"""
improvement_vs_random.py -- merging + packing against an arbitrary qubit assignment.

**Why a second baseline.** The benchmark ablation measures against the *naive* packing,
which fills patches in program order, and that is a deliberately strong reference: program
order on QASMBench is locality-optimised. Measured over 49 proxies at k=2, naive incurs a
median 0.87 of the cross-patch two-qubit gates a random permutation does, and exactly 0.50
on the chain and fan-out families (ising, ghz, cat, wstate) -- algorithms get written
`q[i]-q[i+1]`, and pairing adjacent qubits then captures every other edge for free. Only
the irregular circuits (hhl, sat, multiply) come out slightly worse than chance.

So "beats naive" and "beats an arbitrary assignment" are different claims with different
sizes, and the ablation quotes the harder one. This figure quotes the other, separately,
rather than mixing the two into a single inflated number: the naive comparison stays the
headline and this is the answer to "what do the techniques buy over a compile that never
ran them".

**The arms differ by merging and packing together, not placement alone.** Both arms get
the plan's automatic H-commutation decision -- commutation alone is net positive (see the
benchmark ablation's first segment), so leaving it on in the baseline keeps that arm
honest rather than strawmanned. The baseline is then a random assignment translated and
simulated with merging OFF: a compile with neither of AQORN's own techniques. Ours is the
planned placement with merging ON. The delta is what merging and packing are jointly
worth; the benchmark ablation decomposes the same techniques segment by segment on the
naive substrate, so the two figures answer joint and marginal versions of one question.

**Five metrics, three size groups.** One grouped bar chart: 1q, 2q and depth reduction from
the translated circuit, then LSR and TVD improvement from simulation, each split into
QASMBench's own small / medium / large directories. All five come off one scan row per
placement -- see `simulation.METRICS` -- so the gate counts are the same object the
simulator ran and cost nothing extra.

**The baseline is the mean over `--permutations` random assignments**, each using the same
patch count as AQORN, so both arms pack the same number of patches and the comparison
never becomes packing density against packing density. Improvements are paired within a circuit: circuit-to-
circuit spread runs about twice the placement effect, so a difference of means across
circuits would bury it.

One calibration per circuit, solved on the random baseline. Post-selection is an
exponentially strict filter, so a fixed noise scale keeps every shot or none; alpha is
solved so the *baseline* lands near `--target-selection-rate`, and AQORN is measured at
that same alpha and shot count. Calibrating each arm separately would move the noise along
with the thing being measured.

Run with:

    python -m experiments.figures.improvement_vs_random.improvement_vs_random
    python -m ... --permutations 12 --backend stim
    python -m ... --no-simulate           # gate counts only, no LSR/TVD, seconds not hours
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import COLUMN_WIDTH, data_dir, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import (
    add_sim_arguments,
    check_compile,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from experiments.shared.figures import stamp_provenance
from packing.plan import plan_packing
from permutations.generator import permutation_with_n_codes
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

SCRIPT = Path(__file__).resolve()

SIZE_GROUPS = ("small", "medium", "large")

#: `(column, axis label, higher-is-better?)`. Gate counts are reductions, LSR and TVD are
#: improvements -- both oriented so a positive bar means AQORN won.
METRICS: Sequence[Tuple[str, str]] = (
    ("one_qubit_gates", "1q gates"),
    ("two_qubit_gates", "2q gates"),
    ("circuit_depth", "depth"),
    ("lsr", "LSR"),
    ("tvd", "TVD"),
)

#: Metrics where a *smaller* number is the better outcome, so the improvement is
#: `(baseline - ours) / baseline`. LSR is the exception: a higher selection rate is better,
#: so its improvement is the other way round.
LOWER_IS_BETTER = frozenset(
    {"one_qubit_gates", "two_qubit_gates", "circuit_depth", "tvd"}
)


def _improvement(column: str, baseline: float, ours: float) -> Optional[float]:
    if baseline is None or ours is None or not np.isfinite([baseline, ours]).all():
        return None
    if column in LOWER_IS_BETTER:
        return None if baseline <= 0 else 100.0 * (baseline - ours) / baseline
    return None if baseline <= 0 else 100.0 * (ours - baseline) / baseline


def _counts(
    program: qiskit.QuantumCircuit,
    placement: List[Tuple[int, int]],
    n_code: int,
    *,
    commute: bool,
    merging: bool,
) -> Optional[Dict[str, float]]:
    """Gate counts and depth without simulating. `None` if the placement is rejected."""
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_merging = merging
    config.translator_config.enable_h_commutation = commute
    try:
        circuit = translate_permutation(program, placement, config=config).circuit
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    return {
        "one_qubit_gates": float(one_q),
        "two_qubit_gates": float(two_q),
        "circuit_depth": float(circuit.depth()),
    }


def _random_placements(
    num_qubits: int, patches: int, n_code: int, count: int, seed: int
) -> List[List[Tuple[int, int]]]:
    """
    `count` random assignments at a fixed patch count.

    Same patch count as AQORN on purpose: a random permutation free to use more or fewer
    patches would be measuring packing density, which is a different decision, and one the
    objective settles rather than the placement search.
    """
    placements = []
    for index in range(count):
        placements.append(
            permutation_with_n_codes(
                q=num_qubits, n_codes=patches, seed=seed + index, n=n_code
            )
        )
    return placements


def collect(
    circuits: Sequence[Tuple[str, str, qiskit.QuantumCircuit]],
    n_code: int,
    *,
    permutations: int,
    seed: int,
    simulate: bool,
    sim_settings,
    jobs: Optional[int] = None,
) -> pd.DataFrame:
    """One row per circuit: AQORN's improvement over the mean random assignment."""
    items = [
        (size_group, name, program, n_code, permutations, seed, simulate, sim_settings)
        for size_group, name, program in circuits
    ]
    produced = parallel_map(_run_circuit, items, jobs=jobs, label="circuits")
    return pd.DataFrame([row for row in produced if row is not None])


def _run_circuit(job) -> Optional[Dict[str, float | str]]:
    """
    One circuit: plan, draw its random assignments, then count and simulate both arms.

    Module-level and self-contained because `parallel_map` spawns rather than forks --
    the same shape as `code_size_crossover._run_column`. Placements derive from the
    explicit seed and circuits share nothing, so the parallel rows match the serial
    ones; `parallel_map` preserves input order, so the frame does too.
    """
    size_group, name, program, n_code, permutations, seed, simulate, sim_settings = job
    plan = plan_packing(
        program, n_code, noise=noise_profile_for(sim_settings), verify=True
    )
    # The pipeline can silently miscompile: on `qpe_n9` at k = 2 the combination
    # (packed placement, merging on, H commutation off) computes the wrong distribution
    # while the selection rate stays at 1.0, so neither post-selection nor
    # `plan_packing(verify=True)` catches it. Checked against naive rather than against
    # this figure's random baseline, because naive is the known-good compile and a random
    # placement is not a reference for anything. Guarded even in `--no-sim` mode: a wrong
    # circuit's gate counts are its own, and reporting them as AQORN's is the same fault.
    broken = check_compile(
        program, naive_packing(program.num_qubits, n_code - 2), plan.placement,
        sim_settings, n_code=n_code, h_commutation=plan.h_commutation,
    )
    if broken:
        print(f"[improvement_vs_random] skipping {name} n_code={n_code}: {broken}")
        return None

    randoms = _random_placements(
        program.num_qubits, plan.patches, n_code, permutations, seed
    )

    # Ours carries both techniques; the baseline carries neither (module docstring). The
    # commutation decision is shared so the delta is merging + packing and nothing else.
    ours = _counts(program, plan.placement, n_code,
                   commute=plan.h_commutation, merging=True)
    # Keep the placement alongside its counts. A random assignment is rejected far more
    # often than a planned one -- it can put the operands of a gate the translator
    # cannot route cross-patch in different patches -- and everything downstream has to
    # use a placement that survived. Calibrating on a rejected one silently fails both
    # pilots, `calibrate` returns `fell_back`, and the circuit contributes gate counts
    # but no LSR or TVD: measured 6 of 21 circuits, and every medium and large one.
    survivors = [
        (placement, counts)
        for placement, counts in (
            (placement, _counts(program, placement, n_code,
                                commute=plan.h_commutation, merging=False))
            for placement in randoms
        )
        if counts is not None
    ]
    if ours is None or not survivors:
        return None
    randoms = [placement for placement, _ in survivors]
    theirs = [counts for _, counts in survivors]

    row: Dict[str, float | str] = {
        "size_group": size_group,
        "circuit": name,
        "qubits": program.num_qubits,
        "n_code": n_code,
        "patches": plan.patches,
        "permutations_kept": len(survivors),
        "permutations_rejected": permutations - len(survivors),
        "h_commutation": plan.h_commutation,
        "fell_back": plan.fell_back,
    }
    for column in ("one_qubit_gates", "two_qubit_gates", "circuit_depth"):
        baseline = float(np.mean([entry[column] for entry in theirs]))
        row[f"baseline_{column}"] = baseline
        row[f"ours_{column}"] = ours[column]
        improvement = _improvement(column, baseline, ours[column])
        if improvement is not None:
            row[column] = improvement

    if simulate:
        # Calibrate on the baseline, then measure both arms at that alpha and shot
        # count. The first *surviving* random placement stands in for the baseline:
        # calibrating on all of them would cost `permutations` extra pilot pairs per
        # circuit for a number the mean barely moves.
        reference = randoms[0]
        calibration = calibrate(
            program, reference, sim_settings,
            n_code=n_code,
            baseline_gate_count=baseline_gate_count(program, reference, n_code),
        )
        row["alpha"] = calibration.alpha
        row["repetitions"] = calibration.repetitions
        row["pilot_selection_rate"] = calibration.pilot_selection_rate
        row["calibration_fell_back"] = calibration.fell_back
        # A fallen-back calibration means both pilots failed, so alpha is the untuned
        # `1/m` guess and the selection rate is unknown. Simulating anyway produces
        # numbers from an uncalibrated noise scale, which is worse than no numbers.
        if not calibration.fell_back:
            config_ours = build_config(
                sim_settings, n_code=n_code,
                alpha=calibration.alpha, repetitions=calibration.repetitions,
                h_commutation=plan.h_commutation, merging=True,
            )
            config_base = build_config(
                sim_settings, n_code=n_code,
                alpha=calibration.alpha, repetitions=calibration.repetitions,
                h_commutation=plan.h_commutation, merging=False,
            )
            sim_ours = evaluate(program, plan.placement, config_ours)
            sim_theirs = [evaluate(program, p, config_base) for p in randoms]
            sim_theirs = [entry for entry in sim_theirs if entry is not None]
            if sim_ours is not None and sim_theirs:
                for column in ("lsr", "tvd"):
                    baseline = float(np.mean([e[column] for e in sim_theirs]))
                    row[f"baseline_{column}"] = baseline
                    row[f"ours_{column}"] = sim_ours[column]
                    improvement = _improvement(column, baseline, sim_ours[column])
                    if improvement is not None:
                        row[column] = improvement
    return row


def plot(frame: pd.DataFrame, *, output_name: str) -> List[Path]:
    use_style()
    import matplotlib.pyplot as plt

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    present = [(c, label) for c, label in METRICS if c in frame.columns]
    groups = [g for g in SIZE_GROUPS if (frame["size_group"] == g).any()]

    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    centres = np.arange(len(present), dtype=float)
    width = 0.8 / max(1, len(groups))

    for index, group in enumerate(groups):
        offset = (index - (len(groups) - 1) / 2) * width
        rows = frame[frame["size_group"] == group]
        means, errors = [], []
        for column, _ in present:
            values = rows[column].dropna() if column in rows else pd.Series(dtype=float)
            means.append(float(values.mean()) if len(values) else 0.0)
            # Standard error: the bar is a mean over circuits and the question is whether
            # it clears zero. The per-circuit spread is in the CSV.
            errors.append(
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        axis.bar(
            centres + offset, means, width=width * 0.92,
            color=palette[index % len(palette)], edgecolor="white", linewidth=0.4,
            label=group, zorder=2,
        )
        axis.errorbar(
            centres + offset, means, yerr=errors, fmt="none",
            ecolor="#636363", elinewidth=0.7, capsize=1.6, zorder=3,
        )

    axis.set_xticks(centres)
    axis.set_xticklabels([label for _, label in present])
    # Percent, matching the ablation figures. A factor axis was tried across the paper and
    # kept only on 6 and 9, where the range spans 50x and percent stops being readable.
    axis.set_ylabel("Merging + packing improvement\nover random assignment (%)")
    axis.axhline(0, color="#636363", linewidth=0.8)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.4)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(frameon=False, ncol=len(groups), loc="best")
    # No y floor at zero: a negative group mean is a real result here, and this is a
    # difference rather than a magnitude, so the axis has no natural origin to anchor.
    fig.tight_layout()
    return save_figure(fig, output_name, SCRIPT)


def summarise(frame: pd.DataFrame) -> None:
    print(f"\n[improvement_vs_random] {len(frame)} circuits, "
          f"{int(frame['permutations_kept'].mean())} random placements each")
    header = f"  {'group':<8}{'n':>4}"
    for _, label in METRICS:
        header += f"{label:>10}"
    print(header)
    for group in list(SIZE_GROUPS) + ["all"]:
        rows = frame if group == "all" else frame[frame["size_group"] == group]
        if rows.empty:
            continue
        line = f"  {group:<8}{len(rows):>4}"
        for column, _ in METRICS:
            if column in rows.columns:
                values = rows[column].dropna()
                line += f"{values.mean():>10.2f}" if len(values) else f"{'-':>10}"
            else:
                line += f"{'-':>10}"
        print(line)
    print(f"  verify fell back on {int(frame['fell_back'].sum())}/{len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--n-code", type=int, default=4, help="Code size; k = n_code - 2.")
    parser.add_argument("--qasm-root", type=Path, default=Path("benchmarks/qasmbench"))
    parser.add_argument("--max-gates", type=int, default=2000)
    parser.add_argument("--max-qubits", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=8,
                        help="Random placements averaged to form the baseline.")
    # `add_sim_arguments` owns `--seed` for the simulator; this one seeds the random
    # placements, which have to stay reproducible independently of the shots.
    parser.add_argument("--placement-seed", type=int, default=20260729)
    add_jobs_argument(parser)
    parser.add_argument("--no-simulate", action="store_true",
                        help="Skip LSR and TVD. Gate counts only, and fast.")
    parser.add_argument("--clifford-proxies", choices=["auto", "on", "off"],
                        default="auto",
                        help="Replace rotations with S. auto: on when simulating, since "
                             "Stim refuses rz. Changes the gate-count bars, so 'off' is "
                             "what makes them comparable with the merging ablation.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--persist", action="store_true")
    add_sim_arguments(parser)
    args = parser.parse_args()

    name = f"improvement_vs_random_n{args.n_code}"
    cache = SCRIPT.parent / f"{name}.csv"

    if cache.exists() and not args.refresh:
        frame = pd.read_csv(cache)
        print(f"[improvement_vs_random] reusing {cache} (--refresh to recompute)")
    else:
        # Whether rotations are replaced by S in the proxies. This is not a free choice and
        # it is not cosmetic: Stim refuses `rz`, so simulating *requires* a Clifford corpus,
        # and swapping the corpus moves the gate-count bars. Measured over the same 45
        # circuits at k=2, mean improvement over random placement:
        #
        #                       1q     2q    depth
        #     rotations kept   2.50   4.86   10.49
        #     Clifford         3.83   6.30    7.30
        #
        # and on the medium group depth falls 8.78 -> 0.68. S is mergeable where `rz` is
        # not, so merging captures depth the placement would otherwise have been credited
        # for -- the gain is real either way, it just moves between stages. Gate counts are
        # comparable with `merging_improvement_benchmark_stylized` only with rotations kept,
        # since that is the corpus it uses.
        #
        # So: `auto` follows what the run can actually do, and the flag exists because a
        # reader comparing this figure against the ablation needs to know which corpus
        # produced it. `--clifford-proxies off --backend aer` keeps rotations *and*
        # simulates, at about 128x the wall clock.
        clifford_proxies = (
            not args.no_simulate if args.clifford_proxies == "auto"
            else args.clifford_proxies == "on"
        )
        print(f"[improvement_vs_random] proxies: "
              f"{'rotations replaced by S' if clifford_proxies else 'rotations kept'}")
        circuits: List[Tuple[str, str, qiskit.QuantumCircuit]] = []
        for group in SIZE_GROUPS:
            root = args.qasm_root / group
            if not root.exists():
                continue
            for circuit_name, program in load_proxy_circuits(
                root, args.max_gates, args.max_qubits, 1.0,
                replace_rz_rx_with_s=clifford_proxies,
            ):
                # Re-check the limits against the built circuit. The loader's are
                # advisory: `find_qasm_files` reads them from each directory's README, so a
                # circuit whose metadata is missing or stale is loaded regardless. Passing
                # `--max-qubits 8 --max-gates 200` still yielded a 100-qubit, 95,300-gate
                # proxy, and simulating it against six random placements burned nearly four
                # hours before anyone noticed the run was not the small one it claimed.
                if program.num_qubits < 4 or circuit_name.lower().endswith("_n2"):
                    continue
                if args.max_qubits is not None and program.num_qubits > args.max_qubits:
                    continue
                if args.max_gates is not None and program.size() > args.max_gates:
                    continue
                circuits.append((group, circuit_name, program))
        print(f"[improvement_vs_random] {len(circuits)} circuits after filtering: "
              f"<= {args.max_qubits} qubits, <= {args.max_gates} gates")
        if not circuits:
            raise SystemExit("no circuits found")
        settings = settings_from_args(args)
        frame = collect(
            circuits, args.n_code,
            permutations=args.permutations, seed=args.placement_seed,
            simulate=not args.no_simulate,
            sim_settings=settings,
            jobs=args.jobs,
        )
        if frame.empty:
            raise SystemExit("every placement was rejected; nothing to plot")
        # Stamp the cost model that produced these rows, so a future run can tell
        # whether this cache still describes the current planner. See packing/version.py.
        frame = stamp_provenance(frame)
        frame.to_csv(cache, index=False)
        if args.persist:
            frame.to_csv(data_dir(SCRIPT) / f"{name}.csv", index=False)

    summarise(frame)
    for path in plot(frame, output_name=name):
        print(f"[improvement_vs_random] wrote {path}")


if __name__ == "__main__":
    main()
