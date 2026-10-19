"""
1_best_k_phase_diagram.py -- the best code size moves, and here is the map.

**Claim.** There is no default `k`. Which `[[k+2, k, 2]]` code keeps the most shots is a
joint property of the machine's noise profile and the size of the program, and the
boundary between regimes runs diagonally through that plane. This is the figure the
conclusion points at when it argues the resource decision has to be made per deployment
rather than fixed once in a compiler.

`code_size_crossover` establishes that a crossover exists along the noise-ratio axis, for
a fixed benchmark window. This adds the second axis: it shows the crossover *moving* as
programs get larger, which is what turns "we found a crossover" into "there is no default
answer". Same harness, same calibration convention, one more dimension.

x = the **measured** idle/movement noise ratio of the reference configuration -- the
same naive k=2 baseline the calibration is solved on, read off `measured_regime`. The
nominal ratios (`SimSettings.with_noise_ratio`) are only the sweep that generates
coverage; each column is *labelled* with the geometric mean of what the schedule
actually charged there. Measuring the axis on a fixed reference is also what makes it
independent of the colour variable: the old nominal axis partly restated `k` (patch
count sets transport sets the regime), so the figure was drawing its own axis. y =
program size, binned. Colour = the `k` with the highest selection rate in that cell.

**Selection rate, not TVD, and that is forced.** At the 20-56 qubit widths this figure
needs for a meaningful patch ladder, TVD is not estimable: an entangled state that wide
spreads over more outcomes than any affordable shot budget populates, so two finite
samples score ~1.0 regardless of noise (measured 0.9997 on `ising_n26` by
`code_size_crossover`, and 0.88 flat across `k` at n=20 while building this figure). LSR
is a binomial proportion and stays precise where TVD collapses. The accuracy half of the
`k` story is owned by `lsr_tvd_vs_k` and `2_tvd_vs_noise_ratio_by_k`, which run narrow
enough to measure it.

**Colour is the winner, not the margin** -- deliberately, and it is the difference between
a claim that survives review and one that does not. A margin heatmap invites "that margin
is inside your error bars" on exactly the cells near the boundary, where the margin is
smallest and the objection is strongest. The winner map's claim is only that a boundary
exists and that it moves, which is robust to that objection. `print_summary` reports every
cell's margin and its runner-up so the large ones can be quoted in prose, and marks cells
where the top two sit within one standard error of each other.

**Why the size axis is gates and not qubits.** Program width is pinned to 20-56 by the
patch ladder -- below 20 the largest codes collapse to a single patch and stop being
distinguishable, above 56 the simulation stops being affordable -- so width has too little
room left to bin. Gate count varies by two orders of magnitude inside that window
(164 to 2059 among the non-trivial circuits) and is what actually drives the idle term,
since translated depth is what a qubit sits through. Bins are quantiles of the loaded set,
so every column carries a comparable number of circuits rather than one bin holding
everything.

**Trivial circuits are excluded here**, unlike in `code_size_crossover`. A benchmark with
fewer than `TRIVIAL_GATE_COUNT` gates on 20+ qubits is a chain -- `ghz_n40` is 40 gates on
40 qubits -- and has no packing decision in it, so every `k` scores alike and the winner is
whichever way the sampling noise fell. `code_size_crossover` keeps them because it reports
a mean over a fixed benchmark set and the dilution is part of that honest number. Here they
would not dilute a mean, they would miscolour a cell, and a phase diagram is read as a
statement about each cell individually.

**One noise calibration per (circuit, ratio), solved at the reference code size.** Never
per `k`: calibrating per `k` would pin selection rate along the exact axis the comparison
lives on and flatten every line. Never once per circuit either: separations scale with
total noise, and one alpha across the whole ratio axis lets total noise vary several-fold
along it, measuring the low-ratio columns where nothing separates and the high-ratio
columns at the post-selection floor. This is the convention `code_size_crossover` arrived
at the expensive way; it is inherited rather than re-derived.

**Cells share a denominator.** A circuit that fails to place at any swept `k`, at any
swept ratio, is dropped from every cell -- otherwise a cell's winner could be an artefact
of which circuits happened to survive there.

**`--standard`: the paper's configuration since 2026-10-07** (`packing.standard.STANDARD`,
"w10T"). Opt-in: without the flag every arm, config and output name is the submitted one.

    planner       `plan_standard` -- silent-aware, weight `STANDARD.silent_weight_for(k + 2)`
                  -- on the same noise profile, with the same naive fallback
    translation   `SimSettings.with_standard`: every translation layer-sorted (pilots, guard,
                  regime, arms), and the merging arms build in-patch CZ / XCX transversally
                  at k = 2; the calibration, guard and regime follow it through `build_config`
    accuracy      `FRAME_SHOTS` Pauli frames per arm on the simulation's own Stim circuit
                  (`evaluate(frames=)`): `silent_per_kept`, `p_silent`, `frame_lsr`. Silent
                  error per kept shot is the accuracy measure that survives these widths,
                  where TVD does not (TVD ~ 0.95 x silent per kept where both resolve)
    outputs       `1_best_k_phase_diagram_standard[_vN].csv` beside the script, and with
                  `--persist` under the same name in results/data -- never overwritten; the
                  figures `..._standard` (best k by selection rate, as above) and
                  `..._standard_accuracy` (the k with the fewest silent errors per kept shot)

Every row also says which configuration produced it (`configuration`, `planner_weight`), and
`--from-csv` reads that back, so a standard CSV redraws as the standard figures.

Run with:

    PYTHONPATH=. python experiments/figures/1_best_k_phase_diagram/1_best_k_phase_diagram.py
    PYTHONPATH=. python .../1_best_k_phase_diagram.py --from-csv <path>
    PYTHONPATH=. python .../1_best_k_phase_diagram.py --circuits ising_n26 knn_n25 --ratios 0.25 1 4
    PYTHONPATH=. python .../1_best_k_phase_diagram.py --standard --jobs 12 --persist
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import (
    COLUMN_WIDTH,
    data_dir,
    factor_from_pct,
    format_factor,
    save_figure,
    use_style,
)
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import (
    check_compile,
    measured_regime,
    SimSettings,
    add_sim_arguments,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from experiments.shared.figures import stamp_provenance
from packing.plan import plan_packing
from packing.standard import STANDARD, plan_standard

NAME = "1_best_k_phase_diagram"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: Stem of every `--standard` output: CSV, persisted copy and both figures.
STANDARD_NAME = f"{NAME}_standard"

#: Pauli frames per simulated arm under `--standard`. The frame run reuses the Stim circuit
#: the simulation already converted, so it costs a FlipSimulator pass, not a conversion; at
#: the lowest selection rate this figure has seen (0.006) it still keeps ~600 frames.
FRAME_SHOTS = 100_000

Perm = List[Tuple[int, int]]

#: Log-spaced in powers of four and symmetric about 1.0, so the movement-dominated and
#: idle-dominated halves get equal resolution. Narrower than `code_size_crossover`'s
#: 1/256..256 because this figure spends its budget on the second axis instead; the
#: boundary in that figure sits between ratio 2 and 4, comfortably inside this span.
DEFAULT_RATIOS: Tuple[float, ...] = (1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0)

#: Matches every other k figure in the suite, so the family is the same one throughout.
#: Odd k is not expressible -- `n_code = k + 2` must be even, and the pipeline rejects
#: odd code sizes outright (verified: k = 3, 5, 7 all refused).
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: Program-qubit window. The floor keeps the patch ladder from collapsing -- below 20,
#: k = 8 and k = 10 both land on a single patch and the largest codes stop being
#: distinguishable. The ceiling keeps the sweep affordable.
DEFAULT_QUBIT_RANGE: Tuple[int, int] = (20, 56)

#: Program gates above this are dropped: `load_proxy_circuits`' own limit filters the
#: pre-proxy circuit, which lets very large benchmarks through at high per-call cost.
DEFAULT_GATE_LIMIT = 4000

#: Below this a benchmark on 20+ qubits is a chain with no packing decision in it. See
#: the module docstring on why these are excluded here and kept in `code_size_crossover`.
TRIVIAL_GATE_COUNT = 100

DEFAULT_SIZE_BINS = 4

#: Sequential rather than categorical, because `k` is ordinal: a reader should be able to
#: see "larger codes win up here" without decoding a legend. This is the positive arm of
#: `figures.HEATMAP_CMAP`, so the two heatmaps in the paper read as the same family. Every
#: cell also carries its `k` as text, which is what makes the panel greyscale-safe.
K_RAMP = LinearSegmentedColormap.from_list(
    "aqorn_k", ["#b3cde0", "#6497b1", "#005b96", "#03396c", "#011f4b"]
)


@dataclass
class Settings:
    ratios: Sequence[float] = DEFAULT_RATIOS
    k_values: Sequence[int] = DEFAULT_K_VALUES
    circuits: Optional[Sequence[str]] = None
    qubit_range: Tuple[int, int] = DEFAULT_QUBIT_RANGE
    gate_limit: int = DEFAULT_GATE_LIMIT
    min_gates: int = TRIVIAL_GATE_COUNT
    size_bins: int = DEFAULT_SIZE_BINS
    #: Fixed *raw* shot budget per point. The shots post-selection discards are the
    #: result, so the budget must not be scaled by the calibrated selection rate.
    repetitions: int = 40_000
    #: Code size the single per-circuit alpha calibration is solved at.
    reference_k: int = 2
    benchmark_root: Path = field(default_factory=lambda: Path("benchmarks/qasmbench"))
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None
    #: The standard configuration (`--standard`); see the module docstring.
    standard: bool = False


def load_circuits(settings: Settings) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """The benchmark set, filtered to a window where every k has a distinct patch count."""
    root = settings.benchmark_root
    if not root.exists():
        raise FileNotFoundError(
            f"benchmark root {root} not found. Run from the repository root, pass "
            f"--benchmark-root, or initialise the submodule with "
            f"`git submodule update --init --recursive`."
        )
    low, high = settings.qubit_range
    loaded = load_proxy_circuits(root, qubit_limit=None if settings.circuits else high)
    if settings.circuits:
        wanted = set(settings.circuits)
        chosen = [(name, c) for name, c in loaded if name in wanted]
        missing = wanted - {name for name, _ in chosen}
        if missing:
            raise ValueError(f"no such benchmark circuit: {sorted(missing)}")
        return chosen
    return [
        (name, circuit)
        for name, circuit in loaded
        if low <= circuit.num_qubits <= high
        and settings.min_gates <= circuit.size() <= settings.gate_limit
    ]


def _placement(circuit: qiskit.QuantumCircuit, k: int, profile) -> Tuple[Perm, bool]:
    """
    The compiler's placement at this code size, and the commutation it was priced under.

    The second return value has to be honoured by the caller: translating with commutation
    in a state the plan was not priced under describes a circuit the pipeline would never
    build. Falls back to naive so a refused plan costs a data point rather than the run.
    """
    try:
        plan = plan_packing(
            circuit, k + 2, h_commutation="auto", verify=True, noise=profile
        )
        return plan.placement, plan.h_commutation
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return naive_packing(circuit.num_qubits, k), True


def _standard_placement(
    circuit: qiskit.QuantumCircuit, k: int, profile, where: str = ""
) -> Tuple[Perm, bool, str]:
    """
    `_placement` under the standard planner, plus what the planner did.

    `plan_standard` is the silent-aware planner at `STANDARD.silent_weight_for(k + 2)`, given
    the same noise profile `_placement` gives `plan_packing`, with the same naive fallback.
    The status goes in the CSV because that fallback is otherwise invisible -- a `k` with no
    packaged silent-error rates (n_code outside 4..12) raises and would be measured as
    naive: "planned", "naive_by_verify" (verify kept the naive packing, a normal outcome) or
    "planner_error" (it raised; naive with commutation on, exactly as `_placement` does).
    """
    try:
        plan = plan_standard(
            circuit, k + 2, noise=profile, h_commutation="auto", verify=True
        )
    except Exception as error:  # noqa: BLE001 - a rejected placement is a normal outcome
        print(f"[{NAME}] {where}k={k}: standard planner raised {error!r}; "
              f"measuring the naive placement")
        return naive_packing(circuit.num_qubits, k), True, "planner_error"
    status = "naive_by_verify" if getattr(plan, "fell_back", False) else "planned"
    return plan.placement, plan.h_commutation, status


def _run_cell(job) -> List[Dict]:
    """
    One (circuit, ratio): calibrate once at the reference k, then measure every k.

    Module-level and self-contained because `parallel_map` spawns rather than forks. The
    unit of work is a whole ratio column for one circuit rather than a single (circuit,
    ratio, k) point, so the two calibration pilots are paid once and shared by every k in
    the column -- which is also what makes the column a fair comparison.
    """
    settings, name, circuit, ratio = job
    sim = settings.sim.with_noise_ratio(ratio)
    if settings.standard:
        # Every translation below -- both pilots, the guard, the regime and the arms -- is
        # layer-sorted, and the merging arms build in-patch CZ / XCX transversally at k = 2:
        # `build_config` and `measured_regime` read both flags off `sim`. The noise profile
        # does not depend on them, so the planner prices the same machine either way.
        sim = sim.with_standard()
    profile = noise_profile_for(sim)
    frames = FRAME_SHOTS if settings.standard else 0

    reference_packing = naive_packing(circuit.num_qubits, settings.reference_k)
    calibration = calibrate(
        circuit,
        reference_packing,
        sim,
        n_code=settings.reference_k + 2,
        baseline_gate_count=circuit.size() * 3,
    )
    if not math.isfinite(calibration.alpha):
        return []

    # The x value: what the reference configuration's schedule actually charges at
    # these scalars. Fixed at the calibration reference (naive, passes off, k=2) so
    # the axis cannot restate the colour variable, and alpha-invariant so the
    # calibrated alpha does not move it.
    try:
        regime = measured_regime(
            circuit, reference_packing, n_code=settings.reference_k + 2,
            idle_time_scalar=sim.idle_time_scalar,
            distance_scalar=sim.distance_scalar,
            asap_order=sim.asap_order,
        )
    except Exception:  # noqa: BLE001 - a rejected reference is a normal outcome
        return []

    rows: List[Dict] = []
    for k in settings.k_values:
        if settings.standard:
            placement, commute, plan_status = _standard_placement(
                circuit, k, profile, where=f"{name} ratio={ratio} "
            )
        else:
            placement, commute = _placement(circuit, k, profile)
        # The pipeline can silently miscompile: on `qpe_n9` at k = 2 the combination
        # (packed placement, merging on, H commutation off) computes the wrong distribution
        # while the selection rate stays at 1.0, so neither post-selection nor
        # `plan_packing(verify=True)` catches it. This figure reads selection rate off
        # every k, so one miscompiled cell would move a *winner*. Skip rather than fall
        # back to naive -- that would hand the cell to whichever k had not miscompiled.
        broken = check_compile(
            circuit, naive_packing(circuit.num_qubits, k), placement, sim,
            n_code=k + 2, h_commutation=commute,
        )
        if broken:
            print(f"[{NAME}] skipping {name} k={k} ratio={ratio}: {broken}")
            continue
        config = build_config(
            sim,
            n_code=k + 2,
            alpha=calibration.alpha,
            repetitions=settings.repetitions,
            h_commutation=commute,
            merging=True,
        )
        # `frames=0` (the submitted configuration) is `evaluate`'s default: no frame run.
        measured = evaluate(circuit, placement, config, frames=frames, frame_seed=sim.seed)
        if measured is None:
            continue
        row = {
            "circuit": name,
            "program_qubits": circuit.num_qubits,
            "program_gates": circuit.size(),
            "ratio": ratio,
            "measured_ratio_ref": regime["measured_ratio"],
            "movement_share_ref": regime["movement_share"],
            "idle_scalar": sim.idle_time_scalar,
            "distance_scalar": sim.distance_scalar,
            "k": k,
            "n_code": k + 2,
            "patches": len({patch for patch, _ in placement}),
            "alpha": calibration.alpha,
            "repetitions": settings.repetitions,
            "h_commutation": commute,
            "lsr": measured["lsr"],
            # Recorded, never plotted: not estimable at these widths. See the docstring.
            "tvd": measured["tvd"],
            "one_qubit_gates": measured["one_qubit_gates"],
            "two_qubit_gates": measured["two_qubit_gates"],
            "circuit_depth": measured["circuit_depth"],
        }
        if settings.standard:
            # Which configuration made the row; a CSV without the column is the submitted
            # one, so the default schema is unchanged.
            nan = float("nan")
            row.update({
                "configuration": "standard",
                "planner_weight": STANDARD.silent_weight_for(k + 2),
                "plan_status": plan_status,
                # Silent logical errors per kept shot, per shot, and the frames' own
                # selection rate; NaN when the backend is not Stim.
                "silent_per_kept": measured.get("silent_per_kept", nan),
                "p_silent": measured.get("p_silent", nan),
                "frame_lsr": measured.get("frame_lsr", nan),
                # The same frames on the lanes that hold a program qubit only (empty lanes at
                # k > 2 carry no program error); the accuracy map uses this when present.
                "silent_per_kept_occupied": measured.get("silent_per_kept_occupied", nan),
            })
        rows.append(row)
    return rows


def compute(settings: Settings) -> pd.DataFrame:
    circuits = load_circuits(settings)
    if not circuits:
        raise RuntimeError(
            f"no benchmark circuit has {settings.qubit_range[0]}-"
            f"{settings.qubit_range[1]} qubits and "
            f"{settings.min_gates}-{settings.gate_limit} gates"
        )
    print(
        f"[{NAME}] {len(circuits)} circuits x {len(settings.ratios)} ratios x "
        f"{len(settings.k_values)} code sizes, {settings.repetitions:,} raw shots/point"
    )
    print(f"[{NAME}] circuits: {', '.join(name for name, _ in circuits)}")
    if settings.standard:
        weights = ", ".join(
            f"k={k}: {STANDARD.silent_weight_for(k + 2):g}" for k in settings.k_values
        )
        print(
            f"[{NAME}] standard configuration: plan_standard (silent weight {weights}), "
            f"every translation layer-sorted, transversal in-patch CZ/XCX on the merging "
            f"arms at k=2, {FRAME_SHOTS:,} Pauli frames per arm"
        )

    jobs = [
        (settings, name, circuit, ratio)
        for name, circuit in circuits
        for ratio in settings.ratios
    ]
    produced = parallel_map(_run_cell, jobs, jobs=settings.jobs, label="circuit x ratio")
    rows = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def complete_circuits(frame: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    """
    Keep only circuits that placed at every k and every ratio, so cells share a denominator.

    Without this a cell's winner could be an artefact of which circuits survived there
    rather than a statement about the regime.
    """
    wanted = len(set(k_values)) * frame["ratio"].nunique()
    counts = frame.groupby("circuit")["lsr"].size()
    keep = counts[counts >= wanted].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"[{NAME}] dropped {len(dropped)} incomplete circuits: {', '.join(dropped)}")
    return frame[frame["circuit"].isin(keep)].copy()


def assign_bins(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    """
    Quantile bins over program gates, so every row of the map carries similar weight.

    Quantiles rather than equal-width because the gate distribution is heavily
    right-skewed -- equal-width bins would put fourteen circuits in the first and one in
    the last, and a cell backed by one circuit is a sampling accident with a colour.
    """
    per_circuit = frame.groupby("circuit")["program_gates"].first().sort_values()
    labels = pd.qcut(per_circuit, q=bins, labels=False, duplicates="drop")
    edges = pd.qcut(per_circuit, q=bins, duplicates="drop").cat.categories

    frame = frame.copy()
    frame["size_bin"] = frame["circuit"].map(labels)
    frame["size_label"] = frame["size_bin"].map(
        {i: f"{int(math.ceil(c.left))}-{int(c.right)}" for i, c in enumerate(edges)}
    )
    return frame


def _log_mean(values: np.ndarray) -> Tuple[float, float]:
    """
    Geometric mean of selection rate and its standard error, taken in log space.

    Selection rate compounds multiplicatively across a circuit -- it is the product of
    per-operation survival -- so the arithmetic mean over circuits is dominated by
    whichever benchmark happened to keep the most shots. `code_size_crossover` uses the
    same convention, and the two figures have to aggregate alike to be quoted together.
    """
    clean = np.asarray([v for v in values if v == v and v > 0], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    logs = np.log(clean)
    centre = float(np.exp(logs.mean()))
    if clean.size < 2:
        return centre, 0.0
    error = float(logs.std(ddof=1) / math.sqrt(clean.size))
    return centre, error


def winners(
    frame: pd.DataFrame,
    settings: Settings,
    metric: str = "lsr",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    One row per (size bin, ratio): the winning k, its margin, and its runner-up.

    Judged on `metric`, geometric-meaned over the cell's circuits (`_log_mean`): selection
    rate by default, higher better. `--standard` also judges `silent_per_kept`, lower better;
    the margin is then how many times more silent errors the runner-up has.
    """
    rows: List[Dict] = []
    for (size_bin, label), block in frame.groupby(["size_bin", "size_label"]):
        for ratio in sorted(block["ratio"].unique()):
            column = block[block["ratio"] == ratio]
            scores: Dict[int, Tuple[float, float]] = {}
            for k in settings.k_values:
                values = column[column["k"] == k][metric].to_numpy()
                centre, error = _log_mean(values)
                if centre == centre:
                    scores[k] = (centre, error)
            if not scores:
                continue
            ordered = sorted(
                scores.items(), key=lambda item: item[1][0], reverse=higher_is_better
            )
            (best_k, (best, best_err)), = ordered[:1]
            runner_k, (runner, runner_err) = (
                ordered[1] if len(ordered) > 1 else (best_k, (best, best_err))
            )
            # Errors are in log space, so combine there and read the margin as a ratio.
            # `top / bottom` is the winner's advantage as a ratio >= 1 in either direction.
            spread = math.sqrt(best_err ** 2 + runner_err ** 2)
            top, bottom = (best, runner) if higher_is_better else (runner, best)
            margin = (top / bottom - 1.0) * 100.0 if bottom > 0 else float("nan")
            rows.append({
                "size_bin": size_bin,
                "size_label": label,
                "ratio": ratio,
                "best_k": best_k,
                f"best_{metric}": best,
                "runner_k": runner_k,
                f"runner_{metric}": runner,
                "margin_pct": margin,
                "margin_log": math.log(top / bottom) if bottom > 0 else float("nan"),
                "margin_se_log": spread,
                "significant": bool(
                    bottom > 0 and math.log(top / bottom) > spread
                ),
                "circuits": column["circuit"].nunique(),
                "patches_at_best": float(
                    column[column["k"] == best_k]["patches"].mean()
                ),
            })
    return pd.DataFrame(rows)


def print_summary(frame: pd.DataFrame, table: pd.DataFrame, settings: Settings) -> None:
    print(f"\n[{NAME}] {frame['circuit'].nunique()} circuits after the completeness filter")
    print(f"[{NAME}] winning k by (size bin, ratio); * marks a margin above 1 SE\n")

    ratios = sorted(table["ratio"].unique())
    header = "  gates".ljust(14) + "".join(f"{_tick_label(r):>10}" for r in ratios)
    print(header)
    for size_bin in sorted(table["size_bin"].unique()):
        block = table[table["size_bin"] == size_bin]
        label = block["size_label"].iloc[0]
        cells = []
        for ratio in ratios:
            row = block[block["ratio"] == ratio]
            if row.empty:
                cells.append(f"{'-':>10}")
                continue
            row = row.iloc[0]
            mark = "*" if row["significant"] else " "
            cells.append(f"{'k=' + str(int(row['best_k'])) + mark:>10}")
        print(f"  {label:<12}" + "".join(cells))

    print(f"\n[{NAME}] margins over the runner-up (x), same layout")
    print(header)
    for size_bin in sorted(table["size_bin"].unique()):
        block = table[table["size_bin"] == size_bin]
        label = block["size_label"].iloc[0]
        cells = []
        for ratio in ratios:
            row = block[block["ratio"] == ratio]
            # `margin_pct` is `(best/runner - 1) * 100`, a plain change, so the factor is
            # `1 + pct/100`. Reported as a factor to match the rest of the paper.
            cells.append(
                f"{'-':>10}" if row.empty
                else f"{format_factor(factor_from_pct(row.iloc[0]['margin_pct'], reduction=False, floor=0.0)):>10}"
            )
        print(f"  {label:<12}" + "".join(cells))

    distinct = table["best_k"].nunique()
    significant = int(table["significant"].sum())
    print(
        f"\n[{NAME}] {distinct} distinct code sizes win somewhere on the map; "
        f"{significant}/{len(table)} cells have a margin above one standard error"
    )
    largest = table.nlargest(3, "margin_pct")
    for _, row in largest.iterrows():
        print(
            f"  gates {row['size_label']}, ratio {_tick_label(row['ratio'])}: "
            f"k={int(row['best_k'])} beats k={int(row['runner_k'])} by "
            f"{format_factor(factor_from_pct(row['margin_pct'], reduction=False, floor=0.0))}"
        )

    print(f"\n[{NAME}] measured reference ratio by nominal column "
          f"(geometric mean +- spread over circuits)")
    for ratio in ratios:
        values = frame[frame["ratio"] == ratio]["measured_ratio_ref"].to_numpy()
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        logs = np.log(values)
        centre = float(np.exp(logs.mean()))
        spread = float(np.exp(logs.std(ddof=1))) if len(logs) > 1 else 1.0
        print(f"  nominal {_tick_label(ratio):>6} -> measured {centre:8.3f}  "
              f"(x/ {spread:.2f} over circuits)")

    print(f"\n[{NAME}] absolute selection rate by (size bin, ratio, k)")
    for size_bin in sorted(frame["size_bin"].unique()):
        block = frame[frame["size_bin"] == size_bin]
        print(f"  gates {block['size_label'].iloc[0]}")
        for ratio in ratios:
            column = block[block["ratio"] == ratio]
            parts = []
            for k in settings.k_values:
                centre, _ = _log_mean(column[column["k"] == k]["lsr"].to_numpy())
                parts.append(f"k{k}={centre:.4f}" if centre == centre else f"k{k}=  -   ")
            print(f"    ratio {_tick_label(ratio):>6}: " + "  ".join(parts))


def accuracy_rows(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    `frame` with `silent_per_kept` blanked where it is not a measurement; None if none is.

    A rate counts when it is finite and at least one frame was kept: `split_from_stim`
    divides by `max(1, kept)`, so an arm whose every frame was detected reports 0 -- which
    would read as the most accurate arm. `_log_mean` skips NaN (and 0), so such a row drops
    out of its cell; the counts are printed rather than left silent.
    """
    if "silent_per_kept" not in frame.columns:
        return None
    frame = frame.copy()
    if "silent_per_kept_occupied" in frame.columns and \
            pd.to_numeric(frame["silent_per_kept_occupied"], errors="coerce").notna().any():
        # Count only flips on lanes that hold a program qubit: on every lane, k > 2 at fewer
        # qubits than slots is charged for errors no program output can show.
        print(f"[{NAME}] accuracy map on occupied lanes (silent_per_kept_occupied); the "
              f"all-lane rate stays in silent_per_kept_all_lanes")
        frame["silent_per_kept_all_lanes"] = frame["silent_per_kept"]
        frame["silent_per_kept"] = frame["silent_per_kept_occupied"]
    rates = pd.to_numeric(frame["silent_per_kept"], errors="coerce")
    if "frame_lsr" in frame.columns:
        kept = pd.to_numeric(frame["frame_lsr"], errors="coerce") > 0
    else:
        kept = pd.Series(True, index=frame.index)
    usable = rates.notna() & kept
    if not usable.any():
        return None
    if not usable.all():
        print(f"[{NAME}] {int((~usable).sum())}/{len(frame)} rows carry no silent-error "
              f"rate (no kept frame, or not Stim); their cells average fewer circuits")
    zeros = int((rates[usable] == 0).sum())
    if zeros:
        print(f"[{NAME}] {zeros} rows measured no silent error at all; the geometric mean "
              f"skips them, which counts against their k -- check before quoting")
    out = frame.copy()
    out["silent_per_kept"] = rates.where(usable)
    return out


def print_accuracy_summary(
    frame: pd.DataFrame, table: pd.DataFrame, lsr_table: pd.DataFrame, settings: Settings
) -> None:
    """The `--standard` accuracy map in text, and how often it agrees with the LSR map."""
    ratios = sorted(table["ratio"].unique())
    header = "  gates".ljust(14) + "".join(f"{_tick_label(r):>10}" for r in ratios)
    print(f"\n[{NAME}] fewest silent errors per kept shot: winning k by (size bin, "
          f"ratio); * marks a margin above 1 SE\n")
    print(header)
    for size_bin in sorted(table["size_bin"].unique()):
        block = table[table["size_bin"] == size_bin]
        cells = []
        for ratio in ratios:
            row = block[block["ratio"] == ratio]
            if row.empty:
                cells.append(f"{'-':>10}")
                continue
            row = row.iloc[0]
            mark = "*" if row["significant"] else " "
            cells.append(f"{'k=' + str(int(row['best_k'])) + mark:>10}")
        print(f"  {block['size_label'].iloc[0]:<12}" + "".join(cells))

    both = table.merge(lsr_table, on=["size_bin", "ratio"], suffixes=("", "_lsr"))
    agree = int((both["best_k"] == both["best_k_lsr"]).sum())
    print(f"\n[{NAME}] the most accurate k is also the selection-rate winner in "
          f"{agree}/{len(both)} cells")

    print(f"\n[{NAME}] silent errors per kept shot by (size bin, ratio, k), geometric mean")
    for size_bin in sorted(frame["size_bin"].unique()):
        block = frame[frame["size_bin"] == size_bin]
        print(f"  gates {block['size_label'].iloc[0]}")
        for ratio in ratios:
            column = block[block["ratio"] == ratio]
            parts = []
            for k in settings.k_values:
                centre, _ = _log_mean(column[column["k"] == k]["silent_per_kept"].to_numpy())
                parts.append(f"k{k}={centre:.4f}" if centre == centre else f"k{k}=  -   ")
            print(f"    ratio {_tick_label(ratio):>6}: " + "  ".join(parts))


def _tick_label(ratio: float) -> str:
    """Powers of four read as fractions below one and integers above it."""
    if ratio >= 1:
        return f"{int(round(ratio))}"
    return f"1/{int(round(1 / ratio))}"


def _measured_label(frame: pd.DataFrame, ratio: float) -> str:
    """
    Geometric mean of the measured reference ratio in one nominal column.

    The nominal ratio indexes the sweep; the label is what the reference schedules
    actually charged there, which is the number a hardware reader can locate their
    machine against.
    """
    values = frame[frame["ratio"] == ratio]["measured_ratio_ref"].to_numpy()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return _tick_label(ratio)
    centre = float(np.exp(np.log(values).mean()))
    return f"{centre:.2g}"


def plot(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    settings: Settings,
    *,
    name: str = NAME,
    label: str = "best code size $k$",
) -> List[Path]:
    """The winner map of `table` (any `winners` table), saved versioned as `name`."""
    use_style()

    ratios = sorted(table["ratio"].unique())
    bins = sorted(table["size_bin"].unique())
    k_values = list(settings.k_values)

    grid = np.full((len(bins), len(ratios)), np.nan)
    for row_index, size_bin in enumerate(bins):
        for col_index, ratio in enumerate(ratios):
            row = table[(table["size_bin"] == size_bin) & (table["ratio"] == ratio)]
            if not row.empty:
                grid[row_index, col_index] = row.iloc[0]["best_k"]

    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.78))

    # Discrete norm: one colour band per k, so the colourbar reads as five categories
    # rather than a continuum that happens to be sampled at five points.
    boundaries = [k - 1 for k in k_values] + [k_values[-1] + 1]
    norm = BoundaryNorm(boundaries, K_RAMP.N)
    mesh = axis.imshow(
        grid, aspect="auto", origin="lower", cmap=K_RAMP, norm=norm,
        extent=(-0.5, len(ratios) - 0.5, -0.5, len(bins) - 0.5),
    )

    # Thin separators in the page colour, so cells read as discrete states rather than a
    # continuous field. A phase diagram's claim is about which cell you are in.
    for edge in range(len(ratios) + 1):
        axis.axvline(edge - 0.5, color="#ffffff", linewidth=0.8)
    for edge in range(len(bins) + 1):
        axis.axhline(edge - 0.5, color="#ffffff", linewidth=0.8)

    # The k in every cell: this is what keeps the panel readable in greyscale, where the
    # darkest two steps of the ramp are hard to tell apart.
    for row_index in range(len(bins)):
        for col_index in range(len(ratios)):
            value = grid[row_index, col_index]
            if value != value:
                continue
            cell = table[
                (table["size_bin"] == bins[row_index])
                & (table["ratio"] == ratios[col_index])
            ].iloc[0]
            # Contrast against the colour actually drawn, not against k. Reading the
            # threshold off the raw k value breaks whenever the swept range does not span
            # the whole ramp -- a two-value sweep puts k=4 at the darkest step, and this
            # printed black on navy.
            red, green, blue, _ = K_RAMP(norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            colour = "#ffffff" if luminance < 0.5 else "#1a1a1a"
            axis.text(
                col_index, row_index,
                f"{int(value)}" + ("" if cell["significant"] else " ·"),
                ha="center", va="center", fontsize=8, color=colour,
            )

    axis.set_xticks(range(len(ratios)))
    axis.set_xticklabels([_measured_label(frame, r) for r in ratios])
    axis.set_yticks(range(len(bins)))
    axis.set_yticklabels(
        [table[table["size_bin"] == b]["size_label"].iloc[0] for b in bins]
    )
    axis.set_xlabel("measured idle / movement noise ratio (column mean)")
    axis.set_ylabel("program gates")

    # A heatmap's own cells carry the reading; gridlines would only compete with them.
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)

    bar = fig.colorbar(mesh, ax=axis, ticks=k_values, pad=0.03)
    bar.set_label(label)
    bar.outline.set_visible(False)

    paths = save_figure(fig, name, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(defaults.ratios))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values))
    parser.add_argument("--circuits", nargs="+", default=None,
                        help="Explicit benchmark names, bypassing the width filter.")
    parser.add_argument("--qubit-range", nargs=2, type=int,
                        default=list(defaults.qubit_range))
    parser.add_argument("--gate-limit", type=int, default=defaults.gate_limit)
    parser.add_argument("--min-gates", type=int, default=defaults.min_gates,
                        help="Below this a wide benchmark is a chain with no packing "
                             "decision in it, and its winning k is sampling noise.")
    parser.add_argument("--size-bins", type=int, default=defaults.size_bins)
    parser.add_argument("--repetitions", type=int, default=defaults.repetitions,
                        help="Raw shots per point. Fixed, not scaled by selection rate: "
                             "the shots post-selection discards are the result.")
    parser.add_argument("--reference-k", type=int, default=defaults.reference_k)
    parser.add_argument("--benchmark-root", type=Path, default=defaults.benchmark_root)
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=None,
                        help=f"Default: {NAME}.csv beside the script, replaced each run. "
                             f"With --standard: {STANDARD_NAME}.csv beside the script, "
                             f"and this path or that one is never overwritten -- an "
                             f"existing name moves on to _v2, _v3, ...")
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it here: this "
                             "run is hours of simulation.")
    parser.add_argument("--standard", action="store_true",
                        help="The paper's configuration since 2026-10-07 (packing.standard): "
                             "plan_standard, every translation layer-sorted, transversal "
                             "in-patch CZ/XCX on the merging arms at k=2, and "
                             f"{FRAME_SHOTS:,} Pauli frames per arm for silent errors per "
                             "kept shot. See the module docstring.")
    args = parser.parse_args()
    if args.csv is None:
        args.csv = (HERE / f"{STANDARD_NAME}.csv") if args.standard else defaults.csv_path
    return Settings(
        ratios=args.ratios,
        k_values=args.k_values,
        circuits=args.circuits,
        qubit_range=(args.qubit_range[0], args.qubit_range[1]),
        gate_limit=args.gate_limit,
        min_gates=args.min_gates,
        size_bins=args.size_bins,
        repetitions=args.repetitions,
        reference_k=args.reference_k,
        benchmark_root=args.benchmark_root,
        sim=settings_from_args(args, defaults.sim),
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        jobs=args.jobs,
        standard=args.standard,
    )


def free_csv(path: Path, *elsewhere: Path) -> Path:
    """
    `path`, or `<stem>_v2.csv`, `_v3`, ... -- the first name free beside it and in every
    `elsewhere` directory, so a rerun is additive and a persisted copy keeps the local
    copy's name. `next_version` does the same for figures.
    """
    def taken(candidate: Path) -> bool:
        return candidate.exists() or any((d / candidate.name).exists() for d in elsewhere)

    candidate, version = path, 2
    while taken(candidate):
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        version += 1
    return candidate


def write_standard_csv(frame: pd.DataFrame, settings: Settings) -> Path:
    """
    The `--standard` CSV, at `--csv` (default `STANDARD_NAME.csv` beside the script) and with
    `--persist` in results/data, under the first name free in both. Exclusive creation
    (`mode="x"`) makes "never overwritten" a property of the write, not only of the check.
    """
    kept_dir = data_dir(SCRIPT) if settings.persist else None
    path = free_csv(settings.csv_path, *([kept_dir] if kept_dir else []))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")
    print(f"[{NAME}] wrote {len(frame)} rows to {path}")
    if kept_dir is not None:
        kept = kept_dir / path.name
        frame.to_csv(kept, index=False, mode="x")
        print(f"[{NAME}] persisted to {kept}")
    return path


def csv_configuration(frame: pd.DataFrame) -> str:
    """
    Which configuration produced a cached CSV: "standard" or "submitted". A CSV from before
    `--standard` existed has no `configuration` column and is the submitted configuration.
    """
    if "configuration" not in frame.columns:
        return "submitted"
    found = sorted({str(value) for value in frame["configuration"].dropna().unique()})
    if len(found) != 1:
        raise RuntimeError(
            f"expected one configuration in the CSV, found {found or 'none'}; "
            f"these rows cannot be drawn as one map"
        )
    return found[0]


def main() -> None:
    settings = parse_args()

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
        # The rows say which configuration made them, and that decides which figures
        # they draw as -- so a standard CSV redraws as the standard figures, flag or not.
        configuration = csv_configuration(frame)
        if settings.standard and configuration != "standard":
            raise SystemExit(
                f"[{NAME}] --standard, but {settings.from_csv} holds {configuration}-"
                f"configuration rows; drop the flag to redraw them as what they are"
            )
        if configuration == "standard" and not settings.standard:
            print(f"[{NAME}] {settings.from_csv} holds standard-configuration rows: "
                  f"drawing the standard figures")
        settings.standard = configuration == "standard"
    else:
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Stamp the cost model that produced these rows, so a future run can tell
        # whether this cache still describes the current planner. See packing/version.py.
        frame = stamp_provenance(frame)
        if settings.standard:
            write_standard_csv(frame, settings)
        else:
            frame.to_csv(settings.csv_path, index=False)
            print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
            if settings.persist:
                kept = data_dir(SCRIPT) / f"{NAME}.csv"
                frame.to_csv(kept, index=False)
                print(f"[{NAME}] persisted to {kept}")

    frame = complete_circuits(frame, settings.k_values)
    if frame.empty:
        raise RuntimeError("no circuit placed at every k and ratio")
    frame = assign_bins(frame, settings.size_bins)
    table = winners(frame, settings)
    print_summary(frame, table, settings)
    if not settings.standard:
        plot(frame, table, settings)
        return

    plot(frame, table, settings, name=STANDARD_NAME)
    # Same cells, same circuits, same layout: the k with the fewest silent logical errors
    # per kept shot -- the accuracy half of the code-size decision, measurable at this width.
    accuracy = accuracy_rows(frame)
    if accuracy is None:
        print(f"[{NAME}] these rows carry no silent-error rates; accuracy map skipped")
        return
    accurate = winners(accuracy, settings, metric="silent_per_kept", higher_is_better=False)
    print_accuracy_summary(accuracy, accurate, table, settings)
    plot(accuracy, accurate, settings, name=f"{STANDARD_NAME}_accuracy",
         label="most accurate code size $k$")


if __name__ == "__main__":
    main()
