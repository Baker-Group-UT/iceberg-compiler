"""
lsr_tvd_vs_k.py -- what a larger code costs in post-selection, and what it costs in error.

**Claim.** Raising k costs correctness for no gain in throughput: a bigger patch puts
more logical qubits behind the same pair of detectors, so the selection rate stays flat
while the distance-2 code catches a smaller share of what goes wrong and TVD climbs.

The AQORN-against-naive comparison is the secondary reading, and an honest one: at these
widths the two placements sit inside each other's spread at every k. That is a real
result rather than a missing one -- packing buys gate count and depth (see
`ablation_vs_k`), and on a circuit this narrow that does not translate into a measurable
selection-rate or TVD difference. Read the gap, not the ordering.

Two panels sharing the k axis:

    LSR   logical selection rate -- the fraction of shots that clear post-selection
    TVD   logical total variation distance against the noiseless reference

Two series, both at the same k: the naive index-order packing, and the placement AQORN
plans. Merging is on for both, and H commutation follows the planner's own per-circuit
decision -- applied to both rows, so the only difference between them is where the qubits
sit rather than which passes ran.

**Both series share one noise calibration per circuit.** Post-selection is an
exponentially strict filter, so a fixed noise scale either keeps everything or keeps
nothing; alpha is solved per (circuit, k) so the *naive baseline* lands near
`--target-selection-rate`, and the AQORN row is then measured at that same alpha and
shot count. Calibrating each placement separately would move the noise with the thing
being measured. Shot counts follow from the calibrated selection rate rather than being
fixed, so a circuit that keeps 3% of its shots is run harder than one that keeps 30%
and the error bars are comparable across the sweep.

Random Clifford circuits, `clifford_only=True`: the Stim backend rejects Rx and Rz
outright, which is why the paper simulates rotation-bearing benchmarks through Clifford
proxies rather than directly.

**Structured circuits, no immutable gates.** Uniform-random two-qubit pairs give an
Erdos-Renyi interaction graph with no communities, so every partition is equally good and
the packing segment reads zero for a structural reason -- see
`experiments/shared/random_circuits.structured_program_circuit`. Blocks are planted on
shuffled labels, so recovering them is an achievement rather than a coincidence.
`block_size` is held fixed across the sweep rather than tracking k: the circuit has to be
the same at every point or the x axis confounds "what k does" with "what circuit we drew".
`immutable_ratio=0` because S and the rotations are inert here -- they neither merge nor
move with the placement, so they only add variance to a figure about k.

Run with:

    python -m experiments.figures.lsr_tvd_vs_k.lsr_tvd_vs_k
    python -m experiments.figures.lsr_tvd_vs_k.lsr_tvd_vs_k --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.random_circuits import (
    plant_blocks,
    structured_program_circuit,
)
from experiments.shared.simulation import (
    check_compile,
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from packing.plan import plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

NAME = "lsr_tvd_vs_k"
HERE = Path(__file__).resolve().parent

Perm = List[Tuple[int, int]]

#: (column, axis label, whether higher is better). LSR and TVD are both dimensionless,
#: so neither label carries a unit.
PANELS: Sequence[Tuple[str, str]] = (
    ("lsr", "selection rate"),
    ("tvd", "TVD reduction"),
)

#: Series order fixes appearance: AQORN takes palette position 1, the naive baseline
#: position 2, matching every other figure that compares the two.
SERIES: Sequence[Tuple[str, str]] = (
    ("aqorn", "AQORN packing"),
    ("naive", "naive packing"),
)

DEFAULT_K_VALUES = (2, 4, 6, 8, 10)


@dataclass
class Settings:
    k_values: Sequence[int] = DEFAULT_K_VALUES
    num_qubits: int = 12
    gates_per_qubit: int = 10
    immutable_ratio: float = 0.0
    block_size: int = 4
    locality: float = 0.85
    circuits: int = 10
    circuit_seed: int = 20260729
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    #: Worker processes for the (circuit, k) sweep. None takes `parallel.DEFAULT_JOBS`.
    jobs: Optional[int] = None



def _run_point(job: Tuple[Settings, int, int]) -> List[Dict]:
    """
    One (circuit, k) point: calibrate, plan, and measure both placements.

    Module-level and taking everything it needs as an argument, because `parallel_map`
    spawns rather than forks and a closure cannot cross that boundary. Returns the rows
    for this point -- a list, since one point contributes two series -- and an empty
    list where the old in-line loop would have hit `continue`.
    """
    settings, index, k = job
    n_code = k + 2
    seed = settings.circuit_seed + index
    blocks = plant_blocks(settings.num_qubits, settings.block_size, seed=seed)
    circuit = structured_program_circuit(
        settings.num_qubits,
        settings.gates_per_qubit,
        seed=seed,
        blocks=blocks,
        locality=settings.locality,
        immutable_ratio=settings.immutable_ratio,
        clifford_only=True,
    )
    naive = naive_packing(settings.num_qubits, k)

    calibration = calibrate(
        circuit, naive, settings.sim,
        n_code=n_code,
        baseline_gate_count=baseline_gate_count(circuit, naive, n_code),
    )

    # Let the planner decide whether H commutation pays on this circuit, then honour
    # its answer for *both* rows. `plan.h_commutation` is what the plan was priced
    # under, and translating the placement with the pass in the other state gives a
    # circuit the plan never costed -- measured 30% off, and worse than the naive arm
    # `verify` had just guaranteed it beat.
    try:
        # Bug B6: the planner's noise costs default to zero, so without this it
        # optimises gates and depth while the simulator charges movement and
        # idling -- answering a different question from the one measured.
        plan = plan_packing(
            circuit, n_code, h_commutation="auto", verify=True,
            noise=noise_profile_for(settings.sim),
        )
        aqorn, commute = plan.placement, plan.h_commutation
    except Exception:  # noqa: BLE001
        aqorn, commute = naive, True

    # The pipeline can silently miscompile. On `qpe_n9` at k = 2 the combination
    # (packed placement, merging on, H commutation off) computes the wrong distribution
    # while the selection rate stays at 1.0, so neither post-selection nor
    # `plan_packing(verify=True)` -- which compares cost, not correctness -- catches it.
    # Skip rather than fall back to naive: substituting the baseline for the arm under
    # test pulls the cell towards zero improvement and reads as a result.
    broken = check_compile(
        circuit, naive, aqorn, settings.sim, n_code=n_code, h_commutation=commute,
    )
    if broken:
        print(f"[{NAME}] skipping k {k}: {broken}")
        return []

    # One config, both placements: same alpha, same shot count, same passes, so the
    # difference between the two rows is the placement and nothing else.
    config = build_config(
        settings.sim, n_code=n_code,
        alpha=calibration.alpha, repetitions=calibration.repetitions,
        h_commutation=commute, merging=True,
    )
    measured = {
        "naive": evaluate(circuit, naive, config),
        "aqorn": evaluate(circuit, aqorn, config),
    }
    if any(value is None for value in measured.values()):
        return []

    return [
        {
            "seed": seed,
            "num_qubits": settings.num_qubits,
            "program_gates": circuit.size(),
            "k": k,
            "n_code": n_code,
            "series": series,
            "patches": len({patch for patch, _ in (naive if series == "naive" else aqorn)}),
            "alpha": calibration.alpha,
            "repetitions": calibration.repetitions,
            "pilot_selection_rate": calibration.pilot_selection_rate,
            "h_commutation_chosen": commute,
            "lsr": result["lsr"],
            "tvd": result["tvd"],
        }
        for series, result in measured.items()
    ]


def compute(settings: Settings) -> pd.DataFrame:
    points = [
        (settings, index, k)
        for index in range(settings.circuits)
        for k in settings.k_values
    ]
    produced = parallel_map(_run_point, points, jobs=settings.jobs, label="circuit x k")
    rows: List[Dict] = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def print_summary(frame: pd.DataFrame) -> None:
    print(f"\n[{NAME}] mean over circuits")
    for k in sorted(frame["k"].unique()):
        block = frame.loc[frame["k"] == k]
        parts = []
        for series, label in SERIES:
            sub = block.loc[block["series"] == series]
            if sub.empty:
                continue
            parts.append(
                f"{label}: LSR={sub['lsr'].mean():.4f} TVD={sub['tvd'].mean():.4f}"
            )
        circuits = block["seed"].nunique()
        reps = block["repetitions"].mean()
        commuted = int(block.loc[block["series"] == "aqorn", "h_commutation_chosen"].sum())
        print(
            f"  k={k:<3d} {'   '.join(parts)}   "
            f"[{circuits} circuits, {reps:.0f} reps avg, commutation chosen {commuted}/{circuits}]"
        )


def paired_improvement(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per-circuit percent improvement of AQORN over naive, both signed so up is better.

    Paired, not a difference of the two series' means. The circuits differ from each
    other far more than the placements differ on any one circuit -- circuit-to-circuit
    spread was about twice the effect -- so two lines with their own error bands hide a
    consistent within-circuit effect inside overlapping ribbons. Pairing cancels the
    circuit and leaves the placement, which is the only thing the figure is asking about.
    """
    wide = frame.pivot_table(
        index=["seed", "k"], columns="series", values=[column for column, _ in PANELS]
    )
    rows: List[Dict] = []
    for (seed, k), row in wide.iterrows():
        record = {"seed": seed, "k": k}
        for column, _ in PANELS:
            naive, aqorn = row[(column, "naive")], row[(column, "aqorn")]
            if not naive:
                record[column] = np.nan
                continue
            # LSR: higher is better, so improvement is (aqorn - naive). TVD: lower is
            # better, so it is (naive - aqorn). Both end up "positive means AQORN won".
            gain = (aqorn - naive) if column == "lsr" else (naive - aqorn)
            record[column] = 100.0 * gain / naive
        rows.append(record)
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    improvement = paired_improvement(frame)
    k_values = sorted(improvement["k"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        nrows=len(PANELS), ncols=1,
        figsize=(COLUMN_WIDTH, 1.35 * len(PANELS) + 0.7),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for axis, (column, label) in zip(axes, PANELS):
        means, spreads = [], []
        for k in k_values:
            sub = improvement.loc[improvement["k"] == k, column].dropna()
            means.append(float(sub.mean()) if len(sub) else np.nan)
            spreads.append(float(sub.std(ddof=0)) if len(sub) > 1 else 0.0)
        means, spreads = np.asarray(means), np.asarray(spreads)

        # Zero first and underneath, so "no better than naive" is the thing the eye
        # measures against rather than something it has to find.
        axis.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)
        axis.plot(k_values, means, color=palette[0], marker="o", zorder=3)
        axis.fill_between(
            k_values, means - spreads, means + spreads,
            color=palette[0], alpha=0.15, linewidth=0, zorder=2,
        )
        axis.set_ylabel(label)
        # A signed difference has to show both signs, so this is the documented
        # exception to the y-starts-at-zero rule rather than a truncated axis: zero is
        # on the plot, drawn, and labelled.
        axis.set_axisbelow(True)

    axes[-1].set_xticks(k_values)
    axes[-1].set_xlabel("logical qubits per patch, $k$   ($[[k{+}2,\\,k,\\,2]]$ code)")
    fig.supylabel("AQORN improvement over naive packing (%)")

    # Versioned: the n=12 run is already on disk and the n=20 rerun is a different
    # experiment, not a correction of it. Overwriting would destroy the comparison.
    paths = save_figure(
        fig, f"{NAME}_n{settings.num_qubits}", Path(__file__),
        settings.output_override, versioned=True,
    )
    plt.close(fig)
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values))
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits,
                        help="Program width. Must exceed max(k) so the largest code "
                             "still needs more than one patch.")
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--immutable-ratio", type=float, default=defaults.immutable_ratio,
                        help="Zero here: immutables are inert for a k sweep.")
    parser.add_argument("--block-size", type=int, default=defaults.block_size,
                        help="Planted community size, fixed across the sweep.")
    parser.add_argument("--locality", type=float, default=defaults.locality,
                        help="Probability a 2q gate stays inside its block.")
    parser.add_argument("--circuits", type=int, default=defaults.circuits,
                        help="Random circuits averaged per k. Each costs two "
                             "calibration pilots plus two measured runs.")
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it here: this "
                             "run is hours of simulation.")
    args = parser.parse_args()
    return Settings(
        k_values=args.k_values,
        num_qubits=args.num_qubits,
        gates_per_qubit=args.gates_per_qubit,
        immutable_ratio=args.immutable_ratio,
        block_size=args.block_size,
        locality=args.locality,
        circuits=args.circuits,
        circuit_seed=args.circuit_seed,
        sim=settings_from_args(args, defaults.sim),
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        jobs=args.jobs,
    )


def main() -> None:
    settings = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
    else:
        if settings.num_qubits <= max(settings.k_values):
            raise ValueError(
                f"--num-qubits {settings.num_qubits} is not larger than max k "
                f"{max(settings.k_values)}; the largest code would need one patch."
            )
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            from experiments.shared.figures import data_dir

            target = data_dir(Path(__file__)) / f"{NAME}.csv"
            frame.to_csv(target, index=False)
            print(f"[{NAME}] persisted to {target}")

    print_summary(frame)
    for path in plot(frame, settings):
        print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
