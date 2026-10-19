"""
2_tvd_vs_noise_ratio_by_k.py -- what a patch can still detect, as the noise changes shape.

**Claim.** A `[[k+2, k, 2]]` patch puts `k` logical qubits behind one pair of detectors, so
raising `k` costs accuracy. This figure asks whether the machine's noise profile ever pays
that cost back: it sweeps which mechanism dominates -- qubits idling, or gates travelling
between patches -- and reads total variation distance at each code size.

The companion figure `code_size_crossover` shows that *selection rate* has a crossover
along this axis: a large code keeps more shots where movement dominates, because it needs
fewer patches and so fewer crossings. The question here is whether the shots it keeps are
any good. If TVD crosses too, a large code is simply better on movement-limited hardware.
If TVD does not cross, the throughput win is bought with accuracy and the two halves of
the resource decision pull against each other -- which is a sharper statement, and the one
the selector has to arbitrate.

x = the **measured** idle/movement noise ratio -- what the schedule actually charged,
read off `measured_regime` for each arm's own translated circuit. The nominal ratios
(`SimSettings.with_noise_ratio`, geometric mean pinned at one) are only the sweep that
generates coverage; plotting the nominal value asserted a balance the model does not
have, since the mapping from scalars to regime depends on the circuit, the placement
and `k`. Each k line therefore samples its own x positions -- that is the honest
version of the axis, not a defect. y = TVD against the ideal unencoded distribution.
One line per `k`.

**Placement is naive at every k, deliberately.** This figure is about what a patch can
detect, not about how well AQORN arranges patches. Running the planner here would let
placement quality vary along the axis and confound it with the code size, so every arm
uses `naive_packing` -- fill patches in program order. `code_size_crossover` and
`1_best_k_phase_diagram` measure the deployed compiler; this one measures the code.

**Logical width is fixed and the patch count follows from k.** All arms run the same
16-qubit programs, so `k` sets the patch count: 8, 4, 3, 2, 2 across k = 2..10. That is
the real deployment tradeoff -- choosing a code size *is* choosing how thinly the program
spreads -- and it is what makes the ratio axis bite, since patch count is exactly what the
movement term charges for. Holding the patch count fixed instead would isolate detection
strength, but every k would then run a different-width program and circuit content would
no longer be controlled.

**Why 16 qubits, measured rather than assumed.** TVD has a sampling floor: two finite
samples of a state spread over many outcomes score a nonzero distance however clean the
circuit is, and past some width that floor swallows the result -- which is why
`code_size_crossover` records TVD at 24-40 qubits and refuses to plot it. A probe across
widths 10-18 at this shot budget measured the floor (alpha ~ 0) against the signal:

    n     floor   TVD k=2 -> k=10   signal/floor
    10    0.010     0.021 -> 0.221     2.5 - 24.1
    12    0.032     0.057 -> 0.091     1.8 -  2.9
    14    0.052     0.081 -> 0.126     1.6 -  2.5
    16    0.037     0.059 -> 0.108     1.6 -  2.8
    18    0.132     0.189 -> 0.249     1.4 -  1.9

16 gives the best floor-to-spread ratio at a width whose patch ladder still separates. It
is also why `--target-kept-shots` defaults far above the shared default here: the floor
falls as 1/sqrt(accepted shots), so shots buy headroom directly.

**The floor is measured and drawn, not argued away.** Every (circuit, k) is additionally
run at alpha ~ 0, and the mean of those is plotted as a reference line. A reader asking
"is that separation just the sampling floor?" can see the answer instead of taking the
caption's word for it. It also guards the figure against its own failure mode: if a line
sits on the floor, the figure says so on its face.

**One noise calibration per (circuit, ratio), solved at the reference code size.** Never
per `k` -- that would pin the error budget along the exact axis the comparison lives on and
flatten every line. This is the convention `code_size_crossover` arrived at the expensive
way, inherited rather than re-derived.

Bands are +-1 standard deviation over circuits, the house default; the spread is over
circuits rather than shots, and it is wide because benchmarks differ from each other more
than the code sizes do at a fixed ratio. The caption has to say so.

Run with:

    PYTHONPATH=. python experiments/figures/2_tvd_vs_noise_ratio_by_k/2_tvd_vs_noise_ratio_by_k.py
    PYTHONPATH=. python .../2_tvd_vs_noise_ratio_by_k.py --from-csv <path>
    PYTHONPATH=. python .../2_tvd_vs_noise_ratio_by_k.py --circuits 2 --ratios 0.25 1 4
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.figures import COLUMN_WIDTH, data_dir, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
from experiments.shared.simulation import (
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    measured_regime,
    naive_packing,
    settings_from_args,
)

NAME = "2_tvd_vs_noise_ratio_by_k"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: Same span and spacing as `1_best_k_phase_diagram`, so the two ratio axes are directly
#: comparable. Log-spaced in powers of four and symmetric about 1.0.
DEFAULT_RATIOS: Tuple[float, ...] = (1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0)

#: Matches every other k figure in the suite. Odd k is not expressible -- `n_code = k + 2`
#: must be even, and the pipeline rejects odd code sizes outright.
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: Program width. See the module docstring: measured as the best floor-to-spread tradeoff
#: at a width whose patch ladder (8, 4, 3, 2, 2) still separates.
DEFAULT_NUM_QUBITS = 16

#: alpha for the floor arm. Not zero, because zero noise short-circuits parts of the
#: pipeline; small enough that any TVD measured is sampling, not physics.
FLOOR_ALPHA = 1e-9


@dataclass
class Settings:
    ratios: Sequence[float] = DEFAULT_RATIOS
    k_values: Sequence[int] = DEFAULT_K_VALUES
    num_qubits: int = DEFAULT_NUM_QUBITS
    gates_per_qubit: int = 10
    block_size: int = 4
    locality: float = 0.85
    immutable_ratio: float = 0.0
    circuits: int = 8
    circuit_seed: int = 20260730
    #: Whether to measure the alpha ~ 0 sampling floor alongside the sweep.
    measure_floor: bool = True
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None


def _circuit_for(settings: Settings, index: int):
    """The same program for every k and ratio at this index, so only the arm varies."""
    seed = settings.circuit_seed + index
    blocks = plant_blocks(settings.num_qubits, settings.block_size, seed=seed)
    return seed, structured_program_circuit(
        settings.num_qubits,
        settings.gates_per_qubit,
        seed=seed,
        blocks=blocks,
        locality=settings.locality,
        immutable_ratio=settings.immutable_ratio,
        clifford_only=True,
    )


def _run_column(job) -> List[Dict]:
    """
    One (circuit, ratio): calibrate once at the reference k, then measure every k.

    Module-level and self-contained because `parallel_map` spawns rather than forks. The
    unit of work is a whole ratio column so the two calibration pilots are paid once and
    shared by every k in it -- which is also what makes the column a fair comparison.
    """
    settings, index, ratio = job
    seed, circuit = _circuit_for(settings, index)
    sim = settings.sim.with_noise_ratio(ratio)
    reference = min(settings.k_values)
    naive_reference = naive_packing(settings.num_qubits, reference)

    calibration = calibrate(
        circuit, naive_reference, sim,
        n_code=reference + 2,
        baseline_gate_count=baseline_gate_count(circuit, naive_reference, reference + 2),
    )
    if not math.isfinite(calibration.alpha):
        return []

    rows: List[Dict] = []
    for k in settings.k_values:
        placement = naive_packing(settings.num_qubits, k)
        config = build_config(
            sim, n_code=k + 2,
            alpha=calibration.alpha, repetitions=calibration.repetitions,
        )
        measured = evaluate(circuit, placement, config)
        if measured is None:
            continue
        # The x value: what this arm's schedule actually charges at these scalars,
        # measured on the same translated configuration `evaluate` simulated.
        # Alpha-invariant, so the calibrated alpha above does not move it.
        regime = measured_regime(
            circuit, placement, n_code=k + 2,
            idle_time_scalar=sim.idle_time_scalar,
            distance_scalar=sim.distance_scalar,
            h_commutation=True, merging=True,
        )
        rows.append({
            "seed": seed,
            "circuit_index": index,
            "num_qubits": settings.num_qubits,
            "program_gates": circuit.size(),
            "ratio": ratio,
            "measured_ratio": regime["measured_ratio"],
            "movement_share": regime["movement_share"],
            "idle_scalar": sim.idle_time_scalar,
            "distance_scalar": sim.distance_scalar,
            "k": k,
            "n_code": k + 2,
            "patches": len({patch for patch, _ in placement}),
            "arm": "measured",
            "alpha": calibration.alpha,
            "repetitions": calibration.repetitions,
            "lsr": measured["lsr"],
            "tvd": measured["tvd"],
            "accepted_shots": calibration.repetitions * measured["lsr"],
            "one_qubit_gates": measured["one_qubit_gates"],
            "two_qubit_gates": measured["two_qubit_gates"],
            "circuit_depth": measured["circuit_depth"],
        })
    return rows


def _run_floor(job) -> List[Dict]:
    """
    The alpha ~ 0 arm: whatever TVD reads here is the sampling floor, not noise.

    Independent of the ratio -- with no noise there is nothing for the mechanisms to
    scale -- so this runs once per (circuit, k) rather than per column.
    """
    settings, index, k = job
    seed, circuit = _circuit_for(settings, index)
    placement = naive_packing(settings.num_qubits, k)
    # Shots matched to what the measured arm will land on, so the floor is the floor of
    # this figure's budget rather than of some other one.
    repetitions = settings.sim.max_repetitions
    config = build_config(
        settings.sim, n_code=k + 2, alpha=FLOOR_ALPHA, repetitions=repetitions
    )
    measured = evaluate(circuit, placement, config)
    if measured is None:
        return []
    return [{
        "seed": seed,
        "circuit_index": index,
        "num_qubits": settings.num_qubits,
        "program_gates": circuit.size(),
        "ratio": float("nan"),
        "idle_scalar": float("nan"),
        "distance_scalar": float("nan"),
        "k": k,
        "n_code": k + 2,
        "patches": len({patch for patch, _ in placement}),
        "arm": "floor",
        "alpha": FLOOR_ALPHA,
        "repetitions": repetitions,
        "lsr": measured["lsr"],
        "tvd": measured["tvd"],
        "accepted_shots": repetitions * measured["lsr"],
        "one_qubit_gates": measured["one_qubit_gates"],
        "two_qubit_gates": measured["two_qubit_gates"],
        "circuit_depth": measured["circuit_depth"],
    }]


def compute(settings: Settings) -> pd.DataFrame:
    columns = [
        (settings, index, ratio)
        for index in range(settings.circuits)
        for ratio in settings.ratios
    ]
    print(
        f"[{NAME}] {settings.circuits} circuits x {len(settings.ratios)} ratios x "
        f"{len(settings.k_values)} code sizes at n = {settings.num_qubits}"
    )
    produced = parallel_map(_run_column, columns, jobs=settings.jobs, label="circuit x ratio")
    rows = [row for batch in produced if batch for row in batch]

    if settings.measure_floor:
        floors = [
            (settings, index, k)
            for index in range(settings.circuits)
            for k in settings.k_values
        ]
        print(f"[{NAME}] {len(floors)} floor points at alpha = {FLOOR_ALPHA:g}")
        produced = parallel_map(_run_floor, floors, jobs=settings.jobs, label="floor")
        rows += [row for batch in produced if batch for row in batch]

    return pd.DataFrame(rows)


def curves(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Mean TVD and LSR per (k, ratio), with the spread over circuits."""
    measured = frame[frame["arm"] == "measured"]
    rows: List[Dict] = []
    for k in settings.k_values:
        for ratio in sorted(measured["ratio"].unique()):
            block = measured[(measured["k"] == k) & (measured["ratio"] == ratio)]
            if block.empty:
                continue
            count = max(1, len(block))
            root = math.sqrt(count)
            tvd_sd = float(block["tvd"].std(ddof=1)) if len(block) > 1 else 0.0
            lsr_sd = float(block["lsr"].std(ddof=1)) if len(block) > 1 else 0.0
            finite = block[np.isfinite(block["measured_ratio"])]
            rows.append({
                "k": k,
                "ratio": ratio,
                # Geometric mean: the axis is a log-scale ratio, and each circuit
                # lands at its own measured value.
                "measured_ratio": float(
                    np.exp(np.log(finite["measured_ratio"]).mean())
                ) if not finite.empty else float("nan"),
                "movement_share": float(block["movement_share"].mean()),
                "patches": float(block["patches"].mean()),
                "tvd": float(block["tvd"].mean()),
                "tvd_sd": tvd_sd,
                # Standard *error*, and the band is drawn from this rather than the SD.
                # Every k here is measured on the same circuits at the same alpha, so the
                # comparison is paired within circuit and the SD describes how much
                # benchmarks differ from each other -- which is large, and not the
                # question. Measured: SD spans 0.03-0.20 against means of 0.088-0.131, so
                # SD bands overlap into a single block and hide the ordering the figure
                # exists to show. `code_size_crossover` reaches the same conclusion for
                # the same reason. The caption must say the band is +-1 SE over circuits.
                "tvd_se": tvd_sd / root,
                "lsr": float(block["lsr"].mean()),
                "lsr_sd": lsr_sd,
                "lsr_se": lsr_sd / root,
                "accepted_shots": float(block["accepted_shots"].mean()),
                "circuits": int(block["circuit_index"].nunique()),
            })
    return pd.DataFrame(rows)


def floor_level(frame: pd.DataFrame) -> Optional[float]:
    """Mean sampling floor across every (circuit, k), or None when it was not measured."""
    floor = frame[frame["arm"] == "floor"]
    if floor.empty:
        return None
    return float(floor["tvd"].mean())


def print_summary(frame: pd.DataFrame, table: pd.DataFrame, settings: Settings) -> None:
    level = floor_level(frame)
    ratios = sorted(table["ratio"].unique())

    print(f"\n[{NAME}] measured idle/movement ratio by nominal ratio and k")
    header_m = "  nominal".ljust(12) + "".join(
        f"{'k=' + str(k):>10}" for k in settings.k_values
    )
    print(header_m)
    for ratio in ratios:
        cells = []
        for k in settings.k_values:
            row = table[(table["k"] == k) & (table["ratio"] == ratio)]
            cells.append(
                f"{'-':>10}" if row.empty
                else f"{row.iloc[0]['measured_ratio']:>10.3f}"
            )
        print(f"  {_tick_label(ratio):<10}" + "".join(cells))

    print(f"\n[{NAME}] TVD by ratio and k (mean over circuits)")
    header = "  ratio".ljust(12) + "".join(f"{'k=' + str(k):>10}" for k in settings.k_values)
    print(header)
    for ratio in ratios:
        cells = []
        for k in settings.k_values:
            row = table[(table["k"] == k) & (table["ratio"] == ratio)]
            cells.append(f"{'-':>10}" if row.empty else f"{row.iloc[0]['tvd']:>10.4f}")
        print(f"  {_tick_label(ratio):<10}" + "".join(cells))

    if level is not None:
        print(f"\n[{NAME}] sampling floor (alpha = {FLOOR_ALPHA:g}): TVD = {level:.4f}")
        worst = table["tvd"].min()
        print(
            f"[{NAME}] lowest plotted TVD is {worst:.4f}, "
            f"{worst / level:.2f}x the floor"
        )

    print(f"\n[{NAME}] selection rate by ratio and k (the other half of the tradeoff)")
    print(header)
    for ratio in ratios:
        cells = []
        for k in settings.k_values:
            row = table[(table["k"] == k) & (table["ratio"] == ratio)]
            cells.append(f"{'-':>10}" if row.empty else f"{row.iloc[0]['lsr']:>10.4f}")
        print(f"  {_tick_label(ratio):<10}" + "".join(cells))

    print(f"\n[{NAME}] best (lowest-TVD) k at each ratio")
    crossed = set()
    for ratio in ratios:
        column = table[table["ratio"] == ratio]
        if column.empty:
            continue
        best = column.loc[column["tvd"].idxmin()]
        runner = column[column["k"] != best["k"]]["tvd"].min()
        crossed.add(int(best["k"]))
        margin = (runner / best["tvd"] - 1.0) * 100.0 if best["tvd"] > 0 else float("nan")
        print(
            f"  ratio {_tick_label(ratio):>6}: k={int(best['k'])} "
            f"(TVD {best['tvd']:.4f}, {margin:.1f}% better than the next)"
        )

    if len(crossed) == 1:
        only = crossed.pop()
        print(
            f"\n[{NAME}] NO CROSSOVER: k={only} has the lowest TVD at every ratio. "
            f"The selection-rate crossover in code_size_crossover is therefore bought "
            f"with accuracy rather than shared with it."
        )
    else:
        print(
            f"\n[{NAME}] CROSSOVER: the lowest-TVD code size changes across the axis "
            f"({sorted(crossed)})."
        )

    print(f"\n[{NAME}] patch count by k: " + ", ".join(
        f"k={k}: {int(table[table['k'] == k]['patches'].iloc[0])}"
        for k in settings.k_values if not table[table["k"] == k].empty
    ))
    print(f"[{NAME}] accepted shots per point: "
          f"{table['accepted_shots'].min():,.0f}-{table['accepted_shots'].max():,.0f}")


def _tick_label(ratio: float) -> str:
    """Powers of four read as fractions below one and integers above it."""
    if ratio >= 1:
        return f"{int(round(ratio))}"
    return f"1/{int(round(1 / ratio))}"


def plot(frame: pd.DataFrame, table: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.80))

    # Series in k order, so each k keeps one appearance and the style's zipped colour,
    # marker and dash cyclers advance together -- greyscale-safe without passing colours.
    for k in settings.k_values:
        block = table[table["k"] == k].sort_values("measured_ratio")
        block = block[np.isfinite(block["measured_ratio"])]
        if block.empty:
            continue
        x = block["measured_ratio"].to_numpy()
        y = block["tvd"].to_numpy()
        spread = block["tvd_se"].to_numpy()
        patches = int(block["patches"].iloc[0])
        line, = axis.plot(x, y, label=f"$k={k}$ ({patches} patches)")
        axis.fill_between(
            x, y - spread, y + spread,
            color=line.get_color(), alpha=0.15, linewidth=0,
        )

    level = floor_level(frame)
    if level is not None:
        # #636363 is the style's reserved reference-line colour.
        axis.axhline(level, color="#636363", linestyle="--", linewidth=1.0, zorder=1)
        axis.annotate(
            "sampling floor",
            xy=(x[0], level), xytext=(0, 3), textcoords="offset points",
            fontsize=8, color="#636363", va="bottom", ha="left",
        )

    axis.set_xscale("log")
    axis.set_xlabel("measured idle / movement noise ratio")
    axis.set_ylabel("total variation distance")
    axis.set_ylim(bottom=0.0)
    # Two columns below the data rather than "best": five entries stacked in a corner
    # cover the movement-dominated end, which is where two of the lines cross.
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(defaults.ratios))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values))
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits,
                        help="Program width. See the module docstring: 16 is where the "
                             "TVD sampling floor and the patch ladder both behave.")
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--locality", type=float, default=defaults.locality)
    parser.add_argument("--circuits", type=int, default=defaults.circuits,
                        help="Random circuits averaged per (k, ratio).")
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    parser.add_argument("--no-floor", action="store_true",
                        help="Skip the alpha~0 arm. The floor is what makes the "
                             "separation interpretable, so only skip it for a smoke test.")
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/.")
    args = parser.parse_args()
    # TVD's floor falls as 1/sqrt(accepted shots), so this figure buys headroom with
    # shots and defaults well above the shared default.
    sim = settings_from_args(args, defaults.sim)
    if args.target_kept_shots == SimSettings().target_kept_shots:
        sim = replace(sim, target_kept_shots=40_000)
    return Settings(
        ratios=args.ratios,
        k_values=args.k_values,
        num_qubits=args.num_qubits,
        gates_per_qubit=args.gates_per_qubit,
        block_size=args.block_size,
        locality=args.locality,
        circuits=args.circuits,
        circuit_seed=args.circuit_seed,
        measure_floor=not args.no_floor,
        sim=sim,
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        jobs=args.jobs,
    )


def main() -> None:
    settings = parse_args()

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
        if "measured_ratio" not in frame.columns:
            raise SystemExit(
                f"[{NAME}] {settings.from_csv} predates the measured-regime axis "
                "(no measured_ratio column) -- regenerate rather than replot."
            )
    else:
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            kept = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(kept, index=False)
            print(f"[{NAME}] persisted to {kept}")

    table = curves(frame, settings)
    if table.empty:
        raise RuntimeError("no measured arm survived")
    print_summary(frame, table, settings)
    plot(frame, table, settings)


if __name__ == "__main__":
    main()
