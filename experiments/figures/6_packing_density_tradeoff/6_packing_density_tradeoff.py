"""
6_packing_density_tradeoff.py -- what spreading a program thin actually costs.

**Claim.** Packing density -- logical qubits per patch -- is not a free knob. Spreading a
program over more, emptier patches buys parallelism and cuts depth, and it is paid for in
two-qubit gates, in physical qubits, and in the selection rate those physical qubits drag
down. There is no setting that wins on every metric, which is why the resource decision
has to be made against a machine rather than fixed in a compiler.

x = packing density, `logical qubits / patches`. Density `k` means every patch is full;
density 1 means every qubit sits alone in its own patch. Two panels sharing that axis:
(a) the resource cost -- two-qubit gates, depth, and spacetime volume; (b) the measured
outcome -- selection rate and TVD. Both as percent change from the densest packing, so
five quantities with four different units share one axis honestly.

**Percent change from full density, not absolute values.** Gate counts run to the
hundreds, depth to the tens, and LSR and TVD are bare numbers below one; plotting them
together in absolute terms needs five axes and communicates nothing. Referencing every
series to its own value at the densest packing puts them on one comparable scale and
makes the *shape* of each tradeoff the thing being read, which is the claim.

**Sign convention, stated because it is the one thing a reader can misread.** Every series
is plotted so that **up is worse**: cost metrics are plotted as their raw percent increase,
and selection rate is negated (a fall in LSR is a rise on this axis). TVD needs no
negation, since higher TVD is already worse. So a line rising to the left means "spreading
out hurts this", and lines that diverge are the tradeoff.

**One calibration per (circuit, code size), solved at full density.** Alpha is never solved
per density point -- that would pin the error budget along the exact axis the comparison
lives on and flatten the outcome panel. Every density for a given (circuit, k) shares the
alpha calibrated at that k's densest packing, so the panel measures what spreading does
rather than what the calibrator did.

**n = 12, where TVD is measurable.** A probe across widths found TVD's sampling floor at
0.032 here against a signal several times that, while past about 18 qubits the floor
starts to swallow the result. `1_best_k_phase_diagram` runs wide and reports selection
rate; this figure runs narrow so that panel (b) can carry TVD at all.

**Placement is a deterministic spread, not the planner.** Qubits are dealt round-robin
across the chosen number of patches. Letting AQORN place each point would confound the
density axis with how well the planner happens to do at each density, and this figure is
about the shape of the resource space that the planner is choosing *within*.

The same sweep also carries the code-size axis, which `9_code_size_tradeoff` plots from
this CSV -- one simulation, two readings, and they cannot disagree.

Run with:

    PYTHONPATH=. python experiments/figures/6_packing_density_tradeoff/6_packing_density_tradeoff.py
    PYTHONPATH=. python .../6_packing_density_tradeoff.py --from-csv <path>
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

from experiments.shared.figures import (
    COLUMN_WIDTH,
    MAX_FACTOR,
    apply_factor_axis,
    data_dir,
    factor_from_pct,
    format_factor,
    save_figure,
    use_style,
)
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
from experiments.shared.simulation import (
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    settings_from_args,
)
from experiments.shared.figures import stamp_provenance

NAME = "6_packing_density_tradeoff"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

Perm = List[Tuple[int, int]]

#: Matches every other k figure in the suite. Odd k is not expressible -- `n_code = k + 2`
#: must be even, and the pipeline rejects odd code sizes outright.
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: The code size the density panel is drawn at. k = 6 at n = 12 gives the widest clean
#: density range (patches 2..12, so density 6 down to 1) of any code size in the family.
DEFAULT_DENSITY_K = 6

#: `(column, label)` for the cost panel. Volume is `patches * (n_code + 2) * depth`, the
#: spacetime product `packing.resource.noise.estimate` charges idling against.
COST_SERIES: Sequence[Tuple[str, str]] = (
    ("two_qubit_gates", "2q gates"),
    ("circuit_depth", "depth"),
    ("volume", "volume"),
)

#: `(column, label, negate)` for the outcome panel. See the docstring on sign convention:
#: LSR is negated so that up is worse on both panels.
OUTCOME_SERIES: Sequence[Tuple[str, str, bool]] = (
    ("lsr", "selection rate (inverted)", True),
    ("tvd", "TVD", False),
)


@dataclass
class Settings:
    k_values: Sequence[int] = DEFAULT_K_VALUES
    density_k: int = DEFAULT_DENSITY_K
    num_qubits: int = 12
    gates_per_qubit: int = 10
    block_size: int = 4
    locality: float = 0.85
    circuits: int = 8
    circuit_seed: int = 20260730
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None


def spread_packing(num_qubits: int, patches: int, k: int) -> Perm:
    """
    Deal qubits round-robin across `patches` patches, each holding at most `k`.

    Round-robin rather than block-fill so that a partly-filled configuration spreads the
    shortfall evenly instead of leaving one patch empty -- an empty patch is a different
    resource shape, not a lower density.
    """
    if patches * k < num_qubits:
        raise ValueError(f"{patches} patches of {k} cannot hold {num_qubits} qubits")
    counts = [0] * patches
    placement: Perm = []
    for index in range(num_qubits):
        patch = index % patches
        placement.append((patch, counts[patch]))
        counts[patch] += 1
    return placement


def patch_counts_for(num_qubits: int, k: int) -> List[int]:
    """Every patch count from the densest packing up to one qubit per patch."""
    return list(range(math.ceil(num_qubits / k), num_qubits + 1))


def _run_circuit_k(job) -> List[Dict]:
    """
    One (circuit, k): calibrate at full density, then walk the density ladder.

    The whole ladder is one work item so the calibration is paid once and shared, which
    is what makes the points on a ladder comparable to each other. Module-level because
    `parallel_map` spawns rather than forks.
    """
    settings, index, k = job
    seed = settings.circuit_seed + index
    blocks = plant_blocks(settings.num_qubits, settings.block_size, seed=seed)
    circuit = structured_program_circuit(
        settings.num_qubits, settings.gates_per_qubit, seed=seed, blocks=blocks,
        locality=settings.locality, immutable_ratio=0.0, clifford_only=True,
    )
    n_code = k + 2
    full = naive_packing(settings.num_qubits, k)

    calibration = calibrate(
        circuit, full, settings.sim, n_code=n_code,
        baseline_gate_count=baseline_gate_count(circuit, full, n_code),
    )
    if not math.isfinite(calibration.alpha):
        return []

    rows: List[Dict] = []
    for patches in patch_counts_for(settings.num_qubits, k):
        try:
            placement = spread_packing(settings.num_qubits, patches, k)
        except ValueError:
            continue
        config = build_config(
            settings.sim, n_code=n_code,
            alpha=calibration.alpha, repetitions=calibration.repetitions,
        )
        measured = evaluate(circuit, placement, config)
        if measured is None:
            continue
        rows.append({
            "seed": seed,
            "circuit_index": index,
            "num_qubits": settings.num_qubits,
            "k": k,
            "n_code": n_code,
            "patches": patches,
            "density": settings.num_qubits / patches,
            "alpha": calibration.alpha,
            "repetitions": calibration.repetitions,
            "lsr": measured["lsr"],
            "tvd": measured["tvd"],
            "one_qubit_gates": measured["one_qubit_gates"],
            "two_qubit_gates": measured["two_qubit_gates"],
            "circuit_depth": measured["circuit_depth"],
            # The spacetime product the noise model charges idling against.
            "volume": patches * (n_code + 2) * measured["circuit_depth"],
        })
    return rows


def compute(settings: Settings) -> pd.DataFrame:
    points = [
        (settings, index, k)
        for index in range(settings.circuits)
        for k in settings.k_values
    ]
    total = sum(len(patch_counts_for(settings.num_qubits, k)) for k in settings.k_values)
    print(
        f"[{NAME}] {settings.circuits} circuits x {len(settings.k_values)} code sizes, "
        f"{total} density points per circuit, n = {settings.num_qubits}"
    )
    produced = parallel_map(_run_circuit_k, points, jobs=settings.jobs, label="circuit x k")
    rows = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def relative_to_full(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Percent change from each circuit's densest packing, at one code size.

    Referenced within circuit rather than against a pooled mean: circuits differ in
    absolute gate count by more than the density effect being measured, so a pooled
    reference would report the spread between benchmarks rather than the tradeoff.
    """
    block = frame[frame["k"] == k].copy()
    if block.empty:
        return block

    columns = [name for name, _ in COST_SERIES] + [name for name, _, _ in OUTCOME_SERIES]
    pieces: List[pd.DataFrame] = []
    for index, circuit in block.groupby("circuit_index"):
        circuit = circuit.sort_values("patches")
        reference = circuit.iloc[0]  # fewest patches = densest
        scaled = circuit.copy()
        for column in columns:
            base = reference[column]
            scaled[f"{column}_pct"] = (
                (circuit[column] / base - 1.0) * 100.0 if base else float("nan")
            )
        pieces.append(scaled)
    return pd.concat(pieces, ignore_index=True)


def curves(scaled: pd.DataFrame) -> pd.DataFrame:
    """Mean and +-1 SD per density, over circuits."""
    rows: List[Dict] = []
    columns = [name for name, _ in COST_SERIES] + [name for name, _, _ in OUTCOME_SERIES]
    for density in sorted(scaled["density"].unique()):
        block = scaled[scaled["density"] == density]
        row: Dict = {
            "density": density,
            "patches": int(block["patches"].iloc[0]),
            "circuits": int(block["circuit_index"].nunique()),
        }
        for column in columns:
            values = block[f"{column}_pct"]
            row[f"{column}_pct"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(frame: pd.DataFrame, table: pd.DataFrame, settings: Settings) -> None:
    # Raw percent change, with plain labels. The plot negates selection rate so that up
    # is worse on both panels; the table does not, because a table of signed changes is
    # read directly and an inverted column would be a trap.
    print(f"\n[{NAME}] percent change from the densest packing, k = {settings.density_k}")
    print(f"[{NAME}] raw signs (the figure inverts selection rate so up is worse there)")
    outcome_labels = {"lsr": "selection rate", "tvd": "TVD"}
    header = (
        "  density  patches" + "".join(f"{label:>16}" for _, label in COST_SERIES)
        + "".join(
            f"{outcome_labels.get(column, label):>16}"
            for column, label, _ in OUTCOME_SERIES
        )
    )
    print(header)
    for _, row in table.sort_values("density", ascending=False).iterrows():
        cells = "".join(f"{row[f'{c}_pct']:>16.1f}" for c, _ in COST_SERIES)
        cells += "".join(f"{row[f'{c}_pct']:>16.1f}" for c, _, _ in OUTCOME_SERIES)
        print(f"  {row['density']:>7.2f}  {int(row['patches']):>7}" + cells)

    if len(table) >= 2:
        densest = table.loc[table["density"].idxmax()]
        thinnest = table.loc[table["density"].idxmin()]
        print(
            f"\n[{NAME}] going from density {densest['density']:.2f} "
            f"({int(densest['patches'])} patches) to {thinnest['density']:.2f} "
            f"({int(thinnest['patches'])} patches):"
        )
        # Factors, matching the axis, and uninverted for the same reason the table above
        # is: every line reads "this many times the densest packing".
        for column, label in COST_SERIES:
            value = factor_from_pct(thinnest[f"{column}_pct"], reduction=False, floor=0.0)
            print(f"    {label:<12} {format_factor(value):>8}")
        for column, label, _ in OUTCOME_SERIES:
            plain = {"lsr": "selection rate", "tvd": "TVD"}.get(column, label)
            value = factor_from_pct(thinnest[f"{column}_pct"], reduction=False, floor=0.0)
            print(f"    {plain:<12} {format_factor(value):>8}")

    print(f"\n[{NAME}] absolute values at each code size, full density")
    for k in sorted(frame["k"].unique()):
        block = frame[frame["k"] == k]
        full = block[block["patches"] == block["patches"].min()]
        if full.empty:
            continue
        print(
            f"  k={int(k):<3} patches={int(full['patches'].iloc[0]):<3} "
            f"LSR {full['lsr'].mean():.4f}  TVD {full['tvd'].mean():.4f}  "
            f"2q {full['two_qubit_gates'].mean():.0f}  "
            f"depth {full['circuit_depth'].mean():.0f}  "
            f"volume {full['volume'].mean():.0f}"
        )


def plot(table: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.05), sharex=True)
    top, bottom = axes

    table = table.sort_values("density")
    x = table["density"].to_numpy()

    # Factors of the densest packing, still drawn so **up is worse**. `_pct` is a plain
    # change -- `(value/base - 1) * 100` -- hence `reduction=False`; and a higher-is-better
    # series is inverted rather than negated, since on a ratio axis the mirror of 1.4x is
    # 1/1.4 and a minus sign has nothing to mean. See 9_code_size_tradeoff, which plots the
    # same shape against code size.
    def factors(column: str, shift: int = 0, invert: bool = False) -> np.ndarray:
        pct = table[f"{column}_pct"].to_numpy()
        if shift:
            pct = pct + shift * table[f"{column}_sd"].to_numpy()
        values = np.array(
            [factor_from_pct(p, reduction=False, floor=1.0 / MAX_FACTOR) for p in pct]
        )
        return 1.0 / values if invert else values

    spans: Dict[str, List[float]] = {"top": [], "bottom": []}

    def draw(axis, key: str, column: str, label: str, invert: bool) -> None:
        centre = factors(column, invert=invert)
        # Band converted at its endpoints: a symmetric +-1 SD in percent is asymmetric in
        # factors, so centre +- something would understate one side.
        edges = (factors(column, -1, invert), factors(column, +1, invert))
        low, high = np.minimum(*edges), np.maximum(*edges)
        line, = axis.plot(x, centre, label=label)
        axis.fill_between(x, low, high, color=line.get_color(), alpha=0.15, linewidth=0)
        spans[key].extend([float(np.nanmin(low)), float(np.nanmax(high))])

    for column, label in COST_SERIES:
        draw(top, "top", column, label, False)
    for column, label, negate in OUTCOME_SERIES:
        draw(bottom, "bottom", column, label, negate)

    for axis, key in ((top, "top"), (bottom, "bottom")):
        # 1x is "no different from the densest packing", the reference every series is
        # measured against; `apply_factor_axis` gives it the reserved reference-line style.
        reach = spans[key] + [1.0]
        apply_factor_axis(axis, max(reach), bottom=min(reach))
        axis.legend(frameon=False, fontsize=8, loc="best")
        # Density falls left to right in the natural reading of "spreading out", so the
        # axis is reversed: the densest packing (the reference) sits at the left.
        axis.set_xlim(x.max(), x.min())

    # Literal U+00D7 rather than mathtext, which `svg.fonttype: none` would re-outline.
    top.set_ylabel("change from densest (×)")
    bottom.set_ylabel("change from densest (×)")
    bottom.set_xlabel(
        f"packing density (logical qubits per patch), $k={settings.density_k}$"
    )

    for axis, tag in zip(axes, "ab"):
        axis.annotate(
            f"({tag})", xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-32, 2), textcoords="offset points",
            fontsize=10, fontweight="bold", va="bottom",
        )

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values))
    parser.add_argument("--density-k", type=int, default=defaults.density_k,
                        help="Code size the density panels are drawn at. The sweep still "
                             "covers every --k-values entry, for 7_code_size_tradeoff.")
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits)
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--locality", type=float, default=defaults.locality)
    parser.add_argument("--circuits", type=int, default=defaults.circuits)
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    return Settings(
        k_values=args.k_values,
        density_k=args.density_k,
        num_qubits=args.num_qubits,
        gates_per_qubit=args.gates_per_qubit,
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

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
    else:
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Stamp the cost model that produced these rows, so a future run can tell
        # whether this cache still describes the current planner. See packing/version.py.
        frame = stamp_provenance(frame)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            kept = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(kept, index=False)
            print(f"[{NAME}] persisted to {kept}")

    scaled = relative_to_full(frame, settings.density_k)
    if scaled.empty:
        raise RuntimeError(f"no rows at k = {settings.density_k}")
    table = curves(scaled)
    print_summary(frame, table, settings)
    plot(table, settings)


if __name__ == "__main__":
    main()
