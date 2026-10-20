"""
7_improvement_vs_immutable_fraction.py -- does AQORN stop helping as gates stop fusing?

**Claim.** Merging works by fusing adjacent single-qubit gates into whole-patch macros, and
S, T and the rotations cannot be fused -- they bound its reach. A reader is entitled to
ask whether AQORN's benefit is therefore an artefact of benchmarks that happen to be
merge-friendly, and whether it evaporates on circuits dominated by immutable gates. This
figure answers that on real benchmarks by regressing the measured improvement on the share
of gates that cannot fuse.

x = immutable share of program operations (`s`, `sdg`, `rz`, `rx`, `t`, `tdg`), which runs
from 0 to 83% across the 71 QASMBench proxies. y = full-pipeline improvement over naive
packing. One series per metric, each with a least-squares fit and its correlation.

**On real benchmarks, because the random-circuit version of this axis was already measured
and retired.** `IMMUTABLE_AXIS.md` records why: the structured generator holds gate count
and two-qubit ratio fixed, so raising the immutable share only swaps H/X/Z for S and never
touches the interaction graph. That gave r = -0.09 on TVD and about a third of locality's
leverage elsewhere -- a null result about the generator rather than about compilation. On
benchmarks the immutable share co-varies with everything else a real program does, which is
the measurement the question actually asks for.

**No simulation of its own.** The improvement columns come from the 71-circuit benchmark
sweep in `merging_improvement/`, which already measures gate counts, depth, selection rate
and TVD per circuit under the full pipeline. The immutable share is static analysis of the
same proxy circuits -- `random_circuits.immutable_fraction`, counting operations. So this
figure is a join, and re-running it is free.

**Improvements are summed over the three pipeline stages.** The source CSV records H
commutation, merging and packing as *incremental* contributions, so the full-pipeline
figure for a circuit is their sum. That is the quantity a reader means by "how much did
AQORN help on this circuit", and it is what the abstract's numbers aggregate.

**Read the fit, not the individual points.** A single benchmark's improvement depends on far
more than its immutable share -- width, interaction locality, and depth all move it -- so
the scatter is wide by construction. The claim is about the slope: if it is flat, the
immutable fraction does not govern whether AQORN helps, which is the answer to the question
asked. Correlations for every metric are printed, including the ones not plotted.

Run with:

    PYTHONPATH=. python experiments/figures/7_improvement_vs_immutable_fraction/7_improvement_vs_immutable_fraction.py
    PYTHONPATH=. python .../7_improvement_vs_immutable_fraction.py --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style
from experiments.shared.random_circuits import immutable_fraction

NAME = "7_improvement_vs_immutable_fraction"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: The 71-circuit benchmark sweep this reads. See the module docstring: this figure runs
#: no simulation, it joins that sweep against a static count.
DEFAULT_SOURCE = (
    SCRIPT.parents[1] / "merging_improvement"
    / "merging_improvement_benchmark_stylized_simulated.csv"
)

#: `(column, label, plotted)`. Every metric is reported; three are drawn, because seven
#: fitted scatters at column width is texture rather than a reading. The three drawn are
#: the ones the paper leads with.
METRICS: Sequence[Tuple[str, str, bool]] = (
    ("pct_reduction_two_qubit_gates", "2q gates", True),
    ("pct_reduction_circuit_depth", "depth", True),
    ("pct_reduction_logical_total_variation_distance", "TVD", True),
    ("pct_reduction_one_qubit_gates", "1q gates", False),
    ("pct_improvement_logical_selection_rate", "selection rate", False),
)


@dataclass
class Settings:
    source: Path = field(default_factory=lambda: DEFAULT_SOURCE)
    benchmark_root: Path = field(default_factory=lambda: Path("benchmarks/qasmbench"))
    output_override: Optional[Path] = None
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")


def build(settings: Settings) -> pd.DataFrame:
    """Join per-circuit full-pipeline improvement against the static immutable share."""
    frame = pd.read_csv(settings.source)
    columns = [column for column, _, _ in METRICS]

    # Incremental per stage, so the full-pipeline number is the sum over stages.
    totals = frame.groupby("circuit_name")[columns].sum()
    context = frame.groupby("circuit_name").agg(
        num_qubits=("num_qubits", "first"),
        size_group=("size_group", "first"),
        baseline_gate_count=("baseline_gate_count", "first"),
    )

    loaded = dict(load_proxy_circuits(settings.benchmark_root, qubit_limit=None))
    rows: List[Dict] = []
    missing: List[str] = []
    for name, row in totals.iterrows():
        circuit = loaded.get(name)
        if circuit is None:
            missing.append(name)
            continue
        entry: Dict = {
            "circuit": name,
            "immutable_fraction": immutable_fraction(circuit),
            "program_gates": circuit.size(),
        }
        entry.update({column: float(row[column]) for column in columns})
        entry.update(context.loc[name].to_dict())
        rows.append(entry)

    if missing:
        print(f"[{NAME}] {len(missing)} circuits in the sweep were not loadable: "
              f"{', '.join(sorted(missing)[:6])}")
    return pd.DataFrame(rows)


def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Least squares plus Pearson r and a two-sided p value for the slope.

    The p value is what decides whether a visually flat line is genuinely flat or merely
    under-powered, which is the whole question this figure is asked to settle.
    """
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    count = len(x)
    if abs(correlation) >= 1.0:
        return float(slope), float(intercept), correlation, 0.0
    # t = r sqrt(n-2) / sqrt(1-r^2), against a t distribution with n-2 dof. The normal
    # approximation is used for the tail, which is close enough at n = 71.
    statistic = correlation * np.sqrt(count - 2) / np.sqrt(1 - correlation ** 2)
    from math import erfc, sqrt
    p_value = float(erfc(abs(statistic) / sqrt(2)))
    return float(slope), float(intercept), correlation, p_value


def print_summary(table: pd.DataFrame) -> None:
    x = table["immutable_fraction"].to_numpy() * 100.0
    print(f"\n[{NAME}] {len(table)} QASMBench circuits")
    print(
        f"[{NAME}] immutable share: {x.min():.1f}% to {x.max():.1f}%, "
        f"median {np.median(x):.1f}%, {(x > 0).sum()} circuits above zero"
    )
    print(f"\n  metric            mean impr.   slope (pp per 10pp)      r      p")
    for column, label, _ in METRICS:
        y = table[column].to_numpy()
        slope, _, correlation, p_value = _fit(x, y)
        print(
            f"  {label:<16} {y.mean():>8.2f}%   {slope * 10:>16.2f}   "
            f"{correlation:>6.3f}  {p_value:>5.3f}"
        )

    significant = [
        label for column, label, _ in METRICS
        if _fit(x, table[column].to_numpy())[3] < 0.05
    ]
    if significant:
        print(
            f"\n[{NAME}] READING: the immutable share does predict improvement for "
            f"{', '.join(significant)} (p < 0.05). Report the slope with the mean."
        )
    else:
        print(
            f"\n[{NAME}] READING: no metric's improvement depends on the immutable share "
            f"at p < 0.05. AQORN's benefit is not an artefact of merge-friendly "
            f"benchmarks, which is the answer to the question."
        )


def plot(table: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.80))

    x = table["immutable_fraction"].to_numpy() * 100.0
    grid = np.linspace(0.0, x.max(), 20)

    for column, label, plotted in METRICS:
        if not plotted:
            continue
        y = table[column].to_numpy()
        slope, intercept, correlation, _ = _fit(x, y)
        handle = axis.scatter(x, y, s=10, alpha=0.55, edgecolors="none")
        # marker="" because the style's cycler would otherwise stud the regression line
        # with data markers, making a fit read as a measured series.
        axis.plot(
            grid, slope * grid + intercept,
            color=handle.get_facecolor()[0], linewidth=1.4, marker="",
            label=f"{label}  ($r={correlation:+.2f}$)",
        )

    # Zero is "no better than naive packing", so it gets the reserved reference styling.
    axis.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)
    # Plain "%", not "\%": matplotlib's mathtext is not LaTeX and renders the backslash
    # literally, which printed "(\% of program operations)" on the first draw.
    axis.set_xlabel("immutable gates (% of program operations)")
    axis.set_ylabel("improvement over naive packing (%)")
    axis.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--from-csv", type=Path, default=defaults.source,
        help="Per-circuit benchmark sweep. This figure runs no simulation of its own.",
    )
    parser.add_argument("--benchmark-root", type=Path, default=defaults.benchmark_root)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    return Settings(
        source=args.from_csv,
        benchmark_root=args.benchmark_root,
        csv_path=args.csv,
        output_override=args.output_dir,
    )


def main() -> None:
    settings = parse_args()
    if not settings.source.exists():
        raise FileNotFoundError(
            f"{settings.source} not found. It is produced by "
            f"merging_improvement_benchmark_stylized_simulated.py."
        )
    table = build(settings)
    if table.empty:
        raise RuntimeError("no circuit matched between the sweep and the benchmark set")
    settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(settings.csv_path, index=False)
    print(f"[{NAME}] wrote {len(table)} rows to {settings.csv_path}")
    print_summary(table)
    plot(table, settings)


if __name__ == "__main__":
    main()
