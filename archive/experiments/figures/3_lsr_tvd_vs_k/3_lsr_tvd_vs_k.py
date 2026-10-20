"""
3_lsr_tvd_vs_k.py -- what a larger code costs, in absolute terms.

**Claim.** Raising `k` buys nothing in throughput and costs accuracy. Selection rate is
roughly flat across the family -- it even peaks in the middle -- while total variation
distance degrades monotonically, because more logical qubits sit behind the same single
pair of detectors. That is the cost side of the code-size decision, and it is what makes
`k` a real choice rather than a free parameter to turn up.

x = logical qubits per patch `k`. Two stacked panels sharing it: (a) logical selection
rate, (b) TVD. Two series in each, AQORN packing and the naive baseline.

**This plots absolute LSR and TVD, where `lsr_tvd_vs_k` plots the improvement between the
two placements.** Both readings come from the same sweep and both are worth having, but
they answer different questions and the improvement framing cannot answer this one. At
n = 20 the paired AQORN-vs-naive gains carry a standard deviation of 3-12 points against
means of 2-24, so the improvement panels are bands straddling zero -- `FIGURE_PLAN` §P5
reaches the same conclusion and says to read this figure for the `k` trend and let the
placement figures carry the placement comparison. The absolute series are far better
behaved because the `k` trend is within-circuit and monotone, and they are what the claim
above actually needs.

**No simulation of its own.** The rows come from `lsr_tvd_vs_k`'s sweep, which already
measures exactly what this needs: both placements, every `k`, one alpha per circuit. This
script is a second reading of that CSV, so restyling it is free and the two figures can
never disagree about the underlying numbers.

**Read at n = 20, not the n = 12 the sweep used to run at.** At 12 qubits the patch ladder
saturates -- k = 6, 8 and 10 are all two patches -- so the top three points of the x axis
are the same resource shape and the trend flattens for a reason that has nothing to do
with the code. At 20 the ladder is 10 / 5 / 4 / 3 / 2 and every point is a distinct shape.

Bands are +-1 standard deviation over the circuits in the sweep. The spread is over
circuits, not shots; the caption has to say so and say how many.

Run with:

    PYTHONPATH=. python experiments/figures/3_lsr_tvd_vs_k/3_lsr_tvd_vs_k.py
    PYTHONPATH=. python .../3_lsr_tvd_vs_k.py --from-csv <path>
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
import pandas as pd

from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

NAME = "3_lsr_tvd_vs_k"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: The sweep this reads. `lsr_tvd_vs_k` writes it; see the module docstring on why this
#: figure does not simulate anything itself.
DEFAULT_SOURCE = (
    SCRIPT.parents[1] / "lsr_tvd_vs_k" / "lsr_tvd_vs_k_n20.csv"
)

#: `(column value, legend label)`. AQORN takes palette position 1 and the naive baseline
#: position 2, matching every other figure that compares the two.
SERIES: Sequence[Tuple[str, str]] = (
    ("aqorn", "AQORN packing"),
    ("naive", "naive packing"),
)

#: `(column, panel label)`. Selection rate first: it is the one that stays flat, and the
#: TVD panel below it is the cost that flatness conceals.
PANELS: Sequence[Tuple[str, str]] = (
    ("lsr", "logical selection rate"),
    ("tvd", "total variation distance"),
)


@dataclass
class Settings:
    source: Path = field(default_factory=lambda: DEFAULT_SOURCE)
    output_override: Optional[Path] = None


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and +-1 SD per (series, k), plus the patch count that k implies."""
    rows: List[Dict] = []
    for series, _ in SERIES:
        block = frame[frame["series"] == series]
        for k in sorted(block["k"].unique()):
            cell = block[block["k"] == k]
            row: Dict = {
                "series": series,
                "k": int(k),
                "patches": float(cell["patches"].mean()),
                "circuits": int(len(cell)),
            }
            for column, _ in PANELS:
                row[column] = float(cell[column].mean())
                row[f"{column}_sd"] = (
                    float(cell[column].std(ddof=1)) if len(cell) > 1 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(table: pd.DataFrame, frame: pd.DataFrame) -> None:
    qubits = int(frame["num_qubits"].iloc[0]) if "num_qubits" in frame else -1
    print(f"\n[{NAME}] absolute LSR and TVD by k, n = {qubits}")
    for series, label in SERIES:
        block = table[table["series"] == series].sort_values("k")
        if block.empty:
            continue
        print(f"\n  {label}")
        print("    k   patches      LSR       TVD    circuits")
        for _, row in block.iterrows():
            print(
                f"    {int(row['k']):<3} {int(row['patches']):>7}   "
                f"{row['lsr']:.4f}    {row['tvd']:.4f}   {int(row['circuits']):>5}"
            )

    for series, label in SERIES:
        block = table[table["series"] == series].sort_values("k")
        if len(block) < 2:
            continue
        first, last = block.iloc[0], block.iloc[-1]
        peak = block.loc[block["lsr"].idxmax()]
        lsr_change = (last["lsr"] / first["lsr"] - 1.0) * 100.0
        tvd_change = (last["tvd"] / first["tvd"] - 1.0) * 100.0
        print(
            f"\n[{NAME}] {label}: across k = {int(first['k'])}..{int(last['k'])}, "
            f"LSR {lsr_change:+.1f}% (peaks at k = {int(peak['k'])}), "
            f"TVD {tvd_change:+.1f}%"
        )


def plot(table: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.05), sharex=True
    )

    for axis, (column, ylabel) in zip(axes, PANELS):
        for series, label in SERIES:
            block = table[table["series"] == series].sort_values("k")
            if block.empty:
                continue
            x = block["k"].to_numpy()
            y = block[column].to_numpy()
            spread = block[f"{column}_sd"].to_numpy()
            line, = axis.plot(x, y, label=label)
            axis.fill_between(
                x, y - spread, y + spread,
                color=line.get_color(), alpha=0.15, linewidth=0,
            )
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)

    # Stacked panels share the x quantity, so only the bottom one carries its labels.
    axes[-1].set_xlabel("logical qubits per patch, $k$   ($[[k{+}2,\\,k,\\,2]]$ code)")
    ticks = sorted(table["k"].unique())
    axes[-1].set_xticks(ticks)
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    for axis, tag in zip(axes, "ab"):
        axis.annotate(
            f"({tag})", xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-28, 2), textcoords="offset points",
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
    parser.add_argument(
        "--from-csv", type=Path, default=defaults.source,
        help="Raw rows from lsr_tvd_vs_k. This figure runs no simulation of its own.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    return Settings(source=args.from_csv, output_override=args.output_dir)


def main() -> None:
    settings = parse_args()
    if not settings.source.exists():
        raise FileNotFoundError(
            f"{settings.source} not found. Run lsr_tvd_vs_k first, e.g.\n"
            f"  PYTHONPATH=. python -m experiments.figures.lsr_tvd_vs_k.lsr_tvd_vs_k "
            f"--num-qubits 20 --csv {settings.source}"
        )
    frame = pd.read_csv(settings.source)
    table = summarise(frame)
    if table.empty:
        raise RuntimeError(f"no rows in {settings.source}")
    print_summary(table, frame)
    plot(table, settings)


if __name__ == "__main__":
    main()
