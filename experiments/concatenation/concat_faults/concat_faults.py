"""
concat_faults.py -- single faults that evade detection at both levels, by layout (exact).

From `distance.py`: on deterministic 4-qubit test programs, the total probability of the detector
error model's mechanisms that flip an output and fire no check -- the silent error post-selection
keeps, to first order. One bar per compilation; error bars +-1 SD over the test programs; the
number above each bar is the mean count of such mechanisms. Single column.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_faults.concat_faults \
        --from-csv results/concatenation/concat_distance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
NAME = "concat_faults"
ARMS = [("single", "1 level"), ("single_passes", "1 level\n+passes"),
        ("concat_index", "2 levels\nindex"), ("concat_index_passes", "2 levels\nindex\n+passes"),
        ("concat_stride", "2 levels\nspread"), ("concat_stride_passes", "2 levels\nspread\n+passes")]


def draw(frame: pd.DataFrame, output_dir=None):
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.487, 2.75))
    alpha = float(frame["alpha"].iloc[0])
    colours = ["#bdbdbd", "#bdbdbd", "#636363", "#636363", "#0097fd", "#0097fd"]
    hatches = [None, "//", None, "//", None, "//"]
    for i, (arm, _) in enumerate(ARMS):
        sub = frame[frame["arm"] == arm]
        mean = sub["first_order_silent_probability"].mean()
        sd = sub["first_order_silent_probability"].std()
        ax.bar(i, mean, 0.7, yerr=sd, capsize=2, color=colours[i], hatch=hatches[i],
               edgecolor="white", error_kw={"elinewidth": 0.6})
        ax.text(i, (mean + sd) * 1.25, f"{sub['first_order_silent_mechanisms'].mean():.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(3e-5, 2e-2)
    ax.set_xticks(range(len(ARMS)), [label for _, label in ARMS], fontsize=8)
    ax.set_ylabel(f"First-order silent probability\n" + rf"($\alpha$ = {alpha:g})")
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", default="results/concatenation/concat_distance.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for path in draw(pd.read_csv(args.from_csv), Path(args.output_dir) if args.output_dir else None):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
