"""
figure.py -- the equal-qubit code-size figure, from equal_qubit_code_size.csv.

Single column, two stacked panels sharing x = idle/distance ratio (log): (a) shots kept and
(b) logical error per kept shot on occupied lanes, each relative to naive k = 2 at the budget
Q2 = 4 ceil(n/2) data qubits (dashed line at 1), for AQORN at every k on the same budget. Each
point is the geometric mean over the 13 circuits with a 95% bootstrap interval (resampling
circuits). k = 2 at Q2 is its tightest packing; every larger k runs below full occupancy.

    PYTHONPATH=. venv/bin/python -m experiments.revision.equal_qubit.figure \\
        --from-csv results/data/teddy_final_2026-10-11/equal_qubit_code_size.csv --output-dir DIR
Writes equal_qubit_code_size.{svg,pdf}.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision.equal_qubit.equal_qubit_code_size import K_VALUES, NAME, summary

MARKERS = {2: "o", 4: "s", 6: "^", 8: "D", 10: "v"}
#: Blue for k = 2 (the house AQORN colour), then a sequential ramp so the larger codes stay
#: distinguishable in print.
COLOURS = {2: "#0096ff", 4: "#2b3a47", 6: "#7a5c99", 8: "#c27c2e", 10: "#b0413e"}


def draw(frame: pd.DataFrame, out_dir: Path, budget: str = "Q2", note=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    tab = summary(frame)
    tab = tab[(tab.budget == budget) & (tab.arm == "aqorn")]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.75), sharex=True)
    dx = {2: 0.86, 4: 0.93, 6: 1.0, 8: 1.07, 10: 1.15}
    panels = (("lsr", "shots kept / naive $k$=2", "(a)"),
              ("logical_error", "logical error / naive $k$=2", "(b)"))
    for ax, (metric, label, letter) in zip(axes, panels):
        part = tab[tab.metric == metric]
        lo_all, hi_all = [], []
        for k in K_VALUES:
            s = part[part.k == k].sort_values("ratio")
            if s.empty:
                continue
            ax.errorbar(s.ratio * dx[k], s.gm, yerr=[s.gm - s.lo, s.hi - s.gm],
                        color=COLOURS[k], marker=MARKERS[k], markersize=3.4, capsize=1.2,
                        lw=1.0, linestyle="-", label=f"AQORN $k$={k}")
            lo_all.append(s.lo.min())
            hi_all.append(s.hi.max())
        ax.axhline(1.0, color="#7a7a7a", lw=1.0, linestyle=(0, (4, 2)), zorder=0,
                   label="naive $k$=2")
        ax.set_yscale("log")
        lo, hi = min(lo_all + [1.0]) * 0.92, max(hi_all + [1.0]) * 1.08
        ax.set_ylim(lo, hi)
        candidates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] if hi / lo > 4 else \
            [0.5, 0.7, 1.0, 1.4, 2.0, 2.8]
        ticks = [t for t in candidates if lo <= t <= hi]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(label, fontsize=8.2)
        ax.text(0.985, 0.95, letter, transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=6.4, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.55, 1.0), handlelength=1.6, columnspacing=0.9)
    ax = axes[-1]
    ax.set_xscale("log", base=2)
    ticks = sorted(frame.ratio.unique())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1/{1 / v:g}" if v < 1 else f"{v:g}"))
    ax.set_xlabel("idle / distance noise ratio")
    if note:
        fig.suptitle(note, fontsize=6, color="#b00000", y=0.905)
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.align_ylabels(axes)
    name = NAME if budget == "Q2" else f"{NAME}_{budget.replace('+', '_')}"
    paths = save_figure(fig, name, Path(__file__), override=out_dir)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", default="Q2")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.from_csv)
    for path in draw(frame, Path(args.output_dir), args.budget, args.note):
        print(f"[{NAME} figure] saved {path}")


if __name__ == "__main__":
    main()
