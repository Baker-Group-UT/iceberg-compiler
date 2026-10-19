"""
sampled_front_grid.py -- the four sampled fronts in one column, 2 x 2.

Each panel is one program too large to enumerate (`experiments/optimality/sampled_front.py`):
logical selection rate against logical error rate at the circuit's Fig 6 noise level for
2000 uniform random placements (grey), AQORN's grouping with random lanes and patch order
(light blue), the front of all of them (line), and each packer's placement. Panels follow
`sampled_front.py` exactly -- same arms hidden, same front -- but at a quarter of a column
there is no room for a text label per point, so the packers carry distinct markers and one
legend above the grid names them.

    PYTHONPATH=. venv/bin/python -m experiments.figures.sampled_front_grid.sampled_front_grid \\
        --from-csv <dir with sampled_front_<program>.csv> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

from experiments.figures.sampled_front.sampled_front import HIDDEN, front
from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

NAME = "sampled_front_grid"
PROGRAMS = ("multiplier_n15", "qec9xz_n17", "adder_n10", "ghz_state_n23")

#: (arm, legend label, plot style). AQORN, naive and the interaction partition take the
#: first three palette positions, as in `sampled_front.py`; every other packer is a small
#: black marker drawn on top, so a marker landing on AQORN's stays visible -- on these
#: programs the searches on AQORN's objective land on or beside AQORN.
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()
ARMS = (
    ("aqorn", "AQORN", dict(marker="o", color="#0097fd", markersize=5.5,
                            markeredgecolor="black", markeredgewidth=0.4, zorder=4)),
    ("naive", "naive", dict(marker="s", color="#3b4c56", markersize=4.5,
                            markeredgecolor="black", markeredgewidth=0.4, zorder=4)),
    ("interaction_partition", "int. partition",
     dict(marker="^", color="#cdc2c0", markersize=5, markeredgecolor="black",
          markeredgewidth=0.4, zorder=4)),
    ("interaction_matching", "int. matching",
     dict(marker="+", color="black", markersize=5, markeredgewidth=0.8, zorder=5)),
    ("hypergraph_24", "[24]", dict(marker="x", color="black", markersize=4,
                                   markeredgewidth=0.8, zorder=5)),
    ("greedy_pairs", "greedy pairs",
     dict(marker="v", color="white", markersize=4, markeredgecolor="black",
          markeredgewidth=0.6, zorder=5)),
    ("annealing", "annealing", dict(marker="*", color="black", markersize=5,
                                    markeredgewidth=0.0, zorder=6)),
    ("random_budget", "best random",
     dict(marker="D", color="white", markersize=3, markeredgecolor="black",
          markeredgewidth=0.6, zorder=6)),
)
CLOUD = dict(s=1.5, color="#cfcfcf", linewidths=0, zorder=1, rasterized=True)
GROUPING = dict(s=2, color="#9fd3fb", linewidths=0, zorder=2, rasterized=True)


def panel(ax, frame: pd.DataFrame, alpha: str, title: str) -> None:
    x, y = f"ler_{alpha}", f"lsr_{alpha}"
    frame = frame[frame.get("correct", True) != False].dropna(subset=[x, y])  # noqa: E712
    frame = frame[~((frame["kind"] == "arm") & frame["label"].isin(HIDDEN))]
    cloud = frame[frame["kind"] == "random"]
    grouping = frame[frame["kind"] == "aqorn_grouping"]
    ax.scatter(cloud[x], cloud[y], **CLOUD)
    ax.scatter(grouping[x], grouping[y], **GROUPING)
    edge = front(frame, x, y)
    ax.step(edge[x], edge[y], where="post", color="#636363", linewidth=0.7, marker="none",
            linestyle="-", zorder=3)
    arms = frame[frame["kind"] == "arm"].set_index("label")
    for arm, _, style in ARMS:
        if arm in arms.index:
            ax.plot(arms.at[arm, x], arms.at[arm, y], linestyle="none", **style)
    ax.margins(0.07)
    ax.set_title(title, fontsize=8, pad=2)
    # Three ticks a side: four 9pt labels do not fit across a quarter-column panel.
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)


def draw(frames: dict, alpha: str, output_dir=None):
    use_style()
    # Taller than the 2.4 in default: two rows of panels plus a three-row legend.
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH, 3.4))
    for i, (ax, program) in enumerate(zip(axes.ravel(), PROGRAMS)):
        panel(ax, frames[program], alpha, f"({'abcd'[i]}) {program}")
    handles = [Line2D([], [], linestyle="none", **{k: v for k, v in style.items()
                                                   if k != "zorder"}, label=label)
               for _, label, style in ARMS]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=3, color="#cfcfcf",
                       label="random"),
                Line2D([], [], linestyle="none", marker="o", markersize=3, color="#9fd3fb",
                       label="AQORN groups")]
    # Column-major fill, 3/3/2/2: short labels share the first column so the four columns
    # fit the column width.
    order = ("AQORN", "naive", "[24]", "int. partition", "int. matching", "greedy pairs",
             "annealing", "best random", "random", "AQORN groups")
    handles = sorted(handles, key=lambda h: order.index(h.get_label()))
    # Legend entries at 8pt (house style says 9): ten entries in four columns only fit a
    # column width at the minor size. The caption spells out the two clouds.
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8,
               handletextpad=0.1, columnspacing=0.7, handlelength=1.3, borderpad=0.0, labelspacing=0.25,
               markerscale=0.85)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.86))
    fig.supxlabel("logical error rate")
    fig.supylabel("logical selection rate")
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", type=Path, required=True,
                        help="directory holding sampled_front_<program>.csv for the four programs")
    parser.add_argument("--alpha", default="fig6", choices=("fig6", "a1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    frames = {p: pd.read_csv(args.from_csv / f"sampled_front_{p}.csv") for p in PROGRAMS}
    for path in draw(frames, args.alpha, args.output_dir):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
