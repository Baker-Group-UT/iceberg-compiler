"""
sampled_front.py -- AQORN's placement of a program too large to enumerate, among thousands of
random placements and every other packer, on kept shots against logical error.

One panel, single column: every measured placement of `experiments/optimality/sampled_front.py`
(`multiplier_n15` by default) as logical selection rate against logical error rate at the
circuit's Fig 6 noise level -- uniform random placements (grey), AQORN's grouping with random
lanes and patch order (light blue: what *which atoms move* is worth), the front of all of them
(line), and each packer's placement, labelled. Simulated annealing on AQORN's own objective
estimates that objective's optimum.

    PYTHONPATH=. venv/bin/python -m experiments.figures.sampled_front.sampled_front \\
        --from-csv results/optimality_front/sampled_front_multiplier_n15.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

NAME = "sampled_front"
LABELS = {
    "aqorn": "AQORN", "naive": "naive", "annealing": "annealing (AH*)",
    "interaction_partition": "int. partition", "interaction_partition_descent": "int. part. + descent",
    "interaction_matching": "int. matching", "hypergraph_24": "[24]", "greedy_pairs": "greedy pairs",
    "random_budget": "best random (AH*)",
}
#: Arms drawn with a series marker (palette order); the rest get a small black marker.
STYLED = ("aqorn", "naive", "interaction_partition")
#: Arms not drawn. AQORN's own ablations, and interaction partition + AQORN's descent: that is
#: AQORN's search from a cost-blind seed, so on one program it lands among AQORN's own samples
#: and reads as a rival; across the corpus it ties AQORN on accuracy and keeps fewer shots
#: (packers_corpus), which is where it is reported.
HIDDEN = ("aqorn_raw", "aqorn_seed_only", "aqorn_descent_from_naive",
          "interaction_partition_descent")


def front(points: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    ordered = points.sort_values([x, y], ascending=[True, False])
    keep, best = [], -np.inf
    for index, value in ordered[y].items():
        if value > best:
            keep.append(index)
            best = value
    return ordered.loc[keep]


def draw(frame: pd.DataFrame, alpha: str, output_dir=None):
    use_style()
    x, y = f"ler_{alpha}", f"lsr_{alpha}"
    frame = frame[frame.get("correct", True) != False].dropna(subset=[x, y])  # noqa: E712
    frame = frame[~((frame["kind"] == "arm") & frame["label"].isin(HIDDEN))]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.7))
    cloud = frame[frame["kind"] == "random"]
    grouping = frame[frame["kind"] == "aqorn_grouping"]
    ax.scatter(cloud[x], cloud[y], s=2, color="#cfcfcf", linewidths=0, zorder=1,
               label="random placements")
    ax.scatter(grouping[x], grouping[y], s=3, color="#9fd3fb", linewidths=0, zorder=2,
               label="AQORN grouping, random lanes")
    edge = front(frame, x, y)
    ax.step(edge[x], edge[y], where="post", color="#636363", linewidth=0.8, marker="none",
            linestyle="-", zorder=3)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    arms = frame[frame["kind"] == "arm"].set_index("label")
    for arm, row in arms.iterrows():
        if arm in STYLED:
            i = STYLED.index(arm)
            style = dict(marker=cycle["marker"][i], color=cycle["color"][i], markersize=5,
                         markeredgecolor="black", markeredgewidth=0.4)
        else:
            style = dict(marker="*" if arm == "annealing" else "x", color="black",
                         markersize=5 if arm == "annealing" else 3.5, markeredgewidth=0.6)
        ax.plot(row[x], row[y], linestyle="none", zorder=5, **style)
        ax.annotate(LABELS.get(arm, arm), (row[x], row[y]), xytext=(4, 1),
                    textcoords="offset points", fontsize=8, ha="left", va="bottom", zorder=6)
    ax.set_xlabel("Logical error rate")
    ax.set_ylabel("Logical selection rate")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.legend(loc="lower left", fontsize=8, markerscale=3, handletextpad=0.2)
    program = frame["program"].iloc[0] if "program" in frame else ""
    paths = save_figure(fig, f"{NAME}_{program}" if program else NAME, Path(__file__),
                        override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", type=Path,
                        default=Path("results/optimality_front/sampled_front_multiplier_n15.csv"))
    parser.add_argument("--alpha", default="fig6", choices=("fig6", "a1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.from_csv)
    for path in draw(frame, args.alpha, args.output_dir):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
