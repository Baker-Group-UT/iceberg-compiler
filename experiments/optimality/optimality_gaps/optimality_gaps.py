"""
optimality_gaps.py -- how close AQORN's packing is to optimal: small programs exhaustively, and
constructed programs whose optimal packing is known at any size.

(a) 4-6 qubit programs, every placement of the tightest packing under both H-commutation settings
    (`exhaustive_silent.py`): mean regret above the exhaustive optimum on the weight-10 planner's
    objective (expected detectable error + 10 x model silent error), every arm scored on the
    translator it runs on. Error bars: +-1 SD over circuits.
(b) planted circuits (`planted.py`): fraction of the hidden pairs each packer recovers, against
    program width, without decoys and with decoys at rate 0.3.
(c) planted circuits: weighted gate work of the logical gates above the gate-wise lower bound
    (which the planted packing attains), decoy-free circuits.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.optimality_gaps.optimality_gaps
    ... --exhaustive results/optimality/exhaustive_silent.csv --planted results/optimality/planted.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
NAME = "optimality_gaps"
ARMS = [("w10", "AQORN (weight 10)"), ("w10T", "AQORN w10T"), ("v4", "AQORN v4"),
        ("naive", "Naive"), ("interaction_partition", "Interaction partition")]
CONFIGS = ["4q,k=2", "5q,k=2", "6q,k=2"]


def panel_exhaustive(ax, frame: pd.DataFrame) -> None:
    frame = frame[frame["in_space"].astype(bool)]
    width = 0.8 / len(ARMS)
    x = np.arange(len(CONFIGS))
    for i, (arm, label) in enumerate(ARMS):
        sub = frame[frame["arm"] == arm].groupby("config")["score_regret_pct"]
        means = [sub.mean().get(c, np.nan) for c in CONFIGS]
        sds = [sub.std().get(c, np.nan) for c in CONFIGS]
        ax.bar(x + (i - (len(ARMS) - 1) / 2) * width, means, width, yerr=sds, capsize=2,
               label=label, error_kw={"elinewidth": 0.6})
    ax.set_xticks(x, [c.split(",")[0].replace("q", " qubits") for c in CONFIGS])
    ax.set_ylabel("Regret above optimum (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("(a) Exhaustive, weight-10 objective", loc="left")


def _style(i: int):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    return {key: values[i % len(values)] for key, values in cycle.items()}


def panel_recovery(ax, frame: pd.DataFrame) -> None:
    shipped = frame[(frame["translator"] == "shipped") | (frame["arm"] == "w10T")]
    for i, (arm, label) in enumerate(ARMS):
        style = _style(i)
        for decoy in (0.0, 0.3):
            sub = shipped[(shipped["arm"] == arm) & (shipped["decoy"] == decoy)]
            if sub.empty:
                continue
            series = sub.groupby("n")["pairs_recovered"].mean() * 100
            ax.plot(series.index, series.values, color=style.get("color"),
                    marker=style.get("marker"), markersize=3,
                    linestyle=style.get("linestyle", "-") if decoy == 0.0 else ":",
                    label=label if decoy == 0.0 else None)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Program qubits")
    ax.set_ylabel("Planted pairs recovered (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Planted (dotted: decoys 0.3)", loc="left")


def panel_bound(ax, frame: pd.DataFrame) -> None:
    clean = frame[(frame["decoy"] == 0.0) & ((frame["translator"] == "shipped")
                                            | (frame["arm"] == "w10T"))]
    for i, (arm, label) in enumerate(ARMS):
        sub = clean[clean["arm"] == arm]
        if sub.empty:
            continue
        style = _style(i)
        series = sub.groupby("n")["logical_work_vs_bound_pct"]
        ax.errorbar(series.mean().index, series.mean().values, yerr=series.std().values,
                    capsize=2, elinewidth=0.6, color=style.get("color"),
                    marker=style.get("marker"), markersize=3,
                    linestyle=style.get("linestyle", "-"), label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Program qubits")
    ax.set_ylabel("Gate work above bound (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("(c) Planted, no decoys", loc="left")


def draw(exhaustive: pd.DataFrame, planted: pd.DataFrame, output_dir=None):
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(7.14, 2.85))
    panel_exhaustive(axes[0], exhaustive)
    panel_recovery(axes[1], planted)
    panel_bound(axes[2], planted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False,
               fontsize=9)
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--exhaustive", default="results/optimality/exhaustive_silent.csv")
    parser.add_argument("--planted", default="results/optimality/planted.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    paths = draw(pd.read_csv(args.exhaustive), pd.read_csv(args.planted),
                 Path(args.output_dir) if args.output_dir else None)
    for path in paths:
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
