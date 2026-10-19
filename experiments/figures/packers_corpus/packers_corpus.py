"""
packers_corpus.py -- how each packer compares with AQORN over Fig 6's 71 circuits, under the
current standard with the logical optimisation pass. The successor of
Experiment1's packer figure (submitted planner, 2026-10-01).

Full width, one row per packer, three panels, each at the circuit's Fig 6 noise level:
    (a) kept shots relative to AQORN: geometric mean over circuits of LSR / AQORN's LSR
    (b) logical error rate reduction against naive packing, % (median over circuits)
    (c) TVD reduction against naive packing, % (median over the circuits whose ideal output is
        not uniform, where TVD can see an error at all)
Placements that fail the noiseless correctness check are dropped (and counted).

    PYTHONPATH=. venv/bin/python -m experiments.figures.packers_corpus.packers_corpus \\
        --from-csv results/optimality_front/packers_corpus.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

NAME = "packers_corpus"
LABELS = {
    "aqorn": "AQORN", "aqorn_raw": "AQORN, verify off", "aqorn_seed_only": "AQORN seed only",
    "aqorn_descent_from_naive": "AQORN descent from naive", "naive": "naive (program order)",
    "random_budget": "random, AQORN's budget", "interaction_matching": "interaction matching",
    "interaction_partition": "interaction partition",
    "interaction_partition_descent": "interaction partition + descent",
    "hypergraph_24": "[24] stage 1", "greedy_pairs": "pairwise greedy (earlier)",
}


def summary(frame: pd.DataFrame, alpha: str) -> pd.DataFrame:
    lsr, ler, tvd = f"lsr_{alpha}", f"ler_{alpha}", f"tvd_{alpha}"
    ok = frame[frame.get("correct", True) != False].dropna(subset=[lsr, ler])  # noqa: E712
    wide = {m: ok.pivot_table(index="circuit", columns="arm", values=m) for m in (lsr, ler, tvd)}
    rows = []
    for arm in wide[lsr].columns:
        ratio = (wide[lsr][arm] / wide[lsr]["aqorn"]).dropna()
        ler_red = (100 * (1 - wide[ler][arm] / wide[ler]["naive"])).dropna()
        visible = wide[tvd]["naive"] > 0
        tvd_red = (100 * (1 - wide[tvd][arm][visible] / wide[tvd]["naive"][visible])).dropna()
        rows.append({"arm": arm, "lsr_vs_aqorn": float(np.exp(np.log(ratio).mean())),
                     "ler_reduction": float(ler_red.median()),
                     "tvd_reduction": float(tvd_red.median()), "circuits": len(ratio),
                     "tvd_circuits": len(tvd_red)})
    return pd.DataFrame(rows).sort_values("lsr_vs_aqorn", ascending=False)


def draw(frame: pd.DataFrame, alpha: str, output_dir=None):
    use_style()
    table = summary(frame, alpha)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.0), sharey=True)
    y = np.arange(len(table))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    for ax, column, label, reference, i in (
            (axes[0], "lsr_vs_aqorn", "Kept shots vs. AQORN", 1.0, 0),
            (axes[1], "ler_reduction", "LER reduction vs. naive (%)", 0.0, 1),
            (axes[2], "tvd_reduction", "TVD reduction vs. naive (%)", 0.0, 1)):
        ax.plot(table[column], y, linestyle="none", marker=cycle["marker"][i],
                color=cycle["color"][i], markersize=5)
        ax.axvline(reference, color="#636363", linestyle="--", linewidth=1.0, zorder=0)
        ax.set_xlabel(label)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticks(y, [LABELS.get(a, a) for a in table["arm"]])
    axes[0].set_ylim(len(table) - 0.5, -0.5)
    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", loc="left", fontweight="bold")
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    print(table.round(3).to_string(index=False))
    print(f"[{NAME}] (a) geometric mean over {int(table['circuits'].max())} circuits; (c) over "
          f"the {int(table['tvd_circuits'].max())} whose ideal output is not uniform")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", type=Path,
                        default=Path("results/optimality_front/packers_corpus.csv"))
    parser.add_argument("--alpha", default="fig6", choices=("fig6", "a1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.from_csv)
    bad = int((frame.get("correct") == False).sum()) if "correct" in frame else 0  # noqa: E712
    print(f"[{NAME}] {frame['circuit'].nunique()} circuits; {bad} placements fail the noiseless check")
    for path in draw(frame, args.alpha, args.output_dir):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
