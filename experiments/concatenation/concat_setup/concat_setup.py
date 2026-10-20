"""
concat_setup.py -- how the concatenated [[4,2,2]] x [[4,2,2]] code is laid out and compiled.

(a) the spread layout: two outer patches A and B (4 data + 2 syndrome ancillas each) in six inner
    patches, each inner patch holding the same role of A and of B. On the data this is [[16,4,4]].
(b) the index layout: an inner patch holds two qubits of ONE outer patch, so one undetected inner
    error (Xbar_1 Xbar_2 = X_1 X_2) is a weight-2 error on that outer patch -- distance 2.
(c) the compilation path: the translator run twice.

A diagram, so no data: no --from-csv. Full width.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_setup.concat_setup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
NAME = "concat_setup"
A_COLOUR, B_COLOUR = "#0097fd", "#3b4c56"
ROLES = ("d0", "d1", "d2", "d3", "a0", "a1")


def _cell(ax, x, y, w, h, label, colour, *, hatch=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.04",
                                facecolor=colour, edgecolor="white", linewidth=0.8, hatch=hatch))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8,
            color="white" if colour in (A_COLOUR, B_COLOUR) else "black")


def _layout_panel(ax, inner, title, note, broken: int):
    """inner: list of six (slot0, slot1) labels like ('A', 'd0')."""
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-1.15, 2.35)
    ax.axis("off")
    for j, lanes in enumerate(inner):
        ax.add_patch(FancyBboxPatch((j + 0.05, 0.0), 0.9, 1.7,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="none", edgecolor="#636363", linewidth=0.9,
                                    linestyle="-" if j != broken else "--"))
        for slot, (patch, role) in enumerate(lanes):
            colour = A_COLOUR if patch == "A" else B_COLOUR
            _cell(ax, j + 0.12, 0.92 - 0.82 * slot, 0.76, 0.7,
                  f"{patch}$\\cdot${role}", colour)
        ax.text(j + 0.5, -0.17, f"inner {j}", ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(broken + 0.5, 1.78), xytext=(broken + 0.5, 2.15),
                arrowprops=dict(arrowstyle="-|>", color="#636363", lw=1.0))
    ax.text(broken + 0.62, 2.2, r"undetected $\bar X_1\bar X_2$", fontsize=8, ha="left",
            va="center")
    ax.text(3.0, -0.62, note, ha="center", va="top", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")


def _pipeline_panel(ax):
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    steps = [
        ("logical program\n$m$ qubits", "#cdc2c0"),
        ("outer translation\nAQORN: packing,\ncommutation, merging", "#cdc2c0"),
        ("C1: $m/2$ outer patches\n6 qubits each\n(4 data, 2 ancillas)", "#bdbdbd"),
        ("flatten: SWAP $\\to$\nrelabel, measurements\ndeferred to the end", "#cdc2c0"),
        ("inner translation\nAQORN restricted to\nspread placements", "#cdc2c0"),
        ("C2: $9m$ physical\nqubits, decoded at\nboth levels", "#bdbdbd"),
    ]
    width, gap = 1.02, 0.17
    for i, (text, colour) in enumerate(steps):
        x = 0.05 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, 0.1), width, 0.8, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    facecolor=colour, edgecolor="#636363", linewidth=0.8))
        ax.text(x + width / 2, 0.5, text, ha="center", va="center", fontsize=8)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.01, 0.5), (x - 0.01, 0.5),
                                         arrowstyle="-|>", mutation_scale=8, color="#3b4c56",
                                         linewidth=0.9))
    ax.set_title("(c) Compilation: the same translator, run once per level", fontsize=10,
                 loc="left")


def draw(output_dir: Path | None = None):
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig = plt.figure(figsize=(7.14, 4.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 0.95])
    spread = [(("A", r), ("B", r)) for r in ROLES]
    index = [(("A", "d0"), ("A", "d1")), (("A", "d2"), ("A", "d3")), (("A", "a0"), ("A", "a1")),
             (("B", "d0"), ("B", "d1")), (("B", "d2"), ("B", "d3")), (("B", "a0"), ("B", "a1"))]
    _layout_panel(fig.add_subplot(grid[0, 0]), spread, "(a) Spread: one role of A and of B per inner patch",
                  "one error on each of A and B: both outer patches detect it\n"
                  "data qubits (inner 0-3): the [[16,4,4]] code", broken=1)
    _layout_panel(fig.add_subplot(grid[0, 1]), index, "(b) Index order: two qubits of one outer patch",
                  r"$X_{d0}X_{d1}$ on outer patch A: an outer logical error" "\n"
                  "undetected at both levels -- distance 2", broken=0)
    _pipeline_panel(fig.add_subplot(grid[1, :]))
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for path in draw(Path(args.output_dir) if args.output_dir else None):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
