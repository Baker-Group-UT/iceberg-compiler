"""
merging_rules.py -- gate merging in one figure: the three merges, each as before -> after with
its one rule and its cost, and the pass that applies them.

    (a) whole-patch H   H on every occupied lane of a patch, in one layer -> one HH
                        (H on 3 of 4 lanes is not merged: an HH would also flip the fourth)
    (b) Pauli           X (or Z) on any subset of a patch's lanes -> one XX (ZZ)
    (c) transversal     CX from every lane of one patch to every lane of another -> one CXT;
                        crossed lanes are first aligned by a routing swap (an atom exchange, no
                        gates; its movement is charged)
    the pass            layer the circuit, propose candidates in every patch, keep the
                        non-conflicting set with the largest saving, rewrite, repeat until
                        nothing merges (`translator.merging.preprocessing`)

Costs are (1q, 2q) physical gates at k = 4 ([[6,4,2]]) from `iceberg.costs.work_table(6)`, the
code size of the paper's merging example (`merging_example_circuit.py`), whose three merges
these are.

    PYTHONPATH=. venv/bin/python -m experiments.figures.algorithm_diagrams.merging_rules \\
        [--output-dir DIR]
Writes merging_rules.{svg,pdf}; text stays text in the SVG (Inkscape-editable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

from experiments.figures.algorithm_diagrams.merging_algorithm import (
    CXT_C, HH_C, INK, SWAP_C, XX_C, Canvas, costs, cx, tick, wires,
)

PATCH_BAND = "#e3f1ec"
PATCH_BAND_2 = "#e6ecf5"
NOTE = dict(size=6.8, color="#4a4a4a", linespacing=1.25)


def band(c: Canvas, x0, x1, ys, lanes, face, label=None):
    top, bottom = ys[lanes[0]] + 0.075, ys[lanes[-1]] - 0.075
    c.band(x0, bottom, x1 - x0, top - bottom, face)
    if label:
        c.text(x0 - 0.04, (top + bottom) / 2, label, size=6.2, color="#6a6a6a", ha="right",
               rotation=90)


def macro(c: Canvas, x, ys, lanes, label, face, width=0.24, marks=None):
    top, bottom = ys[lanes[0]] + 0.055, ys[lanes[-1]] - 0.055
    c.box(x, (top + bottom) / 2, width, top - bottom, label, face=face, edge=face, color="white",
          size=6.8, rounding=0.025)
    for q in marks or ():
        c.dot(x - width / 2 - 0.02, ys[q], r=0.016, color=face)


def draw(out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    k = costs()
    W, H = FULL_WIDTH, 2.95
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = Canvas(ax)
    title_y = H - 0.14

    def lanes4(top, pitch=0.15):
        return {q: top - q * pitch for q in range(4)}

    # ---- (a) whole-patch H --------------------------------------------------------------------
    ys = lanes4(H - 0.50)
    a0 = 0.30
    c.text(1.15, title_y, "(a) Whole-patch Hadamard", size=8.8)
    band(c, a0, a0 + 0.62, ys, [0, 1, 2, 3], PATCH_BAND)
    c.text(a0 + 0.31, ys[0] + 0.15, "one patch, k = 4", size=6.2, color="#6a6a6a")
    wires(c, a0, a0 + 0.62, ys)
    for q in range(4):
        c.box(a0 + 0.31, ys[q], 0.13, 0.11, "H")
    c.arrow((a0 + 0.72, (ys[1] + ys[2]) / 2), (a0 + 1.02, (ys[1] + ys[2]) / 2), width=1.0)
    band(c, a0 + 1.10, a0 + 1.72, ys, [0, 1, 2, 3], PATCH_BAND)
    wires(c, a0 + 1.10, a0 + 1.72, ys, labels=False)
    macro(c, a0 + 1.41, ys, [0, 1, 2, 3], "HH", HH_C)
    c.text(1.15, ys[3] - 0.24, "rule: every occupied lane, one layer", size=7.2)
    c.text(1.15, ys[3] - 0.42, f"4 H {k['H']} $\\rightarrow$ HH {k['HH']}", **NOTE)
    tick(c, a0 + 0.10, ys[3] - 0.66, False)
    c.text(a0 + 0.22, ys[3] - 0.66, "H on 3 of 4 lanes: not merged,\nan HH would also flip the 4th",
           ha="left", **NOTE)

    # ---- (b) Pauli ----------------------------------------------------------------------------
    b0 = 2.68
    c.text(b0 + 0.86, title_y, "(b) Pauli merge", size=8.8)
    band(c, b0, b0 + 0.62, ys, [0, 1, 2, 3], PATCH_BAND)
    c.text(b0 + 0.31, ys[0] + 0.15, "one patch, k = 4", size=6.2, color="#6a6a6a")
    wires(c, b0, b0 + 0.62, ys)
    for q in (0, 2):
        c.box(b0 + 0.31, ys[q], 0.13, 0.11, "X")
    c.arrow((b0 + 0.72, (ys[1] + ys[2]) / 2), (b0 + 1.02, (ys[1] + ys[2]) / 2), width=1.0)
    band(c, b0 + 1.10, b0 + 1.72, ys, [0, 1, 2, 3], PATCH_BAND)
    wires(c, b0 + 1.10, b0 + 1.72, ys, labels=False)
    macro(c, b0 + 1.41, ys, [0, 1, 2], "XX", XX_C, marks=(0, 2))
    c.text(b0 + 0.86, ys[3] - 0.24, "rule: any subset of lanes (X or Z)", size=7.2)
    c.text(b0 + 0.86, ys[3] - 0.42, f"2 X {k['X']} $\\rightarrow$ XX {k['XX2']}", **NOTE)
    c.text(b0 + 0.86, ys[3] - 0.62, "dots: the lanes it acts on", **NOTE)

    # ---- (c) transversal ----------------------------------------------------------------------
    c0 = 5.02
    pitch, gap, top = 0.105, 0.05, H - 0.38
    ys8 = {q: top - q * pitch - (gap if q >= 4 else 0.0) for q in range(8)}
    c.text(c0 + 0.98, title_y, "(c) Transversal CX", size=8.8)
    band(c, c0, c0 + 0.78, ys8, [0, 1, 2, 3], PATCH_BAND, "patch A")
    band(c, c0, c0 + 0.78, ys8, [4, 5, 6, 7], PATCH_BAND_2, "patch B")
    wires(c, c0, c0 + 0.78, ys8, labels=False)
    for i, (control, target) in enumerate(((0, 5), (1, 4), (2, 6), (3, 7))):
        cx(c, c0 + 0.15 + i * 0.16, ys8, control, target)
    c.arrow((c0 + 0.86, ys8[4] - 0.02), (c0 + 1.10, ys8[4] - 0.02), width=1.0)
    d0 = c0 + 1.18
    band(c, d0, d0 + 0.80, ys8, [0, 1, 2, 3], PATCH_BAND)
    band(c, d0, d0 + 0.80, ys8, [4, 5, 6, 7], PATCH_BAND_2)
    wires(c, d0, d0 + 0.80, ys8, labels=False)
    for x in (d0 + 0.14, d0 + 0.66):
        c.line([x, x], [ys8[5], ys8[4]], color=SWAP_C, width=0.7)
        c.cross(x, ys8[4], s=0.026)
        c.cross(x, ys8[5], s=0.026)
    macro(c, d0 + 0.40, ys8, list(range(8)), "CXT", CXT_C, width=0.26)
    cy = ys8[7] - 0.20
    c.text(c0 + 0.98, cy, "rule: every lane of both patches, lane i to lane i", size=7.2)
    c.text(c0 + 0.98, cy - 0.18, f"4 CX {k['CX']} $\\rightarrow$ CXT {k['CXT']}", **NOTE)
    c.text(c0 + 0.98, cy - 0.38,
           "crossed lanes (here 0 and 1) are aligned by a routing\n"
           "swap: an atom exchange, no gates, movement charged", **NOTE)

    # ---- the pass -----------------------------------------------------------------------------
    steps = ["layer the circuit", "propose (a)$-$(c) in\nevery patch",
             "keep the non-conflicting\nset with the largest saving", "rewrite"]
    y, h, w, g = 0.30, 0.38, 1.38, 0.30
    x = (W - (len(steps) * w + (len(steps) - 1) * g)) / 2
    xs = []
    for i, step in enumerate(steps):
        c.box(x + w / 2, y + h / 2, w, h, "", face="#f3f0ea", edge="#9a948a", rounding=0.06)
        c.text(x + w / 2, y + h / 2, step, size=7.0, linespacing=1.15)
        if i:
            c.arrow((x - g + 0.03, y + h / 2), (x - 0.03, y + h / 2))
        xs.append(x)
        x += w + g
    c.arrow((xs[-1] + w / 2, y + h + 0.01), (xs[0] + w / 2, y + h + 0.01),
            connectionstyle="arc3,rad=0.055", gid="repeat")
    c.text(W / 2, y + h + 0.255, "repeat until nothing merges", size=6.8, color="#5a5a5a")
    c.text(0.12, y + h / 2, "The\npass", size=7.4, ha="left", color=INK)

    paths = save_figure(fig, "merging_rules", Path(__file__),
                        override=Path(out_dir) if out_dir else None)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for path in draw(args.output_dir):
        print(f"[merging_rules] saved {path}")


if __name__ == "__main__":
    main()
