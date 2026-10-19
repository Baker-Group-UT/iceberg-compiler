"""
merging_algorithm.py -- the gate-merging pass on the paper's toy circuit, step by step.

The circuit and every step are the pass's own (`experiments/figures/merging_example_circuit/
merging_example_circuit.py --trace`, the Fig 5 example): two patches at k = 4 (q0..q3 in patch
0, q4..q7 in patch 1), 13 logical gates.

    (1) layers        the program in time slices on the packing's lanes
    (2) candidates    per patch and gate kind (`translator.merging.single_qubit`): a whole-patch
                      HH must cover every occupied lane; Pauli XX / ZZ fuse on any subset (an
                      even one cancels the anchor). Three H on 3 of 4 lanes stay targeted
    (3) conflicts     candidates that share a gate, or whose fusion would move a gate past a
                      two-qubit gate, conflict; the conflict-free set with the largest weighted
                      saving is kept (`translator.merging.selection`)
    (4) transversal   a layer of cross-patch CX covering every lane of both patches becomes one
                      CXT; a crossed lane map is aligned by routing swaps -- atom exchanges, no
                      gates, movement charged -- and the macro is taken only if it still pays
                      (`translator.merging.transversal`)
    (5) rewrite       and repeat until a pass merges nothing (two passes here)

Gate costs at k = 4 ([[6,4,2]], `iceberg.costs.work_table(6)`), (1q, 2q): targeted H (7, 4),
HH (6, 0); X (2, 0), XX on two lanes (2, 0); cross-patch CX (0, 4), CXT (0, 6). Saved: 24 1q and
26 2q physical gates, as the trace reports.

    PYTHONPATH=. venv/bin/python -m experiments.figures.algorithm_diagrams.merging_algorithm \\
        [--output-dir DIR]
Writes merging_algorithm.{svg,pdf}; text stays text in the SVG (Inkscape-editable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

INK = "#2b2b2b"
WIRE = "#9a9a9a"
GATE = "#ffffff"
HH_C = "#3b4c56"       # the H-commutation / HH colour of Fig 6
XX_C = "#9a8f8c"
CXT_C = "#0097fd"      # packing blue of Fig 6
SWAP_C = "#5a5a5a"
OK = "#2f7d62"
BAD = "#c0392b"
BAND = "#f3f0ea"
PATCH0 = "#e3f1ec"
PATCH1 = "#e8eef6"

#: The toy circuit as drawn, column by column; layers from the pass's trace.
COLUMNS = [
    ("L0", [("h", 0), ("h", 1), ("h", 2), ("h", 3)]),
    ("L1", [("x", 0), ("x", 2), ("cx", 3, 7)]),
    ("L1", [("cx", 1, 4)]),
    ("L2", [("cx", 0, 5)]),
    ("L2", [("cx", 2, 6)]),
    ("L2", [("h", 1), ("h", 3)]),
    ("L3", [("h", 0)]),
]


def costs():
    """(1q, 2q) of the gates and macros at k = 4, from the cost tables the pass uses."""
    from iceberg.costs import work_table
    from translator.merging.conflicts import macro_cost

    table = work_table(6)
    return {"H": table["H"], "HH": table["HH"], "X": table["X"],
            "XX2": macro_cost("XX", 2, table), "CX": table["CX"], "CXT": table["CXT"]}


class Canvas:
    def __init__(self, ax):
        self.ax = ax

    def text(self, x, y, s, size=7.2, color=INK, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        self.ax.text(x, y, s, fontsize=size, color=color, **kw)

    def line(self, xs, ys, color=WIRE, width=0.7, gid=None, **kw):
        self.ax.plot(xs, ys, color=color, linewidth=width, linestyle=kw.pop("linestyle", "-"),
                     marker="none", solid_capstyle="butt", gid=gid, **kw)

    def box(self, x, y, w, h, label="", face=GATE, edge="#4a4a4a", size=6.5, color=INK,
            rounding=0.02, gid=None, width=0.6, z=3):
        from matplotlib.patches import FancyBboxPatch

        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                         boxstyle=f"round,pad=0,rounding_size={rounding}",
                                         facecolor=face, edgecolor=edge, linewidth=width,
                                         zorder=z, gid=gid))
        if label:
            self.ax.text(x, y, label, fontsize=size, color=color, ha="center", va="center",
                         zorder=z + 1)

    def band(self, x, y, w, h, face, gid=None):
        from matplotlib.patches import FancyBboxPatch

        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                         facecolor=face, edgecolor="none", zorder=0, gid=gid))

    def dot(self, x, y, r=0.022, color=INK):
        from matplotlib.patches import Circle

        self.ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=4))

    def target(self, x, y, r=0.045):
        from matplotlib.patches import Circle

        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, linewidth=0.6,
                                 zorder=4))
        self.line([x - r, x + r], [y, y], color=INK, width=0.6, zorder=5)
        self.line([x, x], [y - r, y + r], color=INK, width=0.6, zorder=5)

    def cross(self, x, y, s=0.035, color=SWAP_C):
        self.line([x - s, x + s], [y - s, y + s], color=color, width=0.8, zorder=5)
        self.line([x - s, x + s], [y + s, y - s], color=color, width=0.8, zorder=5)

    def arrow(self, start, end, color="#7a7a7a", style="-|>", width=0.8, gid=None, **kw):
        from matplotlib.patches import FancyArrowPatch

        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7,
                                          color=color, linewidth=width, shrinkA=0, shrinkB=0,
                                          zorder=3, gid=gid, **kw))


def wires(c: Canvas, x0, x1, ys, labels=True):
    for q, y in ys.items():
        c.line([x0, x1], [y, y], gid=f"wire-q{q}")
        if labels:
            c.text(x0 - 0.06, y, f"q{q}", size=6.5, ha="right")


def cx(c: Canvas, x, ys, control, target):
    c.line([x, x], [ys[control], ys[target]], color=INK, width=0.6, zorder=3)
    c.dot(x, ys[control])
    c.target(x, ys[target])


def tick(c: Canvas, x, y, ok: bool):
    """A drawn check / cross (the serif font has no glyphs for them)."""
    from matplotlib.patches import Circle

    colour = OK if ok else BAD
    c.ax.add_patch(Circle((x, y), 0.075, facecolor="white", edgecolor=colour, linewidth=0.8,
                          zorder=3))
    if ok:
        c.line([x - 0.035, x - 0.008, x + 0.04], [y + 0.002, y - 0.03, y + 0.035], color=colour,
               width=1.1, zorder=4)
    else:
        c.line([x - 0.03, x + 0.03], [y - 0.03, y + 0.03], color=colour, width=1.1, zorder=4)
        c.line([x - 0.03, x + 0.03], [y + 0.03, y - 0.03], color=colour, width=1.1, zorder=4)


def draw(out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    k = costs()
    W, H = FULL_WIDTH, 3.85
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = Canvas(ax)
    note = dict(size=6.6, color="#4a4a4a", linespacing=1.25)

    # ---- (1) layers: the program on the packing's lanes --------------------------------------
    pitch, gap, top = 0.15, 0.10, H - 0.40
    ys = {q: top - q * pitch - (gap if q >= 4 else 0.0) for q in range(8)}
    x0, step = 0.30, 0.29
    xs = [x0 + 0.20 + i * step for i in range(len(COLUMNS))]
    c.text((x0 + xs[-1] + 0.2) / 2, H - 0.13, "(1) Program on the packing, in layers", size=8.8)
    spans = {}
    for x, (layer, _) in zip(xs, COLUMNS):
        spans.setdefault(layer, []).append(x)
    for layer, cols in spans.items():
        c.band(min(cols) - 0.12, ys[7] - 0.09, max(cols) - min(cols) + 0.24, top - ys[7] + 0.18,
               BAND, gid=f"layer-{layer}")
        c.text((min(cols) + max(cols)) / 2, ys[7] - 0.17, layer, size=6.5, color="#6a6a6a")
    wires(c, x0, xs[-1] + 0.20, ys)
    for x, (_, gates) in zip(xs, COLUMNS):
        for gate in gates:
            if gate[0] == "cx":
                cx(c, x, ys, gate[1], gate[2])
            else:
                c.box(x, ys[gate[1]], 0.13, 0.11, gate[0].upper())
    bx = xs[-1] + 0.28
    for label, a, b in (("patch 0", 0, 3), ("patch 1", 4, 7)):
        c.line([bx, bx], [ys[b] - 0.04, ys[a] + 0.04], color="#6a6a6a", width=0.7)
        c.text(bx + 0.07, (ys[a] + ys[b]) / 2, label, size=6.3, color="#6a6a6a", rotation=90)

    # ---- (2) single-qubit candidates ----------------------------------------------------------
    px = 3.02
    c.text(px + 1.18, H - 0.13, "(2) Single-qubit candidates", size=8.8)
    rows = [
        (True, "HH on q0$-$q3 (L0)", "covers every occupied lane",
         f"4 H {k['H']} $\\rightarrow$ HH {k['HH']}"),
        (True, "XX on q0, q2 (L1)", "a Pauli fuses on any subset",
         f"2 X {k['X']} $\\rightarrow$ XX {k['XX2']}"),
        (False, "H on q0, q1, q3 (L2$-$L3)", "3 of 4 lanes: an HH would also flip q2",
         "stays targeted"),
    ]
    ry = H - 0.45
    for ok, title, why, cost in rows:
        tick(c, px + 0.08, ry, ok)
        c.text(px + 0.22, ry + 0.06, title, size=7.4, ha="left")
        c.text(px + 0.22, ry - 0.10, why, ha="left", **note)
        c.text(px + 2.36, ry + 0.06, cost, ha="right", **note)
        ry -= 0.44
    c.text(px + 1.18, ry + 0.14, "costs: (1q, 2q) physical gates at k = 4", size=6.3,
           color="#6a6a6a")

    # ---- (3) conflicts -------------------------------------------------------------------------
    qx = 5.62
    c.text(qx + 0.74, H - 0.13, "(3) Resolve conflicts", size=8.8)
    c.text(qx + 0.74, H - 0.32,
           "candidates conflict if they share a\ngate, or if fusing one would move a\n"
           "gate past a two-qubit gate; keep the\nconflict-free set with the largest\n"
           "weighted saving", va="top", **note)
    gy = ys[6]
    c.box(qx + 0.42, gy, 0.42, 0.20, "HH", face=HH_C, edge=HH_C, color="white", size=7,
          rounding=0.05)
    c.box(qx + 1.06, gy, 0.42, 0.20, "XX", face=XX_C, edge=XX_C, color="white", size=7,
          rounding=0.05)
    c.text(qx + 0.74, gy - 0.22, "no conflict here: both kept", size=6.6, color=OK)

    c.arrow((bx + 0.17, ys[3] - gap / 2), (px - 0.05, ys[3] - gap / 2), gid="flow-1-2")
    c.arrow((px + 2.42, ys[3] - gap / 2), (qx + 0.05, ys[3] - gap / 2), gid="flow-2-3")

    # ---- (4) transversal two-qubit ------------------------------------------------------------
    t4 = 1.86
    c.text(1.60, t4, "(4) Transversal two-qubit gates", size=8.8)
    lane_y = [t4 - 0.25 - i * 0.16 for i in range(4)]
    left, right = 0.45, 1.10
    pairs = {0: 5, 1: 4, 2: 6, 3: 7}            # control q -> target q, from the circuit
    for i in range(4):
        c.text(left, lane_y[i], f"q{i}", size=7)
        c.text(right, lane_y[i], f"q{4 + i}", size=7)
    for q, t in pairs.items():
        c.arrow((left + 0.10, lane_y[q]), (right - 0.12, lane_y[t - 4]), color=INK, width=0.6)
    c.text((left + right) / 2, lane_y[3] - 0.17, "CX: lane 0 $\\rightarrow$ 1, 1 $\\rightarrow$ 0",
           size=6.3, color="#6a6a6a")
    c.arrow((1.32, (lane_y[1] + lane_y[2]) / 2), (1.62, (lane_y[1] + lane_y[2]) / 2),
            width=1.0)
    al, ar = 1.92, 2.57
    for i in range(4):
        c.text(al, lane_y[i], f"q{i}", size=7)
        c.text(ar, lane_y[i], f"q{[5, 4, 6, 7][i]}", size=7)
        c.arrow((al + 0.10, lane_y[i]), (ar - 0.12, lane_y[i]), color=INK, width=0.6)
    sx = ar + 0.17
    c.line([sx, sx], [lane_y[1], lane_y[0]], color=SWAP_C, width=0.7)
    c.cross(sx, lane_y[0])
    c.cross(sx, lane_y[1])
    c.text(sx + 0.07, (lane_y[0] + lane_y[1]) / 2, "swap", size=6.3, color=SWAP_C, ha="left")
    c.text((al + ar) / 2, lane_y[3] - 0.17, "aligned: one CXT", size=6.3, color="#6a6a6a")
    c.text(0.12, lane_y[3] - 0.36,
           "a CX layer covering every lane of both patches becomes one CXT. The\n"
           "crossed lane map is aligned by a routing swap q4$\\leftrightarrow$q5 (an atom "
           "exchange:\nno gates, movement charged), and the macro is kept only if it still pays:\n"
           f"4 CX {k['CX']} $\\rightarrow$ CXT {k['CXT']}",
           ha="left", va="top", **note)

    # ---- (5) rewrite and repeat ---------------------------------------------------------------
    ox = 3.85
    c.text(ox + 1.55, t4, "(5) Rewrite, then repeat", size=8.8)
    opitch, ogap, otop = 0.105, 0.06, t4 - 0.22
    oys = {q: otop - q * opitch - (ogap if q >= 4 else 0.0) for q in range(8)}
    wires(c, ox + 0.30, ox + 3.05, oys, labels=False)
    c.text(ox + 0.24, (oys[0] + oys[3]) / 2, "patch 0", size=6.3, color="#6a6a6a", ha="right")
    c.text(ox + 0.24, (oys[4] + oys[7]) / 2, "patch 1", size=6.3, color="#6a6a6a", ha="right")

    def span(a, b):
        return (oys[a] + oys[b]) / 2, oys[a] - oys[b] + 0.075

    yc, hh = span(0, 3)
    c.box(ox + 0.55, yc, 0.26, hh, "HH", face=HH_C, edge=HH_C, color="white", size=6.5)
    yc, hh = span(0, 2)
    c.box(ox + 0.95, yc, 0.26, hh, "XX", face=XX_C, edge=XX_C, color="white", size=6.5)
    for x in (ox + 1.30, ox + 2.02):
        c.line([x, x], [oys[5], oys[4]], color=SWAP_C, width=0.7)
        c.cross(x, oys[4], s=0.028)
        c.cross(x, oys[5], s=0.028)
    yc, hh = span(0, 7)
    c.box(ox + 1.66, yc, 0.32, hh, "CXT", face=CXT_C, edge=CXT_C, color="white", size=6.5)
    for q in (0, 1, 3):
        c.box(ox + 2.40, oys[q], 0.13, 0.085, "H", size=5.8)
    c.text(ox + 1.68, oys[7] - 0.18,
           "13 gates $\\rightarrow$ 8 instructions: HH, XX, swap, CXT, swap, 3 H.\n"
           "A second pass finds nothing new, so the pass stops.\n"
           "Saved: 24 one-qubit and 26 two-qubit physical gates.",
           va="top", **note)
    c.arrow((3.20, (lane_y[1] + lane_y[2]) / 2), (ox + 0.12, (lane_y[1] + lane_y[2]) / 2),
            gid="flow-4-5")

    paths = save_figure(fig, "merging_algorithm", Path(__file__),
                        override=Path(out_dir) if out_dir else None)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(f"[merging_algorithm] costs at k = 4: {costs()}")
    for path in draw(args.output_dir):
        print(f"[merging_algorithm] saved {path}")


if __name__ == "__main__":
    main()
