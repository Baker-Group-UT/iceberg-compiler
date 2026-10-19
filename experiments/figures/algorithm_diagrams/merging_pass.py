"""
merging_pass.py -- how the merging pass decides, on a small example where the decision matters.

Two patches at k = 2 ([[4,2,2]], q0 q1 | q2 q3) and an 8-gate program whose single-qubit
candidates conflict. Everything shown is the pass's own output (`translator.merging`,
reproduced step by step in `trace()`):

    (1) candidates   every gate leads a candidate: it collects one gate of its kind per other
                     lane of its patch, scanning forward in time across gates on other lanes.
                     HH fuses H q0 (layer 0) with H q1 (layer 1); XX fuses X q1 (0) with X q0
                     (1); ZZ fuses the two Z
    (2) conflicts    HH and XX cannot both happen: on q0 the H precedes the X, on q1 it follows
                     it, so the two fusions would cross in time. The pass keeps the
                     non-conflicting set with the largest saving: HH and ZZ; the X's stay
    (3) transversal  the two CX cover both lanes of both patches, lane i to lane i: one CXT
    (4) rewrite      8 gates -> 5; then repeat until a pass merges nothing (here the second
                     pass finds nothing; on Fig 6's corpus later passes add 0.7% of the savings)

Costs (1q, 2q) at k = 2 from `iceberg.costs.work_table(4)`: H (7, 4), HH (4, 0); X, Z (2, 0) and
XX, ZZ on two lanes (2, 0); CX (0, 4), CXT (0, 4). Saved: 12 one-qubit, 12 two-qubit gates.

    PYTHONPATH=. venv/bin/python -m experiments.figures.algorithm_diagrams.merging_pass \\
        [--output-dir DIR]
Writes merging_pass.{svg,pdf}; text stays text in the SVG (Inkscape-editable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

from experiments.figures.algorithm_diagrams.merging_algorithm import (
    CXT_C, HH_C, INK, OK, XX_C, Canvas, cx, tick, wires,
)

PATCH_A = "#e3f1ec"
PATCH_B = "#e6ecf5"
ZZ_C = "#8c6d8f"
PERMUTATION = [(0, 0), (0, 1), (1, 0), (1, 1)]
NOTE = dict(size=6.8, color="#4a4a4a", linespacing=1.25)


def program():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.x(1)
    qc.x(0)
    qc.h(1)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.z(2)
    qc.z(3)
    return qc


def trace():
    """The pass on `program()`: candidates with their savings and conflicts, the picks, the
    transversal groups, the rewritten circuit and the (1q, 2q) saving."""
    from qiskit.converters import circuit_to_dag

    from translator.merging.conflicts import (
        compute_merge_conflicts, filter_single_qubit_conflicting_with_transversal,
        merge_work_table,
    )
    from translator.merging.layer import layer_by_node
    from translator.merging.preprocessing import _build_patch_lanes, _merge_savings_for_records
    from translator.merging.rewrite import build_rewritten_circuit
    from translator.merging.selection import _merge_savings_weight, select_best_group
    from translator.merging.settings import DEFAULT_SETTINGS
    from translator.merging.single_qubit import (
        find_hh_merges_for_patch, find_xx_merges_for_patch, find_zz_merges_for_patch,
    )
    from translator.merging.transversal import find_transversal_merges

    work = merge_work_table(4)
    circuit = program()
    dag = circuit_to_dag(circuit)
    layer, order = layer_by_node(dag)
    index = {q: circuit.find_bit(q).index for q in circuit.qubits}
    lanes = _build_patch_lanes(PERMUTATION, circuit.qubits)
    candidates = []
    for qubits in lanes.values():
        for finder in (find_hh_merges_for_patch, find_xx_merges_for_patch,
                       find_zz_merges_for_patch):
            candidates.extend(finder(layer, order, qubits, DEFAULT_SETTINGS, set()))
    infos = compute_merge_conflicts(candidates, layer, index, dag=dag, order_index=order,
                                    work_table=work)
    names = [r.macro_gate.name.upper() for r in candidates]
    conflicts = sorted({tuple(sorted((names[i], names[j])))
                        for i, info in enumerate(infos) for j in info.conflicts})
    picked = select_best_group(candidates, layer, order, index, dag=dag, work_table=work)
    consumed = {r.first for r in picked} | {r.second for r in picked}
    transversal = find_transversal_merges(layer, order, lanes, index, consumed, True, n_code=4)
    kept = filter_single_qubit_conflicting_with_transversal(picked, transversal, layer, order)
    records = list(kept) + list(transversal)
    saved = _merge_savings_for_records(records, work)
    rewritten = build_rewritten_circuit(circuit, dag, records, layer, order)
    return dict(
        candidates={n: _merge_savings_weight(r, work) for n, r in zip(names, candidates)},
        conflicts=conflicts, kept=[r.macro_gate.name.upper() for r in kept],
        transversal=[r.macro_gate.name.upper() for r in transversal], saved=saved,
        before=len(circuit.data), after=len(rewritten.data),
        rewritten=[(i.operation.name, [rewritten.find_bit(q).index for q in i.qubits])
                   for i in rewritten.data],
    )


def link(c: Canvas, a, b, colour, label, label_at, gid):
    """A candidate fusion: a translucent band joining the gates it would fuse."""
    c.ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, linewidth=7, alpha=0.35,
              solid_capstyle="round", linestyle="-", marker="none", zorder=2, gid=gid)
    c.text(label_at[0], label_at[1], label, size=6.8, color=colour)


def draw(t: dict, out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    W, H = FULL_WIDTH, 1.92
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = Canvas(ax)
    pitch, gap, top = 0.17, 0.08, H - 0.45
    ys = {q: top - q * pitch - (gap if q >= 2 else 0.0) for q in range(4)}
    title_y = H - 0.14
    cap_y = ys[3] - 0.22

    def patches(x0, x1, label=True):
        for lanes_, face, name in (((0, 1), PATCH_A, "patch A"), ((2, 3), PATCH_B, "patch B")):
            hi, lo = ys[lanes_[0]] + 0.075, ys[lanes_[1]] - 0.075
            c.band(x0, lo, x1 - x0, hi - lo, face)

    def gate(x, q, name):
        c.box(x, ys[q], 0.13, 0.11, name)

    # ---- (1) candidates on the program --------------------------------------------------------
    x0 = 0.30
    cols = [x0 + 0.17, x0 + 0.66, x0 + 0.92, x0 + 1.14, x0 + 1.38]
    c.text(1.05, title_y, "(1) Propose candidates", size=8.8)
    patches(x0, cols[-1] + 0.16)
    wires(c, x0, cols[-1] + 0.16, ys)
    for x, q, name in ((cols[0], 0, "H"), (cols[0], 1, "X"), (cols[1], 0, "X"),
                       (cols[1], 1, "H"), (cols[4], 2, "Z"), (cols[4], 3, "Z")):
        gate(x, q, name)
    cx(c, cols[2], ys, 0, 2)
    cx(c, cols[3], ys, 1, 3)
    link(c, (cols[0], ys[0]), (cols[1], ys[1]), HH_C, "", (0, 0), "cand-hh")
    link(c, (cols[0], ys[1]), (cols[1], ys[0]), XX_C, "", (0, 0), "cand-xx")
    mid = (cols[0] + cols[1]) / 2
    c.text(mid, ys[0] + 0.15, "XX", size=6.8, color=XX_C)
    c.text(mid, ys[1] - 0.15, "HH", size=6.8, color=HH_C)
    link(c, (cols[4], ys[2]), (cols[4], ys[3]), ZZ_C, "ZZ", (cols[4] + 0.20, (ys[2] + ys[3]) / 2),
         "cand-zz")
    c.text(1.05, cap_y,
           "each gate collects one gate of its kind\nper other lane of its patch, scanning\n"
           "forward in time past other lanes' gates", va="top", **NOTE)

    # ---- (2) conflicts ------------------------------------------------------------------------
    gx = 2.30
    c.text(gx + 0.68, title_y, "(2) Resolve conflicts", size=8.8)
    s = t["candidates"]
    nodes = {"HH": (gx + 0.30, ys[0] - 0.05), "XX": (gx + 1.06, ys[0] - 0.05),
             "ZZ": (gx + 0.68, ys[3] + 0.02)}
    colours = {"HH": HH_C, "XX": XX_C, "ZZ": ZZ_C}
    for a, b in t["conflicts"]:
        (xa, ya), (xb, yb) = nodes[a], nodes[b]
        c.line([xa, xb], [ya, yb], color="#c0392b", width=1.0, linestyle=(0, (3, 2)))
        c.text((xa + xb) / 2, ya + 0.16, "cross in time", size=6.4, color="#c0392b")
    for name, (x, y) in nodes.items():
        kept = name in t["kept"]
        c.box(x, y, 0.40, 0.19, name, face=colours[name] if kept else "white",
              edge=colours[name], color="white" if kept else colours[name], size=7,
              rounding=0.05, width=0.9)
        c.text(x, y - 0.17, f"saves {s[name]:g}", size=6.3, color="#5a5a5a")
    c.text(gx + 0.68, cap_y,
           "two candidates conflict if they share a\ngate or would cross in time (on q0 the H\n"
           "precedes the X, on q1 it follows). Keep\nthe conflict-free set with the largest\n"
           "saving: HH and ZZ (filled); the two X\nstay as they are",
           va="top", **NOTE)

    # ---- (3) transversal ----------------------------------------------------------------------
    tx = 4.25
    c.text(tx + 0.48, title_y, "(3) Two-qubit gates", size=8.8)
    patches(tx, tx + 0.95)
    wires(c, tx, tx + 0.95, ys, labels=False)
    from matplotlib.patches import FancyBboxPatch

    cx(c, tx + 0.30, ys, 0, 2)
    cx(c, tx + 0.53, ys, 1, 3)
    c.ax.add_patch(FancyBboxPatch((tx + 0.18, ys[3] - 0.09), 0.47, ys[0] - ys[3] + 0.18,
                                  boxstyle="round,pad=0,rounding_size=0.05", facecolor=CXT_C,
                                  edgecolor="none", alpha=0.22, zorder=2, gid="cand-cxt"))
    c.text(tx + 0.80, (ys[1] + ys[2]) / 2, "CXT", size=6.8, color=CXT_C)
    c.text(tx + 0.48, cap_y,
           "CX on every lane of both\npatches, lane i to lane i:\none transversal CXT",
           va="top", **NOTE)

    # ---- (4) rewrite --------------------------------------------------------------------------
    rx = 5.62
    c.text(rx + 0.72, title_y, "(4) Rewrite", size=8.8)
    patches(rx, rx + 1.40)
    wires(c, rx, rx + 1.40, ys, labels=False)

    def macro(x, lanes_, label, colour, width=0.22):
        hi, lo = ys[lanes_[0]] + 0.055, ys[lanes_[-1]] - 0.055
        c.box(x, (hi + lo) / 2, width, hi - lo, label, face=colour, edge=colour, color="white",
              size=6.6, rounding=0.025)

    gate(rx + 0.15, 1, "X")
    macro(rx + 0.40, [0, 1], "HH", HH_C)
    gate(rx + 0.65, 0, "X")
    macro(rx + 0.92, [0, 1, 2, 3], "CXT", CXT_C, width=0.26)
    macro(rx + 1.20, [2, 3], "ZZ", ZZ_C)
    one, two = t["saved"]
    c.text(rx + 0.72, cap_y,
           f"{t['before']} gates $\\rightarrow$ {t['after']}: saves {one:g} one-qubit\n"
           f"and {two:g} two-qubit physical gates.\nRepeat until nothing merges.",
           va="top", **NOTE)

    for a, b in ((cols[-1] + 0.24, gx - 0.02), (gx + 1.40, tx - 0.06), (tx + 1.03, rx - 0.06)):
        c.arrow((a, (ys[1] + ys[2]) / 2), (b, (ys[1] + ys[2]) / 2), gid="flow")

    paths = save_figure(fig, "merging_pass", Path(__file__),
                        override=Path(out_dir) if out_dir else None)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    t = trace()
    for key, value in t.items():
        print(f"[merging_pass] {key}: {value}")
    for path in draw(t, args.output_dir):
        print(f"[merging_pass] saved {path}")


if __name__ == "__main__":
    main()
