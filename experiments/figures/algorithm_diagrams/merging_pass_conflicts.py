"""
merging_pass_conflicts.py -- the selection step alone, on a richer conflict graph (kept as
an alternative; `merging_pass.py` is the paper version).

One [[4,2,2]] patch (k = 2, lanes q0 q1) and a 10-gate program, q0: H S X S H, q1: X S H S X.
Everything shown is the pass's own output (`translator.merging`), reproduced by `trace()`:

    (1) candidates   every H / X / Z leads a candidate: it collects one gate of its kind per
                     other lane, scanning forward in time past other gates. Here: two HH (A, B)
                     and two XX (C, D), each fusing gates two layers apart
    (2) conflicts    `compute_merge_conflicts`: two candidates conflict if they claim the same
                     gate, if their fusions would cross in time, or if one's first gate sits in
                     the other's window and would be moved past its macro
    (3) groups       `select_best_group`: candidates in time order go into the first group
                     holding nothing they conflict with; on a clash the one with the larger
                     saving stays and the other moves down; a tie goes to the earlier (the
                     priority's conflict weight is always 0, `selection._build_priorities`, so
                     that tie-break never fires). Every group is conflict-free; the group with the largest
                     total saving is kept -- a greedy maximum-weight independent set. Here A
                     alone (114) loses to B + C (116): A blocks every other candidate
    (4) rewrite      the kept macros replace their gates; transversal two-qubit macros are found
                     next on what is left (none here), and the pass repeats until nothing merges

Savings are weighted gate counts, 1q + 13 x 2q (`iceberg.costs.DEFAULT_WORK_WEIGHTS`), at k = 2:
an HH saves two targeted H (7, 4) for one HH (4, 0); an XX two X (2, 0) for one XX (2, 0).

    PYTHONPATH=. venv/bin/python -m experiments.figures.algorithm_diagrams.merging_pass_conflicts \\
        [--output-dir DIR]
Writes merging_pass.{svg,pdf}; text stays text in the SVG (Inkscape-editable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

from experiments.figures.algorithm_diagrams.merging_algorithm import (
    HH_C, INK, OK, XX_C, Canvas, wires,
)

PATCH_A = "#e3f1ec"
PERMUTATION = [(0, 0), (0, 1)]
NOTE = dict(size=6.8, color="#4a4a4a", linespacing=1.25)
KIND_STYLE = {"share": dict(color="#c0392b", linestyle="-"),
              "cross": dict(color="#c0392b", linestyle=(0, (3, 2))),
              "window": dict(color="#c0392b", linestyle=(0, (1, 1.6)))}
KIND_LABEL = {"share": "share a gate", "cross": "cross in time",
              "window": "would reorder a gate"}


def program():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    for gate, q in (("h", 0), ("x", 1), ("s", 0), ("s", 1), ("x", 0), ("h", 1), ("s", 0),
                    ("s", 1), ("h", 0), ("x", 1)):
        getattr(qc, gate)(q)
    return qc


def trace():
    """Candidates (gates, saving), typed conflict edges, the groups, the kept group, the
    rewritten circuit and its saving -- all from the merging pass."""
    from qiskit.converters import circuit_to_dag

    from translator.merging import conflicts as C
    from translator.merging import selection as S
    from translator.merging.layer import layer_by_node
    from translator.merging.preprocessing import _build_patch_lanes, _merge_savings_for_records
    from translator.merging.rewrite import build_rewritten_circuit
    from translator.merging.settings import DEFAULT_SETTINGS
    from translator.merging.single_qubit import (
        find_hh_merges_for_patch, find_xx_merges_for_patch, find_zz_merges_for_patch,
    )

    work = C.merge_work_table(4)
    circuit = program()
    dag = circuit_to_dag(circuit)
    layer, order = layer_by_node(dag)
    index = {q: circuit.find_bit(q).index for q in circuit.qubits}
    lanes = _build_patch_lanes(PERMUTATION, circuit.qubits)
    cands = []
    for qubits in lanes.values():
        for finder in (find_hh_merges_for_patch, find_xx_merges_for_patch,
                       find_zz_merges_for_patch):
            cands.extend(finder(layer, order, qubits, DEFAULT_SETTINGS, set()))
    priorities = S._build_priorities(cands, range(len(cands)), layer, order, work)
    # typed edges, from the detectors `compute_merge_conflicts` combines
    kinds = {}
    for i, js in C._bentley_ottmann_conflicts(C._build_segments(cands, layer, index)).items():
        for j in js:
            kinds.setdefault(tuple(sorted((i, j))), set()).add("cross")
    shared = {}
    C._add_shared_endpoint_conflicts(cands, shared)
    for i, js in shared.items():
        for j in js:
            kinds.setdefault(tuple(sorted((i, j))), set()).add("share")
    window = {}
    C._add_first_in_window_conflicts(cands, layer, order, window)
    for i, js in window.items():
        for j in js:
            kinds.setdefault(tuple(sorted((i, j))), set()).add("window")
    bridged = {}
    C._add_cross_gate_conflicts(cands, layer, order, dag, bridged)
    for i, js in bridged.items():
        for j in js:
            kinds.setdefault(tuple(sorted((i, j))), set()).add("via a 2q gate")
    infos = C.compute_merge_conflicts(cands, layer, index, dag=dag, order_index=order,
                                      work_table=work)
    assert {tuple(sorted((i, j))) for i, info in enumerate(infos) for j in info.conflicts} \
        == set(kinds), "typed edges must be exactly the pass's conflict graph"
    # the greedy grouping, exactly as `select_best_group` runs it
    info = S._filter_conflicts(infos, range(len(cands)), priorities)
    group_of = {i: None for i in range(len(cands))}
    pending = [(i, 0) for i in sorted(range(len(cands)), key=lambda i: priorities[i].order_key)]
    seen = set()
    while pending:
        i, g = pending.pop(0)
        key = (i, g, group_of.get(i))
        if key in seen:
            continue
        seen.add(key)
        group_of[i] = None
        S._place_merge(i, g, group_of, info, priorities, pending)
    kept = S.select_best_group(cands, layer, order, index, dag=dag, work_table=work)
    kept_ids = [cands.index(r) for r in kept]
    saved = _merge_savings_for_records(kept, work)
    rewritten = build_rewritten_circuit(circuit, dag, kept, layer, order)

    def gates_of(r):
        return sorted(((layer[n], index[n.qargs[0]], n.op.name.upper()) for n in r.nodes))

    order_ids = sorted(range(len(cands)), key=lambda i: priorities[i].order_key)
    letters = {i: "ABCD"[k] for k, i in enumerate(sorted(
        order_ids, key=lambda i: (cands[i].macro_gate.name != "hh", priorities[i].order_key)))}
    return dict(
        candidates=[dict(id=i, letter=letters[i], macro=cands[i].macro_gate.name.upper(),
                         gates=gates_of(cands[i]), saving=priorities[i].savings_weight,
                         group=group_of[i]) for i in range(len(cands))],
        edges={(letters[i], letters[j]): sorted(k) for (i, j), k in kinds.items()},
        kept=sorted(letters[i] for i in kept_ids), saved=saved,
        before=len(circuit.data), after=len(rewritten.data),
        rewritten=[(i.operation.name, [rewritten.find_bit(q).index for q in i.qubits])
                   for i in rewritten.data],
    )


def draw(t: dict, out_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    W, H = FULL_WIDTH, 2.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = Canvas(ax)
    title_y = H - 0.14
    cands = {d["letter"]: d for d in t["candidates"]}
    colour = {L: (HH_C if d["macro"] == "HH" else XX_C) for L, d in cands.items()}

    # ---- (1) candidates on the program --------------------------------------------------------
    pitch, top = 0.30, H - 0.55
    ys = {0: top, 1: top - pitch}
    x0, step = 0.28, 0.30
    cols = [x0 + 0.17 + i * step for i in range(5)]
    c.text((x0 + cols[-1] + 0.15) / 2, title_y, "(1) Propose candidates", size=8.8)
    c.band(x0, ys[1] - 0.10, cols[-1] + 0.15 - x0, pitch + 0.20, PATCH_A)
    wires(c, x0, cols[-1] + 0.15, ys)
    sequence = {0: ["H", "S", "X", "S", "H"], 1: ["X", "S", "H", "S", "X"]}
    for q, gates in sequence.items():
        for x, g in zip(cols, gates):
            c.box(x, ys[q], 0.13, 0.11, g)
    for i, L in enumerate(sorted(cands)):
        (l1, q1, _), (l2, q2, _) = cands[L]["gates"]
        a, b = (cols[l1], ys[q1]), (cols[l2], ys[q2])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour[L], linewidth=6.5, alpha=0.33,
                solid_capstyle="round", linestyle="-", marker="none", zorder=2, gid=f"cand-{L}")
        # the letter sits on the band a quarter of the way along, between gate columns
        lx, ly_ = a[0] + 0.25 * (b[0] - a[0]), a[1] + 0.25 * (b[1] - a[1])
        ax.text(lx, ly_, L, fontsize=6.8, color="white", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=colour[L], edgecolor="none"))
    for i, x in enumerate(cols):
        c.text(x, ys[1] - 0.20, f"L{i}", size=6.2, color="#6a6a6a")
    c.text((x0 + cols[-1] + 0.15) / 2, ys[1] - 0.36,
           "A, B: HH (fuse two H)    C, D: XX (fuse two X)\n"
           "each pairs gates two layers apart", va="top", **NOTE)

    # ---- (2) conflict graph -------------------------------------------------------------------
    gx, gy = 2.30, top - 0.05
    c.text(gx + 0.62, title_y, "(2) Conflict graph", size=8.8)
    at = {"A": (gx + 0.20, gy), "B": (gx + 1.04, gy), "C": (gx + 0.20, gy - 0.58),
          "D": (gx + 1.04, gy - 0.58)}
    for (a, b), ks in t["edges"].items():
        kind = "window" if "window" in ks else ("share" if "share" in ks else "cross")
        style = KIND_STYLE[kind]
        (xa, ya), (xb, yb) = at[a], at[b]
        c.line([xa, xb], [ya, yb], color=style["color"], width=1.1, linestyle=style["linestyle"],
               zorder=1)
    for L, (x, y) in at.items():
        c.box(x, y, 0.30, 0.19, f"{L}  {cands[L]['macro']}", face="white", edge=colour[L],
              color=colour[L], size=6.8, rounding=0.05, width=0.9)
        c.text(x, y - (0.17 if y < gy else -0.17), f"saves {cands[L]['saving']:g}", size=6.2,
               color="#5a5a5a")
    ly = gy - 0.95
    for k, kind in enumerate(("share", "cross", "window")):
        style = KIND_STYLE[kind]
        yy = ly - k * 0.13
        c.line([gx - 0.05, gx + 0.20], [yy, yy], color=style["color"], width=1.1,
               linestyle=style["linestyle"])
        c.text(gx + 0.26, yy, KIND_LABEL[kind], size=6.4, color="#4a4a4a", ha="left")

    # ---- (3) groups: greedy colouring, keep the best ------------------------------------------
    sx = 3.95
    c.text(sx + 0.80, title_y, "(3) Group, keep the best", size=8.8)
    groups = {}
    for L, d in cands.items():
        groups.setdefault(d["group"], []).append(L)
    for g, (members) in sorted(groups.items()):
        x = sx + 0.10 + g * 0.55
        total = sum(cands[L]["saving"] for L in members)
        keep = set(members) == set(t["kept"])
        h = 0.30 + 0.26 * len(members)
        ax.add_patch(FancyBboxPatch((x - 0.04, gy - h + 0.20), 0.46, h,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor="#e2f3ec" if keep else "#f3f0ea",
                                    edgecolor=OK if keep else "#9a948a", linewidth=0.9 if keep else 0.6,
                                    zorder=0, gid=f"group-{g}"))
        c.text(x + 0.19, gy + 0.08, f"group {g}", size=6.4, color="#5a5a5a")
        for k, L in enumerate(sorted(members)):
            c.box(x + 0.19, gy - 0.14 - k * 0.26, 0.30, 0.19, f"{L}  {cands[L]['macro']}",
                  face=colour[L], edge=colour[L], color="white", size=6.8, rounding=0.05)
        c.text(x + 0.19, gy - h + 0.10, f"{total:g}", size=7.2, color=OK if keep else INK)
    c.text(sx + 0.80, ly + 0.10,
           "in time order, each candidate joins the first\ngroup with no conflict; on a clash "
           "the larger\nsaving stays, a tie goes to the earlier (A).\n"
           "Keep the group with the largest total:\nB + C (116) beat A (114)", va="top", **NOTE)

    # ---- (4) rewrite --------------------------------------------------------------------------
    rx = 5.85
    c.text(rx + 0.62, title_y, "(4) Rewrite", size=8.8)
    c.band(rx, ys[1] - 0.10, 1.22, pitch + 0.20, PATCH_A)
    wires(c, rx, rx + 1.22, ys, labels=False)

    def macro(x, label, col):
        c.box(x, (ys[0] + ys[1]) / 2, 0.20, pitch + 0.11, label, face=col, edge=col,
              color="white", size=6.4, rounding=0.025)

    c.box(rx + 0.10, ys[0], 0.12, 0.10, "H", size=6)
    c.box(rx + 0.27, ys[0], 0.12, 0.10, "S", size=6)
    macro(rx + 0.47, "XX", XX_C)
    c.box(rx + 0.67, ys[0], 0.12, 0.10, "S", size=6)
    c.box(rx + 0.67, ys[1], 0.12, 0.10, "S", size=6)
    macro(rx + 0.87, "HH", HH_C)
    c.box(rx + 1.05, ys[1], 0.12, 0.10, "S", size=6)
    c.box(rx + 1.18, ys[1], 0.12, 0.10, "X", size=6)
    one, two = t["saved"]
    c.text(rx + 0.60, ys[1] - 0.36,
           f"{t['before']} gates $\\rightarrow$ {t['after']}; saves\n{one:g} one-qubit and "
           f"{two:g} two-qubit\ngates (A alone: 10 and 8).\nThen two-qubit macros;\n"
           "repeat until nothing merges", va="top", **NOTE)

    for a, b, y in ((cols[-1] + 0.25, gx - 0.12, (ys[0] + ys[1]) / 2),
                    (gx + 1.30, sx - 0.02, (ys[0] + ys[1]) / 2),
                    (sx + 1.72, rx - 0.08, (ys[0] + ys[1]) / 2)):
        c.arrow((a, y), (b, y), gid="flow")

    paths = save_figure(fig, "merging_pass_conflicts", Path(__file__),
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
