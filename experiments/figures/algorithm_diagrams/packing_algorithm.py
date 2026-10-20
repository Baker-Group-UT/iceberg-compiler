"""
packing_algorithm.py -- AQORN's packing search on one small program, step by step.

The program (`packing_example.qpy` / `.qasm` beside this script) is a 6-qubit, 26-gate program:
`experiments.shared.random_circuits.structured_program_circuit(6, 4, 3, ...)` after the
verified logical-optimisation pass, i.e. what the planner sees in the standard pipeline. It was
picked from 1800 such programs because every choice in it can be read off the signed graph and
none is trivial: the seed matters, its descent improves it, and the descent from the naive
packing gets stuck. Every number is read off the final planner (`plan_standard`, branch
v2-teddy-final) by `trace()`:

    (a) signed graph    edge weight = patch benefit of sharing a patch minus 10 x the silent
                        price of the in-patch gates (`packing.silent.silent_aware_graph`); CZ and
                        XCX pairs are cheaper inside a patch (transversal, no silent error), CX
                        pairs are not
    (b) partition       the seed: pair the most positive edges first; the last pair is forced
                        (`packing.silent._seed`)
    (c) seed descent    one swap breaks the forced negative pair (`packing.search.descend`)
    (d) naive descent   the planner also descends from the naive packing; it stops in a worse
                        local optimum: every swap that brings q1 and q4 together also pairs q3
                        with q5 across a negative edge (checked: `naive_best_move` >= 0)
    then                keep the lower; lanes inside each patch (`assign_lanes`); (a)-(d) run
                        once per H-commutation setting, keeping the lower standard objective
                        (detected + 10 x silent errors per shot on the standard schedule);
                        optionally the final check against the naive packing

    PYTHONPATH=. venv/bin/python -m experiments.figures.algorithm_diagrams.packing_algorithm \\
        [--output-dir DIR] [--with-final-check]
Writes packing_algorithm.{svg,pdf}; text stays text in the SVG (Inkscape-editable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / "packing_example.qpy"

PATCH = "#cde8df"
NODE = "#f4f0e9"
NEG = "#d6542b"
POS = "#8a8a8a"
INK = "#2b2b2b"
BOX = "#f3f0ea"
KEEP = "#e2f3ec"
KEEP_EDGE = "#2f7d62"

#: Node positions (column, row): every patch of the seed, the seed's descent and the naive
#: descent is a pair of neighbours, and no two edges cross.
POS_XY = {0: (0, 1), 3: (1, 1), 4: (2, 1), 2: (0, 0), 5: (1, 0), 1: (2, 0)}
#: Edges drawn as arcs because a straight line would run through a node.
ARCS = {(1, 2): -0.30}
#: Where each weight label sits relative to its edge's midpoint (inches), clear of lines.
LABEL_AT = {(0, 2): (-0.17, 0.0), (0, 3): (0.0, 0.10), (1, 4): (0.16, 0.14),
            (2, 3): (0.12, -0.07), (3, 5): (0.17, 0.0)}

R = 0.105            # node radius, inches
DX, DY = 0.56, 0.56  # node spacing, inches


def _sweep(packing):
    """`packing.search.descend`'s sweep, logging each accepted move."""
    from packing.search import moves_for

    log = []
    for _ in range(8):
        improved = False
        for qubit in sorted(packing.patch_of):
            current, best = packing.value(), None
            for move in moves_for(packing, qubit):
                value = packing.value_after(move.as_dict())
                if value < current - 1e-9 and (best is None or value < best[0]):
                    best = (value, move)
            if best is not None:
                before = dict(packing.patch_of)
                packing.apply(best[1].as_dict())
                moved = sorted(q for q, p in packing.patch_of.items() if before[q] != p)
                log.append({"qubits": moved, "value": best[0]})
                improved = True
        if not improved:
            break
    return log


def trace():
    """The numbers in the figure, from the final planner on the example program."""
    from qiskit import qpy

    from experiments.optimality.standard_packers import standard_gadgets
    from iceberg.utils import logical_capacity
    from packing.search import moves_for, naive_seed
    from packing.seed import DEFAULT_PARTITIONER
    from packing.silent import _seed, build_costs, load_silent_rates, silent_aware_graph
    from packing.standard import STANDARD, plan_standard, standard_noise_profile

    with open(EXAMPLE, "rb") as handle:
        program = qpy.load(handle)[0]
    noise = standard_noise_profile()
    with standard_gadgets():
        plan = plan_standard(program)
        commute = bool(plan.h_commutation)
        rates = load_silent_rates(4, transversal_intra_two_qubit=True)
        profiles, cost = build_costs(program, 4, commute=commute, noise=noise, rates=rates,
                                     silent_weight=STANDARD.silent_weight_for(4),
                                     solo_hh=STANDARD.solo_hh_aware)
        signed = {pair: w for pair, w in silent_aware_graph(cost, profiles, noise).weights.items()
                  if abs(w) > 1e-9}
        gates = {pair: sorted({kind for _, kind in events}) for pair, events in cost.gates.items()}
        slots = max(1, -(-program.num_qubits // logical_capacity(4)))
        seed = _seed(cost, profiles, slots, noise=noise, partitioner=DEFAULT_PARTITIONER)
        seed_groups, seed_value = [sorted(g) for g in seed.groups], seed.value()
        seed_moves = _sweep(seed)
        naive = naive_seed(cost, slots)
        naive_groups, naive_value = [sorted(g) for g in naive.groups], naive.value()
        naive_moves = _sweep(naive)
        naive_best_move = min((naive.value_after(m.as_dict()) - naive.value()
                               for q in naive.patch_of for m in moves_for(naive, q)), default=0.0)
        considered = dict(plan.considered)
    placement = [tuple(p) for p in plan.placement]
    final = sorted(sorted(q for q, (p, _) in enumerate(placement) if p == patch)
                   for patch in {p for p, _ in placement})
    return dict(
        gates_in_program=len(program.data), commute=commute, signed=signed, gates=gates,
        seed=seed_groups, seed_value=seed_value, seed_moves=seed_moves,
        seed_after=[sorted(g) for g in seed.groups], seed_after_value=seed.value(),
        naive=naive_groups, naive_value=naive_value, naive_moves=naive_moves,
        naive_after=[sorted(g) for g in naive.groups], naive_after_value=naive.value(),
        naive_best_move=naive_best_move, final=final,
        commuted=considered.get("commute=True"), not_commuted=considered.get("commute=False"),
        packed=considered.get("packed"), naive_check=considered.get("naive"),
        fell_back=bool(plan.fell_back),
    )


def _xy(origin, q):
    return origin[0] + POS_XY[q][0] * DX, origin[1] + POS_XY[q][1] * DY


def _node(ax, x, y, label):
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), R, facecolor=NODE, edgecolor="#5a5a5a", linewidth=0.7,
                        zorder=4, gid=f"node-{label}"))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color=INK, zorder=5)


def _patch(ax, members, origin, gid):
    """A capsule around a patch's nodes, along the line between them."""
    pts = [_xy(origin, q) for q in members]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if len(pts) == 1:
        xs, ys = xs * 2, ys * 2
    ax.plot(xs, ys, color=PATCH, linewidth=2 * (R + 0.05) * 72, solid_capstyle="round",
            linestyle="-", marker="none", alpha=0.9, zorder=1, gid=gid)


def _graph(ax, origin, signed, patches=(), labels=False, highlight=(), gid="graph"):
    from matplotlib.patches import FancyArrowPatch

    for i, group in enumerate(patches):
        _patch(ax, group, origin, f"{gid}-patch-{i}")
    for (a, b), w in signed.items():
        key = (min(a, b), max(a, b))
        (xa, ya), (xb, yb) = _xy(origin, a), _xy(origin, b)
        negative = w < 0
        width = 2.6 if key in highlight else (1.5 if negative else 1.1)
        colour = NEG if negative else POS
        if key in ARCS:
            ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-", linewidth=width,
                                         color=colour, connectionstyle=f"arc3,rad={ARCS[key]}",
                                         shrinkA=0, shrinkB=0, zorder=2,
                                         gid=f"{gid}-edge-{a}-{b}"))
        else:
            ax.plot([xa, xb], [ya, yb], color=colour, linewidth=width, linestyle="-",
                    marker="none", solid_capstyle="round", zorder=2, gid=f"{gid}-edge-{a}-{b}")
        if labels:
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            if key in ARCS:
                my = min(ya, yb) - 0.27
            else:
                dx, dy = LABEL_AT.get(key, (0.12, 0.05))
                mx, my = mx + dx, my + dy
            ax.text(mx, my, f"+{w:.0f}" if w > 0 else f"$-${abs(w):.0f}", fontsize=7,
                    color=NEG if negative else INK, ha="center", va="center", zorder=6)
    for q in POS_XY:
        x, y = _xy(origin, q)
        _node(ax, x, y, f"q{q}")


def _box(ax, x, y, w, h, title, body, keep=False, gid="box"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=KEEP if keep else BOX,
                                edgecolor=KEEP_EDGE if keep else "#9a948a", linewidth=0.7,
                                zorder=1, gid=gid))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=8.5,
            color=KEEP_EDGE if keep else INK, zorder=2)
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=7.2,
            color=KEEP_EDGE if keep else "#4a4a4a", zorder=2, linespacing=1.25)


def _arrow(ax, start, end, gid="arrow", **kwargs):
    from matplotlib.patches import FancyArrowPatch

    kwargs.setdefault("arrowstyle", "-|>")
    ax.add_patch(FancyArrowPatch(start, end, mutation_scale=7, color="#7a7a7a",
                                 linewidth=0.8, zorder=3, gid=gid, shrinkA=0, shrinkB=0,
                                 **kwargs))


def _groups(groups):
    return ", ".join("{" + ", ".join(f"q{q}" for q in g) + "}" for g in groups)


def _caption(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="top", fontsize=7.2, color="#3a3a3a", linespacing=1.3)


def _weight(t, group):
    if len(group) < 2:
        return 0.0
    a, b = group
    return t["signed"].get((min(a, b), max(a, b)), 0.0)


def _signed(w):
    return f"+{w:.0f}" if w > 0 else f"$-${abs(w):.0f}"


def draw(t: dict, out_dir=None, with_final_check: bool = False):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    W, H = FULL_WIDTH, 3.62
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    signed = t["signed"]

    panel = W / 4
    centres = [panel * (i + 0.5) for i in range(4)]
    bottom_row = 2.36
    origins = [(c - DX, bottom_row) for c in centres]
    titles = ["(a) Signed graph", "(b) Partition: the seed", "(c) Descent from the seed",
              "(d) Descent from naive"]
    for c, title in zip(centres, titles):
        ax.text(c, H - 0.13, title, ha="center", va="center", fontsize=9, color=INK)
    cap_y = bottom_row - R - 0.30

    # (a) the signed graph, with what makes an edge positive or negative
    kinds = {}
    for pair, w in signed.items():
        for kind in t["gates"].get(pair, []):
            kinds.setdefault(kind.upper(), set()).add("+" if w > 0 else "-")
    pos_kinds = " and ".join(sorted(k for k, s in kinds.items() if s == {"+"}))
    neg_kinds = " and ".join(sorted(k for k, s in kinds.items() if s == {"-"}))
    _graph(ax, origins[0], signed, labels=True, gid="a")
    _caption(ax, centres[0], cap_y,
             "weight = patch benefit $-$ 10 $\\times$ silent price:\n"
             f"{pos_kinds} pairs are cheaper inside a\npatch (transversal, no silent error);\n"
             f"{neg_kinds} pairs are not")

    # (b) the partition seed, its forced pair highlighted
    seed = t["seed"]
    forced = min(seed, key=lambda g: _weight(t, g))
    first = sorted((g for g in seed if g != forced), key=lambda g: -_weight(t, g))
    _graph(ax, origins[1], signed, patches=seed, highlight={tuple(forced)}, gid="b")
    _caption(ax, centres[1], cap_y,
             "most positive pairs first:\n"
             + ", ".join(f"{_groups([g])} {_signed(_weight(t, g))}" for g in first) + ";\n"
             f"the last two are forced together:\n{_groups([forced])} across "
             f"{_signed(_weight(t, forced))}.  Objective {t['seed_value']:.1f}")

    # (c) the descent from the seed: the swap that breaks the forced pair
    _graph(ax, origins[2], signed, patches=t["seed_after"], gid="c")
    moved = t["seed_moves"][0]["qubits"] if t["seed_moves"] else []
    if len(moved) == 2:
        (xa, ya), (xb, yb) = _xy(origins[2], moved[0]), _xy(origins[2], moved[1])
        _arrow(ax, (xa + 0.10, ya - 0.05), (xb - 0.10, yb + 0.05), arrowstyle="<|-|>",
               connectionstyle="arc3,rad=-0.35", gid="c-swap")
    swap = "$\\leftrightarrow$".join(f"q{q}" for q in moved)
    _caption(ax, centres[2], cap_y,
             f"swap {swap} breaks the forced pair,\nkeeps {{q1, q4}}, trades {{q0, q2}} for\n"
             f"{{q0, q3}} (+50 either way):\n{t['seed_value']:.1f} $\\rightarrow$ "
             f"{t['seed_after_value']:.1f}; then no move lowers it")

    # (d) the descent from naive and where it stops
    _graph(ax, origins[3], signed, patches=t["naive_after"], gid="d")
    _caption(ax, centres[3], cap_y,
             f"start {_groups(t['naive'])}:\n"
             f"{len(t['naive_moves'])} swaps, {t['naive_value']:.1f} $\\rightarrow$ "
             f"{t['naive_after_value']:.1f}, then stuck:\n"
             "every swap pairing q1 with q4 also\npairs q3 with q5 across a negative edge")

    for i in range(3):
        _arrow(ax, (centres[i] + DX + R + 0.12, bottom_row + DY / 2),
               (centres[i + 1] - DX - R - 0.12, bottom_row + DY / 2), gid="row-arrow")

    # bottom row: keep the lower, lanes, the H decision, (optionally) the final check
    y, h = 0.50, 0.50
    boxes = [("Keep the lower", f"seed {t['seed_after_value']:.1f} < naive "
              f"{t['naive_after_value']:.1f}", True),
             ("Assign lanes", "lane order inside each patch", False),
             ("Both H settings", f"(a)$-$(d) once per setting; keep the\nlower: commuted "
              f"{t['commuted']:.3f} vs not {t['not_commuted']:.3f}", False)]
    if with_final_check:
        boxes.append(("Final check", f"plan {t['packed']:.3f} vs\nnaive {t['naive_check']:.3f}",
                      False))
    width, gap = (2.00, 0.32) if not with_final_check else (1.55, 0.22)
    total = len(boxes) * width + (len(boxes) - 1) * gap
    bx0 = (W - total) / 2
    for i, (title, body, keep) in enumerate(boxes):
        x = bx0 + i * (width + gap)
        _box(ax, x, y, width, h, title, body, keep=keep, gid=f"box-{i}")
        if i:
            _arrow(ax, (x - gap + 0.04, y + h / 2), (x - 0.04, y + h / 2), gid="box-arrow")
    # (c) and (d) both feed "keep the lower"
    turn = y + h + 0.15
    for c in (centres[2], centres[3]):
        ax.plot([c, c, bx0 + width * 0.5], [cap_y - 0.66, turn, turn], color="#7a7a7a",
                linewidth=0.8, linestyle="-", marker="none", zorder=3, gid="down-line")
    _arrow(ax, (bx0 + width * 0.5, turn), (bx0 + width * 0.5, y + h + 0.01), gid="down-arrow")
    ax.text(W / 2, 0.30,
            f"Result: patches {_groups(t['final'])}, H commuted.   (a)$-$(d): packing objective, "
            "lower is better; H settings: detected + 10 $\\times$ silent errors per shot.",
            ha="center", va="center", fontsize=6.8, color="#4a4a4a")

    # legend
    from matplotlib.patches import FancyBboxPatch

    lx, ly = W / 2 - 1.75, 0.10
    ax.add_patch(FancyBboxPatch((lx, ly - 0.035), 0.28, 0.07, boxstyle="round,pad=0,"
                                "rounding_size=0.03", facecolor=PATCH, edgecolor="none"))
    ax.text(lx + 0.34, ly, "patch", va="center", fontsize=7.2, color=INK)
    ax.plot([lx + 0.95, lx + 1.25], [ly, ly], color=NEG, linewidth=1.5, linestyle="-",
            marker="none")
    ax.text(lx + 1.31, ly, "negative: keep apart", va="center", fontsize=7.2, color=INK)
    ax.plot([lx + 2.45, lx + 2.75], [ly, ly], color=POS, linewidth=1.1, linestyle="-",
            marker="none")
    ax.text(lx + 2.81, ly, "positive: sharing pays", va="center", fontsize=7.2, color=INK)

    name = "packing_algorithm_with_check" if with_final_check else "packing_algorithm"
    paths = save_figure(fig, name, Path(__file__), override=Path(out_dir) if out_dir else None)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--with-final-check", action="store_true",
                        help="also draw the final check against the naive packing")
    args = parser.parse_args()
    t = trace()
    for key in ("gates_in_program", "seed", "seed_value", "seed_moves", "seed_after",
                "seed_after_value", "naive", "naive_value", "naive_moves", "naive_after",
                "naive_after_value", "naive_best_move", "final", "commuted", "not_commuted",
                "packed", "naive_check", "fell_back"):
        print(f"[packing_algorithm] {key}: {t[key]}")
    if t["naive_best_move"] < -1e-9:
        raise SystemExit("[packing_algorithm] the naive descent is not at a local optimum")
    for path in draw(t, args.output_dir, args.with_final_check):
        print(f"[packing_algorithm] saved {path}")


if __name__ == "__main__":
    main()
