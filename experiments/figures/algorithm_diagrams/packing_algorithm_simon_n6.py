"""
packing_algorithm_simon_n6.py -- the first version of the packing diagram (simon_n6),
kept for reference: there the seed's descent finds nothing and the plan comes from the naive
start, so it does not show why the seed matters. `packing_algorithm.py` is the paper version.

The program is simon_n6 (Fig 6's corpus), planned by the final compiler (`plan_standard`,
v2-teddy-final: no random restarts, H-commutation decision on `standard_objective`). Every
number is read off that plan by `trace()` below, so the figure cannot drift from the code:

    (a) signed graph   edge weight = the patch benefit of sharing a patch minus 10 x the silent
                       price of the in-patch gates (`packing.silent.silent_aware_graph`); the
                       q0-q1-q3 triangle (4 CX per pair) is negative, the two XCX pairs positive,
                       q5 is idle
    (b) partition      the seed: merge the most positive pair first, keep negative pairs apart
                       (`packing.silent._seed`); its descent finds no improving move
    (c) naive start    the planner also descends from the naive packing, where q0 and q1 share a
                       patch across a negative edge; one swap separates them
    (d) keep the lower the two descents' objectives; the lower is the plan
    then               lanes inside each patch (`assign_lanes`), and (a)-(d) run once per
                       H-commutation setting, keeping the lower standard objective
                       (detected + 10 x silent errors per shot on the standard schedule)

    PYTHONPATH=. venv/bin/python -m experiments.figures.algorithm_diagrams.packing_algorithm_simon_n6 \\
        [--output-dir DIR] [--with-final-check]
Writes packing_algorithm.{svg,pdf}; text stays text in the SVG (Inkscape-editable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

PATCH = "#cde8df"
NODE = "#f4f0e9"
NEG = "#d6542b"
POS = "#8a8a8a"
INK = "#2b2b2b"
BOX = "#f3f0ea"
KEEP = "#e2f3ec"
KEEP_EDGE = "#2f7d62"

#: Node positions in a panel's local frame: two rows of three, as in the program's layout.
POS_XY = {0: (0, 1), 4: (1, 1), 2: (2, 1), 5: (0, 0), 1: (1, 0), 3: (2, 0)}


def trace():
    """The numbers in the figure, from the final planner on simon_n6."""
    from experiments.optimality.standard_packers import standard_gadgets
    from experiments.revision.common import load_fig6_circuits
    from packing.search import descend, moves_for, naive_seed
    from packing.seed import DEFAULT_PARTITIONER
    from packing.silent import _seed, build_costs, load_silent_rates, silent_aware_graph
    from packing.standard import STANDARD, plan_standard, standard_noise_profile

    program = next(c for c in load_fig6_circuits() if c.name == "simon_n6").original
    noise = standard_noise_profile()
    with standard_gadgets():
        plan = plan_standard(program)
        commute = bool(plan.h_commutation)
        rates = load_silent_rates(4, transversal_intra_two_qubit=True)
        profiles, cost = build_costs(program, 4, commute=commute, noise=noise, rates=rates,
                                     silent_weight=STANDARD.silent_weight_for(4),
                                     solo_hh=STANDARD.solo_hh_aware)
        signed = silent_aware_graph(cost, profiles, noise).weights
        gates = {pair: [kind for _, kind in events] for pair, events in cost.gates.items()}
        seed = _seed(cost, profiles, 3, noise=noise, partitioner=DEFAULT_PARTITIONER)
        seed_groups = [sorted(g) for g in seed.groups]
        seed_value = seed.value()
        best_seed_move = min(seed.value_after(m.as_dict()) - seed_value
                             for q in seed.patch_of for m in moves_for(seed, q))
        descend(seed, max_rounds=8)
        naive = naive_seed(cost, 3)
        naive_groups, naive_value = [sorted(g) for g in naive.groups], naive.value()
        descend(naive, max_rounds=8)
        considered = dict(plan.considered)
    placement = [tuple(p) for p in plan.placement]
    final = sorted(sorted(q for q, (p, _) in enumerate(placement) if p == patch)
                   for patch in {p for p, _ in placement})
    return dict(
        commute=commute, signed=signed, gates=gates,
        seed=seed_groups, seed_value=seed_value, seed_after=seed.value(),
        best_seed_move=best_seed_move,
        naive=naive_groups, naive_value=naive_value,
        naive_after=[sorted(g) for g in naive.groups], naive_after_value=naive.value(),
        final=final,
        commuted=considered.get("commute=True"), not_commuted=considered.get("commute=False"),
        packed=considered.get("packed"), naive_check=considered.get("naive"),
        fell_back=bool(plan.fell_back),
    )


R = 0.105          # node radius, inches
DX, DY = 0.56, 0.56  # node spacing, inches


def _xy(origin, q):
    return origin[0] + POS_XY[q][0] * DX, origin[1] + POS_XY[q][1] * DY


def _node(ax, x, y, label):
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), R, facecolor=NODE, edgecolor="#5a5a5a", linewidth=0.7,
                        zorder=4, gid=f"node-{label}"))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color=INK, zorder=5)


def _patch(ax, members, origin, gid):
    """A capsule around a patch's 1-2 nodes, along the line between them (a diagonal pair, as in
    the naive start, gets a diagonal capsule rather than a box that would swallow its
    neighbours)."""
    pts = [_xy(origin, q) for q in members]
    width_pt = 2 * (R + 0.05) * 72
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if len(pts) == 1:
        xs, ys = xs * 2, ys * 2
    ax.plot(xs, ys, color=PATCH, linewidth=width_pt, solid_capstyle="round", linestyle="-",
            marker="none", alpha=0.9, zorder=1, gid=gid)


#: Edges that carry a weight label in panel (a): one triangle edge, both positive edges.
LABELLED = {(1, 3): (0.0, -0.11), (2, 3): (-0.15, 0.0), (2, 4): (0.0, 0.11)}


def _graph(ax, origin, signed, patches=(), labels=False, highlight=None, gid="graph"):
    for i, group in enumerate(patches):
        _patch(ax, group, origin, f"{gid}-patch-{i}")
    for (a, b), w in signed.items():
        (xa, ya), (xb, yb) = _xy(origin, a), _xy(origin, b)
        negative = w < 0
        width = 2.4 if (highlight and {a, b} == set(highlight)) else (1.5 if negative else 1.0)
        ax.plot([xa, xb], [ya, yb], color=NEG if negative else POS, linewidth=width,
                linestyle="-", marker="none", solid_capstyle="round", zorder=2,
                gid=f"{gid}-edge-{a}-{b}")
        key = (min(a, b), max(a, b))
        if labels and key in LABELLED:
            dx, dy = LABELLED[key]
            ax.text((xa + xb) / 2 + dx, (ya + yb) / 2 + dy,
                    f"+{w:.0f}" if w > 0 else f"$-${abs(w):.0f}", fontsize=7, color=INK,
                    ha="center", va="center", zorder=6)
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


def draw(t: dict, out_dir=None, with_final_check: bool = False):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    W, H = FULL_WIDTH, 3.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    signed = {pair: w for pair, w in t["signed"].items() if abs(w) > 1e-9}

    panel = W / 4
    centres = [panel * (i + 0.5) for i in range(4)]
    bottom_row = 2.30                                  # y of the q5 / q1 / q3 row
    origins = [(c - DX, bottom_row) for c in centres]
    titles = ["(a) Signed graph", "(b) Partition: the seed", "(c) Descent from naive",
              "(d) Keep the lower"]
    for c, title in zip(centres, titles):
        ax.text(c, H - 0.13, title, ha="center", va="center", fontsize=9, color=INK)

    neg = [w for w in signed.values() if w < 0]
    pos = [w for w in signed.values() if w > 0]
    cap_y = bottom_row - R - 0.13

    # (a) the signed graph
    _graph(ax, origins[0], signed, labels=True, gid="a")
    _caption(ax, centres[0], cap_y,
             "weight = patch benefit $-$ 10 $\\times$ silent price\n"
             f"triangle q0, q1, q3: 4 CX per pair, $-${abs(min(neg)):.0f}\n"
             f"q2$-$q3, q2$-$q4: one XCX each, +{max(pos):.0f}\nq5 idle")

    # (b) the partition seed and its (empty) descent
    _graph(ax, origins[1], signed, patches=t["seed"], gid="b")
    _caption(ax, centres[1], cap_y,
             "most positive pair first: {q2, q3};\nthe triangle kept apart\n"
             f"objective {t['seed_value']:.1f}\n"
             f"descent: no move lowers it\n(best: swap q0$\\leftrightarrow$q1, "
             f"$\\Delta$ = {t['best_seed_move']:.0f})")

    # (c) the naive packing, its bad pair, and the swap that fixes it
    _graph(ax, origins[2], signed, patches=t["naive"], highlight=(0, 1), gid="c")
    (x0, y0), (x4, y4) = _xy(origins[2], 0), _xy(origins[2], 4)
    _arrow(ax, (x0 + 0.04, y0 + R + 0.01), (x4 - 0.04, y4 + R + 0.01),
           connectionstyle="arc3,rad=-0.45", gid="c-swap", arrowstyle="<|-|>")
    ax.text((x0 + x4) / 2, y0 + R + 0.20, "swap", ha="center", va="bottom", fontsize=7,
            color="#5a5a5a")
    _caption(ax, centres[2], cap_y,
             "the planner also descends from\nthe naive packing: q0 and q1 share\n"
             "a patch across a negative edge\n"
             f"swap q0$\\leftrightarrow$q4: {t['naive_value']:.1f} $\\rightarrow$ "
             f"{t['naive_after_value']:.1f}")

    # (d) the lower of the two descents
    _graph(ax, origins[3], signed, patches=t["naive_after"], gid="d")
    _caption(ax, centres[3], cap_y,
             f"from naive {t['naive_after_value']:.1f} < seed {t['seed_after']:.1f}\n"
             "the same grouping as the seed\nup to q0$\\leftrightarrow$q1 (symmetric);\n"
             "only the patch order differs")

    for i in range(3):
        _arrow(ax, (centres[i] + DX + R + 0.10, bottom_row + DY / 2),
               (centres[i + 1] - DX - R - 0.10, bottom_row + DY / 2), gid="row-arrow")

    # bottom row: lanes, the H decision, (optionally) the final check
    y, h = 0.50, 0.50
    boxes = [("Assign lanes", "lane order inside each patch", False),
             ("Both H settings", f"(a)$-$(d) once per setting; keep the lower:\n"
              f"commuted {t['commuted']:.3f} vs not {t['not_commuted']:.3f}", False)]
    if with_final_check:
        boxes.append(("Final check", f"plan {t['packed']:.3f} vs naive {t['naive_check']:.3f}",
                      True))
    width, gap = (2.35, 0.45) if not with_final_check else (1.95, 0.30)
    total = len(boxes) * width + (len(boxes) - 1) * gap
    bx0 = (W - total) / 2
    for i, (title, body, keep) in enumerate(boxes):
        x = bx0 + i * (width + gap)
        _box(ax, x, y, width, h, title, body, keep=keep, gid=f"box-{i}")
        if i:
            _arrow(ax, (x - gap + 0.04, y + h / 2), (x - 0.04, y + h / 2), gid="box-arrow")
    # from (d) down, across, and into the first box
    turn = y + h + 0.17
    ax.plot([centres[3], centres[3], bx0 + width * 0.5], [cap_y - 0.66, turn, turn],
            color="#7a7a7a", linewidth=0.8, linestyle="-", marker="none", zorder=3,
            gid="down-line")
    _arrow(ax, (bx0 + width * 0.5, turn), (bx0 + width * 0.5, y + h + 0.01), gid="down-arrow")
    ax.text(W / 2, 0.30,
            f"Result: patches {_groups(t['final'])}, H commuted.   (a)$-$(d): packing objective, "
            "lower is better; H settings: detected + 10 $\\times$ silent errors per shot.",
            ha="center", va="center", fontsize=6.8, color="#4a4a4a")

    # legend, drawn by hand so it stays inside the canvas
    from matplotlib.patches import FancyBboxPatch

    lx, ly = W / 2 - 1.75, 0.10
    ax.add_patch(FancyBboxPatch((lx, ly - 0.035), 0.28, 0.07, boxstyle="round,pad=0,"
                                "rounding_size=0.03", facecolor=PATCH, edgecolor="none"))
    ax.text(lx + 0.34, ly, "patch", va="center", fontsize=7.2, color=INK)
    ax.plot([lx + 0.95, lx + 1.25], [ly, ly], color=NEG, linewidth=1.5, linestyle="-",
            marker="none")
    ax.text(lx + 1.31, ly, "negative: keep apart", va="center", fontsize=7.2, color=INK)
    ax.plot([lx + 2.45, lx + 2.75], [ly, ly], color=POS, linewidth=1.0, linestyle="-",
            marker="none")
    ax.text(lx + 2.81, ly, "positive: sharing pays", va="center", fontsize=7.2, color=INK)

    name = "packing_algorithm_simon_n6_with_check" if with_final_check else "packing_algorithm_simon_n6"
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
    for key in ("seed", "seed_value", "best_seed_move", "naive", "naive_value", "naive_after",
                "naive_after_value", "final", "commuted", "not_commuted", "packed",
                "naive_check", "fell_back"):
        print(f"[packing_algorithm] {key}: {t[key]}")
    for path in draw(t, args.output_dir, args.with_final_check):
        print(f"[packing_algorithm] saved {path}")


if __name__ == "__main__":
    main()
