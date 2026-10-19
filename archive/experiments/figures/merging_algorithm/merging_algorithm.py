"""
merging_algorithm.py
--------------------

The four steps gate merging takes, on one worked example.

A schematic, but not a hand-drawn one: every candidate, conflict edge, committed
pick and emitted instruction in the figure is read out of the real detectors in
``translator/merging``. The panels are

    (a) lanes      the permutation -- which program qubits share a patch, in which slot
    (b) propose    every gate leads one group, grown forward until it cannot extend
    (c) choose     conflict graph, greedy colouring, heaviest conflict-free class
    (d) emit       the rewritten instruction stream

Why the example looks the way it does. The interesting behaviour only exists at
k > 2, because at two lanes per patch a group of two *is* the whole patch and the
coverage condition is vacuous. So this runs at ``n_code = 6``, i.e. ``[[6, 4, 2]]``,
four lanes per patch and eight program qubits over two patches. The circuit is
chosen for what it makes the selector do rather than for being a useful program: two
layers of Hadamards over one patch produce *five* overlapping candidates -- four
rival whole-patch ``hh`` groups plus a per-lane ``xx`` pair -- with eight conflict
edges between them, so panel (c) shows a real choice instead of a trivial one.

The load-bearing claim of panel (c) is that nothing here solves an optimality
problem. Picking a maximum-weight set of pairwise-compatible macros is maximum-weight
independent set, which is NP-hard; ``selection.select_best_group`` instead does
first-fit graph colouring with savings-weighted eviction and returns the heaviest
single colour class. Every member of a class beyond the first conflicts with
something in class 0, so the classes cannot be unioned, and the heaviest class is not
in general the heaviest independent set. The caption should say "heuristic", not
"optimal".

Cost: negligible. One 8-qubit circuit through the detectors once, so ``--from-csv``
exists for symmetry with the other figure scripts rather than because recomputing
hurts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from matplotlib.patches import FancyBboxPatch, Rectangle
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.shared.figures import (  # noqa: E402
    FULL_WIDTH,
    data_dir,
    save_figure,
    use_style,
)
from translator.merging.conflicts import compute_merge_conflicts, merge_work_table  # noqa: E402
from translator.merging.layer import layer_by_node  # noqa: E402
from translator.merging.preprocessing import (  # noqa: E402
    _build_patch_lanes,
    preprocess_program_for_merging_with_stats,
)
from translator.merging.selection import (  # noqa: E402
    _build_priorities,
    _filter_conflicts,
    _is_single_qubit_merge,
    select_best_group,
)
from translator.merging.settings import DEFAULT_SETTINGS  # noqa: E402
from translator.merging.single_qubit import (  # noqa: E402
    find_hh_merges_for_patch,
    find_xx_merges_for_patch,
    find_zz_merges_for_patch,
)
from translator.merging.transversal import find_transversal_merges  # noqa: E402

NAME = "merging_algorithm"

#: [[6, 4, 2]] -- four lanes per patch. k = 2 would make the coverage rule vacuous,
#: which is exactly the case the figure is not about.
N_CODE = 6
K = N_CODE - 2

#: Ordered house palette, positions 1/2/4/5. See docs/figure-style.md.
ACCENT = "#0097fd"
DARK = "#3b4c56"
MID = "#636363"
LIGHT = "#bdbdbd"

MACRO_NAMES = {"hh", "xx", "zz", "cxt", "czt", "xcxt"}


# --------------------------------------------------------------------------- compute


def build_example() -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    """
    Eight program qubits over two four-lane patches.

    Two Hadamard layers on patch 0 with a pair of X gates between them, then a full
    transversal CNOT group across the patches. The Hadamards are what generate the
    overlapping candidates: every ``h`` is tried as a group leader, and a leader
    starting on a later lane can reach across the X layer to pick up a lane's
    *second* Hadamard, so the groups interleave rather than partition.
    """
    circuit = QuantumCircuit(2 * K)
    for qubit in range(K):
        circuit.h(qubit)
    circuit.x(0)
    circuit.x(2)
    for qubit in (0, 1, 2):
        circuit.h(qubit)
    for qubit in range(K):
        circuit.cx(qubit, qubit + K)
    permutation = [(i // K, i % K) for i in range(2 * K)]
    return circuit, permutation


def _row(kind: str, **fields) -> Dict[str, object]:
    base = {
        "kind": kind,
        "ident": "",
        "label": "",
        "patch": "",
        "slot": "",
        "qubit": "",
        "layer": "",
        "lanes": "",
        "weight": "",
        "conflicts": "",
        "committed": "",
        "detail": "",
    }
    base.update(fields)
    return base


def compute() -> pd.DataFrame:
    """Run the real merging stages on the worked example and tabulate every stage."""
    circuit, permutation = build_example()
    work_table = merge_work_table(N_CODE)
    dag = circuit_to_dag(circuit)
    layer_index, order_index = layer_by_node(dag)
    patch_lanes = _build_patch_lanes(permutation, circuit.qubits)
    qubit_indices = {qubit: circuit.find_bit(qubit).index for qubit in circuit.qubits}
    patch_of = {
        qubit_indices[qubit]: patch
        for patch, qubits in patch_lanes.items()
        for qubit in qubits
    }

    rows: List[Dict[str, object]] = []

    # (a) the permutation
    for logical, (patch, slot) in enumerate(permutation):
        rows.append(_row("lane", patch=patch, slot=slot, qubit=logical))

    # (b) the program gates, positioned by the pass's own layering
    for node, layer in layer_index.items():
        name = getattr(node.op, "name", "?")
        if len(node.qargs) != 1:
            continue
        index = qubit_indices[node.qargs[0]]
        rows.append(
            _row(
                "gate",
                label=name,
                qubit=index,
                layer=layer,
                patch=patch_of.get(index, ""),
                detail=order_index.get(node, 0),
            )
        )

    # (b)+(c) candidates, in the order the detectors propose them
    candidates = []
    for _, qubits in sorted(patch_lanes.items()):
        for finder in (
            find_hh_merges_for_patch,
            find_xx_merges_for_patch,
            find_zz_merges_for_patch,
        ):
            candidates.extend(
                finder(layer_index, order_index, qubits, DEFAULT_SETTINGS, set())
            )

    infos = compute_merge_conflicts(
        candidates, layer_index, qubit_indices, dag=dag, order_index=order_index
    )
    single = [i for i, merge in enumerate(candidates) if _is_single_qubit_merge(merge)]
    priorities = _build_priorities(candidates, single, layer_index, order_index, work_table)
    filtered = _filter_conflicts(infos, single, priorities)

    picked = select_best_group(
        candidates, layer_index, order_index, qubit_indices, dag=dag, work_table=work_table
    )
    picked_ids = {candidates.index(record) for record in picked}

    for idx, merge in enumerate(candidates):
        nodes = ";".join(
            f"{qubit_indices[node.qargs[0]]}@{layer_index[node]}" for node in merge.nodes
        )
        rows.append(
            _row(
                "candidate",
                ident=f"c{idx}",
                label=getattr(merge.macro_gate, "name", "?"),
                lanes=len(merge.nodes),
                weight=priorities[idx].savings_weight if idx in priorities else 0.0,
                conflicts=";".join(f"c{j}" for j in sorted(filtered[idx].conflicts))
                if idx in filtered
                else "",
                committed=int(idx in picked_ids),
                detail=nodes,
            )
        )

    # (c) transversal macros -- committed by their own sweep, no set selection
    consumed = {record.first for record in picked} | {record.second for record in picked}
    transversal = find_transversal_merges(
        layer_index, order_index, patch_lanes, qubit_indices, consumed, True, n_code=N_CODE
    )
    for idx, record in enumerate(transversal):
        rows.append(
            _row(
                "transversal",
                ident=f"t{idx}",
                label=getattr(record.macro_gate, "name", "?"),
                lanes=len(record.nodes),
                committed=1,
                detail=len(record.pre_ops),
            )
        )

    # (d) the rewritten instruction stream, from the pass itself
    rewritten, savings_1q, savings_2q, weighted = preprocess_program_for_merging_with_stats(
        circuit, permutation, n_code=N_CODE, allow_transversal_swap_merges=True
    )
    for idx, instruction in enumerate(rewritten.data):
        name = instruction.operation.name
        qubits = [rewritten.find_bit(q).index for q in instruction.qubits]
        rows.append(
            _row(
                "emit",
                ident=str(idx),
                label=name,
                lanes=len(qubits),
                detail=",".join(str(q) for q in qubits),
                committed=int(name.lower() in MACRO_NAMES),
            )
        )

    rows.append(
        _row(
            "summary",
            label="totals",
            detail=(
                f"in={len(circuit.data)};out={len(rewritten.data)};"
                f"s1q={savings_1q:g};s2q={savings_2q:g};weighted={weighted:g};"
                f"n_code={N_CODE};k={K}"
            ),
        )
    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ plot


def _sub(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    return frame[frame["kind"] == kind].copy()


def _summary(frame: pd.DataFrame) -> Dict[str, str]:
    detail = _sub(frame, "summary")["detail"].iloc[0]
    return dict(part.split("=", 1) for part in str(detail).split(";"))


def _bare(ax) -> None:
    """Schematic panel: no axes, no grid. The style file turns the y grid on by default."""
    ax.set_axis_off()
    ax.grid(False)
    ax.set_aspect("equal")


def _rule(ax, xs, ys, *, color, lw=0.6, style="-", z=1) -> None:
    """
    A schematic line.

    Every argument is explicit on purpose. The house style composes colour, marker and
    linestyle into one zipped prop cycle, which is right for data series and wrong here:
    left to the cycle, guide lines sprout diamonds and conflict edges come out dash-dot.
    """
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style, marker="", zorder=z,
            solid_capstyle="round")


def _place_panel_labels(fig, axes, labels: Sequence[str]) -> None:
    """
    Put `(a) lanes` ... above each panel, all on one baseline.

    Every panel is `aspect="equal"`, so the axes boxes end up different heights and
    `transAxes` y=1 is a different figure height in each -- labels anchored per-axes
    stagger visibly. Resolve the layout first, then place them in figure coordinates
    against the topmost box.
    """
    fig.canvas.draw()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, text in zip(axes, labels):
        fig.text(
            ax.get_position().x0,
            min(top + 0.015, 0.995),
            text,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )


def _cell(ax, x, y, w, h, *, face, edge, radius=0.06, lw=0.6, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def draw_lanes(ax, frame: pd.DataFrame) -> None:
    """(a) The permutation: patch x slot, one program qubit per cell."""
    lanes = _sub(frame, "lane")
    lanes["patch"] = lanes["patch"].astype(int)
    lanes["slot"] = lanes["slot"].astype(int)
    lanes["qubit"] = lanes["qubit"].astype(int)

    cw, ch, gap = 0.82, 0.62, 0.42
    for patch, group in lanes.groupby("patch"):
        x = patch * (cw + gap)
        _cell(
            ax,
            x - 0.08,
            -0.08,
            cw + 0.16,
            K * ch + 0.16,
            face="none",
            edge=DARK,
            radius=0.1,
            lw=0.8,
            z=1,
        )
        for _, row in group.iterrows():
            y = (K - 1 - row["slot"]) * ch
            _cell(ax, x, y, cw, ch * 0.86, face="white", edge=LIGHT)
            ax.text(
                x + cw / 2,
                y + ch * 0.43,
                f"$q_{{{row['qubit']}}}$",
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )
        ax.text(
            x + cw / 2,
            K * ch + 0.2,
            f"patch {patch}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )

    ax.text(
        -0.34,
        K * ch / 2,
        "slot",
        ha="center",
        va="center",
        fontsize=8,
        rotation=90,
        color=MID,
    )
    ax.set_xlim(-0.55, 2 * cw + gap + 0.15)
    ax.set_ylim(-0.25, K * ch + 0.72)
    _bare(ax)


def draw_propose(ax, frame: pd.DataFrame) -> None:
    """(b) One patch's gates, with the cells each committed family absorbs marked."""
    gates = _sub(frame, "gate")
    gates = gates[gates["patch"].astype(str) == "0"]
    gates["qubit"] = gates["qubit"].astype(int)
    gates["layer"] = gates["layer"].astype(int)

    cands = _sub(frame, "candidate")
    cands["committed"] = cands["committed"].astype(int)
    shown = {row["label"]: row for _, row in cands[cands["committed"] == 1].iterrows()}

    cw, ch, gw = 0.70, 0.60, 0.50
    n_layers = int(gates["layer"].max()) + 1
    right = (n_layers - 1) * cw + gw

    def cell_x(layer: int) -> float:
        return layer * cw

    def cell_y(lane: int) -> float:
        return (K - 1 - lane) * ch

    for lane in range(K):
        y = cell_y(lane) + ch * 0.42
        _rule(ax, [-0.14, right + 0.14], [y, y], color=LIGHT, z=0)
        ax.text(-0.24, y, f"$q_{{{lane}}}$", ha="right", va="center", fontsize=8)

    for _, row in gates.iterrows():
        x, y = cell_x(row["layer"]), cell_y(row["qubit"])
        _cell(ax, x, y, gw, ch * 0.84, face="white", edge=MID)
        ax.text(x + gw / 2, y + ch * 0.42, row["label"].upper(), ha="center", va="center",
                fontsize=8, color=DARK, zorder=3)

    # Mark the individual cells each family absorbs rather than a bounding box: `xx` takes
    # lanes 0 and 2 and skips the H on lane 1 between them, so a box over that span --
    # or a spine drawn straight through it -- would claim a gate the macro never touches.
    # The spine therefore only joins lanes that are actually adjacent in the group.
    # Solid/dashed as well as colour, so the two families still separate in greyscale.
    top = K * ch
    for name, colour, style in (("hh", ACCENT, "-"), ("xx", DARK, (0, (2.2, 1.4)))):
        row = shown.get(name)
        if row is None:
            continue
        spans = [tuple(int(v) for v in part.split("@")) for part in str(row["detail"]).split(";")]
        lanes_used = sorted(lane for lane, _ in spans)
        layer = spans[0][1]
        x = cell_x(layer)
        for lane, lyr in spans:
            ax.add_patch(
                Rectangle(
                    (cell_x(lyr) - 0.06, cell_y(lane) - 0.06),
                    gw + 0.12,
                    ch * 0.84 + 0.12,
                    facecolor="none",
                    edgecolor=colour,
                    linewidth=1.1,
                    linestyle=style,
                    zorder=4,
                )
            )
        for lower, upper in zip(lanes_used, lanes_used[1:]):
            if upper - lower != 1:
                continue
            _rule(
                ax,
                [x + gw / 2, x + gw / 2],
                [cell_y(upper) + ch * 0.84 + 0.06, cell_y(lower) - 0.06],
                color=colour,
                lw=1.1,
                style=style,
                z=4,
            )
        ax.text(
            x + gw / 2,
            top + 0.06,
            f"{name}\n{len(spans)}/{K}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colour,
            linespacing=1.1,
        )

    ax.text(-0.24, -0.30, f"+{len(cands) - len(shown)} rival groups", ha="left", va="top",
            fontsize=8, color=MID)
    ax.set_xlim(-0.72, right + 0.30)
    ax.set_ylim(-0.72, top + 0.80)
    _bare(ax)


def draw_choose(ax, frame: pd.DataFrame) -> None:
    """(c) Conflict graph; the committed colour class is filled."""
    cands = _sub(frame, "candidate")
    cands["committed"] = cands["committed"].astype(int)
    cands["weight"] = cands["weight"].astype(float)

    # Deterministic layout: the whole-patch groups conflict with each other pairwise, so
    # they go on a square and read as the clique they are; the per-lane group sits apart
    # because it conflicts with only two of them.
    whole = [row for _, row in cands.iterrows() if row["label"] == "hh"]
    perlane = [row for _, row in cands.iterrows() if row["label"] != "hh"]
    pos: Dict[str, Tuple[float, float]] = {}
    square = [(-0.55, 0.62), (0.55, 0.62), (-0.55, -0.62), (0.55, -0.62)]
    for row, (x, y) in zip(whole, square):
        pos[row["ident"]] = (x, y)
    for offset, row in enumerate(perlane):
        pos[row["ident"]] = (1.95, 0.0 - 1.24 * offset)

    for _, row in cands.iterrows():
        a = pos.get(row["ident"])
        if a is None:
            continue
        for other in str(row["conflicts"]).split(";"):
            b = pos.get(other)
            if b is None or other <= row["ident"]:
                continue
            _rule(ax, [a[0], b[0]], [a[1], b[1]], color=LIGHT, lw=0.6, z=1)

    bw, bh = 0.86, 0.62
    for _, row in cands.iterrows():
        x, y = pos[row["ident"]]
        keep = bool(row["committed"])
        _cell(
            ax,
            x - bw / 2,
            y - bh / 2,
            bw,
            bh,
            face=ACCENT if keep else "white",
            edge=ACCENT if keep else LIGHT,
            radius=0.2,
            lw=0.9,
            z=3,
        )
        ax.text(x, y + 0.11, row["label"], ha="center", va="center", fontsize=8,
                color="white" if keep else DARK, zorder=4)
        ax.text(x, y - 0.15, f"$w{{=}}${row['weight']:g}", ha="center", va="center",
                fontsize=8, color="white" if keep else MID, zorder=4)

    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=ACCENT, edgecolor=ACCENT, label="committed"),
            Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=LIGHT, label="not taken"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.46, 0.04),
        ncol=2,
        handlelength=1.0,
        handleheight=0.7,
        handletextpad=0.35,
        columnspacing=0.8,
        fontsize=8,
    )
    ax.set_xlim(-1.15, 2.55)
    ax.set_ylim(-1.62, 1.30)
    _bare(ax)


def _operands(detail: str) -> str:
    """Compact operand label: a patch pair, a contiguous lane run, or a short list."""
    qubits = [int(part) for part in str(detail).split(",") if part != ""]
    if len(qubits) == 2 * K:
        return r"p0$\rightarrow$p1"
    if len(qubits) >= 3 and qubits == list(range(qubits[0], qubits[-1] + 1)):
        return f"$q_{{{qubits[0]}}}$–$q_{{{qubits[-1]}}}$"
    return ",".join(f"$q_{{{q}}}$" for q in qubits)


def draw_emit(ax, frame: pd.DataFrame) -> None:
    """(d) The rewritten instruction stream, macros filled."""
    emitted = _sub(frame, "emit")
    emitted["committed"] = emitted["committed"].astype(int)
    totals = _summary(frame)

    x = 0.0
    height = 0.62
    for _, row in emitted.iterrows():
        macro = bool(row["committed"])
        label = row["label"]
        width = 0.56 + 0.15 * max(0, len(label) - 2)
        _cell(
            ax,
            x,
            0.0,
            width,
            height,
            face=ACCENT if macro else "white",
            edge=ACCENT if macro else LIGHT,
            radius=0.09,
            lw=0.9,
        )
        ax.text(x + width / 2, height / 2, label, ha="center", va="center", fontsize=8,
                color="white" if macro else DARK, zorder=3)
        # Operands only under the macros. Under the leftover singles they are three more
        # numbers competing for the same 8pt of vertical space and say nothing.
        if macro:
            ax.text(x + width / 2, -0.10, _operands(row["detail"]), ha="center", va="top",
                    fontsize=8, color=MID)
        x += width + 0.12

    span = x - 0.12
    ax.annotate(
        "",
        xy=(span, height + 0.34),
        xytext=(0.0, height + 0.34),
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=MID, shrinkA=0, shrinkB=0),
    )
    ax.text(0.0, height + 0.46, f"{totals['in']} instructions $\\rightarrow$ {totals['out']}",
            ha="left", va="bottom", fontsize=8, color=DARK)
    ax.text(span, -0.44, f"saves {totals['s1q']} 1q, {totals['s2q']} 2q physical gates",
            ha="right", va="top", fontsize=8, color=DARK)
    ax.set_xlim(-0.10, span + 0.10)
    ax.set_ylim(-0.78, height + 0.90)
    _bare(ax)


def plot(frame: pd.DataFrame, script: Path, override: Path | None) -> List[Path]:
    use_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(FULL_WIDTH, 2.15))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.66, 1.14, 1.26, 1.64], wspace=0.02)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]

    draw_lanes(axes[0], frame)
    draw_propose(axes[1], frame)
    draw_choose(axes[2], frame)
    draw_emit(axes[3], frame)

    _place_panel_labels(
        fig, axes, ("(a) lanes", "(b) propose", "(c) choose", "(d) emit")
    )
    return save_figure(fig, NAME, script, override)


# ------------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3].strip())
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="redraw from a cached CSV instead of re-running the detectors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override where the SVG and PDF land",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="also copy the CSV into the project's results/data/ (rarely worth it here:"
        " the computation is one 8-qubit circuit)",
    )
    args = parser.parse_args(argv)

    script = Path(__file__).resolve()
    cache = script.parent / f"{NAME}.csv"

    if args.from_csv is not None:
        frame = pd.read_csv(args.from_csv, keep_default_na=False)
    else:
        frame = compute()
        frame.to_csv(cache, index=False)
        if args.persist:
            frame.to_csv(data_dir(script) / f"{NAME}.csv", index=False)

    paths = plot(frame, script, args.output_dir)
    totals = _summary(frame)
    print(f"n_code={totals['n_code']} k={totals['k']}: "
          f"{totals['in']} -> {totals['out']} instructions, "
          f"{totals['s1q']} 1q + {totals['s2q']} 2q gates saved "
          f"({totals['weighted']} weighted)")
    print(f"cache: {cache}")
    for path in paths:
        print(f"wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
