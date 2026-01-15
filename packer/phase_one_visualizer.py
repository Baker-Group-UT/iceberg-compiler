from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from .phase_one_graph import (
    GraphHeuristics,
    EdgeItem,
    build_graph,
    DEFAULT_WORK_TABLE,
    DEFAULT_WORK_WEIGHTS,
)

EdgeGroups = Dict[str, List[EdgeItem]]


def summarize_graph(
    graph: GraphHeuristics,
    *,
    top_edges: Optional[int] = 10,
    precision: int = 4,
) -> str:
    """
    Build a human-readable summary of the phase-one heuristic graph.

    Args:
        graph: Graph produced by ``build_graph``.
        top_edges: Trim the per-kind edge listings to this many entries (None for all).
        precision: Number of decimals to display for weights.
    """
    lines: List[str] = []
    lines.append("GraphHeuristics summary")
    lines.append(
        f"- qubits: {len(graph.nodes)} | edges: {len(graph.edges)} "
        f"| patch_capacity={graph.patch_capacity} | max_patches={graph.max_patches}"
    )
    lines.append(
        f"- solo_budget_range=[{graph.solo_budget_min}, {graph.solo_budget_max}]"
    )
    lines.append("")
    lines.append("Nodes:")
    for node in sorted(graph.nodes.values(), key=lambda n: n.qubit_index):
        lines.append(
            f"  q{node.qubit_index}: base={node.base_cost:.{precision}f}, "
            f"solo={node.solo_cost:.{precision}f}"
        )

    grouped = _group_edges(graph.edges)
    lines.append("")
    for kind in ("solo", "same", "diff"):
        edges = grouped.get(kind, [])
        if not edges:
            continue
        lines.append(f"{kind} edges ({len(edges)} total):")
        for edge in _trim_edges(edges, top_edges):
            label = _format_edge(edge, precision=precision)
            lines.append(f"  {label}")
        if top_edges is not None and len(edges) > top_edges:
            lines.append(f"  ... {len(edges) - top_edges} more")
        lines.append("")

    return "\n".join(lines).rstrip()


def graph_to_dot(
    graph: GraphHeuristics,
    *,
    precision: int = 3,
    include_solo_edges: bool = False,
) -> str:
    """
    Convert the heuristic graph into a GraphViz DOT description.

    Args:
        graph: Graph produced by ``build_graph``.
        precision: Number of decimals to display for weights.
        include_solo_edges: When True, emit self-loop edges for solo entries.
    """
    lines = ["graph PhaseOneHeuristics {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=circle, fontsize=10];")

    for node in graph.nodes.values():
        label = (
            f"q{node.qubit_index}\\n"
            f"base={node.base_cost:.{precision}f}\\n"
            f"solo={node.solo_cost:.{precision}f}"
        )
        lines.append(f'  q{node.qubit_index} [label="{label}"];')

    for edge in graph.edges:
        if edge.kind == "solo" and not include_solo_edges:
            continue
        if edge.j is None or edge.i == edge.j:
            loop = edge.i
            label = f"{edge.weight:.{precision}f} ({edge.kind})"
            lines.append(
                f'  q{loop} -- q{loop} [label="{label}", color="#888888", style="dashed"];'
            )
            continue
        color = "#00897b" if edge.kind == "same" else "#f4511e"
        label = f"{edge.weight:.{precision}f}"
        style = "solid" if edge.kind == "same" else "dotted"
        lines.append(
            f'  q{edge.i} -- q{edge.j} '
            f'[label="{label}", color="{color}", style="{style}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def draw_graph_matplotlib(
    graph: GraphHeuristics,
    *,
    include_solo_edges: bool = False,
    figsize: Tuple[int, int] = (7, 5),
    precision: int = 3,
    show_legend: bool = True,
    ax=None,
):
    """
    Render the heuristic graph using Matplotlib primitives.

    Args:
        graph: Graph produced by ``build_graph``.
        include_solo_edges: Whether to draw dashed self-loop edges.
        figsize: Size of the figure if ``ax`` is not provided.
        precision: Number of decimals to display for weights.
        ax: Optional Matplotlib Axes to draw onto.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    qubits = sorted(graph.nodes.keys())
    positions: Dict[int, Tuple[float, float]] = {}
    radius = 0.4
    center = (0.5, 0.5)
    count = max(1, len(qubits))
    for idx, q in enumerate(qubits):
        angle = 2 * math.pi * idx / count
        positions[q] = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )

    for edge in graph.edges:
        if edge.kind == "solo" and not include_solo_edges:
            continue
        x0, y0 = positions[edge.i]
        if edge.j is None or edge.i == edge.j:
            circle = plt.Circle((x0, y0), 0.045, color="gray", fill=False, linestyle="dashed")
            ax.add_patch(circle)
            ax.text(
                x0,
                y0 + 0.06,
                f"{edge.weight:.{precision}f}",
                fontsize=9,
                color="gray",
                ha="center",
            )
            continue
        x1, y1 = positions[edge.j]
        color = "#00897b" if edge.kind == "same" else "#f4511e"
        linestyle = "-" if edge.kind == "same" else "--"
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy) or 1.0
        norm = (-dy / dist, dx / dist)
        offset_mag = 0.05 if edge.kind == "same" else -0.05
        offset = (norm[0] * offset_mag, norm[1] * offset_mag)
        start = (x0 + offset[0], y0 + offset[1])
        end = (x1 + offset[0], y1 + offset[1])
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
            f"{edge.weight:.{precision}f}",
            fontsize=10,
            color=color,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1),
        )

    node_radius = 0.065
    for node_id, node in graph.nodes.items():
        x, y = positions[node_id]
        circle = plt.Circle(
            (x, y),
            node_radius,
            facecolor="#263238",
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            f"q{node_id}\n{node.solo_cost:.{precision}f}",
            color="white",
            fontsize=12,
            ha="center",
            va="center",
            weight="bold",
            zorder=4,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Phase-One Heuristic Graph", fontsize=12, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    if show_legend:
        handles = [
            Patch(facecolor="#263238", edgecolor="white", label="Node (solo weight)"),
            Line2D(
                [0], [0], color="#00897b", linewidth=2, label="same edge", linestyle="-"
            ),
            Line2D(
                [0], [0], color="#f4511e", linewidth=2, label="diff edge", linestyle="--"
            ),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9)
    if ax.figure:
        ax.figure.tight_layout()


def _group_edges(edges: Iterable[EdgeItem]) -> EdgeGroups:
    grouped: EdgeGroups = {"solo": [], "same": [], "diff": []}
    for edge in edges:
        grouped.setdefault(edge.kind, []).append(edge)
    for items in grouped.values():
        items.sort(key=lambda e: e.weight)
    return grouped


def _trim_edges(edges: Sequence[EdgeItem], top_edges: Optional[int]) -> Sequence[EdgeItem]:
    if top_edges is None or top_edges >= len(edges):
        return edges
    return edges[:top_edges]


def _format_edge(edge: EdgeItem, *, precision: int) -> str:
    if edge.kind == "solo" or edge.j is None:
        return f"solo(q{edge.i}) -> weight={edge.weight:.{precision}f}"
    return f"{edge.kind}(q{edge.i}, q{edge.j}) -> weight={edge.weight:.{precision}f}"


__all__ = [
    "summarize_graph",
    "graph_to_dot",
    "draw_graph_matplotlib",
]


def _generate_random_circuit(
    num_qubits: int = 4,
    num_gates: int = 10,
    gate_set: Sequence[str] = ("x", "z", "h", "cx"),
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_gates):
        gate = random.choice(gate_set)
        if gate == "cx":
            control, target = random.sample(range(num_qubits), 2)
            qc.cx(control, target)
        elif gate == "x":
            qc.x(random.randrange(num_qubits))
        elif gate == "z":
            qc.z(random.randrange(num_qubits))
        else:
            qc.h(random.randrange(num_qubits))
    return qc


def _smoke_test() -> None:
    random.seed(42)
    # random.seed(45)
    # qc = _generate_random_circuit(num_qubits=6, num_gates=20)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(3)
    qc.x(1)
    qc.cx(0,1)
    qc.cx(3,2)
    qc.z(2)

    graph = build_graph(
        qc,
        gate_work_table=DEFAULT_WORK_TABLE,
        work_weights=DEFAULT_WORK_WEIGHTS,
        patch_capacity=2,
        max_patches=8,
    )
    print("=== Phase-One Graph Summary ===")
    print(qc)
    print(summarize_graph(graph, top_edges=5))
    print("\n=== DOT Representation ===")
    print(graph_to_dot(graph))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - visualization optional
        print(f"\n(Matplotlib unavailable: {exc})")
    else:  # pragma: no cover - visualization optional
        draw_graph_matplotlib(graph)
        plt.show()


if __name__ == "__main__":
    _smoke_test()
