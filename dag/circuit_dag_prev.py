# circuit_dag.py – v1.7
# Nodes are gate operations in this version, edges show their shared progression.
"""Qiskit DAG visualiser – qubit‑aligned lanes & better framing."""
from __future__ import annotations
import itertools
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import networkx as nx
from qiskit import QuantumCircuit, QiskitError

__all__ = [
    "CircuitDAG",
    "qubit_interaction_graph",
    "draw_interaction_graph",
    "draw_detailed_interaction_graph",
]

# ---------------------------------------------------------------------------
# Helper: positions with optional qubit‑aligned lanes
# ---------------------------------------------------------------------------

def _layered_pos(
    G: nx.DiGraph,
    *,
    orientation: str = "LR",
    horiz: float = 2.0,
    vert: float = 1.3,
    qubit_lanes: bool = False,
):
    """Return (pos, layers) with optional *qubit lanes* alignment."""
    try:
        layers = list(nx.topological_generations(G))
    except nx.NetworkXUnfeasible:
        layers = [list(G.nodes)]

    pos: Dict[int, Tuple[float, float]] = {}
    for depth, layer in enumerate(layers):
        ordered = sorted(layer)
        for i, node in enumerate(ordered):
            if qubit_lanes:
                q = min(G.nodes[node]["qubits"])  # primary qubit lane
                x = depth * horiz
                y = -q * vert
            else:
                if orientation == "LR":
                    x = depth * horiz
                    y = -(i - (len(ordered) - 1) / 2) * vert
                else:  # TB
                    x = (i - (len(ordered) - 1) / 2) * horiz
                    y = -depth * vert
            pos[node] = (x, y)
    return pos, layers

# ---------------------------------------------------------------------------
class CircuitDAG:
    _BASE_STYLE = {
        "node_size": 680,
        "font_size": 8,
        "arrows": True,
        "edgecolors": "#333",
        "linewidths": 1.2,
        "node_shape": "o",
    }

    def __init__(self, qc: QuantumCircuit):
        if qc is None:
            raise QiskitError("Circuit cannot be None.")
        self.qc = qc
        self.graph: nx.DiGraph = nx.DiGraph()
        self._build()

    # ........................................ build
    def _qidx(self, q):
        return self.qc.qubits.index(q)

    def _primary_qubit(self, inst_qubits):
        return self._qidx(inst_qubits[-1]) if inst_qubits else -1

    def _build(self):
        last: Dict[int, int] = {}
        for idx, inst in enumerate(self.qc.data):
            qubits = [self._qidx(q) for q in inst.qubits]
            gate = inst.operation.name
            q_txt = ",".join(map(str, qubits[:3])) + (",…" if len(qubits) > 3 else "")
            label = f"{idx}:{gate}\nq[{q_txt}]"
            self.graph.add_node(idx, gate=gate, qubits=qubits, label=label)
            for q in qubits or [-1]:
                if q in last:
                    self.graph.add_edge(last[q], idx)
                last[q] = idx

    # ........................................ helpers
    def _colormap(self):
        unique = sorted({d["gate"] for _n, d in self.graph.nodes(data=True)})
        cm = plt.get_cmap("tab20")
        return {g: mcolors.to_hex(cm(i % 20)) for i, g in enumerate(unique)}

    # ........................................ compute positions
    def _compute_pos(self, layout, prog, orient, h, v, lanes):
        if layout == "graphviz":
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog=prog, args="-Grankdir=LR")
            return pos, None
        return _layered_pos(self.graph, orientation=orient, horiz=h, vert=v, qubit_lanes=lanes)

    # ........................................ draw
    def draw(
        self,
        *,
        layout: str = "layered",
        orient: str = "LR",
        figsize: Tuple[int, int] = (14, 6),
        horiz_spacing: float = 2.0,
        vert_spacing: float = 1.3,
        zoom: float = 0.85,       # more padding so depth fits
        qubit_lanes: bool = True, # align lanes
        **kw,
    ):
        pos, layers = self._compute_pos(layout, "dot", orient, horiz_spacing, vert_spacing, qubit_lanes)

        colours = self._colormap()
        node_cols = [colours[self.graph.nodes[n]["gate"]] for n in self.graph.nodes]
        style = {**self._BASE_STYLE, **kw, "node_color": node_cols}

        plt.figure(figsize=figsize)
        nx.draw(self.graph, pos, with_labels=True, labels=nx.get_node_attributes(self.graph, "label"), **style)

        # lane/grid lines for qubits
        if qubit_lanes:
            qs = sorted({q for data in self.graph.nodes.values() for q in data["qubits"]})
            xs = [x for x, _ in pos.values()]
            for q in qs:
                y = -q * vert_spacing
                plt.hlines(y, min(xs)-horiz_spacing, max(xs)+horiz_spacing, colors="#e5e5e5", linestyles="dotted", linewidth=0.6)

        # depth markers top
        xs = sorted({x for x, _ in pos.values()})
        ys = [y for _x, y in pos.values()]
        ymax = max(ys) + vert_spacing
        for depth, x in enumerate(xs):
            plt.text(x, ymax + 0.4, f"depth {depth}", fontsize=7, ha="center", va="bottom", color="#555")

        # legend bottom‑left
        patches = [Patch(color=c, label=g) for g, c in colours.items()]
        plt.legend(handles=patches, fontsize=7, frameon=True, loc="lower left", bbox_to_anchor=(0.02, 0.02))

        # zoom framing
        x_vals = [x for x, _ in pos.values()]
        y_vals = [y for _, y in pos.values()] + [ymax]
        xpad = (max(x_vals) - min(x_vals)) * (1 - zoom)
        ypad = (max(y_vals) - min(y_vals)) * (1 - zoom)
        plt.xlim(min(x_vals) - xpad, max(x_vals) + xpad)
        plt.ylim(min(y_vals) - ypad, max(y_vals) + ypad)

        plt.title("Circuit Dependency DAG – qubit lanes")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ........................................
    def get_graph(self):
        return self.graph

# ---------------------------------------------------------------------------
# 2) Interaction & detailed graph helpers (unchanged from v1.5)
# ---------------------------------------------------------------------------

def qubit_interaction_graph(qc: QuantumCircuit, weight: bool = True) -> nx.Graph:
    G = nx.Graph()
    for idx in range(len(qc.qubits)):
        G.add_node(idx, label=f"q{idx}")
    for inst in qc.data:
        if len(inst.qubits) >= 2:
            idxs = [qc.qubits.index(q) for q in inst.qubits]
            for u, v in itertools.combinations(idxs, 2):
                if G.has_edge(u, v):
                    if weight:
                        G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1 if weight else None)
    return G


def draw_interaction_graph(qc: QuantumCircuit, figsize: Tuple[int, int] = (6, 6), **kw):
    G = qubit_interaction_graph(qc)
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="neato")
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    widths = [G[u][v].get("weight", 1) for u, v in G.edges]
    plt.figure(figsize=figsize)
    nx.draw(G, pos, labels=nx.get_node_attributes(G, "label"), width=widths, node_color="#ffdead", node_size=650, **kw)
    plt.title("Inter-Qubit Interaction Graph (edge width = #gates)")
    plt.axis("off")
    plt.show()

# ---------------------------------------------------------------------------
# 3) draw_detailed_interaction_graph kept unchanged
# ---------------------------------------------------------------------------


def _multi_edges_with_order(qc: QuantumCircuit) -> nx.MultiGraph:
    """Create MultiGraph; each edge stores *order* for pair (u,v)."""
    G = nx.MultiGraph()
    for idx, _ in enumerate(qc.qubits):
        G.add_node(idx, label=f"q{idx}")
    pair_counter: Dict[Tuple[int, int], int] = {}
    for step, inst in enumerate(qc.data):
        if len(inst.qubits) < 2:
            continue
        idxs = [qc.qubits.index(q) for q in inst.qubits]
        for u, v in itertools.combinations(idxs, 2):
            pair = tuple(sorted((u, v)))
            order = pair_counter.get(pair, 0)
            label = f"[{order}] {inst.operation.name.upper()}({u},{v})"
            G.add_edge(u, v, label=label, order=order)
            pair_counter[pair] = order + 1
    return G

def draw_detailed_interaction_graph(
    qc: QuantumCircuit,
    figsize: Tuple[int, int] = (8, 8),
    *,
    cmap: str = "viridis",
    base_rad: float = 0.15,
    label_offset: float = 0.08,
    **edge_kw,
):
    """Detailed multi-edge graph – label now hugs its arc accurately.

    The mid‑curve label is computed from the *exact* quadratic Bézier that
    :pydata:`matplotlib` draws when using connectionstyle ``"arc3,rad=r"``.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to visualise.
    figsize : tuple
        Figure size in inches.
    cmap : str
        Matplotlib colormap name for order‑based colour.
    base_rad : float
        Curvature for the first edge (next edges scale by order+1).
    label_offset : float
        Extra outward shift (in data units) so label sits clearly above the arc.
    edge_kw : dict
        Extra kwargs forwarded to ``networkx.draw_networkx_edges``.  ``linewidths``
        is mapped to ``width`` automatically.
    """
    G = _multi_edges_with_order(qc)
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="neato")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # ``linewidths`` compatibility shim
    if "linewidths" in edge_kw:
        edge_kw["width"] = edge_kw.pop("linewidths")

    # colour mapping by order
    max_order = max((d["order"] for *_, d in G.edges(data=True)), default=0)
    norm = mcolors.Normalize(vmin=0, vmax=max_order or 1)
    cmap_func = plt.get_cmap(cmap)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color="#c0ffee", node_size=1400)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"))

    for (u, v, k, data) in G.edges(keys=True, data=True):
        order = data["order"]
        # alternate bend direction to separate parallel edges
        rad = base_rad * (order + 1) * (1 if k % 2 == 0 else -1)
        color = cmap_func(norm(order))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            edge_color=[color],
            **edge_kw,
        )

        # ------ Accurate label placement on quadratic Bézier ------
        x0, y0 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x0, y2 - y0
        xm, ym = (x0 + x2) / 2, (y0 + y2) / 2
        # Control point = midpoint + rad * perpendicular
        cx = xm + rad * dy
        cy = ym - rad * dx
        # Quadratic Bézier at t=0.5: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
        lx = 0.25 * x0 + 0.5 * cx + 0.25 * x2
        ly = 0.25 * y0 + 0.5 * cy + 0.25 * y2
        # Outward shift along perpendicular for readability
        length = (dx**2 + dy**2) ** 0.5 or 1e-9
        px, py = (-dy / length, dx / length)  # unit perpendicular
        lx += px * label_offset * (1 if rad >= 0 else -1)
        ly += py * label_offset * (1 if rad >= 0 else -1)

        plt.text(
            lx,
            ly,
            data["label"],
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )

    plt.title("Detailed Interaction Graph (order‑coded)")
    plt.axis("off")
    plt.show()