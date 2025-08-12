"""
# circuit_dag.py – v2.3  (timeline graph + laneless clustering + 3-D tower view)

Timeline-style graph for Qiskit circuits (nodes = qubit×timeslice)
=================================================================
* **Nodes**  = (qubit_index, t)
* **Edges**  = gate operations (labelled & coloured by gate)

New in v2.3
-----------
* **orient="tower"** – 3-D "vertical tower" view
    * Qubits are arranged on a ring in the *x-y* plane.
    * Time advances **upwards** along *z* so every qubit forms its own pillar.
    * Multi-qubit gate edges span between pillars at the relevant height.
* Existing improvements from v2.2: tight same-qubit clustering in laneless.

Public API
~~~~~~~~~~
>>> g = build_qubit_graph(qc)
>>> draw_qubit_graph(g, orient="tower")          # 3-D view
>>> draw_qubit_graph(g, orient="laneless")       # 2-D spring view
>>> draw_qubit_graph(g, orient="horiz")          # classic lanes
"""
from __future__ import annotations

# stdlib / typing
from typing import Dict, Tuple, List

# third-party
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – activates the 3-D backend
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit

__all__ = [
    "build_qubit_graph",
    "draw_qubit_graph",
]

# ---------------------------------------------------------------------------
# 1) Build qubit-timeline graph (unchanged from v2.2)
# ---------------------------------------------------------------------------

def build_qubit_graph(qc: QuantumCircuit, *, decompose: bool = True) -> nx.MultiDiGraph:
    """Return a qubit-timeline graph whose edge *weight* encodes interaction
    strength and is later interpreted as spring strength in force-directed
    layouts.
    """
    if decompose:
        qc = qc.decompose()
    idx_map = {qb: i for i, qb in enumerate(qc.qubits)}
    G = nx.MultiDiGraph()

    head = {i: (i, 0) for i in range(qc.num_qubits)}
    G.add_nodes_from(head.values())

    t = 1
    for inst in qc.data:
        qidxs = [idx_map[q] for q in inst.qubits]

        # advance head for each participating qubit -------------------------
        for q in qidxs:
            new = (q, t)
            G.add_node(new)
            w = 0.6 if len(qidxs) == 1 else 0.2  # vertical idle vs busy
            G.add_edge(head[q], new, gate=inst.operation.name if len(qidxs) == 1 else "i", w=w)
            head[q] = new

        # multi-qubit gate edges -------------------------------------------
        if len(qidxs) == 2:
            ctl, tgt = qidxs
            w = 1.5
            if inst.operation.name == "swap":
                G.add_edge(head[ctl], head[tgt], gate="swap", w=w)
                G.add_edge(head[tgt], head[ctl], gate="swap", w=w)
            else:
                G.add_edge(head[ctl], head[tgt], gate=inst.operation.name, w=w)
        elif len(qidxs) == 3:
            c1, c2, tgt = qidxs
            w = 2.0
            G.add_edge(head[c1], head[c2], gate=inst.operation.name, w=w)
            G.add_edge(head[c1], head[tgt], gate=inst.operation.name, w=w)
        t += 1
    return G

# ---------------------------------------------------------------------------
# 2) Helper: add strong springs along each qubit’s timeline (laneless only)
# ---------------------------------------------------------------------------

def _reinforce_qubit_chains(
    G: nx.MultiDiGraph,
    *,
    strength: float = 5.0,
) -> nx.MultiDiGraph:
    """Return *H* = G ∪ extra, invisible edges that pull consecutive nodes
    belonging to **the same logical qubit** closer together in spring layout.
    """
    H = G.copy()
    chains: Dict[int, List[Tuple[int, int]]] = {}
    for q, t in H.nodes:
        chains.setdefault(q, []).append((q, t))
    for q, nodes in chains.items():
        nodes.sort(key=lambda n: n[1])  # by time slice
        for u, v in zip(nodes, nodes[1:]):
            H.add_edge(u, v, gate="__cluster__", w=strength)
    return H

# ---------------------------------------------------------------------------
# 3) Colour helpers (unchanged)
# ---------------------------------------------------------------------------
_GATE_COLORS: Dict[str, str] = {}
_Q_COLORS: Dict[int, str] = {}


def _gate_colour(g: str) -> str:
    if g not in _GATE_COLORS:
        cm = plt.get_cmap("tab20")
        _GATE_COLORS[g] = mcolors.to_hex(cm(len(_GATE_COLORS) % 20))
    return _GATE_COLORS[g]


def _qubit_colour(q: int) -> str:
    if q not in _Q_COLORS:
        cm = plt.get_cmap("tab10")
        _Q_COLORS[q] = mcolors.to_hex(cm(q % 10))
    return _Q_COLORS[q]

# ---------------------------------------------------------------------------
# 4) Main visualiser – supports 'horiz', 'laneless', and new 'tower'
# ---------------------------------------------------------------------------

def draw_qubit_graph(
    G: nx.MultiDiGraph,
    *,
    figsize: Tuple[int, int] = (14, 6),
    node_size: int = 90,
    orient: str = "horiz",          # 'horiz' | 'laneless' | 'tower'
    spring_k: float | None = None,
    seed: int = 42,
    tower_radius: float = 3.0,
    tower_zscale: float = 1.0,
    arrow_scale: float = 1.4,       # <— overall arrow-head size multiplier
    tower_head_ratio: float = 0.1, # <— fraction of arrow length used for head
):
    """Visualise *G*.

    orient="tower"  – 3-D ring-and-pillar view (requires Matplotlib ≥ 3.1).
    orient="laneless" – 2-D spring layout with tight same-qubit clustering.
    orient="horiz"   – classic timeline lanes.
    """

    if orient not in {"horiz", "laneless", "tower"}:
        raise ValueError("orient must be 'horiz', 'laneless', or 'tower'")

    # ---------------------------------------------------------------------
    # 1. Compute node positions
    # ---------------------------------------------------------------------
    if orient == "horiz":
        pos = {(q, t): (t, -q) for (q, t) in G.nodes}

    elif orient == "laneless":
        pos0 = {(q, t): (q, -t) for (q, t) in G.nodes}  # seed positions
        H = _reinforce_qubit_chains(G, strength=5.0)
        if spring_k is None:
            spring_k = 0.35
        pos = nx.spring_layout(H, weight="w", seed=seed, k=spring_k, pos=pos0)

    else:  # orient == 'tower'
        n_qubits = 1 + max(q for q, _ in G.nodes)
        R = tower_radius
        pos = {}
        for (q, t) in G.nodes:
            theta = 2 * np.pi * q / n_qubits
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            z = t * tower_zscale
            pos[(q, t)] = (x, y, z)

    # ---------------------------------------------------------------------
    # 2. Create figure / axes (3-D for tower, 2-D otherwise)
    # ---------------------------------------------------------------------
    if orient == "tower":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # ---------------------------------------------------------------------
    # 3. Draw edges
    # ---------------------------------------------------------------------
    for (u, v, data) in G.edges(data=True):
        gate = data["gate"]
        if gate == "__cluster__":
            continue

        color = _gate_colour(gate)

        if orient == "tower":
            # --- 3-D arrows ------------------------------------------------
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)
            vec_len = np.linalg.norm([dx, dy, dz])
            if vec_len == 0:      # self-edge (shouldn’t happen, but be safe)
                continue
            # quiver needs a *direction* (unit vec) and a *length*
            ux, uy, uz = dx / vec_len, dy / vec_len, dz / vec_len
            ax.quiver(
                x0, y0, z0,
                ux, uy, uz,
                length=vec_len,
                arrow_length_ratio=tower_head_ratio * arrow_scale,
                linewidth=1.1,
                color=color,
                alpha=0.9,
            )
            if gate not in ("i",):
                ax.text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    (z0 + z1) / 2,
                    gate,
                    color=color,
                    size=7,
                    ha="center",
                    va="center",
                )
        else:
            # --- 2-D arrows ----------------------------------------------
            x0, y0, _ = (*pos[u], 0.0)
            x1, y1, _ = (*pos[v], 0.0)
            dx, dy = x1 - x0, y1 - y0
            hw = 0.04 * arrow_scale   # head width
            hl = 0.07 * arrow_scale   # head length
            ax.arrow(
                x0, y0, dx, dy,
                length_includes_head=True,
                head_width=hw,
                head_length=hl,
                fc=color,
                ec=color,
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )
            if gate not in ("i",):
                ax.text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    gate,
                    fontsize=7,
                    ha="center",
                    va="center",
                    color=color,
                )

    # ---------------------------------------------------------------------
    # 4. Draw nodes
    # ---------------------------------------------------------------------
    node_cols = [_qubit_colour(q) for q, _ in G.nodes]
    if orient == "tower":
        xs, ys, zs = zip(*[pos[n] for n in G.nodes])
        ax.scatter(xs, ys, zs, s=node_size, c=node_cols, depthshade=False,
                   edgecolors="#333", linewidths=0.6)
    else:
        xs, ys = zip(*[pos[n] for n in G.nodes])
        ax.scatter(xs, ys, s=node_size, c=node_cols, zorder=3,
                   edgecolors="#333", linewidths=0.6)

    # ---------------------------------------------------------------------
    # 5. Legend & cosmetics
    # ---------------------------------------------------------------------
    patches = [Patch(color=c, label=g) for g, c in _GATE_COLORS.items()
               if g not in ("i", "__cluster__")]
    if patches:
        ax.legend(handles=patches, fontsize=7, loc="upper right")

    if orient == "tower":
        ax.set_title("3-D tower view")
        ax.set_axis_off()
        ax.view_init(elev=20, azim=35)
    elif orient == "laneless":
        ax.set_title("Laneless qubit graph")
        ax.axis("off")
    else:
        ax.set_title("Qubit lanes")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 5) Re-export previous DAG visualiser for convenience
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from circuit_dag_prev import CircuitDAG  # type: ignore
