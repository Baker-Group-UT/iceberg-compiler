"""Layering helpers for merge analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from .models import Lane


def layer_by_node(dag: DAGCircuit) -> Tuple[Dict[DAGOpNode, int], Dict[DAGOpNode, int]]:
    """Assign earliest layer index and insertion order to each op node."""
    availability = {qubit: 0 for qubit in dag.qubits}
    layers: Dict[DAGOpNode, int] = {}
    order: Dict[DAGOpNode, int] = {}
    for idx, node in enumerate(dag.topological_op_nodes()):
        start = 0
        if node.qargs:
            start = max(availability[q] for q in node.qargs)
        layers[node] = start
        order[node] = getattr(node, "_node_id", idx)
        duration = 1
        for q in node.qargs:
            availability[q] = start + duration
    return layers, order


def collect_events(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    lane_map: Dict[object, Lane],
) -> List[Tuple[int, int, List[Lane], DAGOpNode]]:
    events: List[Tuple[int, int, List[Lane], DAGOpNode]] = []
    for node, layer_idx in layer_index.items():
        lanes: List[Lane] = []
        for q in node.qargs:
            if q in lane_map:
                lanes.append(lane_map[q])
        if lanes:
            events.append((layer_idx, order_index.get(node, 0), lanes, node))
    events.sort(key=lambda item: (item[0], item[1], len(item[2])))
    return events
