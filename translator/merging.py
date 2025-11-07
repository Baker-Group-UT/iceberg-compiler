"""
HH merging pre-processing
=========================
Given a permutation and a program circuit:
    For each patch:
        two pointers, one for each qubit
        increment pointers along layers until they encounter an H
            if p0 == H and p1 == H:
                HH
            if only one pointer has an H, we create a ghost pointer for the H lane
                then increment ghost pointer and other pointer, checking for two qubit gates
                increment until either GP and OP see two-qubit gates (scary vibes, could be path)
                OR we run out lookahead
                if we encounter a H on OP and neither of the exit conditions have been met,
                we delay the original H to an HH

This way, we minimize runtime, we create HH's where reasonable without introducing much delay to the circuit.

Pseudo-code sketch
------------------
def preprocess(program_circuit, permutation, settings):
    dag = circuit_to_dag(program_circuit)
    layers = list(dag.serial_layers())  # cirq-style "moments"
    patch_pairs = build_patch_pairs(permutation)  # patch -> (lane_a, lane_b)
    merges = []
    for patch, (qa, qb) in patch_pairs.items():
        merges += find_hh_merges(qa, qb, layers, settings)
    dag = apply_merges(dag, merges)  # replace delayed H pairs with HH macro gates
    return dag_to_circuit(dag), merges

def find_hh_merges(qa, qb, layers, settings):
    cursors = {qa: LayerCursor(qa), qb: LayerCursor(qb)}
    ghost = None  # stores pending H info (lane, layer_idx, node, remaining budget)
    results = []
    while not cursors_exhausted(cursors, layers):
        for lane in (qa, qb):
            cursors[lane].advance_to_next_gate(layers)

        ga = cursors[qa].current_gate()
        gb = cursors[qb].current_gate()
        if ga is None or gb is None:
            break  # one lane ran out of gates

        if ga.name == "h" and gb.name == "h":
            results.append(make_merge(qa, qb, ga, gb))
            cursors[qa].consume()
            cursors[qb].consume()
            ghost = None
            continue

        # spawn or update ghost pointer when exactly one lane sees H
        if ga.name == "h" and gb.name != "h" and ghost is None:
            ghost = GhostState(qubit=qa, node=ga, budget=settings.hh_budget)
            cursors[qa].pause()
            cursors[qb].consume()
            continue
        if gb.name == "h" and ga.name != "h" and ghost is None:
            ghost = GhostState(qubit=qb, node=gb, budget=settings.hh_budget)
            cursors[qb].pause()
            cursors[qa].consume()
            continue

        if ghost:
            lead = qb if ghost.qubit == qa else qa
            lead_gate = cursors[lead].current_gate()
            if is_two_qubit(lead_gate) or touches_other_patch(lead_gate):
                pending = ghost
                ghost = None
                cursors[pending.qubit].consume_original_h()
                continue

            cost = settings.delay_costs.get(lead_gate.name, settings.default_cost)
            if ghost.budget < cost:
                pending = ghost
                ghost = None
                cursors[pending.qubit].consume_original_h()
                continue

            if lead_gate.name == "h":
                results.append(make_merge(ghost.qubit, lead, ghost.node, lead_gate))
                cursors[lead].consume()
                cursors[ghost.qubit].consume_original_h()
                ghost = None
            else:
                ghost.budget -= cost
                cursors[lead].consume()
                continue

        # neither lane is on an H or ghost was resolved
        if not ghost:
            cursors[qa].consume()
            cursors[qb].consume()

    return results

def apply_merges(dag, merges):
    for merge in merges:
        dag.remove_op_node(merge.left)
        dag.remove_op_node(merge.right)
        dag.apply_operation_back(HHGate(), qargs=[merge.qa, merge.qb], metadata=merge.timing)
    return dag

The actual implementation can build atop these helpers, keeping the delay-cost
table and budget in a config object so HH, CX, and CZ merging share the same tuning knobs.

CNOT
look at ipad
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

Lane = int


@dataclass
class HHMergeSettings:
    """Tunable parameters for HH merging."""

    budget: float = 5.0
    default_cost: float = 1.0
    delay_costs: Dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.1,
            "z": 0.1,
            "s": 0.75,
            "sdg": 0.75,
            "t": 0.9,
            "tdg": 0.9,
            "cx": 3.5,
            "cz": 3.5,
            "measure": float("inf"),
        }
    )

    def cost_for(self, name: str) -> float:
        return self.delay_costs.get(name, self.default_cost)


class HHGate(Gate):
    """Custom logical placeholder for a merged HH gate."""

    def __init__(self):
        super().__init__("hh", 2, [])

    def inverse(self):
        return HHGate()


@dataclass(frozen=True)
class MergeRecord:
    """Book-keeping for replacing two H nodes with one HH instruction."""

    first: DAGOpNode
    second: DAGOpNode
    waiting_qubit: object
    lead_qubit: object


DEFAULT_SETTINGS = HHMergeSettings()

SKIP_NAMES = {"barrier"}


def preprocess_program_for_merging(
    program_circuit: QuantumCircuit,
    permutation: Sequence[Tuple[int, int]],
    *,
    settings: Optional[HHMergeSettings] = None,
) -> QuantumCircuit:
    """
    Return a possibly rewritten circuit where certain pairs of H gates inside a
    patch are replaced with an HH macro instruction.
    """
    if program_circuit.num_qubits == 0:
        return program_circuit

    if settings is None:
        settings = DEFAULT_SETTINGS

    dag = circuit_to_dag(program_circuit)
    layer_index, order_index = _layer_by_node(dag)
    patch_pairs = _build_patch_pairs(permutation, program_circuit.qubits)
    if not patch_pairs:
        return program_circuit

    skip_nodes = set()
    consumed_nodes: set = set()
    replacements: Dict[DAGOpNode, MergeRecord] = {}
    waiting_lookup: Dict[DAGOpNode, MergeRecord] = {}
    for patch, qubits in patch_pairs.items():
        merges = _find_merges_for_patch(layer_index, order_index, qubits, settings, consumed_nodes)
        for first, second in merges:
            waiting_qubit = first.qargs[0]
            lead_qubit = second.qargs[0]
            record = MergeRecord(first=first, second=second, waiting_qubit=waiting_qubit, lead_qubit=lead_qubit)
            skip_nodes.add(first)
            replacements[second] = record
            waiting_lookup[first] = record

    if not replacements:
        return program_circuit

    return _build_rewritten_circuit(
        program_circuit,
        dag,
        skip_nodes,
        replacements,
        waiting_lookup,
        layer_index,
        order_index,
    )


def _build_patch_pairs(
    permutation: Sequence[Tuple[int, int]],
    qubits: Sequence,
) -> Dict[int, Tuple]:
    """Map patch id -> tuple of two logical qubits (ordered by slot)."""
    patch_slots: Dict[int, Dict[int, object]] = {}
    for logical_i, (patch, slot) in enumerate(permutation):
        if logical_i >= len(qubits):
            break
        patch_slots.setdefault(patch, {})[slot] = qubits[logical_i]

    patch_pairs: Dict[int, Tuple] = {}
    for patch, slots in patch_slots.items():
        if len(slots) < 2:
            continue
        ordered = [q for _, q in sorted(slots.items(), key=lambda item: item[0])]
        patch_pairs[patch] = (ordered[0], ordered[1])
    return patch_pairs


def _find_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Tuple,
    settings: HHMergeSettings,
    consumed: set,
) -> List[Tuple[DAGOpNode, DAGOpNode]]:
    lane_map = {qubits[0]: 0, qubits[1]: 1}
    events = _collect_events(layer_index, order_index, lane_map)
    merges: List[Tuple[DAGOpNode, DAGOpNode]] = []
    waiting: Optional[_WaitingState] = None

    for _, _, lanes, node in events:
        if node in consumed or getattr(node.op, "name", "").lower() in SKIP_NAMES:
            continue

        name = getattr(node.op, "name", "").lower()
        touches_both = len(set(lanes)) > 1
        is_measure = name == "measure"
        multi_qubit = len(node.qargs) > 1

        if waiting is None:
            if name == "h" and not touches_both:
                waiting = _WaitingState(lane=lanes[0], node=node, remaining=settings.budget)
            continue

        if touches_both or is_measure:
            waiting = None
            continue

        lane = lanes[0]
        if lane == waiting.lane:
            if multi_qubit:
                waiting = None
                continue
            if name == "h":
                waiting = _WaitingState(lane=lane, node=node, remaining=settings.budget)
                continue
            cost = settings.cost_for(name)
            if cost == float("inf"):
                waiting = None
                continue
            waiting.remaining -= cost
            if waiting.remaining < 0:
                waiting = None
            continue

        # lead lane encountered a gate
        if name == "h":
            merges.append((waiting.node, node))
            consumed.add(waiting.node)
            consumed.add(node)
            waiting = None
            continue

    return merges


def _collect_events(
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


def _build_rewritten_circuit(
    original: QuantumCircuit,
    dag: DAGCircuit,
    skip_nodes: set,
    replacements: Dict[DAGOpNode, MergeRecord],
    waiting_lookup: Dict[DAGOpNode, MergeRecord],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
) -> QuantumCircuit:
    new_qc = QuantumCircuit(*original.qregs, *original.cregs)
    new_qc.global_phase = getattr(original, "global_phase", 0)
    if hasattr(original, "calibrations"):
        new_qc.calibrations = copy.deepcopy(original.calibrations)
    if hasattr(original, "metadata"):
        new_qc.metadata = copy.deepcopy(original.metadata)

    hh_gate = HHGate()
    blocked_qubits: Dict[object, MergeRecord] = {}
    delayed_nodes: Dict[MergeRecord, List[DAGOpNode]] = {}
    ordered_nodes = sorted(
        dag.topological_op_nodes(),
        key=lambda n: (layer_index.get(n, 0), order_index.get(n, 0)),
    )
    for node in ordered_nodes:
        if node in skip_nodes:
            record = waiting_lookup.get(node)
            if record is not None:
                blocked_qubits[record.waiting_qubit] = record
                delayed_nodes.setdefault(record, [])
            continue
        blocking_records = {blocked_qubits[q] for q in node.qargs if q in blocked_qubits}
        if blocking_records:
            record = next(iter(blocking_records))
            delayed_nodes.setdefault(record, []).append(node)
            continue
        if node in replacements:
            record = replacements[node]
            new_qc.append(hh_gate, [record.waiting_qubit, record.lead_qubit], [])
            for delayed in delayed_nodes.get(record, []):
                new_qc.append(delayed.op.copy(), delayed.qargs, delayed.cargs)
            delayed_nodes.pop(record, None)
            blocked_qubits.pop(record.waiting_qubit, None)
            continue
        new_qc.append(node.op.copy(), node.qargs, node.cargs)

    return new_qc


@dataclass
class _WaitingState:
    lane: Lane
    node: DAGOpNode
    remaining: float


def _layer_by_node(dag: DAGCircuit) -> Tuple[Dict[DAGOpNode, int], Dict[DAGOpNode, int]]:
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
