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
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import SwapGate
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


class _TransversalMacroGate(Gate):
    """Base helper to stash metadata for transversal macros."""

    def __init__(
        self,
        name: str,
        num_qubits: int,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(name, num_qubits, [])
        self.metadata = {
            "control_patch": control_patch,
            "target_patch": target_patch,
            "control_count": control_count,
            "logical_qubits": list(logical_qubits),
        }


class CXTGate(_TransversalMacroGate):
    def __init__(
        self,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(
            "cxt",
            len(logical_qubits),
            control_patch=control_patch,
            target_patch=target_patch,
            control_count=control_count,
            logical_qubits=logical_qubits,
        )


class CZTGate(_TransversalMacroGate):
    def __init__(
        self,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(
            "czt",
            len(logical_qubits),
            control_patch=control_patch,
            target_patch=target_patch,
            control_count=control_count,
            logical_qubits=logical_qubits,
        )


@dataclass(frozen=True)
class PlannedOp:
    gate: Gate
    qargs: Tuple[object, ...]


@dataclass(frozen=True, eq=False)
class MacroReplacement:
    """Instruction replacement plan covering HH, CXT, and CZT merges."""

    first: DAGOpNode
    second: DAGOpNode
    macro_gate: Gate
    macro_qargs: Tuple[object, ...]
    blocked_qubits: Tuple[object, ...]
    pre_ops: Tuple[PlannedOp, ...] = ()
    post_ops: Tuple[PlannedOp, ...] = ()


DEFAULT_SETTINGS = HHMergeSettings()

SKIP_NAMES = {"barrier"}


def preprocess_program_for_merging(
    program_circuit: QuantumCircuit,
    permutation: Sequence[Tuple[int, int]],
    *,
    settings: Optional[HHMergeSettings] = None,
    allow_transversal_swap_merges: bool = False,
) -> QuantumCircuit:
    """
    Return a possibly rewritten circuit where compatible H, CX, and CZ pairs are
    replaced with HH / CXT / CZT macro instructions. When
    ``allow_transversal_swap_merges`` is true the pass is also permitted to wrap
    the macro in explicit SWAP gates to temporarily align control/target lanes.
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

    qubit_indices = {qubit: program_circuit.find_bit(qubit).index for qubit in program_circuit.qubits}

    consumed_nodes: set = set()
    records: List[MacroReplacement] = []
    for patch, qubits in patch_pairs.items():
        records.extend(
            _find_hh_merges_for_patch(layer_index, order_index, qubits, settings, consumed_nodes)
        )

    records.extend(
        _find_transversal_merges(
            layer_index,
            order_index,
            patch_pairs,
            qubit_indices,
            consumed_nodes,
            allow_transversal_swap_merges,
        )
    )

    if not records:
        return program_circuit

    return _build_rewritten_circuit(
        program_circuit,
        dag,
        records,
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


def _find_hh_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Tuple,
    settings: HHMergeSettings,
    consumed: set,
) -> List[MacroReplacement]:
    lane_map = {qubits[0]: 0, qubits[1]: 1}
    events = _collect_events(layer_index, order_index, lane_map)
    merges: List[MacroReplacement] = []
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
            record = _build_hh_record(waiting.node, node)
            merges.append(record)
            consumed.add(waiting.node)
            consumed.add(node)
            waiting = None
            continue

    return merges


def _build_hh_record(first: DAGOpNode, second: DAGOpNode) -> MacroReplacement:
    waiting_qubit = first.qargs[0]
    lead_qubit = second.qargs[0]
    return MacroReplacement(
        first=first,
        second=second,
        macro_gate=HHGate(),
        macro_qargs=(waiting_qubit, lead_qubit),
        blocked_qubits=(waiting_qubit,),
    )


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


def _find_transversal_merges(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    patch_pairs: Dict[int, Tuple],
    qubit_indices: Dict[object, int],
    consumed: set,
    allow_transversal_swap_merges: bool,
) -> List[MacroReplacement]:
    patch_ids = sorted(patch_pairs.keys())
    records: List[MacroReplacement] = []
    for idx, left_patch in enumerate(patch_ids):
        for right_patch in patch_ids[idx + 1 :]:
            records.extend(
                _scan_patch_pair_for_transversal(
                    layer_index,
                    order_index,
                    left_patch,
                    right_patch,
                    patch_pairs[left_patch],
                    patch_pairs[right_patch],
                    qubit_indices,
                    consumed,
                    allow_transversal_swap_merges,
                )
            )
    return records


def _scan_patch_pair_for_transversal(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    patch_a: int,
    patch_b: int,
    qubits_a: Tuple,
    qubits_b: Tuple,
    qubit_indices: Dict[object, int],
    consumed: set,
    allow_transversal_swap_merges: bool,
) -> List[MacroReplacement]:
    lane_map = {q: slot for slot, q in enumerate(qubits_a + qubits_b)}
    events = _collect_events(layer_index, order_index, lane_map)
    patches_by_qubit = {q: patch_a for q in qubits_a}
    patches_by_qubit.update({q: patch_b for q in qubits_b})

    slots_by_patch: Dict[int, Dict[int, object]] = {
        patch_a: {slot: qubit for slot, qubit in enumerate(qubits_a)},
        patch_b: {slot: qubit for slot, qubit in enumerate(qubits_b)},
    }
    slot_lookup: Dict[object, int] = {}
    for patch, slots in slots_by_patch.items():
        for slot, qubit in slots.items():
            slot_lookup[qubit] = slot

    pending: Optional[_PendingTransversal] = None
    records: List[MacroReplacement] = []

    for _, _, _, node in events:
        if node in consumed:
            continue

        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES:
            continue

        if name == "swap" and _apply_real_swap(node, patches_by_qubit, slots_by_patch, slot_lookup):
            continue

        info = _gate_info_for_pair(
            node,
            name,
            patch_a,
            patch_b,
            patches_by_qubit,
            slot_lookup,
        )

        if info is not None:
            if pending is None:
                pending = info
                continue

            record = _evaluate_transversal_pair(
                pending,
                info,
                slots_by_patch,
                slot_lookup,
                qubit_indices,
                allow_transversal_swap_merges,
            )

            if record is None:
                pending = info
                continue

            records.append(record)
            consumed.add(record.first)
            consumed.add(record.second)
            pending = None
            continue

        if pending and _node_blocks_pending(node, pending, patches_by_qubit):
            pending = None

    return records


def _evaluate_transversal_pair(
    first: "_PendingTransversal",
    second: "_PendingTransversal",
    slots_by_patch: Dict[int, Dict[int, object]],
    slot_lookup: Dict[object, int],
    qubit_indices: Dict[object, int],
    allow_transversal_swap_merges: bool,
) -> Optional[MacroReplacement]:
    if first.gate_name != second.gate_name:
        return None

    if first.gate_name == "cx" and first.control_patch != second.control_patch:
        return None

    if first.target_patch != second.target_patch:
        return None

    control_pairs = {
        first.control_slot: first.target_slot,
        second.control_slot: second.target_slot,
    }
    if len(control_pairs) != 2 or len(set(control_pairs.values())) != 2:
        return None

    ordered_control_slots = sorted(control_pairs.keys())
    slot_lo, slot_hi = ordered_control_slots
    identity_map = control_pairs[slot_lo] == slot_lo and control_pairs[slot_hi] == slot_hi
    cross_map = control_pairs[slot_lo] == slot_hi and control_pairs[slot_hi] == slot_lo

    pre_ops: Tuple[PlannedOp, ...] = ()
    post_ops: Tuple[PlannedOp, ...] = ()
    swapped_patch: Optional[int] = None
    swapped_slots: Optional[Tuple[int, int]] = None

    if not identity_map:
        if not allow_transversal_swap_merges or not cross_map:
            return None
        # Prefer swapping the target patch when alignment forms a cross map.
        swapped_patch = first.target_patch
        swapped_slots = tuple(sorted({first.target_slot, second.target_slot}))
        swap_qubits = tuple(slots_by_patch[swapped_patch][slot] for slot in swapped_slots)
        swap_gate = SwapGate()
        planned = PlannedOp(gate=swap_gate, qargs=swap_qubits)
        pre_ops = (planned,)
        post_ops = (PlannedOp(gate=SwapGate(), qargs=swap_qubits),)
        _apply_slot_swap(slots_by_patch, slot_lookup, swapped_patch, swapped_slots[0], swapped_slots[1])

    control_order = tuple(qubit for _, qubit in sorted(slots_by_patch[first.control_patch].items()))
    target_order = tuple(qubit for _, qubit in sorted(slots_by_patch[first.target_patch].items()))
    macro_qargs = control_order + target_order
    logical_indices = tuple(qubit_indices[q] for q in macro_qargs)

    if first.gate_name == "cx":
        macro_gate = CXTGate(
            control_patch=first.control_patch,
            target_patch=first.target_patch,
            control_count=len(control_order),
            logical_qubits=logical_indices,
        )
    else:
        macro_gate = CZTGate(
            control_patch=first.control_patch,
            target_patch=first.target_patch,
            control_count=len(control_order),
            logical_qubits=logical_indices,
        )

    record = MacroReplacement(
        first=first.node,
        second=second.node,
        macro_gate=macro_gate,
        macro_qargs=macro_qargs,
        blocked_qubits=(first.control_qubit, first.target_qubit),
        pre_ops=pre_ops,
        post_ops=post_ops,
    )

    if swapped_patch is not None and swapped_slots is not None:
        _apply_slot_swap(slots_by_patch, slot_lookup, swapped_patch, swapped_slots[0], swapped_slots[1])

    return record


def _apply_real_swap(
    node: DAGOpNode,
    patch_lookup: Dict[object, int],
    slots_by_patch: Dict[int, Dict[int, object]],
    slot_lookup: Dict[object, int],
) -> bool:
    qargs = list(node.qargs)
    if len(qargs) != 2:
        return False
    patch = patch_lookup.get(qargs[0])
    if patch is None or patch != patch_lookup.get(qargs[1]):
        return False
    slot_a = slot_lookup.get(qargs[0])
    slot_b = slot_lookup.get(qargs[1])
    if slot_a is None or slot_b is None:
        return False
    _apply_slot_swap(slots_by_patch, slot_lookup, patch, slot_a, slot_b)
    return True


def _apply_slot_swap(
    slots_by_patch: Dict[int, Dict[int, object]],
    slot_lookup: Dict[object, int],
    patch: int,
    slot_a: int,
    slot_b: int,
) -> None:
    if slot_a == slot_b:
        return
    qa = slots_by_patch[patch][slot_a]
    qb = slots_by_patch[patch][slot_b]
    slots_by_patch[patch][slot_a], slots_by_patch[patch][slot_b] = qb, qa
    slot_lookup[qa], slot_lookup[qb] = slot_b, slot_a


def _gate_info_for_pair(
    node: DAGOpNode,
    name: str,
    patch_a: int,
    patch_b: int,
    patch_lookup: Dict[object, int],
    slot_lookup: Dict[object, int],
) -> Optional["_PendingTransversal"]:
    if name not in {"cx", "cz"}:
        return None
    qargs = list(node.qargs)
    if len(qargs) != 2:
        return None
    patch0 = patch_lookup.get(qargs[0])
    patch1 = patch_lookup.get(qargs[1])
    if patch0 is None or patch1 is None:
        return None
    if {patch0, patch1} != {patch_a, patch_b}:
        return None
    slot0 = slot_lookup.get(qargs[0])
    slot1 = slot_lookup.get(qargs[1])
    if slot0 is None or slot1 is None:
        return None
    if patch0 == patch1:
        return None

    control_patch = patch0
    target_patch = patch1
    control_qubit = qargs[0]
    target_qubit = qargs[1]
    control_slot = slot0
    target_slot = slot1

    if name == "cz" and patch1 < patch0:
        control_patch, target_patch = target_patch, control_patch
        control_qubit, target_qubit = target_qubit, control_qubit
        control_slot, target_slot = target_slot, control_slot

    return _PendingTransversal(
        node=node,
        gate_name=name,
        control_patch=control_patch,
        target_patch=target_patch,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        control_slot=control_slot,
        target_slot=target_slot,
    )


def _node_blocks_pending(
    node: DAGOpNode,
    pending: "_PendingTransversal",
    patch_lookup: Dict[object, int],
) -> bool:
    qargs = set(node.qargs)
    if pending.control_qubit in qargs or pending.target_qubit in qargs:
        return True
    patches = {patch_lookup.get(q) for q in qargs if patch_lookup.get(q) is not None}
    return len(patches) > 1


def _build_rewritten_circuit(
    original: QuantumCircuit,
    dag: DAGCircuit,
    records: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
) -> QuantumCircuit:
    new_qc = QuantumCircuit(*original.qregs, *original.cregs)
    new_qc.global_phase = getattr(original, "global_phase", 0)
    if hasattr(original, "calibrations"):
        new_qc.calibrations = copy.deepcopy(original.calibrations)
    if hasattr(original, "metadata"):
        new_qc.metadata = copy.deepcopy(original.metadata)

    skip_nodes = {record.first for record in records}
    replacements = {record.second: record for record in records}
    waiting_lookup = {record.first: record for record in records}

    blocked_qubits: Dict[object, MacroReplacement] = {}
    delayed_nodes: Dict[MacroReplacement, List[DAGOpNode]] = {}
    ordered_nodes = sorted(
        dag.topological_op_nodes(),
        key=lambda n: (layer_index.get(n, 0), order_index.get(n, 0)),
    )
    for node in ordered_nodes:
        if node in skip_nodes:
            record = waiting_lookup.get(node)
            if record is not None:
                for qubit in record.blocked_qubits:
                    blocked_qubits[qubit] = record
                delayed_nodes.setdefault(record, [])
            continue
        if node in replacements:
            record = replacements[node]
            for planned in record.pre_ops:
                new_qc.append(planned.gate.copy(), list(planned.qargs), [])
            new_qc.append(record.macro_gate, list(record.macro_qargs), [])
            for planned in record.post_ops:
                new_qc.append(planned.gate.copy(), list(planned.qargs), [])
            for delayed in delayed_nodes.get(record, []):
                new_qc.append(delayed.op.copy(), delayed.qargs, delayed.cargs)
            delayed_nodes.pop(record, None)
            for qubit in record.blocked_qubits:
                blocked_qubits.pop(qubit, None)
            continue
        blocking_records = {blocked_qubits[q] for q in node.qargs if q in blocked_qubits}
        if blocking_records:
            record = next(iter(blocking_records))
            delayed_nodes.setdefault(record, []).append(node)
            continue
        new_qc.append(node.op.copy(), node.qargs, node.cargs)

    return new_qc


@dataclass
class _WaitingState:
    lane: Lane
    node: DAGOpNode
    remaining: float


@dataclass
class _PendingTransversal:
    node: DAGOpNode
    gate_name: str
    control_patch: int
    target_patch: int
    control_qubit: object
    target_qubit: object
    control_slot: int
    target_slot: int


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
