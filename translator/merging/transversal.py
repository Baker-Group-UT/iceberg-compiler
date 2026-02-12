"""Transversal CX/CZ merge detection."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGOpNode

from .gates import CXTGate, CZTGate, XCXTGate
from .layer import collect_events
from .models import MacroReplacement, PlannedOp, _PendingTransversal
from .single_qubit import MACRO_NAMES, SKIP_NAMES


def find_transversal_merges(
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
    events = collect_events(layer_index, order_index, lane_map)
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
            if pending and _node_blocks_pending(node, pending, patches_by_qubit):
                pending = None
            continue

        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES:
            continue
        if name in MACRO_NAMES:
            if pending and _node_blocks_pending(node, pending, patches_by_qubit):
                pending = None
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
    first: _PendingTransversal,
    second: _PendingTransversal,
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
    elif first.gate_name == "cz":
        macro_gate = CZTGate(
            control_patch=first.control_patch,
            target_patch=first.target_patch,
            control_count=len(control_order),
            logical_qubits=logical_indices,
        )
    else:
        macro_gate = XCXTGate(
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
) -> Optional[_PendingTransversal]:
    if name not in {"cx", "cz", "xcx"}:
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

    if name in {"cz", "xcx"} and patch1 < patch0:
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
    pending: _PendingTransversal,
    patch_lookup: Dict[object, int],
) -> bool:
    qargs = set(node.qargs)
    if pending.control_qubit in qargs or pending.target_qubit in qargs:
        return True
    patches = {patch_lookup.get(q) for q in qargs if patch_lookup.get(q) is not None}
    return len(patches) > 1
