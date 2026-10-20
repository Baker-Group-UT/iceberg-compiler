"""
Transversal CX/CZ/XCX merge detection, at any number of slots per patch.

A transversal macro is a *whole-patch* operation: `CNOT_k_Transversal` emits one
physical CNOT per data qubit, which is a logical CNOT on every lane at once. So
fusing program gates into one is only legal when the group covers every occupied
lane of both patches, and when lane i of the control patch is paired with lane i of
the target. At k=2 a group of two gates is automatically the whole patch and the
only misalignment possible is a single transposition, which is why the pairwise
version got away with enumerating `identity_map` and `cross_map` by hand.

Beyond k=2 both conditions become general:

*coverage*  a group needs one gate per occupied lane, so k gates rather than two.
            Partial coverage cannot be patched up by fusing a subset -- transversal
            CX on a subset of positions maps the full-weight stabilizer
            `S_X = X^(x)n` to something outside the stabilizer group, so it is not a
            logical operation at all. Coverage is all-or-nothing.

*alignment* the control-slot -> target-slot map is a permutation of k slots, of
            which only the identity can be emitted directly. Any other permutation
            is fixed by physically permuting the target patch, decomposed into
            transpositions and emitted as routing swaps around the macro. A routing
            swap is an atom exchange -- two carried atoms, `move_error` each -- so
            realignment is *not* free: `_macro_gain` prices the macro against the
            singles it replaces minus its swaps, and a macro that does not pay for
            itself is not taken. Coverage plus a positive gain plus surviving
            scheduling is now the whole test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from qiskit.dagcircuit import DAGOpNode

from ..custom_gates import RoutingSwapGate
from .gates import CXTGate, CZTGate, XCXTGate
from .layer import collect_events
from .models import MacroReplacement, PlannedOp, _PendingTransversal
from .single_qubit import MACRO_NAMES, SKIP_NAMES

_MACRO_FACTORIES = {"cx": CXTGate, "cz": CZTGate, "xcx": XCXTGate}

#: Macro key and the cross-patch single it replaces, per program gate name.
_MACRO_COST_KEYS = {"cx": ("CXT", "CX"), "cz": ("CZT", "CZ"), "xcx": ("XCXT", "XCX")}


def _macro_gain(
    gate_name: str, lane_count: int, n_code: int, transpositions: int
) -> float:
    """
    Weighted-work gain of one transversal macro over the singles it replaces.

    `lane_count` cross-patch singles against one macro at code size `n_code`, minus
    the routing swaps realignment emits (one before and one after the macro per
    transposition) priced as atom exchanges -- `iceberg.costs.carriage_cost`. This is
    the comparison that used to not exist: a macro was taken unconditionally once
    coverage held, which both accepted misaligned macros whose swaps outweighed the
    saving and fired n-gate macros over two occupied lanes of a large patch, where
    2 x (0, 4) singles are cheaper than one (0, n) macro.

    Movement is deliberately not decided here beyond the swaps: one grouped macro trip
    against `lane_count` single trips is a *further* argument for the macro that needs
    distances this pass does not have, so the gate is conservative for aligned macros.
    """
    from iceberg.costs import CROSS_PATCH_COST, carriage_cost, work_table

    from .conflicts import DEFAULT_MERGE_WORK_WEIGHTS

    keys = _MACRO_COST_KEYS.get(gate_name)
    if keys is None:
        return 0.0
    macro_key, single_key = keys
    table = work_table(n_code)
    single = CROSS_PATCH_COST.get(single_key, table[single_key])
    macro = table[macro_key]
    w1, w2 = DEFAULT_MERGE_WORK_WEIGHTS
    realign = 2 * transpositions * carriage_cost("SWAP", (w1, w2))
    return (
        lane_count * (single[0] * w1 + single[1] * w2)
        - (macro[0] * w1 + macro[1] * w2)
        - realign
    )


def find_transversal_merges(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    patch_lanes: Dict[int, Tuple],
    qubit_indices: Dict[object, int],
    consumed: set,
    allow_transversal_swap_merges: bool,
    n_code: int = 4,
) -> List[MacroReplacement]:
    patch_ids = sorted(patch_lanes.keys())
    records: List[MacroReplacement] = []
    for idx, left_patch in enumerate(patch_ids):
        for right_patch in patch_ids[idx + 1 :]:
            # A macro pairs lane i with lane i, so the two patches must hold the
            # same number of qubits. Unequal occupancy has no valid pairing.
            if len(patch_lanes[left_patch]) != len(patch_lanes[right_patch]):
                continue
            records.extend(
                _scan_patch_pair_for_transversal(
                    layer_index,
                    order_index,
                    left_patch,
                    right_patch,
                    patch_lanes[left_patch],
                    patch_lanes[right_patch],
                    qubit_indices,
                    consumed,
                    allow_transversal_swap_merges,
                    n_code=n_code,
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
    n_code: int = 4,
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

    lane_count = len(qubits_a)
    if lane_count < 2:
        # One occupied lane per patch fuses nothing: a transversal macro over a
        # single lane replaces one gate with one gate. `_build_patch_lanes` already
        # drops such patches, but `analysis.PairMergeAnalyzer` calls in here with a
        # solo lane on each side to ask what two qubits would gain from *separate*
        # patches, and the answer is nothing. The pairwise implementation reached the
        # same conclusion by a longer route -- it needed two candidates before
        # evaluating, then rejected them because their slot map had one entry rather
        # than two -- so those counts have always been zero. See the note in
        # `PairMergeStats` about the packer relying on them.
        return []

    pending: List[_PendingTransversal] = []
    records: List[MacroReplacement] = []

    from translator import explore_switches

    def flush() -> None:
        # Branch v2-tvd-explore: a group that can grow no further fires as macro +
        # undo when most lane pairs want the gate and the work pays. Off, this is the
        # shipped `pending.clear()`.
        if explore_switches.TRANSVERSAL_MAJORITY_UNDO and 2 * len(pending) > lane_count:
            record = _build_transversal_record(
                list(pending),
                slots_by_patch,
                slot_lookup,
                qubit_indices,
                allow_transversal_swap_merges,
                n_code=n_code,
                allow_undo=True,
            )
            if record is not None:
                records.append(record)
                consumed.update(record.nodes)
        pending.clear()

    for _, _, _, node in events:
        if node in consumed:
            if _node_blocks_pending(node, pending, patches_by_qubit):
                flush()
            continue

        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES:
            continue
        if name in MACRO_NAMES:
            if _node_blocks_pending(node, pending, patches_by_qubit):
                flush()
            continue

        if name == "swap" and _is_real_swap(node, patches_by_qubit, slot_lookup):
            # Flush first: a majority-undo record is built against the pre-SWAP slots.
            flush()
            _apply_real_swap(node, patches_by_qubit, slots_by_patch, slot_lookup)
            # A program SWAP relabels the patch's slots, so a candidate found before
            # it cannot be fused with one found after: the macro would have to be
            # reordered across the SWAP, and the slot map the earlier candidate was
            # measured against no longer holds. Fusing anyway produced macros that
            # paired the wrong lanes.
            flush()
            continue

        info = _gate_info_for_pair(
            node, name, patch_a, patch_b, patches_by_qubit, slot_lookup
        )

        if info is not None:
            if not _extends_group(pending, info):
                # Not compatible with what is already collected: restart from here
                # rather than dropping this gate, since it may lead its own group.
                flush()
            pending.append(info)
            if len(pending) < lane_count:
                continue

            record = _build_transversal_record(
                list(pending),
                slots_by_patch,
                slot_lookup,
                qubit_indices,
                allow_transversal_swap_merges,
                n_code=n_code,
            )
            pending.clear()
            if record is None:
                continue
            records.append(record)
            consumed.update(record.nodes)
            continue

        if _node_blocks_pending(node, pending, patches_by_qubit):
            flush()

    flush()
    return records


def _extends_group(
    pending: Sequence[_PendingTransversal], info: _PendingTransversal
) -> bool:
    """
    Whether `info` can join the group already collected.

    Same gate kind, same orientation of the same patch pair, and a lane neither side
    has used yet -- the group has to end up a bijection between the two patches'
    slots, so a repeated control or target slot can never be completed.
    """
    if not pending:
        return True
    head = pending[0]
    if info.gate_name != head.gate_name:
        return False
    if info.control_patch != head.control_patch or info.target_patch != head.target_patch:
        return False
    if any(info.control_slot == other.control_slot for other in pending):
        return False
    if any(info.target_slot == other.target_slot for other in pending):
        return False
    return True


def _build_transversal_record(
    group: List[_PendingTransversal],
    slots_by_patch: Dict[int, Dict[int, object]],
    slot_lookup: Dict[object, int],
    qubit_indices: Dict[object, int],
    allow_transversal_swap_merges: bool,
    n_code: int = 4,
    allow_undo: bool = False,
) -> Optional[MacroReplacement]:
    head = group[0]
    control_patch = head.control_patch
    target_patch = head.target_patch

    # control slot -> target slot. `_extends_group` has already guaranteed both
    # sides are free of repeats, so this is a bijection on the slots present.
    alignment = {item.control_slot: item.target_slot for item in group}
    occupied = set(slots_by_patch[control_patch])
    undone: List[int] = []
    if set(alignment) != occupied or set(alignment.values()) != set(
        slots_by_patch[target_patch]
    ):
        # The group misses an occupied lane, so a whole-patch macro would apply the
        # gate to a lane the program never asked about.
        if not allow_undo:
            return None
        undone = _complete_alignment(alignment, occupied, set(slots_by_patch[target_patch]))
        if undone is None:
            return None

    pre_ops: Tuple[PlannedOp, ...] = ()
    post_ops: Tuple[PlannedOp, ...] = ()
    transpositions: List[Tuple[int, int]] = []
    misaligned = any(
        control_slot != target_slot for control_slot, target_slot in alignment.items()
    )

    if misaligned:
        if not allow_transversal_swap_merges:
            return None
        transpositions = _transpositions_for(alignment)

    # The macro must actually be worth it. This gate did not exist: coverage plus
    # alignment used to be the whole test, so a misaligned macro was taken however
    # many swaps realignment cost, and an n-gate macro fired over two occupied lanes
    # of a large patch where two targeted singles are cheaper. See `_macro_gain`.
    # Each undone pair costs a single back, so the macro must pay for (2w - k) of them.
    if _macro_gain(head.gate_name, len(group) - len(undone), n_code, len(transpositions)) <= 0:
        return None

    if misaligned:
        swap_ops = []
        for slot_a, slot_b in transpositions:
            qubits = (
                slots_by_patch[target_patch][slot_a],
                slots_by_patch[target_patch][slot_b],
            )
            # A routing swap, not a program SWAP: it must physically move the data
            # so the macro's lane-i-to-lane-i pairing lines up. See
            # translator.handlers.swap for why the two cannot share an
            # implementation.
            swap_ops.append(PlannedOp(gate=RoutingSwapGate(), qargs=qubits))
            _apply_slot_swap(slots_by_patch, slot_lookup, target_patch, slot_a, slot_b)
        pre_ops = tuple(swap_ops)
        # Undo in reverse so the patch is left as the rest of the scan expects.
        post_ops = tuple(reversed(swap_ops))

    control_order = tuple(q for _, q in sorted(slots_by_patch[control_patch].items()))
    target_order = tuple(q for _, q in sorted(slots_by_patch[target_patch].items()))
    macro_qargs = control_order + target_order
    # After realignment control slot c faces target slot c, so the macro paired the
    # qubits sitting there; the undo is the same gate on the same two program qubits.
    undo_ops = tuple(
        PlannedOp(
            gate=_single_gate(head.gate_name),
            qargs=(slots_by_patch[control_patch][slot], slots_by_patch[target_patch][slot]),
        )
        for slot in undone
    )
    logical_indices = tuple(qubit_indices[q] for q in macro_qargs)

    macro_gate = _MACRO_FACTORIES[head.gate_name](
        control_patch=control_patch,
        target_patch=target_patch,
        control_count=len(control_order),
        logical_qubits=logical_indices,
    )

    nodes = tuple(item.node for item in group)
    record = MacroReplacement(
        nodes=nodes,
        macro_gate=macro_gate,
        macro_qargs=macro_qargs,
        blocked_qubits=tuple(
            qubit
            for item in group[:-1]
            for qubit in (item.control_qubit, item.target_qubit)
        ),
        pre_ops=pre_ops,
        post_ops=post_ops,
        undo_ops=undo_ops,
    )

    # Restore the slot map: the swaps above are emitted around the macro, so from
    # the scan's point of view the patch never moved.
    for slot_a, slot_b in reversed(transpositions):
        _apply_slot_swap(slots_by_patch, slot_lookup, target_patch, slot_a, slot_b)

    return record


def _complete_alignment(
    alignment: Dict[int, int], control_occupied: set, target_occupied: set
) -> Optional[List[int]]:
    """
    Branch v2-tvd-explore: extend a partial slot bijection to every occupied lane.

    Uncovered control lanes are paired with uncovered target lanes -- the same slot
    wherever both are free, so an aligned group stays aligned -- and returned: those
    are the pairs the macro touches that the program did not ask for, and they get an
    undo gate each. Mutates `alignment`.
    """
    free_c = sorted(control_occupied - set(alignment))
    free_t = sorted(target_occupied - set(alignment.values()))
    if len(free_c) != len(free_t) or not free_c:
        return None
    undone: List[int] = []
    for slot in list(free_c):
        if slot in free_t:
            alignment[slot] = slot
            free_c.remove(slot)
            free_t.remove(slot)
            undone.append(slot)
    for control_slot, target_slot in zip(free_c, free_t):
        alignment[control_slot] = target_slot
        undone.append(control_slot)
    return undone


def _single_gate(gate_name: str):
    """The program gate a transversal macro of `gate_name` replaces, for undoing it."""
    from qiskit.circuit.library import CXGate, CZGate

    from ..custom_gates import XCXGate

    return {"cx": CXGate, "cz": CZGate, "xcx": XCXGate}[gate_name]()


def _is_real_swap(
    node: DAGOpNode, patch_lookup: Dict[object, int], slot_lookup: Dict[object, int]
) -> bool:
    """`_apply_real_swap`'s test, without applying it."""
    qargs = list(node.qargs)
    if len(qargs) != 2:
        return False
    patch = patch_lookup.get(qargs[0])
    if patch is None or patch != patch_lookup.get(qargs[1]):
        return False
    return slot_lookup.get(qargs[0]) is not None and slot_lookup.get(qargs[1]) is not None


def _transpositions_for(alignment: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    Transpositions that permute the target patch so lane i faces lane i.

    `alignment` maps a control slot to the target slot holding its partner, so the
    partner of control slot c has to end up at target slot c. Selection sort over
    the control slots: at most k-1 transpositions, emitted in a deterministic order.

    `occupant[s]` tracks which original target slot's data currently sits at slot s,
    which is exactly what `_apply_slot_swap` does to the real slot map.
    """
    occupant = {slot: slot for slot in alignment.values()}
    moves: List[Tuple[int, int]] = []
    for control_slot in sorted(alignment):
        wanted = alignment[control_slot]
        if occupant[control_slot] == wanted:
            continue
        holder = next(slot for slot, held in occupant.items() if held == wanted)
        occupant[control_slot], occupant[holder] = occupant[holder], occupant[control_slot]
        moves.append((min(control_slot, holder), max(control_slot, holder)))
    return moves


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
    if name not in _MACRO_FACTORIES:
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

    # CZ and XCX are symmetric in their operands, so orient them canonically by
    # patch id; that way two gates written in opposite orders still group.
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
    pending: Sequence[_PendingTransversal],
    patch_lookup: Dict[object, int],
) -> bool:
    """
    Whether `node` prevents the collected group from being fused.

    It does if it touches a qubit the group already claims -- that gate would have to
    move across the macro -- or if it is itself a cross-patch gate, which cannot be
    reordered against the macro either.
    """
    if not pending:
        return False
    qargs = set(node.qargs)
    for item in pending:
        if item.control_qubit in qargs or item.target_qubit in qargs:
            return True
    patches = {patch_lookup.get(q) for q in qargs if patch_lookup.get(q) is not None}
    return len(patches) > 1
