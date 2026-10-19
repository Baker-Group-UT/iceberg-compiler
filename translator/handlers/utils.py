"""
translator.gate_handlers.utils

Shared helpers for logical gate placement logic on Iceberg circuits.
"""

from __future__ import annotations
from typing import Dict, List

def is_patch_solo(logical_idx: int, active_by_patch: Dict[int, List[int]], logical_to_patch: Dict[int, int]) -> bool:
    """
    Returns True if the given logical qubit is the only active qubit in its patch.

    Args:
        logical_idx: Index of the logical qubit.
        active_by_patch: Mapping from patch index to list of active logical qubits.
        logical_to_patch: Mapping from logical qubit to its patch index.

    Returns:
        True if the patch contains only this qubit, False otherwise.
    """
    patch = logical_to_patch[logical_idx]
    return active_by_patch.get(patch, []).count(logical_idx) == 1 and len(active_by_patch[patch]) == 1

def active_slots_in_patch(context, patch: int) -> set:
    """
    Slots of `patch` that currently carry a logical qubit.

    `active_by_patch` lists *static* physical indices, so each has to be mapped through
    `current_pos_of` to find where its qubit actually sits.

    This used to derive the slot arithmetically as `index % k_log` and skip that step,
    justified by the claim that the occupied-slot *set* is invariant under the swaps the
    translator emits -- intra-patch swaps permuting occupied slots among themselves, and
    cross-patch swaps being rejected. **That holds only when the patch is full.** With
    k=6 and two lanes in use, `realign_target_slots` will swap an occupied slot with a
    *free* one, which moves the set: patch 1 of `seed 32` went from {0, 3} to {2, 4}
    while this function kept reporting {0, 3}. The following transversal macro then read
    its own live lanes as uncovered and was rejected for it.

    At k=2 an occupied patch is full, so static and live agreed and the shortcut was
    invisible.
    """
    live = set()
    for phys_index in context.active_by_patch.get(patch, []):
        current_patch, current_slot = context.current_pos_of(phys_index)
        if current_patch == patch:
            live.add(current_slot)
    return live


def covers_active_lanes(context, patch: int, slots) -> bool:
    """
    Whether `slots` accounts for every occupied lane of `patch`.

    The gate this guards -- whole-patch HH, or a transversal CX/CZ/XCX -- acts on
    *all* lanes of the patch whether or not a program gate asked for it. Applying it
    with only some lanes covered would silently add the gate to the rest, so a macro
    that does not cover them must be rejected and expanded back into single-lane
    gates.

    Lanes left empty by an uneven packing are excluded, so a macro covering the two
    occupied slots of a four-slot patch is admissible: nothing reads the idle lanes,
    and `apply_logical_hadamard` already fires a whole-patch HH for a patch holding
    a single qubit.
    """
    return set(slots) == active_slots_in_patch(context, patch)


def can_transversal_pair(q1: int, q2: int, idx_in_patch: Dict[int, int]) -> bool:
    """
    Checks whether two logical qubits are aligned for transversal operation.

    Args:
        q1: First logical qubit index.
        q2: Second logical qubit index.
        idx_in_patch: Mapping from logical qubit index to index within its patch.

    Returns:
        True if the two qubits are aligned for transversal ops (i.e., same index in patch).
    """
    return idx_in_patch[q1] == idx_in_patch[q2]

def verbose_log(verbose: bool, message: str):
    """
    Prints a message if verbosity is enabled.

    Args:
        verbose: Boolean flag for verbosity.
        message: String message to print.
    """
    if verbose:
        print(message)

def are_all_partners_zero(logical_idx: int, active_by_patch: Dict[int, List[int]], logical_to_patch: Dict[int, int], state_map: Dict[int, str]) -> bool:
    """
    Returns True if all active qubits in the same patch as logical_idx (excluding itself) are in the |0> state.
    """
    patch = logical_to_patch[logical_idx]
    return all(state_map[q] == "0" for q in active_by_patch[patch] if q != logical_idx)

def is_full_transversal(ctrls: List[int], targs: List[int], idx_in_patch: Dict[int, int]) -> bool:
    """
    Returns True if every control qubit has at least one target qubit with matching index in patch.
    """
    return all(any(idx_in_patch[q] == idx_in_patch[r] for r in targs) for q in ctrls)

def is_alone_in_patch(logical_idx: int, active_by_patch: Dict[int, List[int]], logical_to_patch: Dict[int, int]) -> bool:
    """
    Returns True if the given logical qubit is the only active one in its patch.
    """
    patch = logical_to_patch[logical_idx]
    return len(active_by_patch.get(patch, [])) == 1


def resolve_transversal_macro(op, context, label: str):
    """
    Validate a transversal macro against the live slot map and realign if needed.

    Returns `(control_patch, target_patch)` when the macro may fire, or None when it
    must be rejected. Shared by CXT, CZT and XCXT, which differ only in the physical
    construction they then call.

    Three conditions, in order of how badly they bite:

    1. *One patch per side.* The macro names logical qubits; a SWAP may since have
       scattered them.
    2. *Full coverage.* `*_k_Transversal` emits one physical gate per data qubit, so
       it is a logical gate on every lane. If the macro's lanes are not all the
       occupied lanes of both patches, the uncovered ones would receive a gate the
       program never asked for. This is what makes fusing a *subset* of lanes
       inexpressible: partial transversal is not a logical operation, since it sends
       the full-weight stabilizer `S_X = X^(x)n` outside the stabilizer group.
    3. *Alignment.* Lane i must face lane i; anything else is fixed by permuting the
       target patch with SWAPs.
    """
    metadata = getattr(op.gate, "metadata", {}) or {}
    control_count = int(metadata.get("control_count", max(1, len(op.qubits) // 2)))

    logical_qubits = metadata.get("logical_qubits")
    if logical_qubits is None:
        logical_qubits = [context.qc.find_bit(qubit).index for qubit in op.qubits]
    else:
        logical_qubits = [int(index) for index in logical_qubits]

    control_positions = current_positions_for_logicals(
        logical_qubits[:control_count], context
    )
    target_positions = current_positions_for_logicals(
        logical_qubits[control_count:], context
    )

    def reject(reason: str):
        message = (
            f"{label} macro {reason}: "
            f"controls={control_positions}, targets={target_positions}"
        )
        context.fault_log.append(message)
        verbose_log(context.verbose, f"[skip] {message}")
        return None

    control_patches = {patch for patch, _ in control_positions}
    target_patches = {patch for patch, _ in target_positions}
    if len(control_patches) != 1 or len(target_patches) != 1:
        return reject("spans multiple current patches")

    control_patch = next(iter(control_patches))
    target_patch = next(iter(target_patches))
    control_slots = [slot for _, slot in control_positions]
    target_slots = [slot for _, slot in target_positions]

    if not covers_active_lanes(context, control_patch, control_slots):
        return reject(f"does not cover every occupied lane of patch {control_patch}")
    if not covers_active_lanes(context, target_patch, target_slots):
        return reject(f"does not cover every occupied lane of patch {target_patch}")

    if control_slots == target_slots:
        return control_patch, target_patch

    # A prior gate may have permuted one patch's lanes; realign before the macro.
    if realign_target_slots(context, target_patch, control_slots, target_slots):
        return control_patch, target_patch

    return reject("logical lanes are misaligned with current positions")


def emit_targeted_cross_patch(context, emit, q_ctl: int, q_tgt: int) -> bool:
    """
    Emit a targeted cross-patch two-qubit gate at whatever slots the qubits occupy.

    The targeted construction touches only the two weight-2 logical supports, so it is
    correct for *any* pair of slots — pinned by
    `tests/test_iceberg_logical_action.py::test_cross_patch_targeted_ops_any_slot_pair`
    at n ∈ {4,6,8} in both bare and fault-tolerant modes. No alignment is required.

    `place_logical_cnot` / `_cz` / `_xcx` nonetheless each branch on slot equality, try an
    alignment SWAP, and *reject the placement* when neither `are_all_partners_zero` nor
    `is_alone_in_patch` nor `is_full_transversal` holds. That is vestigial: every one of
    their success branches ends in this same targeted call, and the `*_k_Transversal`
    invocation the alignment existed to enable is commented out in each of them. The
    conditions are the k=2 slot-equality artifact the iceberg layer already discarded.

    Keeping them cost real coverage. At k > 2 with a patch only partly occupied the three
    escape hatches routinely all miss — control patch holding slots {1,3} against a target
    patch holding {0,2} has no matching lane, no solo target and no all-`|0>` partner — so
    the gate was refused and the whole placement declared infeasible. It cost 3 of 15
    medium and 2-3 of 27 large QASMBench circuits at every k > 2, and it is why the
    code-size sweep's k > 2 columns were taken over a survivorship-selected subset.

    Applied to the **bare** path only. A logical-action test shows the ideal action is
    right; it cannot show that a gate preserves the code's error-detecting property, and
    `are_all_partners_zero` / `is_full_transversal` read like they are about not spreading
    errors between patches. So fault-tolerant mode keeps the original rules until someone
    checks that question properly.
    """
    patch_c, pos_c = context.current_pos_of(q_ctl)
    patch_t, pos_t = context.current_pos_of(q_tgt)
    emit(
        patch_c * context.k_log + pos_c,
        patch_t * context.k_log + pos_t,
        fault_tolerant=context.fault_tolerant,
    )
    return True


def expand_transversal_macro(op, context, single_handler, label: str) -> bool:
    """
    Undo a rejected transversal macro, one two-qubit gate per lane pair.

    `hh` has had this since the coverage rule was introduced
    (`_expand_to_single_hadamards`); CXT/CZT/XCXT did not, and simply returned False
    having emitted nothing. Combined with `translate_and_apply` discarding that False,
    a rejected macro deleted the program gates it stood in for.

    The pairing is `qubits[i]` with `qubits[control_count + i]`. `transversal.py` builds
    `macro_qargs` as `control_order + target_order` with each half sorted by slot, and a
    transversal group is a slot bijection, so equal rank means the pair the program
    actually asked for. `logical_qubits` in the metadata is derived from the same tuple,
    so the two orderings cannot disagree.

    Returns False if any of the expanded gates is itself rejected -- under the FT rules,
    say -- so the caller can still report failure rather than lose those gates in turn.
    """
    metadata = getattr(op.gate, "metadata", {}) or {}
    control_count = int(metadata.get("control_count", max(1, len(op.qubits) // 2)))
    controls = list(op.qubits[:control_count])
    targets = list(op.qubits[control_count:])

    applied = 0
    for control, target in zip(controls, targets):
        faux_op = type("TmpOp", (), {"qubits": [control, target], "gate": op.gate})
        if single_handler(faux_op, context) is False:
            context.fault_log.append(
                f"{label} macro expansion failed on lane pair {applied}; "
                f"{len(controls) - applied} gates not emitted."
            )
            return False
        applied += 1

    context.fault_log.append(
        f"{label} macro expanded into {applied} single gates."
    )
    return True


def current_positions_for_logicals(logical_qubits, context) -> List[tuple]:
    """Current `(patch, slot)` of each logical qubit, following any SWAPs."""
    return [
        context.current_pos_of(context.phys_index_for_logical(index))
        for index in logical_qubits
    ]


def realign_target_slots(
    context,
    target_patch: int,
    control_slots: List[int],
    target_slots: List[int],
) -> bool:
    """
    SWAP within `target_patch` so that target slot i sits on control slot i.

    Transversal macros pair lane i of the control patch with lane i of the target
    patch, so a prior gate that permuted one patch's lanes has to be undone before
    the macro fires. This is general in k: the required slot permutation is
    decomposed into transpositions by selection sort, each emitted as one logical
    SWAP. The k=2 code special-cased the single transposition that exists there.

    The desired slots need not be occupied. `resolve_transversal_macro` checks coverage
    before calling this, so `target_slots` is *every* occupied lane of `target_patch` and
    any other lane is free -- and swapping a qubit into a free lane simply moves it.

    That used to be refused. The old code required `sorted(control_slots) ==
    sorted(target_slots)`, i.e. that the two lists be a permutation of each other, which
    is only *necessary* when the patch is full. At k=6 with two lanes in use, control
    slots [2, 4] against target slots [0, 3] was rejected as unalignable even though
    lanes 2 and 4 of the target patch were empty and the move was two SWAPs away. At k=2
    an occupied patch is full, so the condition was always satisfied and the restriction
    never showed.

    Returns False (emitting nothing) only when the request is malformed: differing
    lengths, or a repeated slot on either side.
    """
    if len(control_slots) != len(target_slots):
        return False
    if len(set(control_slots)) != len(control_slots):
        return False
    if len(set(target_slots)) != len(target_slots):
        return False

    # position[state] = the slot that state currently occupies
    position = list(target_slots)
    for state, desired in enumerate(control_slots):
        current = position[state]
        if current == desired:
            continue
        # Whoever holds `desired`, if anyone. Read before the swap, since the swap is
        # what moves them to `current`. `None` means the lane is free, which is the case
        # the permutation condition used to rule out.
        occupant = position.index(desired) if desired in position else None

        # A SWAP is symmetric; emit the slots in ascending order so the recorded
        # operation is canonical regardless of which end we scanned from.
        low, high = sorted((current, desired))
        context.iceberg_set.SWAP(
            [[target_patch, low, high]], fault_tolerant=context.fault_tolerant
        )
        context.update_states_on_swap(target_patch, low, high)

        position[state] = desired
        if occupant is not None:
            position[occupant] = current
    return True
