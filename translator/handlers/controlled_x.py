"""
translator.handlers.controlled_x

Handler for logical XCX gate placement on Iceberg circuits (Qiskit).
Uses the same FT placement rules as CZ, but delegates to IcebergSet.XCX.
"""

from __future__ import annotations

from translator.context import TranslationContext
from translator.handlers.utils import (
    are_all_partners_zero,
    is_full_transversal,
    is_alone_in_patch,
    verbose_log,
)


def apply_logical_xcx(
    op,
    context: TranslationContext,
) -> bool:
    """
    Attempts to place a logical XCX gate under FT rules.

    Args:
        op: Qiskit op (normalized by dispatcher; has .qubits).
        context: TranslationContext with iceberg_set, patch, and state info.

    Returns:
        True if placement succeeds, False if gate must be skipped.
    """
    q_ctl_log = context.qc.find_bit(op.qubits[0]).index
    q_tgt_log = context.qc.find_bit(op.qubits[1]).index

    pc, sc = context.permutation[q_ctl_log]
    pt, st = context.permutation[q_tgt_log]
    q_ctl_phys = pc * context.k_log + sc
    q_tgt_phys = pt * context.k_log + st

    return place_logical_xcx(q_ctl_phys, q_tgt_phys, context)


def apply_logical_xcxt(op, context: TranslationContext) -> bool:
    """Apply a pre-merged transversal XCX macro (XCXTGate)."""

    control_patch, target_patch = _patch_pair_from_macro(op, context)
    context.iceberg_set.XCX_k_Transversal(control_patch, target_patch)
    return True


def place_logical_xcx(
    q_ctl: int,
    q_tgt: int,
    context: TranslationContext,
) -> bool:
    """
    Core logic for placing a logical XCX gate under FT rules.

    Args:
        q_ctl: Physical index of control logical qubit.
        q_tgt: Physical index of target logical qubit.
        context: TranslationContext containing placement metadata.

    Returns:
        True if the XCX gate was placed, False otherwise.
    """
    iceberg_set = context.iceberg_set

    patch_c, pos_c = context.current_pos_of(q_ctl)
    patch_t, pos_t = context.current_pos_of(q_tgt)

    if patch_c == patch_t:
        iceberg_set.XCX(q_ctl, q_tgt, fault_tolerant=context.fault_tolerant)
        return True

    ctrls = context.active_by_patch[patch_c]
    targs = context.active_by_patch[patch_t]
    slot_view = {i: context.current_slot_of(i) for i in set(ctrls + targs)}

    control_ok = (
        are_all_partners_zero(q_ctl, context.active_by_patch, context.logical_to_patch, context.state_map)
        or is_full_transversal(ctrls, targs, slot_view)
    )
    target_alone = is_alone_in_patch(q_tgt, context.active_by_patch, context.logical_to_patch)

    if pos_c == pos_t and (control_ok or target_alone):
        iceberg_set.XCX(patch_c * context.k_log + pos_c, patch_t * context.k_log + pos_t)
        return True

    if pos_c != pos_t and target_alone:
        swap_slots = (pos_t, pos_c)
        iceberg_set.SWAP([[patch_t, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_t, swap_slots[0], swap_slots[1])

        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (target branch)."
        iceberg_set.XCX(patch_c * context.k_log + pos_c2, patch_t * context.k_log + pos_t2)

        return True

    if pos_c != pos_t and control_ok:
        swap_slots = (pos_c, pos_t)
        iceberg_set.SWAP([[patch_c, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_c, swap_slots[0], swap_slots[1])

        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (control branch)."
        iceberg_set.XCX(patch_c * context.k_log + pos_c2, patch_t * context.k_log + pos_t2)

        return True

    msg = f"XCX({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False


def _patch_pair_from_macro(op, context: TranslationContext) -> tuple[int, int]:
    metadata = getattr(op.gate, "metadata", {}) or {}
    control_patch = metadata.get("control_patch")
    target_patch = metadata.get("target_patch")
    if control_patch is not None and target_patch is not None:
        return int(control_patch), int(target_patch)

    control_count = int(metadata.get("control_count", max(1, len(op.qubits) // 2)))
    control_qubit = op.qubits[0]
    target_qubit = op.qubits[control_count]
    control_logical = context.qc.find_bit(control_qubit).index
    target_logical = context.qc.find_bit(target_qubit).index
    return context.logical_to_patch[control_logical], context.logical_to_patch[target_logical]
