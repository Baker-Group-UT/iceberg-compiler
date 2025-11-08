"""
translator.handlers.cnot

Logical CNOT gate placement logic for Iceberg codes (Qiskit).
Handles both intra-patch and inter-patch scenarios with
fallbacks to SWAP-based transversal correction when possible.
"""

from __future__ import annotations

from translator.context import TranslationContext
from translator.handlers.utils import (
    are_all_partners_zero,
    is_full_transversal,
    is_alone_in_patch,
    verbose_log,
)


def apply_logical_cnot(
    op,
    context: TranslationContext,
) -> bool:
    """
    Applies a logical CNOT gate from a Qiskit operation.

    Args:
        op: Qiskit op (typically normalized by dispatcher; has .qubits).
        context: TranslationContext with circuit and patch state.

    Returns:
        True if placement succeeds, False otherwise.
    """
    # Program (logical) indices
    q_ctl_log = context.qc.find_bit(op.qubits[0]).index
    q_tgt_log = context.qc.find_bit(op.qubits[1]).index

    # Map to stable physical ids via permutation: phys = p * k_log + s
    pc, sc = context.permutation[q_ctl_log]
    pt, st = context.permutation[q_tgt_log]
    q_ctl_phys = pc * context.k_log + sc
    q_tgt_phys = pt * context.k_log + st

    return place_logical_cnot(q_ctl_phys, q_tgt_phys, context)


def place_logical_cnot(
    q_ctl: int,
    q_tgt: int,
    context: TranslationContext,
) -> bool:
    """
    Places a logical CNOT gate under FT rules given control and target *physical* indices.

    Args:
        q_ctl: Physical index of control qubit in the program-translated circuit.
        q_tgt: Physical index of target qubit in the program-translated circuit.
        context: Translation context containing static mapping (immutable) and dynamic state view.

    Returns:
        True if placement succeeded, False otherwise.
    """
    iceberg_set = context.iceberg_set

    # --- CURRENT (dynamic) positions for decision-making ---
    patch_c, pos_c = context.current_pos_of(q_ctl)
    patch_t, pos_t = context.current_pos_of(q_tgt)

    # Intra-patch CNOT (always valid)
    if patch_c == patch_t:
        iceberg_set.CNOT(patch_c * context.k_log + pos_c, patch_t * context.k_log + pos_t, fault_tolerant=context.fault_tolerant)
        return True

    # Extract physical indices considered active in each patch (static list of indices)
    ctrls = context.active_by_patch[patch_c]
    targs = context.active_by_patch[patch_t]

    # Build a lightweight slot view reflecting the *current* (dynamic) slot of any qubit we test
    slot_view = {i: context.current_slot_of(i) for i in set(ctrls + targs)}

    # Cross-patch constraints
    control_ok = (
        are_all_partners_zero(
            q_ctl, context.active_by_patch, context.logical_to_patch, context.state_map
        )
        or is_full_transversal(ctrls, targs, slot_view)
    )
    target_alone = is_alone_in_patch(q_tgt, context.active_by_patch, context.logical_to_patch)

    # Aligned for transversal
    if pos_c == pos_t and (control_ok or target_alone):
        # iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        iceberg_set.CNOT(patch_c * context.k_log + pos_c, patch_t * context.k_log + pos_t)
        return True

    # Misaligned → try SWAP on target patch if solo
    if pos_c != pos_t and target_alone:
        swap_slots = (pos_t, pos_c)
        iceberg_set.SWAP([[patch_t, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_t, swap_slots[0], swap_slots[1])

        # Re-read current positions to ensure alignment before transversal
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (target branch)."
        iceberg_set.CNOT(patch_c * context.k_log + pos_c2, patch_t * context.k_log + pos_t2)

        # Swap back so the logical ordering matches the original program
        iceberg_set.SWAP([[patch_t, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_t, swap_slots[0], swap_slots[1])

        return True

    # Misaligned → try SWAP on control patch if compatible
    if pos_c != pos_t and control_ok:
        swap_slots = (pos_c, pos_t)
        iceberg_set.SWAP([[patch_c, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_c, swap_slots[0], swap_slots[1])

        # Re-read positions to ensure alignment
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (control branch)."

        iceberg_set.CNOT(patch_c * context.k_log + pos_c2, patch_t * context.k_log + pos_t2)

        iceberg_set.SWAP([[patch_c, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_c, swap_slots[0], swap_slots[1])

        return True

    # Placement rejected under FT rules
    msg = f"CNOT({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False


def apply_logical_cxt(op, context: TranslationContext) -> bool:
    """Apply a pre-merged transversal CNOT macro (CXTGate)."""

    control_patch, target_patch = _patch_pair_from_macro(op, context)
    context.iceberg_set.CNOT_k_Transversal(control_patch, target_patch)
    return True


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
