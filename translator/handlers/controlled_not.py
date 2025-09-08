"""
translator.handlers.cnot

Logical CNOT gate placement logic for Iceberg codes.
Handles both intra-patch and inter-patch scenarios with
fallbacks to SWAP-based transversal correction when possible.
"""

from __future__ import annotations
import cirq

from translator.context import TranslationContext
from translator.handlers.utils import (
    are_all_partners_zero,
    is_full_transversal,
    is_alone_in_patch,
    verbose_log,
)

def apply_logical_cnot(
    op: cirq.Operation,
    context: TranslationContext,
) -> bool:
    """
    Applies a logical CNOT gate from a Cirq operation.

    Args:
        op: Cirq Operation for logical CNOT.
        context: TranslationContext with circuit and patch state.

    Returns:
        True if placement succeeds, False otherwise.
    """
    q_ctl = op.qubits[0].x
    q_tgt = op.qubits[1].x
    return place_logical_cnot(q_ctl, q_tgt, context)

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
        iceberg_set.CNOT(q_ctl, q_tgt, fault_tolerant=context.fault_tolerant)
        return True

    # Extract physical indices considered active in each patch (static list of indices)
    ctrls = context.active_by_patch[patch_c]
    targs = context.active_by_patch[patch_t]

    # Build a lightweight slot view reflecting the *current* (dynamic) slot of any qubit we test
    # (If is_full_transversal accepts a callable instead of a dict, you can pass context.current_slot_of)
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
        iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # Misaligned → try SWAP on target patch if solo
    if pos_c != pos_t and target_alone:
        iceberg_set.SWAP([[patch_t, pos_t, pos_c]], fault_tolerant=context.fault_tolerant)
        # Update dynamic state tracker (permutation-like)
        context.update_states_on_swap(patch_t, pos_t, pos_c)

        # Re-read current positions to ensure alignment before transversal
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (target branch)."
        iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # Misaligned → try SWAP on control patch if compatible
    if pos_c != pos_t and control_ok:
        iceberg_set.SWAP([[patch_c, pos_c, pos_t]], fault_tolerant=context.fault_tolerant)
        # Update dynamic state tracker
        context.update_states_on_swap(patch_c, pos_c, pos_t)

        # Re-read positions to ensure alignment
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (control branch)."

        iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # Placement rejected under FT rules
    msg = f"CNOT({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False