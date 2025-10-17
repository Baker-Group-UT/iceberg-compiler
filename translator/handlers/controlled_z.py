"""
translator.handlers.cz_handler

Handler for logical CZ gate placement on Iceberg circuits (Qiskit).
Follows fault-tolerant (FT) placement rules:

1. If both qubits are in the same patch → standard in-patch CZ.
2. If in different patches:
   a. Allowed if either:
      • All partners in control patch are |0⟩, OR
      • Full transversal compatibility with target patch
   b. Also allowed if the target is the only active qubit in its patch
   → If indices misalign, perform SWAP on the free patch to align.

Falls back with logging if no valid FT placement exists.
"""

from __future__ import annotations
from translator.context import TranslationContext
from translator.handlers.utils import (
    are_all_partners_zero,
    is_full_transversal,
    is_alone_in_patch,
    verbose_log,
)


def apply_logical_cz(
    op,
    context: TranslationContext,
) -> bool:
    """
    Attempts to place a logical CZ gate under FT rules.

    Args:
        op: Qiskit op (normalized by dispatcher; has .qubits).
        context: TranslationContext with iceberg_set, patch, and state info.

    Returns:
        True if placement succeeds, False if gate must be skipped.
    """
    # Program (logical) indices
    q_ctl_log = context.qc.find_bit(op.qubits[0]).index
    q_tgt_log = context.qc.find_bit(op.qubits[1]).index

    # Map to stable physical ids via permutation: phys = p * k_log + s
    pc, sc = context.permutation[q_ctl_log]
    pt, st = context.permutation[q_tgt_log]
    q_ctl_phys = pc * context.k_log + sc
    q_tgt_phys = pt * context.k_log + st

    return place_logical_cz(q_ctl_phys, q_tgt_phys, context)


def place_logical_cz(
    q_ctl: int,
    q_tgt: int,
    context: TranslationContext,
) -> bool:
    """
    Core logic for placing a logical CZ gate under FT rules.
    Can be used independently from apply_logical_cz.

    Args:
        q_ctl: Physical index of control logical qubit.
        q_tgt: Physical index of target logical qubit.
        context: TranslationContext containing placement metadata.

    Returns:
        True if the CZ gate was placed, False otherwise.
    """
    iceberg_set = context.iceberg_set

    # --- CURRENT (dynamic) positions for decision-making ---
    patch_c, pos_c = context.current_pos_of(q_ctl)
    patch_t, pos_t = context.current_pos_of(q_tgt)

    # Intra-patch → standard CZ
    if patch_c == patch_t:
        iceberg_set.CZ(q_ctl, q_tgt, fault_tolerant=context.fault_tolerant)
        return True

    # Extract active physical indices (static lists)
    ctrls = context.active_by_patch[patch_c]
    targs = context.active_by_patch[patch_t]

    # Build a dynamic slot view for current positions
    slot_view = {i: context.current_slot_of(i) for i in set(ctrls + targs)}

    # FT eligibility
    control_ok = (
        are_all_partners_zero(q_ctl, context.active_by_patch, context.logical_to_patch, context.state_map)
        or is_full_transversal(ctrls, targs, slot_view)
    )
    target_alone = is_alone_in_patch(q_tgt, context.active_by_patch, context.logical_to_patch)

    # Already aligned → FT transversal CZ
    if pos_c == pos_t and (control_ok or target_alone):
        iceberg_set.CZ_k_Transversal(patch_c, patch_t)
        return True

    # Misaligned → try SWAP on target patch if solo
    if pos_c != pos_t and target_alone:
        iceberg_set.SWAP([[patch_t, pos_t, pos_c]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_t, pos_t, pos_c)

        # Verify alignment then do transversal CZ
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (target branch)."
        iceberg_set.CZ_k_Transversal(patch_c, patch_t)
        return True

    # Misaligned → try SWAP on control patch if control_ok
    if pos_c != pos_t and control_ok:
        iceberg_set.SWAP([[patch_c, pos_c, pos_t]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_c, pos_c, pos_t)

        # Verify alignment then do transversal CZ
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (control branch)."
        iceberg_set.CZ_k_Transversal(patch_c, patch_t)
        return True

    # Rejected
    msg = f"CZ({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False
