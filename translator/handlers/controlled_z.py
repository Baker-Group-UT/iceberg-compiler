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
        # iceberg_set.CZ_k_Transversal(patch_c, patch_t)
        iceberg_set.CZ(patch_c * context.k_log + pos_c, patch_t * context.k_log + pos_t)
        return True

    # Misaligned → try SWAP on target patch if solo
    if pos_c != pos_t and target_alone:
        swap_slots = (pos_t, pos_c)
        iceberg_set.SWAP([[patch_t, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_t, swap_slots[0], swap_slots[1])

        # Verify alignment then do transversal CZ
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (target branch)."
        iceberg_set.CZ(patch_c * context.k_log + pos_c2, patch_t * context.k_log + pos_t2)

        return True

    # Misaligned → try SWAP on control patch if control_ok
    if pos_c != pos_t and control_ok:
        swap_slots = (pos_c, pos_t)
        iceberg_set.SWAP([[patch_c, swap_slots[0], swap_slots[1]]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(patch_c, swap_slots[0], swap_slots[1])

        # Verify alignment then do transversal CZ
        _, pos_c2 = context.current_pos_of(q_ctl)
        _, pos_t2 = context.current_pos_of(q_tgt)
        assert pos_c2 == pos_t2, "Post-SWAP misalignment before transversal (control branch)."
        
        iceberg_set.CZ(patch_c * context.k_log + pos_c2, patch_t * context.k_log + pos_t2)

        return True

    # Rejected
    msg = f"CZ({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False


def apply_logical_czt(op, context: TranslationContext) -> bool:
    """Apply a pre-merged transversal CZ macro (CZTGate)."""

    resolved = _resolve_czt_macro_alignment(op, context)
    if resolved is None:
        return False
    control_patch, target_patch = resolved
    context.iceberg_set.CZ_k_Transversal(control_patch, target_patch)
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


def _resolve_czt_macro_alignment(
    op,
    context: TranslationContext,
) -> tuple[int, int] | None:
    metadata = getattr(op.gate, "metadata", {}) or {}
    control_count = int(metadata.get("control_count", max(1, len(op.qubits) // 2)))

    logical_qubits = metadata.get("logical_qubits")
    if logical_qubits is None:
        logical_qubits = [context.qc.find_bit(qubit).index for qubit in op.qubits]
    else:
        logical_qubits = [int(logical_i) for logical_i in logical_qubits]

    control_logicals = logical_qubits[:control_count]
    target_logicals = logical_qubits[control_count:]
    control_positions = _current_positions_for_logicals(control_logicals, context)
    target_positions = _current_positions_for_logicals(target_logicals, context)

    control_patches = {patch for patch, _ in control_positions}
    target_patches = {patch for patch, _ in target_positions}
    if len(control_patches) != 1 or len(target_patches) != 1:
        msg = f"CZT macro spans multiple current patches: controls={control_positions}, targets={target_positions}"
        context.fault_log.append(msg)
        verbose_log(context.verbose, f"[skip] {msg}")
        return None

    control_patch = next(iter(control_patches))
    target_patch = next(iter(target_patches))
    control_slots = [slot for _, slot in control_positions]
    target_slots = [slot for _, slot in target_positions]

    if control_slots == target_slots:
        return control_patch, target_patch

    if (
        context.k_log == 2
        and len(control_slots) == len(target_slots) == 2
        and sorted(control_slots) == [0, 1]
        and sorted(target_slots) == [0, 1]
        and [1 - slot for slot in target_slots] == control_slots
    ):
        context.iceberg_set.SWAP([[target_patch, 0, 1]], fault_tolerant=context.fault_tolerant)
        context.update_states_on_swap(target_patch, 0, 1)
        return control_patch, target_patch

    msg = (
        "CZT macro logical lanes are misaligned with current positions: "
        f"controls={control_positions}, targets={target_positions}"
    )
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return None


def _current_positions_for_logicals(
    logical_qubits: list[int],
    context: TranslationContext,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for logical_i in logical_qubits:
        phys_id = context.phys_index_for_logical(logical_i)
        positions.append(context.current_pos_of(phys_id))
    return positions
