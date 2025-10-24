"""
translator.handlers.h.py – Logical Hadamard (H, HH) handler (Qiskit)

Implements the rules for applying logical Hadamard gates on Iceberg codes,
including auto-merging adjacent H gates into a single FT HH gate when possible.
"""

from __future__ import annotations

from translator.context import TranslationContext
from translator.handlers.utils import is_patch_solo

ENABLE_HH_MERGE = False


def apply_logical_hadamard(
    op,
    context: TranslationContext,
    next_op=None,
) -> int | None:
    """
    Applies a logical Hadamard gate (H) or HH macro on an Iceberg Set.

    If the patch contains only one logical qubit, applies HH fault-tolerantly.
    Otherwise, attempts to merge with the next operation to form HH.
    Falls back to non-FT H with fault logging if needed.

    Args:
        op: The logical H operation (Qiskit op normalized by dispatcher; has .gate, .qubits).
        context: Shared translation state.
        next_op: Optional next operation for HH merge detection.

    Returns:
        If an HH merge occurred, returns the index of the consumed qubit.
        Otherwise, returns None.
    """
    # Resolve *program* (logical) index, then map through permutation and current swaps
    logical_i = context.qc.find_bit(op.qubits[0]).index
    p, s = context.permutation[logical_i]
    phys_id = p * context.k_log + s
    patch, slot = context.current_pos_of(phys_id)
    global_log_i = patch * context.k_log + slot

    # FIXME: This branch is broken because HH is not working I THINK
    if is_patch_solo(global_log_i, context.active_by_patch, context.logical_to_patch):
        context.iceberg_set.HH([patch])
        context.state_map[global_log_i] = "+"
        return None

    if ENABLE_HH_MERGE:
        pass
        # Try to merge H + H -> HH *only if both H's are in the same patch*
        if next_op and getattr(next_op.gate, "name", "").lower() == "h":
            logical_j = context.qc.find_bit(next_op.qubits[0]).index
            pj, sj = context.permutation[logical_j]
            phys_j = pj * context.k_log + sj
            patch_j, _ = context.current_pos_of(phys_j)
            if logical_j != logical_i and patch_j == patch:
                context.iceberg_set.HH([patch], fault_tolerant=context.fault_tolerant)
                context.state_map[global_log_i] = context.state_map[patch_j * context.k_log] = "+"
                return logical_j

    # Otherwise: always perform a *single-qubit* logical H at the resolved location
    # context.iceberg_set.H(global_log_i, fault_tolerant=context.fault_tolerant)
    context.iceberg_set.H_at(patch, slot, fault_tolerant=context.fault_tolerant)
    context.state_map[global_log_i] = "+"
    if not context.fault_tolerant:
        context.fault_log.append(f"non-FT H on logical {global_log_i} in patch {patch}")
    return None