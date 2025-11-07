"""
translator.handlers.h.py – Logical Hadamard (H, HH) handler (Qiskit)

Implements the rules for applying logical Hadamard gates on Iceberg codes.
HH merging is orchestrated by the translator preprocessing layer.
"""

from __future__ import annotations

from translator.context import TranslationContext
from translator.handlers.utils import is_patch_solo


def apply_logical_hadamard(
    op,
    context: TranslationContext,
) -> None:
    """
    Applies a logical Hadamard gate (H) or HH macro on an Iceberg Set.

    If the patch contains only one logical qubit, applies HH fault-tolerantly.
    Otherwise, falls back to non-FT H with fault logging if needed.

    Args:
        op: The logical H operation (Qiskit op normalized by dispatcher; has .gate, .qubits).
        context: Shared translation state.

    Returns:
        None. HH macro insertion is handled by preprocessing passes.
    """
    # Resolve *program* (logical) index, then map through permutation and current swaps
    logical_i = context.qc.find_bit(op.qubits[0]).index
    p, s = context.permutation[logical_i]
    phys_id = p * context.k_log + s
    patch, slot = context.current_pos_of(phys_id)
    global_log_i = patch * context.k_log + slot

    if is_patch_solo(global_log_i, context.active_by_patch, context.logical_to_patch):
        context.iceberg_set.HH([patch])
        context.state_map[global_log_i] = "+"
        return None

    # Otherwise: always perform a *single-qubit* logical H at the resolved location
    # context.iceberg_set.H(global_log_i, fault_tolerant=context.fault_tolerant)
    context.iceberg_set.H_at(patch, slot, fault_tolerant=context.fault_tolerant)
    context.state_map[global_log_i] = "+"
    if not context.fault_tolerant:
        context.fault_log.append(f"non-FT H on logical {global_log_i} in patch {patch}")
    return None


def apply_logical_hh_merge(
    op,
    context: TranslationContext,
) -> None:
    """
    Apply a pre-merged HH macro on the given patch.
    """
    logical_indices = [context.qc.find_bit(q).index for q in op.qubits]
    patch_ids = []
    global_indices = []
    for logical_i in logical_indices:
        p, s = context.permutation[logical_i]
        phys_id = p * context.k_log + s
        patch, slot = context.current_pos_of(phys_id)
        patch_ids.append(patch)
        global_indices.append(patch * context.k_log + slot)

    if len(set(patch_ids)) != 1:
        # Fallback: treat as two single H operations.
        for q in op.qubits:
            faux_op = type("TmpOp", (), {"qubits": [q], "gate": op.gate})
            apply_logical_hadamard(faux_op, context)
        context.fault_log.append("HH merge applied across multiple patches; fell back to single H gates.")
        return None

    patch = patch_ids[0]
    context.iceberg_set.HH([patch])
    for idx in global_indices:
        context.state_map[idx] = "+"
    return None
