"""
translator.handlers.z.py – logic for applying a logical Z (Qiskit version).
"""

from __future__ import annotations
from translator.context import TranslationContext


def apply_logical_z(op, context: TranslationContext) -> bool:
    """
    Applies a logical Z gate to the *current* position of the given logical qubit.

    This resolves the target using the dynamic tracker (which is updated by SWAPs),
    so that Z always lands on the qubit that the logical wire refers to *now*,
    not where it started before alignment moves.

    Args:
        op: The logical Z operation.
        context: Translation state containing patch and qubit mappings.

    Returns:
        True if successfully placed.
    """
    # Get the flat qubit index
    logical_i = context.qc.find_bit(op.qubits[0]).index
    phys_id   = context.phys_index_for_logical(logical_i)

    # Resolve to (patch, slot) after any logical swaps
    patch, slot = context.current_pos_of(phys_id)

    # Apply logical Z at the resolved physical location
    context.iceberg_set.Z_at(patch, slot)

    if getattr(context, "verbose", False):
        context.fault_log.append(f"[trace] Z on logical {phys_id} -> (p{patch},s{slot})")

    return True


def apply_logical_zz_merge(op, context: TranslationContext) -> None:
    """
    Apply a pre-merged ZZ macro on the given patch.
    """
    logical_indices = [context.qc.find_bit(q).index for q in op.qubits]
    patch_ids = []
    slots = []
    global_indices = []
    for logical_i in logical_indices:
        p, s = context.permutation[logical_i]
        phys_id = p * context.k_log + s
        patch, slot = context.current_pos_of(phys_id)
        patch_ids.append(patch)
        slots.append(slot)
        global_indices.append(patch * context.k_log + slot)

    if len(set(patch_ids)) != 1:
        # Fallback: treat as two single Z operations.
        for q in op.qubits:
            faux_op = type("TmpOp", (), {"qubits": [q], "gate": op.gate})
            apply_logical_z(faux_op, context)
        context.fault_log.append("ZZ merge applied across multiple patches; fell back to single Z gates.")
        return None

    patch = patch_ids[0]
    context.iceberg_set.Z_at(patch, slots)
    return None
