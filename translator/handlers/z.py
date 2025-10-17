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
    patch, slot = context.current_pos_of(phys_id)

    # Resolve to (patch, slot) after any logical swaps
    patch, slot = context.current_pos_of(phys_id)

    # Apply logical Z at the resolved physical location
    context.iceberg_set.Z_at(patch, slot)

    if getattr(context, "verbose", False):
        context.fault_log.append(f"[trace] Z on logical {phys_id} -> (p{patch},s{slot})")

    return True
