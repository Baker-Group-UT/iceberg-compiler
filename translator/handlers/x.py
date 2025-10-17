"""
translator.handlers.x.py - logic for applying a logical X.
"""

from __future__ import annotations

from translator.context import TranslationContext

def apply_logical_x(op, context: TranslationContext) -> bool:
    """
    Applies a logical X gate to the *current* position of the given logical qubit.

    This resolves the target using the dynamic tracker (which is updated by SWAPs),
    so that X always lands on the qubit that the logical wire refers to *now*,
    not where it started before alignment moves.

    Args:
        op: The logical X operation.
        context: Translation state containing patch and qubit mappings.

    Returns:
        True if successfully placed.
    """
    # Get the flat qubit index (Qiskit equivalent of Cirq’s qubit.x)
    logical_i = context.qc.find_bit(op.qubits[0]).index
    phys_id   = context.phys_index_for_logical(logical_i)
    patch, slot = context.current_pos_of(phys_id)

    # Current (patch, slot) after any SWAPs performed to align transversals
    patch, slot = context.current_pos_of(phys_id)

    # Apply logical X at the *current* resolved location
    context.iceberg_set.X_at(patch, slot)

    # (Optional verbose trace)
    if getattr(context, "verbose", False):
        context.fault_log.append(f"[trace] X on logical {phys_id} -> (p{patch},s{slot})")

    return True
