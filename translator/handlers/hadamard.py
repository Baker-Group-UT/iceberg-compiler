"""
translator.handlers.h.py – Logical Hadamard (H, HH) handler

Implements the rules for applying logical Hadamard gates on Iceberg codes,
including auto-merging adjacent H gates into a single FT HH gate when possible.
"""

from __future__ import annotations
import cirq

from translator.context import TranslationContext
from translator.handlers.utils import is_patch_solo

ENABLE_HH_MERGE = False

def apply_logical_hadamard(
    op: cirq.Operation,
    context: TranslationContext,
    next_op: cirq.Operation | None = None,
) -> int | None:
    """
    Applies a logical Hadamard gate (H) or HH macro on an Iceberg Set.

    If the patch contains only one logical qubit, applies HH fault-tolerantly.
    Otherwise, attempts to merge with the next operation to form HH.
    Falls back to non-FT H with fault logging if needed.

    Args:
        op: The logical H operation.
        context: Shared translation state.
        next_op: Optional next operation for HH merge detection.

    Returns:
        If an HH merge occurred, returns the index of the consumed qubit.
        Otherwise, returns None.
    """
    q = op.qubits[0].x
    patch, _ = context.current_pos_of(q)

    # FIXME: This branch is broken because HH is not working I THINK
    if is_patch_solo(q, context.active_by_patch, context.logical_to_patch):
        context.iceberg_set.HH([patch])
        context.state_map[q] = "+"
        return None

    if ENABLE_HH_MERGE:
        pass
        # Try to merge H + H -> HH *only if both H's are in the same patch*
        # if (
        #     next_op and isinstance(next_op.gate, cirq.HPowGate)
        #     and abs(next_op.gate.exponent) % 2 == 1 and next_op.gate.global_shift == 0
        # ):
        #     q2 = next_op.qubits[0].x
        #     patch2, _ = context.current_pos_of(q2)
        #     if q2 != q and patch2 == patch:
        #         context.iceberg_set.HH([patch], fault_tolerant=context.fault_tolerant)
        #         context.state_map[q] = context.state_map[q2] = "+"
        #         return q2

    # Otherwise: always perform a *single-qubit* logical H
    context.iceberg_set.H(q, fault_tolerant=context.fault_tolerant)
    context.state_map[q] = "+"
    if not context.fault_tolerant:
        context.fault_log.append(f"non-FT H on logical {q} in patch {patch}")
    return None