"""
translator.handlers.swap.py - logic for applying a logical swap (Qiskit).
"""

from __future__ import annotations

from iceberg.iceberg_set import IcebergSet
from translator.context import TranslationContext
from translator.handlers.utils import verbose_log


def apply_logical_swap(op, context: TranslationContext) -> bool:
    """
    Applies a fault-tolerant SWAP between two logical qubits in the same patch.

    Args:
        op: The logical SWAP operation (Qiskit op: typically (inst, qargs, cargs) normalized by dispatcher).
        context: Translation state with patch mappings and layout.

    Returns:
        True if the gate was placed; False otherwise.
    """
    # Get logical (program) qubit indices from the program circuit
    q0 = context.qc.find_bit(op.qubits[0]).index
    q1 = context.qc.find_bit(op.qubits[1]).index

    patch0 = context.logical_to_patch[q0]
    patch1 = context.logical_to_patch[q1]

    if patch0 != patch1:
        msg = f"SWAP({q0},{q1}) rejected – cross-patch SWAP not supported"
        context.fault_log.append(msg)
        verbose_log(context.verbose, f"[skip] {msg}")
        return False

    # Indices within the patch
    i0 = context.idx_in_patch[q0]
    i1 = context.idx_in_patch[q1]

    iceberg_set: IcebergSet = context.iceberg_set
    iceberg_set.SWAP([[patch0, i0, i1]], fault_tolerant=context.fault_tolerant)

    # If you track dynamic positions, you can reflect the swap like this:
    # context.update_states_on_swap(patch0, i0, i1)

    return True