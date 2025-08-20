"""
translator.handlers.swap.py - logic for applying a logical swap.
"""

from __future__ import annotations
import cirq

from iceberg.iceberg_set import IcebergSet
from translator.context import TranslationContext
from translator.handlers.utils import verbose_log

def apply_logical_swap(op: cirq.Operation, context: TranslationContext) -> bool:
    """
    Applies a fault-tolerant SWAP between two logical qubits in the same patch.

    Args:
        op: The logical SWAP operation.
        context: Translation state with patch mappings and layout.

    Returns:
        True if the gate was placed; False otherwise.
    """
    q0, q1 = (q.x for q in op.qubits)
    patch0 = context.logical_to_patch[q0]
    patch1 = context.logical_to_patch[q1]

    if patch0 != patch1:
        msg = f"SWAP({q0},{q1}) rejected – cross-patch SWAP not supported"
        context.fault_log.append(msg)
        verbose_log(context.verbose, f"[skip] {msg}")
        return False

    # Call Iceberg Set Swap function and pass in positions and fault tolerance
    i0 = context.idx_in_patch[q0]
    i1 = context.idx_in_patch[q1]
    iceberg_set = context.iceberg_set
    iceberg_set.SWAP([[patch0, i0, i1]], faulty=context.faulty)

    return True
