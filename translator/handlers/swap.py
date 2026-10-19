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

    phys0 = context.phys_index_for_logical(q0)
    phys1 = context.phys_index_for_logical(q1)
    patch0, slot0 = context.current_pos_of(phys0)
    patch1, slot1 = context.current_pos_of(phys1)

    if patch0 != patch1:
        msg = f"SWAP({q0},{q1}) rejected – cross-patch SWAP not supported"
        context.fault_log.append(msg)
        verbose_log(context.verbose, f"[skip] {msg}")
        return False

    if slot0 == slot1:
        return True  # nothing to do

    iceberg_set: IcebergSet = context.iceberg_set
    iceberg_set.SWAP([[patch0, slot0, slot1]], fault_tolerant=context.fault_tolerant)

    context.update_states_on_swap(patch0, slot0, slot1)

    return True
