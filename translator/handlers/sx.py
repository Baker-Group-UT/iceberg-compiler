"""
translator.handlers.sx.py - logic for applying a logical SX (Qiskit).
"""

from __future__ import annotations

from translator.context import TranslationContext
from translator.handlers.utils import verbose_log


def apply_logical_sx(op, context: TranslationContext) -> bool:
    """
    Applies a logical SX gate at the *current* location of the logical qubit.
    """
    logical_i = context.qc.find_bit(op.qubits[0]).index
    phys_id = context.phys_index_for_logical(logical_i)
    patch, slot = context.current_pos_of(phys_id)
    context.iceberg_set.SX_at(patch, slot, fault_tolerant=context.fault_tolerant)

    if getattr(context, "verbose", False):
        context.fault_log.append(
            f"[trace] SX on logical {logical_i} -> phys_id {phys_id} "
            f"-> (p{patch},s{slot})"
        )
        verbose_log(
            context.verbose,
            f"[trace] SX applied at (patch={patch}, slot={slot})",
        )

    return True
