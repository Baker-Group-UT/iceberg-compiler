"""
translator.handlers.rz.py - logic for applying a logical RZ.
"""

from __future__ import annotations
import math
import cirq

from iceberg.iceberg_set import IcebergSet
from translator.context import TranslationContext
from translator.handlers.utils import verbose_log

def apply_logical_rz(op: cirq.Operation, context: TranslationContext) -> bool:
    """
    Applies a logical RZ gate by decomposing it into an RZ and CXs.

    Args:
        op: The logical RZ operation.
        context: Translation state containing patch and qubit mappings.

    Returns:
        True if successfully placed.
    """
    qlog = op.qubits[0].x
    angle = float(op.gate.exponent) * math.pi
    context.iceberg_set.RZ(qlog, angle)
    return True
