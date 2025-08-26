'''
translator.gate_handlers.gate_handlers.py – Gate dispatch function.

Provides `translate_and_apply`, the main entrypoint used by all circuit translators
for applying fault-tolerant or faulty versions of logical gates on an Iceberg iceberg_set.
'''

from __future__ import annotations
from typing import Dict, Callable
import math
import cirq

from cirq.ops import (
    Rz,
    HPowGate,
    ZPowGate,
    SwapPowGate,
    CXPowGate,
    CZPowGate,
)
from cirq.ops import MeasurementGate
from iceberg.gates import BarrierGate

from .context import TranslationContext
from .handlers.hadamard import apply_logical_hadamard
from .handlers.controlled_not import apply_logical_cnot
from .handlers.controlled_z import apply_logical_cz
from .handlers.rz import apply_logical_rz
from .handlers.swap import apply_logical_swap

# --- Canonical gate translation table (simple one-to-one mappings) ---
TABLE_SIMPLE: Dict[type, tuple[str, Callable]] = {
    cirq.X:         ("X", lambda ctx, q: ctx.iceberg_set.X(q)),
    cirq.Z:         ("Z", lambda ctx, q: ctx.iceberg_set.Z(q)),
    Rz:             ("RZ", apply_logical_rz),
    ZPowGate:       ("RZ", apply_logical_rz),
    SwapPowGate:    ("SWAP", apply_logical_swap),
}

# --- Special compound gate handlers ---
TABLE_COMPLEX: Dict[type, Callable] = {
    HPowGate:       apply_logical_hadamard,
    CXPowGate:      apply_logical_cnot,
    CZPowGate:      apply_logical_cz,
}

SKIPPED_GATES = (BarrierGate, MeasurementGate)

def _rz_angle_from_gate(gate) -> float:
    """Extracts rotation angle in radians from Rz/ZPowGate."""
    if hasattr(gate, "rads"):
        return float(gate.rads)
    if hasattr(gate, "_rads"):
        return float(gate._rads)
    if isinstance(gate, ZPowGate):
        return float(gate.exponent) * math.pi
    raise TypeError(f"Unsupported RZ gate type: {type(gate)}")

def translate_and_apply(
    context: TranslationContext,
    op: cirq.Operation,
    *,
    next_op: cirq.Operation | None = None,
):
    """
    Translates and applies a logical operation onto the IcebergSet circuit
    using dispatch tables and specialized handlers.

    Args:
        context: TranslationContext holding mapping and circuit state.
        op: Cirq Operation to be translated.
        next_op: Optional lookahead operation (e.g., for detecting HH).

    Returns:
        True if a lookahead gate was consumed, otherwise None.
    """
    gate = op.gate
    g_type = type(gate)

    if isinstance(gate, MeasurementGate) or repr(gate) == "BarrierGate()":
        return None

    # --- Check for HH compound gate ---
    if isinstance(gate, HPowGate) and next_op and isinstance(next_op.gate, HPowGate):
        apply_logical_hadamard(op, context, next_op)
        return True

    # --- Special compound handlers ---
    if g_type in TABLE_COMPLEX:
        TABLE_COMPLEX[g_type](op, context)
        return None

    # --- Simple one-to-one handlers ---
    if g_type in TABLE_SIMPLE:
        label, handler = TABLE_SIMPLE[g_type]
        if label in {"X", "Z"}:
            q = op.qubits[0].x
            handler(context, q)
        else:
            handler(op, context)
        return None

    raise ValueError(f"Unsupported logical gate type: {g_type}")
