'''
translator.gate_handlers.gate_handlers.py – Gate dispatch function (Qiskit).

Provides `translate_and_apply`, the main entrypoint used by all circuit translators
for applying fault-tolerant or faulty versions of logical gates on an Iceberg iceberg_set.
'''

from __future__ import annotations
from typing import Dict, Callable, Optional, Sequence
import math

# Qiskit gate names we’ll dispatch on:
#  - simple:  'x', 'z', 'rz', 'swap'
#  - complex: 'h', 'cx', 'cz'
#  - skipped: 'barrier', 'measure'

from .context import TranslationContext
from .handlers.hadamard import apply_logical_hadamard
from .handlers.controlled_not import apply_logical_cnot
from .handlers.controlled_z import apply_logical_cz
from .handlers.rz import apply_logical_rz
from .handlers.x import apply_logical_x
from .handlers.z import apply_logical_z
from .handlers.swap import apply_logical_swap


# --- Simple shim so handlers can use op.gate and op.qubits like before ---
class _OpLike:
    def __init__(self, inst, qargs: Sequence, cargs: Sequence | None = None):
        self.gate = inst          # Qiskit Instruction
        self.qubits = list(qargs) # list[Qubit]
        self.cargs = list(cargs) if cargs else []


# --- Canonical gate translation table (simple one-to-one mappings) ---
# Use lowercase instruction names for robustness.
TABLE_SIMPLE: Dict[str, tuple[str, Callable]] = {
    "x":    ("X",    apply_logical_x),
    "z":    ("Z",    apply_logical_z),
    "rz":   ("RZ",   apply_logical_rz),
    "swap": ("SWAP", apply_logical_swap),
}

# --- Special compound gate handlers ---
TABLE_COMPLEX: Dict[str, Callable] = {
    "h":  apply_logical_hadamard,
    "cx": apply_logical_cnot,
    "cz": apply_logical_cz,
}

SKIPPED_NAMES = {"barrier", "measure"}


def _rz_angle_from_gate(inst) -> float:
    """Extract rotation angle in radians from a Qiskit RZ or Z instruction."""
    name = inst.name.lower()
    if name == "rz":
        # params[0] is the angle in radians
        return float(inst.params[0])
    if name == "z":
        # Z is equivalent to RZ(pi) up to phase
        return math.pi
    raise TypeError(f"Unsupported RZ gate type: {type(inst)} / name='{name}'")


def translate_and_apply(
    context: TranslationContext,
    op,  # Accept either (inst, qargs, cargs) tuple from qc.data OR an _OpLike
    *,
    next_op: Optional[object] = None,  # optional lookahead; same normalization below if used
):
    """
    Translates and applies a logical operation onto the IcebergSet circuit
    using dispatch tables and specialized handlers.

    Args:
        context: TranslationContext holding mapping and circuit state.
        op: Qiskit (inst, qargs, cargs) triple from QuantumCircuit.data, or an _OpLike.
        next_op: Optional lookahead (same accepted shapes) — currently unused.

    Returns:
        True if a lookahead gate was consumed, otherwise None.
        False if the gate couldn't be applied
    """
    context.iceberg_set.BARRIER()
    # before the tuple/_OpLike checks
    if hasattr(op, "operation") and hasattr(op, "qubits"):  # CircuitInstruction (Qiskit 1.0+)
        inst = op.operation
        qargs = op.qubits
        cargs = getattr(op, "clbits", ())
        op_like = _OpLike(inst, qargs, cargs)
    elif isinstance(op, tuple) and len(op) >= 2:
        inst, qargs = op[0], op[1]
        cargs = op[2] if len(op) > 2 else []
        op_like = _OpLike(inst, qargs, cargs)
    elif isinstance(op, _OpLike):
        op_like = op
    else:
        raise TypeError("translate_and_apply expected CircuitInstruction, (inst,qargs,cargs) or _OpLike")

    gate = op_like.gate
    gname = getattr(gate, "name", "").lower()

    # Skip measurements and barriers
    if gname in SKIPPED_NAMES:
        return None

    # --- Special compound handlers ---
    if gname in TABLE_COMPLEX:
        TABLE_COMPLEX[gname](op_like, context)
        return None

    # FIXME: HH lookahead merging broken
    # # --- Check for HH compound gate ---
    # # None = Success, True = Gate consumed, False = Gate application failed
    # if isinstance(gate, HPowGate) and next_op and isinstance(next_op.gate, HPowGate):
    #     apply_logical_hadamard(op, context, next_op)
    #     return True

    # --- Simple one-to-one handlers ---
    if gname in TABLE_SIMPLE:
        _, handler = TABLE_SIMPLE[gname]
        flag = handler(op_like, context)
        # This looks stupid but handler mostly returns None
        # None = Success, True = Gate consumed, False = Gate application failed
        if flag is False:
            return False
        return None

    raise ValueError(f"Unsupported logical gate type/name: {type(gate)} ('{gname}')")
