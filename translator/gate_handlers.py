# translator/gate_handlers/gate_handlers.py

from __future__ import annotations
from typing import Dict, Callable
import math
import cirq

from cirq.ops import (
    XPowGate,
    Rz,
    HPowGate,
    ZPowGate,
    SwapPowGate,
    CXPowGate,
    CZPowGate,
    MeasurementGate,
)
from iceberg.gates import BarrierGate

from .context import TranslationContext
from .handlers.hadamard import apply_logical_hadamard  # kept, but we’ll call module-level H here
from .handlers.controlled_not import apply_logical_cnot
from .handlers.controlled_z import apply_logical_cz
from .handlers.rz import apply_logical_rz            # update this to use current slot (see note below)
from .handlers.swap import apply_logical_swap

PAULI_X_TYPE = type(cirq.X)
PAULI_Z_TYPE = type(cirq.Z)

SKIPPED_GATES = (BarrierGate, MeasurementGate)

def _rz_angle_from_gate(gate) -> float:
    if hasattr(gate, "rads"):
        return float(gate.rads)
    if hasattr(gate, "_rads"):
        return float(gate._rads)
    if isinstance(gate, ZPowGate):
        return float(gate.exponent) * math.pi
    raise TypeError(f"Unsupported RZ gate type: {type(gate)}")

def _current_patch_slot(context: TranslationContext, q_phys: int) -> tuple[int, int]:
    """Resolve the *current* (patch, slot) for a program physical index."""
    return context.current_pos_of(q_phys)

def translate_and_apply(
    context: TranslationContext,
    op: cirq.Operation,
    *,
    next_op: cirq.Operation | None = None,
):
    """
    Translates and applies a logical operation using the *current* address of each qubit.
    Returns True if a lookahead gate was consumed, otherwise None.
    """
    gate = op.gate

    # Skip barriers/measurements at translation time
    if isinstance(gate, MeasurementGate) or repr(gate) == "BarrierGate()":
        return None

    # -------- Single-qubit gates: always use current (patch,slot) --------
    if isinstance(gate, (XPowGate, ZPowGate, Rz, HPowGate)):
        q_phys = op.qubits[0].x
        patch, slot = _current_patch_slot(context, q_phys)
        mod = context.iceberg_set.modules[patch]

        if isinstance(gate, XPowGate):
            # Apply logical X to the *current* slot
            mod.X(slot)
            return None

        if isinstance(gate, ZPowGate):
            mod.Z(slot)
            return None

        if isinstance(gate, Rz) or isinstance(gate, ZPowGate):
            # If you want RZ support via ZPowGate exponent as well:
            theta = _rz_angle_from_gate(gate)
            mod.RZ(slot, theta, fault_tolerant=context.fault_tolerant)
            return None

        if isinstance(gate, HPowGate):
            # HH merge disabled; do single-qubit H at the *current* slot
            mod.H(slot, fault_tolerant=context.fault_tolerant)
            context.state_map[q_phys] = "+"
            if not context.fault_tolerant:
                context.fault_log.append(f"non-FT H on logical {q_phys} in patch {patch}")
            return None

    # -------- Two-qubit gates: delegate, but handlers must use current positions --------
    if isinstance(gate, CXPowGate):
        # apply_logical_cnot already reads current_pos_of(...) internally
        apply_logical_cnot(op, context)
        return None

    if isinstance(gate, CZPowGate):
        apply_logical_cz(op, context)
        return None

    if isinstance(gate, SwapPowGate):
        # Your SWAP handler must call context.update_states_on_swap(...)
        apply_logical_swap(op, context)
        return None

    raise ValueError(f"Unsupported logical gate type: {type(gate)}")