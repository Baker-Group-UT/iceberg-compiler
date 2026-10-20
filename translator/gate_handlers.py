'''
translator.gate_handlers.gate_handlers.py – Gate dispatch function (Qiskit).

Provides `translate_and_apply`, the main entrypoint used by all circuit translators
for applying fault-tolerant or faulty versions of logical gates on an Iceberg iceberg_set.
'''

from __future__ import annotations
from typing import Dict, Callable, Sequence
import math

from .context import TranslationContext
from .handlers.hadamard import apply_logical_hadamard, apply_logical_hh_merge
from .handlers.controlled_not import apply_logical_cnot, apply_logical_cxt
from .handlers.controlled_x import apply_logical_xcx, apply_logical_xcxt
from .handlers.controlled_z import apply_logical_cz, apply_logical_czt
from .handlers.rz import apply_logical_rz
from .handlers.rx import apply_logical_rx
from .handlers.s import apply_logical_s
from .handlers.sx import apply_logical_sx
from .handlers.x import apply_logical_x, apply_logical_xx_merge
from .handlers.z import apply_logical_z, apply_logical_zz_merge
from .handlers.swap import apply_logical_swap, apply_routing_swap

BARRIERS = False

# --- Simple shim so handlers can use op.gate and op.qubits like before ---
class _OpLike:
    def __init__(self, inst, qargs: Sequence, cargs: Sequence | None = None):
        self.gate = inst          # Qiskit Instruction
        self.qubits = list(qargs) # list[Qubit]
        self.cargs = list(cargs) if cargs else []


# --- Canonical gate translation table (simple one-to-one mappings) ---
# Use lowercase instruction names for robustness.
TABLE_SIMPLE: Dict[str, tuple[str, Callable]] = {
    "rx":   ("RX",   apply_logical_rx),
    "rz":   ("RZ",   apply_logical_rz),
    "s":    ("S",    apply_logical_s),
    "sx":   ("SX",   apply_logical_sx),
    "x":    ("X",    apply_logical_x),
    "z":    ("Z",    apply_logical_z),
    "swap": ("SWAP", apply_logical_swap),
    "routing_swap": ("SWAP", apply_routing_swap),
}

# --- Special compound gate handlers ---
TABLE_COMPLEX: Dict[str, Callable] = {
    "h":  apply_logical_hadamard,
    "hh": apply_logical_hh_merge,
    "xx": apply_logical_xx_merge,
    "zz": apply_logical_zz_merge,
    "cx": apply_logical_cnot,
    "cz": apply_logical_cz,
    "xcx": apply_logical_xcx,
    "xcxt": apply_logical_xcxt,
    "cxt": apply_logical_cxt,
    "czt": apply_logical_czt,
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
):
    """
    Translates and applies a logical operation onto the IcebergSet circuit
    using dispatch tables and specialized handlers.

    Args:
        context: TranslationContext holding mapping and circuit state.
        op: Qiskit (inst, qargs, cargs) triple from QuantumCircuit.data, or an _OpLike.

    Returns:
        False if the gate couldn't be applied, otherwise None.
    """
    if BARRIERS:
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

    # Handler protocol, for both tables below:
    #   None / True -> the gate was applied (or deliberately consumed)
    #   False       -> the gate could NOT be applied and nothing was emitted
    #
    # `False` has to reach the caller. `Translation.translate_circuit` raises on it, and
    # the pipeline's callers read that as "this placement is infeasible" -- which is how
    # the fault-tolerance rejection statistics are produced. Swallowing it means the
    # gate silently does not happen.
    #
    # It was swallowed here for every compound gate: this branch called the handler and
    # discarded the result. Since the merging pass replaces k program gates with one
    # macro and drops the originals, a macro the handler then rejected took those gates
    # with it and the translation still reported success. `simon_n6`-style programs came
    # out at fidelity 0.5 with nothing in the return value to say why. The guard in
    # `translate_circuit` was dead code for all ten names in TABLE_COMPLEX, `cx`, `cz`,
    # `xcx` and `h` among them.
    for table in (TABLE_COMPLEX, TABLE_SIMPLE):
        if gname not in table:
            continue
        entry = table[gname]
        handler = entry[1] if isinstance(entry, tuple) else entry
        return False if handler(op_like, context) is False else None

    raise ValueError(f"Unsupported logical gate type/name: {type(gate)} ('{gname}')")

    raise ValueError(f"Unsupported logical gate type/name: {type(gate)} ('{gname}')")
