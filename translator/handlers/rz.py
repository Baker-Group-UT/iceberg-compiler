"""
translator.handlers.rz.py - logic for applying a logical RZ (Qiskit).
"""

from __future__ import annotations
import math

from translator.context import TranslationContext
from translator.handlers.utils import verbose_log


def apply_logical_rz(op, context: TranslationContext) -> bool:
    """
    Applies a logical RZ gate at the *current* location of the logical qubit.

    Steps:
      1) Get the program (logical) qubit index from the program circuit.
      2) Map to a stable physical id via the permutation.
      3) Resolve the current (patch, slot) (after any SWAPs).
      4) Convert (patch, slot) -> global logical index and apply RZ there.
    """
    # 1) program/logical index
    logical_i = context.qc.find_bit(op.qubits[0]).index

    # 2) stable physical id from the permutation (p * k_log + s)
    p, s = context.permutation[logical_i]
    phys_id = p * context.k_log + s

    # 3) current (patch, slot) after any in-translation SWAPs
    patch, slot = context.current_pos_of(phys_id)

    # 4) convert to global logical index and apply RZ
    global_log_i = patch * context.k_log + slot

    # Angle from Qiskit instruction
    name = getattr(op.gate, "name", "").lower()
    if name == "rz":
        angle = float(op.gate.params[0])  # radians
    elif name == "z":
        angle = math.pi
    else:
        raise TypeError(f"apply_logical_rz received unsupported gate '{name}'")

    context.iceberg_set.RZ(global_log_i, angle)

    if getattr(context, "verbose", False):
        context.fault_log.append(
            f"[trace] RZ({angle:.6f}) on logical {logical_i} "
            f"-> phys_id {phys_id} -> (p{patch},s{slot}) -> global {global_log_i}"
        )
        verbose_log(
            context.verbose,
            f"[trace] RZ applied at (patch={patch}, slot={slot}) with θ={angle:.6f}",
        )

    return True
