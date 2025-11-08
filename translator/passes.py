"""
translator.passes.py - Shared Qiskit transpiler passes for circuit cleanup.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    XGate,
    ZGate,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation

__all__ = ["cancel_inverse_gates"]


_GATE_CANCELLATION_PASS = PassManager(
    InverseCancellation(
        gates_to_cancel=[
            XGate(),
            ZGate(),
            HGate(),
            CXGate(),
            CZGate(),
            (SGate(), SdgGate()),
            (SXGate(), SXdgGate()),
        ],
        run_default=True,
    )
)


def cancel_inverse_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Run a lightweight inverse-gate cancellation pass across the circuit.

    The pass removes back-to-back pairs of gates that undo each other, reducing
    redundant X/Z/S/H/CX/CZ/... instructions before further processing.
    """
    if circuit is None or circuit.num_qubits == 0 or len(circuit) == 0:
        return circuit
    return _GATE_CANCELLATION_PASS.run(circuit)
