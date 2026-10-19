"""Custom Qiskit gate definitions used by the translator."""

from __future__ import annotations

from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister


class XCXGate(Gate):
    """
    X-controlled-X gate (H⊗H · CZ · H⊗H).

    Diagonal in the X eigenbasis of both qubits, with a -1 on |-,->.

    The definition below is what lets Qiskit simulate, transpile or build a
    tableau for a program containing this gate. Without it the gate is opaque and
    everything that needs a matrix — `Clifford`, `Operator`, Aer — fails on it,
    which is why logical programs using XCX could not previously be checked
    against the encoded circuit.

    `iceberg.IcebergModule.XCX` and `IcebergSet.XCX` implement the encoded
    version of exactly this unitary.
    """

    def __init__(self):
        super().__init__("xcx", 2, [])

    def _define(self):
        qubits = QuantumRegister(2, "q")
        definition = QuantumCircuit(qubits, name=self.name)
        definition.h(qubits[0])
        definition.h(qubits[1])
        definition.cz(qubits[0], qubits[1])
        definition.h(qubits[0])
        definition.h(qubits[1])
        self.definition = definition

    def inverse(self, annotated: bool = False):
        return XCXGate()


class RoutingSwapGate(Gate):
    """
    A SWAP inserted by the compiler to move data between lanes, not by the user.

    The distinction matters because the two have opposite implementations. A SWAP
    in the *program* exchanges the values of two program qubits, and the cheapest
    correct way to do that is to exchange their labels and emit nothing. A routing
    SWAP must physically move the data — that is its entire purpose, since the
    transversal macros pair lane i with lane i — and the labels follow the data,
    so its net effect on the program is the identity.

    `translator.merging.transversal` emits these around transversal macros when
    the two patches' lanes are crossed rather than aligned. Conflating them with
    program SWAPs is what made `allow_transversal_swap_merges` produce wrong
    circuits: whichever single semantics was chosen, one of the two callers broke.
    """

    def __init__(self):
        super().__init__("routing_swap", 2, [])

    def _define(self):
        qubits = QuantumRegister(2, "q")
        definition = QuantumCircuit(qubits, name=self.name)
        definition.swap(qubits[0], qubits[1])
        self.definition = definition

    def inverse(self, annotated: bool = False):
        return RoutingSwapGate()
