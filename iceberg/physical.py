"""
physical.py — emission of physical gates and measurements.

Split out of `iceberg_module.py` so that the logical constructions there read as
the algebra they implement, rather than as plumbing. Two concerns live here:

* classical registers and measurement (`measure_one`, `measure_many`), whose
  naming convention (`m_data_*`, `m_parity_*`, `flag*`) is part of the contract
  with the postselector and the simulator's key ordering;
* two-qubit gate emission (`FlaggedGates`), which either drops a bare gate or
  wraps it in the flagged fault-tolerant gadget with two fresh ancillas.
"""

from __future__ import annotations

from typing import Iterable

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .utils import SharedInt

__all__ = ["FlaggedGates", "measure_one", "measure_many", "ensure_creg"]


def ensure_creg(circuit: QuantumCircuit, name: str, size: int = 1) -> ClassicalRegister:
    """Get or create a ClassicalRegister with the given name and size."""
    for creg in circuit.cregs:
        if creg.name == name:
            if len(creg) != size:
                raise ValueError(f"Existing creg '{name}' has size {len(creg)} != {size}")
            return creg
    creg = ClassicalRegister(size, name=name)
    circuit.add_register(creg)
    return creg


def measure_one(circuit: QuantumCircuit, qubit, creg_name: str) -> None:
    """Measure one qubit into a 1-bit ClassicalRegister named `creg_name`."""
    creg = ensure_creg(circuit, creg_name, 1)
    circuit.measure(qubit, creg[0])


def measure_many(circuit: QuantumCircuit, qubits: Iterable, creg_name: str) -> None:
    """
    Measure `qubits` into a ClassicalRegister of matching length.

    Qubit i maps to bit i, left-to-right as provided. Note that Qiskit prints
    registers most-significant-bit first, so bit i appears at string position
    `len - 1 - i`; `iceberg_postselect` is the one place that reverses this.
    """
    qubits = list(qubits)
    creg = ensure_creg(circuit, creg_name, len(qubits))
    circuit.measure(qubits, creg)


class FlaggedGates:
    """
    Two-qubit gate emitter for one circuit, with optional flagged FT gadgets.

    In fault-tolerant mode each two-qubit gate is replaced by the flagged
    construction: two fresh ancillas are allocated, the gate is sandwiched
    between parity checks, and the ancillas are measured into `flag*` registers
    that postselection rejects on. That costs 2 ancillas and 6 extra gates per
    two-qubit gate, independent of n.

    Individual call sites may override the mode with `flagged=`; the transversal
    constructions use that to leave deliberately-unflagged gates unflagged.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        ids: SharedInt,
        *,
        fault_tolerant: bool = False,
        allow_swap: bool = True,
    ):
        self.circuit = circuit
        self.ids = ids
        self.fault_tolerant = fault_tolerant
        self.allow_swap = allow_swap

    def for_mode(self, fault_tolerant: bool) -> "FlaggedGates":
        """A view of this emitter with a different default flagging mode."""
        return FlaggedGates(
            self.circuit,
            self.ids,
            fault_tolerant=fault_tolerant,
            allow_swap=self.allow_swap,
        )

    # --- primitives ---

    def cx(self, control, target, *, flagged: bool | None = None) -> None:
        self._emit("CNOT", control, target, flagged)

    def cz(self, a, b, *, flagged: bool | None = None) -> None:
        self._emit("CZ", a, b, flagged)

    def x_basis_cz(self, a, b, *, flagged: bool | None = None) -> None:
        """
        The X-basis dual of CZ: (H_a) CX(a->b) (H_a).

        Diagonal in the X eigenbasis of both qubits with a -1 on |-,->, i.e.
        `CZ` conjugated by H on both qubits. This is to logical X-type operations
        what a bare `cz` is to Z-type ones, and it is what makes the logical XCX
        constructions mirror the CZ ones exactly. Costs 1 two-qubit gate and 2
        single-qubit gates.
        """
        self.circuit.h(a)
        self.cx(a, b, flagged=flagged)
        self.circuit.h(a)

    def swap(self, a, b, *, flagged: bool | None = None) -> None:
        """
        Exchange two physical qubits.

        Whether a native SWAP is available is an architecture question, answered
        by `allow_swap`; on hardware that has one it is a relabeling and there is
        nothing to flag. Only the three-CNOT fallback is subject to `flagged`.
        """
        if self.allow_swap:
            self.circuit.swap(a, b)
            return
        self.cx(a, b, flagged=flagged)
        self.cx(b, a, flagged=flagged)
        self.cx(a, b, flagged=flagged)

    # --- internals ---

    def _emit(self, gate: str, i, j, flagged: bool | None) -> None:
        use_flags = self.fault_tolerant if flagged is None else flagged
        if not use_flags:
            self._bare(gate, i, j)
            return

        a0, a1 = self._add_flag_ancillas()
        # Flag ancilla a1 checks the gate's own support, a0 checks qubit i.
        self.circuit.h(a1)
        self.circuit.cx(i, a0)
        self._flag_coupling(gate, a1, j)
        self._bare(gate, i, j)
        self.circuit.cz(a1, i)
        self.circuit.cx(i, a0)
        self._flag_coupling(gate, a1, j)
        self.circuit.h(a1)

        measure_one(self.circuit, a0, f"flag{int(self.ids) - 2}")
        measure_one(self.circuit, a1, f"flag{int(self.ids) - 1}")

    def _bare(self, gate: str, i, j) -> None:
        if gate == "CZ":
            self.circuit.cz(i, j)
        elif gate == "CNOT":
            self.circuit.cx(i, j)
        else:
            raise ValueError(f"gate must be 'CZ' or 'CNOT', got {gate!r}")

    def _flag_coupling(self, gate: str, a1, j) -> None:
        """The flag ancilla couples to `j` in the basis the gate does not act in."""
        if gate == "CZ":
            self.circuit.cx(a1, j)
        else:
            self.circuit.cz(a1, j)

    def _add_flag_ancillas(self, count: int = 2):
        qreg = QuantumRegister(count, name=f"flag_{int(self.ids)}")
        self.circuit.add_register(qreg)
        self.ids += count
        return list(qreg)
