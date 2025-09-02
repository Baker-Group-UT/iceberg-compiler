"""
iceberg_module.py — Defines the IcebergModule class,
an instance of an Iceberg quantum code which contains
n phsyical qubits and k (n-2) logical qubits.
"""

from logging import warning
from typing import Collection

import cirq

from .base import CircuitClass
from .gates import BarrierGate
from .utils import rand_clifford_angles, SharedInt


class IcebergModule(CircuitClass):
    """
    [[n, n-2, 2]] Iceberg error detection code, representing a single encoded module.
    """

    def __init__(
        self,
        n: int,
        circuit: cirq.Circuit,
        ids: SharedInt | None = None,
        initialize: str | None = None,
    ):
        """
        Initialize an IcebergModule of size `n` with optional logical initialization.
        """
        if n & 1:
            raise ValueError("n must be even")
        self.n = n
        self.k = n - 2
        self._ids = ids
        self.data_qubits = cirq.LineQubit.range(ids, ids + n)
        ids += n
        self.parity_qubits = cirq.LineQubit.range(ids, ids + 2)
        ids += 2
        self.circuit = circuit

        if initialize is not None:
            self.encode(initialize)

    def encode(self, initialize: str | None = None):
        """
        Initialize the module into the \\ket{initialize_L} state.
        Accepts a binary string of length k (e.g., "00" or "0+"). Defaults to all 0s.
        """
        if self._encoded:
            warning(f"{self.__class__.__name__} is already encoded")
            return

        if initialize is None:
            initialize = "0" * self.k
        elif (
            initialize.count("0") + initialize.count("+") != self.k
            or len(initialize) != self.k
        ):
            raise ValueError("Invalid initialization string")

        self._encoded = True

        if "+" in initialize:
            warning(
                f"{self.__class__.__name__}: + state initialization not implemented"
            )
            return

        # Default: |0^k> encoding
        self.circuit.append(cirq.H(self.data_qubits[0]))
        for i in range(0, self.n - 1):
            self.circuit.append(cirq.CNOT(self.data_qubits[i], self.data_qubits[i + 1]))

    def decode(self, key_prefix: str = ""):
        """
        Measure both ancilla qubits:
        - {prefix}anc0 = X-type parity
        - {prefix}anc1 = Z-type parity
        """
        # X-type parity
        for dq in self.data_qubits:
            self.circuit.append(cirq.CNOT(dq, self.parity_qubits[0]))
        self.circuit.append(
            cirq.measure(self.parity_qubits[0], key=f"{key_prefix}anc0")
        )

        # Z-type parity
        for dq in self.data_qubits:
            self.circuit.append(cirq.H(dq))
            self.circuit.append(cirq.CNOT(dq, self.parity_qubits[1]))
            self.circuit.append(cirq.H(dq))
        self.circuit.append(
            cirq.measure(self.parity_qubits[1], key=f"{key_prefix}anc1")
        )

    def measure(self, key: str):
        """
        Measure the data qubits into a register labeled `key`.
        """
        self.circuit.append(cirq.measure(*self.data_qubits, key=key))

    def _physical_two_qubit_gate(
        self, i: int, j: int, gate: str, fault_tolerant: bool = True
    ):
        """
        Apply a CZ or CNOT gate between physical qubits i and j.
        """
        if gate not in ("CZ", "CNOT"):
            raise ValueError("gate must be 'CZ' or 'CNOT'")
        if fault_tolerant:
            flag_ancillas = cirq.LineQubit.range(self._ids, self._ids + 2)
            self.circuit.append(cirq.H(flag_ancillas[1]))
            self.circuit.append(cirq.CNOT(self.data_qubits[i], flag_ancillas[0]))
            if gate == "CZ":
                self.circuit.append(cirq.CNOT(flag_ancillas[1], self.data_qubits[j]))
            else:
                self.circuit.append(cirq.CZ(flag_ancillas[1], self.data_qubits[j]))
            if gate == "CZ":
                self.circuit.append(cirq.CZ(self.data_qubits[i], self.data_qubits[j]))
            else:
                self.circuit.append(cirq.CNOT(self.data_qubits[i], self.data_qubits[j]))
            self.circuit.append(cirq.CZ(flag_ancillas[1], self.data_qubits[i]))
            self.circuit.append(cirq.CNOT(self.data_qubits[i], flag_ancillas[0]))
            if gate == "CZ":
                self.circuit.append(cirq.CNOT(flag_ancillas[1], self.data_qubits[j]))
            else:
                self.circuit.append(cirq.CZ(flag_ancillas[1], self.data_qubits[j]))
            self.circuit.append(cirq.H(flag_ancillas[1]))
            self.circuit.append(cirq.measure(flag_ancillas[0], key=f"flag{self._ids}"))
            self.circuit.append(
                cirq.measure(flag_ancillas[1], key=f"flag{self._ids + 1}")
            )
            self._ids += 2
        else:
            if gate == "CZ":
                self.circuit.append(cirq.CZ(self.data_qubits[i], self.data_qubits[j]))
            else:
                self.circuit.append(cirq.CNOT(self.data_qubits[i], self.data_qubits[j]))

    def X(self, i: int | Collection[int]):
        """
        Apply the logical X gate to logical qubit(s) `i`.
        """
        self._apply_pauli(i, gate=cirq.X, target_ancilla=self.data_qubits[0])

    def Z(self, i: int | Collection[int]):
        """
        Apply the logical Z gate to logical qubit(s) `i`.
        """
        self._apply_pauli(i, gate=cirq.Z, target_ancilla=self.data_qubits[-1])

    def _apply_pauli(self, i, gate, target_ancilla):
        if isinstance(i, int):
            i = [i]
        if any(j < 0 or j >= self.k for j in i):
            raise ValueError("Index out of range")
        if len(set(i)) != len(i):
            raise ValueError("Duplicate indices")

        if len(i) % 2:
            self.circuit.append(gate(target_ancilla))
        for j in i:
            self.circuit.append(gate(self.data_qubits[j + 1]))

    def S(self, i: int | Collection[int], fault_tolerant: bool = False):
        """
        Apply the logical S gate to logical qubit(s) `i`.
        """
        if isinstance(i, int):
            i = [i]
        for j in i:
            if j < 0 or j >= self.k:
                raise ValueError("Index out of range")
        for j in i:
            self.circuit.append(cirq.S(self.data_qubits[-1]))
            self._physical_two_qubit_gate(j + 1, -1, "CZ", fault_tolerant)
            self.circuit.append(cirq.S(self.data_qubits[j + 1]))

    def H(self, i: int | Collection[int], fault_tolerant: bool = False):
        """
        Apply the logical H gate to logical qubit(s) `i`.
        """
        if isinstance(i, int):
            i = [i]
        if any(j < 0 or j >= self.k for j in i):
            raise ValueError("Index out of range")
        if len(set(i)) != len(i):
            raise ValueError("Duplicate indices")

        if len(i) == self.k:
            for q in self.data_qubits:
                self.circuit.append(cirq.H(q))

            if fault_tolerant:
                self._physical_two_qubit_gate(0, -1, "CNOT", fault_tolerant)
                self._physical_two_qubit_gate(-1, 0, "CNOT", fault_tolerant)
                self._physical_two_qubit_gate(0, -1, "CNOT", fault_tolerant)
            else:
                self.circuit.append(
                    cirq.SWAP(self.data_qubits[0], self.data_qubits[-1])
                )

        else:
            for j in i:
                dq = self.data_qubits[j + 1]
                self._physical_two_qubit_gate(j + 1, -1, "CZ", fault_tolerant)
                self.circuit.append(cirq.H(self.data_qubits[0]))
                self.circuit.append(cirq.H(dq))
                self._physical_two_qubit_gate(0, j + 1, "CZ", fault_tolerant)
                self.circuit.append(cirq.H(dq))
                self._physical_two_qubit_gate(0, j + 1, "CZ", fault_tolerant)
                self.circuit.append(cirq.H(self.data_qubits[0]))
                self.circuit.append(cirq.H(dq))
                self._physical_two_qubit_gate(j + 1, -1, "CZ", fault_tolerant)
                self.circuit.append(cirq.X(self.data_qubits[0]))
                self.circuit.append(cirq.Z(self.data_qubits[-1]))

    def HH(self, fault_tolerant: bool = False):
        """
        Apply an HH operation to this Iceberg module (n=4 only).
        Equivalent to H on all data qubits + SWAP(1,2).
        """
        # FIXME: remove this and incorporate all references into H
        if self.n != 4:
            raise ValueError("HH only implemented for n=4")
        for q in self.data_qubits:
            self.circuit.append(cirq.H(q))
        if fault_tolerant:
            self._physical_two_qubit_gate(1, 2, "CNOT", fault_tolerant)
            self._physical_two_qubit_gate(2, 1, "CNOT", fault_tolerant)
            self._physical_two_qubit_gate(1, 2, "CNOT", fault_tolerant)
        else:
            self.circuit.append(cirq.SWAP(self.data_qubits[1], self.data_qubits[2]))

    def CNOT(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical CNOT from qubit `i` \\to `j`.
        - Faulty mode (n=4): SWAP-only shortcut for (0 \\to 1) or (1 \\to 0)
        - Standard mode: 4-step logical CNOT sequence
        """
        if i == j:
            raise ValueError("control and target must differ")
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")

        if not fault_tolerant and self.n == 4:
            # FIXME: find a way to make this independent of n = 4
            if (i, j) == (0, 1):
                self.circuit.append(cirq.SWAP(self.data_qubits[0], self.data_qubits[1]))
                return
            elif (i, j) == (1, 0):
                self.circuit.append(cirq.SWAP(self.data_qubits[0], self.data_qubits[2]))
                return
            raise NotImplementedError(
                "Faulty CNOT only defined for (0 \to 1) or (1 \to 0)"
            )

        else:
            self._physical_two_qubit_gate(i + 1, 0, "CNOT", fault_tolerant)
            self._physical_two_qubit_gate(-1, 0, "CNOT", fault_tolerant)
            self._physical_two_qubit_gate(-1, j + 1, "CNOT", fault_tolerant)
            self._physical_two_qubit_gate(i + 1, j + 1, "CNOT", fault_tolerant)

    def CZ(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical CZ gate between qubit `i` and `j`.
        """
        if i == j:
            raise ValueError("i and j must differ")
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")

        self._physical_two_qubit_gate(i + 1, j + 1, "CZ", fault_tolerant)
        self.circuit.append(cirq.Z(self.data_qubits[-1]))
        self._physical_two_qubit_gate(i + 1, -1, "CZ", fault_tolerant)
        self._physical_two_qubit_gate(j + 1, -1, "CZ", fault_tolerant)

    def _RZZ(
        self,
        i: int,
        j: int,
        rads: float,
        fault_tolerant: bool = False,
        two_qubit: bool = True,
    ):
        if i == j:
            raise ValueError("i and j must differ")
        if (
            i < 0
            or i >= self.k
            or j < 0
            or (two_qubit and j >= self.k)
            or (not two_qubit and j > self.k)
        ):
            raise ValueError("Index out of range")

        self._physical_two_qubit_gate(i + 1, j + 1, "CNOT", fault_tolerant)
        self.circuit.append(cirq.rz(rads)(self.data_qubits[j + 1]))
        self._physical_two_qubit_gate(i + 1, j + 1, "CNOT", fault_tolerant)

    def RZZ(self, i: int, j: int, rads: float, fault_tolerant: bool = False):
        """
        Apply a logical RZZ(i, j) = exp(-i rads Z_i Z_j / 2)
        """
        self._RZZ(i, j, rads, fault_tolerant=fault_tolerant, two_qubit=True)

    def RZ(self, i: int, rads: float, fault_tolerant: bool = False):
        """
        Apply a logical RZ(i) = exp(-i rads Z_i / 2)
        """
        self._RZZ(i, self.k, rads, fault_tolerant=fault_tolerant, two_qubit=False)

    def initialize_arbitrary_state(self, indices: Collection[int] | None = None):
        """
        Prepare Haar-random states on `indices` logical qubits **before** calling `encode()`.
        If indices is None, initializes all logical qubits.
        """
        if indices is None:
            indices = range(self.k)
        elif isinstance(indices, int):
            indices = [indices]
        for j in indices:
            theta, phi = rand_clifford_angles()
            q = self.data_qubits[j + 1]
            self.circuit.append(cirq.rz(phi).on(q))
            self.circuit.append(cirq.H(q))
            self.circuit.append(cirq.rz(theta).on(q))
            self.circuit.append(cirq.H(q))

    def BARRIER(self):
        """
        Insert a visual barrier across all physical qubits in the module.
        """
        barrier_gate = BarrierGate()
        for q in self.data_qubits + self.parity_qubits:
            self.circuit.append(barrier_gate.on(q))

    def SWAP(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical SWAP between qubits `i` and `j`.
        - Faulty mode (n=4, {0,1}): shortcut SWAP(q1, q2)
        - Default: FT SWAP via 3 logical CNOTs
        """
        if i == j:
            raise ValueError("Cannot swap a qubit with itself")
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i > j:
            i, j = j, i

        if not fault_tolerant and self.n == 4:
            # FIXME: find a way to make this independent of n = 4
            self.circuit.append(cirq.SWAP(self.data_qubits[1], self.data_qubits[2]))
        else:
            self.CNOT(i, j, fault_tolerant=fault_tolerant)
            self.CNOT(j, i, fault_tolerant=fault_tolerant)
            self.CNOT(i, j, fault_tolerant=fault_tolerant)