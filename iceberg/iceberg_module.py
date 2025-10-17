"""
iceberg_module.py — Defines the IcebergModule class,
an instance of an Iceberg quantum code which contains
n phsyical qubits and k (n-2) logical qubits.


iceberg_module_qiskit.py — Qiskit port of IcebergModule.

Defines the IcebergModule class, an instance of an Iceberg quantum code which
contains n physical qubits and k = (n-2) logical qubits.

Notes on the port:
- cirq.LineQubit.range(...) is replaced by adding dedicated QuantumRegisters for
  data and parity qubits within the circuit.
- Measurement "keys" are emulated by creating ClassicalRegisters with those names.
- The SharedInt `ids` is still used to name new ancilla registers and flag cregs,
  and is incremented consistently.
"""

from logging import warning
from typing import Collection, Iterable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .base import CircuitClass
from .utils import SharedInt, rand_clifford_angles


def _ensure_creg(circ: QuantumCircuit, name: str, size: int = 1) -> ClassicalRegister:
    """
    Get or create a ClassicalRegister with the given name and size.
    If it already exists and the size matches, reuse it; otherwise create a new one.
    """
    for creg in circ.cregs:
        if creg.name == name:
            if len(creg) != size:
                raise ValueError(f"Existing creg '{name}' has size {len(creg)} != {size}")
            return creg
    creg = ClassicalRegister(size, name=name)
    circ.add_register(creg)
    return creg


def _measure_one(circ: QuantumCircuit, q, creg_name: str):
    """
    Measure a single qubit into a 1-bit ClassicalRegister named `creg_name`.
    """
    creg = _ensure_creg(circ, creg_name, 1)
    circ.measure(q, creg[0])


def _measure_many(circ: QuantumCircuit, qubits: Iterable, creg_name: str):
    """
    Measure a list of qubits into a ClassicalRegister named `creg_name`
    of matching length, mapping qubit i -> bit i (left-to-right as provided).
    """
    qubits = list(qubits)
    creg = _ensure_creg(circ, creg_name, len(qubits))
    circ.measure(qubits, creg)  # Qiskit supports list-to-creg measure


def _add_flag_ancillas(circ: QuantumCircuit, ids: SharedInt, count: int = 2) -> QuantumRegister:
    """
    Allocate `count` ancilla qubits in a new QuantumRegister named with the current `ids`.
    Increments `ids` by `count`.
    """
    qreg = QuantumRegister(count, name=f"flag_{int(ids)}")
    circ.add_register(qreg)
    ids += count
    return qreg


def _physical_two_qubit_gate(
    circuit: QuantumCircuit,
    ids: SharedInt,
    i,  # Qubit
    j,  # Qubit
    gate: str,
    fault_tolerant: bool = True,
):
    """
    Apply a CZ or CNOT gate between physical qubits i and j.
    In FT mode, use a flagged construction with two temporary ancillas (freshly allocated).
    """
    if gate not in ("CZ", "CNOT"):
        raise ValueError("gate must be 'CZ' or 'CNOT'")

    if fault_tolerant:
        flag_anc = _add_flag_ancillas(circuit, ids, 2)
        a0, a1 = flag_anc[0], flag_anc[1]

        circuit.h(a1)
        circuit.cx(i, a0)
        if gate == "CZ":
            circuit.cx(a1, j)
        else:
            circuit.cz(a1, j)

        if gate == "CZ":
            circuit.cz(i, j)
        else:
            circuit.cx(i, j)

        circuit.cz(a1, i)
        circuit.cx(i, a0)

        if gate == "CZ":
            circuit.cx(a1, j)
        else:
            circuit.cz(a1, j)

        circuit.h(a1)

        _measure_one(circuit, a0, f"flag{int(ids) - 2}")
        _measure_one(circuit, a1, f"flag{int(ids) - 1}")
    else:
        if gate == "CZ":
            circuit.cz(i, j)
        else:
            circuit.cx(i, j)


class IcebergModule(CircuitClass):
    """
    [[n, n-2, 2]] Iceberg error-detection code, representing a single encoded module.

    Qubit layout per module (Qiskit registers created per instance):
      - data:   n qubits  (data[0] and data[-1] act as ancillas in some protocols)
      - parity: 2 qubits  (parity[0] = X-type parity ancilla, parity[1] = Z-type parity ancilla)
    """

    def __init__(
        self,
        n: int,
        circuit: QuantumCircuit,
        ids: SharedInt,
        initialize: str | None = None,
    ):
        if n & 1:
            raise ValueError("n must be even")
        self.n = n
        self.k = n - 2
        self._ids = ids
        self.circuit = circuit
        self._encoded = False

        # Allocate registers dedicated to this module
        self._data = QuantumRegister(self.n, name=f"data_{int(ids)}")
        circuit.add_register(self._data)

        self._parity = QuantumRegister(2, name=f"parity_{int(ids)}")
        circuit.add_register(self._parity)
        ids += 1

        # For convenience, make these lists of qubit references (indexable like before)
        self.data_qubits = list(self._data)
        self.parity_qubits = list(self._parity)

        if initialize is not None:
            self.encode(initialize)

    # --- Initialization / encoding / decoding ---

    def encode(self, initialize: str | None = None):
        """
        Initialize the module into the |initialize_L> state.
        Accepts a binary string of length k (e.g., "00" or "0+"). Defaults to all 0s.
        """
        if self._encoded:
            warning(f"{self.__class__.__name__} is already encoded")
            return

        if initialize is None:
            initialize = "0" * self.k
        elif (initialize.count("0") + initialize.count("+") != self.k) or (len(initialize) != self.k):
            raise ValueError("Invalid initialization string")

        self._encoded = True

        if "+" in initialize:
            warning(f"{self.__class__.__name__}: + state initialization not implemented")
            return

        # Default: |0^k> encoding via GHZ-like fanout across data qubits
        self.circuit.h(self.data_qubits[0])
        for i in range(0, self.n - 1):
            self.circuit.cx(self.data_qubits[i], self.data_qubits[i + 1])


    def decode(self, key_prefix: str = ""):
        """
        Measure both parity ancilla qubits:
          - {prefix}anc0 = X-type parity
          - {prefix}anc1 = Z-type parity
        """
        # X-type parity (CNOT from each data into parity_qubits[0])
        for dq in self.data_qubits:
            self.circuit.cx(dq, self.parity_qubits[0])
        self.BARRIER()

        # Z-type parity: H, CX to parity[1], H (on data)
        for dq in self.data_qubits:
            self.circuit.h(dq)
            self.circuit.cx(dq, self.parity_qubits[1])
            self.circuit.h(dq)
        self.BARRIER()

        _measure_many(self.circuit, self.parity_qubits, f"m_parity_{key_prefix}")

    def measure(self, key: str):
        """
        Measure the data qubits into a classical register labeled `key`.
        """
        _measure_many(self.circuit, self.data_qubits, key)

    # --- Logical Paulis ---

    def X(self, i: int | Collection[int]):
        """
        Apply the logical X gate to logical qubit(s) `i`.
        """
        self._apply_pauli(i, gate="X", target_ancilla=self.data_qubits[0])

    def Z(self, i: int | Collection[int]):
        """
        Apply the logical Z gate to logical qubit(s) `i`.
        """
        self._apply_pauli(i, gate="Z", target_ancilla=self.data_qubits[-1])

    def _apply_pauli(self, i, gate: str, target_ancilla):
        if isinstance(i, int):
            i = [i]
        if any(j < 0 or j >= self.k for j in i):
            raise ValueError("Index out of range")
        if len(set(i)) != len(i):
            raise ValueError("Duplicate indices")

        # odd parity of selected logicals -> also flip target ancilla
        if len(i) % 2:
            if gate == "X":
                self.circuit.x(target_ancilla)
            elif gate == "Z":
                self.circuit.z(target_ancilla)
            else:
                raise ValueError("gate must be 'X' or 'Z'")

        for j in i:
            q = self.data_qubits[j + 1]
            if gate == "X":
                self.circuit.x(q)
            else:
                self.circuit.z(q)

    # --- Clifford / 2-qubit logical ops ---

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
            bottom_ancilla = self.data_qubits[-1]
            data_qubit = self.data_qubits[j + 1]
            self.circuit.s(bottom_ancilla)
            _physical_two_qubit_gate(self.circuit, self._ids, data_qubit, bottom_ancilla, "CZ", fault_tolerant)
            self.circuit.s(data_qubit)

    def H(self, i: int | Collection[int], fault_tolerant: bool = False):
        if isinstance(i, int):
            i = [i]
        if any(j < 0 or j >= self.k for j in i):
            raise ValueError("Index out of range")
        if len(set(i)) != len(i):
            raise ValueError("Duplicate indices")

        if len(i) == self.k:
            # Equivalent to HH (Hadamard on all + SWAP(1,2) for n=4)
            for q in self.data_qubits:
                self.circuit.h(q)
            if fault_tolerant:
                # Operate on the *middle* pair (1,2) for n=4 using flagged CNOT sandwich
                _physical_two_qubit_gate(self.circuit, self._ids, self.data_qubits[1], self.data_qubits[2], "CNOT", True)
                _physical_two_qubit_gate(self.circuit, self._ids, self.data_qubits[2], self.data_qubits[1], "CNOT", True)
                _physical_two_qubit_gate(self.circuit, self._ids, self.data_qubits[1], self.data_qubits[2], "CNOT", True)
            else:
                self.circuit.swap(self.data_qubits[1], self.data_qubits[2])
            return

        # Per-qubit H with ancilla interactions
        for j in i:
            top_ancilla = self.data_qubits[0]
            data_qubit = self.data_qubits[j + 1]
            bottom_ancilla = self.data_qubits[-1]

            _physical_two_qubit_gate(self.circuit, self._ids, data_qubit, bottom_ancilla, "CZ", fault_tolerant)
            self.circuit.h(top_ancilla)
            self.circuit.h(data_qubit)
            _physical_two_qubit_gate(self.circuit, self._ids, top_ancilla, data_qubit, "CZ", fault_tolerant)
            self.circuit.h(data_qubit)
            _physical_two_qubit_gate(self.circuit, self._ids, top_ancilla, data_qubit, "CZ", fault_tolerant)
            self.circuit.h(top_ancilla)
            self.circuit.h(data_qubit)
            _physical_two_qubit_gate(self.circuit, self._ids, data_qubit, bottom_ancilla, "CZ", fault_tolerant)
            self.circuit.x(top_ancilla)
            self.circuit.z(bottom_ancilla)

    def HH(self, fault_tolerant: bool = False):
        """
        Apply an HH operation to this Iceberg module (n=4 only).
        Equivalent to H on all data qubits + SWAP(1,2).
        """
        if self.n != 4:
            raise ValueError("HH only implemented for n=4")
        for q in self.data_qubits:
            self.circuit.h(q)
        if fault_tolerant:
            _physical_two_qubit_gate(self.circuit, self._ids, self.data_qubits[1], self.data_qubits[2], "CNOT", True)
            _physical_two_qubit_gate(self.circuit, self._ids, self.data_qubits[2], self.data_qubits[1], "CNOT", True)
            _physical_two_qubit_gate(self.circuit, self._ids, self.data_qubits[1], self.data_qubits[2], "CNOT", True)
        else:
            self.circuit.swap(self.data_qubits[1], self.data_qubits[2])

    def CNOT(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical CNOT from qubit `i` -> `j`.
        - Faulty mode (n=4): direct CX-based shortcut used (as in the original)
        - Standard mode: 4-step logical CNOT sequence via ancillas
        """
        if i == j:
            raise ValueError("control and target must differ")
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")

        if not fault_tolerant and self.n == 4:
            # Shortcut from original implementation
            self.circuit.cx(self.data_qubits[i + 1], self.data_qubits[0])
            self.circuit.cx(self.data_qubits[-1], self.data_qubits[0])
            self.circuit.cx(self.data_qubits[-1], self.data_qubits[j + 1])
            self.circuit.cx(self.data_qubits[i + 1], self.data_qubits[j + 1])
        else:
            top_ancilla = self.data_qubits[0]
            control_qubit = self.data_qubits[i + 1]
            target_qubit = self.data_qubits[j + 1]
            bottom_ancilla = self.data_qubits[-1]
            _physical_two_qubit_gate(self.circuit, self._ids, control_qubit, top_ancilla, "CNOT", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._ids, bottom_ancilla, top_ancilla, "CNOT", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._ids, bottom_ancilla, target_qubit, "CNOT", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._ids, control_qubit, target_qubit, "CNOT", fault_tolerant)

    def CZ(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical CZ gate between qubit `i` and `j`.
        """
        if i == j:
            raise ValueError("i and j must differ")
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")

        control_qubit = self.data_qubits[i + 1]
        target_qubit = self.data_qubits[j + 1]
        bottom_ancilla = self.data_qubits[-1]
        _physical_two_qubit_gate(self.circuit, self._ids, control_qubit, target_qubit, "CZ", fault_tolerant)
        self.circuit.z(bottom_ancilla)
        _physical_two_qubit_gate(self.circuit, self._ids, control_qubit, bottom_ancilla, "CZ", fault_tolerant)
        _physical_two_qubit_gate(self.circuit, self._ids, target_qubit, bottom_ancilla, "CZ", fault_tolerant)

    # --- Parametric rotations ---

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
        if (i < 0 or i >= self.k or
            (two_qubit and (j < 0 or j >= self.k)) or
            (not two_qubit and j != self.k)):
            raise ValueError("Index out of range")

        control_qubit = self.data_qubits[i + 1]
        target_qubit = self.data_qubits[j + 1]
        _physical_two_qubit_gate(self.circuit, self._ids, control_qubit, target_qubit, "CNOT", fault_tolerant)
        self.circuit.rz(rads, target_qubit)
        _physical_two_qubit_gate(self.circuit, self._ids, control_qubit, target_qubit, "CNOT", fault_tolerant)

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

    # --- Misc helpers ---

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
            self.circuit.rz(phi, q)
            self.circuit.h(q)
            self.circuit.rz(theta, q)
            self.circuit.h(q)

    def BARRIER(self):
        """
        Insert a visual barrier across all physical qubits in the module.
        """
        # Use Qiskit's built-in barrier across this module's qubits.
        self.circuit.barrier(*(self.data_qubits + self.parity_qubits))

    def SWAP(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical SWAP between qubits `i` and `j`.
        - Faulty mode (n=4, {0,1}): shortcut direct SWAP of data[1] <-> data[2]
        - Default: FT SWAP via 3 logical CNOTs
        """
        if i == j:
            raise ValueError("Cannot swap a qubit with itself")
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i > j:
            i, j = j, i

        if not fault_tolerant and self.n == 4:
            # self.circuit.swap(self.data_qubits[1], self.data_qubits[2])
            self.circuit.cx(self.data_qubits[1], self.data_qubits[2])
            self.circuit.cx(self.data_qubits[2], self.data_qubits[1])
            self.circuit.cx(self.data_qubits[1], self.data_qubits[2])
        else:
            self.CNOT(i, j, fault_tolerant=fault_tolerant)
            self.CNOT(j, i, fault_tolerant=fault_tolerant)
            self.CNOT(i, j, fault_tolerant=fault_tolerant)
