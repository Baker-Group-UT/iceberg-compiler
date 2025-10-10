"""
iceberg_set.py — Defines the IcebergSet class,
a container of multiple Iceberg quantum codes (modules)
managing logical-level operations across patches.
"""

from __future__ import annotations

from typing import Collection

import cirq

from .gates import BarrierGate
from .iceberg_module import IcebergModule, _physical_two_qubit_gate
from .utils import SharedInt


class IcebergSet:
    """
    A set of [[n, n-2, 2]] IcebergModules. Must be consistent.
    """

    def __init__(
        self,
        n_program: int,
        n: int,
        circuit: cirq.Circuit,
        initialize: str | None = None,
    ):
        """
        Create a set of IcebergModules to encode `n_program`
        logical qubits using [[n, n−2, 2]] error-detecting codes.

        Parameters
        ----------
        n_program : int
            Total number of logical qubits to encode.
        n : int
            Number of physical qubits per module. Must be even.
        circuit : cirq.Circuit
            The circuit to which all operations are appended.
        initialize : str | None
            Optional binary/+ string to initialize logical state.
        """
        if n % 2 != 0:
            raise ValueError("n must be even")

        self.n_program = n_program
        self.n = n
        self.k = n - 2
        self.num_modules = (self.n_program + self.k - 1) // self.k
        self._n_total = SharedInt()
        self.modules = [
            IcebergModule(n, circuit, self._n_total) for _ in range(self.num_modules)
        ]
        self.circuit = circuit
        self._encoded = False

        if initialize is not None:
            self.encode(initialize)

    def encode(self, initialize: str | None = None):
        """
        Encode all modules to \\ket{initialize_L} state or according to `initialize` string.

        If `initialize` is provided, it must have length `self.k * self.num_modules`.
        """
        if self._encoded:
            return
        self._encoded = True

        for i, mod in enumerate(self.modules):
            init_sub = initialize[self.k * i : self.k * (i + 1)] if initialize else None
            mod.encode(init_sub)

    def decode(self):
        """Measure and collect syndrome bits from all IcebergModules."""
        for idx, mod in enumerate(self.modules):
            mod.decode(key_prefix=f"patch{idx}_")

    def measure(self):
        """Measure all data qubits from each module into registers."""
        for idx, mod in enumerate(self.modules[::-1]):
            self.circuit.append(cirq.measure(*mod.data_qubits, key=f"data_{idx}"))

    def initialize_arbitrary_state(
        self,
        codes: Collection[int] | None = None,
        indices: Collection[int] | None = None,
    ):
        """
        Initialize Haar-random logical states on selected modules/qubits.

        Parameters
        ----------
        codes : Collection[int] | None
            Which module indices to initialize. `None` → all.
        indices : Collection[int] | None
            Logical-qubit indices within each selected module. `None` → all.
        """
        if codes is None:
            codes = range(self.num_modules)
        elif isinstance(codes, int):
            codes = [codes]

        for i in codes:
            self.modules[i].initialize_arbitrary_state(indices)

    def X(self, i: int | Collection[int]):
        """Apply logical X to specified logical qubit(s)."""
        if isinstance(i, int):
            i = [i]
        s: list[set[int]] = [set() for _ in range(self.num_modules)]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")
            patch = j // self.k
            index = j % self.k
            if index in s[patch]:
                raise ValueError("Duplicate index")
            s[patch].add(index)
        for patch, qset in enumerate(s):
            self.modules[patch].X(qset)

    def Z(self, i: int | Collection[int]):
        """Apply logical Z to specified logical qubit(s)."""
        if isinstance(i, int):
            i = [i]
        s = [set() for _ in range(self.num_modules)]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")
            patch = j // self.k
            index = j % self.k
            if index in s[patch]:
                raise ValueError("Duplicate index")
            s[patch].add(index)
        for patch, qset in enumerate(s):
            self.modules[patch].Z(qset)

    def S(self, i: int | Collection[int]):
        """Apply logical S to specified logical qubit(s)."""
        if isinstance(i, int):
            i = [i]
        s = [set() for _ in range(self.num_modules)]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")
            patch = j // self.k
            index = j % self.k
            if index in s[patch]:
                raise ValueError("Duplicate index")
            s[patch].add(index)
        for patch, qset in enumerate(s):
            self.modules[patch].S(qset)

    def H(self, i: int | Collection[int], fault_tolerant: bool = False):
        # FIXME: remove this and incorporate all references into H
        """Apply logical H to specified logical qubit(s)."""
        if isinstance(i, int):
            i = [i]
        s = [set() for _ in range(self.num_modules)]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")
            patch = j // self.k
            index = j % self.k
            if index in s[patch]:
                raise ValueError("Duplicate index")
            s[patch].add(index)
        for patch, qset in enumerate(s):
            self.modules[patch].H(qset, fault_tolerant=fault_tolerant)

    def HH(self, patches: int | Collection[int], fault_tolerant: bool = False):
        """
        Apply the HH operation (logical Hadamard followed by relabel) to selected modules.
        """
        if isinstance(patches, int):
            patches = [patches]
        for p in patches:
            self.modules[p].HH(fault_tolerant=fault_tolerant)

    def CNOT(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply logical CNOT between two logical qubits (intra-module only).
        """
        if i == j:
            raise ValueError("control and target must differ")
        pi, qi = divmod(i, self.k)
        pj, qj = divmod(j, self.k)
        if pi == pj:
            self.modules[pi].CNOT(qi, qj, fault_tolerant=fault_tolerant)
        elif qi == qj:
            control_data = self.modules[pi].data_qubits[qi + 1]
            control_ancilla = self.modules[pi].data_qubits[-1]
            target_data = self.modules[pj].data_qubits[qj + 1]
            target_ancilla = self.modules[pj].data_qubits[0]

            _physical_two_qubit_gate(self.circuit, self._n_total, control_data, target_data, "CNOT", False) # this is transversal
            _physical_two_qubit_gate(self.circuit, self._n_total, control_data, target_ancilla, "CNOT", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._n_total, control_ancilla, target_data, "CNOT", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._n_total, control_ancilla, target_ancilla, "CNOT", fault_tolerant)
        else:
            raise NotImplementedError("Cross-patch CNOT for non-identical indices not implemented")

    def CNOT_k_Transversal(self, i: int, j: int):
        """Apply transversal CNOT across modules i → j."""
        if i == j:
            raise ValueError("i and j must be different")
        ctrl = self.modules[i].data_qubits
        targ = self.modules[j].data_qubits
        for q in range(self.n):
            self.circuit.append(cirq.CNOT(ctrl[q], targ[q]))

    def CZ(self, i: int, j: int, fault_tolerant: bool = False):
        """Apply logical CZ between two logical qubits (intra-module only)."""
        if i == j:
            raise ValueError("i and j must be different")
        pi, qi = divmod(i, self.k)
        pj, qj = divmod(j, self.k)
        if pi == pj:
            self.modules[pi].CZ(qi, qj, fault_tolerant=fault_tolerant)
        elif qi == qj:
            control_data = self.modules[pi].data_qubits[qi + 1]
            control_ancilla = self.modules[pi].data_qubits[-1]
            target_data = self.modules[pj].data_qubits[qj + 1]
            target_ancilla = self.modules[pj].data_qubits[-1]

            _physical_two_qubit_gate(self.circuit, self._n_total, control_data, target_data, "CZ", False) # this is transversal
            _physical_two_qubit_gate(self.circuit, self._n_total, control_data, target_ancilla, "CZ", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._n_total, control_ancilla, target_data, "CZ", fault_tolerant)
            _physical_two_qubit_gate(self.circuit, self._n_total, control_ancilla, target_ancilla, "CZ", False) # this is transversal
        else:
            raise NotImplementedError("Cross-patch CZ for non-identical indices not implemented")

    def CZ_k_Transversal(self, i: int, j: int):
        """
        Apply transversal CZ across modules i and j (not fault-tolerant).
        """
        if i == j:
            raise ValueError("i and j must be different")
        ctrl = self.modules[i].data_qubits
        targ = self.modules[j].data_qubits
        self.circuit.append(cirq.CZ(ctrl[0], targ[-1]))
        for q in range(1, self.n - 1):
            self.circuit.append(cirq.CZ(ctrl[q], targ[q]))
        self.circuit.append(cirq.CZ(ctrl[-1], targ[0]))

    def RZ(self, i: int, rads: float, fault_tolerant: bool = False):
        """Apply logical RZ to a logical qubit."""
        patch, idx = divmod(i, self.k)
        self.modules[patch].RZ(idx, rads, fault_tolerant=fault_tolerant)

    def RZZ(self, i: int, j: int, rads: float):
        """(Not yet implemented) Apply logical RZZ between logical qubits."""
        raise NotImplementedError("RZZ not implemented")

    def SWAP(self, swaps: list[list[int]], fault_tolerant: bool = False):
        """
        Apply logical SWAP operations within patches.

        Each element of `swaps` is [patch_index, i, j] for logical qubit indices i and j.
        """
        for patch, i, j in swaps:
            self.modules[patch].SWAP(i, j, fault_tolerant=fault_tolerant)

    def swap(self, swaps: list[list[int]], fault_tolerant: bool = False) -> None:
        """
        deprecated.
        """
        raise DeprecationWarning(
            "The name of this function was capitalized in the refactor"
        )

    def BARRIER(self, patches: int | Collection[int] | None = None):
        """
        Insert a visual barrier (||) across all physical qubits in selected patches,
        or across all patches if none are specified.
        """
        barrier_gate = BarrierGate()
        qubits = []

        if patches is None:
            modules = self.modules
        else:
            if isinstance(patches, int):
                patches = [patches]
            modules = [self.modules[i] for i in patches]

        for module in modules:
            qubits.extend(module.data_qubits + module.parity_qubits)

        self.circuit.insert(
            len(self.circuit), cirq.Moment([barrier_gate.on(q) for q in qubits])
        )

    def barrier(self, patches: int | Collection[int] | None = None) -> None:
        """
        deprecated.
        """
        raise DeprecationWarning(
            "The name of this function was capitalized in the refactor"
        )
