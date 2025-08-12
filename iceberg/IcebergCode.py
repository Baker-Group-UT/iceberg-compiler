from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from logging import warning
from typing import Iterable
import cirq
import math
import numpy as np

def mark_check_encoded(method):
    method._check_encoded = True
    return method


def check_encoded(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._encoded:
            raise ValueError("The circuit is not encoded")
        return method(self, *args, **kwargs)

    return wrapper


class AutoCheckEncodedMeta(ABCMeta):
    """
    Metaclass to automatically check if the circuit is encoded before calling gates.
    """

    def __new__(cls, clsname, bases, classdict):
        for name, method in classdict.items():
            if (
                callable(method)
                and not name.startswith("__")
                and not name.endswith("__")
            ):
                for base in bases:
                    if hasattr(base, name) and getattr(
                        getattr(base, name), "_check_encoded", False
                    ):
                        classdict[name] = check_encoded(method)
                        break
        return super().__new__(cls, clsname, bases, classdict)


class CircuitClass(ABC, metaclass=AutoCheckEncodedMeta):
    """
    An abstract class for a quantum circuit.
    """

    circuit: cirq.Circuit
    _encoded: bool = False

    @abstractmethod
    def encode(self, initialize: str | None): ...

    @mark_check_encoded
    @abstractmethod
    def decode(self): ...

    @mark_check_encoded
    @abstractmethod
    def measure(self): ...

    @mark_check_encoded
    @abstractmethod
    def X(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def Z(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def S(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def H(self, i: int | Iterable[int]): ...

    @mark_check_encoded
    @abstractmethod
    def CNOT(self, i: int, j: int): ...

    @mark_check_encoded
    @abstractmethod
    def CZ(self, i: int, j: int): ...

    @mark_check_encoded
    @abstractmethod
    def RZZ(self, i: int, j: int, rads: float): ...

    @mark_check_encoded
    @abstractmethod
    def RZ(self, i: int, rads: float): ...


CLIFFORD_Z = [0, math.pi / 2, math.pi, -math.pi / 2]   # 0, 90, 180, 270°
CLIFFORD_X = [0, math.pi]                               # Rx(0) or Rx(π)

def _rand_clifford_angles():
    """Return (θ, φ) drawn uniformly from the single-qubit Clifford group."""
    θ = np.random.choice(CLIFFORD_X)            # 0 or π  (Rx axis)
    φ = np.random.choice(CLIFFORD_Z)            # 0, ±π/2, π (Rz axis)
    return θ, φ

class BarrierGate(cirq.Gate):
    def num_qubits(self):
        return 1
    def _circuit_diagram_info_(self, args):
        return "||"

class IcebergCode(CircuitClass):
    """
    [[n, n-2, 2]] quantum error detection code.
    """

    def __init__(
        self,
        n: int,
        circuit: cirq.Circuit,
        start_index: int = 0,
        initialize: str | None = None,
    ):
        """
        Initialize the IcebergCode with n qubits.
        """
        if n & 1:
            raise ValueError("n must be even")
        self.n = n
        self.k = n - 2
        qubits = cirq.LineQubit.range(start_index, start_index + n + 2)
        self.data_qubits = qubits[:n]
        self.parity_qubits = qubits[n:]
        self.circuit = circuit

        if initialize is not None:
            self.encode(initialize)

    def encode(self, initialize: str | None = None):
        """
        Initialize the qubit to the \\ket{initialize_L} state.
        """
        if self._encoded:
            warning(f"Class {self.__class__.__name__} is already encoded")
            return
        if initialize is None:
            initialize = "0" * self.k
        elif (
            initialize.count("0") + initialize.count("+") != self.k
            and len(initialize) != self.k
        ):
            raise ValueError("Invalid initialization string")
        self._encoded = True

        # TODO: implement usage of `initialize`, i.e., initializing each logical qubit to the 0 or + state
        if initialize.count("+") > 0:
            warning(
                f"Class {self.__class__.__name__}: Trying to initialize to + state, but not implemented yet"
            )
            return

        # @TODO: make the \ket{0^k} encoding Fault-Tolerant
        self.circuit.append(cirq.H(self.data_qubits[0]))
        for i in range(0, self.n - 1):
            self.circuit.append(cirq.CNOT(self.data_qubits[i], self.data_qubits[i + 1]))

    def decode(self, *, key_prefix: str = ""):
        """
        Measure both ancilla qubits with explicit keys:
            patch{idx}_anc0   (X-type parity)
            patch{idx}_anc1   (Z-type parity)
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

    def measure(self, *, key: str) -> None:
        """Measure the *data* qubits of **this** patch into a named register."""
        self.circuit.append(cirq.measure(*self.data_qubits, key=key))

    def X(self, i: int | Iterable[int]):
        """
        Apply the Logical X gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]

        for j in i:
            if j < 0 or j >= self.k:
                raise ValueError("Index out of range")

        s = set()
        for j in i:
            if j not in s:
                s.add(j)
            else:
                raise ValueError("Duplicate index")

        # @TODO: Patch complement rule to minimize weight
        # if len(s) > self.k / 2:
        #     s = set(range(self.k)) - s

        if len(s) & 1:
            self.circuit.append(cirq.X(self.data_qubits[0]))
        for j in s:
            self.circuit.append(cirq.X(self.data_qubits[j + 1]))

    def Z(self, i: int | Iterable[int]):
        """
        Apply the Logical Z gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]

        for j in i:
            if j < 0 or j >= self.k:
                raise ValueError("Index out of range")

        s = set()
        for j in i:
            if j not in s:
                s.add(j)
            else:
                raise ValueError("Duplicate index")

        # @TODO: Patch complement rule to minimize weight
        # if len(s) > self.k / 2:
        #     s = set(range(self.k)) - s

        if len(s) & 1:
            self.circuit.append(cirq.Z(self.data_qubits[-1]))
        for j in s:
            self.circuit.append(cirq.Z(self.data_qubits[j + 1]))

    def S(self, i: int | Iterable[int]):
        """
        Apply the Logical S gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]

        for j in i:
            if j < 0 or j >= self.k:
                raise ValueError("Index out of range")

        s = set()
        for j in i:
            if j not in s:
                s.add(j)
            else:
                raise ValueError("Duplicate index")

        for j in s:
            self.circuit.append(cirq.S(self.data_qubits[-1]))
            self.circuit.append(cirq.CZ(self.data_qubits[j + 1], self.data_qubits[-1]))
            self.circuit.append(cirq.S(self.data_qubits[j + 1]))

    def H(self, i: int | Iterable[int]):
        """
        Apply the Logical H gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]

        for j in i:
            if j < 0 or j >= self.k:
                raise ValueError("Index out of range")

        s = set()
        for j in i:
            if j not in s:
                s.add(j)
            else:
                raise ValueError("Duplicate index")

        if len(s) == self.k:
            for i in range(self.n):
                self.circuit.append(cirq.H(self.data_qubits[i]))
            # @TODO: remove swap and insert relabelling (caveat - relabelling shouldn't mess with the final measurement)
            self.circuit.append(cirq.SWAP(self.data_qubits[0], self.data_qubits[-1]))
            return

        for j in s:
            self.circuit.append(cirq.CZ(self.data_qubits[j + 1], self.data_qubits[-1]))
            self.circuit.append(cirq.H(self.data_qubits[0]))
            self.circuit.append(cirq.H(self.data_qubits[j + 1]))
            self.circuit.append(cirq.CZ(self.data_qubits[0], self.data_qubits[j + 1]))
            self.circuit.append(cirq.H(self.data_qubits[j + 1]))
            self.circuit.append(cirq.CZ(self.data_qubits[0], self.data_qubits[j + 1]))
            self.circuit.append(cirq.H(self.data_qubits[0]))
            self.circuit.append(cirq.H(self.data_qubits[j + 1]))
            self.circuit.append(cirq.CZ(self.data_qubits[j + 1], self.data_qubits[-1]))
            self.circuit.append(cirq.X(self.data_qubits[0]))
            self.circuit.append(cirq.Z(self.data_qubits[-1]))

    def HH(self):
        """
        Apply HH to this Iceberg code:
        - For n=4, apply H to all data qubits, then swap qubits 1 and 2.
        """
        if self.n != 4:
            raise ValueError("HH operation is only implemented for n=4")

        # Apply Hadamard only to data qubits
        for q in self.data_qubits:
            self.circuit.append(cirq.H(q))

        # Swap qubits 1 and 2 in the data block
        self.circuit.append(cirq.SWAP(self.data_qubits[1], self.data_qubits[2]))

    def CNOT(self, i: int, j: int, faulty: bool = False):
        """
        Logical CNOT between qubits i (control) and j (target).

        • Fault-tolerant (default): depth-4 sequence of logical CNOTs.
        • faulty=True on n=4:
            0→1  ⇒  SWAP(q0, q1)
            1→0  ⇒  SWAP(q0, q2)
        (Any other pair raises NotImplementedError.)
        """
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("control and target must differ")

        # —— Non-fault-tolerant “single-swap” shortcuts ————————————————
        if faulty and self.n == 4:
            if i == 0 and j == 1:
                self.circuit.append(cirq.SWAP(self.data_qubits[0],
                                            self.data_qubits[1]))
                return
            if i == 1 and j == 0:
                self.circuit.append(cirq.SWAP(self.data_qubits[0],
                                            self.data_qubits[2]))
                return
            raise NotImplementedError("faulty CNOT only defined for (0→1) or (1→0)")

        # —— Standard distance-2 implementation ——————————————
        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[0]))
        self.circuit.append(cirq.CNOT(self.data_qubits[-1],     self.data_qubits[0]))
        self.circuit.append(cirq.CNOT(self.data_qubits[-1],     self.data_qubits[j + 1]))
        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[j + 1]))


    def CZ(self, i: int, j: int):
        """
        Apply the Logical CZ gate to the i-th and j-th qubits.
        """
        if i < 0 or i >= self.k:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        self.circuit.append(cirq.CZ(self.data_qubits[i + 1], self.data_qubits[j + 1]))
        self.circuit.append(cirq.Z(self.data_qubits[-1]))
        self.circuit.append(cirq.CZ(self.data_qubits[i + 1], self.data_qubits[-1]))
        self.circuit.append(cirq.CZ(self.data_qubits[j + 1], self.data_qubits[-1]))

    def RZZ(self, i: int, j: int, rads: float):
        """
        Apply the Logical RZZ gate to the i-th and j-th qubits.
        """
        if i < 0 or i >= self.k:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[j + 1]))
        self.circuit.append(cirq.Rz(rads=rads)(self.data_qubits[j + 1]))
        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[j + 1]))

    def RZ(self, i: int, rads: float):
        """
        Apply the Logical RZ gate to the i-th qubit.
        """
        if i < 0 or i >= self.k:
            raise ValueError("Index out of range")

        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[-1]))
        self.circuit.append(cirq.Rz(rads=rads)(self.data_qubits[-1]))
        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[-1]))

    def initialize_arbitrary_state(self, indices: Iterable[int] | None = None):
        """
        Prepare Haar-random states on `indices` logical qubits of *this* Iceberg
        patch *before* calling `encode()`.  If `indices` is None, initialise
        all `self.k` logical qubits.
        """
        if indices is None:
            indices = range(self.k)
        elif isinstance(indices, int):
            indices = [indices]

        for j in indices:
            θ, φ = _rand_clifford_angles()
            q = self.data_qubits[j + 1]
            self.circuit.append(cirq.rz(φ).on(q))
            self.circuit.append(cirq.H(q))
            self.circuit.append(cirq.rz(θ).on(q))
            self.circuit.append(cirq.H(q))

    def barrier(self):
        all_qubits = list(self.data_qubits) + list(self.parity_qubits)
        barrier_gate = BarrierGate()
        for q in all_qubits:
            self.circuit.append(barrier_gate.on(q))

    def swap(self, i: int, j: int, faulty: bool = False):
        """
        Logical SWAP between qubits i and j.

        • Fault-tolerant (default): three logical CNOTs.
        • faulty=True on n=4 and {i,j} == {0,1}: single SWAP(q1,q2).
        """
        if i < 0 or i >= self.k or j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("Cannot swap a qubit with itself")

        # —— Non-fault-tolerant shortcut ————————————————
        if faulty and self.n == 4 and {i, j} == {0, 1}:
            self.circuit.append(cirq.SWAP(self.data_qubits[1],
                                        self.data_qubits[2]))
            return

        # —— Standard distance-2 implementation ——————————
        self.CNOT(i, j, faulty=faulty)
        self.CNOT(j, i, faulty=faulty)
        self.CNOT(i, j, faulty=faulty)


class IcebergGrid(CircuitClass):
    """
    A grid of Iceberg codes.
    """

    def __init__(
        self,
        n_program: int,
        n: int,
        circuit: cirq.Circuit,
        initialize: str | None = None,
    ):
        """
        Initialize the IcebergGrid with n_program qubits encoded using [[n, n-2, 2]] Iceberg qubits.
        """
        if n & 1:
            raise ValueError("n must be even")
        self.n_program = n_program
        self.n = n
        self.k = n - 2
        self.num_icebergs = (self.n_program + self.k - 1) // self.k
        self.icebergs = [
            IcebergCode(n, circuit, (n + 2) * i) for i in range(self.num_icebergs)
        ]
        self.circuit = circuit
        if initialize is not None:
            self.encode(initialize)

    def encode(self, initialize: str | None = None):
        """
        Initialize all Icebergs to the \\ket{0^{n-2}_L} state.
        """
        if self._encoded:
            warning(f"Class {self.__class__.__name__} is already encoded")
            return
        self._encoded = True
        # Apply the encoding circuit
        for i, iceberg in enumerate(self.icebergs):
            iceberg.encode(
                initialize
                if initialize is None
                else initialize[self.k * i : self.k * (i + 1)]
            )

    def decode(self):
        for idx, iceberg in enumerate(self.icebergs):
            iceberg.decode(key_prefix=f"patch{idx}_")

    def measure(self):
        # for idx, iceberg in enumerate(self.icebergs[::-1]):
        for idx, iceberg in enumerate(self.icebergs[::-1]):
            self.circuit.append(
                cirq.measure(*iceberg.data_qubits, key=f"data_{idx}")
            )

    def X(self, i: int | Iterable[int]):
        """
        Apply the Logical X gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")

        s: list[set[int]] = [set() for _ in range(self.num_icebergs)]
        for j in i:
            iceberg_index = j // self.k
            qubit_index = j % self.k
            if qubit_index not in s[iceberg_index]:
                s[iceberg_index].add(qubit_index)
            else:
                raise ValueError("Duplicate index")

        for iceberg_index, iceberg in enumerate(self.icebergs):
            iceberg.X(s[iceberg_index])

    def Z(self, i: int | Iterable[int]):
        """
        Apply the Logical Z gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")

        s: list[set[int]] = [set() for _ in range(self.num_icebergs)]
        for j in i:
            iceberg_index = j // self.k
            qubit_index = j % self.k
            if qubit_index not in s[iceberg_index]:
                s[iceberg_index].add(qubit_index)
            else:
                raise ValueError("Duplicate index")

        for iceberg_index, iceberg in enumerate(self.icebergs):
            iceberg.Z(s[iceberg_index])

    def S(self, i: int | Iterable[int]):
        """
        Apply the Logical S gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")

        s: list[set[int]] = [set() for _ in range(self.num_icebergs)]
        for j in i:
            iceberg_index = j // self.k
            qubit_index = j % self.k
            if qubit_index not in s[iceberg_index]:
                s[iceberg_index].add(qubit_index)
            else:
                raise ValueError("Duplicate index")

        for iceberg_index, iceberg in enumerate(self.icebergs):
            iceberg.S(s[iceberg_index])

    def H(self, i: int | Iterable[int]):
        """
        Apply the Logical H gate to the i-th qubit.
        """
        if isinstance(i, int):
            i = [i]
        for j in i:
            if j < 0 or j >= self.n_program:
                raise ValueError("Index out of range")

        s: list[set[int]] = [set() for _ in range(self.num_icebergs)]
        for j in i:
            iceberg_index = j // self.k
            qubit_index = j % self.k
            if qubit_index not in s[iceberg_index]:
                s[iceberg_index].add(qubit_index)
            else:
                raise ValueError("Duplicate index")

        for iceberg_index, iceberg in enumerate(self.icebergs):
            iceberg.H(s[iceberg_index])

    def HH(self, patches: int | Iterable[int]):
        """
        Apply HH to all specified patches (Iceberg codes).
        If patches is an int, apply to that single patch.
        """
        if isinstance(patches, int):
            patches = [patches]
        
        for p in patches:
            if p < 0 or p >= self.num_icebergs:
                raise ValueError(f"Patch index {p} out of range")
            self.icebergs[p].HH()

    def CNOT(self, i: int, j: int, faulty: bool = False):
        """
        Grid-level logical CNOT (same-patch only for now).
        """
        if i < 0 or i >= self.n_program or j < 0 or j >= self.n_program:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("control and target must differ")

        patch_i, q_i = divmod(i, self.k)
        patch_j, q_j = divmod(j, self.k)

        if patch_i != patch_j:
            raise NotImplementedError("Cross-patch CNOT not yet supported")

        self.icebergs[patch_i].CNOT(q_i, q_j, faulty=faulty)


    def CNOT_k_Transversal(self, i: int, j: int):
        """
        Apply the Logical CNOT gate between the i-th and j-th Iceberg patches.
        """
        if i < 0 or i >= self.num_icebergs:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.num_icebergs:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        control_qubits = self.icebergs[i].data_qubits
        target_qubits = self.icebergs[j].data_qubits
        for i in range(self.n):
            self.circuit.append(cirq.CNOT(control_qubits[i], target_qubits[i]))

    def CZ(self, i: int, j: int):
        """
        Apply the Logical CZ gate to the i-th and j-th qubits.
        """
        if i < 0 or i >= self.n_program:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.n_program:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        iceberg_index_i = i // self.k
        qubit_index_i = i % self.k
        iceberg_index_j = j // self.k
        qubit_index_j = j % self.k
        if iceberg_index_i == iceberg_index_j:
            self.icebergs[iceberg_index_i].CZ(qubit_index_i, qubit_index_j)
            return

        # @TODO: implement this
        raise NotImplementedError("CZ not implemented yet")

    def CZ_k_Transversal(self, i: int, j: int):
        """
        Apply the Logical CZ gate between the i-th and j-th Iceberg patches.
        """
        if i < 0 or i >= self.num_icebergs:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.num_icebergs:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        # @WARN: This is not Fault-Tolerant, find a way to fix this
        control_qubits = self.icebergs[i].data_qubits
        target_qubits = self.icebergs[j].data_qubits
        self.circuit.append(cirq.CZ(control_qubits[0], target_qubits[-1]))
        for i in range(1, self.n - 1):
            self.circuit.append(cirq.CZ(control_qubits[i], target_qubits[i]))
        self.circuit.append(cirq.CZ(control_qubits[-1], target_qubits[0]))

    def RZZ(self, i: int, j: int, rads: float):
        """
        Apply the Logical RZZ gate to the i-th and j-th qubits.
        """
        if i < 0 or i >= self.n_program:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.n_program:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        # @TODO: implement this
        raise NotImplementedError("RZZ not implemented yet")

    def RZ(self, i: int, rads: float):
        """
        Apply the Logical RZ gate to the i-th qubit.
        """
        if i < 0 or i >= self.n_program:
            raise ValueError("Index out of range")

        iceberg_index = i // self.k
        qubit_index = i % self.k
        self.icebergs[iceberg_index].RZ(qubit_index, rads)

    def initialize_arbitrary_state(
        self,
        codes: Iterable[int] | None = None,
        indices: Iterable[int] | None = None,
    ):
        """
        Initialise Haar-random states on specified logical qubits across patches.

        Parameters
        ----------
        codes
            Patch indices (0 … `self.num_icebergs`-1) to act on.  `None` → all.
        indices
            Logical-qubit indices *within each selected patch*.
            `None` → all logical qubits in that patch.
        """
        if codes is None:
            codes = range(self.num_icebergs)
        elif isinstance(codes, int):
            codes = [codes]

        for ci in codes:
            self.icebergs[ci].initialize_arbitrary_state(indices)

    def barrier(self, patches: int | Iterable[int] | None = None):
        barrier_gate = BarrierGate()

        if patches is None:
            # All patches
            qubits = []
            for iceberg in self.icebergs:
                qubits.extend(iceberg.data_qubits)
                qubits.extend(iceberg.parity_qubits)
        else:
            if isinstance(patches, int):
                patches = [patches]
            qubits = []
            for p in patches:
                qubits.extend(self.icebergs[p].data_qubits)
                qubits.extend(self.icebergs[p].parity_qubits)

        # Add all barriers in ONE Moment for alignment
        self.circuit.append(cirq.Moment([barrier_gate.on(q) for q in qubits]))

    def swap(self, swaps: list[list[int]], faulty: bool = False):
        """
        Grid-level logical SWAP(s).

        Each entry in `swaps` is [patch_idx, q0, q1] in *logical* indices
        relative to that patch.
        """
        for code_idx, q0, q1 in swaps:
            if code_idx < 0 or code_idx >= self.num_icebergs:
                raise ValueError("patch index out of range")

            if q0 == q1:
                raise ValueError("Cannot swap a qubit with itself")

            self.icebergs[code_idx].swap(q0, q1, faulty=faulty)
