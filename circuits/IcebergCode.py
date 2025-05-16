from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from logging import warning
from typing import Iterable
import cirq


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

    def decode(self):
        """
        Add parity check measurements.
        """
        for i in range(self.n):
            self.circuit.append(cirq.CNOT(self.data_qubits[i], self.parity_qubits[0]))
        self.circuit.append(cirq.measure(self.parity_qubits[0]))

        for i in range(self.n):
            self.circuit.append(cirq.H(self.data_qubits[i]))
            self.circuit.append(cirq.CNOT(self.data_qubits[i], self.parity_qubits[1]))
            self.circuit.append(cirq.H(self.data_qubits[i]))
        self.circuit.append(cirq.measure(self.parity_qubits[1]))

    def measure(self):
        """
        Measure the data qubits.
        """
        self.circuit.append(cirq.measure(*self.data_qubits))

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
        if len(s) > self.k / 2:
            s = set(range(self.k)) - s

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
        if len(s) > self.k / 2:
            s = set(range(self.k)) - s

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

    def CNOT(self, i: int, j: int):
        """
        Apply the Logical CNOT gate to the i-th and j-th qubits.
        """
        if i < 0 or i >= self.k:
            raise ValueError("Index out of range")
        if j < 0 or j >= self.k:
            raise ValueError("Index out of range")
        if i == j:
            raise ValueError("i and j must be different")

        self.circuit.append(cirq.CNOT(self.data_qubits[i + 1], self.data_qubits[0]))
        self.circuit.append(cirq.CNOT(self.data_qubits[-1], self.data_qubits[0]))
        self.circuit.append(cirq.CNOT(self.data_qubits[-1], self.data_qubits[j + 1]))
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
        """
        Add parity check measurements.
        """
        for iceberg in self.icebergs:
            iceberg.decode()

    def measure(self):
        """
        Measure the data qubits.
        """
        for iceberg in self.icebergs:
            iceberg.measure()

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

    def CNOT(self, i: int, j: int):
        """
        Apply the Logical CNOT gate to the i-th and j-th qubits.
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
            self.icebergs[iceberg_index_i].CNOT(qubit_index_i, qubit_index_j)
            return

        # @TODO: implement this
        raise NotImplementedError("CNOT across patches not implemented yet")

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
