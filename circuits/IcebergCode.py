import cirq


class IcebergCode:
    """
    [[n, n-2, 2]] quantum error detection code.
    """

    def __init__(self, n: int, circuit: cirq.Circuit):
        """
        Initialize the IcebergCode with n qubits.
        """
        if n & 1:
            raise ValueError("n must be even")
        self.n = n
        self.k = n - 2
        self.data_qubits = cirq.LineQubit.range(n)
        self.parity_qubits = cirq.LineQubit.range(2)
        self.circuit = circuit
        self._encoded = False
        self.encode()

    def encode(self):
        """
        Initialize the qubit to the \\ket{0^{n-2}_L} state.
        """
        if self._encoded:
            return
        self._encoded = True

        # Apply the encoding circuit
        self.circuit.append(cirq.H(self.data_qubits[0]))
        for i in range(1, self.n):
            self.circuit.append(cirq.CNOT(self.data_qubits[0], self.data_qubits[i]))

    def decode(self):
        """
        Add parity check measurements.
        """
        for i in range(self.n):
            self.circuit.append(cirq.CNOT(self.data_qubits[i], self.parity_qubits[0]))
        self.circuit.append(cirq.measure(self.parity_qubits[0], key="parity_X"))

        self.circuit.append(cirq.H(*self.data_qubits))
        for i in range(self.n):
            self.circuit.append(cirq.CNOT(self.data_qubits[i], self.parity_qubits[1]))
        self.circuit.append(cirq.H(*self.data_qubits))
        self.circuit.append(cirq.measure(self.parity_qubits[1], key="parity_Z"))

    def measure(self):
        """
        Measure the data qubits.
        """
        self.circuit.append(cirq.measure(*self.data_qubits, key="measure"))

    def X(self, i: int | list[int]):
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

    def Z(self, i: int | list[int]):
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

    def S(self, i: int | list[int]):
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

    def H(self, i: int | list[int]):
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

        for j in s:
            self.circuit.append(cirq.CZ(self.data_qubits[j + 1], self.data_qubits[-1]))
            self.circuit.append(
                cirq.H(self.data_qubits[0]), cirq.H(self.data_qubits[j + 1])
            )
            self.circuit.append(cirq.CZ(self.data_qubits[0], self.data_qubits[j + 1]))
            self.circuit.append(cirq.H(self.data_qubits[j + 1]))
            self.circuit.append(cirq.CZ(self.data_qubits[0], self.data_qubits[j + 1]))
            self.circuit.append(
                cirq.H(self.data_qubits[0]), cirq.H(self.data_qubits[j + 1])
            )
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
