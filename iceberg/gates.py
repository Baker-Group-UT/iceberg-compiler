import cirq


class BarrierGate(cirq.Gate):
    """
    A visual-only 1-qubit gate that inserts a barrier (rendered as '||')
    for human-readable separation of moments. Has no effect on execution.
    """

    def num_qubits(self) -> int:
        return 1

    def _circuit_diagram_info_(self, args) -> str:
        return "||"

    def __repr__(self) -> str:
        return "BarrierGate()"