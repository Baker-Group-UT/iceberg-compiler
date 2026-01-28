"""Custom macro gates used by merging passes."""

from __future__ import annotations

from typing import Sequence

from qiskit.circuit import Gate


class HHGate(Gate):
    """Custom logical placeholder for a merged HH gate."""

    def __init__(self):
        super().__init__("hh", 2, [])

    def inverse(self):
        return HHGate()


class XXGate(Gate):
    """Custom logical placeholder for a merged XX gate."""

    def __init__(self):
        super().__init__("xx", 2, [])

    def inverse(self):
        return XXGate()


class ZZGate(Gate):
    """Custom logical placeholder for a merged ZZ gate."""

    def __init__(self):
        super().__init__("zz", 2, [])

    def inverse(self):
        return ZZGate()


class _TransversalMacroGate(Gate):
    """Base helper to stash metadata for transversal macros."""

    def __init__(
        self,
        name: str,
        num_qubits: int,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(name, num_qubits, [])
        self.metadata = {
            "control_patch": control_patch,
            "target_patch": target_patch,
            "control_count": control_count,
            "logical_qubits": list(logical_qubits),
        }


class CXTGate(_TransversalMacroGate):
    def __init__(
        self,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(
            "cxt",
            len(logical_qubits),
            control_patch=control_patch,
            target_patch=target_patch,
            control_count=control_count,
            logical_qubits=logical_qubits,
        )


class CZTGate(_TransversalMacroGate):
    def __init__(
        self,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(
            "czt",
            len(logical_qubits),
            control_patch=control_patch,
            target_patch=target_patch,
            control_count=control_count,
            logical_qubits=logical_qubits,
        )
