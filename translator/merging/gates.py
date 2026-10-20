"""Custom macro gates used by merging passes."""

from __future__ import annotations

from typing import Sequence

from qiskit.circuit import Gate


class _MultiLaneMacroGate(Gate):
    """
    A macro standing in for the same logical gate on several lanes of one patch.

    The width is the number of lanes fused, which is 2 only at k=2. The names stay
    "hh"/"xx"/"zz" whatever the width, so the handler tables, the H-commutation
    pass and `MACRO_NAMES` keep working unchanged.
    """

    def __init__(self, name: str, num_qubits: int = 2) -> None:
        if num_qubits < 2:
            raise ValueError(f"{name} macro needs at least two lanes, got {num_qubits}")
        super().__init__(name, num_qubits, [])

    def inverse(self, annotated: bool = False):
        return type(self)(self.num_qubits)


class HHGate(_MultiLaneMacroGate):
    """Whole-patch Hadamard: H on every lane, plus the anchor swap.

    Because it is whole-patch, it is only a valid substitution when it covers
    every *active* lane of the patch -- see `translator.handlers.hadamard`.
    """

    def __init__(self, num_qubits: int = 2) -> None:
        super().__init__("hh", num_qubits)


class XXGate(_MultiLaneMacroGate):
    """Logical X on several lanes of one patch.

    Unlike HH this needs no coverage condition: `Xbar_i = X[x_anchor] X[lane i]`, so
    a subset of lanes is a valid operation and an even-sized subset cancels the
    anchor factor outright.
    """

    def __init__(self, num_qubits: int = 2) -> None:
        super().__init__("xx", num_qubits)


class ZZGate(_MultiLaneMacroGate):
    """Logical Z on several lanes of one patch. Z-type dual of `XXGate`."""

    def __init__(self, num_qubits: int = 2) -> None:
        super().__init__("zz", num_qubits)


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


class XCXTGate(_TransversalMacroGate):
    def __init__(
        self,
        *,
        control_patch: int,
        target_patch: int,
        control_count: int,
        logical_qubits: Sequence[int],
    ) -> None:
        super().__init__(
            "xcxt",
            len(logical_qubits),
            control_patch=control_patch,
            target_patch=target_patch,
            control_count=control_count,
            logical_qubits=logical_qubits,
        )
