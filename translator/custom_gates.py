"""Custom Qiskit gate definitions used by the translator."""

from __future__ import annotations

from qiskit.circuit import Gate


class XCXGate(Gate):
    """X-controlled-X gate (H⊗H · CZ · H⊗H)."""

    def __init__(self):
        super().__init__("xcx", 2, [])

    def inverse(self):
        return XCXGate()
