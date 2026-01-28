"""
helpers.random_circuits.py - lightweight random circuit builders.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional

from qiskit import QuantumCircuit

from translator.passes import cancel_inverse_gates

_ONE_QUBIT_GATES = {"h", "x", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "y"}
_TWO_QUBIT_GATES = {"cx", "cz", "swap", "cy"}


def random_gate_circuit(
    num_qubits: int,
    num_gates: int,
    gates: Optional[Iterable[str]] = None,
    *,
    seed: Optional[int] = None,
    max_attempts: int = 50_000,
) -> QuantumCircuit:
    """
    Build a random circuit with uneven qubit depths and no immediate gate repeats.

    The returned circuit is post-processed with `cancel_inverse_gates` and will
    contain at least `num_gates` operations after cancellation.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    if num_gates < 0:
        raise ValueError("num_gates must be >= 0")

    if gates is None:
        gates = ["h", "x", "z", "s", "cx", "cz"]

    gate_list: List[str] = [str(name).lower() for name in gates]
    if not gate_list:
        return QuantumCircuit(num_qubits)

    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)

    last_1q: List[Optional[str]] = [None] * num_qubits
    last_per_qubit: List[Optional[tuple[str, Optional[int], Optional[str]]]] = [None] * num_qubits

    def _append_one_qubit(name: str) -> bool:
        qubits = list(range(num_qubits))
        rng.shuffle(qubits)
        q = None
        for candidate in qubits:
            if last_1q[candidate] != name:
                q = candidate
                break
        if q is None:
            return False
        getattr(qc, name)(q)
        last_1q[q] = name
        last_per_qubit[q] = (name, None, None)
        return True

    def _append_two_qubit(name: str) -> bool:
        if num_qubits < 2:
            return False
        candidates = [
            (control, target)
            for control in range(num_qubits)
            for target in range(num_qubits)
            if control != target
        ]
        rng.shuffle(candidates)
        pair = None
        for control, target in candidates:
            if (
                last_per_qubit[control] == (name, target, "control")
                and last_per_qubit[target] == (name, control, "target")
            ):
                continue
            pair = (control, target)
            break
        if pair is None:
            return False
        control, target = pair
        getattr(qc, name)(control, target)
        last_1q[control] = None
        last_1q[target] = None
        last_per_qubit[control] = (name, target, "control")
        last_per_qubit[target] = (name, control, "target")
        return True

    cleaned = cancel_inverse_gates(qc)
    attempts = 0
    while len(cleaned.data) < num_gates:
        if attempts >= max_attempts:
            raise RuntimeError("Failed to build circuit without excessive cancellations.")
        attempts += 1
        name = rng.choice(gate_list)
        if name in _ONE_QUBIT_GATES:
            appended = _append_one_qubit(name)
        elif name in _TWO_QUBIT_GATES:
            appended = _append_two_qubit(name)
        else:
            raise ValueError(f"Unsupported gate name: {name}")

        if appended:
            cleaned = cancel_inverse_gates(qc)

    return cleaned
