"""
random_programs.py — test support: random logical programs the translator accepts.

The translator's gate set is fixed by `translator.gate_handlers.TABLE_SIMPLE` and
`TABLE_COMPLEX`. Generating from that set (rather than from Qiskit's
`random_clifford`, which transpiles to gates the translator has never seen) is
what makes these programs a fair stress test of placement, merging and the
peephole passes.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Sequence

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__all__ = [
    "CLIFFORD_1Q",
    "CLIFFORD_2Q",
    "random_clifford_program",
    "random_program_with_rotations",
]

# Clifford generators the translator handles directly.
CLIFFORD_1Q = ("h", "s", "sx", "x", "z")
# `xcx` needs an explicit Gate object, so it is added separately below.
CLIFFORD_2Q = ("cx", "cz", "swap")


def random_clifford_program(
    num_qubits: int,
    depth: int,
    *,
    rng: random.Random,
    two_qubit_ratio: float = 0.5,
    include_xcx: bool = False,
    measure: bool = False,
) -> QuantumCircuit:
    """
    A random Clifford circuit over the translator's gate set.

    `depth` counts emitted gates, not circuit layers. `two_qubit_ratio` is the
    probability that each gate is a two-qubit one.
    """
    if num_qubits < 2:
        raise ValueError("need at least 2 qubits")
    circuit = QuantumCircuit(num_qubits)
    two_qubit_names: List[str] = list(CLIFFORD_2Q)
    if include_xcx:
        two_qubit_names.append("xcx")

    for _ in range(depth):
        if rng.random() < two_qubit_ratio:
            control, target = rng.sample(range(num_qubits), 2)
            name = rng.choice(two_qubit_names)
            _append_two_qubit(circuit, name, control, target)
        else:
            qubit = rng.randrange(num_qubits)
            getattr(circuit, rng.choice(CLIFFORD_1Q))(qubit)

    if measure:
        circuit.measure_all()
    return circuit


def random_program_with_rotations(
    num_qubits: int,
    depth: int,
    *,
    rng: random.Random,
    measure: bool = False,
) -> QuantumCircuit:
    """
    A random circuit that also contains rz/rx at generic (non-Clifford) angles.

    Kept separate because such a circuit has no Clifford tableau, so it can only
    be checked against a statevector rather than by the exact tableau comparison.
    """
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.3:
            control, target = rng.sample(range(num_qubits), 2)
            _append_two_qubit(circuit, rng.choice(CLIFFORD_2Q), control, target)
        elif roll < 0.6:
            # Generic angle: not a multiple of pi/2, so Clifford aliasing cannot
            # let an incorrect construction pass.
            angle = rng.uniform(0.1, 2.9)
            getattr(circuit, rng.choice(("rz", "rx")))(angle, rng.randrange(num_qubits))
        else:
            getattr(circuit, rng.choice(CLIFFORD_1Q))(rng.randrange(num_qubits))
    if measure:
        circuit.measure_all()
    return circuit


def _append_two_qubit(circuit: QuantumCircuit, name: str, control: int, target: int) -> None:
    if name == "xcx":
        from translator.custom_gates import XCXGate

        circuit.append(XCXGate(), [control, target])
    else:
        getattr(circuit, name)(control, target)


def permutations_for(
    num_qubits: int,
    n_code: int,
    *,
    rng: random.Random,
    count: int,
) -> List[List[Sequence[int]]]:
    """
    A sample of distinct placements of `num_qubits` logical qubits at code size n.

    Always includes the tightest packing (fill each patch before opening the next)
    since that is where cross-patch traffic and slot misalignment are worst.
    """
    from iceberg.utils import logical_capacity

    k = logical_capacity(n_code)
    num_patches = -(-num_qubits // k)
    slots = [(p, s) for p in range(num_patches) for s in range(k)][:num_qubits]

    out = [list(slots)]
    seen = {tuple(slots)}
    attempts = 0
    while len(out) < count and attempts < 50 * count:
        attempts += 1
        candidate = slots.copy()
        rng.shuffle(candidate)
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out
