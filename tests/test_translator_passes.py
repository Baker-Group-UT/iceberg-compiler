"""
The translator's peephole passes must preserve what the circuit computes.

`commute_hadamards` rewrites gates into a Hadamard-conjugated basis, which is
where three bugs lived: it dropped the basis change when a SWAP moved the qubit,
its single-qubit substitution table was missing `sx`/`sxdg` and mapped `sdg` to
`S` instead of `SX-dagger`, and it could not consume the `xcx` gates it emits
itself. All three silently produced a different circuit.

These tests fuzz each pass against the unitary it started from, which is the
property that actually matters, and pin the substitution table against the
matrices it claims to implement.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate
from qiskit.quantum_info import Operator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.random_programs import random_clifford_program, random_program_with_rotations
from translator.custom_gates import XCXGate
from translator.passes import (
    _H_CONJUGATE,
    cancel_inverse_gates,
    cancel_inverse_gates_through_swaps,
    commute_hadamards,
)

PASSES = [commute_hadamards, cancel_inverse_gates, cancel_inverse_gates_through_swaps]


def assert_same_unitary(before: QuantumCircuit, after: QuantumCircuit, label: str) -> None:
    assert np.allclose(Operator(before).data, Operator(after).data, atol=1e-9), (
        f"{label}: pass changed the unitary\n  before:\n{before}\n  after:\n{after}"
    )


@pytest.mark.parametrize("pass_fn", PASSES, ids=lambda f: f.__name__)
@pytest.mark.parametrize("num_qubits", [2, 3, 4])
def test_passes_preserve_clifford_unitaries(pass_fn, num_qubits):
    rng = random.Random(hash((pass_fn.__name__, num_qubits)) % 10_000)
    for _ in range(60):
        program = random_clifford_program(num_qubits, 12, rng=rng, include_xcx=True)
        assert_same_unitary(program, pass_fn(program), pass_fn.__name__)


@pytest.mark.parametrize("pass_fn", PASSES, ids=lambda f: f.__name__)
def test_passes_preserve_non_clifford_unitaries(pass_fn):
    rng = random.Random(4711)
    for _ in range(40):
        program = random_program_with_rotations(3, 10, rng=rng)
        assert_same_unitary(program, pass_fn(program), pass_fn.__name__)


def test_h_conjugate_table_matches_the_matrices():
    """Every entry of the substitution table must really equal H G H."""
    hadamard = HGate().to_matrix()
    for name, replace in _H_CONJUGATE.items():
        source = QuantumCircuit(1)
        if name in ("rx", "rz"):
            getattr(source, name)(0.9137, 0)
        else:
            getattr(source, name)(0)
        original = Operator(source).data
        replacement = Operator(_single_gate_circuit(replace(source.data[0].operation))).data
        assert np.allclose(hadamard @ original @ hadamard, replacement, atol=1e-9), (
            f"H {name} H is not {replace(source.data[0].operation).name}"
        )


def _single_gate_circuit(gate) -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    circuit.append(gate, [0])
    return circuit


def test_hadamard_follows_the_qubit_through_a_swap():
    """H(0); SWAP(0,1) must become SWAP(0,1); H(1), not H(0)."""
    program = QuantumCircuit(2)
    program.h(0)
    program.swap(0, 1)
    result = commute_hadamards(program)

    assert_same_unitary(program, result, "commute_hadamards over swap")
    trailing = [
        result.find_bit(instruction.qubits[0]).index
        for instruction in result.data
        if instruction.operation.name == "h"
    ]
    assert trailing == [1], f"the trailing H should land on qubit 1, got {trailing}"


def test_xcx_is_consumed_by_the_pass_that_emits_it():
    """
    The pass turns `cx` under an active frame into `xcx`, so it has to be able to
    read `xcx` back in. It used to emit the gate unchanged in the wrong basis.
    """
    for active in ((0,), (1,), (0, 1)):
        program = QuantumCircuit(2)
        for qubit in active:
            program.h(qubit)
        program.append(XCXGate(), [0, 1])
        assert_same_unitary(program, commute_hadamards(program), f"xcx h={active}")


def test_unknown_gate_flushes_the_frame():
    """
    An operation the pass does not understand must not be reinterpreted; the
    pending Hadamards are re-emitted before it instead.
    """
    from qiskit.circuit.library import CCXGate

    program = QuantumCircuit(3)
    program.h(0)
    program.append(CCXGate(), [0, 1, 2])
    result = commute_hadamards(program)
    assert_same_unitary(program, result, "unknown gate")
    assert result.data[0].operation.name == "h", "frame should be flushed before the gate"


def test_xcx_gate_has_a_definition():
    """
    Without a definition the gate is opaque and `Clifford`, `Operator` and Aer all
    fail on any program containing it.
    """
    expected = QuantumCircuit(2)
    expected.h(0)
    expected.h(1)
    expected.cz(0, 1)
    expected.h(0)
    expected.h(1)
    assert np.allclose(
        Operator(XCXGate()).data, Operator(expected).data, atol=1e-9
    ), "XCXGate is not H(x)H . CZ . H(x)H"
