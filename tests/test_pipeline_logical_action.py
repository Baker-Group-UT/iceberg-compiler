"""
Exact end-to-end test of the compiler: for a random Clifford program, the logical
Clifford induced by the *translated physical circuit* must equal the program.

This is the strongest correctness statement available, and it needs no sampling.
It covers everything the distribution-based harness in
`test_pipeline_roundtrip.py` covers except encode/decode/postselect, and unlike
that harness it is exact — a wrong gate cannot hide inside shot noise, and it
distinguishes S from S-dagger.

What it exercises, per random program:
  - placement under the fault-tolerance rules, including the SWAPs the translator
    inserts to align slots
  - the merging pass's macro gates (at n=4, where merging runs)
  - the peephole passes (`cancel_inverse_gates`, `commute_hadamards`)
  - every gate construction in `iceberg/`

Accounting for the permutation
------------------------------
The physical circuit acts on lanes, not on program qubits, and the translator
moves qubits between lanes: program qubit i starts on the lane named by the
permutation and ends on the lane named by the translator's final
`state_positions`. So the statement being checked is, for every program qubit i
and each of P = X, Z::

    U (Pbar on lane start[i]) U†  ==  relabel(Program(P on qubit i), end)

where `relabel` sends program qubit j to lane end[j]. Comparing generator by
generator like this sidesteps having to reconstruct the induced permutation over
*all* lanes, which is not recoverable from `state_positions` alone once a qubit
has been swapped into a lane that started idle.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from qiskit.quantum_info import Clifford, Pauli

from iceberg.utils import logical_capacity
from tests.logical_action import logical_clifford_of, physical_pauli, unitary_part
from tests.random_programs import permutations_for, random_clifford_program
from translator.config import TranslatorConfig
from translator.core import Translator

# n=8 is included but sparingly: the tableau work grows with the patch count.
CODE_SIZES = [4, 6, 8]
PROGRAMS_PER_CASE = 12
DEPTH = 14


def translate(program: QuantumCircuit, permutation, n_code: int, **config):
    """Translate with encode/decode suppressed, so the circuit stays unitary."""
    translator = Translator(
        config=TranslatorConfig(n_code=n_code, encode_override=True, **config)
    )
    return translator.translate(program, permutation)


def lane_of(position: Sequence[int], k: int) -> int:
    patch, slot = position
    return patch * k + slot


def relabel_to_lanes(pauli: Pauli, lanes: Sequence[int], num_lanes: int) -> Pauli:
    """Move a q-qubit Pauli onto `num_lanes` lanes, sending qubit j to lanes[j]."""
    z = np.zeros(num_lanes, dtype=bool)
    x = np.zeros(num_lanes, dtype=bool)
    for qubit, lane in enumerate(lanes):
        z[lane] = pauli.z[qubit]
        x[lane] = pauli.x[qubit]
    out = Pauli((z, x))
    return -out if pauli.phase % 4 == 2 else out


def check_program(program: QuantumCircuit, permutation, n_code: int, **config) -> None:
    """
    Assert the translated circuit's logical action is the program's.

    Compared one generator at a time, so a failure names the qubit and basis that
    disagree rather than dumping two tableaus.
    """
    k = logical_capacity(n_code)
    result = translate(program, permutation, n_code, **config)
    modules = result.context.iceberg_set.modules
    num_lanes = len(modules) * k

    start = [lane_of(position, k) for position in permutation]
    end = [lane_of(position, k) for position in result.context.state_positions]

    induced = logical_clifford_of(result.circuit, modules)
    expected = Clifford(unitary_part(program))
    label = f"n={n_code} perm={list(permutation)} config={config}"

    for qubit in range(program.num_qubits):
        for basis in ("X", "Z"):
            source = physical_pauli(num_lanes, {start[qubit]: basis})
            got = source.evolve(induced, frame="s")

            want_small = physical_pauli(program.num_qubits, {qubit: basis}).evolve(
                expected, frame="s"
            )
            want = relabel_to_lanes(want_small, end, num_lanes)

            assert got == want, (
                f"{label}\n"
                f"  {basis}bar on program qubit {qubit} (lane {start[qubit]})\n"
                f"    got  {got}\n"
                f"    want {want}\n"
                f"  start lanes={start} end lanes={end}"
            )


# --------------------------------------------------------------------------
# random Clifford programs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_random_clifford_programs_translate_exactly(n_code):
    """The headline test: random Clifford programs, tightest packing, several n."""
    rng = random.Random(20250726 + n_code)
    k = logical_capacity(n_code)
    num_qubits = 2 * k  # exactly two patches, so cross-patch traffic is exercised

    checked = 0
    for _ in range(PROGRAMS_PER_CASE):
        program = random_clifford_program(num_qubits, DEPTH, rng=rng, include_xcx=True)
        for permutation in permutations_for(num_qubits, n_code, rng=rng, count=2):
            try:
                check_program(program, permutation, n_code)
            except RuntimeWarning:
                # Placement rejected under the fault-tolerance rules; the
                # permutation is infeasible, which is a translator policy
                # decision rather than a correctness failure.
                continue
            checked += 1
    assert checked > 0, f"no program was placeable at n={n_code}"


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_random_clifford_programs_fault_tolerant(n_code):
    """Same, with the flagged gadgets switched on."""
    rng = random.Random(777 + n_code)
    k = logical_capacity(n_code)
    num_qubits = 2 * k

    checked = 0
    for _ in range(4):
        program = random_clifford_program(num_qubits, 8, rng=rng)
        for permutation in permutations_for(num_qubits, n_code, rng=rng, count=1):
            try:
                check_program(program, permutation, n_code, fault_tolerant=True)
            except RuntimeWarning:
                continue
            checked += 1
    assert checked > 0, f"no program was placeable at n={n_code} in FT mode"


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("enable_merging", [True, False])
def test_merging_preserves_the_logical_action(n_code, enable_merging):
    """
    Merging rewrites gates into whole-patch and transversal macros; it must not
    change what the circuit computes.

    Random programs rarely trigger the whole-patch macros -- they need a gate on
    every lane at once -- so this establishes that merging does no harm rather than
    that it fires. `test_merging_generalized.py` builds the programs that trigger
    each macro and checks those.
    """
    rng = random.Random((4242 if enable_merging else 2424) + n_code)
    k = logical_capacity(n_code)
    checked = 0
    for _ in range(8):
        program = random_clifford_program(2 * k, DEPTH, rng=rng, include_xcx=True)
        for permutation in permutations_for(2 * k, n_code, rng=rng, count=2):
            try:
                check_program(
                    program, permutation, n_code, enable_merging=enable_merging
                )
            except RuntimeWarning:
                continue
            checked += 1
    # Not asserted non-zero: at k >= 8 the fault-tolerance rules reject nearly every
    # random permutation, which `test_merging_generalized.py` sidesteps by placing
    # sequentially. Asserting here would pin a placement-policy detail.


def test_uneven_packing_leaves_idle_slots_alone():
    """A program that does not fill its last patch must not disturb idle slots."""
    rng = random.Random(31337)
    for n_code in (6, 8):
        k = logical_capacity(n_code)
        for num_qubits in (k + 1, k + 2):
            program = random_clifford_program(num_qubits, 10, rng=rng)
            permutation = permutations_for(num_qubits, n_code, rng=rng, count=1)[0]
            try:
                check_program(program, permutation, n_code)
            except RuntimeWarning:
                continue


def test_single_patch_programs():
    """Everything inside one patch: no cross-patch construction should appear."""
    rng = random.Random(555)
    for n_code in CODE_SIZES:
        k = logical_capacity(n_code)
        program = random_clifford_program(k, DEPTH, rng=rng, include_xcx=True)
        permutation = [(0, slot) for slot in range(k)]
        try:
            check_program(program, permutation, n_code)
        except RuntimeWarning:
            pytest.fail(f"single-patch program should always be placeable at n={n_code}")
