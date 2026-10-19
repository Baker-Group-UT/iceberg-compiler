"""
The distribution-level pipeline harness: random circuits through the whole stack.

For each random program this runs the path a real experiment runs — translate,
simulate, postselect on the syndromes, reduce to the tracked logical positions —
and asserts the resulting logical distribution matches the unencoded program's.

That covers what `test_pipeline_logical_action.py` cannot: encoding, syndrome
extraction, the classical decoder, and `reduce_distribution`'s use of the
translator's final state positions. It is a weaker check gate-for-gate (it is
sampled, and a distribution is blind to relative phases), so the two are
complementary — the exact test catches wrong gates, this one catches wrong
plumbing.

Fidelity is the primary assertion because it is quadratically insensitive to shot
noise: sampling error moves it by ~1e-4 while a single misplaced gate moves it to
~0.5. TVD is asserted with a looser bound as a sanity check.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.utils import logical_capacity
from analysis.postselection import postselect_against_reference
from simulator.config import SimulatorConfig
from simulator.simulator import simulate_batch
from tests.random_programs import permutations_for, random_clifford_program, random_program_with_rotations
from translator.config import TranslatorConfig
from translator.core import Translator

CODE_SIZES = [4, 6]
# The backend-agreement test reaches one code size further than the rest of the file.
# It is Clifford-only (Stim cannot take the generic-angle rotations the other tests
# use) and therefore cheap enough to afford n=8, where the [[8, 6, 2]] patch is the
# first size at which *both* a lane permutation and an anchor mix-up are visible.
BACKEND_CODE_SIZES = [4, 6, 8]
SHOTS = 20_000
# Both TVD *and* fidelity grow with the support size under sampling, so both bounds
# are derived per-distribution rather than fixed. A random Clifford on 8 logical
# qubits spreads over up to 256 outcomes, where shot noise alone gives TVD ~0.05 and
# a Bhattacharyya deficit of ~0.003 at 20k shots.
#
# An earlier fixed MIN_FIDELITY = 0.999 was reasoned as having "enormous margin"
# because a single wrong gate collapses fidelity to ~0.5. The first half is right and
# the second is not: the deficit from sampling alone straddles 0.999 once the support
# passes ~160 outcomes, which made the n=6 case fail about 1 run in 70 whether or not
# merging was enabled. The derived floor keeps the margin against a real error --
# three orders of magnitude -- without the false positives.
TVD_SAMPLING_SLACK = 4.0
TVD_FLOOR = 0.02
FIDELITY_SAMPLING_SLACK = 4.0
FIDELITY_CEILING_DEFICIT = 0.001


def translate_all(program: QuantumCircuit, permutations: List, n_code: int, **config):
    """
    Translate every placement, dropping the ones the FT rules reject.

    Returns an empty list when every placement was rejected, which is a translator
    policy decision rather than a failure; callers assert on the total number placed
    across programs.
    """
    translator = Translator(config=TranslatorConfig(n_code=n_code, **config))
    translated = []
    for permutation in permutations:
        try:
            translated.append(translator.translate(program, permutation))
        except RuntimeWarning:
            continue  # placement rejected under the FT rules
    return translated


def simulate_and_postselect(program: QuantumCircuit, translated: List, *, stim: bool = False):
    """Simulate the translations noiselessly on one backend and postselect them."""
    # Aer cannot assemble the `xcx` instruction by name, so the reference
    # simulation gets an unrolled copy. The translator handles xcx natively and
    # keeps the original.
    ideal, noisy = simulate_batch(
        _aer_ready(program),
        translated,
        SimulatorConfig(repetitions=SHOTS, noiseless=True, stim=stim),
        desc="sim",
    )
    return postselect_against_reference(translated, ideal, noisy)


def run_pipeline(program: QuantumCircuit, permutations: List, n_code: int, **config):
    """
    Translate, simulate and postselect — the same path a real scan takes.

    Returns an empty list when every placement was rejected; see `translate_all`.
    """
    translated = translate_all(program, permutations, n_code, **config)
    if not translated:
        return []
    return simulate_and_postselect(program, translated)


def _aer_ready(program: QuantumCircuit) -> QuantumCircuit:
    if any(instruction.operation.name == "xcx" for instruction in program.data):
        return program.decompose(gates_to_decompose=["xcx"])
    return program


def assert_matches_unencoded(results, label: str) -> None:
    for result in results:
        floor = fidelity_floor(result.postselected_ideal_dist)
        assert result.logical_fidelity >= floor, (
            f"{label} perm={result.permutation}: fidelity "
            f"{result.logical_fidelity:.4f} < {floor:.4f} "
            f"(support {len(result.postselected_ideal_dist)})\n"
            f"  encoded : {_top(result.postselected_noisy_dist)}\n"
            f"  expected: {_top(result.postselected_ideal_dist)}"
        )
        budget = tvd_budget(result.postselected_ideal_dist)
        assert result.logical_total_variation_distance <= budget, (
            f"{label} perm={result.permutation}: TVD "
            f"{result.logical_total_variation_distance:.4f} > {budget:.4f} "
            f"(support {len(result.postselected_ideal_dist)})\n"
            f"  encoded : {_top(result.postselected_noisy_dist)}\n"
            f"  expected: {_top(result.postselected_ideal_dist)}"
        )
        assert result.logical_selection_rate == pytest.approx(1.0, abs=1e-6), (
            f"{label}: noiseless run should keep every shot, got "
            f"{result.logical_selection_rate}"
        )


def fidelity_floor(reference: dict) -> float:
    """
    How far below 1 the Bhattacharyya coefficient can fall from sampling alone.

    Both distributions are sampled, so with p_hat = p + d and q_hat = p + e drawn
    independently, sqrt(p_hat q_hat) ~ p + d/2 + e/2 - (d^2 + e^2)/(8p). The linear
    terms sum to zero and E[d^2] = E[e^2] = p(1-p)/N, leaving

        E[BC] ~ 1 - sum (1 - p) / (4N) = 1 - (S - 1) / (4N)

    for a support of S outcomes. So a wide distribution has to be allowed a larger
    deficit than a narrow one, exactly as for TVD.

    The floor is never stricter than `FIDELITY_CEILING_DEFICIT` below 1, so a narrow
    distribution still gets a tight bound.
    """
    if not reference:
        return 1.0 - FIDELITY_CEILING_DEFICIT
    expected_deficit = (len(reference) - 1) / (4.0 * SHOTS)
    deficit = max(FIDELITY_CEILING_DEFICIT, FIDELITY_SAMPLING_SLACK * expected_deficit)
    return 1.0 - deficit


def tvd_budget(reference: dict) -> float:
    """
    How much total-variation distance multinomial sampling alone can explain.

    E|p_hat - p| ~ sqrt(2 p (1-p) / (pi N)) per outcome, and TVD is half their
    sum, so a wide distribution tolerates more than a narrow one.
    """
    import math

    if not reference:
        return TVD_FLOOR
    expected = 0.5 * sum(
        math.sqrt(2.0 * p * (1.0 - p) / (math.pi * SHOTS)) for p in reference.values()
    )
    return max(TVD_FLOOR, TVD_SAMPLING_SLACK * expected)


def cross_backend_tvd_budget(reference: dict) -> float:
    """
    Budget for the TVD *between two backends*, both of which are sampled.

    `tvd_budget` bounds one sample against the exact distribution. Here neither side
    is exact: with p_hat and q_hat drawn independently, E|p_hat - q_hat| per outcome
    picks up the variance of both, which is sqrt(2) times the one-sided figure.
    """
    import math

    return math.sqrt(2.0) * tvd_budget(reference)


def distribution_tvd(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _top(distribution, count: int = 6) -> dict:
    if not distribution:
        return {}
    ordered = sorted(distribution.items(), key=lambda kv: -kv[1])[:count]
    return {key: round(value, 4) for key, value in ordered}


# --------------------------------------------------------------------------
# random Clifford circuits — the headline harness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_random_clifford_circuits_match_unencoded(n_code):
    rng = random.Random(90210 + n_code)
    k = logical_capacity(n_code)
    num_qubits = 2 * k

    placed = 0
    for index in range(4):
        program = random_clifford_program(
            num_qubits, 12, rng=rng, include_xcx=True, measure=True
        )
        permutations = permutations_for(num_qubits, n_code, rng=rng, count=3)
        results = run_pipeline(program, permutations, n_code)
        assert_matches_unencoded(results, f"n={n_code} clifford #{index}")
        placed += len(results)
    assert placed > 0, f"n={n_code}: no random Clifford program was placeable"


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_random_clifford_circuits_fault_tolerant(n_code):
    rng = random.Random(1234 + n_code)
    k = logical_capacity(n_code)
    placed = 0
    for index in range(4):
        program = random_clifford_program(2 * k, 6, rng=rng, measure=True)
        permutations = permutations_for(2 * k, n_code, rng=rng, count=3)
        results = run_pipeline(program, permutations, n_code, fault_tolerant=True)
        assert_matches_unencoded(results, f"n={n_code} clifford FT #{index}")
        placed += len(results)
    assert placed > 0, f"n={n_code}: nothing placeable in FT mode"


def test_swap_containing_programs_match_unencoded():
    """
    Programs with SWAPs, which used to come out with the wrong distribution.

    `apply_logical_swap` emitted the physical SWAP *and* relabelled the two
    program qubits; the two cancel, so the gate was never performed. Independently,
    `commute_hadamards` moved Hadamards across SWAPs without following the qubit.
    Both are fixed; this pins them.
    """
    for n_code in CODE_SIZES:
        k = logical_capacity(n_code)
        # Keep the SWAPs inside one patch: the translator rejects cross-patch
        # SWAPs, so a cross-patch one would test nothing.
        program = QuantumCircuit(2 * k)
        program.h(0)
        program.swap(0, 1)
        program.cx(1, k)
        program.h(1)
        program.swap(0, 1)
        program.h(0)
        program.measure_all()
        permutations = [[(0, slot) for slot in range(k)] + [(1, slot) for slot in range(k)]]
        results = run_pipeline(program, permutations, n_code)
        assert results, f"n={n_code}: intra-patch swap program should be placeable"
        assert_matches_unencoded(results, f"n={n_code} swaps")


def test_alignment_swap_then_intra_patch_gate():
    """
    A misaligned cross-patch CX (which inserts an alignment SWAP) followed by an
    intra-patch CZ/XCX on the qubits that moved.

    `place_logical_cz` and `place_logical_xcx` used to address their intra-patch
    branch with the static qubit ids rather than the current slots, so the gate
    landed on whichever qubits had moved into the original slots.
    """
    for n_code in CODE_SIZES:
        k = logical_capacity(n_code)
        for second in ("cz", "xcx", "cx"):
            program = QuantumCircuit(2 * k)
            program.h(0)
            program.h(k + 1)
            program.cx(k + 1, 0)  # cross-patch, misaligned slots
            getattr(program, second)(k, k + 1) if second != "xcx" else _xcx(program, k, k + 1)
            program.measure_all()
            permutations = [
                [(0, slot) for slot in range(k)] + [(1, slot) for slot in range(k)]
            ]
            results = run_pipeline(program, permutations, n_code)
            assert_matches_unencoded(results, f"n={n_code} align+{second}")


def _xcx(program: QuantumCircuit, a: int, b: int) -> None:
    from translator.custom_gates import XCXGate

    program.append(XCXGate(), [a, b])


# --------------------------------------------------------------------------
# non-Clifford circuits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_programs_with_rotations_match_unencoded(n_code):
    """
    Smaller and fewer, since rz/rx at generic angles put the state outside the
    stabilizer formalism and can only be checked this way.
    """
    rng = random.Random(31415 + n_code)
    k = logical_capacity(n_code)
    for index in range(2):
        program = random_program_with_rotations(k, 10, rng=rng, measure=True)
        permutations = [[(0, slot) for slot in range(k)]]
        results = run_pipeline(program, permutations, n_code)
        assert results, f"n={n_code}: single-patch program should be placeable"
        assert_matches_unencoded(results, f"n={n_code} rotations #{index}")


def test_single_rotation_angles_are_faithful():
    """
    RZ and RX at a generic angle must produce the right *distribution*, which
    pins the angle's magnitude end to end through encode/decode.
    """
    import math

    for n_code in CODE_SIZES:
        k = logical_capacity(n_code)
        for gate, angle in (("rx", 0.7), ("rx", math.pi / 3), ("rz", 1.1)):
            program = QuantumCircuit(k)
            program.h(0)  # so an rz has an observable effect
            getattr(program, gate)(angle, 0)
            program.h(0)
            program.measure_all()
            permutations = [[(0, slot) for slot in range(k)]]
            results = run_pipeline(program, permutations, n_code)
            assert results, f"n={n_code}: single-patch program should be placeable"
            assert_matches_unencoded(results, f"n={n_code} {gate}({angle:.3f})")


# --------------------------------------------------------------------------
# backend agreement — Stim against Aer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_code", BACKEND_CODE_SIZES)
def test_stim_matches_aer_noiseless(n_code):
    """
    Both simulator backends must produce the same logical distribution.

    Everything else in this file runs Aer. Stim was left unexercised, and it had
    been returning a logical distribution with every classical register bit-reversed
    -- `_format_key_by_cregs` emitted Qiskit's *printed* little-endian order while
    `iceberg_postselect` reads position `d` of a data register as data qubit `d`.

    n=4 cannot see that. `CodeLayout.decode_logical` is reversal-invariant on a 4-bit
    even-parity codeword, so the k=2 numbers were right and nothing upstream
    complained; at n=6 the same reversal permutes the lanes and mixes in the wrong
    anchor, and TVD went from the 0.05 sampling floor to 0.78 while still looking
    like a distribution. Hence the assertions here run at every code size, and hence
    the cross-backend comparison rather than only a per-backend one against the
    reference: a shared convention error upstream of both would pass the latter.

    The two backends are compared on identical translations, so any disagreement is
    the simulate-and-decode path and not placement.
    """
    rng = random.Random(5150 + n_code)
    k = logical_capacity(n_code)

    placed = 0
    # Single-patch programs are always placeable and carry the intra-patch gate set;
    # two-patch ones add cross-patch traffic and slot alignment, and are rejected
    # often enough at n=8 that they cannot be the only source of coverage.
    for index, num_qubits in enumerate((k, 2 * k, 2 * k)):
        program = random_clifford_program(
            num_qubits, 12, rng=rng, include_xcx=True, measure=True
        )
        permutations = permutations_for(num_qubits, n_code, rng=rng, count=3)
        translated = translate_all(program, permutations, n_code)
        if not translated:
            continue
        label = f"n={n_code} #{index} ({num_qubits}q)"

        aer = simulate_and_postselect(program, translated, stim=False)
        stim = simulate_and_postselect(program, translated, stim=True)
        assert_matches_unencoded(aer, f"{label} aer")
        assert_matches_unencoded(stim, f"{label} stim")

        for aer_result, stim_result in zip(aer, stim):
            reference = aer_result.postselected_ideal_dist
            budget = cross_backend_tvd_budget(reference)
            observed = distribution_tvd(
                aer_result.postselected_noisy_dist, stim_result.postselected_noisy_dist
            )
            assert observed <= budget, (
                f"{label} perm={aer_result.permutation}: backends disagree, "
                f"TVD(stim, aer) {observed:.4f} > {budget:.4f} "
                f"(support {len(reference)})\n"
                f"  aer : {_top(aer_result.postselected_noisy_dist)}\n"
                f"  stim: {_top(stim_result.postselected_noisy_dist)}"
            )
        placed += len(translated)

    assert placed > 0, f"n={n_code}: nothing placeable, so nothing was compared"
