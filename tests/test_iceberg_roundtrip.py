"""
Round-trip convention tests: encode -> logical ops -> decode -> measure ->
postselect, compared against the same program run unencoded.

These are the tests that pin the *conventions* rather than the gate
constructions: measurement bit order, the logical-index order of the decoded
string, and the decoder's choice of anchor. All three used to be implicit in a
hardcoded k=2 lookup table, and all three are invisible at n=4 — a 4-bit
even-parity codeword decodes to the same logical value whether you read the
register forwards or backwards, and the k=2 anchor mix-up is a lane swap that the
table absorbed. So every test here runs at n=6 as well, where the conventions
have observable consequences.

The end-to-end cases go through `simulator.Simulator` rather than Aer directly,
so that the endianness handling under test is the production one.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.demo import demo_program, encode_program, logical_distribution
from iceberg.demo import reference_distribution as demo_reference
from iceberg.iceberg_postselect import decode_data_register, iceberg_postselect
from iceberg.iceberg_set import IcebergSet
from iceberg.layout import CodeLayout
from simulator.simulator import Simulator, SimulatorConfig

CODE_SIZES = [4, 6, 8]


# --------------------------------------------------------------------------
# decoder, against codewords built straight from the code definition
# --------------------------------------------------------------------------


def codeword_registers(n: int, logical: tuple[int, ...]) -> list[str]:
    """
    The two data-register strings of the codeword |logical>_L.

    |0..0>_L = (|0^n> + |1^n>)/sqrt(2) and Xbar_i = X[x_anchor] X[lane_i], so
    applying the X's to |0^n> gives one branch and its complement is the other.
    """
    layout = CodeLayout(n)
    bits = [0] * n
    for i, value in enumerate(logical):
        if value:
            bits[layout.x_anchor] ^= 1
            bits[layout.lane(i)] ^= 1
    complement = [1 - b for b in bits]
    return ["".join(map(str, bits)), "".join(map(str, complement))]


@pytest.mark.parametrize("n", CODE_SIZES)
def test_decoder_inverts_the_encoding(n):
    """Every codeword branch must decode back to the logical string it encodes."""
    layout = CodeLayout(n)
    for logical in product([0, 1], repeat=layout.k):
        want = "".join(map(str, logical))
        for register in codeword_registers(n, logical):
            assert decode_data_register(register) == want, (
                f"n={n}: register {register} should decode to {want}"
            )


@pytest.mark.parametrize("n", CODE_SIZES)
def test_decoder_rejects_non_codewords(n):
    """Odd-parity outcomes are detected errors, not logical values."""
    for weight_one in range(n):
        register = "".join("1" if i == weight_one else "0" for i in range(n))
        assert decode_data_register(register) is None


@pytest.mark.parametrize("n", CODE_SIZES)
def test_decoding_against_the_wrong_anchor_differs_beyond_k2(n):
    """
    The natural-looking `bit(lane) XOR bit(x_anchor)` rule decodes to C_A . x.

    At k=2 that is a lane swap and looks like an endianness convention; beyond it
    the two rules disagree in a way no relabeling can fix. This test documents
    the trap so a future rewrite cannot silently reintroduce it.
    """
    layout = CodeLayout(n)
    disagreements = 0
    for logical in product([0, 1], repeat=layout.k):
        register = codeword_registers(n, logical)[0]
        bits = [int(b) for b in register]
        wrong = [bits[layout.lane(i)] ^ bits[layout.x_anchor] for i in range(layout.k)]
        parity = sum(wrong) % 2
        # The wrong rule is exactly C_A applied to the right answer.
        assert [w ^ parity for w in wrong] == layout.decode_logical(bits)
        if "".join(map(str, wrong)) != "".join(map(str, logical)):
            disagreements += 1
    if layout.k == 2:
        # k=2: C_A is the lane swap, so it only shows up on |01> and |10>.
        assert disagreements == 2
    else:
        assert disagreements > 2


def test_postselect_rejects_syndromes_and_orders_patches():
    key_order = ["m_parity_0", "m_parity_1", "flag0", "m_data_0", "m_data_1"]
    dist = {
        "00 00 0 0011 0110": 0.5,  # patch0 -> '10', patch1 -> '11'
        "01 00 0 0011 0110": 0.2,  # parity syndrome -> rejected
        "00 00 1 0011 0110": 0.2,  # flag syndrome  -> rejected
        "00 00 0 0001 0110": 0.1,  # odd parity     -> rejected
    }
    logical_dist, sel_rate = iceberg_postselect(dist, key_order)
    assert logical_dist == {"1011": 1.0}
    assert sel_rate == pytest.approx(0.5)


def test_postselect_handles_empty_distribution():
    assert iceberg_postselect({}, ["m_data_0"]) == ({}, 0.0)


# --------------------------------------------------------------------------
# end-to-end through the production simulator
# --------------------------------------------------------------------------


def reference_distribution(program: QuantumCircuit) -> dict[str, float]:
    """
    Exact output distribution of the unencoded program, keyed the same way the
    postselected logical distribution is: qubit index 0 leftmost.
    """
    probabilities = Statevector.from_instruction(program).probabilities_dict()
    # Statevector keys are printed MSB-first; the logical strings are qubit 0
    # first, so reverse each key.
    return {key[::-1]: p for key, p in probabilities.items() if p > 1e-12}


SHOTS = 40_000
# Noiseless, so the only discrepancy is sampling error: ~0.0025 at 40k shots,
# leaving the tolerance below at roughly 8 sigma.
TOLERANCE = 0.02
# Seeded, so a failure here reproduces exactly. This used to be `None` on purpose:
# `Simulator` splits a run into `repetitions // 100` chunks, and a fixed seed once made
# every chunk a replay of the first, leaving `repetitions / 100` independent samples and
# a sampling error ten times the one assumed above. Chunks now draw their own seeds from
# the configured one; `tests/test_simulator_seeding.py` pins that.
SEED = 20260727


def encode_and_run(n: int, program: QuantumCircuit, apply_logical) -> dict[str, float]:
    """Encode `program` by hand onto one patch, simulate, and postselect."""
    circuit = QuantumCircuit()
    iceberg = IcebergSet(n_program=program.num_qubits, n=n, circuit=circuit)
    iceberg.encode()
    apply_logical(iceberg)
    iceberg.decode()
    iceberg.measure()

    simulator = Simulator(
        SimulatorConfig(repetitions=SHOTS, noiseless=True, seed=SEED)
    ).with_shot_progress(False)
    result = simulator.run(circuit)
    logical_dist, sel_rate = iceberg_postselect(result.ideal_dist, result.key_order)
    assert sel_rate == pytest.approx(1.0), "noiseless run should never be rejected"
    return logical_dist


def assert_same_distribution(got: dict[str, float], want: dict[str, float]) -> None:
    keys = set(got) | set(want)
    for key in keys:
        assert got.get(key, 0.0) == pytest.approx(want.get(key, 0.0), abs=TOLERANCE), (
            f"outcome {key}: got {got.get(key, 0.0)}, want {want.get(key, 0.0)}"
        )


@pytest.mark.parametrize("n", [4, 6])
def test_roundtrip_x_on_each_lane(n):
    """X on lane i must show up as a 1 at logical position i, and nowhere else."""
    layout = CodeLayout(n)
    for lane in layout.lanes():
        program = QuantumCircuit(layout.k)
        program.x(lane)
        got = encode_and_run(n, program, lambda ib, lane=lane: ib.X(lane))
        assert_same_distribution(got, reference_distribution(program))


@pytest.mark.parametrize("n", [4, 6])
def test_roundtrip_ghz_via_intra_patch_cnots(n):
    """A GHZ chain exercises H, CNOT and the decoder's lane ordering together."""
    layout = CodeLayout(n)
    program = QuantumCircuit(layout.k)
    program.h(0)
    for lane in range(1, layout.k):
        program.cx(0, lane)

    def apply(iceberg):
        iceberg.H_at(0, 0)
        for lane in range(1, layout.k):
            iceberg.CNOT(0, lane)

    assert_same_distribution(
        encode_and_run(n, program, apply), reference_distribution(program)
    )


@pytest.mark.parametrize("n", [4, 6])
def test_roundtrip_whole_patch_hadamard(n):
    """HH must be H on every lane — an equal superposition over all k bits."""
    layout = CodeLayout(n)
    program = QuantumCircuit(layout.k)
    for lane in layout.lanes():
        program.h(lane)

    assert_same_distribution(
        encode_and_run(n, program, lambda ib: ib.HH([0])),
        reference_distribution(program),
    )


@pytest.mark.parametrize("n", [4, 6])
def test_roundtrip_asymmetric_program(n):
    """
    A program whose output distribution is not symmetric under bit reversal or
    under C_A, so it fails if either convention is wrong.
    """
    layout = CodeLayout(n)
    program = QuantumCircuit(layout.k)
    program.h(0)
    program.cx(0, 1)
    program.x(1)
    if layout.k > 2:
        program.h(2)

    def apply(iceberg):
        iceberg.H_at(0, 0)
        iceberg.CNOT(0, 1)
        iceberg.X(1)
        if layout.k > 2:
            iceberg.H_at(0, 2)

    assert_same_distribution(
        encode_and_run(n, program, apply), reference_distribution(program)
    )


@pytest.mark.parametrize("n", [4, 6])
def test_roundtrip_cross_patch_cnot(n):
    """Two patches: pins the patch-major order of the decoded logical string."""
    layout = CodeLayout(n)
    k = layout.k
    program = QuantumCircuit(2 * k)
    program.h(0)
    program.cx(0, k)  # patch 0 slot 0 -> patch 1 slot 0
    program.x(k + 1)

    def apply(iceberg):
        iceberg.H_at(0, 0)
        iceberg.CNOT(0, k)
        iceberg.X(k + 1)

    assert_same_distribution(
        encode_and_run(n, program, apply), reference_distribution(program)
    )


@pytest.mark.parametrize("n", [4, 6])
def test_roundtrip_cross_patch_cnot_misaligned_slots(n):
    """Cross-patch CNOT between *different* slots, which k=2 could not express."""
    layout = CodeLayout(n)
    k = layout.k
    program = QuantumCircuit(2 * k)
    program.h(0)
    program.cx(0, k + (k - 1))

    def apply(iceberg):
        iceberg.H_at(0, 0)
        iceberg.CNOT(0, k + (k - 1))

    assert_same_distribution(
        encode_and_run(n, program, apply), reference_distribution(program)
    )


# --------------------------------------------------------------------------
# the reference encoder behind `python -m iceberg.demo`
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", CODE_SIZES)
def test_demo_reference_encoder_matches_unencoded_program(n):
    """
    `iceberg.demo.encode_program` must reproduce the program it encodes.

    This is the whole gate set at once, including the paths only the reference
    encoder reaches (cross-patch SWAP, misaligned cross-patch CZ), and at n=8 it
    also covers projecting away idle slots.
    """
    program = demo_program(4)
    program.swap(1, 2)  # cross-patch at n=4; the translator refuses this one
    circuit, _ = encode_program(program, n)
    got, selection_rate = logical_distribution(circuit, program.num_qubits, shots=SHOTS)
    assert selection_rate == pytest.approx(1.0)
    assert_same_distribution(got, demo_reference(program))


def test_demo_reference_encoder_rejects_unsupported_gates():
    program = QuantumCircuit(2)
    program.t(0)
    with pytest.raises(NotImplementedError, match="not in the reference encoder"):
        encode_program(program, 4)
