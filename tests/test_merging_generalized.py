"""
Merging at k > 2: does every macro still compute the program?

`test_pipeline_logical_action.py` runs random Clifford programs, and those almost
never place an H on all k lanes at once or an aligned CX on all k lane pairs. So the
whole-patch macros -- the ones that make merging worth having -- go unexercised there
at k > 2, and the random harness reports a vacuous pass. The programs here are built
to trigger each macro, then checked with the same exact tableau comparison.

The negative cases matter as much as the positive ones. A whole-patch macro applied
with only some lanes covered would silently add the gate to the rest, so
`test_partial_hadamard_coverage_does_not_fuse` pins the refusal.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.utils import logical_capacity
from tests.test_pipeline_logical_action import check_program
from translator.merging import preprocess_program_for_merging_with_stats
from translator.passes import cancel_inverse_gates, commute_hadamards

CODE_SIZES = [4, 6, 8, 10, 12]
MACRO_NAMES = ("hh", "xx", "zz", "cxt", "czt", "xcxt")


def sequential(num_qubits: int, k: int) -> List[Tuple[int, int]]:
    return [(index // k, index % k) for index in range(num_qubits)]


def macros_emitted(
    program: QuantumCircuit, permutation, n_code: int
) -> collections.Counter:
    """Macros the pass puts in the circuit, after the same peepholes the translator runs."""
    processed = cancel_inverse_gates(commute_hadamards(cancel_inverse_gates(program)))
    rewritten, *_ = preprocess_program_for_merging_with_stats(
        processed,
        list(permutation),
        allow_transversal_swap_merges=True,
        n_code=n_code,
    )
    return collections.Counter(
        instruction.operation.name.lower()
        for instruction in rewritten.data
        if instruction.operation.name.lower() in MACRO_NAMES
    )


def check_merged(program: QuantumCircuit, permutation, n_code: int) -> None:
    check_program(
        program,
        permutation,
        n_code,
        enable_merging=True,
        allow_transversal_swap_merges=True,
    )


# --- whole-patch Hadamard ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_all_lane_hadamard_fuses_and_is_exact(n_code):
    """One H per lane becomes a single k-lane HH, and still computes H^(x)k."""
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    for qubit in range(2 * k):
        program.h(qubit)
    permutation = sequential(2 * k, k)

    assert macros_emitted(program, permutation, n_code)["hh"] == 2, (
        "each full patch of Hadamards should collapse to one whole-patch macro"
    )
    check_merged(program, permutation, n_code)


@pytest.mark.parametrize("n_code", [6, 8, 10, 12])
def test_partial_hadamard_coverage_does_not_fuse(n_code):
    """
    Two H's in a k>2 patch must not become an HH.

    HH is H on every lane, so fusing here would apply Hadamards the program never
    asked for. At k=2 the same input *is* full coverage and does fuse, which is
    covered by `test_all_lane_hadamard_fuses_and_is_exact`.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    program.h(0)
    program.h(1)
    permutation = sequential(2 * k, k)

    assert macros_emitted(program, permutation, n_code)["hh"] == 0
    check_merged(program, permutation, n_code)


def test_partial_hadamard_coverage_fuses_at_k2():
    """The same two H's are the whole patch at k=2, so there they should fuse."""
    program = QuantumCircuit(4)
    program.h(0)
    program.h(1)
    permutation = sequential(4, 2)
    assert macros_emitted(program, permutation, 4)["hh"] == 1


# --- transversal two-qubit macros ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("gate,macro", [("cx", "cxt"), ("cz", "czt")])
def test_aligned_cross_patch_gates_fuse_and_are_exact(n_code, gate, macro):
    """Lane i to lane i on every lane is exactly the transversal construction."""
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    for lane in range(k):
        getattr(program, gate)(lane, k + lane)
    permutation = sequential(2 * k, k)

    assert macros_emitted(program, permutation, n_code)[macro] == 1
    check_merged(program, permutation, n_code)


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize(
    "pairing",
    ["reversed", "rotated"],
    ids=["reversed-slots", "rotated-slots"],
)
def test_misaligned_cross_patch_gates_realign_and_are_exact(n_code, pairing):
    """
    A group whose slot map is a non-identity permutation still fuses.

    The target patch is permuted by routing swaps around the macro. At k=2 the only
    possibility is a single transposition; these cases are a full reversal and a
    k-cycle, which need up to k-1 transpositions and so only exist past k=2.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    for lane in range(k):
        partner = (k - 1 - lane) if pairing == "reversed" else (lane + 1) % k
        program.cx(lane, k + partner)
    permutation = sequential(2 * k, k)

    assert macros_emitted(program, permutation, n_code)["cxt"] == 1
    check_merged(program, permutation, n_code)


@pytest.mark.parametrize("n_code", [6, 8])
def test_partial_cross_patch_coverage_does_not_fuse(n_code):
    """
    Cross-patch gates on a subset of lanes must not become a transversal macro.

    Partial transversal is not merely unprofitable, it is not a logical operation:
    on a subset of positions it sends the full-weight stabilizer X^(x)n outside the
    stabilizer group.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    program.cx(0, k + 0)
    program.cx(1, k + 1)
    permutation = sequential(2 * k, k)

    emitted = macros_emitted(program, permutation, n_code)
    assert emitted["cxt"] == 0, f"fused {2} of {k} lanes into a whole-patch macro"
    check_merged(program, permutation, n_code)


# --- Pauli macros, which have no coverage condition ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("gate,macro", [("x", "xx"), ("z", "zz")])
def test_pauli_fuses_on_any_subset_and_is_exact(n_code, gate, macro):
    """
    `Pbar_i` shares an anchor with every other, so a subset is legal and cheap.

    Checked on a strict subset precisely because that is what distinguishes these
    from the whole-patch macros.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    lanes = list(range(0, k, 2)) or [0]
    for lane in lanes:
        getattr(program, gate)(lane)
    permutation = sequential(2 * k, k)

    if len(lanes) >= 2:
        assert macros_emitted(program, permutation, n_code)[macro] == 1
    check_merged(program, permutation, n_code)


# --- macros in sequence, and against gates that move lanes ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_macros_compose_with_each_other_and_with_paulis(n_code):
    """A whole-patch H, a transversal CX and a Pauli layer in one program."""
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    for lane in range(k):
        program.h(lane)
    for lane in range(k):
        program.cx(lane, k + lane)
    for lane in range(k):
        program.z(lane)
    permutation = sequential(2 * k, k)

    assert sum(macros_emitted(program, permutation, n_code).values()) >= 2
    check_merged(program, permutation, n_code)


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_program_swap_between_macros_stays_exact(n_code):
    """
    A SWAP relabels slots, so macros either side of it must not be fused together.

    The pass flushes its candidate group at a SWAP rather than reason across it;
    this checks the result is still correct, whatever it decides to emit.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    for lane in range(k):
        program.h(lane)
    program.swap(0, 1)
    for lane in range(k):
        program.h(lane)
    permutation = sequential(2 * k, k)
    check_merged(program, permutation, n_code)


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_merging_never_changes_the_logical_action(n_code):
    """
    Merging on and merging off must agree, on a program with fusable structure.

    The strongest statement available about the pass: whatever it chooses to fuse,
    the circuit computes the same thing.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(2 * k)
    for lane in range(k):
        program.h(lane)
    for lane in range(k):
        program.cx(lane, k + ((lane + 1) % k))
    for lane in range(k):
        program.x(lane)
        program.h(k + lane)
    permutation = sequential(2 * k, k)

    for enable_merging in (False, True):
        check_program(
            program,
            permutation,
            n_code,
            enable_merging=enable_merging,
            allow_transversal_swap_merges=True,
        )


# --- the packer's analyzer, which calls the same detectors ---


def test_pair_merge_analyzer_still_works():
    """
    `packer.merge_packer` reaches the detectors through `PairMergeAnalyzer`.

    It hands the transversal detector a single lane per patch, which is a shape the
    real pass never produces because `_build_patch_lanes` drops patches holding one
    qubit. That made it the one caller the k-generalization could break without any
    test noticing, since nothing else exercises `packer/`.
    """
    from archive.packer.merge_packer import MergeMaximizingPacker
    from translator.merging import PairMergeAnalyzer

    program = QuantumCircuit(6)
    for qubit in range(6):
        program.h(qubit)
    for qubit in range(0, 6, 2):
        program.cx(qubit, qubit + 1)
    for qubit in range(6):
        program.x(qubit)

    analyzer = PairMergeAnalyzer(program)
    stats = analyzer.analyze_pair(0, 1)
    assert stats.hh_merges >= 1, "two qubits both carrying H should show an HH merge"
    assert (stats.cx_merges, stats.cz_merges, stats.xcx_merges) == (0, 0, 0), (
        "transversal counts come from two solo patches, where nothing can fuse"
    )

    packed = MergeMaximizingPacker().pack(program, num_codes=3)
    assert len(packed.permutation) == 6
    assert len(set(packed.permutation)) == 6


# --- savings accounting ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_reported_savings_scale_with_the_number_of_lanes_fused(n_code):
    """
    A whole-patch HH replaces k Hadamards, so the saving must scale with k.

    Fixing the count at two -- as the pairwise version did -- understates a k=10
    merge fivefold, which would make the selector prefer the wrong candidates.
    """
    k = logical_capacity(n_code)
    program = QuantumCircuit(k)
    for lane in range(k):
        program.h(lane)
    permutation = [(0, lane) for lane in range(k)]

    _, savings_1q, savings_2q, _ = preprocess_program_for_merging_with_stats(
        program, permutation, n_code=n_code
    )
    from iceberg.costs import work_table

    table = work_table(n_code)
    expected_1q = k * table["H"][0] - table["HH"][0]
    expected_2q = k * table["H"][1] - table["HH"][1]
    assert (savings_1q, savings_2q) == (expected_1q, expected_2q)


# ----------------------------------------------------------------------
# the transversal macro must pay for itself (na-packing-costs round)
# ----------------------------------------------------------------------


def test_macro_gain_prices_singles_macro_and_routing_swaps():
    """
    The accept/reject gate that used to be a bare boolean: coverage alone no longer
    fires a macro. Weighted table arithmetic, spelled out so a drift is loud.
    """
    from translator.merging.transversal import _macro_gain

    # k=2, aligned: 2 x CX(0,4) = 104 against CXT(0,4) = 52 -> +52.
    assert _macro_gain("cx", 2, 4, transpositions=0) == pytest.approx(52.0)
    # k=2, one transposition: two routing swaps at 2 carriages x 5 units -> -20.
    assert _macro_gain("cx", 2, 4, transpositions=1) == pytest.approx(32.0)
    # Two occupied lanes of a k=10 patch: 2 x CZ_cross(0,4) = 104 against
    # CZT(0,12) = 156 -> the macro LOSES and must not fire.
    assert _macro_gain("cz", 2, 12, transpositions=0) < 0


def test_unprofitable_macro_is_not_taken():
    """A 2-lane transversal on a large code loses to two targeted singles."""
    from translator.merging.transversal import _build_transversal_record, _macro_gain

    assert _macro_gain("cz", 2, 12, 0) < 0 < _macro_gain("cz", 2, 4, 0)
