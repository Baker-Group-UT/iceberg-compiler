"""
Tests for the `grouping` module.

These check the properties the experiment relies on — that profiles describe the
circuit faithfully, that placements are valid, that the capacity constraint holds at
every k, and that refinement never makes things worse. They deliberately do *not*
assert that any particular linkage beats another: that is an empirical question the
measurements in `grouping.evaluate` answer, and pinning it in a test would freeze a
result that should be free to change.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive.grouping.agglomerative import LINKAGES, cluster_qubits, permutation_from_groups
from archive.grouping.cost_model import GroupCostModel
from archive.grouping.profiles import extract_profiles
from archive.grouping.refine import model_objective, refine_placement
from archive.grouping.similarity import frame_similarity, similarity_matrix
from iceberg.utils import logical_capacity
from tests.random_programs import CLIFFORD_1Q

CODE_SIZES = [4, 6, 8]


def program(num_qubits: int = 6, depth: int = 30, seed: int = 7) -> QuantumCircuit:
    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        if rng.random() < 0.45:
            left, right = rng.sample(range(num_qubits), 2)
            getattr(circuit, rng.choice(["cx", "cz"]))(left, right)
        else:
            getattr(circuit, rng.choice(CLIFFORD_1Q))(rng.randrange(num_qubits))
    return circuit


# --- profiles ---


def test_profiles_account_for_every_gate():
    circuit = program()
    profiles = extract_profiles(circuit)
    seen = sum(len(profile.events) for profile in profiles.values())
    expected = sum(
        len(instruction.qubits)
        for instruction in circuit.data
        if instruction.operation.name not in ("barrier", "measure")
    )
    assert seen == expected


def test_cast_is_symmetric():
    profiles = extract_profiles(program())
    for index, profile in profiles.items():
        for partner in profile.cast():
            assert index in profiles[partner].cast()


def test_frame_records_the_layer_of_each_gate():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(0)
    profiles = extract_profiles(circuit)
    frame = profiles[0].frame()
    assert frame[0] == ("h",)
    assert "cx" in frame[1]
    assert profiles[0].count("h") == 2
    assert profiles[1].cast() == frozenset({0})


# --- cost model ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_group_cost_is_positive_and_grows_with_the_group(n_code):
    circuit = program(num_qubits=logical_capacity(n_code) + 2)
    profiles = extract_profiles(circuit)
    model = GroupCostModel(profiles=profiles, n_code=n_code)
    singles = [model.cost([index]) for index in profiles]
    assert all(cost >= 0 for cost in singles)
    pair = model.cost([0, 1])
    assert pair <= singles[0] + singles[1] + 1e-9, (
        "merging two qubits should never be modelled as more expensive than "
        "keeping them apart, since internal gates are charged once rather than twice"
    )


def test_net_gain_rejects_overlapping_groups():
    profiles = extract_profiles(program())
    model = GroupCostModel(profiles=profiles, n_code=4)
    with pytest.raises(ValueError):
        model.net_gain([0, 1], [1, 2])


# --- similarity ---


def test_similarity_matrix_is_complete_and_symmetric_by_construction():
    profiles = extract_profiles(program(num_qubits=5))
    matrix = similarity_matrix(profiles)
    assert len(matrix) == 5 * 4 // 2
    for (left, right), value in matrix.items():
        assert left < right
        assert value == pytest.approx(
            frame_similarity(profiles[left], profiles[right])
        )


def test_identical_qubits_are_maximally_similar():
    """Two qubits with the same timeline should score above a mismatched pair."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.h(1)  # same frame as qubit 0
    circuit.x(2)  # different
    profiles = extract_profiles(circuit)
    assert frame_similarity(profiles[0], profiles[1]) > frame_similarity(
        profiles[0], profiles[2]
    )


# --- clustering ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("linkage", LINKAGES)
def test_clustering_respects_capacity_and_places_every_qubit(n_code, linkage):
    num_qubits = logical_capacity(n_code) + 3
    circuit = program(num_qubits=num_qubits)
    result = cluster_qubits(circuit, n_code=n_code, linkage=linkage)

    placed = sorted(qubit for group in result.groups for qubit in group)
    assert placed == list(range(num_qubits)), "every qubit must land in exactly one patch"
    assert all(len(group) <= result.k for group in result.groups)

    permutation = result.permutation()
    assert len(permutation) == num_qubits
    assert len(set(permutation)) == num_qubits, "no two qubits may share a slot"
    for patch, slot in permutation:
        assert 0 <= slot < result.k


def test_clustering_rejects_an_unknown_linkage():
    with pytest.raises(ValueError, match="unknown linkage"):
        cluster_qubits(program(), linkage="nonsense")


def test_max_patches_forces_merges_past_the_score_threshold():
    circuit = program(num_qubits=6)
    loose = cluster_qubits(circuit, n_code=4, linkage="interaction")
    tight = cluster_qubits(circuit, n_code=4, linkage="interaction", max_patches=3)
    assert tight.num_patches <= 3
    assert tight.num_patches <= loose.num_patches


def test_permutation_from_groups_rejects_an_oversized_group():
    with pytest.raises(ValueError, match="exceeds capacity"):
        permutation_from_groups([[0, 1, 2]], 2)


def test_clustering_recovers_planted_blocks():
    """
    With a clean block structure, the interaction linkage should find it.

    Two triples that interact only among themselves, at k=4 so both fit.
    """
    circuit = QuantumCircuit(6)
    for _ in range(4):
        for left, right in ((0, 2), (2, 4), (0, 4)):
            circuit.cx(left, right)
        for left, right in ((1, 3), (3, 5), (1, 5)):
            circuit.cx(left, right)
    result = cluster_qubits(circuit, n_code=6, linkage="interaction")
    groups = {frozenset(group) for group in result.groups}
    assert frozenset({0, 2, 4}) in groups
    assert frozenset({1, 3, 5}) in groups


# --- refinement ---


@pytest.mark.parametrize("n_code", [4, 6])
def test_refinement_never_worsens_the_objective(n_code):
    circuit = program(num_qubits=logical_capacity(n_code) + 2)
    profiles = extract_profiles(circuit)
    objective = model_objective(profiles, n_code)
    seed = cluster_qubits(circuit, n_code=n_code, linkage="interaction").permutation()

    before = objective(seed)
    result = refine_placement(seed, objective, n_code=n_code, max_evaluations=400)
    assert result.cost <= before + 1e-9
    assert len(set(result.placement)) == len(result.placement), "slots stay distinct"
    assert len(result.placement) == circuit.num_qubits


def test_refinement_reaches_a_local_optimum():
    """At the end, no single move improves the objective."""
    from archive.grouping.refine import _candidates

    circuit = program(num_qubits=5)
    profiles = extract_profiles(circuit)
    objective = model_objective(profiles, 4)
    seed = [(index // 2, index % 2) for index in range(5)]
    result = refine_placement(seed, objective, n_code=4)

    for _, neighbour in _candidates(result.placement, 2):
        assert objective(neighbour) >= result.cost - 1e-9


def test_reorder_moves_stay_bounded_at_large_k():
    """
    The neighbourhood must not be dominated by one patch's permutations.

    A patch with k occupants admits k! - 1 reorderings, so at k=10 an unbounded
    `reorder` would emit millions of candidates for patch 0 alone and any budgeted
    search would never reach patch 1. Slot transpositions must still be reachable,
    since those are what carry the transversal-macro alignment.
    """
    from archive.grouping.refine import MAX_REORDER_SLOTS, _candidates

    k = 10
    placement = [(index // k, index % k) for index in range(2 * k)]
    kinds = [kind for kind, _ in _candidates(placement, k)]
    assert kinds.count("reorder") == 0, (
        f"patches of {k} exceed MAX_REORDER_SLOTS={MAX_REORDER_SLOTS}, so no full "
        "permutations should be enumerated"
    )

    # A transposition within patch 0 is still a neighbour, reached via `swap`.
    swapped = list(placement)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert any(
        neighbour == swapped for kind, neighbour in _candidates(placement, k)
        if kind == "swap"
    )

    # Small patches keep their reorderings.
    small = [(index // 3, index % 3) for index in range(6)]
    assert any(kind == "reorder" for kind, _ in _candidates(small, 3))
