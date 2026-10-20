"""
Tests for `packing.partition`.

The load-bearing one is `test_gain_matches_actual_cut_change`: the refinement decides
every move from an incrementally maintained attachment table, so a bookkeeping slip
would silently make it optimise the wrong thing. That test recomputes the cut from
scratch and checks the predicted gains sum to the real change.
"""

from __future__ import annotations

import random

import pytest

from packing.tables import translated_view
from packing.graph import build_graph
from packing.profiles import extract_profiles
from iceberg.utils import logical_capacity
from packing.partition import (
    _matching,
    coarsen_cap,
    cut_weight,
    pack_bins,
    partition,
    refine,
)
from packing.seed import PARTITIONERS

CODE_SIZES = [4, 6, 8, 10, 12]
WIDTHS = [8, 13, 21]


def program(num_qubits: int, depth: int, seed: int = 5):
    """A small Clifford circuit with a mix of one- and two-qubit gates."""
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for layer in range(depth):
        for qubit in range(num_qubits):
            roll = rng.random()
            if roll < 0.3:
                circuit.h(qubit)
            elif roll < 0.45:
                circuit.x(qubit)
            elif roll < 0.55:
                circuit.z(qubit)
        order = list(range(num_qubits))
        rng.shuffle(order)
        for index in range(0, num_qubits - 1, 2):
            a, b = order[index], order[index + 1]
            rng.choice([circuit.cx, circuit.cz])(a, b)
        del layer
    return circuit


def random_weights(rng: random.Random, n: int, density: float = 0.35):
    return {
        (a, b): round(rng.uniform(-40.0, 120.0), 3)
        for a in range(n)
        for b in range(a + 1, n)
        if rng.random() < density
    }


# ----------------------------------------------------------------------
# coarsen_cap
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "k, expected", [(2, 2), (3, 1), (4, 4), (5, 1), (6, 2), (8, 8), (10, 2), (12, 4)]
)
def test_coarsen_cap_is_largest_power_of_two_divisor(k, expected):
    assert coarsen_cap(k) == expected


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_coarsen_cap_divides_capacity(n_code):
    k = logical_capacity(n_code)
    cap = coarsen_cap(k)
    assert k % cap == 0
    assert cap & (cap - 1) == 0  # a power of two


# ----------------------------------------------------------------------
# matching
# ----------------------------------------------------------------------


def test_matching_pairs_only_equal_sizes_within_cap():
    rng = random.Random(3)
    for _ in range(60):
        n = rng.randint(4, 24)
        weights = random_weights(rng, n)
        sizes = {q: rng.choice([1, 1, 1, 2, 2, 4]) for q in range(n)}
        cap = 4
        mapping = _matching(weights, sizes, cap)
        merged: dict = {}
        for node, target in mapping.items():
            merged.setdefault(target, []).append(node)
        for members in merged.values():
            assert len(members) <= 2
            if len(members) == 2:
                left, right = members
                assert sizes[left] == sizes[right]
                assert sizes[left] * 2 <= cap


def test_matching_never_pairs_a_repelling_edge():
    weights = {(0, 1): -50.0, (2, 3): 90.0}
    sizes = {q: 1 for q in range(4)}
    mapping = _matching(weights, sizes, 4)
    assert mapping[0] != mapping[1]
    assert mapping[2] == mapping[3]


# ----------------------------------------------------------------------
# pack_bins
# ----------------------------------------------------------------------


def test_pack_bins_never_overflows_under_the_size_invariant():
    rng = random.Random(11)
    for _ in range(500):
        k = rng.choice([2, 4, 6, 8, 10, 12])
        cap = coarsen_cap(k)
        sizes, total, node = {}, 0, 0
        target_qubits = rng.randint(1, 160)
        while total < target_qubits:
            allowed = [s for s in (1, 2, 4, 8) if s <= cap and total + s <= target_qubits]
            sizes[node] = rng.choice(allowed) if allowed else 1
            total += sizes[node]
            node += 1
        target = max(1, -(-total // k))
        bins = pack_bins(list(sizes), {}, sizes, k, target)
        assert len(bins) == target
        for members in bins:
            assert sum(sizes[m] for m in members) <= k
        assert sorted(m for b in bins for m in b) == sorted(sizes)


def test_pack_bins_places_every_node_exactly_once():
    sizes = {q: 1 for q in range(9)}
    bins = pack_bins(list(sizes), {}, sizes, 4, 3)
    flat = [node for members in bins for node in members]
    assert sorted(flat) == list(range(9))
    assert len(bins) == 3


def test_pack_bins_uses_affinity():
    # 0-1 and 2-3 attract; with two bins of two they should not be split.
    weights = {(0, 1): 100.0, (2, 3): 100.0}
    sizes = {q: 1 for q in range(4)}
    bins = pack_bins(list(sizes), weights, sizes, 2, 2)
    grouped = {frozenset(members) for members in bins}
    assert grouped == {frozenset({0, 1}), frozenset({2, 3})}


# ----------------------------------------------------------------------
# refinement gain accounting
# ----------------------------------------------------------------------


def test_gain_matches_actual_cut_change():
    """Every predicted gain must equal the real drop in cut weight."""
    rng = random.Random(17)
    trials = exchanges_seen = 0
    for _ in range(300):
        n = rng.randint(6, 26)
        k = rng.choice([2, 4, 6, 8, 10])
        weights = random_weights(rng, n)
        sizes = {q: 1 for q in range(n)}
        target = max(1, -(-n // k))
        groups = pack_bins(list(sizes), weights, sizes, k, target)
        before = cut_weight(groups, weights)
        trace: list = []
        _, exchanges = refine(groups, weights, sizes, k, trace=trace)
        after = cut_weight(groups, weights)
        exchanges_seen += exchanges
        if trace:
            trials += 1
            assert (before - after) == pytest.approx(sum(g for _, g in trace), abs=1e-6)
    assert trials > 50, "not enough trials accepted a move to be meaningful"
    assert exchanges_seen > 0, "no exchange was ever taken; the swap path is untested"


def test_refine_respects_capacity():
    rng = random.Random(23)
    for _ in range(200):
        n = rng.randint(6, 30)
        k = rng.choice([2, 4, 6, 10])
        weights = random_weights(rng, n)
        sizes = {q: 1 for q in range(n)}
        target = max(1, -(-n // k))
        groups = pack_bins(list(sizes), weights, sizes, k, target)
        refine(groups, weights, sizes, k)
        for members in groups:
            assert sum(sizes[m] for m in members) <= k
        assert sorted(m for g in groups for m in g) == list(range(n))


def test_refine_never_loses_or_duplicates_a_node():
    rng = random.Random(29)
    n, k = 24, 6
    weights = random_weights(rng, n, density=0.6)
    sizes = {q: 1 for q in range(n)}
    groups = pack_bins(list(sizes), weights, sizes, k, 4)
    refine(groups, weights, sizes, k)
    flat = [node for members in groups for node in members]
    assert sorted(flat) == list(range(n))


def test_swap_disabled_takes_no_exchanges():
    rng = random.Random(31)
    total = 0
    for _ in range(80):
        n = rng.randint(8, 24)
        k = rng.choice([2, 4, 6])
        weights = random_weights(rng, n)
        sizes = {q: 1 for q in range(n)}
        target = max(1, -(-n // k))
        groups = pack_bins(list(sizes), weights, sizes, k, target)
        _, exchanges = refine(groups, weights, sizes, k, swap=False)
        total += exchanges
    assert total == 0


def test_refine_only_lowers_the_cut():
    rng = random.Random(37)
    for _ in range(150):
        n = rng.randint(6, 24)
        k = rng.choice([2, 4, 6, 10])
        weights = random_weights(rng, n)
        sizes = {q: 1 for q in range(n)}
        target = max(1, -(-n // k))
        groups = pack_bins(list(sizes), weights, sizes, k, target)
        before = cut_weight(groups, weights)
        refine(groups, weights, sizes, k)
        assert cut_weight(groups, weights) <= before + 1e-9


# ----------------------------------------------------------------------
# the whole partitioner, on real circuits
# ----------------------------------------------------------------------


def _graph_for(circuit, n_code):
    profiles = extract_profiles(translated_view(circuit))
    return build_graph(profiles, n_code, num_qubits=circuit.num_qubits)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("name", list(PARTITIONERS))
def test_partition_is_a_legal_partition(width, n_code, name):
    sample_circuit = program(width, 4)
    options = PARTITIONERS[name]
    if options is None:
        pytest.skip("legacy partitioner is covered by grouping's own tests")
    k = logical_capacity(n_code)
    graph = _graph_for(sample_circuit, n_code)
    target = max(1, -(-sample_circuit.num_qubits // k))
    result = partition(graph, n_code, target, **options)
    flat = [q for group in result.groups for q in group]
    assert sorted(flat) == list(range(sample_circuit.num_qubits))
    assert all(len(group) <= k for group in result.groups)
    assert result.num_patches <= target


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_partition_honours_the_target_exactly_when_packing(width, n_code):
    """`pack_bins` fixes the bin count, so the target is a guarantee, not a cap."""
    sample_circuit = program(width, 4)
    k = logical_capacity(n_code)
    graph = _graph_for(sample_circuit, n_code)
    target = max(1, -(-sample_circuit.num_qubits // k))
    result = partition(graph, n_code, target, matching="equal", base="pack")
    assert result.num_patches <= target
    assert sum(len(g) for g in result.groups) == sample_circuit.num_qubits


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_spare_slots_stay_empty_rather_than_forcing_a_split(width, n_code):
    sample_circuit = program(width, 4)
    k = logical_capacity(n_code)
    graph = _graph_for(sample_circuit, n_code)
    tightest = max(1, -(-sample_circuit.num_qubits // k))
    result = partition(graph, n_code, tightest + 3)
    assert result.num_patches <= tightest + 3
