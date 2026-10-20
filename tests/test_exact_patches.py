"""
A fixed patch count for the standard planner (`patches=`), and the exact modes under it.

Two of these are load-bearing. `test_plan_holds_the_count` runs every stage together at every
count a program admits and checks the safety net is aimed at the same count.
`test_exact_is_the_default_plan_at_the_tightest_count` pins that the new mode cannot move a plan
the default makes -- at the tightest count capacity already forbids every count-changing move --
which is what keeps every recorded w10T plan valid.

The rest pin each stage's own invariant, so a regression names the stage that lost the count:
the partitioner (`exact`, with `pack_bins(fill=)` and `refine(keep_occupied=)` under it), the
descent (`fixed_count`), and the naive packing at a count (`program_order`), which is also the
density sweep's trivial packing at k = 2.
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

from iceberg.utils import logical_capacity
from packing.budget import ResourceBudget
from packing.cost import PackingCost
from packing.graph import InteractionGraph, build_graph
from packing.partition import coarsen_cap, cut_weight, pack_bins, partition, refine
from packing.plan import naive_packing
from packing.profiles import extract_profiles
from packing.search import descend, moves_for, naive_seed, program_order, shuffled_seed
from packing.seed import PARTITIONERS
from packing.standard import plan_standard
from packing.tables import translated_view


def program(num_qubits: int, depth: int, seed: int = 5) -> QuantumCircuit:
    """A random Clifford program without SWAPs, which are rejected cross-patch."""
    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            roll = rng.random()
            if roll < 0.35:
                circuit.h(qubit)
            elif roll < 0.5:
                circuit.x(qubit)
            elif roll < 0.6:
                circuit.z(qubit)
        for _ in range(max(1, num_qubits // 3)):
            a, b = rng.sample(range(num_qubits), 2)
            rng.choice([circuit.cx, circuit.cz])(a, b)
    return circuit


def measured(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    circuit.measure_all()
    return circuit


def random_weights(rng: random.Random, n: int, density: float = 0.35):
    """A signed graph: repelling edges are what leave bins empty in the default mode."""
    return {
        (a, b): round(rng.uniform(-40.0, 120.0), 3)
        for a in range(n)
        for b in range(a + 1, n)
        if rng.random() < density
    }


def tightest(num_qubits: int, k: int) -> int:
    return max(1, -(-num_qubits // k))


def graph_for(circuit: QuantumCircuit, n_code: int):
    profiles = extract_profiles(translated_view(circuit))
    return build_graph(profiles, n_code, num_qubits=circuit.num_qubits)


# ----------------------------------------------------------------------
# the naive packing at a count
# ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4, 10])
def test_program_order_reaches_every_admissible_count(k):
    for num_qubits in range(1, 23):
        for patches in range(tightest(num_qubits, k), num_qubits + 1):
            groups = program_order(num_qubits, k, patches)
            assert len(groups) == patches
            assert all(1 <= len(group) <= k for group in groups)
            assert [q for group in groups for q in group] == list(range(num_qubits))
            sizes = [len(group) for group in groups]
            assert sizes == sorted(sizes, reverse=True), "full patches first"


@pytest.mark.parametrize("k", [2, 4, 6, 10])
def test_naive_packing_at_the_tightest_count_is_the_plain_fill(k):
    for num_qubits in range(1, 30):
        assert naive_packing(num_qubits, k, tightest(num_qubits, k)) == naive_packing(num_qubits, k)


def test_naive_packing_at_k2_is_the_density_sweeps_trivial_packing():
    """The fallback the planner guards against is the baseline the sweep measures against."""
    from experiments.figures.density_ablation_sweep.density_ablation_sweep import (
        tight_trivial_packing,
    )

    for num_qubits in range(2, 30):
        for patches in range(tightest(num_qubits, 2), num_qubits + 1):
            assert naive_packing(num_qubits, 2, patches) == tight_trivial_packing(
                num_qubits, patches
            )


def test_program_order_refuses_unreachable_counts():
    with pytest.raises(ValueError):
        program_order(10, 2, 4)
    with pytest.raises(ValueError):
        program_order(10, 2, 11)


def test_naive_seed_at_a_count_has_that_many_live_patches():
    cost = PackingCost.for_program(program(11, 3), 4)
    for patches in range(6, 12):
        packing = naive_seed(cost, patches + 2, patches=patches)
        assert packing.live == patches
        assert len(packing.groups) == patches + 2


# ----------------------------------------------------------------------
# the partitioner
# ----------------------------------------------------------------------


def test_pack_bins_fill_occupies_every_bin_and_never_overflows():
    rng = random.Random(41)
    for _ in range(400):
        k = rng.choice([2, 4, 6, 8, 10, 12])
        cap = coarsen_cap(k)
        sizes, total, node = {}, 0, 0
        goal = rng.randint(1, 120)
        while total < goal:
            allowed = [s for s in (1, 2, 4, 8) if s <= cap and total + s <= goal]
            sizes[node] = rng.choice(allowed) if allowed else 1
            total += sizes[node]
            node += 1
        target = rng.randint(tightest(total, k), len(sizes))
        bins = pack_bins(list(sizes), random_weights(rng, len(sizes)), sizes, k, target,
                         fill=True)
        assert len(bins) == target
        assert all(bins), "a bin was left empty"
        assert all(sum(sizes[m] for m in members) <= k for members in bins)
        assert sorted(m for members in bins for m in members) == sorted(sizes)


def test_refine_keep_occupied_holds_the_count_and_its_gains_stay_exact():
    rng = random.Random(43)
    for _ in range(200):
        n = rng.randint(6, 26)
        k = rng.choice([2, 4, 6])
        weights = random_weights(rng, n)
        sizes = {q: 1 for q in range(n)}
        target = rng.randint(tightest(n, k), n)
        groups = pack_bins(list(sizes), weights, sizes, k, target, fill=True)
        before = cut_weight(groups, weights)
        trace: list = []
        refine(groups, weights, sizes, k, trace=trace, keep_occupied=True)
        assert sum(1 for group in groups if group) == target
        assert all(len(group) <= k for group in groups)
        assert before - cut_weight(groups, weights) == pytest.approx(
            sum(gain for _, gain in trace), abs=1e-6
        )


@pytest.mark.parametrize("width", [8, 13, 30])
@pytest.mark.parametrize("n_code", [4, 6, 8])
@pytest.mark.parametrize("name", list(PARTITIONERS))
def test_exact_partition_returns_exactly_target_groups(width, n_code, name):
    """Width 30 is past `coarsen_to`, so the per-level contraction limit is exercised."""
    circuit = program(width, 4)
    k = logical_capacity(n_code)
    graph = graph_for(circuit, n_code)
    for target in range(tightest(width, k), width + 1):
        result = partition(graph, n_code, target, exact=True, **PARTITIONERS[name])
        assert result.num_patches == target
        assert all(1 <= len(group) <= k for group in result.groups)
        assert sorted(q for group in result.groups for q in group) == list(range(width))


def test_exact_partition_holds_on_signed_graphs():
    rng = random.Random(47)
    for _ in range(60):
        n = rng.randint(4, 40)
        n_code = rng.choice([4, 6, 8, 10])
        k = logical_capacity(n_code)
        graph = InteractionGraph(random_weights(rng, n), n)
        target = rng.randint(tightest(n, k), n)
        for name, options in PARTITIONERS.items():
            result = partition(graph, n_code, target, exact=True, **options)
            assert result.num_patches == target, name
            assert all(1 <= len(group) <= k for group in result.groups), name


@pytest.mark.parametrize("width", [7, 13, 30])
@pytest.mark.parametrize("n_code", [4, 6, 10])
@pytest.mark.parametrize("name", list(PARTITIONERS))
def test_exact_is_the_default_partition_at_the_tightest_count(width, n_code, name):
    circuit = program(width, 4)
    graph = graph_for(circuit, n_code)
    target = tightest(width, logical_capacity(n_code))
    exact = partition(graph, n_code, target, exact=True, **PARTITIONERS[name])
    default = partition(graph, n_code, target, **PARTITIONERS[name])
    assert exact.groups == default.groups


def test_exact_partition_refuses_unreachable_targets():
    graph = graph_for(program(10, 3), 4)
    with pytest.raises(ValueError):
        partition(graph, 4, 4, exact=True)
    with pytest.raises(ValueError):
        partition(graph, 4, 11, exact=True)


# ----------------------------------------------------------------------
# the descent
# ----------------------------------------------------------------------


def _live_after(packing, move) -> int:
    groups = list(packing.groups)
    for index, group in move.as_dict().items():
        groups[index] = group
    return sum(1 for group in groups if group)


@pytest.mark.parametrize("n_code", [4, 6])
def test_fixed_count_offers_exactly_the_count_preserving_moves(n_code):
    cost = PackingCost.for_program(program(11, 4), n_code)
    for patches in range(tightest(11, cost.k), 12):
        packing = naive_seed(cost, patches + 2, patches=patches)  # spare slots on offer too
        for qubit in sorted(packing.patch_of):
            every = set(moves_for(packing, qubit))
            fixed = set(moves_for(packing, qubit, fixed_count=True))
            assert fixed <= every
            assert all(_live_after(packing, move) == packing.live for move in fixed)
            assert all(_live_after(packing, move) != packing.live for move in every - fixed)


@pytest.mark.parametrize("n_code", [4, 6])
def test_fixed_count_descent_keeps_the_count_and_only_improves(n_code):
    cost = PackingCost.for_program(program(13, 5, seed=21), n_code)
    rng = random.Random(5)
    for patches in range(tightest(13, cost.k), 14):
        for packing in (naive_seed(cost, patches, patches=patches),
                        shuffled_seed(cost, patches, rng)):
            assert packing.live == patches
            before = packing.value()
            descend(packing, fixed_count=True)
            assert packing.live == patches
            assert packing.value() <= before + 1e-9


def test_the_default_descent_does_not_hold_the_count():
    """Why `fixed_count` exists: the objective charges per live patch, so spread seeds collapse."""
    cost = PackingCost.for_program(program(13, 5, seed=21), 4)
    collapsed = []
    for patches in range(tightest(13, 2) + 1, 14):
        packing = naive_seed(cost, patches, patches=patches)
        descend(packing)
        collapsed.append(packing.live < patches)
    assert any(collapsed)


# ----------------------------------------------------------------------
# the planner
# ----------------------------------------------------------------------


def _clifford(num_qubits: int, seed: int) -> QuantumCircuit:
    from tests.random_programs import random_clifford_program

    circuit = random_clifford_program(num_qubits, 6 * num_qubits, rng=random.Random(seed),
                                      include_xcx=True)
    circuit.measure_all()
    return circuit


@pytest.mark.parametrize("n_code, width", [(4, 9), (6, 10)])
def test_plan_holds_the_count(n_code, width):
    circuit = _clifford(width, seed=3)
    k = logical_capacity(n_code)
    for patches in range(tightest(width, k), width + 1):
        plan = plan_standard(circuit, n_code, patches=patches)
        assert plan.patches == patches
        assert len({patch for patch, _ in plan.placement}) == patches
        assert len(set(plan.placement)) == width
        assert plan.physical_qubits == patches * (n_code + 2)
        assert plan.verified
        scores = dict(plan.considered)
        if plan.fell_back:
            assert plan.placement == naive_packing(width, k, patches)
        else:
            assert scores["packed"] <= scores["naive"] + 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_exact_is_the_default_plan_at_the_tightest_count(seed):
    circuit = _clifford(10, seed=seed)
    for n_code in (4, 6):
        default = plan_standard(circuit, n_code)
        exact = plan_standard(circuit, n_code, patches=tightest(10, logical_capacity(n_code)))
        assert exact.placement == default.placement
        assert exact.h_commutation == default.h_commutation
        assert exact.fell_back == default.fell_back
        assert exact.cost == pytest.approx(default.cost)


def test_a_fixed_commutation_decision_is_honoured_at_a_count():
    circuit = _clifford(9, seed=4)
    for commute in (True, False):
        plan = plan_standard(circuit, 4, patches=7, h_commutation=commute)
        assert plan.h_commutation is commute
        assert plan.patches == 7


def test_unreachable_or_conflicting_counts_are_refused():
    circuit = _clifford(8, seed=1)
    with pytest.raises(ValueError):
        plan_standard(circuit, 4, patches=3)            # below ceil(8 / 2)
    with pytest.raises(ValueError):
        plan_standard(circuit, 4, patches=9)            # more patches than qubits
    with pytest.raises(ValueError):
        plan_standard(circuit, 4, patches=5, spread=2)  # two answers to one question
    with pytest.raises(ValueError):
        plan_standard(circuit, 4, patches=6, budget=ResourceBudget(max_patches=5))
