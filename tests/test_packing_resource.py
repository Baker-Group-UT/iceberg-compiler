"""Tests for `packing.resource` — code size and patch density selection."""

from __future__ import annotations

import random

import pytest
from qiskit import QuantumCircuit

from packing.budget import ResourceBudget
from iceberg.utils import logical_capacity
from packing.resource import NoiseKnobs, estimate, select_resources
from packing.resource.select import _densities, _forced_groups


def program(num_qubits: int, depth: int, seed: int = 5) -> QuantumCircuit:
    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            roll = rng.random()
            if roll < 0.3:
                circuit.h(qubit)
            elif roll < 0.45:
                circuit.x(qubit)
        order = list(range(num_qubits))
        rng.shuffle(order)
        for index in range(0, num_qubits - 1, 2):
            rng.choice([circuit.cx, circuit.cz])(order[index], order[index + 1])
    return circuit


# --- noise model -------------------------------------------------------


def test_idle_rate_is_derived_not_tuned():
    """
    NA defaults, priced at the simulator's own twirled thermal rate.

    The 1 us layer is the binding single-qubit gate; the rate is
    1/(2*T2) + 1/(4*T1) at T2 = 1.5 s, T1 = 10 s -- both imported from
    `simulator.noise_models`, not restated. The old pin (88 ns / 341 us) was the
    superconducting machine the selector kept ranking for after the simulator moved.
    """
    knobs = NoiseKnobs()
    assert knobs.idle_per_qubit_layer == pytest.approx(1e-6 * (0.5 / 1.5 + 0.25 / 10.0))


def test_gate_counting_profile_switches_idling_off():
    assert NoiseKnobs.gate_counting().idle_per_qubit_layer == 0.0


def test_movement_is_priced_per_grouped_trip():
    """
    One crossing = one round trip of the control side: two carried atoms pay
    `move_error`, and the whole array pays the trip's idling. Both scale the way the
    transport model says -- handling with crossings, idling with crossings x distance
    x physical qubits.
    """
    knobs = NoiseKnobs()
    still = estimate(knobs, one_qubit=0, two_qubit=0, depth=0, patches=4,
                     n_code=4, logical_qubits=8, crossings=0, mean_hops=9.0)
    moving = estimate(knobs, one_qubit=0, two_qubit=0, depth=0, patches=4,
                      n_code=4, logical_qubits=8, crossings=5, mean_hops=9.0)
    assert still.movement == 0.0 and still.idle == 0.0
    assert moving.movement == pytest.approx(5 * 2.0 * knobs.move_error)
    trip = knobs.round_trip_error_seconds(9.0)
    assert moving.idle == pytest.approx(5 * trip * 4 * 6 * knobs.idle_rate)


def test_large_patches_pay_for_their_own_geometry():
    """
    The blockade threshold at resource-selection time: a k=2 patch is free of
    intra-patch carries, a k=10 patch pays them on ~27% of its pairs -- so the same
    gate count costs more two-qubit work at large k, which is the mechanism that
    keeps large patches honest when code sizes are ranked.
    """
    from packing.resource.noise import intra_patch_movement

    assert intra_patch_movement(6) == (0.0, 0.0)                 # k=2: one blockade sphere
    fraction, cells = intra_patch_movement(14)                    # k=10: 4x4 rectangle
    assert 0.2 < fraction < 0.4 and cells > 0
    small = estimate(NoiseKnobs(), one_qubit=0, two_qubit=100, depth=0, patches=1,
                     n_code=4, logical_qubits=2)
    large = estimate(NoiseKnobs(), one_qubit=0, two_qubit=100, depth=0, patches=1,
                     n_code=12, logical_qubits=10)
    assert small.movement == 0.0
    assert large.movement > 0.0


def test_survival_falls_as_errors_rise():
    low = estimate(NoiseKnobs(), one_qubit=10, two_qubit=10, depth=10, patches=2,
                   n_code=4, logical_qubits=4)
    high = estimate(NoiseKnobs(), one_qubit=1000, two_qubit=1000, depth=100, patches=2,
                    n_code=4, logical_qubits=4)
    assert high.total > low.total
    assert high.survival < low.survival


def test_corrupted_falls_with_patch_density():
    """More patches means fewer logical qubits behind each detector pair."""
    tight = estimate(NoiseKnobs(), one_qubit=100, two_qubit=100, depth=50, patches=2,
                     n_code=4, logical_qubits=20)
    spread = estimate(NoiseKnobs(), one_qubit=100, two_qubit=100, depth=50, patches=10,
                      n_code=4, logical_qubits=20)
    assert spread.corrupted < tight.corrupted


def test_density_weight_changes_the_ranking():
    tight = estimate(NoiseKnobs(), one_qubit=100, two_qubit=100, depth=50, patches=2,
                     n_code=4, logical_qubits=20)
    spread = estimate(NoiseKnobs(), one_qubit=110, two_qubit=110, depth=50, patches=10,
                      n_code=4, logical_qubits=20)
    assert tight.score(0.0) < spread.score(0.0)          # throughput prefers tight
    assert spread.score(100.0) < tight.score(100.0)      # correctness prefers spread


# --- forced spreading --------------------------------------------------


@pytest.mark.parametrize("patches", [3, 5, 7, 9])
def test_forced_groups_hits_the_requested_patch_count(patches):
    weights = {(a, b): 10.0 for a in range(9) for b in range(a + 1, 9)}
    groups = _forced_groups(9, weights, 4, patches)
    assert len(groups) == patches
    assert sorted(q for g in groups for q in g) == list(range(9))
    assert all(len(g) <= 4 for g in groups)


def test_forced_groups_respects_capacity():
    weights = {(a, b): 5.0 for a in range(12) for b in range(a + 1, 12)}
    groups = _forced_groups(12, weights, 3, 4)
    assert all(len(g) <= 3 for g in groups)
    assert sorted(q for g in groups for q in g) == list(range(12))


def test_densities_span_tightest_to_loosest():
    picks = _densities(20, 4, None)
    assert picks[0] == 5 and picks[-1] == 20
    assert picks == sorted(set(picks))


def test_densities_respect_a_ceiling():
    picks = _densities(20, 4, 8)
    assert max(picks) <= 8 and min(picks) == 5


# --- end to end --------------------------------------------------------


def test_returns_a_legal_placement():
    circuit = program(10, 3)
    plan = select_resources(circuit, code_sizes=(4, 6))
    k = logical_capacity(plan.n_code)
    assert len(plan.placement) == circuit.num_qubits
    sizes: dict = {}
    for patch, _ in plan.placement:
        sizes[patch] = sizes.get(patch, 0) + 1
    assert all(count <= k for count in sizes.values())
    assert len({(p, s) for p, s in plan.placement}) == circuit.num_qubits
    assert plan.physical_qubits == plan.patches * (plan.n_code + 2)


def test_budget_is_respected():
    circuit = program(12, 3)
    plan = select_resources(
        circuit, budget=ResourceBudget(max_physical_qubits=80), code_sizes=(4, 6, 8)
    )
    assert plan.physical_qubits <= 80


def test_density_weight_buys_lower_patch_density():
    circuit = program(12, 3, seed=11)
    throughput = select_resources(circuit, noise=NoiseKnobs(density_weight=0.0),
                                  code_sizes=(4, 6))
    correctness = select_resources(circuit, noise=NoiseKnobs(density_weight=200.0),
                                   code_sizes=(4, 6))
    assert correctness.patch_density <= throughput.patch_density
    assert correctness.error.corrupted <= throughput.error.corrupted


def test_frontier_is_ordered_and_measured():
    """
    The frontier is ranked, and the returned plan is no worse than the best of it.

    This used to assert the plan *equals* the top-scoring measured candidate, which was
    only true while `select_resources` returned a candidate untouched. It now runs
    `_refine_and_verify` on the winner -- a descent arm and a naive arm, both exactly
    translated -- so the plan can and should beat everything on the frontier. Equality
    would now be a sign the refinement had stopped working.
    """
    circuit = program(9, 3)
    plan = select_resources(circuit, code_sizes=(4, 6))
    measured = [c for c in plan.frontier if c.measured]
    assert measured, "nothing was measured"
    scores = [c.error.score(0.0) for c in measured]
    assert scores == sorted(scores)
    assert plan.error.total <= measured[0].error.total + 1e-9, (
        f"plan ({plan.error.total:.6f}) is worse than the best measured candidate "
        f"({measured[0].error.total:.6f}); refinement should never lose to its own input"
    )


def test_explicit_densities_are_honoured():
    circuit = program(12, 3)
    plan = select_resources(circuit, code_sizes=(4,), densities=(6, 12))
    assert plan.patches in (6, 12)
