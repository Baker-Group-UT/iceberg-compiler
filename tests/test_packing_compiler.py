"""
Tests for `packing.compiler` — the entry point that runs both stages.

The two things worth pinning here are the ones a caller cannot check for themselves.
First, that the returned circuit really is the returned placement translated under the
returned commutation setting: the plan is self-describing or it is useless, and the same
trap already has a comment on `PackingPlan.h_commutation`. Second, that the arm
comparison is a comparison -- the winner is the best of the arms on the resource
objective, and the naive arm means "best" has a floor.
"""

from __future__ import annotations

import random

import pytest
from qiskit import QuantumCircuit

from iceberg.utils import logical_capacity
from packing import compile_circuit
from packing.budget import ResourceBudget
from packing.graph import NoiseProfile
from packing.passes import clear_cache
from packing.resource import NoiseKnobs
from packing.resource.noise import packing_profile


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


def program(num_qubits: int, depth: int, seed: int = 5) -> QuantumCircuit:
    """The generator `test_packing_resource` uses, so the two suites see one corpus."""
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


# --- the noise bridge --------------------------------------------------


def test_gate_counting_knobs_reach_the_paper_convention():
    """
    One knobs object drives both stages, so `gate_counting` has to survive the crossing.

    `NoiseKnobs.gate_counting()` zeroes `layer_time`, `move_error` and `movement_speed`;
    `NoiseProfile()` is the paper's convention on the other side. If the conversion did
    not land exactly there, asking for gate counting at the top would quietly hand the
    planner a machine that charges for movement.
    """
    assert packing_profile(NoiseKnobs.gate_counting()) == NoiseProfile()


def test_derived_profile_prices_what_the_knobs_price():
    """Neutral-atom defaults charge for both, and idling scales with the knob."""
    derived = packing_profile(NoiseKnobs())
    assert derived.movement_cost > 0 and derived.idle_cost > 0
    assert derived.distance_scale == 1.0
    assert packing_profile(NoiseKnobs(), idle_scale=0.0).idle_cost == 0.0
    off = packing_profile(NoiseKnobs(), distance=False)
    assert off.movement_cost == 0.0 and off.distance_scale == 0.0


# --- end to end --------------------------------------------------------


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_returns_the_circuit_its_own_placement_describes(planner):
    """
    The returned circuit is `placement` translated under `h_commutation`, exactly -- in the
    configuration the planner compiles for (the standard: layer-sorted programs and the
    transversal in-patch CZ / XCX; v4: neither).

    A `CompiledCircuit` whose fields describe a different circuit than the one it carries
    is worse than no plan at all, because the error estimate is then priced against
    something the caller does not have.
    """
    from packing.compiler import _translate
    from packing.standard import STANDARD

    circuit = program(10, 3)
    built = compile_circuit(circuit, code_sizes=(4, 6), planner=planner)
    standard = planner == "standard"
    again = _translate(
        circuit, built.placement, built.n_code,
        commute=built.h_commutation, merging=True, fault_tolerant=False,
        asap_order=standard and STANDARD.asap_order,
        transversal_intra_two_qubit=standard and STANDARD.transversal_intra_two_qubit,
    )
    assert again is not None
    assert again.circuit.depth() == built.circuit.depth()
    assert again.circuit.size() == built.circuit.size()
    assert built.error.depth == built.circuit.depth()
    assert built.physical_qubits == built.patches * (built.n_code + 2)


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_placement_is_legal_at_the_chosen_code_size(planner):
    circuit = program(10, 3)
    built = compile_circuit(circuit, code_sizes=(4, 6), planner=planner)
    k = logical_capacity(built.n_code)
    assert len(built.placement) == circuit.num_qubits
    sizes: dict = {}
    for patch, _ in built.placement:
        sizes[patch] = sizes.get(patch, 0) + 1
    assert all(count <= k for count in sizes.values())
    assert len(set(built.placement)) == circuit.num_qubits
    assert built.patches == len(sizes)


def test_the_winner_is_the_best_arm_and_naive_is_the_floor():
    """
    `considered` is the whole comparison, ordered, and the plan is its head.

    The naive arm is what makes the floor real in the mode actually translated: neither
    stage's internal naive comparison carries here, since the selector measures its own
    in bare mode with merging on and the packer scores its own on gates and depth.
    """
    circuit = program(12, 3, seed=11)
    built = compile_circuit(circuit, code_sizes=(4, 6), planner="v4")   # the submitted contract
    scores = [score for _, score in built.considered]
    assert scores == sorted(scores)
    assert built.considered[0][0] == built.source
    assert built.error.score(0.0) == pytest.approx(scores[0])
    assert "naive" in dict(built.considered), "the floor was not measured"
    assert scores[0] <= dict(built.considered)["naive"] + 1e-12


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_verify_off_drops_the_naive_arm(planner):
    circuit = program(10, 3)
    built = compile_circuit(circuit, code_sizes=(4,), verify=False, planner=planner)
    assert not any(name.startswith("naive") for name, _ in built.considered)
    assert built.source in ("packer", "selector")


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_both_stages_ran_and_are_reported(planner):
    """
    The point of the entry point is that both stages happened; both are on the result.

    `plan` at the resource stage's code size is the join between them -- if the planner
    were run at some other code size the arm comparison would be pricing two different
    machines against each other.
    """
    circuit = program(10, 3)
    built = compile_circuit(circuit, code_sizes=(4, 6), planner=planner)
    assert built.plan is not None and built.resources is not None
    assert built.plan.n_code == built.resources.n_code == built.n_code
    assert built.resources.frontier, "the resource sweep measured nothing"


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_budget_is_respected_end_to_end(planner):
    """Both stages take the budget, so the returned circuit has to fit it."""
    circuit = program(12, 3)
    built = compile_circuit(
        circuit, budget=ResourceBudget(max_physical_qubits=80), code_sizes=(4, 6, 8),
        planner=planner,
    )
    assert built.physical_qubits <= 80
    assert built.circuit.num_qubits <= 80


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_a_pinned_code_size_is_honoured(planner):
    circuit = program(9, 3)
    built = compile_circuit(circuit, code_sizes=(6,), planner=planner)
    assert built.n_code == 6
    assert built.plan.n_code == 6


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_forced_commutation_reaches_every_arm(planner):
    """
    An explicit setting holds for the planner and for the arms it is compared against.

    The default is one measured decision shared by everything; a caller who overrides it
    is overriding it for the whole comparison, not just for the planner.
    """
    circuit = program(8, 3)
    for commute in (True, False):
        built = compile_circuit(circuit, code_sizes=(4,), h_commutation=commute,
                                planner=planner)
        assert built.h_commutation is commute
        assert built.plan.h_commutation is commute


def test_density_weight_buys_patch_density_through_the_composition():
    """
    The resource decision survives the stage below it.

    This is the case the module docstring calls out: above `density_weight=0` the
    selector spreads and the planner cannot follow, so it is the arm comparison rather
    than `spread` that keeps the looser shape. Either way the caller gets it.
    """
    circuit = program(12, 3, seed=11)
    # The submitted compiler's knob: the standard packs every code size at the planner's
    # own (tightest) patch count and does not read `density_weight`.
    throughput = compile_circuit(circuit, noise=NoiseKnobs(density_weight=0.0),
                                 code_sizes=(4, 6), planner="v4")
    correctness = compile_circuit(circuit, noise=NoiseKnobs(density_weight=200.0),
                                  code_sizes=(4, 6), planner="v4")
    assert correctness.patch_density <= throughput.patch_density
    assert correctness.error.corrupted <= throughput.error.corrupted


def test_fault_tolerant_mode_translates_and_still_has_its_floor():
    """
    FT is the case the extra naive translation is really for.

    Neither search stage has a fault-tolerant path -- both rank in bare mode -- so with
    `fault_tolerant=True` the arm comparison is the only thing measuring what the caller
    actually gets. It matters: on this circuit the naive arm scores far off the winner,
    and picking between them on a bare-mode number would have been guesswork.
    """
    circuit = program(8, 3)
    built = compile_circuit(circuit, code_sizes=(4, 6), fault_tolerant=True, planner="v4")
    assert built.translation is not None and built.translation.fault_tolerant
    assert len(built.placement) == circuit.num_qubits
    assert "naive" in dict(built.considered)
    assert built.error.score(0.0) <= dict(built.considered)["naive"] + 1e-12


@pytest.mark.parametrize("planner", ["standard", "v4"])
def test_summary_names_the_shape_and_the_winner(planner):
    circuit = program(8, 3)
    built = compile_circuit(circuit, code_sizes=(4,), planner=planner)
    text = built.summary()
    assert f"[[{built.n_code},{built.k},2]] x{built.patches}" in text
    assert built.source in text
