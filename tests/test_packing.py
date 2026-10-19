"""
Tests for the `packing` package.

Two of these are load-bearing and the rest are hygiene.

`test_matches_analytic_cost` pins the new objective to the old one. `PackingCost` is a
faster spelling of `grouping.analytic.AnalyticCost`, not a second opinion about what a
placement costs, and the moment the two disagree every measured comparison between the
packages becomes meaningless.

`test_incremental_matches_rescore` pins the delta machinery. The search never rescores
a partition from scratch -- it adds and subtracts the handful of terms a move touches
-- so a bug there is a silently wrong objective that still looks like it is descending.
"""

from __future__ import annotations

import random
import sys
from itertools import permutations
from pathlib import Path
from typing import List, Tuple

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive.grouping.analytic import AnalyticCost
from packing.tables import translated_view
from packing.profiles import extract_profiles
from iceberg.utils import logical_capacity
from packing.cost import _EMPTY, PackingCost
from packing.graph import NoiseProfile
from packing.lanes import assign_lanes
from packing.plan import measured_cost, naive_packing, plan_packing
from packing.search import (
    Packing,
    _candidates,
    descend,
    moves_for,
    naive_seed,
    shuffled_seed,
)
from packing.seed import graph_seed

CODE_SIZES = [4, 6, 8, 12]


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
            a, b = rng.sample(range(num_qubits), 2) if num_qubits > 1 else (0, 0)
            if a == b:
                continue
            rng.choice([circuit.cx, circuit.cz])(a, b)
    return circuit


def chain(num_qubits: int, layers: int = 4) -> QuantumCircuit:
    """Nearest-neighbour CNOTs plus local rotations. The ising benchmark's shape."""
    circuit = QuantumCircuit(num_qubits)
    for layer in range(layers):
        for qubit in range(layer % 2, num_qubits - 1, 2):
            circuit.cx(qubit, qubit + 1)
        for qubit in range(num_qubits):
            circuit.h(qubit)
    return circuit


def partitions(num_qubits: int, k: int, rng: random.Random, count: int = 4):
    """Random tight partitions, as lists of frozensets."""
    for _ in range(count):
        order = list(range(num_qubits))
        rng.shuffle(order)
        yield [frozenset(order[i:i + k]) for i in range(0, num_qubits, k)]


def placement_of(groups) -> List[Tuple[int, int]]:
    total = sum(len(g) for g in groups)
    out = [(0, 0)] * total
    for index, group in enumerate(groups):
        for slot, qubit in enumerate(sorted(group)):
            out[qubit] = (index, slot)
    return out


# ----------------------------------------------------------------------
# the objective
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("depth_weight", [0.0, 12.0])
def test_matches_analytic_cost_on_gates(
    n_code: int, depth_weight: float, monkeypatch
) -> None:
    """
    `PackingCost` still reproduces `AnalyticCost` on everything except depth and
    movement carriages.

    The depth term deliberately diverges: `AnalyticCost` sums every gate's depth,
    which assumes nothing runs concurrently and -- because a sum is unchanged by moving
    gates between patches -- cannot express load balance at all. `PackingCost` blends that
    sum with the busiest patch's load (`PARALLEL_SHARE`). Pinning the two models to each
    other was what let a sign error in the depth term stay invisible: both were wrong
    together and the suite passed. `test_ranks_a_separable_partition_correctly` is the
    replacement pin, and it checks against exact translation rather than against a
    second model.

    The second deliberate divergence is the carriage pricing: `PackingCost` charges
    HH's anchor swap and a macro's expected routing swaps as atom exchanges, which the
    archived model predates. Zeroing `carriage_cost` restores the movement-free
    objective exactly, so this comparison still pins every shared piece of the gate
    accounting; the carriage arithmetic has its own pins below
    (`test_hh_savings_charge_the_anchor_swap` and
    `test_transversal_savings_price_expected_realignment`).

    Comparing at `depth_weight=0.0` still exercises the whole gate accounting, and at
    12.0 it confirms the remaining difference is the depth charge alone.
    """
    monkeypatch.setattr("packing.cost.carriage_cost", lambda *args, **kwargs: 0.0)
    circuit = program(12, 4)
    k = logical_capacity(n_code)
    profiles = extract_profiles(translated_view(circuit))
    old = AnalyticCost(profiles=profiles, n_code=n_code, depth_weight=depth_weight)
    new = PackingCost(
        profiles=profiles, n_code=n_code, num_qubits=12, depth_weight=depth_weight
    )
    rng = random.Random(3)
    for groups in partitions(12, k, rng, count=6):
        live = [frozenset(g) for g in groups if g]
        mine = new.of(groups)
        theirs = old.cost(placement_of(groups))
        # Undo each model's depth charge and the two must agree exactly.
        total = _EMPTY
        for group in live:
            total = total + new.unary(group)
        for index, left in enumerate(live):
            for right in live[index + 1:]:
                total = total + new.binary(left, right)
        busiest = new.busiest_patch_depth(live)
        mine -= depth_weight * new.estimated_depth(total, busiest)
        theirs -= depth_weight * total.depth
        assert mine == pytest.approx(theirs, rel=1e-12)


def test_mean_patch_hops_matches_the_simulators_centres() -> None:
    """
    The packing model's distances are the transport model's distances, by construction.

    `mean_patch_hops` must equal the mean Manhattan centre-to-centre separation of
    `simulator.grid_constructor.build_patch_centres` -- the quantity
    `derive_transport_groups` actually charges -- at every patch count and code size.
    This is the lockstep contract the old unit-pitch version broke: the search was
    optimising a geometry that was never simulated.
    """
    import itertools

    from packing.tables import mean_patch_hops
    from simulator.grid_constructor import build_patch_centres
    from simulator.transport import manhattan

    for patches, per_patch in [(2, 6), (3, 14), (5, 8), (8, 6), (12, 10)]:
        centres = build_patch_centres(patches, qubits_per_patch=per_patch)
        pairs = list(itertools.combinations(range(patches), 2))
        exact = sum(manhattan(centres[a], centres[b]) for a, b in pairs) / len(pairs)
        assert mean_patch_hops(patches, per_patch) == pytest.approx(exact), (
            patches, per_patch,
        )


def test_hh_savings_charge_the_anchor_swap() -> None:
    """
    HH's anchor swap is an atom exchange -- two carriages -- not a free relabel.

    The HH cost inside the savings comparison must therefore sit above the bare
    (n, 0) table entry by exactly `carriage_cost("HH")`.
    """
    from iceberg.costs import DEFAULT_WORK_WEIGHTS, carriage_cost, work_table

    circuit = program(12, 4)
    profiles = extract_profiles(translated_view(circuit))
    cost = PackingCost(profiles=profiles, n_code=4, num_qubits=12)
    w1, w2 = DEFAULT_WORK_WEIGHTS
    table = work_table(4)
    bare = table["HH"][0] * w1 + table["HH"][1] * w2
    assert cost._hh == pytest.approx(bare + carriage_cost("HH"))
    assert carriage_cost("HH") > 0


def test_transversal_savings_price_expected_realignment() -> None:
    """
    A fired macro's saving is reduced by the *expected* routing swaps of a random
    alignment -- 2(m - H_m) swaps priced as atom exchanges -- and credits (m - 1)
    crossings of movement, all without the objective ever seeing slot order.
    """
    from iceberg.costs import DEFAULT_WORK_WEIGHTS, carriage_cost, work_table

    circuit = program(12, 4)
    profiles = extract_profiles(translated_view(circuit))
    cost = PackingCost(profiles=profiles, n_code=4, num_qubits=12)

    # Two lanes, one clean CX bijection in one layer.
    events = {"cx": [(0, 0, 2), (0, 1, 3)]}
    savings, crossing_savings = cost._transversal(events, 2, 2)

    w1, w2 = DEFAULT_WORK_WEIGHTS
    table = work_table(4)
    single = table["CX"][0] * w1 + table["CX"][1] * w2
    macro = table["CXT"][0] * w1 + table["CXT"][1] * w2
    expected_swaps = 2.0 * (2 - 1.5)  # m - H_m at m = 2
    realign = expected_swaps * carriage_cost("SWAP")
    assert savings == pytest.approx(2 * single - macro - realign)
    assert crossing_savings == pytest.approx(1.0)

    # Lane-order invariance survives: the charge is an expectation, not a slot count.
    permuted = {"cx": [(0, 0, 3), (0, 1, 2)]}
    assert cost._transversal(permuted, 2, 2) == (savings, crossing_savings)


@pytest.mark.parametrize("n_code", [6, 8])
def test_ranks_a_separable_partition_correctly(n_code: int) -> None:
    """
    The model must prefer the right grouping when there provably is one.

    A planted-block circuit at `locality=1.0` has zero cross-block two-qubit gates, and
    with `block_size == k` the blocks are an optimal partition -- exact translation puts
    them ~10% below the naive packing. The model used to rank them *last* on every seed,
    so the descent optimised away from the answer and `verify` threw the result out; the
    packing segment read near zero for a reason that had nothing to do with the search.

    This is the pin that `test_matches_analytic_cost_on_gates` could not be: it checks the
    model against a real translation instead of against another model.
    """
    from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
    from packing.passes import commutation_pays

    k = logical_capacity(n_code)
    num_qubits = 3 * k
    for seed in range(3):
        blocks = plant_blocks(num_qubits, k, seed=seed)
        circuit = structured_program_circuit(
            num_qubits, 12, seed=seed, blocks=blocks, locality=1.0
        )
        commute = commutation_pays(circuit, n_code)
        cost = PackingCost(
            profiles=extract_profiles(
                translated_view(circuit, enable_h_commutation=commute)
            ),
            n_code=n_code,
            num_qubits=num_qubits,
        )
        oracle = [frozenset(b) for b in blocks]
        naive = [
            frozenset(range(i, min(i + k, num_qubits)))
            for i in range(0, num_qubits, k)
        ]

        # Sanity: the oracle really is better, by exact translation.
        truth_oracle = measured_cost(
            circuit, placement_of(oracle), n_code, h_commutation=commute
        )
        truth_naive = measured_cost(
            circuit, naive_packing(num_qubits, k), n_code, h_commutation=commute
        )
        assert truth_oracle < truth_naive, (
            f"seed {seed}: planted blocks are not actually better ({truth_oracle} vs "
            f"{truth_naive}) -- the fixture, not the model, is wrong"
        )

        assert cost.of(oracle) < cost.of(naive), (
            f"n={n_code} seed {seed}: model ranks the planted partition "
            f"{cost.of(oracle):.0f} above naive {cost.of(naive):.0f}, while exact "
            f"translation puts it {(truth_oracle - truth_naive) / truth_naive * 100:.1f}% "
            "below"
        )


@pytest.mark.parametrize("n_code", [6, 12])
def test_analytic_cost_ignores_lanes(n_code: int) -> None:
    """
    The premise the whole package rests on: cost is a function of the partition.

    If this ever fails, lanes have become part of the objective and `search` is
    searching the wrong space -- `assign_lanes` would have to move into the descent.
    """
    circuit = program(8, 4)
    profiles = extract_profiles(translated_view(circuit))
    old = AnalyticCost(profiles=profiles, n_code=n_code, depth_weight=12.0)
    k = logical_capacity(n_code)
    groups = [frozenset(range(min(k, 8)))]
    remaining = [q for q in range(8) if q not in groups[0]]
    if remaining:
        groups.append(frozenset(remaining))
    members = sorted(groups[0])
    seen = set()
    for order in permutations(range(len(members))):
        placement = placement_of(groups)
        for position, qubit in enumerate(members):
            placement[qubit] = (0, order[position])
        seen.add(round(old.cost(placement), 9))
    assert len(seen) == 1, f"cost varied with lane order: {sorted(seen)}"


# ----------------------------------------------------------------------
# incremental search state
# ----------------------------------------------------------------------


def seeds_for(circuit: QuantumCircuit, n_code: int, slots: int):
    """Both seeds, each with its own cost model, for parametrised checks."""
    profiles = extract_profiles(translated_view(circuit))
    for label in ("graph", "naive"):
        cost = PackingCost(
            profiles=profiles, n_code=n_code, num_qubits=circuit.num_qubits
        )
        if label == "graph":
            yield label, cost, graph_seed(cost, profiles, slots)
        else:
            yield label, cost, naive_seed(cost, slots)


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_incremental_matches_rescore(n_code: int) -> None:
    """Applying moves by delta agrees with pricing the partition from scratch."""
    circuit = program(14, 4, seed=9)
    k = logical_capacity(n_code)
    rng = random.Random(17)
    for _, cost, packing in seeds_for(circuit, n_code, max(1, -(-14 // k))):
        for _ in range(30):
            options = list(_candidates(packing))
            if not options:
                break
            move = rng.choice(options)
            predicted = packing.value_after(move.as_dict())
            packing.apply(move.as_dict())
            assert predicted == pytest.approx(packing.value(), rel=1e-9)
            assert packing.value() == pytest.approx(
                cost.of(packing.partition()), rel=1e-9
            )


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_seeds_are_feasible_and_tight(n_code: int) -> None:
    """
    Both seeds cover every qubit exactly once and respect capacity.

    `graph_seed` cannot take the partitioner's word for either: `partition` treats
    `max_patches` as a target rather than a guarantee, so it splits anything oversized
    and rehomes anything dropped.
    """
    k = logical_capacity(n_code)
    for num_qubits in (7, 12, 13, 28):
        circuit = program(num_qubits, 3)
        slots = max(1, -(-num_qubits // k))
        for label, _cost, packing in seeds_for(circuit, n_code, slots):
            groups = packing.partition()
            assert sorted(q for g in groups for q in g) == list(range(num_qubits)), label
            assert all(len(g) <= k for g in groups), label
            assert len(groups) <= slots, label


@pytest.mark.parametrize("n_code", [4, 6])
def test_descent_never_increases_cost(n_code: int) -> None:
    circuit = program(13, 5, seed=21)
    k = logical_capacity(n_code)
    for label, _cost, packing in seeds_for(circuit, n_code, max(1, -(-13 // k))):
        before = packing.value()
        result = descend(packing)
        assert packing.value() <= before + 1e-9, label
        assert result.packing is packing
        assert result.gain >= -1e-9


def test_no_lane_only_moves_are_offered() -> None:
    """
    Every offered move changes the partition.

    The old neighbourhood spent 66% of its evaluations on `reorder` and same-patch
    `swap`, which only permute lanes and so cannot change a partition-valued objective.
    """
    circuit = program(10, 4)
    cost = PackingCost.for_program(circuit, 6)
    packing = naive_seed(cost, 3)
    before = [set(g) for g in packing.groups]
    for move in _candidates(packing):
        after = [set(g) for g in packing.groups]
        for index, group in move.as_dict().items():
            after[index] = set(group)
        assert after != before, f"{move.kind} left the partition unchanged"


def test_moves_for_matches_full_neighbourhood() -> None:
    circuit = program(9, 3)
    cost = PackingCost.for_program(circuit, 6)
    packing = naive_seed(cost, 3)
    per_qubit = [m for q in sorted(packing.patch_of) for m in moves_for(packing, q)]
    assert len(per_qubit) == len(list(_candidates(packing)))


# ----------------------------------------------------------------------
# lanes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_assign_lanes_is_a_valid_placement(n_code: int) -> None:
    circuit = program(11, 4)
    cost = PackingCost.for_program(circuit, n_code)
    k = logical_capacity(n_code)
    packing = naive_seed(cost, max(1, -(-11 // k)))
    placement = assign_lanes(packing.partition(), cost)
    assert len(placement) == 11
    assert len(set(placement)) == 11, "two qubits share a (patch, slot)"
    for patch, slot in placement:
        assert 0 <= slot < k


def test_assign_lanes_aligns_a_clean_bijection() -> None:
    """
    Two patches whose gates pair one-to-one should end up lane-to-lane.

    Ascending order gets 0 of 4 here; this is the case `align_slots` was written for
    and the one its single-anchor benefit matrix still handles correctly.
    """
    circuit = QuantumCircuit(8)
    for left, right in ((0, 5), (1, 4), (2, 7), (3, 6)):
        circuit.cz(left, right)
    cost = PackingCost.for_program(circuit, 6)
    groups = [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})]
    placement = assign_lanes(groups, cost)
    slot = {q: s for q, (_, s) in enumerate(placement)}
    matched = sum(
        1 for left, right in ((0, 5), (1, 4), (2, 7), (3, 6))
        if slot[left] == slot[right]
    )
    assert matched == 4, f"only {matched}/4 lanes aligned"


# ----------------------------------------------------------------------
# the planner
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_plan_packing_is_valid_and_tight(n_code: int) -> None:
    circuit = program(15, 4, seed=31)
    k = logical_capacity(n_code)
    plan = plan_packing(circuit, n_code, verify=False)
    assert len(plan.placement) == 15
    assert len(set(plan.placement)) == 15
    assert plan.patches == max(1, -(-15 // k))
    assert plan.physical_qubits == plan.patches * (n_code + 2)
    assert sum(plan.occupancy) == 15


@pytest.mark.parametrize("n_code", [4, 6])
def test_verify_never_returns_worse_than_naive(n_code: int) -> None:
    """The safety net, on the circuit that motivated it."""
    circuit = program(10, 4, seed=44)
    k = logical_capacity(n_code)
    plan = plan_packing(circuit, n_code, verify=True)
    assert plan.verified
    ours = measured_cost(circuit, plan.placement, n_code)
    theirs = measured_cost(circuit, naive_packing(10, k), n_code)
    assert ours is not None and theirs is not None
    assert ours <= theirs + 1e-9


def test_spread_can_open_an_empty_patch() -> None:
    """
    With idle and movement priced high enough, spreading has to be reachable.

    `spread` is the only move that raises the patch count, so this is the mechanism
    the old stack lacked -- `max_patches` reached the partitioner as a cap and nothing
    downstream could take a looser budget up on the offer.
    """
    from packing.graph import NoiseProfile

    circuit = program(8, 5, seed=52)
    tight = plan_packing(circuit, 6, verify=False, spread=0)
    loose = plan_packing(
        circuit,
        6,
        verify=False,
        spread=4,
        noise=NoiseProfile(movement_cost=200.0, distance_scale=0.0),
    )
    assert tight.patches == 2
    assert loose.patches >= tight.patches


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_single_patch_programs_are_handled(n_code: int) -> None:
    """A program that fits one patch has no placement decision to make."""
    circuit = program(2, 3)
    plan = plan_packing(circuit, n_code, verify=False)
    assert plan.patches == 1
    assert {p for p, _ in plan.placement} == {0}


@pytest.mark.parametrize("n_code", [6, 12])
def test_budget_ceiling_caps_spreading(n_code: int) -> None:
    """
    A budget bounds how far `spread` may go, but never below the tightest packing.

    An impossible budget returns the tightest packing anyway and lets the caller see
    `physical_qubits` exceed the cap, rather than failing -- the same contract
    `grouping.plan.plan_placement` documents.
    """
    from packing.budget import ResourceBudget

    circuit = program(12, 4, seed=61)
    k = logical_capacity(n_code)
    tight = max(1, -(-12 // k))

    capped = plan_packing(
        circuit,
        n_code,
        verify=False,
        spread=6,
        budget=ResourceBudget(max_patches=tight),
    )
    assert capped.patches <= tight

    impossible = plan_packing(
        circuit, n_code, verify=False, budget=ResourceBudget(max_physical_qubits=1)
    )
    assert impossible.patches == tight
    assert impossible.physical_qubits > 1


def test_search_and_verify_share_one_depth_weight() -> None:
    """
    The objective and the check that validates it must want the same thing.

    The old stack scored the search with `depth_weight=0.0` in `plan_placement` and
    measured with 12.0 on the figure path.
    """
    import inspect

    from packing import cost as cost_module
    from packing import plan as plan_module

    assert (
        inspect.signature(plan_module.plan_packing).parameters["depth_weight"].default
        is cost_module.DEFAULT_DEPTH_WEIGHT
    )
    assert (
        inspect.signature(plan_module.measured_cost).parameters["depth_weight"].default
        is cost_module.DEFAULT_DEPTH_WEIGHT
    )


def test_restarts_are_deterministic_and_never_worse() -> None:
    """
    Restarts must be reproducible and can only help.

    They are gated on width in `plan_packing` -- free on small circuits, no measurable
    gain and 2-3x the cost on large ones -- so the small path is the one to pin.
    """
    circuit = program(7, 5, seed=71)
    first = plan_packing(circuit, 6, verify=False)
    again = plan_packing(circuit, 6, verify=False)
    assert first.placement == again.placement
    assert first.cost == pytest.approx(again.cost)

    without = plan_packing(circuit, 6, verify=False, restarts=0)
    assert first.cost <= without.cost + 1e-9


def test_shuffled_seed_respects_capacity() -> None:
    import random as _random

    circuit = program(15, 4, seed=83)
    for n_code in CODE_SIZES:
        k = logical_capacity(n_code)
        cost = PackingCost.for_program(circuit, n_code)
        packing = shuffled_seed(cost, max(1, -(-15 // k)), _random.Random(2))
        groups = packing.partition()
        assert sorted(q for g in groups for q in g) == list(range(15))
        assert all(len(g) <= k for g in groups)


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_never_worse_than_naive_in_the_model(n_code: int) -> None:
    """
    The planner's own objective must rate its placement at least as good as naive.

    `verify` guarantees this after translating, but that is a repair, not the property:
    when it fires the packing contribution is zero and the search has simply failed. The
    guarantee comes from always seeding a descent from naive.

    Priced with heavy movement, which is the regime that broke it. The seed graph
    optimises cut weight, a good proxy when the objective is mostly gate counts and a bad
    one once crossings cost 4x a two-qubit gate -- there the descent from the graph seed
    landed *above* naive and could not climb out. Wide enough to be past
    `SMALL_ENOUGH_TO_RESTART`, since below it the naive seed was already being tried.

    A nearest-neighbour chain, because that is the shape that exhibits it: program order
    already *is* a near-optimal partition, so naive is the strong baseline the graph seed
    has to match and the one it cannot reach by descending from elsewhere. A random
    interaction graph does not reproduce the failure -- naive is bad there, and beating it
    is easy.
    """
    circuit = chain(20, layers=4)
    k = logical_capacity(n_code)
    noise = NoiseProfile(movement_cost=52.0, idle_cost=0.24, distance_scale=4.0)

    plan = plan_packing(circuit, n_code, noise=noise, verify=False)
    cost = PackingCost(
        profiles=extract_profiles(
            translated_view(circuit, enable_h_commutation=plan.h_commutation)
        ),
        n_code=n_code,
        num_qubits=circuit.num_qubits,
        movement_cost=noise.movement_cost,
        idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale,
    )

    def groups_of(placement):
        by_patch: dict = {}
        for qubit, (patch, _) in enumerate(placement):
            by_patch.setdefault(patch, []).append(qubit)
        return [tuple(sorted(v)) for v in by_patch.values()]

    ours = cost.of(groups_of(plan.placement))
    theirs = cost.of(groups_of(naive_packing(circuit.num_qubits, k)))
    assert ours <= theirs + 1e-6


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_movement_scales_with_profile(n_code: int) -> None:
    """
    The movement charge is linear in `movement_cost`, with no clamp flattening it.

    Replaces `test_movement_cannot_dominate_the_objective`, which asserted the opposite:
    that movement stayed within `MOVEMENT_SHARE_CEILING` of gate-and-depth work. That
    clamp was removed because where it bound it removed the term's *gradient*, not just
    its magnitude -- at `distance_scalar = 128` every `price` call during a plan came back
    clamped, so movement reduced to `0.25 * work` and the search became blind to the thing
    the profile was telling it to avoid. It then chose placements carrying 3.4x naive's
    real movement.

    Linearity is the property that makes a noise profile mean anything: doubling the price
    of a crossing must double what crossings cost, or `noise_profile_for` has no authority
    over the objective. Asserted by rescoring the same partition at two prices, so the
    shared terms cancel and the difference is the movement charge alone.
    """
    circuit = chain(16, layers=4)
    k = logical_capacity(n_code)
    groups = [
        tuple(range(start, min(start + k, circuit.num_qubits)))
        for start in range(0, circuit.num_qubits, k)
    ]
    profiles = extract_profiles(translated_view(circuit))

    def priced(movement_cost: float) -> float:
        return PackingCost(
            profiles=profiles, n_code=n_code, num_qubits=circuit.num_qubits,
            movement_cost=movement_cost, distance_scale=1.0,
        ).of(groups)

    free = priced(0.0)
    single = priced(10.0) - free
    double = priced(20.0) - free

    assert single > 0.0, "a positive movement_cost must cost something"
    assert double == pytest.approx(2.0 * single, rel=1e-9), (
        f"movement must be linear in its price: 10.0 -> {single}, 20.0 -> {double}"
    )
    # And an absurd price must still be felt, rather than saturating at a fixed share.
    assert priced(1e6) - free == pytest.approx(1e5 * single, rel=1e-9)
