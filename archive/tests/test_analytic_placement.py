"""
Tests for the analytic cost model, the signed interaction graph, the multilevel
partitioner and the planner.

The one that matters is `test_model_tracks_the_translator`: everything else in this
stack optimises against the model, so if the model and the translator disagree the
rest is confidently wrong. The pairwise packer's estimate correlates with real cost
at r = -0.17, which is what that failure looks like in practice.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.gate_count_stats import one_two_qubit_gate_totals
from archive.grouping.analytic import AnalyticCost, translated_view, validate
from archive.grouping.graph import NoiseProfile, build_graph, hadamard_affinity
from archive.grouping.multilevel import align_slots, partition, place
from archive.grouping.plan import ResourceBudget, plan_placement, sweep_patch_counts
from archive.grouping.profiles import extract_profiles
from iceberg.utils import logical_capacity
from translator.config import TranslatorConfig
from translator.core import Translator

CODE_SIZES = [4, 6, 8, 12]
ONE_QUBIT = ("h", "s", "sx", "x", "z")


def program(num_qubits: int, depth: int, seed: int = 5) -> QuantumCircuit:
    """A random Clifford program without SWAPs, which are rejected cross-patch."""
    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        if rng.random() < 0.5:
            left, right = rng.sample(range(num_qubits), 2)
            getattr(circuit, rng.choice(("cx", "cz")))(left, right)
        else:
            getattr(circuit, rng.choice(ONE_QUBIT))(rng.randrange(num_qubits))
    return circuit


def sequential(num_qubits: int, k: int) -> List[Tuple[int, int]]:
    return [(index // k, index % k) for index in range(num_qubits)]


def shuffled_placements(num_qubits: int, k: int, count: int, seed: int):
    rng = random.Random(seed)
    slots = [
        (patch, slot)
        for patch in range(-(-num_qubits // k))
        for slot in range(k)
    ][:num_qubits]
    out = []
    for _ in range(count):
        candidate = slots[:]
        rng.shuffle(candidate)
        out.append(candidate)
    return out


# --- the model against reality ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_model_tracks_the_translator(n_code):
    """Rank agreement with real translated gate counts, over random placements."""
    k = logical_capacity(n_code)
    circuit = program(3 * k, 8 * k)
    report = validate(
        circuit, n_code, placements=shuffled_placements(3 * k, k, 25, seed=n_code)
    )
    assert report.samples >= 10, "not enough placements survived to judge the model"
    assert report.usable, f"model is not safe to optimise against: {report.summary()}"


@pytest.mark.parametrize("n_code", [4, 6, 8])
def test_model_is_exact_without_merging(n_code):
    """
    With merging off the cost is a plain sum of per-gate costs, so it must be exact.

    This localises any disagreement: if this passes and
    `test_model_tracks_the_translator` fails, the gap is the macro estimate rather
    than the gate accounting.
    """
    k = logical_capacity(n_code)
    circuit = program(2 * k, 6 * k, seed=17)
    placement = sequential(2 * k, k)

    model = AnalyticCost.for_program(circuit, n_code, depth_weight=0.0)
    model._merge_savings_within = lambda *args, **kwargs: 0.0  # noqa: SLF001
    model._transversal_savings = lambda *args, **kwargs: 0.0  # noqa: SLF001

    translator = Translator(
        config=TranslatorConfig(n_code=n_code, enable_merging=False)
    )
    one, two = one_two_qubit_gate_totals(translator.translate(circuit, placement).circuit)
    measured = one + 13 * two
    assert model.cost(placement) == pytest.approx(measured, rel=0.02)


def test_scoring_the_raw_program_is_wrong():
    """
    `for_program` must apply the translator's own passes.

    Scoring the program as written rather than as translated was worth a 13-49%
    error, because H commutation rewrites and cancels gates before placement. This
    pins the difference so the trap cannot quietly return.
    """
    circuit = program(8, 60, seed=3)
    raw = AnalyticCost(profiles=extract_profiles(circuit), n_code=6, depth_weight=0.0)
    prepared = AnalyticCost.for_program(circuit, 6, depth_weight=0.0)
    placement = sequential(8, 4)
    assert raw.cost(placement) != pytest.approx(prepared.cost(placement), rel=0.01)

    translator = Translator(config=TranslatorConfig(n_code=6))
    one, two = one_two_qubit_gate_totals(translator.translate(circuit, placement).circuit)
    assert prepared.cost(placement) == pytest.approx(one + 13 * two, rel=0.05)


def test_caching_makes_a_local_move_cheap():
    """A swap must reuse the terms it did not touch, or the search gains nothing."""
    circuit = program(24, 200, seed=9)
    model = AnalyticCost.for_program(circuit, 6, depth_weight=0.0)
    placement = sequential(24, 4)
    model.cost(placement)

    before = model.misses
    moved = list(placement)
    moved[0], moved[1] = moved[1], moved[0]
    model.cost(moved)
    recomputed = model.misses - before
    assert recomputed <= 6, f"a single swap recomputed {recomputed} terms"


# --- the signed graph ---


def test_cx_carries_no_gate_signal_and_depth_breaks_the_tie():
    """
    The whole point of signing the weights: not every gate wants co-location.

    CX costs (0, 4) intra *and* cross, so on gate counts alone it carries no grouping
    signal whatsoever -- a CNOT-heavy circuit yields a nearly empty graph and an
    arbitrary partition. Depth breaks the tie in the *splitting* direction, because
    cross-patch CNOT is a layer shallower (3 against 4). CZ is cheaper intra on gates,
    so it pulls together either way.
    """
    cx = QuantumCircuit(2)
    cx.cx(0, 1)
    cz = QuantumCircuit(2)
    cz.cz(0, 1)
    off = dict(include_hadamard_affinity=False, include_transversal_affinity=False)

    gates_only = build_graph(extract_profiles(cx), 6, depth_weight=0.0, **off)
    assert gates_only.weight(0, 1) == pytest.approx(0.0), (
        "gate counts cannot distinguish intra- from cross-patch CNOT"
    )

    with_depth = build_graph(extract_profiles(cx), 6, **off)
    assert with_depth.weight(0, 1) < 0, "cross-patch CNOT is shallower, so CX should split"

    assert build_graph(extract_profiles(cz), 6, **off).weight(0, 1) > 0


def test_movement_noise_makes_every_gate_attract():
    """On a movement-dominated machine even a CX prefers to stay local."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    profiles = extract_profiles(circuit)
    off = dict(include_hadamard_affinity=False, include_transversal_affinity=False)
    free = build_graph(profiles, 6, **off)
    costly = build_graph(profiles, 6, noise=NoiseProfile.movement_dominated(), **off)
    assert costly.weight(0, 1) > free.weight(0, 1)


def test_hadamard_coincidence_creates_an_edge_where_no_gate_exists():
    """
    Two qubits that never share a gate but always Hadamard together should attract.

    Without this the partitioner is blind to the whole-patch macro, which is the
    largest saving available when H commutation is not sweeping Hadamards away.
    """
    circuit = QuantumCircuit(3)
    for _ in range(5):
        circuit.h(0)
        circuit.h(1)
        circuit.barrier()
    circuit.x(2)

    affinity = hadamard_affinity(extract_profiles(circuit), 6)
    assert affinity.get((0, 1), 0.0) > 0
    assert (0, 2) not in affinity and (1, 2) not in affinity


# --- partitioning ---


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_partition_respects_capacity_and_places_everyone(n_code):
    k = logical_capacity(n_code)
    num_qubits = 3 * k + 1
    circuit = program(num_qubits, 10 * k, seed=n_code + 1)
    graph = build_graph(
        extract_profiles(translated_view(circuit)), n_code, num_qubits=num_qubits
    )
    result = partition(graph, n_code, max_patches=-(-num_qubits // k))

    placed = sorted(q for group in result.groups for q in group)
    assert placed == list(range(num_qubits))
    assert all(len(group) <= k for group in result.groups)


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_partition_meets_the_patch_budget(n_code):
    """
    Overshooting the budget is not a cosmetic problem.

    Every patch pays for its own encode and decode, so extra patches made one
    placement 16% *worse* than naive before `_consolidate` existed.
    """
    k = logical_capacity(n_code)
    num_qubits = 4 * k
    circuit = program(num_qubits, 12 * k, seed=n_code + 2)
    graph = build_graph(
        extract_profiles(translated_view(circuit)), n_code, num_qubits=num_qubits
    )
    target = -(-num_qubits // k)
    assert partition(graph, n_code, max_patches=target).num_patches <= target


def test_partition_separates_planted_blocks():
    """Two triples that interact only internally should land in their own patches."""
    circuit = QuantumCircuit(6)
    for _ in range(5):
        for left, right in ((0, 2), (2, 4), (0, 4)):
            circuit.cz(left, right)
        for left, right in ((1, 3), (3, 5), (1, 5)):
            circuit.cz(left, right)
    # Transversal affinity deliberately competes with block structure here (the
    # blocks fire in the same layers), so this checks the cut term in isolation.
    graph = build_graph(
        extract_profiles(circuit), 6, num_qubits=6,
        include_transversal_affinity=False,
    )
    groups = {frozenset(g) for g in partition(graph, 6, max_patches=2).groups}
    assert frozenset({0, 2, 4}) in groups
    assert frozenset({1, 3, 5}) in groups


# --- slot alignment ---


@pytest.mark.parametrize("n_code", [6, 8, 12])
def test_alignment_puts_partners_on_matching_lanes(n_code):
    """
    Transversal macros pair lane i with lane i, so partners belong on equal slots.

    Two patches wired straight across should come out perfectly aligned; the
    assignment problem per patch pair is solved exactly, so there is no excuse for
    a near miss.
    """
    k = logical_capacity(n_code)
    circuit = QuantumCircuit(2 * k)
    for lane in range(k):
        for _ in range(3):
            circuit.cz(lane, k + lane)

    profiles = extract_profiles(circuit)
    groups = [list(range(k)), list(range(k, 2 * k))]
    placement = align_slots(groups, profiles, n_code)
    for lane in range(k):
        assert placement[lane][1] == placement[k + lane][1], (
            f"qubit {lane} and its partner {k + lane} landed on different lanes"
        )


def test_alignment_yields_a_valid_placement():
    circuit = program(12, 90, seed=4)
    profiles = extract_profiles(circuit)
    graph = build_graph(profiles, 6, num_qubits=12)
    placement, _ = place(graph, profiles, 6, max_patches=3)
    assert len(placement) == 12
    assert len(set(placement)) == 12, "two qubits share a slot"
    assert all(0 <= slot < 4 for _, slot in placement)


# --- planning ---


def test_planner_respects_a_physical_qubit_budget():
    """A 64-atom machine at n=8 can afford 6 patches, and the plan must know it."""
    circuit = program(20, 150, seed=6)
    plan = plan_placement(
        circuit, 8, budget=ResourceBudget(max_physical_qubits=64)
    )
    assert plan.physical_qubits <= 64
    assert plan.num_patches <= 64 // (8 + 2)


def test_planner_beats_or_matches_sequential():
    """Placing deliberately should never lose to filling patches in index order."""
    for n_code in (6, 8):
        k = logical_capacity(n_code)
        circuit = program(3 * k, 12 * k, seed=n_code + 5)
        plan = plan_placement(circuit, n_code)
        model = AnalyticCost.for_program(circuit, n_code, depth_weight=0.0)
        assert plan.cost <= model.cost(sequential(3 * k, k)) + 1e-6


def test_patch_count_sweep_is_monotone_in_resources():
    """Each entry should use more physical qubits than the last, for plotting."""
    circuit = program(12, 80, seed=8)
    rows = sweep_patch_counts(circuit, 6)
    assert rows, "sweep produced nothing"
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    assert all(r[1] == r[0] * 8 for r in rows)


def test_planner_scales_to_a_wide_circuit():
    """
    A 140-qubit placement has to be quick, since that was the whole motivation.

    The translate-per-candidate search needed ~5.5 hours for one code size here, so
    the bar is minutes rather than hours. Refinement on the analytic objective is what
    fills most of it -- the patch-count scan is cheap because it skips refinement.
    """
    import time

    circuit = program(140, 600, seed=2)
    started = time.perf_counter()
    plan = plan_placement(circuit, 8)
    elapsed = time.perf_counter() - started
    assert elapsed < 90.0, f"planning 140 qubits took {elapsed:.1f}s"
    assert len(plan.placement) == 140


# --- patch-count selection: the spread mechanism -----------------------------------


def test_free_slots_only_sees_empty_patches_when_told():
    """
    `relocate` cannot spread unless it is offered the empty patches.

    This is why `plan_placement`'s patch-count scan had no effect for so long: the
    partitioner only ever merges, so nothing downstream could act on a looser budget.
    """
    from archive.grouping.refine import _free_slots

    placement = [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert _free_slots(placement, 4) == [], "a full patch has no free slots of its own"
    offered = {patch for patch, _ in _free_slots(placement, 4, num_patches=4)}
    assert offered == {1, 2, 3}, "patches 1-3 are empty and must be reachable"


def test_patch_count_scan_can_spread_when_it_pays():
    """
    The scan must be able to choose more than the tightest packing.

    An extra patch costs `ENCODE + DECODE` -- 234 weighted units at n=6 -- against 3
    units for each CNOT moved from intra- to cross-patch at the default `depth_weight`.
    So it correctly declines on a gate-counting machine and has to be given a reason:
    here, a depth weight large enough for the crossings to pay.
    """
    from archive.grouping import NoiseProfile, plan_placement

    program = QuantumCircuit(4)
    for step in range(30):
        program.cx(step % 4, (step + 1) % 4)

    tight = plan_placement(program, 6, noise=NoiseProfile(), depth_weight=3.0)
    assert tight.num_patches == 1, "one patch holds all four qubits at k=4"

    spread = plan_placement(program, 6, noise=NoiseProfile(), depth_weight=120.0)
    assert spread.num_patches > 1, (
        "with crossings priced high enough to beat encode/decode the scan must spread; "
        f"got {spread.num_patches} patches from {spread.considered}"
    )


def test_extra_patches_are_priced_by_movement_and_idle():
    """Both new charges must rise with the patch count, or spreading is free."""
    from archive.grouping import AnalyticCost, mean_patch_hops

    assert mean_patch_hops(1) == 0.0
    assert mean_patch_hops(4) < mean_patch_hops(16) < mean_patch_hops(64), "grows with P"

    program = QuantumCircuit(8)
    for qubit in range(8):
        program.h(qubit)
    for qubit in range(7):
        program.cx(qubit, qubit + 1)
    tight = [(index // 2, index % 2) for index in range(8)]     # 4 patches
    spread = [(index, 0) for index in range(8)]                 # 8 patches

    model = AnalyticCost.for_program(program, 4, movement_cost=6.0, idle_cost=0.01)
    assert model.breakdown(spread).movement > model.breakdown(tight).movement
    assert model.breakdown(spread).idle > model.breakdown(tight).idle

    free = AnalyticCost.for_program(program, 4)
    assert free.breakdown(spread).movement == 0.0, "inert unless movement_cost is set"
    assert free.breakdown(spread).idle == 0.0, "inert unless idle_cost is set"
