"""
standard_packers.py -- Experiment1's packing baselines (`experiments/revision/packers.py`) under
the current standard: the model-based ones search AQORN's own objective (`SilentAwareCost`, the
weight-10 silent price, transversal in-patch CZ / XCX, solo-HH pricing) instead of the submitted
`PackingCost`, and AQORN is `plan_standard`. The compiler itself is untouched (frozen 2026-10-10).

Arms, every one at AQORN's H-commutation decision so placements differ in nothing else:

    aqorn                          `plan_standard` (verify on): the compiler
    aqorn_raw                      the same search, no verify
    aqorn_seed_only                silent-aware graph -> partition -> lanes; no descent
    aqorn_descent_from_naive       descent from program order only; no graph, no restarts
    annealing                      simulated annealing on AQORN's objective (pair swaps), several
                                   restarts in a time budget: an estimate of the objective's
                                   optimum where exhaustive search is impossible
    naive                          program order
    random_budget                  best of N random balanced packings on AQORN's objective, N =
                                   the evaluations AQORN's own search spent (time-capped)
    interaction_matching           k = 2: max-weight matching of the 2q-interaction graph
    interaction_partition          that graph through AQORN's multilevel partitioner
    interaction_partition_descent  the above, then AQORN's descent
    hypergraph_24                  [24] stage 1 (Pauli-based-computation hypergraph, km1); built
                                   from the program's non-Clifford rotations, so plan on the
                                   original, not the Clifford proxy
    greedy_pairs                   k = 2: the paper's earlier pairwise greedy

Lanes come from `packing.lanes.assign_lanes` for every arm but naive and greedy_pairs (which
return their own), as in Experiment1.
"""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

Perm = List[Tuple[int, int]]

N_CODE = 4
ARMS: Tuple[str, ...] = (
    "naive", "random_budget", "interaction_matching", "interaction_partition",
    "interaction_partition_descent", "hypergraph_24", "greedy_pairs", "aqorn_seed_only",
    "aqorn_descent_from_naive", "annealing", "aqorn_raw", "aqorn",
)


@contextmanager
def standard_gadgets():
    from iceberg.gadget_options import transversal_intra
    from packing.standard import STANDARD

    with transversal_intra(STANDARD.transversal_intra_two_qubit):
        yield


def make_standard_context(program, commute: bool):
    """`revision.packers.PackContext` with AQORN's own objective as `cost`."""
    from experiments.revision.packers import PackContext
    from iceberg.utils import logical_capacity
    from packing.silent import build_costs, load_silent_rates
    from packing.standard import STANDARD, standard_noise_profile

    noise = standard_noise_profile()
    rates = load_silent_rates(N_CODE,
                              transversal_intra_two_qubit=STANDARD.transversal_intra_two_qubit)
    with standard_gadgets():
        profiles, cost = build_costs(program, N_CODE, commute=commute, noise=noise, rates=rates,
                                     silent_weight=STANDARD.silent_weight_for(N_CODE),
                                     solo_hh=STANDARD.solo_hh_aware)
    from packing.tables import translated_view

    k = logical_capacity(N_CODE)
    tightest = max(1, -(-program.num_qubits // k))
    return PackContext(program=program, n_code=N_CODE, commute=commute, noise=noise,
                       requested_noise=noise, k=k,
                       view=translated_view(program, enable_h_commutation=commute),
                       profiles=profiles, cost=cost, tightest=tightest, slots=tightest)


def _lanes(ctx, groups) -> Perm:
    from packing.lanes import assign_lanes

    return [tuple(p) for p in assign_lanes([frozenset(g) for g in groups if g], ctx.cost)]


# --- AQORN and its ablations ---------------------------------------------------------------


def aqorn(ctx) -> Perm:
    from packing.standard import plan_standard

    plan = plan_standard(ctx.program)
    ctx.notes["aqorn_commute"] = bool(plan.h_commutation)
    ctx.notes["aqorn_fell_back"] = bool(plan.fell_back)
    return [tuple(p) for p in plan.placement]


def aqorn_raw(ctx) -> Perm:
    from packing.standard import plan_standard

    plan = plan_standard(ctx.program, h_commutation=ctx.commute, verify=False)
    return [tuple(p) for p in plan.placement]


def aqorn_seed_only(ctx) -> Perm:
    from packing.seed import DEFAULT_PARTITIONER
    from packing.silent import _seed

    packing = _seed(ctx.cost, ctx.profiles, ctx.slots, noise=ctx.noise,
                    partitioner=DEFAULT_PARTITIONER)
    return _lanes(ctx, packing.partition())


def aqorn_descent_from_naive(ctx) -> Perm:
    from packing.search import descend, naive_seed

    packing = naive_seed(ctx.cost, ctx.slots)
    descend(packing, max_rounds=8)
    return _lanes(ctx, packing.partition())


def aqorn_evaluations(ctx) -> int:
    """Move evaluations of AQORN's own search (seed descent + restarts), for random_budget."""
    from packing.plan import SMALL_ENOUGH_TO_RESTART
    from packing.search import descend, naive_seed, shuffled_seed
    from packing.seed import DEFAULT_PARTITIONER
    from packing.silent import _seed

    packing = _seed(ctx.cost, ctx.profiles, ctx.slots, noise=ctx.noise,
                    partitioner=DEFAULT_PARTITIONER)
    evaluations = descend(packing, max_rounds=8).evaluations
    restarts = 4 if ctx.num_qubits <= SMALL_ENOUGH_TO_RESTART else 0
    rng = random.Random(0)
    for candidate in [naive_seed(ctx.cost, ctx.slots)] + [
            shuffled_seed(ctx.cost, ctx.slots, rng) for _ in range(restarts)]:
        evaluations += descend(candidate, max_rounds=8).evaluations
    return int(evaluations)


def random_budget(ctx, seconds: float = 120.0) -> Perm:
    from experiments.revision.packers import random_budget as _random_budget

    return _random_budget(ctx, aqorn_evaluations(ctx), seconds=seconds)


def annealing(ctx, seconds: float = 300.0, seed: int = 0) -> Perm:
    """
    Simulated annealing over balanced partitions (swap two qubits in different patches) on
    AQORN's objective, restarted from random partitions until `seconds` run out; the best
    partition seen. Temperature: geometric from 5% to 0.01% of the start's value per sweep.
    """
    from experiments.revision.packers import _random_groups

    rng = random.Random(seed)
    n, cost = ctx.num_qubits, ctx.cost
    best_value, best_groups = math.inf, None
    began = time.time()
    restarts = 0
    sweeps = 200
    while time.time() - began < seconds:
        restarts += 1
        groups = [set(g) for g in _random_groups(n, ctx.slots, ctx.k, rng)]
        value = cost.of([frozenset(g) for g in groups])
        hot, cold = 0.05 * abs(value), 1e-4 * abs(value)
        if restarts == 1:
            # On a wide program one full schedule can outlast the budget, and a schedule cut
            # off while still hot is a random walk. Size it so a restart takes at most half
            # the budget (unchanged wherever 200 sweeps fit, e.g. multiplier_n15).
            probe = time.time()
            for _ in range(20):
                cost.of([frozenset(g) for g in groups])
            per_move = (time.time() - probe) / 20
            sweeps = int(max(20, min(200, (seconds / 2) / max(per_move * 4 * n, 1e-9))))
            ctx.notes["annealing_sweeps"] = sweeps
        for sweep in range(sweeps):
            temperature = hot * (cold / hot) ** (sweep / (sweeps - 1))
            for _ in range(4 * n):
                a, b = rng.sample(range(len(groups)), 2)
                if not groups[a] or not groups[b]:
                    continue
                qa, qb = rng.choice(sorted(groups[a])), rng.choice(sorted(groups[b]))
                groups[a].remove(qa), groups[b].remove(qb)
                groups[a].add(qb), groups[b].add(qa)
                trial = cost.of([frozenset(g) for g in groups])
                if trial <= value or rng.random() < math.exp(-(trial - value) / temperature):
                    value = trial
                    if value < best_value - 1e-12:
                        best_value, best_groups = value, [set(g) for g in groups]
                else:
                    groups[a].remove(qb), groups[b].remove(qa)
                    groups[a].add(qa), groups[b].add(qb)
            if time.time() - began > seconds:
                break
        cost._unary.clear()
        cost._binary.clear()
    ctx.notes["annealing_restarts"] = restarts
    ctx.notes["annealing_value"] = best_value
    return _lanes(ctx, best_groups)


# --- conventional baselines (Experiment1's, on this context) ---------------------------------


def _from_revision(name: str) -> Callable:
    def run(ctx):
        from experiments.revision import packers

        return getattr(packers, name)(ctx)
    return run


ARM_FUNCTIONS: Dict[str, Callable] = {
    "aqorn": aqorn,
    "aqorn_raw": aqorn_raw,
    "aqorn_seed_only": aqorn_seed_only,
    "aqorn_descent_from_naive": aqorn_descent_from_naive,
    "annealing": annealing,
    "random_budget": random_budget,
    "naive": _from_revision("naive"),
    "interaction_matching": _from_revision("interaction_matching"),
    "interaction_partition": _from_revision("interaction_partition"),
    "interaction_partition_descent": _from_revision("interaction_partition_descent"),
    "hypergraph_24": _from_revision("hypergraph_24"),
    "greedy_pairs": _from_revision("greedy_pairs"),
}


def run_arms(program, *, arms=ARMS, annealing_seconds: float = 300.0,
             random_seconds: float = 120.0) -> Tuple[Dict[str, Optional[Perm]], Dict, bool]:
    """
    ({arm: placement or None}, notes, commute): AQORN decides the commutation setting, every
    other arm runs at it. A failing arm is None, its error in the notes.
    """
    from packing.standard import plan_standard

    commute = bool(plan_standard(program).h_commutation)
    ctx = make_standard_context(program, commute)
    out: Dict[str, Optional[Perm]] = {}
    for arm in arms:
        began = time.time()
        try:
            with standard_gadgets():
                if arm == "annealing":
                    placement = annealing(ctx, seconds=annealing_seconds)
                elif arm == "random_budget":
                    placement = random_budget(ctx, seconds=random_seconds)
                else:
                    placement = ARM_FUNCTIONS[arm](ctx)
            out[arm] = [tuple(int(v) for v in p) for p in placement] if placement else None
        except Exception as error:  # noqa: BLE001 - an arm that cannot run is recorded
            out[arm] = None
            ctx.notes[f"{arm}_error"] = f"{type(error).__name__}: {error}"[:200]
        ctx.notes[f"{arm}_seconds"] = round(time.time() - began, 2)
    return out, ctx.notes, commute
