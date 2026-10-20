"""
packers.py -- Experiment1's packing baselines, each run under AQORN's own pipeline.

Compares conventional interaction- and locality-based partitioning, AQORN's signed-graph
seed and its local refinement under identical compilation passes, plus a simple greedy
search at a comparable budget, with commutation and merging held fixed.

Every packer here returns a placement for a *fixed* commutation decision -- the one the
shipped compiler (`plan_packing(h_commutation="auto")`) chose for that circuit and code
size -- and every caller translates every arm under that decision with merging on, so
packers differ in the placement and nothing else (gotcha 4). The objective objects are
built exactly as `packing.plan._plan_under` builds them, from the commuted view, so the
model-based packers price the circuit the translator will place.

Keys, matching the plan's table:

    naive                        program order
    random_budget                best of N random balanced packings on `PackingCost`, N =
                                 every evaluation AQORN's search spent at this commutation
                                 setting (all seeds' descents, not just the graph seed's)
    interaction_matching         k = 2: exact max-weight matching of the unsigned
                                 interaction graph (2q gate count per pair)
    interaction_partition        the same unsigned graph through the same multilevel
                                 partitioner AQORN's seed uses; only edge weights differ
    interaction_partition_descent  the above, then AQORN's descent (not in the plan's table;
                                 separates the signed weights from the refinement)
    hypergraph_24                [24] stage 1: Pauli-based-computation hypergraph, km1
                                 partition (`experiments/revision/hypergraph24.py`)
    greedy_pairs                 k = 2: the paper's earlier pairwise greedy
                                 (`archive/packer/merge_packer.py`)
    aqorn_seed_only              signed graph -> partition -> lanes; no descent, no verify
    aqorn_descent_from_naive     descent from the naive seed only; no graph, no restarts
    aqorn_raw                    `plan_packing(verify=False)` at the fixed setting
    aqorn                        `plan_packing(verify=True, h_commutation="auto")`, shipped

`with_safety_net` gives any baseline the same "translate both, keep the cheaper vs naive"
check AQORN's `verify` applies, on the same criterion, so `verify` is not an unfair edge.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

Perm = List[Tuple[int, int]]

#: `random_budget` gives up after this long and keeps its best, recording that it did.
#: AQORN's descent on a 140-qubit circuit evaluates O(10^5) moves incrementally; the same
#: number of *from-scratch* random scores costs far more, and the arm is a weak baseline
#: whose result does not change with the last few thousand samples.
RANDOM_BUDGET_SECONDS = 120.0

#: Clear `PackingCost`'s membership caches this often during random sampling. Random
#: partitions almost never repeat a group, so the caches only grow.
_CACHE_FLUSH = 256


@dataclass
class PackContext:
    """Everything `_plan_under` builds before searching, for one (program, n_code, commute)."""

    program: QuantumCircuit
    n_code: int
    commute: bool
    noise: object  # packing.graph.NoiseProfile
    requested_noise: object  # what verify / the safety net are scored with (may be None)
    k: int
    view: QuantumCircuit
    profiles: dict
    cost: object  # packing.cost.PackingCost
    tightest: int
    slots: int
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return self.program.num_qubits


def make_context(
    program: QuantumCircuit, n_code: int, commute: bool, noise=None, *, spread: int = 0
) -> PackContext:
    """Mirror `packing.plan._plan_under` up to the seed: view, profiles, cost, slots."""
    from iceberg.utils import logical_capacity
    from packing.cost import DEFAULT_DEPTH_WEIGHT, PackingCost
    from packing.graph import NoiseProfile
    from packing.profiles import extract_profiles
    from packing.tables import translated_view

    requested = noise
    noise = noise or NoiseProfile()
    k = logical_capacity(n_code)
    view = translated_view(program, enable_h_commutation=commute)
    profiles = extract_profiles(view)
    cost = PackingCost(
        profiles=profiles,
        n_code=n_code,
        num_qubits=program.num_qubits,
        depth_weight=DEFAULT_DEPTH_WEIGHT,
        movement_cost=noise.movement_cost,
        idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale,
    )
    tightest = max(1, -(-program.num_qubits // k))
    return PackContext(
        program=program, n_code=n_code, commute=commute, noise=noise,
        requested_noise=requested, k=k, view=view, profiles=profiles, cost=cost,
        tightest=tightest, slots=tightest + max(0, spread),
    )


def _lanes(ctx: PackContext, groups) -> Perm:
    from packing.lanes import assign_lanes

    return [tuple(p) for p in assign_lanes([frozenset(g) for g in groups if g], ctx.cost)]


def model_value(ctx: PackContext, placement: Perm) -> float:
    """`PackingCost` of a placement's partition, groups in patch-index order."""
    groups: Dict[int, set] = {}
    for qubit, (patch, _) in enumerate(placement):
        groups.setdefault(int(patch), set()).add(qubit)
    ordered = [frozenset(groups[p]) for p in sorted(groups)]
    return float(ctx.cost.of(ordered))


# ----------------------------------------------------------------------------------------
# the packers
# ----------------------------------------------------------------------------------------


def naive(ctx: PackContext) -> Perm:
    from packing.plan import naive_packing

    return [tuple(p) for p in naive_packing(ctx.num_qubits, ctx.k)]


def aqorn(ctx: PackContext) -> Perm:
    """The shipped compiler. Decides commutation itself; callers check it matches `ctx`."""
    from packing.plan import plan_packing

    plan = plan_packing(ctx.program, ctx.n_code, h_commutation="auto", verify=True,
                        noise=ctx.requested_noise)
    ctx.notes["aqorn_commute"] = bool(plan.h_commutation)
    ctx.notes["aqorn_fell_back"] = bool(plan.fell_back)
    ctx.notes["aqorn_graph_seed_evaluations"] = (
        int(plan.descent.evaluations) if plan.descent else 0
    )
    return [tuple(p) for p in plan.placement]


def aqorn_raw(ctx: PackContext) -> Perm:
    from packing.plan import plan_packing

    plan = plan_packing(ctx.program, ctx.n_code, h_commutation=ctx.commute, verify=False,
                        noise=ctx.requested_noise)
    return [tuple(p) for p in plan.placement]


def aqorn_replicated(ctx: PackContext) -> Tuple[Perm, int]:
    """
    `_plan_under` step for step, counting every move evaluation across all seeds.

    The count is `random_budget`'s allowance. The placement must equal `aqorn_raw`'s;
    callers record whether it does, which is the check that this replication is faithful.
    """
    from packing.lanes import assign_lanes
    from packing.plan import SMALL_ENOUGH_TO_RESTART
    from packing.search import descend, naive_seed, shuffled_seed
    from packing.seed import DEFAULT_PARTITIONER, graph_seed

    cost, slots = ctx.cost, ctx.slots
    packing = graph_seed(cost, ctx.profiles, slots, noise=ctx.noise,
                         partitioner=DEFAULT_PARTITIONER)
    result = descend(packing, max_rounds=8)
    evaluations = result.evaluations
    restarts = 4 if ctx.num_qubits <= SMALL_ENOUGH_TO_RESTART else 0
    rng = random.Random(0)
    alternatives = [naive_seed(cost, slots)]
    alternatives += [shuffled_seed(cost, slots, rng) for _ in range(restarts)]
    for candidate in alternatives:
        evaluations += descend(candidate, max_rounds=8).evaluations
        if candidate.value() < packing.value() - 1e-9:
            packing = candidate
    ctx.notes["aqorn_total_evaluations"] = int(evaluations)
    ctx.notes["aqorn_seed_descent_evaluations"] = int(result.evaluations)
    return [tuple(p) for p in assign_lanes(packing.partition(), cost)], int(evaluations)


def aqorn_seed_only(ctx: PackContext) -> Perm:
    from packing.seed import DEFAULT_PARTITIONER, graph_seed

    packing = graph_seed(ctx.cost, ctx.profiles, ctx.slots, noise=ctx.noise,
                         partitioner=DEFAULT_PARTITIONER)
    return _lanes(ctx, packing.partition())


def aqorn_descent_from_naive(ctx: PackContext) -> Perm:
    from packing.search import descend, naive_seed

    packing = naive_seed(ctx.cost, ctx.slots)
    descend(packing, max_rounds=8)
    return _lanes(ctx, packing.partition())


def _random_groups(n: int, slots: int, k: int, rng: random.Random) -> List[frozenset]:
    """A uniformly shuffled balanced packing at `slots` patches (as `shuffled_seed`)."""
    order = list(range(n))
    rng.shuffle(order)
    groups: List[set] = [set() for _ in range(max(1, slots))]
    for position, qubit in enumerate(order):
        groups[position % len(groups)].add(qubit)
    for group in groups:
        while len(group) > k:
            spare = group.pop()
            for other in groups:
                if len(other) < k:
                    other.add(spare)
                    break
    return [frozenset(g) for g in groups if g]


def random_budget(ctx: PackContext, budget: int, *, seed: int = 0,
                  seconds: float = RANDOM_BUDGET_SECONDS) -> Perm:
    """Best of `budget` random balanced packings on the model; time-capped, recorded."""
    rng = random.Random(seed)
    best_value, best_groups = math.inf, None
    started = time.time()
    drawn = 0
    for drawn in range(1, max(1, budget) + 1):
        groups = _random_groups(ctx.num_qubits, ctx.slots, ctx.k, rng)
        value = ctx.cost.of(groups)
        if value < best_value:
            best_value, best_groups = value, groups
        if drawn % _CACHE_FLUSH == 0:
            ctx.cost._unary.clear()
            ctx.cost._binary.clear()
            if time.time() - started > seconds:
                break
    ctx.cost._unary.clear()
    ctx.cost._binary.clear()
    ctx.notes["random_budget_requested"] = int(budget)
    ctx.notes["random_budget_drawn"] = int(drawn)
    ctx.notes["random_budget_capped"] = bool(drawn < budget)
    return _lanes(ctx, best_groups)


def unsigned_graph(ctx: PackContext):
    """Interaction-count graph of the commuted view: weight = 2q gates between the pair."""
    from packing.graph import InteractionGraph

    weights = {pair: float(len(events)) for pair, events in ctx.cost.gates.items()}
    return InteractionGraph(weights, ctx.num_qubits)


def interaction_matching(ctx: PackContext) -> Optional[Perm]:
    """k = 2 only: the exact optimum of interaction-count pairing."""
    import networkx as nx

    if ctx.k != 2:
        return None
    graph = nx.Graph()
    graph.add_nodes_from(range(ctx.num_qubits))
    for (a, b), w in unsigned_graph(ctx).weights.items():
        graph.add_edge(a, b, weight=w)
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    groups = [set(pair) for pair in matching]
    matched = {q for pair in matching for q in pair}
    loose = [q for q in range(ctx.num_qubits) if q not in matched]
    for start in range(0, len(loose), 2):
        groups.append(set(loose[start:start + 2]))
    groups.sort(key=lambda g: min(g))
    return _lanes(ctx, groups)


def _partition_groups(ctx: PackContext, graph) -> List[set]:
    """`graph_seed`'s partition call and repair, on an arbitrary graph."""
    from packing.partition import partition as feasible_partition
    from packing.seed import DEFAULT_PARTITIONER, PARTITIONERS

    result = feasible_partition(graph, ctx.n_code, ctx.slots,
                                **PARTITIONERS[DEFAULT_PARTITIONER])
    tidy: List[set] = []
    for group in result.groups:
        members = sorted(group)
        for start in range(0, len(members), ctx.k):
            tidy.append(set(members[start:start + ctx.k]))
    placed = {q for g in tidy for q in g}
    for qubit in range(ctx.num_qubits):
        if qubit in placed:
            continue
        for group in tidy:
            if len(group) < ctx.k:
                group.add(qubit)
                break
        else:
            tidy.append({qubit})
    return [g for g in tidy if g]


def interaction_partition(ctx: PackContext) -> Perm:
    return _lanes(ctx, _partition_groups(ctx, unsigned_graph(ctx)))


def interaction_partition_descent(ctx: PackContext) -> Perm:
    from packing.search import Packing, descend

    groups = _partition_groups(ctx, unsigned_graph(ctx))
    groups = groups + [set() for _ in range(max(0, ctx.slots - len(groups)))]
    packing = Packing(ctx.cost, groups)
    descend(packing, max_rounds=8)
    return _lanes(ctx, packing.partition())


def hypergraph_24(ctx: PackContext) -> Perm:
    """
    [24] stage 1 (Sethi et al., arXiv 2603.17167): convert to Pauli-based computation, make
    one hyperedge per non-Clifford Pauli-product rotation, minimise connectivity (km1) over
    ceil(n/k) modules with eps = 0.06 -- by multi-start k-way FM, not KaHyPar, which is not
    installed. Built from `ctx.program` (the original, never the proxy). A Clifford-only
    program has no hyperedge and gets program order, flagged `hypergraph_24_fallback`.

    *Not* the interaction graph: [24] reports beating clique-expansion graph partitioners,
    so `interaction_partition` is a separate arm and must not be labelled as [24].
    """
    from experiments.revision.hypergraph24 import hypergraph24_groups

    groups, info = hypergraph24_groups(ctx.program, ctx.k)
    for key in ("fallback", "num_hyperedges", "km1_value", "km1_program_order", "seconds"):
        ctx.notes[f"hypergraph_24_{key}"] = info[key]
    return _lanes(ctx, groups)


def greedy_pairs(ctx: PackContext) -> Optional[Perm]:
    """k = 2 only: the paper's earlier pairwise greedy, unchanged from the archive."""
    if ctx.k != 2:
        return None
    from archive.packer.merge_packer import MergeMaximizingPacker

    packer = MergeMaximizingPacker(patch_capacity=2, rng=random.Random(0))
    result = packer.pack(ctx.program, num_codes=ctx.tightest)
    return [tuple(p) for p in result.permutation]


# ----------------------------------------------------------------------------------------
# fairness: the same safety net AQORN gets
# ----------------------------------------------------------------------------------------


def measured(ctx: PackContext, placement: Perm) -> float:
    """`plan.measured_cost` on the criterion AQORN's verify used. inf when rejected."""
    from packing.plan import measured_cost

    value = measured_cost(ctx.program, placement, ctx.n_code,
                          h_commutation=ctx.commute, noise=ctx.requested_noise)
    return math.inf if value is None else float(value)


def with_safety_net(ctx: PackContext, placement: Perm, naive_score: Optional[float] = None
                    ) -> Tuple[Perm, bool]:
    """Keep `placement` unless naive measures strictly cheaper. Returns (kept, fell_back)."""
    if ctx.tightest <= 1:
        return placement, False
    ours = measured(ctx, placement)
    theirs = measured(ctx, naive(ctx)) if naive_score is None else naive_score
    if ours == math.inf or theirs < ours:
        return naive(ctx), True
    return placement, False


#: Arms whose output depends on k = 2 (skipped elsewhere).
K2_ONLY = {"interaction_matching", "greedy_pairs"}

#: Every arm, in report order.
ARMS: Tuple[str, ...] = (
    "naive", "random_budget", "interaction_matching", "interaction_partition",
    "interaction_partition_descent", "hypergraph_24", "greedy_pairs", "aqorn_seed_only",
    "aqorn_descent_from_naive", "aqorn_raw", "aqorn",
)

#: Arms that get a `<arm>+safety` twin. naive and aqorn need none (naive is the floor;
#: aqorn's verify already is the net).
SAFETY_ARMS: Tuple[str, ...] = tuple(a for a in ARMS if a not in {"naive", "aqorn"})


def run_all_packers(
    program: QuantumCircuit,
    n_code: int,
    *,
    noise=None,
    arms: Sequence[str] = ARMS,
    safety: bool = True,
    random_seed: int = 0,
) -> Tuple[Dict[str, Optional[Perm]], Dict[str, object], Dict[str, float]]:
    """
    Every arm for one circuit at one code size, under the shipped compiler's commutation
    decision. Returns (placements, notes, seconds) -- placements keyed by arm, with
    `<arm>+safety` twins when `safety`; None where an arm does not apply or failed.
    """
    from packing.plan import plan_packing

    timings: Dict[str, float] = {}
    started = time.time()
    shipped = plan_packing(program, n_code, h_commutation="auto", verify=True, noise=noise)
    timings["aqorn"] = time.time() - started
    commute = bool(shipped.h_commutation)
    ctx = make_context(program, n_code, commute, noise)
    ctx.notes.update(
        commute=commute,
        aqorn_fell_back=bool(shipped.fell_back),
        aqorn_patches=int(shipped.patches),
        aqorn_graph_seed_evaluations=int(shipped.descent.evaluations) if shipped.descent else 0,
    )
    placements: Dict[str, Optional[Perm]] = {"aqorn": [tuple(p) for p in shipped.placement]}

    replicated, budget = aqorn_replicated(ctx)
    funcs: Dict[str, Callable[[], Optional[Perm]]] = {
        "naive": lambda: naive(ctx),
        "random_budget": lambda: random_budget(ctx, budget, seed=random_seed),
        "interaction_matching": lambda: interaction_matching(ctx),
        "interaction_partition": lambda: interaction_partition(ctx),
        "interaction_partition_descent": lambda: interaction_partition_descent(ctx),
        "hypergraph_24": lambda: hypergraph_24(ctx),
        "greedy_pairs": lambda: greedy_pairs(ctx),
        "aqorn_seed_only": lambda: aqorn_seed_only(ctx),
        "aqorn_descent_from_naive": lambda: aqorn_descent_from_naive(ctx),
        "aqorn_raw": lambda: aqorn_raw(ctx),
    }
    for arm in arms:
        if arm == "aqorn" or arm not in funcs:
            continue
        begun = time.time()
        try:
            placements[arm] = funcs[arm]()
        except Exception as error:  # noqa: BLE001 - one arm failing costs that arm only
            placements[arm] = None
            ctx.notes[f"{arm}_error"] = repr(error)[:300]
        timings[arm] = time.time() - begun
    if placements.get("aqorn_raw") is not None:
        ctx.notes["replication_matches_aqorn_raw"] = (
            placement_tuple(replicated) == placement_tuple(placements["aqorn_raw"])
        )

    if safety:
        naive_score = measured(ctx, naive(ctx))
        ctx.notes["naive_measured"] = naive_score
        for arm in SAFETY_ARMS:
            placement = placements.get(arm)
            if arm not in arms or placement is None:
                continue
            kept, fell = with_safety_net(ctx, placement, naive_score)
            placements[f"{arm}+safety"] = kept
            ctx.notes[f"{arm}+safety_fell_back"] = fell
    return placements, ctx.notes, timings


def placement_tuple(placement: Optional[Perm]):
    return None if placement is None else tuple((int(p), int(s)) for p, s in placement)
