"""
plan.py — choose how many patches to spend, not just how to fill them.

Everything upstream takes the patch count as given: `ceil(m/k)` patches, packed as
tightly as they will go. That is one point in a trade-off, not the answer to it.

    fewer patches   less encode/decode -- each patch pays `ENCODE + DECODE`, which is
                    `(2n+1, 3n-1)` gates and scales with n -- but more qubits sharing
                    a patch, so more contention for whole-patch macros and a bigger
                    blast radius per undetected error.
    more patches    more encode/decode and more cross-patch traffic, but patches that
                    are easier to align and cheaper to schedule.

Nothing requires a patch to be full. A qubit that interacts with nothing benefits
from being alone; a tightly-coupled trio benefits from being together even if that
leaves a slot empty. The planner searches over patch counts, places at each, scores
with the analytic model, and returns the best under whatever budget applies.

The budget
----------
`ResourceBudget` is what the machine can actually provide -- a cap on physical
qubits, on patches, or neither. The number of physical qubits a placement needs is
`patches * (n + 2)`, so a 256-atom machine running the [[12,10,2]] code can afford
18 patches, and asking for the tightest packing of a 140-qubit circuit (14 patches)
happens to fit while k=2 (70 patches, 420 atoms) does not. Being able to say that
before compiling is the point.

Noise enters through `NoiseProfile`, which prices movement and idling that gate
counts do not see. On a gate-dominated machine the planner will pack tightly; on a
movement-dominated one it spreads work out to keep interacting qubits close. Same
code, different answer, because the hardware differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from iceberg.utils import logical_capacity

from .analytic import AnalyticCost, translated_view
from .graph import NoiseProfile, build_graph
from .multilevel import PartitionResult, align_slots, partition
from .profiles import extract_profiles

__all__ = [
    "ResourceBudget",
    "PlacementPlan",
    "best_placement",
    "plan_placement",
    "sweep_patch_counts",
]

Placement = List[Tuple[int, int]]


def best_placement(
    graph,
    profiles,
    n_code: int,
    model,
    *,
    max_patches: Optional[int] = None,
    refine_rounds: int = 12,
    verify: Optional[Callable[[Placement], Optional[float]]] = None,
    fallback: Optional[Placement] = None,
    spread_to: Optional[int] = None,
) -> Tuple[Placement, PartitionResult]:
    """
    Partition for a seed, then hill-climb on the *analytic cost itself*.

    The graph is a pairwise proxy for an objective that is not pairwise. Whole-patch
    HH and the transversal macros only pay when every lane of a patch (or patch pair)
    cooperates, and no edge weight can express a conjunction -- so cut minimisation
    systematically leaves value on the table. Measured against exhaustive search over
    placements: the seed alone captured about half the available gain at k=2, and on
    `simon_n6` at k=4 it found 0% of a 40% improvement.

    Refining on the analytic cost fixes that, because the analytic cost *does* score
    the conjunction. This was not affordable before: scoring a candidate meant
    translating the circuit, ~700ms at 140 qubits. It is now ~0.1ms, so a few thousand
    evaluations are free and the proxy only has to supply a decent starting point.

    So the division of labour is: the graph gets us near, the real objective finishes.

    The model is not exact, though -- it over-counts merging by 2-3% because macro
    firing depends on scheduling it cannot predict -- and on a few circuits that is
    enough to make it prefer a placement that is genuinely *worse* than the naive
    one. `vqe_n4` came out 8% worse than sequential this way.

    So when `verify` is supplied, the winner is checked against `fallback` by exact
    measurement and the better one is returned. Two translations, negligible next to
    everything else, and it turns "usually better" into "never worse" -- which is
    what makes the packing contribution safe to present as a gain.
    """
    from .refine import interaction_graph, refine_placement

    result = partition(graph, n_code, max_patches=max_patches)
    seed = align_slots(result.groups, profiles, n_code)
    if result.num_patches < 2 and not spread_to:
        return seed, result  # one patch and no room offered: no decision exists

    refined = refine_placement(
        seed,
        model.objective(),
        n_code=n_code,
        max_rounds=refine_rounds,
        # Directed neighbourhood only once the circuit is big enough for the full one
        # to hurt; below that the restriction costs quality for no gain.
        neighbours=(
            interaction_graph(profiles) if len(seed) > 40 else None
        ),
        # Offer empty patches so `relocate` can spread. The partitioner only ever
        # merges, so without this the patch count is fixed at whatever it produced and
        # the whole patch-count decision is decorative. See `_free_slots`.
        num_patches=spread_to,
    )
    placement = refined.placement if refined.cost < float("inf") else seed
    result.refinement_moves += refined.moves

    # Refinement may have emptied or filled patches, so re-derive the grouping rather
    # than reporting the partition's own. `num_patches` and `occupancy` are read off
    # `result.groups` by callers.
    if spread_to:
        by_patch: Dict[int, List[int]] = {}
        for qubit, (patch, _) in enumerate(placement):
            by_patch.setdefault(patch, []).append(qubit)
        result.groups = [members for _, members in sorted(by_patch.items())]

    if verify is not None and fallback is not None:
        ours = verify(placement)
        theirs = verify(list(fallback))
        if ours is None or (theirs is not None and theirs < ours):
            return list(fallback), result

    return placement, result


@dataclass(frozen=True)
class ResourceBudget:
    """What the machine can provide. `None` means unconstrained."""

    max_physical_qubits: Optional[int] = None
    max_patches: Optional[int] = None

    def patch_ceiling(self, n_code: int) -> Optional[int]:
        """Most patches this budget allows at the given code size."""
        limits = []
        if self.max_patches is not None:
            limits.append(self.max_patches)
        if self.max_physical_qubits is not None:
            limits.append(self.max_physical_qubits // (n_code + 2))
        return min(limits) if limits else None

    def admits(self, patches: int, n_code: int) -> bool:
        ceiling = self.patch_ceiling(n_code)
        return ceiling is None or patches <= ceiling


@dataclass
class PlacementPlan:
    """A placement, why it was chosen, and what it costs."""

    placement: Placement
    n_code: int
    cost: float
    partition: PartitionResult
    physical_qubits: int
    considered: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def num_patches(self) -> int:
        return self.partition.num_patches

    @property
    def occupancy(self) -> List[int]:
        return sorted((len(g) for g in self.partition.groups), reverse=True)

    def summary(self) -> str:
        tried = ", ".join(f"{p}p:{c:.0f}" for p, c in self.considered)
        return (
            f"{self.num_patches} patches (sizes {self.occupancy}), "
            f"{self.physical_qubits} physical qubits, cost {self.cost:.0f}"
            f"  [tried {tried}]"
        )


def plan_placement(
    program: QuantumCircuit,
    n_code: int,
    *,
    budget: Optional[ResourceBudget] = None,
    noise: Optional[NoiseProfile] = None,
    depth_weight: float = 0.0,
    candidates: Optional[Sequence[int]] = None,
) -> PlacementPlan:
    """
    Place `program`, choosing the patch count as part of the decision.

    Tries the tightest packing and a few looser ones, scores each with the analytic
    model, and keeps the cheapest that the budget admits. Cheap enough to do
    exhaustively over patch counts because scoring a placement no longer requires
    translating the circuit.

    Spreading
    ---------
    `max_patches` reaches `multilevel.partition` as a *cap*, and the partitioner only
    ever merges. Whether a looser cap changes anything therefore depends on the circuit:
    when the coarse graph splits cleanly `_greedy_partition` does leave a looser grouping,
    and when greedy packs tight nothing downstream could take the offer up. On QASMBench
    small circuits it is the second case -- `qpe_n9` returns 3 patches whether asked for
    3, 4, 6 or 9 -- while a CX-dense 4-qubit program does spread once crossings are priced
    at `depth_weight >= 12`.

    Two changes here, both aimed at the second case. `spread_to` widens `_free_slots` to
    cover patches `0 .. target-1` whether occupied or not, so `refine`'s `relocate` move
    can migrate a qubit into an *empty* patch -- previously invisible, since `_free_slots`
    derived its patches from the occupied set. And each candidate count is now refined
    rather than ranked on its unrefined partition, since refinement is where spreading
    happens; the extra candidates get half the round budget, because the tightest usually
    wins.

    It usually does win, and the arithmetic says why. At n=6 an extra patch costs
    `ENCODE + DECODE` = `(13, 17)` gates, 234 weighted units, while a CNOT moved from
    intra- to cross-patch saves one layer -- 3 units at the default `depth_weight`. So
    spreading needs on the order of 78 crossing CNOTs to break even, and on a
    gate-counting machine it correctly almost never fires. That is the pricing working
    rather than failing: an extra patch is genuinely expensive, and movement and idle are
    priced precisely so they can tip the balance on hardware where they dominate.

    Measured honestly: on every circuit tried so far the chosen count is the same as
    before this change. What is new is that the mechanism *exists* and the scan compares
    refined placements rather than five copies of one partition.
    """
    budget = budget or ResourceBudget()
    noise = noise or NoiseProfile()
    k = logical_capacity(n_code)
    num_qubits = program.num_qubits

    profiles = extract_profiles(translated_view(program))
    graph = build_graph(profiles, n_code, noise=noise, num_qubits=num_qubits)
    model = AnalyticCost(
        profiles=profiles,
        n_code=n_code,
        movement_cost=noise.movement_cost,
        idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale,
        depth_weight=depth_weight,
    )

    tightest = max(1, -(-num_qubits // k))
    options = list(candidates) if candidates is not None else _default_candidates(
        tightest, num_qubits
    )
    ceiling = budget.patch_ceiling(n_code)
    options = [p for p in options if p >= tightest and (ceiling is None or p <= ceiling)]
    if not options:
        # The budget cannot even hold the tightest packing; return it anyway and let
        # the caller see `physical_qubits` exceed the cap rather than silently fail.
        options = [tightest]

    # Each candidate count has to be *refined*, not just partitioned. The partitioner
    # only ever merges, so a looser budget changes nothing on its own -- spreading
    # happens when `relocate` migrates a qubit into an empty patch, which only the
    # refinement pass does. The previous two-pass shortcut ranked candidates on the
    # partition alone and so ranked five identical placements.
    #
    # Affordable because scoring is analytic. Rounds taper with width, and the extra
    # candidates get a shorter budget than the tightest, which is the one most likely to
    # win: on a gate-counting machine every looser option pays encode/decode for nothing.
    rounds = _rounds_for(num_qubits)
    considered: List[Tuple[int, float]] = []
    best: Optional[Tuple[float, Placement, PartitionResult]] = None
    for index, target in enumerate(options):
        placement, result = best_placement(
            graph, profiles, n_code, model,
            # Both mechanisms, because they reach different placements. The cap lets
            # `_greedy_partition` leave a looser grouping in the first place, which it
            # does whenever the coarse graph splits cleanly; `spread_to` lets refinement
            # migrate into empty patches afterwards, which is the only route when greedy
            # packs tight. Pinning this to `tightest` and relying on refinement alone
            # lost the CX-dense case at `depth_weight=12` that the cap already found.
            max_patches=target,
            refine_rounds=rounds if index == 0 else max(1, rounds // 2),
            spread_to=target,
        )
        cost = model.cost(placement)
        considered.append((result.num_patches, cost))
        if best is None or cost < best[0]:
            best = (cost, placement, result)

    _, placement, result = best
    return PlacementPlan(
        placement=placement,
        n_code=n_code,
        cost=model.cost(placement),
        partition=result,
        physical_qubits=result.num_patches * (n_code + 2),
        considered=considered,
    )


def _rounds_for(num_qubits: int) -> int:
    """
    Refinement rounds, tapered with circuit width.

    Each round is a full neighbourhood sweep, so its cost grows with the circuit
    while its marginal gain falls -- the first two or three rounds take almost all of
    the improvement. Tapering keeps a 140-qubit placement in seconds rather than
    minutes without measurably changing the result.
    """
    if num_qubits <= 32:
        return 12
    if num_qubits <= 80:
        return 6
    return 3


def _default_candidates(tightest: int, num_qubits: int) -> List[int]:
    """
    Patch counts worth trying: the tightest, then progressively looser.

    Geometric rather than exhaustive -- the cost curve in patch count is smooth
    (encode/decode rises linearly, cut weight falls), so a handful of samples finds
    the region and the partitioner's own refinement handles the rest.
    """
    options = {tightest}
    step = max(1, tightest // 4)
    for multiple in (1, 2, 3):
        options.add(min(num_qubits, tightest + multiple * step))
    options.add(num_qubits)  # every qubit solo, the sparse extreme
    return sorted(options)


def sweep_patch_counts(
    program: QuantumCircuit,
    n_code: int,
    *,
    noise: Optional[NoiseProfile] = None,
    depth_weight: float = 0.0,
) -> List[Tuple[int, int, float]]:
    """
    Cost at every admissible patch count, as (patches, physical qubits, cost).

    For plotting the resource trade-off directly: how much does buying another patch
    actually buy? Cheap enough to run per benchmark now that scoring is analytic.
    """
    noise = noise or NoiseProfile()
    k = logical_capacity(n_code)
    profiles = extract_profiles(translated_view(program))
    graph = build_graph(profiles, n_code, noise=noise, num_qubits=program.num_qubits)
    model = AnalyticCost(
        profiles=profiles,
        n_code=n_code,
        movement_cost=noise.movement_cost,
        idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale,
        depth_weight=depth_weight,
    )

    tightest = max(1, -(-program.num_qubits // k))
    seen: Dict[int, float] = {}
    for target in range(tightest, program.num_qubits + 1):
        placement, result = best_placement(
            graph, profiles, n_code, model, max_patches=target
        )
        if result.num_patches in seen:
            continue
        seen[result.num_patches] = model.cost(placement)
    return [
        (patches, patches * (n_code + 2), cost) for patches, cost in sorted(seen.items())
    ]
