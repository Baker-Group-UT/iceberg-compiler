"""
refine.py — local search over placements, seeded from a grouping.

The clustering in `agglomerative.py` scores one group at a time, and that turns out
to be the formulation's ceiling: the merging pass's largest wins — the transversal
CXT/CZT/XCXT macros — are properties of a *pair* of patches and of the slot order
within them, neither of which is visible when scoring a single group in isolation.
See the report for the measurements.

This module attacks the same objective from the other end. It hill-climbs on the
placement directly, using moves that can express what clustering cannot:

    swap        exchange the slots of two qubits (changes grouping *and* slot order)
    relocate    move one qubit to a free slot (changes only grouping)
    reorder     permute slots inside one patch (changes only slot order)

Scored against whatever `objective` is supplied. With the measured objective —
translate and count — this is a true local optimum of the real cost, at the price of
one translation per candidate. With the cost model it is fast but inherits the
model's blind spots, which makes the pair worth comparing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from qiskit import QuantumCircuit

from iceberg.utils import logical_capacity

__all__ = [
    "RefinementResult",
    "refine_placement",
    "measured_objective",
    "model_objective",
]

Placement = List[Tuple[int, int]]
Objective = Callable[[Placement], float]


@dataclass
class RefinementResult:
    placement: Placement
    cost: float
    evaluations: int = 0
    accepted: List[Tuple[str, float]] = field(default_factory=list)
    #: True when a budget cut a round short, so the result is not a local optimum.
    truncated: bool = False

    @property
    def moves(self) -> int:
        return len(self.accepted)

    def summary(self) -> str:
        kinds: Dict[str, int] = {}
        for kind, _ in self.accepted:
            kinds[kind] = kinds.get(kind, 0) + 1
        detail = ", ".join(f"{kind}x{count}" for kind, count in sorted(kinds.items()))
        return (
            f"cost={self.cost:.0f} after {self.moves} moves "
            f"({detail or 'none'}) in {self.evaluations} evaluations"
            + (" [truncated]" if self.truncated else "")
        )


def measured_objective(
    program: QuantumCircuit,
    n_code: int,
    *,
    fault_tolerant: bool = False,
    rejected_penalty: float = float("inf"),
) -> Objective:
    """
    An objective that translates the circuit and counts gates.

    Rejected placements score `rejected_penalty`, so the search will step away from
    them but will not crash on them.
    """
    from .evaluate import measure

    cache: Dict[Tuple[Tuple[int, int], ...], float] = {}

    def objective(placement: Placement) -> float:
        key = tuple(placement)
        if key in cache:
            return cache[key]
        counts = measure(program, placement, n_code, fault_tolerant=fault_tolerant)
        value = rejected_penalty if counts is None else counts[2]
        cache[key] = value
        return value

    return objective


def model_objective(
    profiles,
    n_code: int,
    *,
    window: int = 2,
    transversal_confidence: Optional[float] = None,
) -> Objective:
    """
    An objective that predicts cost from `GroupCostModel` instead of translating.

    Roughly three orders of magnitude cheaper per candidate than
    `measured_objective`, which is what would make local search usable on circuits
    too large to translate thousands of times. Whether the prediction is faithful
    enough to find the same optimum is an empirical question — `evaluate.py` answers
    it by measuring where this objective's local optimum actually lands.
    """
    from .cost_model import GroupCostModel

    kwargs = {"window": window}
    if transversal_confidence is not None:
        kwargs["transversal_confidence"] = transversal_confidence
    model = GroupCostModel(profiles=profiles, n_code=n_code, **kwargs)

    def objective(placement: Placement) -> float:
        by_patch: Dict[int, List[int]] = {}
        for qubit, (patch, _) in enumerate(placement):
            by_patch.setdefault(patch, []).append(qubit)
        return sum(model.cost(group) for group in by_patch.values())

    return objective


def _free_slots(
    placement: Placement, k: int, num_patches: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Every slot a `relocate` move may target.

    Without `num_patches` the patches are exactly the occupied ones, so an *empty* patch
    is invisible and the search can never spread a program over more patches than the
    partition gave it. That is why `plan_placement`'s patch-count scan had no effect:
    `multilevel.partition` treats its budget as a cap and only ever merges, so nothing
    downstream could take a looser budget up on the offer.

    Passing `num_patches` offers the slots of patches `0 .. num_patches-1` whether or not
    they currently hold anything, which is what makes spreading reachable. The cost model
    then decides: an empty patch costs a full encode/decode, so it has to pay for itself
    in reduced cross-patch traffic, movement distance or idle volume.
    """
    used = set(placement)
    occupied = {patch for patch, _ in placement}
    highest = max(occupied) + 1 if occupied else 0
    patches = range(max(highest, num_patches or 0))
    return [
        (patch, slot)
        for patch in patches
        for slot in range(k)
        if (patch, slot) not in used
    ]


#: Largest patch occupancy for which `reorder` enumerates full permutations.
#: Beyond this the factorial dominates the whole candidate set -- at k=10 a single
#: patch offers 10! - 1 reorderings, so a bounded search would spend its entire
#: budget permuting patch 0 and never look at patch 1. Slot *transpositions* are
#: still reachable at any k, because a `swap` of two qubits in the same patch is
#: exactly that; capping here only drops the 3-cycles and longer.
MAX_REORDER_SLOTS = 4


def interaction_graph(profiles) -> Dict[int, Set[int]]:
    """qubit -> the qubits it shares a two-qubit gate with."""
    graph: Dict[int, Set[int]] = {index: set() for index in profiles}
    for index, profile in profiles.items():
        for event in profile.two_qubit_events():
            graph[index].add(event.partner)
            graph.setdefault(event.partner, set()).add(index)
    return graph


def _swap_partners(
    placement: Placement, neighbours: Optional[Dict[int, Set[int]]]
) -> List[Tuple[int, int]]:
    """
    Which pairs of qubits are worth trying to exchange.

    Without an interaction graph this is every pair, which is O(m^2) and becomes the
    whole cost of the search at scale: at 140 qubits it is 9,730 swaps per round, and
    with a translate-per-candidate objective that is hours for one placement.

    With one, a swap is only proposed when it moves a qubit *toward a patch holding
    one of its interaction partners*. That is the standard boundary restriction from
    graph partitioning, and it matches what the objective rewards -- cross-patch
    two-qubit gates are what drive physical cost, so a move that changes no qubit's
    co-location with its partners has nothing to gain except through slot order,
    which `reorder` covers separately. On a sparse interaction graph the count drops
    from O(m^2) to roughly O(m . d . k).

    It is a restriction, not a reformulation: a swap that would help only by an
    indirect rearrangement is no longer reachable in one move. Multi-round descent
    recovers many of those, but the search is a heuristic either way.
    """
    count = len(placement)
    if neighbours is None:
        return list(combinations(range(count), 2))

    occupants = {position: index for index, position in enumerate(placement)}
    patches_of: Dict[int, Set[int]] = {}
    for index, (patch, _) in enumerate(placement):
        patches_of.setdefault(index, set()).add(patch)

    pairs: Set[Tuple[int, int]] = set()
    for index in range(count):
        wanted = {
            placement[partner][0]
            for partner in neighbours.get(index, ())
            if partner < count
        }
        for position, other in occupants.items():
            if other != index and position[0] in wanted:
                pairs.add((min(index, other), max(index, other)))
    return sorted(pairs)


def _candidates(
    placement: Placement,
    k: int,
    max_reorder_slots: int = MAX_REORDER_SLOTS,
    neighbours: Optional[Dict[int, Set[int]]] = None,
    num_patches: Optional[int] = None,
):
    """Yield (kind, neighbour) for every single move from `placement`."""
    count = len(placement)

    # swap: exchange two qubits' slots
    for left, right in _swap_partners(placement, neighbours):
        neighbour = list(placement)
        neighbour[left], neighbour[right] = neighbour[right], neighbour[left]
        yield "swap", neighbour

    # relocate: move a qubit into a free slot
    for slot in _free_slots(placement, k, num_patches):
        for index in range(count):
            if placement[index] == slot:
                continue
            neighbour = list(placement)
            neighbour[index] = slot
            yield "relocate", neighbour

    # reorder: permute the occupied slots of one patch
    by_patch: Dict[int, List[int]] = {}
    for index, (patch, _) in enumerate(placement):
        by_patch.setdefault(patch, []).append(index)
    for patch, indices in by_patch.items():
        if not 2 <= len(indices) <= max_reorder_slots:
            continue
        current = [placement[index][1] for index in indices]
        for order in permutations(current):
            if list(order) == current:
                continue
            neighbour = list(placement)
            for index, slot in zip(indices, order):
                neighbour[index] = (patch, slot)
            yield "reorder", neighbour


def refine_placement(
    placement: Sequence[Tuple[int, int]],
    objective: Objective,
    *,
    n_code: int = 4,
    max_rounds: int = 40,
    max_evaluations: Optional[int] = None,
    max_reorder_slots: int = MAX_REORDER_SLOTS,
    neighbours: Optional[Dict[int, Set[int]]] = None,
    time_budget: Optional[float] = None,
    num_patches: Optional[int] = None,
) -> RefinementResult:
    """
    Steepest-descent local search from `placement`.

    Each round evaluates every single move and commits the best strict improvement,
    stopping at a local optimum. Steepest rather than first-improvement because the
    objective is expensive enough that the extra evaluations buy a better path.

    Three ways to bound it, in descending order of preference:

    `neighbours`   an interaction graph, which *narrows* the neighbourhood instead of
                   truncating it -- see `_swap_partners`. Costs nothing in bias and
                   turns the O(m^2) swap set into roughly O(m . d . k). Pass this
                   first; the other two are backstops.
    `max_rounds`   caps the number of complete sweeps, so every round stays unbiased.
    `max_evaluations` / `time_budget`
                   cut a round short, which biases the search toward whichever moves
                   `_candidates` happens to yield first. `truncated` records that it
                   happened so a caller can report it rather than pass off a
                   half-searched placement as a local optimum.

    `num_patches` widens the `relocate` neighbourhood to include *empty* patches up to
    that count, which is the only way the search can spread a program over more patches
    than it was given. See `_free_slots`. The objective decides whether spreading pays.
    """
    k = logical_capacity(n_code)
    current = list(placement)
    best_cost = objective(current)
    result = RefinementResult(placement=current, cost=best_cost, evaluations=1)
    deadline = None if time_budget is None else time.monotonic() + time_budget

    def out_of_budget() -> bool:
        if max_evaluations is not None and result.evaluations >= max_evaluations:
            return True
        return deadline is not None and time.monotonic() >= deadline

    for _ in range(max_rounds):
        best_move: Optional[Tuple[str, Placement, float]] = None
        for kind, neighbour in _candidates(
            current, k, max_reorder_slots, neighbours, num_patches
        ):
            if out_of_budget():
                result.truncated = True
                break
            value = objective(neighbour)
            result.evaluations += 1
            if value < best_cost and (best_move is None or value < best_move[2]):
                best_move = (kind, neighbour, value)
        if best_move is None:
            break
        kind, neighbour, value = best_move
        current = neighbour
        best_cost = value
        result.accepted.append((kind, value))
        if out_of_budget():
            result.truncated = True
            break

    result.placement = current
    result.cost = best_cost
    return result
