"""
search.py — descend on the real objective, from a seed that knows the whole graph.

The design here is the result of measuring an assumption and finding it wrong, so it is
worth stating what was tried.

The old stack had three objectives: signed edge weights drove the seed, cut weight drove
the partitioner, and the analytic cost drove a final local search. Since the analytic
cost is now cheap to evaluate incrementally, the obvious move is to use it everywhere and
delete the proxies. That was tried, and it does not work:

    plain descent on the true objective, from any cheap seed, leaves the naive partition
    untouched on `knn_n31` and `swap_test_n25` at k=10 -- zero improving single moves --
    while a partition 13% cheaper *by that same objective* exists

    Kernighan-Lin rescues some of those (`knn_n31` 0% -> -13.4%, `dnn_n16` +7.9% ->
    -26.4%) and not others (`ising_n42`, `swap_test_n25`: no change)

    coarsening into atoms so that one move relocates half a patch was tried too, scoring
    every merge on the true objective. Better on some circuits, worse overall, and 2 s
    at m=67

    seeding from the signed-graph multilevel partition and then descending on the true
    objective matches or beats the incumbent nearly everywhere, in a third of the time

The lesson is that a smooth global proxy is a *better search guide* than the exact
objective, even though it is a worse description of cost. Cut weight is a pairwise
relaxation: it is defined on every pair at once, so minimising it moves the whole
partition coherently. The true objective is conjunctive -- transversal macros need every
lane of two patches, HH needs every lane of one -- which makes its landscape full of
plateaus and deep local optima that a move-at-a-time search cannot cross.

So the graph stays, in the one role it is good at, and the objective it was standing in
for takes over everything downstream of the seed:

    seed      `grouping.multilevel.partition` on the signed graph -- global structure
    descend   sweep qubits: move / spread / swap, best-improvement, true objective

What is gone, and stays gone
---------------------------
*the patch-count scan* -- five candidates, of which 41 of 63 QASMBench circuits returned
    one distinct count, and the tightest packing won all 63. `spread` moves raise the
    count when it pays and the objective charges per live patch, so it is not a separate
    decision.

*the separate analytic local search* -- `refine.refine_placement` was a fourth stage
    after the partitioner, with a move set two thirds of which could not change the cost.
    `descend` replaces it. The FM passes and `_consolidate` *inside*
    `multilevel.partition` are untouched and still run; they optimise the cut, and this
    descent re-finds their gains on the true objective, so stubbing both out moves the
    final cost on only 1 of 9 hard circuits (`adder_n64` at k=6, +1.94%).

*lane search* -- `reorder`, and same-patch `swap`. The objective is a function of the
    partition (see `cost`), so both are provably zero-gain; they were 72 of 109 candidate
    moves on `qpe_n9`, scored on every round.

*Kernighan-Lin* -- it was here, and it went, which is worth recording because it is the
    textbook answer to the local optima above. Once the graph seeds the search, KL
    changed the outcome on 1 of 8 hard circuits with a patience cutoff and 2 of 87
    without, at 3-6x and 15x the cost of the descent respectively. And every circuit it
    helped was one where the model mis-ranks -- so `verify` was already returning the
    naive packing there, and with the safety net on KL changed nothing at all. A good
    seed removed the need for the escape hatch.

Deltas, not rescores
--------------------
A move changes one or two patches, so it changes those patches' unary terms and the
binary terms incident to them, and nothing else. `Packing._delta` computes exactly that
set, which is what makes thousands of candidates per sweep affordable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .cost import PackingCost, Term

__all__ = [
    "Packing",
    "Move",
    "DescentResult",
    "descend",
    "naive_seed",
    "program_order",
    "shuffled_seed",
]

Group = FrozenSet[int]
_EMPTY_TERM = Term()


@dataclass(frozen=True)
class Move:
    """A candidate change, as the new membership of each patch it touches."""

    kind: str
    changes: Tuple[Tuple[int, Group], ...]

    def as_dict(self) -> Dict[int, Group]:
        return dict(self.changes)


class Packing:
    """
    A partition of the logical qubits into patch slots, scored incrementally.

    Slots may be empty; `live` counts the occupied ones and the objective charges only
    those, so spare slots cost nothing and are what let the search spread a program out
    when movement or idle noise makes that worth doing.
    """

    def __init__(self, cost: PackingCost, groups: Sequence[Iterable[int]]) -> None:
        self.cost = cost
        self.groups: List[Group] = [frozenset(g) for g in groups]
        self.patch_of: Dict[int, int] = {
            q: index for index, group in enumerate(self.groups) for q in group
        }
        self.live = sum(1 for g in self.groups if g)
        self._total = _EMPTY_TERM
        for group in self.groups:
            if group:
                self._total = self._total + cost.unary(group)
        for index, left in enumerate(self.groups):
            if not left:
                continue
            for right in self.groups[index + 1:]:
                if right:
                    self._total = self._total + cost.binary(left, right)

    # ------------------------------------------------------------------
    # value and deltas
    # ------------------------------------------------------------------

    def value(self) -> float:
        # `busiest` comes off the current groups rather than out of `_total`: the
        # accumulator is keyed by group membership and a delta may rescope without
        # every group key changing, which would leave a load stranded.
        return self.cost.price(
            self.live, self._total, self.cost.busiest_patch_depth(self.groups),
            self.cost.movement_hops(self.groups),
        )

    def _neighbour_patches(self, members: Iterable[int]) -> Set[int]:
        out: Set[int] = set()
        partners = self.cost.partners
        for qubit in members:
            for partner in partners.get(qubit, ()):
                index = self.patch_of.get(partner)
                if index is not None:
                    out.add(index)
        return out

    def _delta(self, changes: Dict[int, Group]) -> Tuple[Term, int]:
        """
        Exact change in accumulated terms and live-patch count.

        `others` is the set of untouched patches whose `binary` term with a touched patch
        can move, and which patches those are depends on whether any patch changed *size*:

        size preserved (a swap)
            only the qubits that actually moved can change a crossing, so `others` is
            their gate-neighbourhood. This is the common case -- at the tightest packing
            every patch is full, so swaps are most of the neighbourhood -- and scoping it
            this tightly is what makes the sweep affordable.
        size changed (a move or a spread)
            `_transversal` requires two patches to hold equally many qubits, so a resize
            can switch a macro on or off against a patch this qubit has no gate with at
            all. Fall back to the touched patches' full gate-neighbourhood.

        Overshooting is safe either way: an untouched patch contributes the same term
        before and after, and the two cancel.
        """
        touched = sorted(changes)
        resized = any(len(changes[i]) != len(self.groups[i]) for i in touched)
        if resized:
            others: Set[int] = set()
            for index in touched:
                others |= self._neighbour_patches(self.groups[index])
                others |= self._neighbour_patches(changes[index])
        else:
            moved: Set[int] = set()
            for index in touched:
                moved |= self.groups[index] ^ changes[index]
            others = self._neighbour_patches(moved)
        others -= set(touched)

        removed = added = _EMPTY_TERM
        unary, binary = self.cost.unary, self.cost.binary
        for index in touched:
            removed = removed + unary(self.groups[index])
            added = added + unary(changes[index])
        for position, left in enumerate(touched):
            for right in touched[position + 1:]:
                removed = removed + binary(self.groups[left], self.groups[right])
                added = added + binary(changes[left], changes[right])
        for index in touched:
            for other in others:
                resident = self.groups[other]
                removed = removed + binary(self.groups[index], resident)
                added = added + binary(changes[index], resident)

        live_delta = sum(1 for i in touched if changes[i]) - sum(
            1 for i in touched if self.groups[i]
        )
        return added - removed, live_delta

    def value_after(self, changes: Dict[int, Group]) -> float:
        """
        Cost this partition would have after `changes`, without applying them.

        The busiest-patch load has to be evaluated on the *hypothetical* groups, not the
        current ones. Scoring a candidate move on the old load is how the search ends up
        choosing by one objective and being judged by another -- the exact mismatch the
        old stack had between `plan_placement` and the figure path.
        """
        delta, live_delta = self._delta(changes)
        hypothetical = list(self.groups)
        for index, group in changes.items():
            hypothetical[index] = group
        return self.cost.price(
            self.live + live_delta,
            self._total + delta,
            self.cost.busiest_patch_depth(hypothetical),
            self.cost.movement_hops(hypothetical),
        )

    def apply(self, changes: Dict[int, Group]) -> None:
        delta, live_delta = self._delta(changes)
        self._total = self._total + delta
        self.live += live_delta
        for index, group in changes.items():
            for qubit in self.groups[index]:
                if self.patch_of.get(qubit) == index:
                    del self.patch_of[qubit]
            self.groups[index] = group
        for index in changes:
            for qubit in self.groups[index]:
                self.patch_of[qubit] = index

    def partition(self) -> List[Group]:
        return [g for g in self.groups if g]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        sizes = sorted((len(g) for g in self.groups if g), reverse=True)
        return f"Packing(live={self.live}, sizes={sizes}, cost={self.value():.0f})"


# ----------------------------------------------------------------------
# moves
# ----------------------------------------------------------------------


def moves_for(packing: Packing, qubit: int, *, fixed_count: bool = False) -> Iterable[Move]:
    """
    Every admissible way to change where one qubit sits.

    Three shapes, all changes of *membership* -- there is deliberately no lane move,
    because the objective cannot read lanes:

        move    q joins a patch that has room
        spread  q leaves for an empty slot, raising the patch count
        swap    q and r exchange patches, the only way to move anything once every patch
                is full, which is exactly what the tightest packing is

    Empty slots are interchangeable, so only the first is offered, and a lone qubit is
    offered none -- that yields the same partition with two labels exchanged.

    `fixed_count` offers only the moves that keep the number of live patches: every swap,
    and a move only when q's patch keeps another qubit. The objective charges per live
    patch, so without this a descent from a spread-out seed collapses it to the tightest
    packing (96 of 105 density-sweep cases above the tightest count).
    """
    k = packing.cost.k
    groups = packing.groups
    home = packing.patch_of.get(qubit)
    if home is None:
        return
    here = groups[home]
    without = here - {qubit}
    single = frozenset({qubit})
    first_empty = next((i for i, g in enumerate(groups) if not g), None)

    for target, resident in enumerate(groups):
        if target == home:
            continue
        if not resident:
            if target == first_empty and without and not fixed_count:
                yield Move("spread", ((home, without), (target, single)))
            continue
        if len(resident) < k and (without or not fixed_count):
            yield Move("move", ((home, without), (target, resident | single)))
        for other in resident:
            yield Move(
                "swap",
                (
                    (home, without | {other}),
                    (target, (resident - {other}) | single),
                ),
            )


def _candidates(packing: Packing) -> Iterable[Move]:
    """The whole neighbourhood, for inspection and tests."""
    for qubit in sorted(packing.patch_of):
        yield from moves_for(packing, qubit)


# ----------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------


def program_order(num_qubits: int, k: int, patches: int) -> List[List[int]]:
    """
    The qubits in program order, in exactly `patches` patches of at most `k`.

    Full patches first, then one qubit apiece: each patch takes as many as it can while
    leaving one for every patch after it. At the tightest count that is the plain
    `k`-at-a-time fill; at k = 2 it pairs the first `num_qubits - patches` consecutive pairs
    and leaves the rest solo, the density sweep's trivial packing.
    """
    lowest = max(1, -(-num_qubits // k))
    if not lowest <= patches <= max(1, num_qubits):
        raise ValueError(
            f"{num_qubits} qubits in patches of {k} need between {lowest} and "
            f"{num_qubits} patches, not {patches}"
        )
    groups: List[List[int]] = []
    qubit = 0
    for index in range(patches):
        size = min(k, num_qubits - qubit - (patches - index - 1))
        groups.append(list(range(qubit, qubit + size)))
        qubit += size
    return groups


def naive_seed(cost: PackingCost, slots: int, *, patches: Optional[int] = None) -> Packing:
    """
    Program order, tightest first. The fallback when no graph seed is supplied.

    With `patches`, program order into exactly that many live patches (`program_order`),
    in `max(slots, patches)` slots.
    """
    if patches is not None:
        groups = [set(g) for g in program_order(cost.num_qubits, cost.k, patches)]
        groups += [set() for _ in range(max(0, slots - len(groups)))]
        return Packing(cost, groups)
    groups: List[Set[int]] = [set() for _ in range(max(1, slots))]
    for qubit in range(cost.num_qubits):
        groups[min(qubit // cost.k, len(groups) - 1)].add(qubit)
    return Packing(cost, groups)


def shuffled_seed(cost: PackingCost, slots: int, rng: random.Random) -> Packing:
    """
    A random balanced packing, for restarts.

    Only worth running where a descent is cheap. On small circuits the basins are few
    and small, so which one the graph seed lands in decides the answer -- restarting
    recovered `sat_n7` from 1.6% better than naive to 6.7%, at 4 ms. On large circuits
    restarts changed nothing measurable and cost 2-3x, so `plan_packing` gates them on
    size rather than offering them everywhere.
    """
    order = list(range(cost.num_qubits))
    rng.shuffle(order)
    groups: List[Set[int]] = [set() for _ in range(max(1, slots))]
    for position, qubit in enumerate(order):
        groups[position % len(groups)].add(qubit)
    for group in groups:
        while len(group) > cost.k:
            spare = group.pop()
            for other in groups:
                if len(other) < cost.k:
                    other.add(spare)
                    break
    return Packing(cost, groups)


# ----------------------------------------------------------------------
# descent
# ----------------------------------------------------------------------


@dataclass
class DescentResult:
    packing: Packing
    rounds: int = 0
    accepted: List[Tuple[str, float]] = field(default_factory=list)
    evaluations: int = 0
    seed_cost: float = 0.0

    @property
    def moves(self) -> int:
        return len(self.accepted)

    @property
    def gain(self) -> float:
        """How much the descent improved on the seed, in weighted units."""
        return self.seed_cost - self.packing.value()

    def summary(self) -> str:
        kinds: Dict[str, int] = {}
        for kind, _ in self.accepted:
            kinds[kind] = kinds.get(kind, 0) + 1
        detail = ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items())) or "none"
        return (
            f"cost={self.packing.value():.0f} (seed {self.seed_cost:.0f}, "
            f"gain {self.gain:.0f}) after {self.moves} moves ({detail}) "
            f"in {self.evaluations} evaluations, {self.rounds} rounds"
        )


def descend(
    packing: Packing,
    *,
    max_rounds: int = 8,
    tolerance: float = 1e-9,
    result: Optional[DescentResult] = None,
    fixed_count: bool = False,
) -> DescentResult:
    """
    Sweep the qubits, moving each to its best home, until a sweep changes nothing.

    `fixed_count` descends over the count-preserving moves only (`moves_for`), so the
    live-patch count of the seed is the count of the result.

    Per-qubit best-improvement rather than one globally-steepest move per round. Both
    reach a local optimum, but a global-steepest round costs a full O(m^2) neighbourhood
    scan and buys a single move, so it needs as many rounds as it makes moves: at m=64
    that was still improving when it hit a 12-round cap, which is quality left on the
    table for a lot of work. A sweep applies many moves per scan and converges in a
    handful of rounds. Same per-node scheme as `multilevel._refine`, scored on the real
    objective instead of the cut.

    Terminates because every accepted move strictly lowers a bounded objective.
    """
    result = result or DescentResult(packing=packing, seed_cost=packing.value())
    for _ in range(max_rounds):
        result.rounds += 1
        improved = False
        for qubit in sorted(packing.patch_of):
            current = packing.value()
            best: Optional[Tuple[float, Move]] = None
            for move in moves_for(packing, qubit, fixed_count=fixed_count):
                result.evaluations += 1
                value = packing.value_after(move.as_dict())
                if value < current - tolerance and (best is None or value < best[0]):
                    best = (value, move)
            if best is not None:
                packing.apply(best[1].as_dict())
                result.accepted.append((best[1].kind, best[0]))
                improved = True
        if not improved:
            break
    return result
