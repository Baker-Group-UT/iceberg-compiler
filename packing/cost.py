"""
cost.py — the placement objective, as an incremental function of the partition.

`grouping.analytic.AnalyticCost` scores a *placement*: a (patch, slot) pair per
qubit. It does not need to. Slot order enters none of its terms -- the transversal
macro fires on a lane *bijection*, and while realigning lanes is no longer free
(a routing swap is an atom exchange, two carriages of `move_error`), the charge is
priced in *expectation over alignments* inside `_transversal`, which keeps every
slot permutation of a group scoring identically. So all 24 slot permutations of a
4-lane patch still collapse to one state, and the search explores a space k! times
smaller than placements.

This module drops slots. The state is a *partition* of the logical qubits into
patches, and the cost is

    cost(partition) = solo                                      constant
                    + P * (ENCODE + DECODE)                     per patch
                    + sum over patches      unary(A)
                    + sum over patch pairs  binary(A, B)        only where they interact
                    + movement(crossings, P) + idle(depth, P)   global

`unary` and `binary` are cached by membership, and every term a move does not touch
is reused, so a candidate move costs the few terms it actually changes rather than a
full rescore. That is what lets the search use the real objective everywhere instead
of a pairwise graph proxy.

Agreement with `AnalyticCost` is a test, not a hope: `tests/test_packing.py`
asserts equality on random partitions across code sizes, so this stays a faster
spelling of the same number rather than a second opinion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from qiskit import QuantumCircuit

# These four are pure, tested, and shared with `grouping.analytic`, which is the
# model this module has to agree with numerically. Re-deriving them here would be a
# second implementation of the same combinatorics and a way for the two to drift.
from .tables import (
    CROSS_PATCH_COST,
    DEFAULT_WINDOW,
    GATE_KEYS,
    _coincident_layers,
    _complete_bijections,
    _fusable_group_sizes,
    gate_depths,
    mean_patch_hops,
    patch_hop_matrix,
    translated_view,
)
from .profiles import QubitProfile, extract_profiles
from iceberg.costs import DEFAULT_WORK_WEIGHTS, carriage_cost, work_table
from iceberg.utils import logical_capacity

__all__ = ["Term", "PackingCost", "DEFAULT_DEPTH_WEIGHT", "PARALLEL_SHARE"]

Group = FrozenSet[int]
Pair = Tuple[int, int]

#: Weight on estimated depth, in weighted-gate units per layer.
#:
#: Not optional in practice. Intra- and cross-patch CNOT cost identical gates, so at
#: zero the objective cannot see CNOT placement at all and a CX-dominated program has
#: no signal whatever.
#:
#: 12.0 is the value `grouping/tune.py`'s grid search settled on, and the one the
#: committed figures score with. The old stack used it on the figure path while
#: `grouping.plan.plan_placement` passed 0.0, so the search and the measurement that
#: validated it wanted different things; here one number does both.
DEFAULT_DEPTH_WEIGHT = 12.0

#: How much of the depth estimate comes from the *busiest patch* rather than from the
#: total gate count. Depth is bracketed by two extremes:
#:
#:     sum over gates       nothing runs concurrently -- full serialisation
#:     busiest patch        everything that can run concurrently does
#:
#: This used to be the sum alone, and that is unable to represent the one thing depth is
#: actually sensitive to. A sum does not change when gates move between patches, so the
#: model could not prefer a balanced packing over a lopsided one at the same patch count
#: -- and since real depth is set by the slowest patch, that is precisely the choice that
#: matters. It also made the model rank the *known-best* partition worst on circuits with
#: clean block structure: a planted-block corpus where the best grouping saves 10.6% by
#: exact translation had the search finding 2.1%, because the model was pointing away
#: from it.
#:
#: Measured by rank agreement against exact translation, comparing partitions at a fixed
#: patch count (where the failure above lives):
#:
#:                            structured   QASMBench
#:     sum only  (share 0.0)      0.02        0.19
#:     half and half (0.5)        0.42        0.24
#:     busiest only (1.0)         0.41        0.16
#:
#: Half and half is the only setting better than the sum on both corpora, and it stays
#: as good as the sum when the patch count varies (0.95 / 0.90 against 0.95 / 0.88).
PARALLEL_SHARE = 0.5

#: **Retired. Nothing in the live objective reads this.** `price` no longer clamps the
#: movement term.
#:
#: What it was: a ceiling on movement as a multiple of the gate-and-depth work beside it,
#: fitted to stop a large `movement_cost` from becoming the entire objective -- the failure
#: being that the search would then minimise crossings alone, and since program order
#: already nearly minimises crossings on QASMBench (median 0.87 of a random permutation's,
#: exactly 0.50 on the chain and fan-out families, because algorithms get written
#: `q[i]-q[i+1]`), the descent landed back on naive. At an idle/distance ratio of 1/16,
#: uncapped, 60 of 76 proxies came back metrically identical to naive.
#:
#: Why it had to go: capping the *product* removes the term's gradient wherever it binds,
#: not merely its size. At `distance_scalar = 128` the raw term ran 108x the cap, so 100%
#: of 8,884 `price` calls during a single plan came back clamped -- movement collapsed to
#: `0.25 * work`, a function of `work` alone, and the search became blind to exactly what
#: the profile was telling it to avoid. It chose placements carrying 3.4x naive's real
#: movement while its own model recorded an improvement.
#:
#: Why the original failure no longer bites: it was measured against the pre-transport
#: pricing -- distance applied twice, unit-pitch hops, four round trips per crossing. All
#: of those changed, and `price` now charges *per-pair* distance via `patch_hop_matrix`
#: rather than a placement-blind mean, which is what lets the search reduce movement
#: deliberately instead of only by reducing crossing count. Re-measured on expected error
#: over 8 QASMBench proxies at `distance_scalar` 0.5 to 8, removing the cap strictly helps
#: or ties, and the count of *improved* circuits rose.
#:
#: Kept as a name because `experiments/tests/movement_cap_sweep.py` still reads it.
MOVEMENT_SHARE_CEILING = 0.25


@dataclass(frozen=True)
class Term:
    """
    One additive contribution. `crossings` and `depth` are tallies, not charges.

    `depth` is the summed per-gate depth. The per-patch split that `price` also needs
    is NOT here: a Term is cached by group membership and never learns which patch index
    filed it, so it cannot survive the search's delta scoping -- a patch that changes
    membership without changing a crossing's cost would leave its load under the old key.
    `busiest_patch_depth` reads it off the partition instead.

    `crossing_savings` is the movement counterpart of `savings`: crossings a fired
    transversal macro folds into one grouped movement, credited by `price` against the
    movement tally the same way `savings` is credited against work.
    """

    one_qubit: float = 0.0
    two_qubit: float = 0.0
    savings: float = 0.0
    crossings: float = 0.0
    depth: float = 0.0
    crossing_savings: float = 0.0

    def __add__(self, other: "Term") -> "Term":
        return Term(
            self.one_qubit + other.one_qubit,
            self.two_qubit + other.two_qubit,
            self.savings + other.savings,
            self.crossings + other.crossings,
            self.depth + other.depth,
            self.crossing_savings + other.crossing_savings,
        )

    def __sub__(self, other: "Term") -> "Term":
        return Term(
            self.one_qubit - other.one_qubit,
            self.two_qubit - other.two_qubit,
            self.savings - other.savings,
            self.crossings - other.crossings,
            self.depth - other.depth,
            self.crossing_savings - other.crossing_savings,
        )


_EMPTY = Term()


class PackingCost:
    """
    The objective, precomputed per program and cached per group.

    Build once per (program, code size); reuse across the whole search. Everything
    that does not depend on the partition is resolved in `__init__`, so a term is a
    handful of dictionary lookups rather than a walk over the circuit.
    """

    def __init__(
        self,
        profiles: Mapping[int, QubitProfile],
        n_code: int,
        *,
        num_qubits: Optional[int] = None,
        weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS,
        window: int = DEFAULT_WINDOW,
        depth_weight: float = DEFAULT_DEPTH_WEIGHT,
        movement_cost: float = 0.0,
        idle_cost: float = 0.0,
        distance_scale: float = 1.0,
        solo_hh: bool = False,
    ) -> None:
        self.profiles = profiles
        self.n_code = n_code
        self.k = logical_capacity(n_code)
        self.num_qubits = num_qubits if num_qubits is not None else (
            max(profiles) + 1 if profiles else 0
        )
        self.weights = weights
        self.window = window
        self.depth_weight = depth_weight
        self.movement_cost = movement_cost
        self.idle_cost = idle_cost
        self.distance_scale = distance_scale
        #: Price a solo patch's Hadamards as the whole-patch HH the translator emits for them
        #: (`translator.handlers.hadamard`; `TranslatorConfig.solo_hh_promotion`) rather than as
        #: single-lane gadgets. Off is the historical objective, in which a qubit's own gates
        #: cost the same wherever it sits -- blind to the one placement-dependent single-qubit
        #: cost the translator has, which matters wherever a qubit can be alone. Follows the
        #: translator's switch: with `SOLO_H_TARGETED` on there is no promotion to price.
        from translator import explore_switches

        self.solo_hh = bool(solo_hh) and not explore_switches.SOLO_H_TARGETED

        table = work_table(n_code)
        depths = gate_depths(n_code)
        w1, w2 = weights

        encode, decode = table["ENCODE"], table["DECODE"]
        self.per_patch = (encode[0] + decode[0]) * w1 + (encode[1] + decode[1]) * w2

        # --- single-qubit gates: same cost wherever the qubit sits, so a constant ---
        solo = 0.0
        for profile in profiles.values():
            for event in profile.events:
                if event.partner is not None:
                    continue
                gate = GATE_KEYS.get(event.kind)
                if gate is None:
                    continue
                cost = table.get(gate, (0, 0))
                solo += cost[0] * w1 + cost[1] * w2
        self.solo = solo

        # --- two-qubit gates, bucketed by unordered qubit pair ---
        gates: Dict[Pair, List[Tuple[int, str]]] = {}
        for qubit, profile in profiles.items():
            for event in profile.two_qubit_events():
                partner = event.partner
                if partner is None or partner <= qubit:
                    continue  # each gate is seen from both ends; keep one
                if GATE_KEYS.get(event.kind) is None:
                    continue
                gates.setdefault((qubit, partner), []).append((event.layer, event.kind))
        self.gates = gates

        # Intra-patch cost is deduplicated by (layer, kind) and cross-patch is not,
        # because `AnalyticCost._patch_term` dedupes and `_pair_term` does not. The
        # two differ only if one pair carries two same-kind gates in one layer, which
        # a dependency-respecting layering will not produce; preserved so the numbers
        # match term for term.
        self.intra: Dict[Pair, Term] = {}
        self.cross: Dict[Pair, Term] = {}
        for pair, events in gates.items():
            one_q = two_q = depth = 0.0
            for layer, kind in sorted(set(events)):
                gate = GATE_KEYS[kind]
                cost = table.get(gate, (0, 0))
                one_q += cost[0]
                two_q += cost[1]
                depth += depths.get(gate, (0, 0))[0]
            # Depth is NOT folded into `one_qubit` here. It used to be, which meant
            # the depth contribution was fixed at construction as a plain sum and
            # `price` had no way to reshape it. `price` now owns the whole depth charge.
            self.intra[pair] = Term(
                one_qubit=one_q * w1,
                two_qubit=two_q * w2,
                depth=depth,
            )
            one_q = two_q = depth = 0.0
            for layer, kind in events:
                gate = GATE_KEYS[kind]
                cost = CROSS_PATCH_COST.get(gate, table.get(gate, (0, 0)))
                one_q += cost[0]
                two_q += cost[1]
                depth += depths.get(gate, (0, 0))[1]
            self.cross[pair] = Term(
                one_qubit=one_q * w1,
                two_qubit=two_q * w2,
                crossings=float(len(events)),
                depth=depth,
            )

        # --- who shares a gate with whom ---
        # Used to bound `binary` recomputation: two patches with no gate between them
        # contribute nothing, so a move need only re-price the patches its qubits
        # actually reach.
        self.partners: Dict[int, Set[int]] = {q: set() for q in range(self.num_qubits)}
        for left, right in gates:
            self.partners[left].add(right)
            self.partners[right].add(left)

        self.layers: Dict[str, Dict[int, FrozenSet[int]]] = {
            kind: {q: p.layers_with(kind) for q, p in profiles.items()}
            for kind in ("h", "x", "z")
        }

        # --- macro savings, precomputed where possible ---
        self._h_single = table["H"][0] * w1 + table["H"][1] * w2
        # HH's anchor swap is an atom exchange, not a free relabel: two carriages of
        # movement on top of the n Hadamards. Charged here so the HH-vs-k-singles
        # comparison sees what the simulator charges.
        self._hh = (
            table["HH"][0] * w1 + table["HH"][1] * w2
            + carriage_cost("HH", weights)
        )
        self._pauli = {
            "x": table["X"][0] * w1 + table["X"][1] * w2,
            "z": table["Z"][0] * w1 + table["Z"][1] * w2,
        }
        self._macro = {
            kind: (
                CROSS_PATCH_COST.get(GATE_KEYS[kind], table[GATE_KEYS[kind]]),
                table[key],
            )
            for kind, key in (("cx", "CXT"), ("cnot", "CXT"), ("cz", "CZT"),
                              ("xcx", "XCXT"))
        }

        self._unary: Dict[Group, Term] = {}
        self._binary: Dict[Tuple[Group, Group], Term] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # terms
    # ------------------------------------------------------------------

    def unary(self, group: Group) -> Term:
        """Everything internal to one patch: intra-patch gates, minus HH and Pauli fusion."""
        if not group:
            return _EMPTY
        cached = self._unary.get(group)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1

        members = sorted(group)
        term = _EMPTY
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                inside = self.intra.get((left, right))
                if inside is not None:
                    term = term + inside
        term = Term(
            one_qubit=term.one_qubit,
            two_qubit=term.two_qubit,
            savings=self._within(members),
            depth=term.depth,
        )
        self._unary[group] = term
        return term

    def binary(self, left: Group, right: Group) -> Term:
        """Cross-patch gates between two patches, minus the transversal macros they enable."""
        if not left or not right:
            return _EMPTY
        key = (left, right) if min(left) <= min(right) else (right, left)
        cached = self._binary.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1

        first, second = key
        term = _EMPTY
        events: Dict[str, List[Tuple[int, int, int]]] = {}
        for source in sorted(first):
            for partner in second:
                pair = (source, partner) if source < partner else (partner, source)
                crossing = self.cross.get(pair)
                if crossing is None:
                    continue
                term = term + crossing
                for layer, kind in self.gates[pair]:
                    events.setdefault(kind, []).append((layer, source, partner))

        savings, crossing_savings = self._transversal(events, len(first), len(second))
        term = Term(
            one_qubit=term.one_qubit,
            two_qubit=term.two_qubit,
            savings=savings,
            crossings=term.crossings,
            depth=term.depth,
            crossing_savings=crossing_savings,
        )
        self._binary[key] = term
        return term

    # ------------------------------------------------------------------
    # macro savings
    # ------------------------------------------------------------------

    def _within(self, members: Sequence[int]) -> float:
        """Whole-patch Hadamard, plus Pauli fusion across lanes."""
        if len(members) == 1:
            # A qubit alone in its patch: every one of its Hadamards is a whole-patch HH,
            # coincident or not (`solo_hh`).
            if not self.solo_hh:
                return 0.0
            count = len(self.layers["h"].get(members[0], frozenset()))
            return count * (self._h_single - self._hh)
        if len(members) < 2:
            return 0.0
        savings = 0.0
        coincident = _coincident_layers(
            [self.layers["h"].get(q, frozenset()) for q in members], self.window
        )
        savings += len(coincident) * (len(members) * self._h_single - self._hh)
        for kind in ("x", "z"):
            lane = self._pauli[kind]
            for count in _fusable_group_sizes(
                [self.layers[kind].get(q, frozenset()) for q in members], self.window
            ):
                savings += count * lane - (count + (count % 2)) * self.weights[0]
        return savings

    def _transversal(
        self, events: Mapping[str, Sequence[Tuple[int, int, int]]], left: int, right: int
    ) -> Tuple[float, float]:
        """
        A transversal macro replaces one gate per lane, so it needs every lane.

        Both patches must hold the same number of qubits and the crossings must form
        a bijection over them. Lane *order* is free for the *partition* -- all slot
        permutations of a group score identically -- but it is no longer free for the
        *macro*: a non-identity alignment is fixed by routing swaps, and a routing swap
        is an atom exchange the simulator charges two carriages for. Since this model
        never sees slots, it charges the expectation over a uniformly random alignment:
        a random permutation of `m` lanes needs `m - H_m` transpositions, and the
        merging pass emits each as a swap before and after the macro. That keeps the
        objective lane-order-invariant while pricing what realignment actually costs
        on average.

        Returns `(work savings, crossing savings)`: the second is the movement credit
        -- a fired macro folds its `m` per-gate round trips into one grouped movement,
        so `m - 1` crossings' worth of transport disappears. (The macro's single trip
        carries n = m + 2-ish atoms where the singles carried 2 each, so the handling
        side of the credit is if anything conservative for m >= 2.)
        """
        if not events or left != right:
            return 0.0, 0.0
        w1, w2 = self.weights
        harmonic = sum(1.0 / i for i in range(1, left + 1)) if left else 0.0
        expected_swaps = 2.0 * max(0.0, left - harmonic)
        realign = expected_swaps * carriage_cost("SWAP", self.weights)
        savings = 0.0
        crossing_savings = 0.0
        for kind, crossings in events.items():
            entry = self._macro.get(kind)
            if entry is None:
                continue
            single, macro = entry
            gain = left * (single[0] * w1 + single[1] * w2) - (
                macro[0] * w1 + macro[1] * w2
            ) - realign
            if gain <= 0:
                continue
            fired = _complete_bijections(crossings, left, self.window)
            savings += gain * fired
            crossing_savings += (left - 1) * fired
        return savings, crossing_savings

    # ------------------------------------------------------------------
    # assembling a whole partition
    # ------------------------------------------------------------------

    def busiest_patch_depth(self, groups: Sequence[Group]) -> float:
        """
        Depth load of the most-occupied patch, from the gate inventory.

        A cross-patch gate is charged to *both* endpoints, because it ties up both
        patches while it runs. Computed from the partition rather than accumulated into
        `Term`: patch identity lives with the partition, and a group-keyed accumulator
        silently desynchronises when the search rescopes a delta.

        O(gates touching a changed patch) if a caller memoises per patch; O(gates) here.
        """
        owner: Dict[int, int] = {
            q: index for index, group in enumerate(groups) for q in group
        }
        load: Dict[int, float] = {}
        for (left, right), term in self.intra.items():
            a, b = owner.get(left), owner.get(right)
            if a is not None and a == b:
                load[a] = load.get(a, 0.0) + term.depth
        for (left, right), term in self.cross.items():
            a, b = owner.get(left), owner.get(right)
            if a is None or b is None or a == b:
                continue
            load[a] = load.get(a, 0.0) + term.depth
            load[b] = load.get(b, 0.0) + term.depth
        return max(load.values(), default=0.0)

    def movement_hops(self, groups: Sequence[Group]) -> float:
        """
        Cell-distance the cross-patch traffic of *this* partition actually travels.

        `sum over patch pairs of (crossings between them) x (their centre separation)`,
        against `price`'s fallback of `total crossings x mean separation`. Same quantity
        when every pair happens to sit at the mean, and different exactly when a placement
        has chosen which pairs carry the traffic -- which is the whole point of a placement.

        Read off the partition rather than out of a `Term`, for the reason `Term` documents:
        a Term is cached by group membership and never learns which patch index filed it,
        so it cannot carry a per-pair distance through the search's delta scoping.
        `busiest_patch_depth` is the precedent -- `Packing.value_after` already recomputes
        that on the hypothetical groups for every candidate move, and this joins it.

        Cheap despite the pair loop: `binary` is memoised on `self._binary`, so a rescore
        after a single-qubit move re-derives only the pairs whose membership changed.
        """
        live = [group for group in groups if group]
        if len(live) < 2:
            return 0.0
        matrix = patch_hop_matrix(len(live), self.n_code + 2)
        total = 0.0
        for first in range(len(live)):
            for second in range(first + 1, len(live)):
                term = self.binary(live[first], live[second])
                moved = term.crossings - term.crossing_savings
                if moved > 0.0:
                    total += moved * matrix[first][second]
        return total

    def estimated_depth(self, total: Term, busiest: float) -> float:
        """
        Layers this partition is expected to take, between the two bracketing extremes.

        `total.depth` is every gate's depth added up, which is the answer if nothing runs
        concurrently. The largest per-patch load is the answer if everything that can run
        concurrently does, since the slowest patch gates the circuit. Real circuits sit
        between, and `PARALLEL_SHARE` says where -- see that constant for the measurement.

        The maximum is what makes load *balance* visible at all: the sum is unchanged by
        moving gates between patches, so a model built on it alone cannot prefer an even
        spread, which is exactly what matters when idle noise dominates.
        """
        return (1.0 - PARALLEL_SHARE) * total.depth + PARALLEL_SHARE * busiest

    def price(
        self,
        patches: int,
        total: Term,
        busiest: float = 0.0,
        hop_distance: Optional[float] = None,
    ) -> float:
        """
        Turn accumulated terms plus a patch count into one number.

        `busiest` is `busiest_patch_depth(groups)`; it defaults to 0.0 so a caller with
        only a Term still gets the pure-sum depth estimate rather than an exception.

        `hop_distance` is `movement_hops(groups)` -- the real cell-distance this partition's
        crossings travel. Left at None the movement term falls back to
        `crossings x mean_patch_hops`, which is placement-blind: at a fixed patch count it
        is the same number for every layout, so an objective using it charges a placement
        that keeps interacting qubits adjacent exactly what it charges one that scatters
        them. Supplied by `of` and by `Packing.value`/`value_after`, so the search and both
        rescoring paths all see distance. The fallback exists for callers holding only a
        Term, and for `mean_patch_hops`' remaining legitimate users.
        """
        if patches <= 0:
            return 0.0
        depth = self.estimated_depth(total, busiest)
        work = (
            total.one_qubit + total.two_qubit - total.savings + self.depth_weight * depth
        )
        # Crossings a fired macro folds into one grouped movement are credited first --
        # the transport model charges one round trip per movement, not per gate.
        #
        # **Uncapped, deliberately.** This used to be clamped to
        # `MOVEMENT_SHARE_CEILING * work`, which is indefensible once every term is in the
        # same currency: `packing_profile` converts probabilities to weighted-gate units at
        # `w2 / p2`, so capping movement at a quarter of gate-and-depth work asserts that
        # movement error cannot exceed a quarter of gate error -- a claim about physics,
        # imposed as a numerical guard.
        #
        # It was measurably harmful. At `distance_scalar = 128` the raw term ran 108x the
        # cap, so 100% of 8,884 `price` calls during one plan came back clamped: movement
        # collapsed to `0.25 * work`, a function of `work` alone, with *zero gradient* in
        # crossings or layout. The search could not see movement at all, chose placements
        # carrying 3.4x naive's real movement, and `plan.measured_cost` -- which had no
        # movement term either -- signed off on them.
        #
        # Removing it strictly helps or ties on measured expected error over 8 QASMBench
        # proxies across `distance_scalar` 0.5 to 8: unchanged at 0.5 and 1 where the cap
        # never bound, and at `distance_scalar` 8 the mean improvement over naive goes
        # +0.31% -> +5.52% at k=2 and +2.62% -> +7.50% at k=4.
        #
        # The failure mode it was added for -- "the search minimises crossings alone and
        # returns the naive packing it started from", 10 of 12 proxies unchanged -- was
        # measured against the *old* pricing, where distance was applied twice and hops
        # were unit-pitch. It does not reproduce on the current pricing: the count of
        # improved circuits rose when the cap came off. `test_movement_scales_with_profile`
        # pins the linearity this restores, and the corpus check is the integration guard.
        moved = max(0.0, total.crossings - total.crossing_savings)
        # `distance_scale` interpolates between counting crossings and charging cells, and
        # the shape is kept exactly as it was: the old expression
        # `moved * (1 + scale*(mean_hops - 1))` equals `moved + scale*(moved*mean_hops -
        # moved)`, so substituting the true per-pair sum for `moved * mean_hops` is a
        # strict generalisation -- identical whenever every pair sits at the mean, and
        # different precisely when the placement chose which pairs carry the traffic.
        travelled = moved * mean_patch_hops(patches, self.n_code + 2) \
            if hop_distance is None else hop_distance
        movement = self.movement_cost * max(
            0.0, moved + self.distance_scale * (travelled - moved)
        )
        # Idling is charged on the same estimate: it is every physical qubit sitting
        # through every layer, so it scales with the depth the circuit actually takes
        # rather than with the number of gates in it.
        idle = self.idle_cost * depth * patches * (self.n_code + 2)
        return (
            self.solo
            + patches * self.per_patch
            + work
            + movement
            + idle
        )

    def of(self, groups: Sequence[Group]) -> float:
        """Cost of a partition, from scratch. The search uses deltas instead."""
        live = [g for g in groups if g]
        total = _EMPTY
        for group in live:
            total = total + self.unary(group)
        for index, left in enumerate(live):
            for right in live[index + 1:]:
                total = total + self.binary(left, right)
        return self.price(len(live), total, self.busiest_patch_depth(live),
                          self.movement_hops(live))

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_program(
        cls, program: QuantumCircuit, n_code: int, **kwargs
    ) -> "PackingCost":
        """
        Build from a program, scoring the circuit the translator will actually place.

        `translated_view` applies the translator's own program-level passes first.
        They rewrite gate *kinds* -- a CX with a commuted Hadamard on its control
        becomes an XCX -- and every cost here is keyed by kind, so scoring the raw
        program prices gates that will never exist. Worth 13-49% error.
        """
        view = translated_view(program)
        return cls(
            profiles=extract_profiles(view),
            n_code=n_code,
            num_qubits=program.num_qubits,
            **kwargs,
        )
