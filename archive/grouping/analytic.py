"""
analytic.py — placement cost without translating the circuit.

The local search in `refine.py` used to score a candidate by translating the whole
program and counting gates. That is exact, and it is why the search cost `O(m . g)`
per candidate: at 140 qubits one evaluation took ~700ms, so a single sweep of the
`O(m^2)` neighbourhood ran for hours. This module replaces it with an *additive*
cost that can be recomputed for the two patches a move touches and reused everywhere
else.

The decomposition
-----------------
Every physical gate the translator emits is attributable to one patch or one ordered
pair of patches::

    cost(placement) = sum_P patch_term(P) + sum_{P<Q} pair_term(P, Q)

`patch_term`  single-qubit gates on the patch's members, two-qubit gates with both
              operands inside it, and the savings from whole-patch HH and from Pauli
              fusion among its lanes.
`pair_term`   two-qubit gates with one operand in each patch, and the savings from a
              transversal CXT/CZT/XCXT between them.

Pairs with no gates between them contribute nothing and are never visited, so the
sum is over the *edges of the interaction graph* rather than over all `O(P^2)` pairs.

Why this is exact for gate counts, and where it is not
------------------------------------------------------
Per-gate costs come from `iceberg.costs`, the same table the constructions are
verified against, so the un-merged part is exact rather than a surrogate. Two things
are estimates:

*merging*   a macro depends on several gates being adjacent in time, which is a
            scheduling property. The model applies the same rules the pass does --
            whole-patch macros need every occupied lane, Pauli fusion needs two or
            more -- within a layer window, but it cannot predict how the pass will
            resolve conflicts between overlapping candidates, so it over-counts when
            candidates collide.
*depth*     summed per-gate depth only (`GATE_DEPTH`, off by default), which is a
            proxy: true circuit depth is not decomposable per patch.

Measured against the translator, the split is clean. With merging disabled the model
is exact -- 0.00% error on a k=2 sample, 0.8% at k=4 where alignment SWAPs enter.
With merging enabled the error is 2-3.4%, all of it the macro estimate. Rank
correlation over random placements is +0.94 to +0.98 for k >= 4.

That kind of estimate has gone wrong here before: the pairwise packer's savings
estimate correlates with measured cost at r = -0.17, i.e. it points the wrong way. So
`validate` lives in the module rather than in a test, and anything that optimises
against this model should be able to show its agreement.

One trap, since it cost a 13-49% error before it was found: score the circuit the
*translator sees*, not the program as written. `Translator.translate` runs
`cancel_inverse_gates` and `commute_hadamards` first, which rewrites gates and
cancels Hadamards outright. Use `AnalyticCost.for_program`, which applies them.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from iceberg.costs import DEFAULT_WORK_WEIGHTS, depth_table, work_table
from iceberg.utils import logical_capacity

from packing.profiles import QubitProfile, extract_profiles

__all__ = [
    "AnalyticCost",
    "CostBreakdown",
    "ValidationReport",
    "mean_patch_hops",
    "validate",
]

Placement = Sequence[Tuple[int, int]]

#: Gates whose cross-patch construction differs from the intra-patch one. Everything
#: else costs the same either way, which is why co-location buys less here than it
#: would in a code with expensive inter-block operations.
CROSS_PATCH_COST: Dict[str, Tuple[int, int]] = {
    "CZ": (0, 4),
    "XCX": (4, 4),
}

#: Depth of each two-qubit construction as (intra-patch, cross-patch), from the
#: measured `iceberg.costs.depth_table` rather than from guesswork.
#:
#: This is the other half of the placement trade-off and it does not show up in gate
#: counts at all. Inter-patch CNOT is depth 3 against intra-patch 4, so a CNOT-only
#: circuit is exactly placement-neutral by gate count while still being worth placing
#: well -- and a transversal macro is depth 1 for all k lanes at once, which is the
#: largest single win available.
#:
#: Summing per-gate depth is a proxy for circuit depth, exact only when gates
#: serialise. It is priced through `AnalyticCost.depth_weight`.


def gate_depths(n_code: int) -> Dict[str, Tuple[int, int]]:
    """(intra, cross) depth per two-qubit gate key, at one code size."""
    table = depth_table(n_code)
    return {
        key: (table[key], table.get(f"{key}_CROSS", table[key]))
        for key in ("CX", "CZ", "XCX")
    }


#: Program gate name -> cost-table key.
GATE_KEYS: Dict[str, str] = {
    "x": "X", "y": "Y", "z": "Z", "h": "H", "s": "S", "sdg": "S",
    "sx": "SX", "sxdg": "SX", "rz": "RZ", "rx": "RX",
    "cx": "CX", "cnot": "CX", "cz": "CZ", "xcx": "XCX", "swap": "SWAP",
}

#: Layer tolerance for calling two gates simultaneous, matching the merging pass's
#: willingness to delay a gate by a few layers to make a macro fire.
DEFAULT_WINDOW = 2

#: Mean patch separation, cached by patch count. Small dictionary, computed once.
_MEAN_HOPS: Dict[int, float] = {}


def mean_patch_hops(num_patches: int) -> float:
    """
    Mean Manhattan distance between two distinct patches, in grid steps.

    The layout convention is `simulator.grid_constructor.build_physical_qubit_positions`
    verbatim -- `ceil(sqrt(P))` columns, column-major fill -- so the model predicts the
    same geometry the distance-dependent noise model will later charge for. Enumerated
    rather than approximated: P is at most a few hundred, and the result is cached.

    Grows like `sqrt(P)`, which is the whole point. It is the term that makes spreading a
    program over more patches cost something on a machine where moving atoms is what
    limits fidelity. One patch means no crossings and no distance, hence 0.
    """
    if num_patches < 2:
        return 0.0
    cached = _MEAN_HOPS.get(num_patches)
    if cached is not None:
        return cached

    columns = max(1, math.ceil(math.sqrt(num_patches)))
    rows = math.ceil(num_patches / columns)
    coords = [(index // rows, index % rows) for index in range(num_patches)]

    total = 0.0
    pairs = 0
    for first in range(num_patches):
        for second in range(first + 1, num_patches):
            total += abs(coords[first][0] - coords[second][0])
            total += abs(coords[first][1] - coords[second][1])
            pairs += 1
    result = total / pairs if pairs else 0.0
    _MEAN_HOPS[num_patches] = result
    return result


@dataclass(frozen=True)
class CostBreakdown:
    """
    Where a placement's cost comes from, for reporting and debugging.

    `one_qubit`, `two_qubit` and `merge_savings` are additive over patches and pairs, so
    they can be cached per term. `movement` and `idle` cannot: both depend on the *total*
    patch count, which is a property of the whole placement rather than of any one term.
    So the per-term code tallies the two quantities those charges are computed from --
    `crossings` and `depth` -- and `AnalyticCost.breakdown` prices them once at the end.
    """

    one_qubit: float = 0.0
    two_qubit: float = 0.0
    merge_savings: float = 0.0
    #: Cross-patch two-qubit gates. Each forces the two patches to be brought together,
    #: so this is what the movement charge is levied on.
    crossings: float = 0.0
    #: Summed per-gate depth. Already priced into `one_qubit` via `depth_weight`; kept
    #: separately because the idle charge needs it as a duration.
    depth: float = 0.0
    #: Priced by `breakdown` from `crossings` and the patch count, not by the terms.
    movement: float = 0.0
    idle: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.one_qubit + self.two_qubit + self.movement + self.idle
            - self.merge_savings
        )

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            one_qubit=self.one_qubit + other.one_qubit,
            two_qubit=self.two_qubit + other.two_qubit,
            merge_savings=self.merge_savings + other.merge_savings,
            crossings=self.crossings + other.crossings,
            depth=self.depth + other.depth,
            movement=self.movement + other.movement,
            idle=self.idle + other.idle,
        )


@dataclass
class AnalyticCost:
    """
    Additive placement cost for one program at one code size.

    Terms are cached by their membership, so re-scoring a placement that differs by
    one swap recomputes two patch terms and the pair terms incident to them, and
    reuses everything else. Call `cost` as often as the search likes.
    """

    profiles: Mapping[int, QubitProfile]
    n_code: int
    weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS
    window: int = DEFAULT_WINDOW
    #: Charge for moving a qubit between patches, in weighted-gate units per unit
    #: distance. Zero reproduces the paper's convention that atom movement is free;
    #: raising it makes the model prefer placements that keep interacting qubits in
    #: nearby patches, which is what a movement- or idle-noise-dominated machine
    #: wants. See `graph.NoiseProfile`.
    movement_cost: float = 0.0
    #: Charge per physical-qubit-layer of exposure, same units. Every physical qubit
    #: decoheres for the whole circuit, so this is levied on the space-time volume
    #: `depth * patches * (n + 2)`. That rises with the patch count as well as with depth,
    #: which is the second reason -- alongside encode/decode -- for an extra patch to have
    #: to justify itself. It was declared but never read until this revision.
    #:
    #: Two approximations, both stated because they set how far to trust it: `depth` is
    #: the *summed* per-gate depth rather than the schedule's critical path, so the
    #: absolute figure is an over-estimate; and no credit is given for qubits that are
    #: busy rather than idle, so strictly this is total exposure and not idle time. Both
    #: are monotone in patch count and in depth, which is what the patch-count comparison
    #: needs.
    idle_cost: float = 0.0
    #: How much of `mean_patch_hops` to apply to the movement charge. 0 charges one unit
    #: per crossing regardless of layout, which is what this model did before and what
    #: every committed figure ran with (they all set `movement_cost=0`, so the whole term
    #: is inert there). 1 charges the true mean separation, so a program spread over 16
    #: patches pays about twice what the same program pays over 4.
    distance_scale: float = 1.0
    #: Weight on summed per-gate depth, in weighted-gate units per layer. Nonzero by
    #: default: with depth unpriced the model is blind to CNOT placement entirely,
    #: since intra- and cross-patch CNOT cost identical gates. `validate` sets it to
    #: zero, because there it is checking the gate accounting against gate counts.
    depth_weight: float = 3.0

    _patch_cache: Dict[Tuple, CostBreakdown] = field(default_factory=dict, repr=False)
    _pair_cache: Dict[Tuple, CostBreakdown] = field(default_factory=dict, repr=False)
    _table: Dict[str, Tuple[int, int]] = field(default_factory=dict, repr=False)
    #: How many term evaluations were served from cache; the incremental speedup.
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self._table = work_table(self.n_code)
        self._depths = gate_depths(self.n_code)
        self._edges = _interaction_edges(self.profiles)

    @classmethod
    def for_program(
        cls,
        program: QuantumCircuit,
        n_code: int,
        *,
        enable_h_commutation: bool = True,
        **kwargs,
    ) -> "AnalyticCost":
        """
        Build a model for `program`, scoring the circuit the translator will see.

        Always prefer this to constructing from raw profiles. `Translator.translate`
        applies `cancel_inverse_gates` and `commute_hadamards` before placing
        anything, so the gates it costs are not the gates the program was written
        with -- commutation rewrites X into Z, CX into XCX, and cancels Hadamards
        outright. Scoring the raw program instead was worth a 13-49% error and
        dropped rank correlation to +0.22; scoring the preprocessed circuit makes
        the model reproduce translated gate counts exactly.
        """
        return cls(
            profiles=extract_profiles(translated_view(program, enable_h_commutation)),
            n_code=n_code,
            **kwargs,
        )

    # --- public API ---

    @property
    def k(self) -> int:
        return logical_capacity(self.n_code)

    def cost(self, placement: Placement) -> float:
        """Weighted gate count for `placement`."""
        return self.breakdown(placement).total

    def breakdown(self, placement: Placement) -> CostBreakdown:
        by_patch: Dict[int, List[int]] = {}
        slot_of: Dict[int, int] = {}
        for qubit, (patch, slot) in enumerate(placement):
            by_patch.setdefault(patch, []).append(qubit)
            slot_of[qubit] = slot

        total = CostBreakdown()
        # Every patch is encoded once and decoded once regardless of what runs on
        # it, and both scale with n. A constant offset while the patch count is
        # fixed, but the planner compares placements that use *different* numbers of
        # patches, and there this is the term that pays for an extra patch.
        encode, decode = self._table["ENCODE"], self._table["DECODE"]
        per_patch = CostBreakdown(
            one_qubit=(encode[0] + decode[0]) * self.weights[0],
            two_qubit=(encode[1] + decode[1]) * self.weights[1],
        )
        for _ in by_patch:
            total = total + per_patch

        for patch, members in by_patch.items():
            total = total + self._patch_term(members, slot_of)

        patch_of = {q: p for p, members in by_patch.items() for q in members}
        seen: set = set()
        for left, right in self._edges:
            if left >= len(placement) or right >= len(placement):
                continue
            pl, pr = patch_of[left], patch_of[right]
            if pl == pr:
                continue
            key = (min(pl, pr), max(pl, pr))
            if key in seen:
                continue
            seen.add(key)
            total = total + self._pair_term(by_patch[key[0]], by_patch[key[1]], slot_of)

        # Both of these depend on how many patches the placement used, so neither can
        # live in a term cached by membership -- see CostBreakdown. Priced once, here.
        patches = len(by_patch)
        hops = 1.0 + self.distance_scale * (mean_patch_hops(patches) - 1.0)
        total = CostBreakdown(
            one_qubit=total.one_qubit,
            two_qubit=total.two_qubit,
            merge_savings=total.merge_savings,
            crossings=total.crossings,
            depth=total.depth,
            movement=self.movement_cost * total.crossings * max(0.0, hops),
            idle=self.idle_cost * total.depth * patches * (self.n_code + 2),
        )
        return total

    def objective(self):
        """A `refine.Objective` closure over this model."""

        def evaluate(placement: Placement) -> float:
            return self.cost(placement)

        return evaluate

    # --- patch-local term ---

    def _patch_term(self, members: Sequence[int], slot_of: Mapping[int, int]) -> CostBreakdown:
        key = (tuple(sorted(members)), tuple(slot_of[q] for q in sorted(members)))
        cached = self._patch_cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1

        inside = set(members)
        one_q = two_q = depth = 0.0
        counted: set = set()
        for qubit in members:
            profile = self.profiles.get(qubit)
            if profile is None:
                continue
            for event in profile.events:
                gate = GATE_KEYS.get(event.kind)
                if gate is None:
                    continue
                if event.partner is None:
                    cost = self._table.get(gate, (0, 0))
                    one_q += cost[0]
                    two_q += cost[1]
                elif event.partner in inside:
                    # Charge an intra-patch two-qubit gate once, not once per end.
                    tag = (min(qubit, event.partner), max(qubit, event.partner), event.layer, event.kind)
                    if tag in counted:
                        continue
                    counted.add(tag)
                    cost = self._table.get(gate, (0, 0))
                    one_q += cost[0]
                    two_q += cost[1]
                    depth += self._depths.get(gate, (0, 0))[0]

        savings = self._merge_savings_within(members)
        result = CostBreakdown(
            one_qubit=one_q * self.weights[0] + depth * self.depth_weight,
            two_qubit=two_q * self.weights[1],
            merge_savings=savings,
            depth=depth,
        )
        self._patch_cache[key] = result
        return result

    def _merge_savings_within(self, members: Sequence[int]) -> float:
        """
        Whole-patch Hadamard plus Pauli fusion, in weighted units.

        The HH term is conjunctive: the macro applies H to every lane, so it only
        fires on layers where *all* occupied lanes want one. The Pauli term is
        super-additive: X on m lanes costs m + (m mod 2) single-qubit gates instead
        of 2m, because the anchor factor cancels for an even count.
        """
        if len(members) < 2:
            return 0.0
        w1, w2 = self.weights
        savings = 0.0

        layers = [self._layers_of(q, "h") for q in members]
        for layer in _coincident_layers(layers, self.window):
            del layer
            single = self._table["H"]
            macro = self._table["HH"]
            savings += (
                len(members) * (w1 * single[0] + w2 * single[1])
                - (w1 * macro[0] + w2 * macro[1])
            )

        for kind in ("x", "z"):
            per_lane = self._table["X" if kind == "x" else "Z"]
            lane_cost = w1 * per_lane[0] + w2 * per_lane[1]
            for count in _fusable_group_sizes(
                [self._layers_of(q, kind) for q in members], self.window
            ):
                # m aligned Paulis cost m + (m mod 2) 1q gates, against 2m apart.
                fused = (count + (count % 2)) * w1
                savings += count * lane_cost - fused
        return savings

    # --- cross-patch term ---

    def _pair_term(
        self,
        left: Sequence[int],
        right: Sequence[int],
        slot_of: Mapping[int, int],
    ) -> CostBreakdown:
        key = (
            tuple(sorted(left)),
            tuple(sorted(right)),
            tuple(slot_of[q] for q in sorted(left)),
            tuple(slot_of[q] for q in sorted(right)),
        )
        cached = self._pair_cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1

        right_set = set(right)
        one_q = two_q = depth = 0.0
        crossings: List[Tuple[int, int, int, str]] = []
        for qubit in left:
            profile = self.profiles.get(qubit)
            if profile is None:
                continue
            for event in profile.two_qubit_events():
                if event.partner not in right_set:
                    continue
                gate = GATE_KEYS.get(event.kind)
                if gate is None:
                    continue
                cost = CROSS_PATCH_COST.get(gate, self._table.get(gate, (0, 0)))
                one_q += cost[0]
                two_q += cost[1]
                depth += self._depths.get(gate, (0, 0))[1]
                crossings.append((qubit, event.partner, event.layer, event.kind))

        savings = self._transversal_savings(crossings, left, right, slot_of)
        # Moving data between patches is what `movement_cost` prices. One charge per
        # crossing gate, since that is when the two patches must be brought together.
        # Movement is *not* charged here. It scales with how far apart the two patches
        # sit, which depends on the total patch count -- a property of the placement, not
        # of this pair, and not part of this cache key. Tally the crossings and let
        # `breakdown` price them once it knows the layout.
        result = CostBreakdown(
            one_qubit=one_q * self.weights[0] + depth * self.depth_weight,
            two_qubit=two_q * self.weights[1],
            merge_savings=savings,
            crossings=float(len(crossings)),
            depth=depth,
        )
        self._pair_cache[key] = result
        return result

    def _transversal_savings(
        self,
        crossings: Sequence[Tuple[int, int, int, str]],
        left: Sequence[int],
        right: Sequence[int],
        slot_of: Mapping[int, int],
    ) -> float:
        """
        A transversal macro replaces one gate per lane, so it needs every lane.

        Mirrors `translator.merging.transversal`: same gate kind, one gate per
        occupied lane on each side, and a slot bijection. Alignment itself is free --
        routing swaps are atom movement -- so the condition is that the crossings
        form a bijection, not that they are already lane-i-to-lane-i.
        """
        if not crossings or len(left) != len(right):
            return 0.0
        w1, w2 = self.weights
        savings = 0.0

        by_kind: Dict[str, List[Tuple[int, int, int]]] = {}
        for source, partner, layer, kind in crossings:
            by_kind.setdefault(kind, []).append((layer, source, partner))

        for kind, events in by_kind.items():
            macro_key = {"cx": "CXT", "cnot": "CXT", "cz": "CZT", "xcx": "XCXT"}.get(kind)
            if macro_key is None:
                continue
            single = CROSS_PATCH_COST.get(GATE_KEYS[kind], self._table[GATE_KEYS[kind]])
            macro = self._table[macro_key]
            gain = len(left) * (w1 * single[0] + w2 * single[1]) - (
                w1 * macro[0] + w2 * macro[1]
            )
            if gain <= 0:
                continue
            savings += gain * _complete_bijections(events, len(left), self.window)
        return savings

    # --- helpers ---

    def _layers_of(self, qubit: int, kind: str) -> FrozenSet[int]:
        profile = self.profiles.get(qubit)
        return profile.layers_with(kind) if profile else frozenset()


# --------------------------------------------------------------------------
# combinatorial helpers
# --------------------------------------------------------------------------


def translated_view(
    program: QuantumCircuit, enable_h_commutation: bool = True
) -> QuantumCircuit:
    """
    The circuit the translator actually places, after its program-level passes.

    These passes run before placement and do not depend on it, so they can be
    applied once and reused for every candidate. Everything that scores a placement
    -- this model, the interaction graph, the partitioner -- must work from this
    view rather than the raw program.
    """
    from translator.passes import cancel_inverse_gates, commute_hadamards

    processed = cancel_inverse_gates(program)
    if enable_h_commutation:
        processed = commute_hadamards(processed)
        processed = cancel_inverse_gates(processed)
    return processed


def _interaction_edges(profiles: Mapping[int, QubitProfile]) -> List[Tuple[int, int]]:
    edges = set()
    for qubit, profile in profiles.items():
        for event in profile.two_qubit_events():
            edges.add((min(qubit, event.partner), max(qubit, event.partner)))
    return sorted(edges)


def _coincident_layers(
    layer_sets: Sequence[FrozenSet[int]], window: int
) -> List[int]:
    """
    Layers where *every* set has an entry within `window`.

    Used for the conjunctive HH term: one returned layer is one macro.
    """
    if not layer_sets or any(not s for s in layer_sets):
        return []
    anchor = sorted(layer_sets[0])
    found: List[int] = []
    used: List[set] = [set() for _ in layer_sets]
    for layer in anchor:
        picks = []
        for index, layers in enumerate(layer_sets):
            candidate = next(
                (
                    other
                    for other in sorted(layers)
                    if other not in used[index] and abs(other - layer) <= window
                ),
                None,
            )
            if candidate is None:
                break
            picks.append(candidate)
        if len(picks) == len(layer_sets):
            for index, pick in enumerate(picks):
                used[index].add(pick)
            found.append(layer)
    return found


def _fusable_group_sizes(
    layer_sets: Sequence[FrozenSet[int]], window: int
) -> List[int]:
    """
    Sizes of the groups of coincident Paulis, one entry per fusable group.

    Unlike HH this does not need every lane, so a group of any size >= 2 counts.
    """
    occurrences: List[Tuple[int, int]] = []
    for index, layers in enumerate(layer_sets):
        occurrences.extend((layer, index) for layer in layers)
    occurrences.sort()

    sizes: List[int] = []
    used_lanes: set = set()
    current: List[Tuple[int, int]] = []
    for layer, index in occurrences:
        if current and (layer - current[0][0] > window or index in used_lanes):
            if len(current) >= 2:
                sizes.append(len(current))
            current, used_lanes = [], set()
        current.append((layer, index))
        used_lanes.add(index)
    if len(current) >= 2:
        sizes.append(len(current))
    return sizes


def _complete_bijections(
    events: Sequence[Tuple[int, int, int]], lanes: int, window: int
) -> int:
    """
    How many times `lanes` crossings line up in time with distinct endpoints.

    One such set is one transversal macro. Greedy over layers, which is what the
    detector does too -- it accumulates candidates and fires as soon as it holds a
    full bijection.
    """
    if lanes <= 0:
        return 0
    ordered = sorted(events)
    macros = 0
    sources: set = set()
    partners: set = set()
    start: Optional[int] = None
    for layer, source, partner in ordered:
        if start is not None and layer - start > window:
            sources, partners, start = set(), set(), None
        if source in sources or partner in partners:
            sources, partners, start = set(), set(), None
        if start is None:
            start = layer
        sources.add(source)
        partners.add(partner)
        if len(sources) == lanes and len(partners) == lanes:
            macros += 1
            sources, partners, start = set(), set(), None
    return macros


# --------------------------------------------------------------------------
# validation against the translator
# --------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """How well the model tracks what the translator actually emits."""

    samples: int
    correlation: float
    mean_relative_error: float
    worst_relative_error: float
    model: List[float] = field(default_factory=list)
    measured: List[float] = field(default_factory=list)

    @property
    def distinct_costs(self) -> int:
        """How many different costs the sampled placements actually had."""
        return len(set(self.measured))

    @property
    def spread(self) -> float:
        """Relative range of measured cost across the sample."""
        if not self.measured:
            return 0.0
        low, high = min(self.measured), max(self.measured)
        return (high - low) / low if low else 0.0

    @property
    def regret(self) -> float:
        """
        How much worse the model's pick is than the true best, relatively.

        The decision-relevant number, and the one to judge on. Rank correlation
        answers "does the model order placements correctly", which is stricter than
        anything a search needs and misleading when placements are near-identical: at
        n=12 a sample spanning 3.7% in cost gave r = +0.835 purely because a 0.4%
        model error is enough to shuffle near-ties. Regret answers the question that
        matters -- if I optimise this model, what do I lose? -- and there it was 0.
        """
        if not self.measured:
            return float("nan")
        pick = min(range(len(self.model)), key=lambda i: self.model[i])
        best = min(self.measured)
        return (self.measured[pick] - best) / best if best else 0.0

    @property
    def usable(self) -> bool:
        """
        Whether the model is safe to optimise against.

        Low regret, or high rank agreement. A model that points the *wrong* way is
        worse than no model -- the pairwise packer's estimate manages exactly that at
        r = -0.17 -- and either criterion rules that out.
        """
        if self.samples and self.regret <= 0.02:
            return True
        return self.correlation >= 0.9

    def summary(self) -> str:
        verdict = "usable" if self.usable else "NOT USABLE"
        return (
            f"{self.samples} placements: regret={self.regret:.2%} "
            f"r={self.correlation:+.3f} mean|err|={self.mean_relative_error:.1%} "
            f"worst={self.worst_relative_error:.1%} "
            f"(cost spread {self.spread:.1%}) -> {verdict}"
        )


def validate(
    program: QuantumCircuit,
    n_code: int,
    *,
    placements: Sequence[Placement],
    model: Optional[AnalyticCost] = None,
) -> ValidationReport:
    """
    Compare the model against real translated gate counts over `placements`.

    Returns rank correlation rather than absolute agreement, because the search only
    ever asks "is A cheaper than B". Absolute error is reported too so a systematic
    offset is visible.

    Measured counts come from `one_two_qubit_gate_totals`, which excludes SWAPs by
    the paper's convention that atom movement is not a gate -- matching the model,
    which prices SWAP at (0, 0). `evaluate.measure` counts them, so using it here
    would compare two different currencies.
    """
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from translator.config import TranslatorConfig
    from translator.core import Translator

    if model is None:
        # depth_weight=0: this compares the model against measured *gate* counts,
        # so pricing depth here would guarantee a mismatch.
        model = AnalyticCost.for_program(program, n_code, depth_weight=0.0)
    w1, w2 = model.weights
    translator = Translator(config=TranslatorConfig(n_code=n_code))

    predicted: List[float] = []
    actual: List[float] = []
    for placement in placements:
        try:
            result = translator.translate(program, list(placement))
        except Exception:  # noqa: BLE001 - rejected placements have nothing to compare
            continue
        one, two = one_two_qubit_gate_totals(result.circuit)
        predicted.append(model.cost(placement))
        actual.append(w1 * one + w2 * two)

    if len(predicted) < 2:
        return ValidationReport(len(predicted), float("nan"), float("nan"), float("nan"))

    correlation = _spearman(predicted, actual)
    errors = [
        abs(p - a) / a for p, a in zip(predicted, actual) if a > 0
    ]
    return ValidationReport(
        samples=len(predicted),
        correlation=correlation,
        mean_relative_error=statistics.fmean(errors) if errors else float("nan"),
        worst_relative_error=max(errors) if errors else float("nan"),
        model=predicted,
        measured=actual,
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Rank correlation, which is what matters for a search that only compares."""
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    if n < 2:
        return float("nan")
    mean_l, mean_r = statistics.fmean(left), statistics.fmean(right)
    num = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    den_l = sum((a - mean_l) ** 2 for a in left) ** 0.5
    den_r = sum((b - mean_r) ** 2 for b in right) ** 0.5
    if den_l == 0 or den_r == 0:
        return float("nan")
    return num / (den_l * den_r)
