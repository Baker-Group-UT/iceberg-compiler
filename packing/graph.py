"""
graph.py — the interaction graph the partitioner actually wants.

`agglomerative._interaction_linkage` counts two-qubit gates crossing a patch
boundary and treats every crossing as equally bad. That is a reasonable first guess
and it measured well (r = +0.80 against cost), but it is wrong in a specific and
fixable way: **not every gate wants its operands together.**

Signed weights, over gates *and* depth
-------------------------------------
An edge's weight is what co-locating its endpoints saves, priced on both axes::

    weight(a, b) = [gate_cost(cross) - gate_cost(intra)]
                 + depth_weight * [depth(cross) - depth(intra)]

Positive pulls two qubits together, negative pushes them apart. The two axes
disagree, and for CNOT they disagree in opposite directions (`iceberg.costs`):

    CX    gates: intra (0,4) == cross (0,4)   ->  0
          depth: intra 4     >  cross 3       -> -1 per gate, wants to SPLIT
    CZ    gates: intra (1,3) <  cross (0,4)   -> +12, wants to be together
          depth: intra 3     == cross 3       ->  0
    XCX   gates: intra (5,3) <  cross (4,4)   -> +12
          depth: intra 6     >  cross 5       -> -1

So on gate counts alone every CX edge weighs exactly zero, and a CNOT-heavy circuit
carries **no grouping signal at all** -- the graph is nearly empty and the partition
is arbitrary. That is not a modelling nicety: it is why cut-minimisation produced
negligible or negative packing benefit on the CX-dominated benchmarks (ghz, cat, bv,
adder). Counting every crossing equally, as the first version of this module did,
invents a preference the cost table does not have and points it the wrong way.

Transversal alignment is the prize, and it is not a clustering objective
-----------------------------------------------------------------------
A transversal macro is depth 1 and costs `(0, n)` while doing the work of k separate
cross-patch gates -- at n=6, 78 weighted units and depth 1 against 208 and depth 3
each. It wins on both axes at once. But it fires only when *all* k lanes of two
patches pair up, so what it rewards is not "interacting qubits together" but
"interacting qubits spread across two patches, one per lane, aligned".

`transversal_affinity` supplies that signal. Two qubits belong in the same patch when
their two-qubit gates point at *different* partners in the same layer -- then they can
be two lanes of one side of a macro. This is neighbourhood similarity rather than
adjacency, and it is the opposite of what a cut-minimising weight asks for.

Movement and idle noise
-----------------------
Neither appears in a gate table. `NoiseProfile` adds a per-crossing movement charge
and a per-idle-layer charge, which makes splitting less attractive on a machine where
shuttling atoms dominates. Set them from the hardware, not from taste.

Hadamard coincidence
--------------------
Without H commutation to sweep Hadamards away, the whole-patch HH macro is the
largest single saving available (depth 2 for all k lanes against 8 each), and it needs
an H on every occupied lane at once. That is a property of timelines, not of any gate
between two qubits, so an adjacency graph cannot see it.

`hadamard_affinity` adds positive edges between qubits whose Hadamards coincide in
time. Like the transversal term it is a pairwise stand-in for a conjunctive effect --
HH needs all k lanes, not two -- so it over-rewards pairs. The analytic cost model
applies the real all-lanes condition once a candidate placement exists; these terms
only have to get the ordering of candidate merges roughly right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from iceberg.costs import DEFAULT_WORK_WEIGHTS, depth_table, work_table

from .tables import CROSS_PATCH_COST, GATE_KEYS
from .profiles import QubitProfile

__all__ = [
    "DEFAULT_DEPTH_WEIGHT",
    "DEFAULT_TRANSVERSAL_CONFIDENCE",
    "NoiseProfile",
    "InteractionGraph",
    "build_graph",
    "hadamard_affinity",
    "transversal_affinity",
]

Edge = Tuple[int, int]


@dataclass(frozen=True)
class NoiseProfile:
    """
    What the hardware charges beyond gate counts.

    `movement_cost`  weighted-gate units per cross-patch two-qubit gate, standing in
                     for the round-trip atom movement it forces. The paper's
                     convention is 0 -- movement is not a gate -- which is right for
                     counting gates and wrong for predicting fidelity.
    `idle_cost`      units per qubit per layer of idling. Rises with circuit depth,
                     so it rewards placements that keep work parallel.
    `distance_scale` multiplies movement by patch distance when a layout is known.
                     Left at 1.0 the model is distance-agnostic.

    `idle_time_scalar` and `distance_scalar` are the raw machine knobs, carried
                     alongside the converted costs so a caller that needs to price a
                     *translated* circuit against the simulator's own schedule can do
                     so. `movement_cost` and `idle_cost` already fold them in, so
                     nothing in `cost.price` should read these -- doing that is what
                     applied `distance_scalar` twice once before. They exist for
                     `plan.measured_cost`, which reprices the real schedule rather than
                     the model, and which needs the knobs and not their conversion.
    """

    movement_cost: float = 0.0
    idle_cost: float = 0.0
    distance_scale: float = 1.0
    idle_time_scalar: float = 1.0
    distance_scalar: float = 1.0

    @classmethod
    def gate_counting(cls) -> "NoiseProfile":
        """The paper's convention: movement is free, only gates count."""
        return cls()

    @classmethod
    def movement_dominated(cls, movement_cost: float = 6.0) -> "NoiseProfile":
        """A machine where shuttling atoms, not gate error, sets the fidelity."""
        return cls(movement_cost=movement_cost)


@dataclass
class InteractionGraph:
    """Signed, weighted affinities between logical qubits."""

    weights: Dict[Edge, float]
    num_qubits: int

    def neighbours(self) -> Dict[int, set]:
        out: Dict[int, set] = {index: set() for index in range(self.num_qubits)}
        for (left, right) in self.weights:
            out[left].add(right)
            out[right].add(left)
        return out

    def weight(self, left: int, right: int) -> float:
        return self.weights.get((min(left, right), max(left, right)), 0.0)

    def positive_only(self) -> "InteractionGraph":
        """Drop the repulsive edges, for partitioners that require non-negative weights."""
        return InteractionGraph(
            {edge: w for edge, w in self.weights.items() if w > 0}, self.num_qubits
        )

    def summary(self) -> str:
        values = list(self.weights.values())
        if not values:
            return f"{self.num_qubits} qubits, no edges"
        attract = sum(1 for v in values if v > 0)
        repel = sum(1 for v in values if v < 0)
        return (
            f"{self.num_qubits} qubits, {len(values)} edges "
            f"({attract} attracting, {repel} repelling, "
            f"{len(values) - attract - repel} neutral), "
            f"total weight {sum(values):.0f}"
        )


#: How many weighted-gate units one layer of depth is worth.
#:
#: Not a free parameter to taste: it sets the exchange rate between the two things a
#: placement trades off, and the figures are sensitive to it. Calibrated so that the
#: depth difference on a CNOT (one layer) is worth roughly a quarter of a two-qubit
#: gate, which is enough to break the exact tie that gate counts leave on CX without
#: swamping the genuine gate savings on CZ and XCX. `evaluate_depth_weight` in
#: `tune.py` measures the alternatives rather than assuming this one.
DEFAULT_DEPTH_WEIGHT = 3.0

#: How much of a speculative transversal macro's per-lane gain to bank.
#:
#: A coincident pair is only 2/k of the way to a macro -- the other k-2 lanes must
#: align too -- so counting the full gain lets the term overwhelm savings that are
#: certain. Unconfident it did exactly that: on a circuit with two cleanly separated
#: CZ triples, full confidence pulled qubits *across* the blocks because their gates
#: shared layers, discarding an all-intra-patch placement that was strictly cheaper.
#:
#: Set from measurement, not taste: `tune.py` grid-searches this against translated
#: gate counts and depth on a benchmark sample.
DEFAULT_TRANSVERSAL_CONFIDENCE = 0.25


def build_graph(
    profiles: Mapping[int, QubitProfile],
    n_code: int,
    *,
    noise: Optional[NoiseProfile] = None,
    weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    transversal_confidence: float = DEFAULT_TRANSVERSAL_CONFIDENCE,
    include_hadamard_affinity: bool = True,
    include_transversal_affinity: bool = True,
    window: int = 2,
    num_qubits: Optional[int] = None,
) -> InteractionGraph:
    """
    Build the signed affinity graph for one program at one code size.

    Four effects, summed: what co-location saves in gates, what it saves in depth,
    what Hadamard coincidence would enable, and what transversal alignment would.
    The first two can be zero or negative, and on CNOT they cancel unless depth is
    priced -- see the module docstring.
    """
    noise = noise or NoiseProfile()
    table = work_table(n_code)
    depths = depth_table(n_code)
    w1, w2 = weights
    size = num_qubits if num_qubits is not None else (max(profiles) + 1 if profiles else 0)

    def scalar(cost: Tuple[int, int]) -> float:
        return w1 * cost[0] + w2 * cost[1]

    edges: Dict[Edge, float] = {}
    for qubit, profile in profiles.items():
        for event in profile.two_qubit_events():
            partner = event.partner
            if partner is None or partner <= qubit:
                continue  # each gate is seen from both ends; count it once
            key = GATE_KEYS.get(event.kind)
            if key is None:
                continue
            intra = table.get(key, (0, 0))
            cross = CROSS_PATCH_COST.get(key, intra)
            gate_gain = scalar(cross) - scalar(intra)
            depth_gain = depths.get(f"{key}_CROSS", depths.get(key, 0)) - depths.get(key, 0)
            gain = (
                gate_gain
                + depth_weight * depth_gain
                + noise.movement_cost * noise.distance_scale
            )
            edge = (qubit, partner)
            edges[edge] = edges.get(edge, 0.0) + gain

    if include_hadamard_affinity:
        for edge, gain in hadamard_affinity(
            profiles, n_code, weights=weights, window=window,
            depth_weight=depth_weight,
        ).items():
            edges[edge] = edges.get(edge, 0.0) + gain

    if include_transversal_affinity:
        for edge, gain in transversal_affinity(
            profiles, n_code, weights=weights, window=window,
            depth_weight=depth_weight, confidence=transversal_confidence,
        ).items():
            edges[edge] = edges.get(edge, 0.0) + gain

    return InteractionGraph(edges, size)


def transversal_affinity(
    profiles: Mapping[int, QubitProfile],
    n_code: int,
    *,
    weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    confidence: float = DEFAULT_TRANSVERSAL_CONFIDENCE,
    window: int = 2,
) -> Dict[Edge, float]:
    """
    Positive edges between qubits that could be two lanes of the same macro.

    A transversal macro needs both patches fully paired, one lane facing one lane. So
    two qubits belong *together* when their two-qubit gates, in the same layer and of
    the same kind, point at *different* partners: then they can sit on two lanes of
    one side while their partners sit on the matching lanes of the other.

    This is neighbourhood similarity, not adjacency, and it is the term a
    cut-minimising weight cannot express -- adjacency says "put a and b together"
    while this says "put a and b together *because* their partners can also be
    together". Both matter, and for CNOT only this one carries any signal.

    The reward is what one lane contributes to a macro: a cross-patch gate costs
    `c(gate)` and depth `d(gate)`, the macro costs `c(macro)/k` and `d(macro)/k` per
    lane, so joining is worth the difference.
    """
    table = work_table(n_code)
    depths = depth_table(n_code)
    k = max(1, n_code - 2)
    w1, w2 = weights

    def per_lane_gain(kind: str, macro: str) -> float:
        single = CROSS_PATCH_COST.get(kind, table.get(kind, (0, 0)))
        gate = (w1 * single[0] + w2 * single[1]) - (
            w1 * table[macro][0] + w2 * table[macro][1]
        ) / k
        depth = depths.get(f"{kind}_CROSS", depths[kind]) - depths[macro] / k
        # Speculative: the macro needs all k lanes, and this pair supplies two.
        return confidence * (gate + depth_weight * depth)

    gains = {
        "cx": per_lane_gain("CX", "CXT"),
        "cnot": per_lane_gain("CX", "CXT"),
        "cz": per_lane_gain("CZ", "CZT"),
        "xcx": per_lane_gain("XCX", "XCXT"),
    }

    # layer -> kind -> [(qubit, partner)]
    by_layer: Dict[int, Dict[str, list]] = {}
    for qubit, profile in profiles.items():
        for event in profile.two_qubit_events():
            if event.partner is None or event.kind not in gains:
                continue
            bucket = by_layer.setdefault(event.layer // max(1, window), {})
            bucket.setdefault(event.kind, []).append((qubit, event.partner))

    affinity: Dict[Edge, float] = {}
    for kinds in by_layer.values():
        for kind, pairs in kinds.items():
            gain = gains[kind]
            if gain <= 0:
                continue
            for index, (left, left_partner) in enumerate(pairs):
                for right, right_partner in pairs[index + 1 :]:
                    # Same-lane candidates only: distinct qubits with distinct
                    # partners, which is exactly what a bijection needs.
                    if len({left, right, left_partner, right_partner}) < 4:
                        continue
                    edge = (min(left, right), max(left, right))
                    affinity[edge] = affinity.get(edge, 0.0) + gain
                    partner_edge = (
                        min(left_partner, right_partner),
                        max(left_partner, right_partner),
                    )
                    affinity[partner_edge] = affinity.get(partner_edge, 0.0) + gain
    return affinity


def hadamard_affinity(
    profiles: Mapping[int, QubitProfile],
    n_code: int,
    *,
    weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    window: int = 2,
) -> Dict[Edge, float]:
    """
    Positive edges between qubits whose Hadamards land at the same time.

    The per-coincidence reward is what one lane contributes to a whole-patch macro:
    a targeted H costs `c(H)` and the macro costs `c(HH)/k` per lane, so the saving
    attributable to one lane joining is `c(H) - c(HH)/k`. Pairs that coincide often
    accumulate a large weight and get pulled together early in the partitioning.

    A pairwise stand-in for a conjunctive effect, so it over-rewards -- two qubits
    Hadamarding together do not by themselves make a macro at k > 2. The analytic
    cost model applies the real all-lanes condition once there is a placement to
    score; this only has to get the *ordering* of candidate merges roughly right.
    """
    table = work_table(n_code)
    depths = depth_table(n_code)
    k = max(1, n_code - 2)
    w1, w2 = weights
    # A targeted H costs c(H) and depth 8; HH costs c(HH)/k and depth 2/k per lane.
    # Depth is where the macro really wins, so it belongs in this term too.
    per_lane_gain = (
        (w1 * table["H"][0] + w2 * table["H"][1])
        - (w1 * table["HH"][0] + w2 * table["HH"][1]) / k
        + depth_weight * (depths["H"] - depths["HH"] / k)
    )
    if per_lane_gain <= 0:
        return {}

    layers_by_qubit = {
        qubit: sorted(profile.layers_with("h")) for qubit, profile in profiles.items()
    }
    qubits = [q for q, layers in layers_by_qubit.items() if layers]

    affinity: Dict[Edge, float] = {}
    for index, left in enumerate(qubits):
        for right in qubits[index + 1 :]:
            shared = _coincidences(layers_by_qubit[left], layers_by_qubit[right], window)
            if shared:
                affinity[(min(left, right), max(left, right))] = shared * per_lane_gain
    return affinity


def _coincidences(left: List[int], right: List[int], window: int) -> int:
    """
    How many layers the two lists share, within `window`, without reusing an entry.

    A two-pointer sweep, so building the affinity map stays linear in the number of
    Hadamards rather than quadratic.
    """
    i = j = matched = 0
    while i < len(left) and j < len(right):
        if abs(left[i] - right[j]) <= window:
            matched += 1
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return matched
