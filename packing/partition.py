"""
partition.py — multilevel partitioning where capacity is an invariant, not a repair.

This replaces `grouping.multilevel.partition` for seeding. Same shape -- coarsen,
solve, uncoarsen with refinement -- and the same objective, cut weight on the signed
graph, because a smooth pairwise relaxation is a better *search guide* than the
conjunctive true cost even though it is a worse description of it (see `seed`). Two
things are different, and both came out of watching the old one fail.

Feasibility is maintained, not repaired
--------------------------------------
The old base case merged by affinity until no admissible merge remained, and could
strand itself: on `bv_n70` at k=10 it reached sizes `[10,8,8,8,8,8,8,8,4]`, where
every pair exceeds capacity, so it stopped at 9 groups against a target of 7.
`_consolidate` then existed purely to bin-pack the wreckage afterwards -- and before
it was written, `bv_n140` came out 16% *worse* than naive for exactly this reason.

The cause is upstream: coarsening built 4-qubit super-nodes, and 4 does not tile 10.
So this coarsener only ever matches nodes of *equal* size, and caps that size at the
largest power of two dividing k. Every node size is then a power of two that divides
k, which makes packing provably safe:

    process nodes largest-first into `ceil(m/k)` bins of capacity k. If some node of
    size s does not fit, every bin has remaining < s. All sizes are powers of two and
    all placed so far are >= s, so each bin's load is a multiple of s, so each
    remaining is a multiple of s, so each remaining is 0 -- every bin is full. Then
    m > target * k, contradicting target = ceil(m/k).

So `pack_bins` cannot fail, there is nothing to consolidate, and the bin count is
fixed up front rather than emerging from a merge loop that has no way to aim at it.

    k:    2   4   6   8  10  12
    cap:  2   4   2   8   2   4

For k = 10 that means one coarsening level instead of two. The base case runs on m/2
nodes rather than m/4, which is affordable because it is now O(N * bins * k) bin
packing rather than the old O(N^3) merge loop.

Nodes can be exchanged, not only relocated -- and that turned out to be a mistake
--------------------------------------------------------------------------------
`multilevel._refine` only ever *moves* a node, and only into a group with spare
capacity. At the tightest packing there is none, so the pass is inert -- 0 moves on
60 of 63 QASMBench circuits at k=4. An exchange of two equal-size nodes is
capacity-neutral by construction, so it is always admissible, and at coarse levels it
relocates a whole block in one step, which single-qubit moves cannot reach because the
intermediate states are worse.

All of that is true and it still measured *worse*: 0.6 points at k=6 and 0.8 at k=10.
The reason is instructive. Exchange minimises the **cut**, and the cut is a loose
proxy for cost -- where FM fires it moves the cut 1.5% and the true cost 10-20%. So a
stronger cut minimiser does not find a better placement, it overfits the proxy. The
seed's job is global structure; `search.descend` is what optimises the real objective.

So `swap` defaults to False, and is kept because a recorded negative result is cheaper
than rediscovering it. Fiduccia-Mattheyses replaced Kernighan-Lin's pair exchange with
single-node moves to get linear time; under a hard capacity bound the exchange becomes
admissible again, but it turns out not to be *wanted*. Implemented with FM's
incremental gain bookkeeping either way, so a pass stays O(N^2).

`target` is a ceiling unless `exact` is set
------------------------------------------
By default `target` bounds the patch count from above and nothing holds it from below:
the merge base keeps merging while any pair attracts, `pack_bins` lets a bin stay empty
when no node prefers it, refinement may relocate a group's last node, and the empty
groups are dropped at the end. Measured on the standard planner's silent-aware graph for 29
of the density sweep's circuits, at every patch count above the tightest, the default
`feasible` partitioner returned exactly `target` groups in 35 of 105 cases and `packed` in 4.

`exact=True` holds the count at every stage instead. A coarsening level contracts at
most `len(nodes) - target` pairs, heaviest first, so the coarse graph never has fewer
nodes than groups wanted. The merge base stops at `target`. `pack_bins` opens an empty
bin whenever the nodes left are no more than the bins still empty. Refinement never moves
a group's only node. At the tightest count none of these can bind -- capacity already
forbids every move that would change the count -- so `exact` returns exactly the default
partition there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from iceberg.utils import logical_capacity

__all__ = [
    "PartitionResult",
    "coarsen_cap",
    "cut_weight",
    "pack_bins",
    "partition",
    "refine",
]

Pair = Tuple[int, int]
Weights = Mapping[Pair, float]


# ----------------------------------------------------------------------
# ported from the multilevel partitioner this package replaced
# ----------------------------------------------------------------------

def _legacy_matching(
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    capacity: int,
    limit: Optional[int] = None,
) -> Dict[int, int]:
    """
    Pair up nodes along their heaviest attracting edge.

    Greedy over edges sorted by weight, skipping any contraction that would exceed
    capacity and any edge that is not attracting. At most `limit` pairs when given.
    Returns node -> super-node id.
    """
    order = sorted(
        (w for w in weights.items() if w[1] > 0), key=lambda item: -item[1]
    )
    matched: Dict[int, int] = {}
    taken: Set[int] = set()
    next_id = 0
    for (left, right), _ in order:
        if limit is not None and next_id >= limit:
            break
        if left in taken or right in taken:
            continue
        if sizes.get(left, 1) + sizes.get(right, 1) > capacity:
            continue
        matched[left] = matched[right] = next_id
        taken.add(left)
        taken.add(right)
        next_id += 1
    for node in sizes:
        if node not in matched:
            matched[node] = next_id
            next_id += 1
    return matched


def _contract(
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    mapping: Mapping[int, int],
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, int]]:
    """Collapse nodes by `mapping`, summing parallel edges."""
    coarse: Dict[Tuple[int, int], float] = {}
    for (left, right), weight in weights.items():
        a, b = mapping[left], mapping[right]
        if a == b:
            continue  # internal to a super-node; no longer a cut candidate
        key = (min(a, b), max(a, b))
        coarse[key] = coarse.get(key, 0.0) + weight
    coarse_sizes: Dict[int, int] = {}
    for node, size in sizes.items():
        coarse_sizes[mapping[node]] = coarse_sizes.get(mapping[node], 0) + size
    return coarse, coarse_sizes


def _legacy_base(
    nodes: Sequence[int],
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    capacity: int,
    max_patches: Optional[int],
    min_patches: Optional[int] = None,
) -> List[List[int]]:
    """
    Seed partition: repeatedly merge the best-scoring admissible pair of groups.

    Score is the summed weight between them, so attracting groups merge and
    repelling ones do not. Stops when no positive merge remains, unless a patch
    budget forces more, and never merges below `min_patches` groups.
    """
    groups: List[List[int]] = [[node] for node in sorted(nodes)]
    group_size = {index: sizes.get(node, 1) for index, node in enumerate(sorted(nodes))}

    def between(a: Sequence[int], b: Sequence[int]) -> float:
        total = 0.0
        for left in a:
            for right in b:
                total += weights.get((min(left, right), max(left, right)), 0.0)
        return total

    while len(groups) > 1:
        if min_patches is not None and len(groups) <= min_patches:
            break
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if group_size[i] + group_size[j] > capacity:
                    continue
                score = between(groups[i], groups[j])
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is None:
            break
        score, i, j = best
        forced = max_patches is not None and len(groups) > max_patches
        if score <= 0 and not forced:
            break
        groups[i] = groups[i] + groups[j]
        group_size[i] += group_size[j]
        groups.pop(j)
        group_size = {index: group_size[index if index < j else index + 1]
                      for index in range(len(groups))}
    return groups


def coarsen_cap(k: int) -> int:
    """
    Largest power of two dividing `k` -- the biggest super-node that still tiles a patch.

    Capping here is what makes `pack_bins` safe. A 4-qubit block cannot tile a patch of
    10, so allowing one guarantees the leftovers the old `_consolidate` had to clean up.
    """
    cap = 1
    while k % (cap * 2) == 0:
        cap *= 2
    return cap


def _weight(weights: Weights, left: int, right: int) -> float:
    return weights.get((left, right) if left < right else (right, left), 0.0)


def cut_weight(groups: Sequence[Sequence[int]], weights: Weights) -> float:
    """Total weight of edges running between different groups."""
    where = {node: index for index, group in enumerate(groups) for node in group}
    total = 0.0
    for (left, right), weight in weights.items():
        if where.get(left) != where.get(right):
            total += weight
    return total


# ----------------------------------------------------------------------
# coarsening
# ----------------------------------------------------------------------


def _matching(
    weights: Weights, sizes: Mapping[int, int], cap: int, limit: Optional[int] = None
) -> Dict[int, int]:
    """
    Pair nodes along their heaviest attracting edge, equal sizes only.

    Three filters, each load-bearing: only attracting edges (a repelling pair must
    never be glued together), each node at most once (so the level halves, which is
    what bounds the level count by log2(cap)), and equal sizes capped at `cap` (so
    every size stays a power of two dividing k -- the invariant `pack_bins` needs).
    `limit` stops after that many pairs, the heaviest, which is how `exact` keeps a
    level from contracting below the group count it has to reach.
    """
    order = sorted(
        ((weight, edge) for edge, weight in weights.items() if weight > 0),
        key=lambda item: -item[0],
    )
    matched: Dict[int, int] = {}
    taken: Set[int] = set()
    next_id = 0
    for _, (left, right) in order:
        if limit is not None and next_id >= limit:
            break
        if left in taken or right in taken:
            continue
        size = sizes.get(left, 1)
        if size != sizes.get(right, 1) or size * 2 > cap:
            continue
        matched[left] = matched[right] = next_id
        taken.add(left)
        taken.add(right)
        next_id += 1
    for node in sizes:
        if node not in matched:
            matched[node] = next_id
            next_id += 1
    return matched


# ----------------------------------------------------------------------
# base case
# ----------------------------------------------------------------------


def pack_bins(
    nodes: Sequence[int],
    weights: Weights,
    sizes: Mapping[int, int],
    capacity: int,
    target: int,
    *,
    fill: bool = False,
) -> List[List[int]]:
    """
    Assign nodes into exactly `target` bins, largest node first.

    Affinity chooses the bin; a tight fit breaks ties. Fixing the bin count up front
    is the point -- the old merge loop had no way to aim at a count, which is why five
    scanned candidates returned one partition on 41 of 63 circuits and why
    `_consolidate` had to exist. Provably never overflows given the size invariant
    from `_matching`; the emptiest-bin fallback is a guard for callers that supply
    sizes violating it, not a path the coarsener can reach.

    Fixing the bin count does not fix how many are *occupied*: a tight fit breaking ties
    fills the bins already started, so above the tightest count most bins stay empty.
    `fill` occupies every bin -- once the nodes left are no more than the bins still
    empty, each of them opens one. An empty bin holds any node, so the overflow
    argument is untouched.
    """
    bins: List[List[int]] = [[] for _ in range(max(1, target))]
    load = [0] * len(bins)
    ordered = sorted(nodes, key=lambda n: -sizes.get(n, 1))
    for position, node in enumerate(ordered):
        size = sizes.get(node, 1)
        must_open = fill and len(ordered) - position <= load.count(0)
        best: Optional[Tuple[Tuple[float, int], int]] = None
        for index, members in enumerate(bins):
            if load[index] + size > capacity:
                continue
            if must_open and load[index]:
                continue
            affinity = sum(_weight(weights, node, other) for other in members)
            key = (affinity, load[index])
            if best is None or key > best[0]:
                best = (key, index)
        chosen = best[1] if best is not None else min(
            range(len(bins)), key=lambda i: load[i]
        )
        bins[chosen].append(node)
        load[chosen] += size
    return bins


# ----------------------------------------------------------------------
# refinement
# ----------------------------------------------------------------------


@dataclass
class _State:
    """Group membership plus the attachment table the gains are read from."""

    groups: List[List[int]]
    where: Dict[int, int]
    load: List[int]
    #: node -> group index -> summed weight from that node to that group's members,
    #: always excluding the node itself.
    attach: Dict[int, Dict[int, float]]


def _build_state(
    groups: Sequence[Sequence[int]], weights: Weights, sizes: Mapping[int, int]
) -> _State:
    live = [list(group) for group in groups]
    where = {node: index for index, group in enumerate(live) for node in group}
    load = [sum(sizes.get(n, 1) for n in group) for group in live]
    attach: Dict[int, Dict[int, float]] = {node: {} for node in where}
    for (left, right), weight in weights.items():
        if not weight:
            continue
        here, there = where.get(left), where.get(right)
        if here is None or there is None:
            continue
        attach[left][there] = attach[left].get(there, 0.0) + weight
        attach[right][here] = attach[right].get(here, 0.0) + weight
    return _State(groups=live, where=where, load=load, attach=attach)


def _rehome(state: _State, weights: Weights, node: int, old: int, new: int) -> None:
    """Tell every other node's attachment table that `node` changed group."""
    for other in state.where:
        if other == node:
            continue
        weight = _weight(weights, node, other)
        if weight:
            table = state.attach[other]
            table[old] = table.get(old, 0.0) - weight
            table[new] = table.get(new, 0.0) + weight


def refine(
    groups: List[List[int]],
    weights: Weights,
    sizes: Mapping[int, int],
    capacity: int,
    *,
    max_passes: int = 4,
    swap: bool = True,
    trace: Optional[List[Tuple[str, float]]] = None,
    keep_occupied: bool = False,
) -> Tuple[int, int]:
    """
    Lower the cut by relocating or exchanging nodes. Returns (relocations, exchanges).

    `keep_occupied` never relocates a group's only node, so the number of occupied groups
    is invariant (a relocation never targets an empty group, and an exchange is neutral).

    Gains, both exact and both read straight off the attachment table:

        relocate v from A to B     att[v][B] - att[v][A]

        exchange a in A, b in B    att[a][B] + att[b][A] - 2*w(a,b)
                                     - att[a][A] - att[b][B]

    The `2*w(a,b)` matters and is easy to miss. `att[a][B]` counts a's weight to *all*
    of B including b, but after the exchange b is no longer there -- and symmetrically
    for `att[b][A]`. The a-b edge is cut before and after, so it must not be credited
    to either side. `tests/test_packing_partition.py` checks every accepted move's
    predicted gain against a from-scratch recomputation of the cut.
    """
    state = _build_state(groups, weights, sizes)
    relocations = exchanges = 0

    for _ in range(max_passes):
        improved = False
        for node in list(state.where):
            home = state.where[node]
            table = state.attach[node]
            here = table.get(home, 0.0)
            size = sizes.get(node, 1)
            best: Optional[Tuple[float, str, int]] = None
            alone = keep_occupied and len(state.groups[home]) == 1

            for target, pull in (() if alone else table.items()):
                if target == home or not state.groups[target]:
                    continue
                if state.load[target] + size > capacity:
                    continue
                gain = pull - here
                if gain > 1e-12 and (best is None or gain > best[0]):
                    best = (gain, "move", target)

            for other, away in (state.where.items() if swap else ()):
                if away == home:
                    continue
                other_size = sizes.get(other, 1)
                if state.load[away] - other_size + size > capacity:
                    continue
                if state.load[home] - size + other_size > capacity:
                    continue
                shared = _weight(weights, node, other)
                partner = state.attach[other]
                gain = (
                    table.get(away, 0.0)
                    + partner.get(home, 0.0)
                    - 2.0 * shared
                    - here
                    - partner.get(away, 0.0)
                )
                if gain > 1e-12 and (best is None or gain > best[0]):
                    best = (gain, "swap", other)

            if best is None:
                continue
            gain, kind, payload = best
            if trace is not None:
                trace.append((kind, gain))
            if kind == "move":
                target = payload
                _rehome(state, weights, node, home, target)
                state.groups[home].remove(node)
                state.groups[target].append(node)
                state.load[home] -= size
                state.load[target] += size
                state.where[node] = target
                relocations += 1
            else:
                other = payload
                away = state.where[other]
                other_size = sizes.get(other, 1)
                _rehome(state, weights, node, home, away)
                _rehome(state, weights, other, away, home)
                state.groups[home].remove(node)
                state.groups[home].append(other)
                state.groups[away].remove(other)
                state.groups[away].append(node)
                state.load[home] += other_size - size
                state.load[away] += size - other_size
                state.where[node], state.where[other] = away, home
                exchanges += 1
            improved = True
        if not improved:
            break

    groups[:] = state.groups
    return relocations, exchanges


# ----------------------------------------------------------------------
# the partitioner
# ----------------------------------------------------------------------


@dataclass
class PartitionResult:
    groups: List[List[int]]
    n_code: int
    cut_weight: float
    levels: int = 0
    relocations: int = 0
    exchanges: int = 0

    @property
    def num_patches(self) -> int:
        return len(self.groups)

    def summary(self) -> str:
        sizes = sorted((len(g) for g in self.groups), reverse=True)
        return (
            f"{self.num_patches} patches, sizes {sizes}, cut {self.cut_weight:.0f}, "
            f"{self.levels} levels, {self.relocations} relocations, "
            f"{self.exchanges} exchanges"
        )


def partition(
    graph,
    n_code: int,
    target: int,
    *,
    coarsen_to: int = 24,
    max_levels: int = 24,
    max_passes: int = 4,
    matching: str = "equal",
    base: str = "pack",
    swap: bool = True,
    exact: bool = False,
) -> PartitionResult:
    """
    Coarsen, solve the small graph, then uncoarsen refining at each level.

    `exact` returns exactly `target` occupied groups rather than at most `target` (see the
    module docstring), and requires ceil(m / k) <= target <= m for m qubits.

    Three knobs, because two of these ideas measured well and one did not, and the
    switch is how that stays checkable rather than remembered:

    `matching`  ``"equal"`` pairs only equal-size nodes and caps the result at
                `coarsen_cap(k)`, which is what makes `pack_bins` provably safe.
                ``"any"`` is `grouping.multilevel`'s heavy-edge matching, which packs
                more per level but produces sizes that need not tile a patch.
    `base`      ``"pack"`` is `pack_bins`, which honours `target` exactly.
                ``"merge"`` is `grouping.multilevel`'s agglomerative merge, which picks
                the globally best pair each step -- better structure, but it can stall
                above `target` and then needs a repair pass.
    `swap`      whether refinement may exchange nodes as well as relocate them.

    With `base="merge"` the count is not guaranteed, so `pack_bins` runs as a repair on
    the stalled groups' own nodes -- the same job `_consolidate` did, but as bin packing
    with the size invariant behind it rather than smallest-first merging.
    """
    capacity = logical_capacity(n_code)
    if exact:
        lowest = max(1, -(-graph.num_qubits // capacity))
        if not lowest <= target <= max(1, graph.num_qubits):
            raise ValueError(
                f"{graph.num_qubits} qubits in patches of {capacity} need between "
                f"{lowest} and {graph.num_qubits} groups; an exact partition into "
                f"{target} does not exist"
            )
    cap = min(coarsen_cap(capacity), capacity)
    weights: Dict[Pair, float] = dict(graph.weights)
    sizes: Dict[int, int] = {node: 1 for node in range(graph.num_qubits)}

    history: List[Dict[int, int]] = []
    level_sizes: List[Dict[int, int]] = [dict(sizes)]
    level_weights: List[Dict[Pair, float]] = [dict(weights)]

    while len(sizes) > max(coarsen_to, target) and len(history) < max_levels:
        # Each contracted pair ends up in one group, so an exact partition can afford
        # only `len(sizes) - target` of them per level.
        limit = len(sizes) - target if exact else None
        if matching == "any":
            mapping = _legacy_matching(weights, sizes, capacity, limit=limit)
        else:
            mapping = _matching(weights, sizes, cap, limit=limit)
        if len(set(mapping.values())) == len(sizes):
            break  # nothing contracted; another level would not shrink
        weights, sizes = _contract(weights, sizes, mapping)
        history.append(mapping)
        level_sizes.append(dict(sizes))
        level_weights.append(dict(weights))

    if base == "merge":
        groups = _legacy_base(
            list(sizes), weights, sizes, capacity, target,
            min_patches=target if exact else None,
        )
        if len(groups) > target:
            # Stalled above the budget. Redistribute the same nodes into exactly
            # `target` bins; safe because node sizes still respect `cap`.
            groups = pack_bins(
                [node for group in groups for node in group],
                weights, sizes, capacity, target, fill=exact,
            )
    else:
        groups = pack_bins(list(sizes), weights, sizes, capacity, target, fill=exact)
    relocations, exchanges = refine(
        groups, weights, sizes, capacity, max_passes=max_passes, swap=swap,
        keep_occupied=exact,
    )

    for level in range(len(history) - 1, -1, -1):
        mapping = history[level]
        expanded: List[List[int]] = [[] for _ in groups]
        for index, group in enumerate(groups):
            members = set(group)
            expanded[index] = [
                node for node, coarse in mapping.items() if coarse in members
            ]
        groups = expanded
        moved, swapped = refine(
            groups,
            level_weights[level],
            level_sizes[level],
            capacity,
            max_passes=max_passes,
            swap=swap,
            keep_occupied=exact,
        )
        relocations += moved
        exchanges += swapped

    live = [sorted(group) for group in groups if group]
    live.sort(key=lambda group: group[0])
    if exact and len(live) != target:
        raise RuntimeError(
            f"exact partition produced {len(live)} groups against a target of {target}"
        )
    return PartitionResult(
        groups=live,
        n_code=n_code,
        cut_weight=cut_weight(live, graph.weights),
        levels=len(history),
        relocations=relocations,
        exchanges=exchanges,
    )
