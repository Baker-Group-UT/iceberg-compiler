"""
multilevel.py — partition first, then align slots.

Two stages, deliberately separate, because they answer different questions and the
answers matter to different degrees in different codes:

*partition*  which qubits share a patch. Minimise cut weight subject to at most k
             per patch. This is balanced k-way graph partitioning.
*align*      which lane each qubit occupies. Transversal macros pair lane i with
             lane i, so for every interacting pair of patches there is a best slot
             correspondence. Solved exactly.

Keeping them apart means a code where inter-block operations are expensive (surface
code, most LDPC families) leans on stage one, and a code where they are cheap --
like Iceberg, where inter-patch CNOT is *cheaper* than intra-patch -- leans on stage
two. Neither has to be rewritten to serve the other.

Why multilevel
--------------
Direct search on the placement is O(m^2) neighbours per round and does not survive
past ~50 qubits. Multilevel partitioning is the standard answer and scales to
millions of nodes:

    coarsen     repeatedly contract strongly-connected pairs into super-nodes, each
                carrying the number of qubits it stands for, until the graph is
                small. Contraction never exceeds capacity k, so every coarse node is
                a legal patch.
    solve       partition the small graph. Cheap enough to be careful with.
    uncoarsen   expand one level at a time, running a bounded boundary-refinement
                pass after each expansion. Starting from a good coarse answer means
                only local fixes are needed.

The big decisions are taken on a small problem, which is the whole trick.

Signed weights
--------------
Unlike the classical formulation, edges here can be negative: a pair of qubits whose
gates are *cheaper* apart should be separated. Coarsening only ever contracts
positive edges, and refinement gains are computed on the signed weights directly, so
a repelling pair is never glued together and is actively pushed apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from iceberg.utils import logical_capacity

from packing.graph import InteractionGraph

__all__ = ["PartitionResult", "partition", "align_slots", "place"]

Placement = List[Tuple[int, int]]


@dataclass
class PartitionResult:
    groups: List[List[int]]
    n_code: int
    cut_weight: float
    levels: int = 0
    refinement_moves: int = 0

    @property
    def k(self) -> int:
        return logical_capacity(self.n_code)

    @property
    def num_patches(self) -> int:
        return len(self.groups)

    def summary(self) -> str:
        sizes = sorted((len(g) for g in self.groups), reverse=True)
        return (
            f"{self.num_patches} patches, sizes {sizes}, cut {self.cut_weight:.0f}, "
            f"{self.levels} levels, {self.refinement_moves} refinement moves"
        )


# --------------------------------------------------------------------------
# coarsening
# --------------------------------------------------------------------------


def _heavy_edge_matching(
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    capacity: int,
) -> Dict[int, int]:
    """
    Pair up nodes along their heaviest attracting edge.

    Greedy over edges sorted by weight, skipping any contraction that would exceed
    capacity and any edge that is not attracting. Returns node -> super-node id.
    """
    order = sorted(
        (w for w in weights.items() if w[1] > 0), key=lambda item: -item[1]
    )
    matched: Dict[int, int] = {}
    taken: Set[int] = set()
    next_id = 0
    for (left, right), _ in order:
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


# --------------------------------------------------------------------------
# base case and refinement
# --------------------------------------------------------------------------


def _greedy_partition(
    nodes: Sequence[int],
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    capacity: int,
    max_patches: Optional[int],
) -> List[List[int]]:
    """
    Seed partition: repeatedly merge the best-scoring admissible pair of groups.

    Score is the summed weight between them, so attracting groups merge and
    repelling ones do not. Stops when no positive merge remains, unless a patch
    budget forces more.
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


def _refine(
    groups: List[List[int]],
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    capacity: int,
    max_passes: int = 4,
) -> int:
    """
    Boundary refinement: move a node to another patch when that lowers the cut.

    Only nodes with at least one edge leaving their patch are considered, which is
    the Fiduccia-Mattheyses observation that interior nodes cannot improve anything.
    Bounded passes, since each is O(boundary x patches).
    """
    where = {node: index for index, group in enumerate(groups) for node in group}
    moves = 0

    for _ in range(max_passes):
        improved = False
        for node in list(where):
            home = where[node]
            if len(groups[home]) <= 1:
                continue
            # gain(target) = weight kept inside target - weight lost from home
            attachment: Dict[int, float] = {}
            for other, index in where.items():
                if other == node:
                    continue
                weight = weights.get((min(node, other), max(node, other)), 0.0)
                if weight:
                    attachment[index] = attachment.get(index, 0.0) + weight
            here = attachment.get(home, 0.0)
            best_target, best_gain = None, 0.0
            for target, pull in attachment.items():
                if target == home:
                    continue
                if sum(sizes.get(n, 1) for n in groups[target]) + sizes.get(node, 1) > capacity:
                    continue
                gain = pull - here
                if gain > best_gain:
                    best_target, best_gain = target, gain
            if best_target is None:
                continue
            groups[home].remove(node)
            groups[best_target].append(node)
            where[node] = best_target
            moves += 1
            improved = True
        groups[:] = [g for g in groups if g]
        where = {node: index for index, group in enumerate(groups) for node in group}
        if not improved:
            break
    return moves


def _consolidate(
    groups: List[List[int]],
    weights: Mapping[Tuple[int, int], float],
    sizes: Mapping[int, int],
    capacity: int,
    max_patches: int,
) -> None:
    """
    Pack groups together until the patch budget is met.

    Greedy merging stalls whenever every remaining group is over half capacity --
    at k=10, two groups of six cannot join even though a different split would have
    fit. Measured on QASMBench, that left `bv_n140` on 18 patches against a budget of
    14, and since every patch pays for its own encode and decode the placement came
    out 16% *worse* than naive.

    So finish with explicit bin packing: take the smallest group and put it in the
    admissible group it interacts with most, largest-first. Purely a repair pass --
    it runs after the cut has been optimised and only when the budget is missed.
    """
    def group_size(group: Sequence[int]) -> int:
        return sum(sizes.get(node, 1) for node in group)

    def affinity(a: Sequence[int], b: Sequence[int]) -> float:
        return sum(
            weights.get((min(x, y), max(x, y)), 0.0) for x in a for y in b
        )

    while len(groups) > max_patches:
        groups.sort(key=group_size)
        moved = False
        for index, group in enumerate(groups):
            candidates = [
                other
                for other in range(len(groups))
                if other != index
                and group_size(groups[other]) + group_size(group) <= capacity
            ]
            if not candidates:
                continue
            # Highest affinity, then emptiest, so packing stays balanced.
            target = max(
                candidates,
                key=lambda other: (affinity(group, groups[other]), -group_size(groups[other])),
            )
            groups[target].extend(group)
            groups.pop(index)
            moved = True
            break
        if not moved:
            break  # genuinely cannot pack tighter at this capacity


def cut_weight(groups: Sequence[Sequence[int]], graph: InteractionGraph) -> float:
    """Total weight of edges running between different patches."""
    where = {node: index for index, group in enumerate(groups) for node in group}
    total = 0.0
    for (left, right), weight in graph.weights.items():
        if where.get(left) != where.get(right):
            total += weight
    return total


# --------------------------------------------------------------------------
# the partitioner
# --------------------------------------------------------------------------


def partition(
    graph: InteractionGraph,
    n_code: int,
    *,
    max_patches: Optional[int] = None,
    coarsen_to: int = 24,
    max_levels: int = 24,
) -> PartitionResult:
    """
    Capacity-constrained partitioning by coarsen / solve / uncoarsen.

    `max_patches` is a target rather than a guarantee -- greedy merging can strand
    clusters that are each over half capacity, so check `num_patches` if the budget
    is hard. Patches are *not* required to be full: leaving a qubit alone is often
    right, and `plan.py` searches over how many patches to spend.
    """
    capacity = logical_capacity(n_code)
    weights = dict(graph.weights)
    sizes = {node: 1 for node in range(graph.num_qubits)}

    history: List[Dict[int, int]] = []
    level_sizes: List[Dict[int, int]] = [dict(sizes)]
    level_weights: List[Dict[Tuple[int, int], float]] = [dict(weights)]

    # --- coarsen ---
    while len(sizes) > coarsen_to and len(history) < max_levels:
        mapping = _heavy_edge_matching(weights, sizes, capacity)
        if len(set(mapping.values())) == len(sizes):
            break  # nothing contracted; further levels would not shrink
        weights, sizes = _contract(weights, sizes, mapping)
        history.append(mapping)
        level_sizes.append(dict(sizes))
        level_weights.append(dict(weights))

    # --- solve on the coarsest graph ---
    groups = _greedy_partition(
        list(sizes), weights, sizes, capacity, max_patches
    )
    moves = _refine(groups, weights, sizes, capacity)

    # --- uncoarsen, refining at each level ---
    for level in range(len(history) - 1, -1, -1):
        mapping = history[level]
        expanded: List[List[int]] = []
        for group in groups:
            members: List[int] = []
            for coarse_node in group:
                members.extend(
                    node for node, target in mapping.items() if target == coarse_node
                )
            expanded.append(members)
        groups = expanded
        moves += _refine(groups, level_weights[level], level_sizes[level], capacity)

    # After uncoarsening the groups hold original qubit ids again, so every node
    # weighs one -- `sizes` at this point still describes the coarsest level.
    unit_sizes = {node: 1 for node in range(graph.num_qubits)}
    if max_patches is not None and len(groups) > max_patches:
        _consolidate(groups, graph.weights, unit_sizes, capacity, max_patches)
        moves += _refine(groups, graph.weights, unit_sizes, capacity)

    groups = [sorted(group) for group in groups if group]
    groups.sort(key=lambda group: group[0])
    return PartitionResult(
        groups=groups,
        n_code=n_code,
        cut_weight=cut_weight(groups, graph),
        levels=len(history),
        refinement_moves=moves,
    )


# --------------------------------------------------------------------------
# slot alignment
# --------------------------------------------------------------------------


def align_slots(
    groups: Sequence[Sequence[int]],
    profiles: Mapping,
    n_code: int,
    *,
    window: int = 2,
) -> Placement:
    """
    Choose each qubit's lane so transversal macros line up.

    A transversal macro pairs lane i of one patch with lane i of the other, so for a
    pair of patches the question is which qubit faces which -- an assignment problem
    over k slots, solved exactly by the Hungarian algorithm in O(k^3). With k <= 10
    that is free, and unlike the local search it cannot get stuck.

    Patches are processed in order of how much cross traffic they carry, each pinned
    against the already-placed patches it interacts with most. Greedy across pairs
    (the joint problem over all patches at once is quadratic assignment, which is
    NP-hard) but exact within each pair, which is where the structure is.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    k = logical_capacity(n_code)
    patch_of: Dict[int, int] = {
        qubit: index for index, group in enumerate(groups) for qubit in group
    }

    # How much each pair of patches interacts, so the heaviest pair is aligned first.
    traffic: Dict[Tuple[int, int], int] = {}
    partners: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for qubit, profile in profiles.items():
        if qubit not in patch_of:
            continue
        for event in profile.two_qubit_events():
            other = event.partner
            if other is None or other not in patch_of or other <= qubit:
                continue
            a, b = patch_of[qubit], patch_of[other]
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            traffic[key] = traffic.get(key, 0) + 1
            partners.setdefault(key, []).append(
                (qubit, other, event.layer) if a < b else (other, qubit, event.layer)
            )

    slots: Dict[int, int] = {}
    placed: Set[int] = set()
    for (left, right), _ in sorted(traffic.items(), key=lambda item: -item[1]):
        for patch in (left, right):
            if patch in placed:
                continue
            anchor = right if patch == left else left
            if anchor not in placed:
                # First patch of the pair: fix it in ascending qubit order and let
                # the partner align against it.
                for slot, qubit in enumerate(sorted(groups[patch])):
                    slots[qubit] = slot
                placed.add(patch)
                continue

            members = sorted(groups[patch])
            anchor_members = sorted(groups[anchor])
            benefit = np.zeros((len(members), k))
            for row, qubit in enumerate(members):
                for partner_qubit, layer in _partners_of(qubit, partners[(left, right)]):
                    if partner_qubit not in slots:
                        continue
                    del layer
                    benefit[row, slots[partner_qubit]] += 1.0
            del anchor_members
            rows, cols = linear_sum_assignment(-benefit)
            for row, col in zip(rows, cols):
                slots[members[row]] = int(col)
            placed.add(patch)

    # Patches with no cross traffic, and any qubit missed above.
    for index, group in enumerate(groups):
        used = {slots[q] for q in group if q in slots}
        free = [slot for slot in range(k) if slot not in used]
        for qubit in sorted(group):
            if qubit not in slots:
                slots[qubit] = free.pop(0)

    total = sum(len(group) for group in groups)
    placement: Placement = [(0, 0)] * total
    for index, group in enumerate(groups):
        for qubit in group:
            placement[qubit] = (index, slots[qubit])
    return placement


def _partners_of(qubit: int, records: Sequence[Tuple[int, int, int]]):
    for left, right, layer in records:
        if left == qubit:
            yield right, layer
        elif right == qubit:
            yield left, layer


def place(
    graph: InteractionGraph,
    profiles: Mapping,
    n_code: int,
    *,
    max_patches: Optional[int] = None,
) -> Tuple[Placement, PartitionResult]:
    """Partition, then align. The whole pipeline in one call."""
    result = partition(graph, n_code, max_patches=max_patches)
    return align_slots(result.groups, profiles, n_code), result
