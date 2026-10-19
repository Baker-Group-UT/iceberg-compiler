"""
agglomerative.py — capacity-constrained agglomerative clustering of logical qubits.

Every qubit starts in its own patch. Repeatedly merge the two clusters with the best
linkage score, provided the union still fits in a patch (k = n - 2 slots). Stop when
no admissible merge has a positive score, or when the patch budget is reached.

Why this extends past pairs where the packer does not
----------------------------------------------------
The packer scores *pairs of qubits* and commits them, so nothing it computes ever
refers to a set of three or more. Here the linkage is evaluated on the **union of two
clusters**, which may be any size up to k. The algorithm is pairwise only in the
sense that it merges two things at a time; what it scores is the full group. So the
conjunctive whole-patch-Hadamard term and the super-additive Pauli term — the two
places a sum-of-pairs surrogate goes wrong — are seen exactly.

That also means the subset-enumeration problem never arises. Reaching a group of size
k takes k-1 merges, each considering O(m^2) cluster pairs, so a placement costs
O(m^3) linkage evaluations rather than the O(2^m) of enumerating groups.

Three linkages are provided so they can be compared rather than assumed:

`cost`         net gain from `cost_model.GroupCostModel`, in weighted gate units.
`similarity`   average pairwise `similarity.frame_similarity` across the union
               (average, not sum, so it does not simply favour large clusters).
`interaction`  cut weight -- two-qubit gates running between the two clusters. The
               measurements say this is the one that matters; see the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from iceberg.utils import logical_capacity

from .cost_model import GroupCostModel
from .profiles import QubitProfile, extract_profiles
from .similarity import DEFAULT_WEIGHTS, SimilarityWeights, frame_similarity

__all__ = ["GroupingResult", "cluster_qubits", "permutation_from_groups", "LINKAGES"]


@dataclass
class GroupingResult:
    """A placement plus the trace of how it was reached."""

    groups: List[List[int]]
    n_code: int
    linkage: str
    #: (left, right, score) for each merge that was committed, in order.
    merges: List[Tuple[Tuple[int, ...], Tuple[int, ...], float]] = field(
        default_factory=list
    )
    #: Merges considered and rejected because the score was not positive.
    rejected: int = 0

    @property
    def k(self) -> int:
        return logical_capacity(self.n_code)

    @property
    def num_patches(self) -> int:
        return len(self.groups)

    def permutation(self) -> List[Tuple[int, int]]:
        return permutation_from_groups(self.groups, self.k)

    def summary(self) -> str:
        sizes = sorted((len(group) for group in self.groups), reverse=True)
        return (
            f"{self.linkage}: {self.num_patches} patches, sizes {sizes}, "
            f"{len(self.merges)} merges ({self.rejected} rejected)"
        )


def permutation_from_groups(
    groups: Sequence[Sequence[int]], k: int
) -> List[Tuple[int, int]]:
    """
    Turn groups into the translator's permutation format.

    `permutation[logical] = (patch, slot)`. Groups map to patches in order, and
    members to slots in ascending qubit order, which keeps the output deterministic.
    """
    total = sum(len(group) for group in groups)
    permutation: List[Optional[Tuple[int, int]]] = [None] * total
    for patch, group in enumerate(groups):
        if len(group) > k:
            raise ValueError(f"group {group} exceeds capacity {k}")
        for slot, qubit in enumerate(sorted(group)):
            permutation[qubit] = (patch, slot)
    if any(entry is None for entry in permutation):
        missing = [i for i, entry in enumerate(permutation) if entry is None]
        raise ValueError(f"qubits {missing} were not placed")
    return [entry for entry in permutation if entry is not None]


# --- linkages ---


def _cost_linkage(model: GroupCostModel) -> Callable[[Sequence[int], Sequence[int]], float]:
    def score(left: Sequence[int], right: Sequence[int]) -> float:
        return model.net_gain(left, right)

    return score


def _similarity_linkage(
    profiles: Mapping[int, QubitProfile], weights: SimilarityWeights
) -> Callable[[Sequence[int], Sequence[int]], float]:
    def score(left: Sequence[int], right: Sequence[int]) -> float:
        pairs = [
            frame_similarity(profiles[a], profiles[b], weights)
            for a in left
            for b in right
        ]
        # Average, not sum: a sum would grow with cluster size and make the first
        # merge decide everything.
        return sum(pairs) / len(pairs) if pairs else 0.0

    return score


def _interaction_linkage(
    profiles: Mapping[int, QubitProfile],
) -> Callable[[Sequence[int], Sequence[int]], float]:
    """
    Cut weight: how many two-qubit gates run between the two clusters.

    Merging the most strongly interacting clusters first is plain agglomerative
    min-cut on the interaction graph. It is here because measurement says the number
    of two-qubit gates that cross a patch boundary is what actually drives physical
    cost (r = +0.80 against measured cost, versus -0.17 for the merging pass's own
    savings estimate) -- so the objective is a partitioning problem, not the
    merge-maximisation problem both the packer and `cost_model` are built around.
    """

    def score(left: Sequence[int], right: Sequence[int]) -> float:
        right_set = set(right)
        crossing = 0
        for qubit in left:
            for event in profiles[qubit].two_qubit_events():
                if event.partner in right_set:
                    crossing += 1
        return float(crossing)

    return score


LINKAGES = ("cost", "similarity", "interaction")


def cluster_qubits(
    circuit: QuantumCircuit,
    *,
    n_code: int = 4,
    linkage: str = "cost",
    max_patches: Optional[int] = None,
    min_score: float = 0.0,
    profiles: Optional[Mapping[int, QubitProfile]] = None,
    similarity_weights: SimilarityWeights = DEFAULT_WEIGHTS,
    window: int = 2,
    transversal_confidence: Optional[float] = None,
) -> GroupingResult:
    """
    Group `circuit`'s logical qubits into patches of at most k = n_code - 2 slots.

    Args:
        linkage: "cost" scores the union with the physical cost model; "similarity"
            scores it with the structural frame/cast kernel.
        max_patches: keep merging past `min_score` while more than this many patches
            remain. A target, not a guarantee: greedy merging can reach a partition
            where no two remaining clusters fit in k slots together even though some
            other partition would have. Measured on QASMBench, 8 of 225 placements
            overshot, 7 of them at k=6 -- the size where clusters most often settle
            at just over half capacity, so that no two of them can be joined.
            Callers that need an exact patch budget must check `num_patches`.
        min_score: merges scoring at or below this are not worth making.
        window: layer tolerance for calling two gates simultaneous.
    """
    if linkage not in LINKAGES:
        raise ValueError(f"unknown linkage {linkage!r}; expected one of {LINKAGES}")

    profiles = profiles if profiles is not None else extract_profiles(circuit)
    k = logical_capacity(n_code)

    if linkage == "interaction":
        score = _interaction_linkage(profiles)
    elif linkage == "cost":
        model_kwargs = {"window": window}
        if transversal_confidence is not None:
            model_kwargs["transversal_confidence"] = transversal_confidence
        model = GroupCostModel(profiles=profiles, n_code=n_code, **model_kwargs)
        score = _cost_linkage(model)
    else:
        score = _similarity_linkage(profiles, similarity_weights)

    clusters: List[List[int]] = [[index] for index in sorted(profiles)]
    result = GroupingResult(groups=clusters, n_code=n_code, linkage=linkage)

    while len(clusters) > 1:
        best: Optional[Tuple[float, int, int]] = None
        for left, right in combinations(range(len(clusters)), 2):
            if len(clusters[left]) + len(clusters[right]) > k:
                continue
            value = score(clusters[left], clusters[right])
            if best is None or value > best[0]:
                best = (value, left, right)

        if best is None:
            break  # every admissible union would overflow a patch

        value, left, right = best
        forced = max_patches is not None and len(clusters) > max_patches
        if value <= min_score and not forced:
            result.rejected += 1
            break

        result.merges.append((tuple(clusters[left]), tuple(clusters[right]), value))
        merged = sorted(clusters[left] + clusters[right])
        for index in sorted((left, right), reverse=True):
            clusters.pop(index)
        clusters.append(merged)

    # Deterministic output order: by smallest member.
    clusters.sort(key=lambda group: group[0])
    result.groups = clusters
    return result
