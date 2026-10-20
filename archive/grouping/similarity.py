"""
similarity.py — how alike are two qubits' gate timelines?

This is the cheap linkage, and the direct expression of the idea that you can decide
grouping by comparing "the frame or cast of a qubit's gate order" to another's. It
never costs anything in physical gates; it is a pure structural comparison, so it
can be computed once into a matrix and reused.

Three signals, each mapping to a specific thing the merging pass can do:

`frame agreement`
    Fraction of layers where both qubits have the *same kind* of gate. High agreement
    means Hadamards, Paulis and same-type two-qubit gates line up, which is exactly
    what HH / XX / ZZ / transversal macros need. Computed with a layer tolerance,
    because the merging pass will delay a gate to make a macro fire.

`cast overlap`
    Jaccard similarity of the two qubits' interaction partners. Two qubits that talk
    to the same third parties are candidates for transversal fusion; and if they talk
    to *each other*, co-locating them turns cross-patch traffic into intra-patch.

`schedule conflict`
    Fraction of layers where both are busy with *different* kinds. This is the
    negative signal a pure "affinity" score misses: two qubits can have high cast
    overlap and still be bad partners because their gates never coincide in kind, so
    every layer forces two separate operations on the same patch.

The combined score is a weighted sum. It is deliberately *not* in units of physical
gates — it is a heuristic ordering — so `evaluate.py` measures it against the
cost-model linkage rather than assuming either is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from .profiles import QubitProfile

__all__ = ["SimilarityWeights", "frame_similarity", "similarity_matrix", "explain"]


@dataclass(frozen=True)
class SimilarityWeights:
    """
    Relative importance of the three signals.

    Defaults put most of the weight on frame agreement, since the whole-patch
    Hadamard is the largest single saving the merging pass can find, and give cast
    overlap half as much. The conflict penalty is small: a conflict costs depth, not
    gates.
    """

    frame: float = 1.0
    cast: float = 0.5
    conflict: float = 0.25
    #: Layer tolerance for calling two gates simultaneous.
    window: int = 2
    #: Kinds that count toward frame agreement, most valuable first.
    kind_value: Mapping[str, float] = None  # type: ignore[assignment]

    def values(self) -> Mapping[str, float]:
        if self.kind_value is not None:
            return self.kind_value
        # Hadamard alignment is worth the most (HH is the biggest macro), then
        # two-qubit alignment (transversal macros), then Paulis (cheap either way).
        return {
            "h": 1.0,
            "cx": 0.6,
            "cz": 0.6,
            "xcx": 0.6,
            "x": 0.3,
            "z": 0.3,
        }


DEFAULT_WEIGHTS = SimilarityWeights()


def frame_similarity(
    left: QubitProfile,
    right: QubitProfile,
    weights: SimilarityWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    Structural similarity of two qubits' timelines, in [-1, 1]-ish.

    Positive means "these want the same things at the same times".
    """
    agreement = _frame_agreement(left, right, weights)
    overlap = _cast_overlap(left, right)
    conflict = _schedule_conflict(left, right, weights)
    return weights.frame * agreement + weights.cast * overlap - weights.conflict * conflict


def _frame_agreement(
    left: QubitProfile, right: QubitProfile, weights: SimilarityWeights
) -> float:
    """Weighted count of coinciding same-kind gates, normalised by the opportunity."""
    values = weights.values()
    matched = 0.0
    possible = 0.0
    for kind, value in values.items():
        left_layers = sorted(left.layers_with(kind))
        right_layers = sorted(right.layers_with(kind))
        possible += value * min(len(left_layers), len(right_layers))
        matched += value * _greedy_pairing(left_layers, right_layers, weights.window)
    if possible <= 0.0:
        return 0.0
    return matched / possible


def _greedy_pairing(left: Sequence[int], right: Sequence[int], window: int) -> int:
    """How many layers can be paired within `window`, each gate used once."""
    available = list(right)
    paired = 0
    for layer in left:
        best = None
        for candidate in available:
            if abs(candidate - layer) <= window:
                if best is None or abs(candidate - layer) < abs(best - layer):
                    best = candidate
        if best is not None:
            available.remove(best)
            paired += 1
    return paired


def _cast_overlap(left: QubitProfile, right: QubitProfile) -> float:
    """
    Jaccard overlap of interaction partners, with talking to each other counted.

    Interacting directly is the strongest possible reason to share a patch, so it is
    folded in as if each were in the other's cast.
    """
    left_cast = set(left.cast()) | {left.index}
    right_cast = set(right.cast()) | {right.index}
    union = left_cast | right_cast
    if not union:
        return 0.0
    return len(left_cast & right_cast) / len(union)


def _schedule_conflict(
    left: QubitProfile, right: QubitProfile, weights: SimilarityWeights
) -> float:
    """Fraction of shared busy layers where the two want *different* kinds."""
    left_frame = left.frame()
    right_frame = right.frame()
    shared = set(left_frame) & set(right_frame)
    if not shared:
        return 0.0
    conflicts = sum(
        1 for layer in shared if not set(left_frame[layer]) & set(right_frame[layer])
    )
    return conflicts / len(shared)


def similarity_matrix(
    profiles: Mapping[int, QubitProfile],
    weights: SimilarityWeights = DEFAULT_WEIGHTS,
) -> Dict[Tuple[int, int], float]:
    """
    The fully connected weighted graph over logical qubits.

    Keys are sorted pairs, so this is an upper-triangular adjacency map.
    """
    indices = sorted(profiles)
    matrix: Dict[Tuple[int, int], float] = {}
    for position, left in enumerate(indices):
        for right in indices[position + 1 :]:
            matrix[(left, right)] = frame_similarity(
                profiles[left], profiles[right], weights
            )
    return matrix


def explain(
    left: QubitProfile, right: QubitProfile, weights: SimilarityWeights = DEFAULT_WEIGHTS
) -> str:
    """Per-signal breakdown, for the experiment log."""
    return (
        f"q{left.index}~q{right.index}: "
        f"frame={_frame_agreement(left, right, weights):.3f} "
        f"cast={_cast_overlap(left, right):.3f} "
        f"conflict={_schedule_conflict(left, right, weights):.3f} "
        f"-> {frame_similarity(left, right, weights):.3f}"
    )
