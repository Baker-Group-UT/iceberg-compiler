"""
cost_model.py — what a group of logical qubits costs when it shares one patch.

This is the piece that makes grouping extensible past pairs. It scores an arbitrary
*set* of qubits directly, in time linear in the group size and circuit depth, so
there is never any need to enumerate subsets.

What it models, and why each term is not a sum over pairs
--------------------------------------------------------
`whole-patch Hadamard`
    The dominant win, and strictly conjunctive: HH applies H to *every* lane, so it
    only fires on layers where all members of the group want a Hadamard. A
    sum-of-pairs surrogate counts a win whenever any two members align, which at
    k=4 overestimates badly. Savings = |G| * c(H) - c(HH).

`Pauli fusion`
    X (or Z) on m lanes of a patch costs m + (m mod 2) single-qubit gates rather
    than 2m, because the anchor flip cancels for an even count. So the saving grows
    with how many members align, and it is *super*-additive in a way pairs cannot
    see: three aligned X gates save more than any pairing of them.

`co-location`
    A two-qubit gate between two members becomes intra-patch. For CX that is free
    (intra and cross both cost (0,4)) but for CZ and XCX the intra construction is
    cheaper, and in all cases a cross-patch gate is charged to *both* patches while
    an intra-patch gate is charged once. That difference is what rewards putting
    interacting qubits together.

`transversal potential`
    Two members whose two-qubit gates point at a *common* other qubit at nearby
    layers can be fused into one transversal macro — but only if the partners also
    end up together, which is not knowable while scoring one group. It is therefore
    modelled as a bounded bonus rather than a firm saving; `TRANSVERSAL_CONFIDENCE`
    is how much of the macro's savings we are willing to bank on speculatively.

Costs come from `iceberg.costs`, so the model follows the code size rather than
restating n=4 numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Mapping, Sequence, Set, Tuple

from iceberg.costs import DEFAULT_WORK_WEIGHTS, work_table

from .profiles import (
    QubitProfile,
    TWO_QUBIT_KINDS,
    aligned_layers,
    pairwise_aligned_count,
)

__all__ = ["GroupCostModel", "group_cost", "net_gain"]

#: How much of a speculative transversal macro's savings to count. The macro needs
#: the partners to be co-located too, which this scoring pass cannot see, so the
#: full saving would be optimistic and zero would ignore a real effect.
TRANSVERSAL_CONFIDENCE = 0.5

#: Layer tolerance when deciding whether two gates "line up". The merging pass will
#: delay a gate by a few layers to make a macro fire, so exact equality understates.
DEFAULT_WINDOW = 2


@dataclass
class GroupCostModel:
    """
    Scores groups of logical qubits for one program at one code size.

    Cost is in the packer's weighted units (1 per single-qubit gate, 13 per
    two-qubit gate by default), and is *lower is better*.
    """

    profiles: Mapping[int, QubitProfile]
    n_code: int = 4
    window: int = DEFAULT_WINDOW
    weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS
    transversal_confidence: float = TRANSVERSAL_CONFIDENCE
    _cache: Dict[FrozenSet[int], float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._table = work_table(self.n_code)

    # --- cost of one gate ---

    @property
    def k(self) -> int:
        return self.n_code - 2

    def cost_of(self, gate: str) -> float:
        one_qubit, two_qubit = self._table[gate]
        w1, w2 = self.weights
        return w1 * one_qubit + w2 * two_qubit

    def cross_patch_cost(self, kind: str) -> float:
        """
        A two-qubit gate between patches, charged to one endpoint.

        Cross-patch CZ and XCX use different constructions from their intra-patch
        counterparts (see `iceberg.costs`); CX is the same either way.
        """
        w1, w2 = self.weights
        if kind == "cz":
            return w2 * 4
        if kind == "xcx":
            return w1 * 4 + w2 * 4
        return self.cost_of("CX")

    def intra_patch_cost(self, kind: str) -> float:
        return self.cost_of({"cx": "CX", "cz": "CZ", "xcx": "XCX"}[kind])

    # --- cost of a group ---

    def cost(self, group: Iterable[int]) -> float:
        """
        Predicted cost of every gate touching `group`, given the group shares a patch.

        Cross-patch two-qubit gates are charged in full to this group; the partner's
        group is charged in full too. That double counting is deliberate and
        consistent: it makes `net_gain` reward co-location by exactly the traffic it
        removes.
        """
        members = frozenset(group)
        if not members:
            return 0.0
        if members in self._cache:
            return self._cache[members]

        profiles = [self.profiles[index] for index in sorted(members)]
        total = 0.0
        total += self._single_qubit_cost(profiles)
        total += self._two_qubit_cost(profiles, members)
        total -= self._transversal_bonus(profiles, members)

        self._cache[members] = total
        return total

    def _single_qubit_cost(self, profiles: Sequence[QubitProfile]) -> float:
        size = len(profiles)
        total = 0.0

        # Hadamards: whole-patch HH where all lanes align, individually otherwise.
        hh_layers = aligned_layers(profiles, "h", window=self.window) if size > 1 else []
        merged_h = len(hh_layers) * size
        remaining_h = sum(profile.count("h") for profile in profiles) - merged_h
        total += len(hh_layers) * self.cost_of("HH")
        total += max(0, remaining_h) * self.cost_of("H")

        # Paulis: fuse aligned pairs, which for X/Z removes the anchor flip.
        for kind, single, fused in (("x", "X", "XX"), ("z", "Z", "ZZ")):
            fused_pairs = (
                pairwise_aligned_count(profiles, kind, window=self.window)
                if size > 1
                else 0
            )
            singles = sum(profile.count(kind) for profile in profiles) - 2 * fused_pairs
            total += fused_pairs * self.cost_of(fused)
            total += max(0, singles) * self.cost_of(single)

        # Everything else is placement-independent.
        for profile in profiles:
            for event in profile.events:
                if event.is_two_qubit or event.kind in ("h", "x", "z"):
                    continue
                total += self.cost_of(_neutral_gate_name(event.kind))
        return total

    def _two_qubit_cost(
        self, profiles: Sequence[QubitProfile], members: FrozenSet[int]
    ) -> float:
        total = 0.0
        counted_internal: Set[Tuple[int, int, int, str]] = set()
        for profile in profiles:
            for event in profile.two_qubit_events():
                kind = event.kind if event.kind in TWO_QUBIT_KINDS else "cx"
                if event.partner in members:
                    # Charge an internal gate once, not once per endpoint.
                    key = (
                        event.layer,
                        min(profile.index, event.partner),
                        max(profile.index, event.partner),
                        event.kind,
                    )
                    if key in counted_internal:
                        continue
                    counted_internal.add(key)
                    total += self.intra_patch_cost(kind)
                else:
                    total += self.cross_patch_cost(kind)
        return total

    def _transversal_bonus(
        self, profiles: Sequence[QubitProfile], members: FrozenSet[int]
    ) -> float:
        """
        Speculative credit for two-qubit gates that could fuse into one macro.

        Two members pointing at a common outside partner, with the same gate kind and
        within `window` layers, is the pattern the transversal scan looks for. The
        macro replaces two cross-patch gates costing 2 * c(kind) with one costing
        (0, n), so the saving is scaled by `transversal_confidence` because whether
        it actually fires also depends on the partners' placement.
        """
        if len(profiles) < 2 or self.transversal_confidence <= 0.0:
            return 0.0

        w1, w2 = self.weights
        macro_cost = {
            "cx": w2 * self.n_code,
            "cz": w2 * self.n_code,
            "xcx": w1 * 2 * self.n_code + w2 * self.n_code,
        }

        bonus = 0.0
        used: Set[Tuple[int, int, str]] = set()
        for position, profile in enumerate(profiles):
            for event in profile.two_qubit_events():
                if event.partner in members:
                    continue
                for other in profiles[position + 1 :]:
                    for candidate in other.two_qubit_events():
                        if candidate.partner in members:
                            continue
                        if candidate.kind != event.kind:
                            continue
                        if abs(candidate.layer - event.layer) > self.window:
                            continue
                        key = (candidate.layer, candidate.partner, candidate.kind)
                        if key in used:
                            continue
                        used.add(key)
                        kind = event.kind if event.kind in TWO_QUBIT_KINDS else "cx"
                        pair_cost = 2 * self.cross_patch_cost(kind)
                        saving = pair_cost - macro_cost.get(kind, pair_cost)
                        if saving > 0:
                            bonus += self.transversal_confidence * saving
                        break
        return bonus

    # --- linkage ---

    def net_gain(self, left: Iterable[int], right: Iterable[int]) -> float:
        """
        How much merging two groups into one patch saves. Higher is better.

        This is the agglomerative linkage. Unlike the packer's `w_ij - w_ii - w_jj`
        it is evaluated on the *union*, so it sees the conjunctive HH term and the
        super-additive Pauli term that a pairwise score cannot express.
        """
        left = frozenset(left)
        right = frozenset(right)
        if left & right:
            raise ValueError("groups overlap")
        return self.cost(left) + self.cost(right) - self.cost(left | right)


def _neutral_gate_name(kind: str) -> str:
    return {
        "s": "S",
        "sdg": "S",
        "sx": "SX",
        "sxdg": "SX",
        "rz": "RZ",
        "rx": "RX",
        "y": "Y",
        "swap": "SWAP",
    }.get(kind, "S")


def group_cost(
    profiles: Mapping[int, QubitProfile], group: Iterable[int], **kwargs
) -> float:
    """Convenience wrapper around `GroupCostModel.cost`."""
    return GroupCostModel(profiles=profiles, **kwargs).cost(group)


def net_gain(
    profiles: Mapping[int, QubitProfile],
    left: Iterable[int],
    right: Iterable[int],
    **kwargs,
) -> float:
    """Convenience wrapper around `GroupCostModel.net_gain`."""
    return GroupCostModel(profiles=profiles, **kwargs).net_gain(left, right)
