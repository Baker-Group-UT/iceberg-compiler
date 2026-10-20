"""
incremental.py -- an opt-in, exact re-evaluation of the descent's two whole-partition terms.

`packing.search.Packing.value_after` prices every candidate move with two terms computed over
the whole partition:

    `PackingCost.movement_hops`         every one of the P(P-1)/2 live patch pairs, each a
                                        `binary` lookup (or computation, for a new group)
    `PackingCost.busiest_patch_depth`   every gate pair of the program

A move changes the membership of one or two patches, so only patch pairs that touch them can
change either term -- and of those, only pairs with a gate between them, because a pair with
no crossing contributes nothing to movement. This prices both terms as the current state's
value plus the change on those pairs: every patch holding a gate partner of any member of a
touched patch, before or after the move. That is a superset of `Packing._delta`'s scope for
swaps (the moved qubits' partners only), deliberately: a pair's macro credit depends on the
whole membership, so nothing here relies on a swap leaving an untouched pair's term alone.
Accepted moves update the state the same way instead of re-walking the partition. Moves that
change the set of live patches (a spread into an empty slot, or a move that empties one) shift
every later patch's grid position, so they fall back to the library's full evaluation.

What it cannot remove: the descent still prices O(n^2) candidates per sweep, and each new
patch pair it meets is still computed and memoised by `PackingCost.binary`, which is now the
dominant per-candidate cost at large k and the reason memory still grows with n.

**Exact, not approximate.** Every summand of both terms is an integer-valued double: hop
distances are integer cell counts (`patch_hop_matrix`, integer strides), crossings and
crossing savings are counts, and per-gate depths are integers. Sums of such values are exact
in any order (far below 2^53), so the prices -- hence every comparison the descent makes, the
moves it accepts and the placement it returns -- are bit-identical to the library's.
`compile_scaling.py` checks this per cell against the library planner's placement and
descent counts rather than assuming it.

Installed only inside a measurement child, by `install()`; nothing in the library changes and
nothing outside that process sees the patch.
"""

from __future__ import annotations

import heapq
from typing import Dict, Optional

_INSTALLED: Dict[str, object] = {}


class _State:
    """The current partition's movement and depth terms, by patch, valid until `apply`."""

    __slots__ = ("position", "matrix", "hops", "loads", "top", "value")

    def __init__(self, packing) -> None:
        from packing.tables import patch_hop_matrix

        cost = packing.cost
        live = [index for index, group in enumerate(packing.groups) if group]
        self.position = {index: position for position, index in enumerate(live)}
        self.matrix = patch_hop_matrix(len(live), cost.n_code + 2) if len(live) >= 2 else None
        # Full evaluations, exactly the library's, once per accepted move.
        self.hops = cost.movement_hops(packing.groups)
        self.loads = _loads(cost, packing.groups)
        self.top = heapq.nlargest(3, ((load, index) for index, load in self.loads.items()))
        self.value = cost.price(packing.live, packing._total,
                                max(self.loads.values(), default=0.0), self.hops)


def _loads(cost, groups) -> Dict[int, float]:
    """`busiest_patch_depth`'s per-patch loads, before its max."""
    owner = {q: index for index, group in enumerate(groups) for q in group}
    load: Dict[int, float] = {}
    for (left, right), term in cost.intra.items():
        a, b = owner.get(left), owner.get(right)
        if a is not None and a == b:
            load[a] = load.get(a, 0.0) + term.depth
    for (left, right), term in cost.cross.items():
        a, b = owner.get(left), owner.get(right)
        if a is None or b is None or a == b:
            continue
        load[a] = load.get(a, 0.0) + term.depth
        load[b] = load.get(b, 0.0) + term.depth
    return load


def _patch_load(cost, members) -> float:
    """Load of one patch holding `members`, whatever the rest of the partition is."""
    total = 0.0
    partners, intra, cross = cost.partners, cost.intra, cost.cross
    for left in members:
        for right in partners.get(left, ()):
            pair = (left, right) if left < right else (right, left)
            if right in members:
                if left < right:
                    total += intra[pair].depth
            else:
                total += cross[pair].depth
    return total


def _moved(cost, left, right) -> float:
    term = cost.binary(left, right)
    moved = term.crossings - term.crossing_savings
    return moved if moved > 0.0 else 0.0


def _effect(packing, changes, current: _State):
    """
    What `changes` does to the partition, priced on `current` without applying it:
    `(delta, live_delta, hops, busiest, touched_loads)`. Live set preserved by the caller.
    """
    cost = packing.cost
    touched = sorted(changes)
    groups = packing.groups
    delta, live_delta = packing._delta(changes)

    hops = current.hops
    if current.matrix is not None:
        others = set()
        for index in touched:
            others |= packing._neighbour_patches(groups[index])
            others |= packing._neighbour_patches(changes[index])
        others.difference_update(touched)
        position, matrix = current.position, current.matrix
        change = 0.0
        for index in touched:
            here = position[index]
            for other in others:
                distance = matrix[here][position[other]]
                resident = groups[other]
                change += (_moved(cost, changes[index], resident)
                           - _moved(cost, groups[index], resident)) * distance
        for first, left in enumerate(touched):
            for right in touched[first + 1:]:
                distance = matrix[position[left]][position[right]]
                change += (_moved(cost, changes[left], changes[right])
                           - _moved(cost, groups[left], groups[right])) * distance
        hops = hops + change

    busiest: Optional[float] = None
    for load, index in current.top:
        if index not in changes:
            busiest = load
            break
    if busiest is None:  # every top load was touched: scan the untouched remainder
        busiest = max((load for index, load in current.loads.items()
                       if index not in changes), default=0.0)
    touched_loads = {index: _patch_load(cost, changes[index]) for index in touched}
    for load in touched_loads.values():
        busiest = max(busiest, load)
    return delta, live_delta, hops, busiest, touched_loads


def _preserves_live(packing, changes) -> bool:
    return all(bool(changes[i]) == bool(packing.groups[i]) for i in changes)


def install() -> None:
    """Patch `Packing.value`, `value_after` and `apply` in this process. Idempotent."""
    from packing.search import Packing

    if _INSTALLED:
        return
    original_after = Packing.value_after
    original_apply = Packing.apply
    original_value = Packing.value
    _INSTALLED.update(value_after=original_after, apply=original_apply, value=original_value)

    def state(self) -> _State:
        current = self.__dict__.get("_incremental")
        if current is None:
            current = _State(self)
            self.__dict__["_incremental"] = current
        return current

    def value(self) -> float:
        return state(self).value

    def apply(self, changes) -> None:
        current = self.__dict__.get("_incremental")
        if current is None or not _preserves_live(self, changes):
            original_apply(self, changes)
            self.__dict__["_incremental"] = None  # positions may shift: rebuild on demand
            return
        # Priced on the state before the move, then carried across it: the same exact
        # integer sums a rebuild would produce, without re-walking every patch pair.
        _, _, hops, _, touched_loads = _effect(self, changes, current)
        original_apply(self, changes)
        current.loads.update(touched_loads)
        current.hops = hops
        current.top = heapq.nlargest(3, ((load, index)
                                         for index, load in current.loads.items()))
        current.value = self.cost.price(self.live, self._total,
                                        max(current.loads.values(), default=0.0), hops)

    def value_after(self, changes) -> float:
        if not _preserves_live(self, changes):
            return original_after(self, changes)  # live set changes: positions shift
        delta, live_delta, hops, busiest, _ = _effect(self, changes, state(self))
        return self.cost.price(self.live + live_delta, self._total + delta, busiest, hops)

    Packing.value = value
    Packing.apply = apply
    Packing.value_after = value_after


def uninstall() -> None:
    from packing.search import Packing

    for name, original in _INSTALLED.items():
        setattr(Packing, name, original)
    _INSTALLED.clear()
