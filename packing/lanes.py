"""
lanes.py — which slot each qubit takes, decided once, at the end.

The objective is a function of the partition, so lanes are not a search variable.
They still have to be *chosen*, and the choice is not quite free: a transversal macro
whose control-slot-to-target-slot map is not the identity is realigned by permuting
the target patch, and while those routing swaps are priced `(0, 0)` they do displace
gates in time, which can stop a later macro firing. So lane order is worth getting
right cheaply, and worth nothing at all to search over.

One pass, patches heaviest-traffic first, each solved exactly against everything
already placed. That last clause is the one difference from `multilevel.align_slots`,
which builds its benefit matrix from `partners[(left, right)]` -- the gates to a
single anchor patch -- and discards every gate to any other already-placed patch. On
`sat_n11` that was 38 of 150 relevant gates thrown away. Using all of them is the
same k-by-k assignment and the same exact solve, with more of the information.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, List, Sequence, Tuple

from .cost import PackingCost

__all__ = ["assign_lanes"]

Group = FrozenSet[int]
Placement = List[Tuple[int, int]]


def assign_lanes(groups: Sequence[Group], cost: PackingCost) -> Placement:
    """Turn a partition into a (patch, slot) placement."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    k = cost.k
    live = [g for g in groups if g]
    patch_of: Dict[int, int] = {q: i for i, g in enumerate(live) for q in g}

    traffic: Dict[int, int] = defaultdict(int)
    for (left, right), events in cost.gates.items():
        here, there = patch_of.get(left), patch_of.get(right)
        if here is None or there is None or here == there:
            continue
        traffic[here] += len(events)
        traffic[there] += len(events)

    slots: Dict[int, int] = {}
    for patch in sorted(range(len(live)), key=lambda i: -traffic[i]):
        members = sorted(live[patch])
        benefit = np.zeros((len(members), k))
        pull = False
        for row, qubit in enumerate(members):
            for partner in cost.partners.get(qubit, ()):
                slot = slots.get(partner)
                if slot is None or patch_of.get(partner) == patch:
                    continue
                pair = (min(qubit, partner), max(qubit, partner))
                benefit[row, slot] += len(cost.gates.get(pair, ()))
                pull = True
        if not pull:
            # Nothing placed yet pulls on this patch; any order is as good, and
            # ascending keeps the result reproducible.
            for slot, qubit in enumerate(members):
                slots[qubit] = slot
            continue
        rows, cols = linear_sum_assignment(-benefit)
        for row, col in zip(rows, cols):
            slots[members[row]] = int(col)

    placement: Placement = [(0, 0)] * cost.num_qubits
    for qubit in range(cost.num_qubits):
        placement[qubit] = (patch_of[qubit], slots[qubit])
    return placement
