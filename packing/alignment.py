"""
alignment.py — which grid site each patch gets, once the partition is fixed.

The packer decides *which qubits share a patch*. It does not decide *where that patch
sits*, and until now nothing did: patch index came out of the partitioner's emission order
and `simulator.grid_constructor` drops patch `p` at site `(p // rows, p % rows)`, so the
geometry a circuit ran on was an accident of list order. The packing objective could not
have exploited it either -- `tables.mean_patch_hops` is a mean over *all* patch pairs and
depends only on how many there are.

That leaves a free optimisation on the table, and under the transport model it is worth
something real: a cross-patch operation is charged a round trip between patch centres, so
putting the patches that talk most in the middle of the array shortens the trips that
happen most often.

This is a *relabelling*. Groups are untouched -- the same logical qubits share the same
patch afterwards -- and only the patch id changes, which is what selects the grid site.
So gate counts and depth are invariant and only distance moves.

    align_to_grid(permutation, program)   ->  permutation with patches renumbered

**How the order is chosen.** Two greedy steps, following `archive/packer/grid_layout.py`,
which is where this idea was first tried:

    *patch order*  start from the patch with the most cross-patch traffic, then
        repeatedly take whichever unplaced patch talks most to the ones already placed.
        Pure traffic ranking ignores *who* a patch talks to and scatters cliques; this
        keeps a communicating group together as it fills outward.

    *site order*   sites sorted by Manhattan distance from the grid centre, ties broken
        by Euclidean distance and then index. Manhattan because that is what transport is
        actually charged in; the tie-breaks only exist to make the result deterministic.

The archive defaulted to `row_major`, which puts the busiest patch at (0, 0) rather than
the middle. That is not centre-out and it is not what this does.

**What it cannot fix.** The partition itself was chosen without any knowledge of geometry,
so a packing that spreads a clique across many patches stays expensive no matter where
those patches sit. Alignment is the cheap second half of a routing decision whose first
half nobody made.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit

__all__ = ["align_to_grid", "patch_traffic", "weighted_patch_distance"]

Perm = List[Tuple[int, int]]


def _two_qubit_pairs(program: QuantumCircuit) -> Dict[Tuple[int, int], int]:
    """Count two-qubit gates per unordered logical pair, on the placed view."""
    from .tables import translated_view

    view = translated_view(program)
    counts: Dict[Tuple[int, int], int] = {}
    for instruction in view.data:
        if instruction.operation.num_qubits != 2:
            continue
        left, right = (view.find_bit(q).index for q in instruction.qubits)
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        counts[key] = counts.get(key, 0) + 1
    return counts


def patch_traffic(
    permutation: Sequence[Tuple[int, int]], program: QuantumCircuit
) -> Dict[Tuple[int, int], int]:
    """
    Cross-patch gate count per unordered patch pair.

    Intra-patch gates are excluded: they cost no transport, so they say nothing about
    where a patch wants to sit.
    """
    patch_of = {qubit: patch for qubit, (patch, _) in enumerate(permutation)}
    traffic: Dict[Tuple[int, int], int] = {}
    for (left, right), count in _two_qubit_pairs(program).items():
        a, b = patch_of.get(left), patch_of.get(right)
        if a is None or b is None or a == b:
            continue
        key = (a, b) if a < b else (b, a)
        traffic[key] = traffic.get(key, 0) + count
    return traffic


def _site_grid(num_patches: int) -> Tuple[int, int]:
    """`(columns, rows)` — must match `simulator.grid_constructor.patch_grid_shape`."""
    columns = max(1, math.ceil(math.sqrt(num_patches)))
    return columns, math.ceil(num_patches / columns)


def _sites_by_centrality(num_patches: int) -> List[int]:
    """
    Patch *indices*, most central site first.

    Inverts the simulator's own convention: index `p` lands at site
    `(p // rows, p % rows)`, so to put a patch at a chosen cell we choose its index.
    """
    columns, rows = _site_grid(num_patches)
    centre_x, centre_y = (columns - 1) / 2.0, (rows - 1) / 2.0

    def key(index: int) -> Tuple[float, float, int]:
        x, y = index // rows, index % rows
        manhattan = abs(x - centre_x) + abs(y - centre_y)
        return (manhattan, math.hypot(x - centre_x, y - centre_y), index)

    return sorted(range(num_patches), key=key)


def _patches_by_connectivity(
    patches: Sequence[int], traffic: Dict[Tuple[int, int], int]
) -> List[int]:
    """Busiest patch first, then whichever unplaced patch talks most to those placed."""
    if len(patches) <= 1:
        return list(patches)

    total: Dict[int, int] = {patch: 0 for patch in patches}
    neighbours: Dict[int, Dict[int, int]] = {patch: {} for patch in patches}
    for (left, right), count in traffic.items():
        if left not in total or right not in total:
            continue
        total[left] += count
        total[right] += count
        neighbours[left][right] = neighbours[left].get(right, 0) + count
        neighbours[right][left] = neighbours[right].get(left, 0) + count

    # `-patch` in the key so ties resolve to the lower id, which keeps the result
    # deterministic across runs rather than depending on set iteration order.
    order = [max(patches, key=lambda patch: (total[patch], -patch))]
    remaining = set(patches) - set(order)
    while remaining:
        nxt = max(
            remaining,
            key=lambda patch: (
                sum(neighbours[patch].get(placed, 0) for placed in order),
                total[patch],
                -patch,
            ),
        )
        order.append(nxt)
        remaining.discard(nxt)
    return order


def weighted_patch_distance(
    permutation: Sequence[Tuple[int, int]], program: QuantumCircuit
) -> float:
    """
    Total cross-patch gates times the grid distance they travel.

    The quantity `align_to_grid` exists to reduce, in gate-cells. Distances are Manhattan
    between patch *sites*, matching how transport is charged; this deliberately ignores the
    patch rectangle's own size, because that is identical under any relabelling.
    """
    traffic = patch_traffic(permutation, program)
    if not traffic:
        return 0.0
    live = {patch for patch, _ in permutation}
    _, rows = _site_grid(len(live))
    total = 0.0
    for (left, right), count in traffic.items():
        lx, ly = left // rows, left % rows
        rx, ry = right // rows, right % rows
        total += count * (abs(lx - rx) + abs(ly - ry))
    return total


def align_to_grid(
    permutation: Sequence[Tuple[int, int]], program: QuantumCircuit
) -> Perm:
    """
    Renumber patches so the busiest ones sit near the centre of the grid.

    Groups are preserved exactly: the qubits that shared a patch still share one, and each
    qubit keeps its slot. Only the patch id changes, and the patch id is what picks the
    grid site. Gate counts and depth are therefore unchanged by construction.

    Patch ids are compacted first, so a permutation using a sparse set of ids (which
    `packing.lanes` does not produce, but callers may) still maps onto a dense grid.

    **Kept only when it wins.** The ordering is greedy and greedy is not optimal here --
    measured across 12 QASMBench proxies it improves weighted distance by 12.7% overall but
    *regresses* on three of them, worst at +16% on `cat_state_n22`, where the traffic graph
    is a star and centring the hub does nothing the default order was not already doing.
    Scoring both layouts costs one pass over the gate list and no simulation, so the
    comparison is free and this returns whichever is cheaper. That is the same bargain
    `packing.plan`'s verify stage makes: it turns "usually better" into "never worse",
    which is what makes the arm safe to read as a contribution.
    """
    if not permutation:
        return []

    live = sorted({patch for patch, _ in permutation})
    compact = {patch: index for index, patch in enumerate(live)}
    dense: Perm = [(compact[patch], slot) for patch, slot in permutation]
    if len(live) <= 1:
        return dense

    traffic = patch_traffic(dense, program)
    if not traffic:
        # Nothing crosses a patch boundary, so every layout costs the same.
        return dense

    ordered = _patches_by_connectivity(range(len(live)), traffic)
    sites = _sites_by_centrality(len(live))
    old_to_new = {patch: sites[rank] for rank, patch in enumerate(ordered)}
    aligned: Perm = [(old_to_new[patch], slot) for patch, slot in dense]

    if weighted_patch_distance(aligned, program) < weighted_patch_distance(dense, program):
        return aligned
    return dense
