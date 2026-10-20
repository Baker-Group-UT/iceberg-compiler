"""
seed.py — the one job the interaction graph is good at.

The graph is a bad description of cost. Its Hadamard-coincidence term makes the graph
*complete* on Hadamard-rich programs -- 9730 of 9730 possible edges on `bv_n140`, every
one attracting -- which is no grouping signal at all; its transversal term credits
control-to-target lane pairs that a CXT macro cannot realise, because `GateEvent.is_control`
is set and never read; and cut weight tracks the real objective so loosely that where FM
refinement fires it moves the cut 1.5% and the true cost 10-20%.

It is nevertheless a much better *search guide* than the true objective, and that is not
a contradiction. Cut weight is a pairwise relaxation defined on every pair at once, so
minimising it moves the whole partition coherently. The true objective is conjunctive --
transversal macros need every lane of two patches, HH every lane of one -- so its
landscape is plateaus and deep local optima. Descending it from a cheap seed leaves the
naive partition untouched on `knn_n31` at k=10 while a partition 13% cheaper by that same
objective exists.

So: partition on the graph to get global structure, then hand the result to
`search.descend`, which never consults the graph again. Measured, that beats both the
old stack (which kept refining on the cut) and a graph-free descent (which cannot find
the basin).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .graph import NoiseProfile, build_graph

from .cost import PackingCost
from .partition import partition as feasible_partition
from .search import Packing, naive_seed

__all__ = ["graph_seed", "naive_seed"]

#: Named seed configurations, so each idea stays independently measurable.
#:
#: Mean improvement against naive over 64 QASMBench proxies at n = 4/6/8/12, exact
#: translation, `verify` on -- the full pipeline, so these are the numbers that matter:
#:
#:     feasible  -4.88 / -5.20 / -5.37 / -4.48   size invariant, merge base, no swap
#:     swapped   -4.88 / -5.27 / -4.76 / -3.69   the same, plus node exchange
#:     packed    -4.77 / -4.13 / -3.86 / -3.23   the same, with bin packing as the base
#:
#: The `legacy` arm was `grouping.multilevel.partition`, at -4.88 / -5.20 / -5.18 /
#: -4.25. It went with that package; `feasible` matched or beat it everywhere and the
#: measurement is recorded here so the comparison survives the code.
#:
#: One of the three ideas worked. The other two stay selectable because a recorded
#: negative result is cheaper than repeating the experiment.
#:
#: *the size invariant works.* Equal-size capped matching ties `legacy` at k = 2 and 4
#: and beats it at k = 6 and 10, where `coarsen_cap` is 2 and the old coarsener was
#: building 4-qubit blocks that cannot tile a patch. It also makes `pack_bins` provably
#: overflow-free, which matters as soon as it is used as a repair: pairing it with
#: legacy heavy-edge matching, whose sizes need not divide k, produced **11 rejected
#: placements** across the sweep. With the invariant, zero.
#:
#: *node exchange hurts*, by 0.6 points at k = 6 and 0.8 at k = 10. It minimises the
#: *cut*, and the cut is a loose proxy -- FM moves it 1.5% while moving true cost
#: 10-20%. A stronger cut minimiser drives the seed to a lower cut and a worse
#: placement. The seed's job is global structure, not a cut record, and `descend` is
#: what optimises the objective that counts.
#:
#: *bin packing is a much weaker base case* than agglomerative merging -- up to 1.3
#: points. It commits each node on affinity to whatever is already in a bin, so the
#: first placements into empty bins are arbitrary and anchor everything after them.
#: Merging takes a global argmax each step. Kept only as the repair path.
PARTITIONERS: Dict[str, Dict[str, object]] = {
    "feasible": {"matching": "equal", "base": "merge", "swap": False},
    "swapped": {"matching": "equal", "base": "merge", "swap": True},
    "packed": {"matching": "equal", "base": "pack", "swap": True},
}

DEFAULT_PARTITIONER = "feasible"


def graph_seed(
    cost: PackingCost,
    profiles,
    slots: int,
    *,
    noise: Optional[NoiseProfile] = None,
    partitioner: str = DEFAULT_PARTITIONER,
) -> Packing:
    """
    Multilevel-partition the signed interaction graph, padded out to `slots` patches.

    The padding is what lets `search`'s `spread` move raise the patch count afterwards.
    In the old stack `max_patches` reached the partitioner as a *cap* and nothing
    downstream could take a looser budget up on the offer, so all five scanned candidates
    returned the identical tightest partition on 41 of 63 circuits.
    """
    graph = build_graph(
        profiles, cost.n_code, noise=noise, num_qubits=cost.num_qubits
    )
    options = PARTITIONERS.get(partitioner, PARTITIONERS[DEFAULT_PARTITIONER])
    result = feasible_partition(graph, cost.n_code, slots, **options)
    groups: List[Set[int]] = [set(group) for group in result.groups if group]

    # The partitioner is not guaranteed to hit the target, and a partition over capacity
    # is not a legal placement. Split anything oversized rather than trusting it.
    k = cost.k
    tidy: List[Set[int]] = []
    for group in groups:
        members = sorted(group)
        for start in range(0, len(members), k):
            tidy.append(set(members[start:start + k]))

    placed = {q for group in tidy for q in group}
    missing = [q for q in range(cost.num_qubits) if q not in placed]
    for qubit in missing:
        for group in tidy:
            if len(group) < k:
                group.add(qubit)
                break
        else:
            tidy.append({qubit})

    while len(tidy) < slots:
        tidy.append(set())
    return Packing(cost, tidy)
