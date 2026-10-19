"""
space.py -- the exhaustive placement space, streamed instead of materialised.

`permutations.generator.generate_permutations` enumerates every ordered choice of slots
and deduplicates afterwards, so it touches C(q*k, q) * q! orderings to keep a few
thousand -- 1.86 million at (5 qubits, k = 4), which it refuses. The *unique* space is
small: a placement is unique up to patch relabelling and a global slot offset
(`canonical_relative_index_key`), so it is exactly

    a set partition of the qubits into blocks of size <= k   (patches, unordered)
  x an injective slot map per block                           (lanes)
  with the smallest slot used anywhere equal to 0             (offset removed)

which this module enumerates directly: per partition, prod P(k, |b|) lane maps minus the
prod P(k-1, |b|) that never use slot 0. Patches are labelled by first appearance in qubit
order, which is the generator's canonical form, so every yielded placement equals its own
`canonical_relative_index_key`. `verify_against_generator` checks the two spaces are the
same set wherever the generator will run.

**That canonical space is not complete for the translator, and this was measured.** The
translator is *not* invariant under patch relabelling: on a 5-qubit random program at
k = 2, swapping which of two paired patches gets the lower index moved work 1065 -> 1079
and depth 43 -> 47, and flipping the lanes of one patch moved depth 43 -> 49. (Which patch
of a transversal pair is the target, and whether lanes line up, are both read off the
labels.) So an "exhaustive optimum" over canonical representatives -- which is what the
archived script reported -- can be beaten by a placement outside it, and was, by 2%.

`complete_placements` therefore enumerates every partition x every patch ordering x every
lane map, with no canonicalisation at all. It is affordable at the tightest patch count,
which is the space `plan_packing(spread=0)` searches: 720 placements at (6, k = 2), 6,720
at (5, k = 4). `exhaustive_space` uses it there, and canonical representatives only for
the sparser patch counts, where complete enumeration runs to 10^5 and beyond; the sparse
part of any "full space" optimum is therefore a canonical-only bound, and is labelled so.
"""

from __future__ import annotations

from itertools import permutations, product
from math import perm
from typing import Iterator, List, Sequence, Tuple

Perm = List[Tuple[int, int]]


def set_partitions(n: int, max_block: int) -> Iterator[List[int]]:
    """Restricted growth strings: block id per element, ids in first-appearance order."""
    labels = [0] * n
    sizes: List[int] = []

    def rec(i: int):
        if i == n:
            yield list(labels)
            return
        for block in range(len(sizes)):
            if sizes[block] < max_block:
                labels[i] = block
                sizes[block] += 1
                yield from rec(i + 1)
                sizes[block] -= 1
        labels[i] = len(sizes)
        sizes.append(1)
        yield from rec(i + 1)
        sizes.pop()

    if n == 0:
        yield []
        return
    yield from rec(0)


def placements(q: int, k: int, *, patches: int | None = None) -> Iterator[Perm]:
    """Every canonical placement of `q` qubits on k-slot patches; optionally exactly `patches`."""
    for labels in set_partitions(q, k):
        blocks = max(labels) + 1 if labels else 0
        if patches is not None and blocks != patches:
            continue
        members = [[i for i, b in enumerate(labels) if b == block] for block in range(blocks)]
        lane_choices = [permutations(range(k), len(m)) for m in members]
        for lanes in product(*[list(c) for c in lane_choices]):
            slot = [0] * q
            for block, assignment in enumerate(lanes):
                for qubit, s in zip(members[block], assignment):
                    slot[qubit] = s
            if min(slot) != 0:
                continue
            yield [(labels[i], slot[i]) for i in range(q)]


def space_size(q: int, k: int, *, patches: int | None = None) -> int:
    total = 0
    for labels in set_partitions(q, k):
        blocks = max(labels) + 1 if labels else 0
        if patches is not None and blocks != patches:
            continue
        sizes = [labels.count(b) for b in range(blocks)]
        with_zero = 1
        without = 1
        for s in sizes:
            with_zero *= perm(k, s)
            without *= perm(k - 1, s) if k - 1 >= s else 0
        total += with_zero - without
    return total


def verify_against_generator(q: int, k: int) -> Tuple[bool, int, int]:
    """Same set as `generate_permutations` (canonicalised)? Returns (equal, ours, theirs)."""
    import contextlib
    import io

    from permutations.code_sets import generate_uniform_code_sets
    from permutations.generator import canonical_relative_index_key, generate_permutations

    with contextlib.redirect_stdout(io.StringIO()):
        theirs = {
            canonical_relative_index_key([tuple(s) for s in p])
            for p in generate_permutations(generate_uniform_code_sets(q, n=k + 2), q)
        }
    ours = {tuple(p) for p in placements(q, k)}
    return ours == theirs, len(ours), len(theirs)


def complete_placements(q: int, k: int, patches: int) -> Iterator[Perm]:
    """Every placement on exactly `patches` patches: partitions x patch orders x lane maps."""
    for labels in set_partitions(q, k):
        blocks = max(labels) + 1 if labels else 0
        if blocks != patches:
            continue
        members = [[i for i, b in enumerate(labels) if b == block] for block in range(blocks)]
        lane_choices = [list(permutations(range(k), len(m))) for m in members]
        for order in permutations(range(blocks)):
            for lanes in product(*lane_choices):
                placement = [(0, 0)] * q
                for block, assignment in enumerate(lanes):
                    for qubit, s in zip(members[block], assignment):
                        placement[qubit] = (order[block], s)
                yield placement


def complete_size(q: int, k: int, patches: int) -> int:
    from math import factorial

    total = 0
    for labels in set_partitions(q, k):
        blocks = max(labels) + 1 if labels else 0
        if blocks != patches:
            continue
        lanes = 1
        for b in range(blocks):
            lanes *= perm(k, labels.count(b))
        total += factorial(blocks) * lanes
    return total


def exhaustive_space(q: int, k: int, mode: str) -> Iterator[Tuple[Perm, bool]]:
    """
    (placement, complete) pairs. `mode="tightest"`: the complete tightest space only.
    `mode="full"`: that, plus canonical representatives of every sparser patch count.
    """
    tight = -(-q // k)
    for placement in complete_placements(q, k, tight):
        yield placement, True
    if mode == "full":
        for placement in placements(q, k):
            if len({p for p, _ in placement}) > tight:
                yield placement, False
