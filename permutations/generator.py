"""
generator.py - Responsible for generating the list of permutations
searched in a permutation scan. Generates permutations with respect to the logical
level. Follows the rules:
 - module order does not matter
 - qubit order within module matters
"""
import time
import math
from itertools import combinations, permutations as permute
from typing import List, Tuple, Dict

from iceberg.utils import logical_capacity


def canonical_key(assign: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """
    Normalize patch labels to enable unique identification of permutation structure.

    Parameters
    ----------
    assign : List[Tuple[int, int]]
        List of (patch_id, slot_index) assignments.

    Returns
    -------
    Tuple[Tuple[int, int], ...]
        Canonical form of assignment, where patch_ids are relabeled to eliminate
        differences due solely to patch order.
    """
    patch_map: Dict[int, int] = {}
    next_id = 0
    canon = []

    for patch, slot in assign:
        if patch not in patch_map:
            patch_map[patch] = next_id
            next_id += 1
        canon.append((patch_map[patch], slot))

    return tuple(canon)


def generate_permutations(
    code_sizes: List[int],
    q: int,
    *,
    status_interval: float = 0.01,
) -> List[List[Tuple[int, int]]]:
    """
    Generate all unique assignments of `q` logical qubits to patch slots.

    A unique assignment is defined up to patch relabeling. This function returns
    one representative from each equivalence class.

    Parameters
    ----------
    code_sizes : List[int]
        Sizes of each Iceberg code patch. Each code contributes
        `logical_capacity(n)` slots.
    q : int
        Number of logical qubits to assign.
    status_interval : float, optional
        Time in seconds between progress updates (default is 0.01).

    Returns
    -------
    List[List[Tuple[int, int]]]
        Unique permutations, each as a list of (patch_id, slot_index) pairs.

    Raises
    ------
    ValueError
        If the estimated number of permutations exceeds 1 million.
    """
    # Build slot list: (patch_id, slot_index)
    slots: List[Tuple[int, int]] = []
    for pid, n in enumerate(code_sizes):
        for k in range(logical_capacity(n)):
            slots.append((pid, k))

    reps = []
    seen: set[Tuple[Tuple[int, int], ...]] = set()
    last_print = time.time()

    # Estimate total workload
    total_combos = math.comb(len(slots), q)
    est_total = total_combos * math.factorial(q)
    if est_total > 1_000_000:
        raise ValueError(
            f"Estimated {est_total} permutations exceed safety threshold. "
            "Use a reduced set or batch-mode instead."
        )

    # Main enumeration loop
    for subset in combinations(slots, q):
        for ordering in permute(subset, q):
            sig = canonical_key(list(ordering))
            if sig in seen:
                continue
            seen.add(sig)
            reps.append(list(ordering))

            # Live progress print
            now = time.time()
            if now - last_print >= status_interval:
                print(
                    f"\r[permutations] Found {len(reps)} unique assignments "
                    f"(est. max ≈ {est_total})",
                    end="", flush=True
                )
                last_print = now

    print(
        f"\r[permutations] Found {len(reps)} unique assignments (done)          "
    )

    return reps
