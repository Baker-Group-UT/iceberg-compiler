"""
generator.py - Responsible for generating the list of permutations
searched in a permutation scan. Generates permutations with respect to the logical
level. Follows the rules:
 - module order does not matter
 - qubit order within module matters
"""
import time
import math
import random
from itertools import combinations, permutations as permute
from typing import List, Tuple, Dict, Any, Union

from qiskit import QuantumCircuit

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

def canonical_relative_index_key(assign: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """
    Canonicalize a permutation by removing patch order *and* normalizing slot indices 
    to be relative (i.e., subtracting the minimum index used across all qubits).

    Parameters
    ----------
    assign : List[Tuple[int, int]]
        List of (patch_id, slot_index) assignments.

    Returns
    -------
    Tuple[Tuple[int, int], ...]
        Canonical representation up to patch reordering and global slot offset.
    """
    # Step 1: Canonicalize patch order
    canon = canonical_key(assign)

    # Step 2: Normalize slot indices (subtract global minimum)
    min_slot = min(slot for _, slot in canon)
    rel_indexed = tuple((patch, slot - min_slot) for patch, slot in canon)

    return rel_indexed

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

    perms = []
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
            sig = canonical_relative_index_key(list(ordering))
            if sig in seen:
                continue
            seen.add(sig)
            perms.append(list(ordering))

            # Live progress print
            now = time.time()
            if now - last_print >= status_interval:
                print(
                    f"\r[permutations] Found {len(perms)} unique assignments "
                    f"(est. max ≈ {est_total})",
                    end="", flush=True
                )
                last_print = now

    print(
        f"\r[permutations] Found {len(perms)} unique assignments (done)                              "
    )

    return perms

def random_permutation_from_circuit(
    circuit: QuantumCircuit,
    code_sizes: List[int],
    *,
    seed: int | None = None,
    return_nodes: bool = False,
    canonicalize: bool = False,
) -> Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], int]]:
    """
    Sample a single random assignment of logical qubits to patch slots.

    This is O(total_slots) to build the slot list and O(q) to sample; it does not
    enumerate permutations. The sampling is uniform over ordered selections of q
    distinct slots, which matches the rule that *qubit order within a module matters*
    while module order does not impact validity.

    Args:
        circuit: Logical circuit object (only .all_qubits() is used; treated as read-only).
        code_sizes: Sizes n for each Iceberg patch/module. Each contributes logical_capacity(n) slots.
        seed: Optional PRNG seed for reproducibility.
        return_nodes: If True, also return the number of distinct patches used.
        canonicalize: If True, returns the canonical relative-index key representative
            (removes patch label symmetry and global slot offset) using the existing helpers.

    Returns:
        Either:
            - List[(patch_id, slot_index)] of length q, or
            - (assignment, nodes_used) if return_nodes=True.

    Raises:
        ValueError: If total available slots < number of logical qubits.
    """
    # Derive q without introducing circuit/gate awareness beyond counting qubits.
    try:
        q = circuit.num_qubits
    except Exception as exc:  # pragma: no cover
        raise TypeError("circuit must expose .all_qubits()") from exc

    # Build the available slot list once.
    slots: List[Tuple[int, int]] = []
    for pid, n in enumerate(code_sizes):
        cap = logical_capacity(n)
        for k in range(cap):
            slots.append((pid, k))

    if len(slots) < q:
        raise ValueError(
            f"Insufficient capacity: need {q} logical slots, have {len(slots)} "
            f"(from code_sizes={code_sizes})."
        )

    rng = random.Random(seed)

    # Choose q distinct slots uniformly without replacement.
    chosen = rng.sample(slots, q)

    # Order matters (logical qubit order), so randomize the permutation order.
    rng.shuffle(chosen)

    assignment: List[Tuple[int, int]] = list(chosen)

    # Optional canonical representative (removes patch relabeling & global slot offset).
    if canonicalize:
        assignment = list(canonical_relative_index_key(assignment))

    if return_nodes:
        nodes_used = len({p for p, _ in assignment})
        return assignment, nodes_used
    return assignment
