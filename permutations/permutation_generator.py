import time
import math
from itertools import combinations, permutations
from typing import List, Tuple, Dict

# ------------------- capacity helper ---------------------------------
def logical_capacity(n_code: int) -> int:
    return n_code - 2                        # Iceberg codes, n even

# ------------------- canonical key (patch labels removed) ------------
def canonical_key(assign: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    patch_map: Dict[int, int] = {}
    next_id, canon = 0, []
    for p, s in assign:
        if p not in patch_map:
            patch_map[p] = next_id; next_id += 1
        canon.append((patch_map[p], s))
    return tuple(canon)

# ------------------- main generator ----------------------------------
def unique_logical_permutations(code_sizes: List[int], q: int,
                                *, status_interval: float = 0.01):
    """Generate one representative assignment for every unique scenario.

    Prints an updating counter every `status_interval` seconds.
    """
    # Build slot list
    slots: List[Tuple[int, int]] = []
    for pid, n in enumerate(code_sizes):
        for k in range(logical_capacity(n)):
            slots.append((pid, k))

    reps, seen = [], set()
    last_print  = time.time()

    total_combo = len(list(combinations(slots, q)))   # upper bound
    est_total   = total_combo * math.factorial(q)                    # worst-case permutations
    # small q, so factorial inline:
    q_fact = 1
    for i in range(2, q + 1):
        q_fact *= i
    est_total = total_combo * q_fact

    if est_total > 1_000_000:
        raise ValueError("Expected permutations would take too long to compute." + 
            f"Est. {est_total} permutations max")


    for subset in combinations(slots, q):
        for ordered in permutations(subset, q):
            sig = canonical_key(list(ordered))
            if sig in seen:
                continue
            seen.add(sig)
            reps.append(list(ordered))

            # ---- live status ----------------------------------------
            now = time.time()
            if now - last_print >= status_interval:
                print(f"\rFound {len(reps)} unique scenarios "
                      f"(searching ≈{est_total} permutations max)", 
                      end="", flush=True)
                last_print = now

    # final status line
    print(f"\rFound {len(reps)} unique scenarios (done)                                    ")

    return reps

# Below are the functions that can generate lists of code sizes.
# These shouldn't be utilized until there are functions that support 
# operations between codes of different sizes. 
# -------------------------------------------------------------------
# generate_code_size_sets(q)
# -------------------------------------------------------------------
def generate_code_size_sets(q: int) -> List[List[int]]:
    """
    Produce every unordered list of code sizes (n values) that

      1) are even and ≥ 4
      2) do not exceed n_max = 2 q + 2
      3) use at most q individual codes (adding further n=4 codes is redundant)
      4) together provide capacity  ≥ q  ( Σ n_i/2 ≥ q )

    The returned lists are sorted ascending and represent unordered multisets
    (patch order is irrelevant).
    """
    n_min, n_max = 4, 2 * q + 2

    # Pre-compute every legal n and its logical capacity k = n/2
    legal_ns = [(n, n - 2) for n in range(n_min, n_max + 1, 2)]

    results = []

    def backtrack(start_idx: int,
                  remaining_q: int,
                  current: List[int]):
        """
        Recursive multiset enumeration.
        - start_idx: restricts candidates to retain ascending order (multiset)
        - remaining_q: logical qubits still to cover
        - current: current multiset (ascending)
        """
        # If we've satisfied capacity, store the multiset
        if remaining_q <= 0:
            results.append(current.copy())
            return
        # Stop if we've already used q codes (rule-3)
        if len(current) == q:
            return

        for idx in range(start_idx, len(legal_ns)):
            n, cap = legal_ns[idx]
            current.append(n)
            backtrack(idx, remaining_q - cap, current)   # idx (=same) ⇒ allow repeats
            current.pop()

    backtrack(0, q, [])
    return results

# ----------------------------------------------------------
# ECONOMIC search  (capacity K in [q , 2q])
# ----------------------------------------------------------
def economic_code_size_sets(q: int) -> List[List[int]]:
    n_min, n_max = 4, 2 * q           # upper bound tightened
    legal = [n for n in range(n_min, n_max + 1, 2)]

    results = []
    def backtrack(start: int, total_k: int, current: List[int]):
        # accept when q ≤ K ≤ 2q
        if q <= total_k <= 2 * q:
            results.append(current.copy())
        # stop if capacity already 2q  or #codes == q
        if total_k >= 2 * q or len(current) == q:
            return
        for i in range(start, len(legal)):
            n = legal[i]
            current.append(n)
            backtrack(i, total_k + logical_capacity(n), current)
            current.pop()

    backtrack(0, 0, [])
    return results

def uniform_code_families(q: int, *, n: int = 4) -> List[int]:
    """
    Return a single list with *q* identical code-sizes `n`.

    Parameters
    ----------
    q : int
        Number of logical qubits (⇒ number of codes returned).
    n : int, optional
        Physical size of each Iceberg code.  Must be even and ≥ 4.
        Default is 4.

    Returns
    -------
    List[int]
        `[n, n, …, n]`  (length = q)

    Raises
    ------
    ValueError
        If `n < 4` or `n` is odd.
    """
    if n < 4 or n % 2:
        raise ValueError("Code size n must be even and ≥ 4")

    return [n] * q

if __name__ == "__main__":
    print(uniform_code_families(q=3))  

    code_sizes = [4,4,4] # three n=4 codes (capacity 2 each)
    # calculate unique permutations with 5 logical qubits
    scenarios  = unique_logical_permutations(code_sizes, q=3)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print("unique scenarios:", len(scenarios))
    print("  ", end="")
    for i in range(len(scenarios[0])):
        print(alphabet[i % 26], end="       ")
    print()
    for s in scenarios:
        print(s)

    for q in range(1, 5):
        print(f"\nq = {q}")
        econ = economic_code_size_sets(q)
        for cs in econ:
            cap = sum(logical_capacity(n) for n in cs)
            print(f"  {cs}   capacity={cap}")

    for q in range(1, 5):
        print(f"\nq = {q}")
        for cs in generate_code_size_sets(q):
            cap = sum(n-2 for n in cs)
            print(f"  {cs}   capacity={cap}")
