"""
code_sets.py - Responsible for generating the lists of code
sizes to search in a permutation scan. I.e. [4,4,4] = 3 n=4 modules.
"""
from typing import List
from iceberg.utils import logical_capacity


def generate_all_code_sets(q: int) -> List[List[int]]:
    """
    Generate all unordered multisets of code sizes (even n ≥ 4)
    that provide capacity ≥ q logical qubits.

    Constraints:
        - Each code size n must be even and ≥ 4.
        - Total number of codes used ≤ q.
        - Each list must satisfy: sum(logical_capacity(n_i)) ≥ q.
        - n values are in ascending order (multisets).

    Parameters
    ----------
    q : int
        Number of logical qubits needed.

    Returns
    -------
    List[List[int]]
        All valid multisets of code sizes (ascending order).
    """
    n_min, n_max = 4, 2 * q + 2
    legal_ns = [(n, logical_capacity(n)) for n in range(n_min, n_max + 1, 2)]

    results = []

    def backtrack(start: int, remaining_q: int, current: List[int]):
        if remaining_q <= 0:
            results.append(current.copy())
            return
        if len(current) == q:
            return
        for i in range(start, len(legal_ns)):
            n, cap = legal_ns[i]
            current.append(n)
            backtrack(i, remaining_q - cap, current)
            current.pop()

    backtrack(0, q, [])
    return results


def generate_economic_code_sets(q: int) -> List[List[int]]:
    """
    Generate efficient code size sets with logical capacity K in [q, 2q].

    These multisets aim to minimize unused capacity by ensuring:
        q ≤ total capacity ≤ 2q.

    Parameters
    ----------
    q : int
        Number of logical qubits needed.

    Returns
    -------
    List[List[int]]
        All valid multisets of code sizes with minimal waste.
    """
    n_min, n_max = 4, 2 * q
    legal_ns = [n for n in range(n_min, n_max + 1, 2)]

    results = []

    def backtrack(start: int, total_k: int, current: List[int]):
        if q <= total_k <= 2 * q:
            results.append(current.copy())
        if total_k >= 2 * q or len(current) == q:
            return
        for i in range(start, len(legal_ns)):
            n = legal_ns[i]
            current.append(n)
            backtrack(i, total_k + logical_capacity(n), current)
            current.pop()

    backtrack(0, 0, [])
    return results


def generate_uniform_code_sets(q: int, *, n: int = 4) -> List[int]:
    """
    Generate a list of `q` identical code sizes.

    Parameters
    ----------
    q : int
        Number of logical qubits (also number of code patches).
    n : int, optional
        Size of each code (even, ≥ 4). Default is 4.

    Returns
    -------
    List[int]
        List containing `q` copies of `n`.

    Raises
    ------
    ValueError
        If `n` is not even or < 4.
    """
    if n < 4 or n % 2 != 0:
        raise ValueError("Code size n must be even and ≥ 4")
    return [n] * q
