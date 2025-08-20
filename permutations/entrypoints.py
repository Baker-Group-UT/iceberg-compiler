"""
create.py - Entry point for the permutations package. 
Returns the permutations for [4] * q.
"""
from typing import List, Tuple, Optional
import cirq
from .code_sets import generate_uniform_code_sets
from .generator import generate_permutations


def create_permutations(
    *,
    q: Optional[int] = None,
    program_circuit: Optional[cirq.Circuit] = None,
    status_interval: float = 0.25,
) -> List[List[Tuple[int, int]]]:
    """
    Generate all unique permutations of logical qubits into a uniform [n=4] layout.

    Parameters
    ----------
    q : int, optional
        Number of logical qubits.
    program_circuit : cirq.Circuit, optional
        the program circuit whose program qubit count will be inferred.
    status_interval : float, optional
        Progress print interval (default is 0.25s).

    Returns
    -------
    List[List[Tuple[int, int]]]
        Unique permutations of (patch_id, slot_index) pairs.

    Raises
    ------
    ValueError
        If neither or both of `q` and `circuit` are provided.
    """
    if (q is None) == (program_circuit is None):
        raise ValueError("Specify exactly one of `q` or `circuit`")

    q_final = q if q is not None else len(program_circuit.all_qubits())

    code_sizes = generate_uniform_code_sets(q_final)
    return generate_permutations(code_sizes, q_final, status_interval=status_interval)