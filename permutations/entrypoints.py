"""
create.py - Entry point for the permutations package.
Returns the permutations for [n] * q, n=4 by default.
"""

from typing import List, Tuple, Optional
from qiskit import QuantumCircuit
from .code_sets import generate_uniform_code_sets
from .generator import generate_permutations


def create_permutations(
    *,
    q: Optional[int] = None,
    program_circuit: Optional[QuantumCircuit] = None,
    status_interval: float = 0.25,
    n: int = 4,
) -> List[List[Tuple[int, int]]]:
    """
    Generate all unique permutations of logical qubits into a uniform layout.

    Parameters
    ----------
    q : int, optional
        Number of logical qubits.
    program_circuit : QuantumCircuit, optional
        The program circuit whose qubit count will be inferred.
    status_interval : float, optional
        Progress print interval (default is 0.25s).
    n : int, optional
        Code size of every patch. Default is 4, i.e. the [[4,2,2]] code.

    Returns
    -------
    List[List[Tuple[int, int]]]
        Unique permutations of (patch_id, slot_index) pairs.

    Raises
    ------
    ValueError
        If neither or both of `q` and `program_circuit` are provided.
    """
    if (q is None) == (program_circuit is None):
        raise ValueError("Specify exactly one of `q` or `program_circuit`")

    q_final = q if q is not None else program_circuit.num_qubits

    code_sizes = generate_uniform_code_sets(q_final, n=n)
    return generate_permutations(code_sizes, q_final, status_interval=status_interval)
