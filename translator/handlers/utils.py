"""
translator.gate_handlers.utils

Shared helpers for logical gate placement logic on Iceberg circuits.
"""

from __future__ import annotations
from typing import Dict, List

def is_patch_solo(logical_idx: int, active_by_patch: Dict[int, List[int]], logical_to_patch: Dict[int, int]) -> bool:
    """
    Returns True if the given logical qubit is the only active qubit in its patch.

    Args:
        logical_idx: Index of the logical qubit.
        active_by_patch: Mapping from patch index to list of active logical qubits.
        logical_to_patch: Mapping from logical qubit to its patch index.

    Returns:
        True if the patch contains only this qubit, False otherwise.
    """
    patch = logical_to_patch[logical_idx]
    return active_by_patch.get(patch, []).count(logical_idx) == 1 and len(active_by_patch[patch]) == 1

def can_transversal_pair(q1: int, q2: int, idx_in_patch: Dict[int, int]) -> bool:
    """
    Checks whether two logical qubits are aligned for transversal operation.

    Args:
        q1: First logical qubit index.
        q2: Second logical qubit index.
        idx_in_patch: Mapping from logical qubit index to index within its patch.

    Returns:
        True if the two qubits are aligned for transversal ops (i.e., same index in patch).
    """
    return idx_in_patch[q1] == idx_in_patch[q2]

def verbose_log(verbose: bool, message: str):
    """
    Prints a message if verbosity is enabled.

    Args:
        verbose: Boolean flag for verbosity.
        message: String message to print.
    """
    if verbose:
        print(message)

def are_all_partners_zero(logical_idx: int, active_by_patch: Dict[int, List[int]], logical_to_patch: Dict[int, int], state_map: Dict[int, str]) -> bool:
    """
    Returns True if all active qubits in the same patch as logical_idx (excluding itself) are in the |0> state.
    """
    patch = logical_to_patch[logical_idx]
    return all(state_map[q] == "0" for q in active_by_patch[patch] if q != logical_idx)

def is_full_transversal(ctrls: List[int], targs: List[int], idx_in_patch: Dict[int, int]) -> bool:
    """
    Returns True if every control qubit has at least one target qubit with matching index in patch.
    """
    return all(any(idx_in_patch[q] == idx_in_patch[r] for r in targs) for q in ctrls)

def is_alone_in_patch(logical_idx: int, active_by_patch: Dict[int, List[int]], logical_to_patch: Dict[int, int]) -> bool:
    """
    Returns True if the given logical qubit is the only active one in its patch.
    """
    patch = logical_to_patch[logical_idx]
    return len(active_by_patch.get(patch, [])) == 1
