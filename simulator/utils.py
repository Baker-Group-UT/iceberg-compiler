"""
simulator/utils.py

Helper functions for distribution postprocessing and fidelity computation.
As well as noise scaling utilities for use with the SQALE leakage simulator.
Includes helpers for creating scaled Hilbert noise parameter sets.
"""

from typing import Dict
from math import sqrt


def bhattacharyya(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute Bhattacharyya coefficient fidelity between two distributions.

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        Classical Bhattacharyya fidelity.
    """
    keys = set(p) | set(q)
    return sum(sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)

def normalize_distribution(dist: dict[str, float]) -> dict[str, float]:
    """
    Normalize a probability distribution so values sum to 1.
    Handles empty dicts safely.
    """
    total = sum(dist.values())
    if total == 0:
        return {k: 0.0 for k in dist}
    return {k: v / total for k, v in dist.items()}

def split_qiskit_bitstring(
    bitstring: str,
    key_order: list[str],
) -> dict[str, str]:
    """
    Split a space-delimited Qiskit bitstring into a dict keyed by register name.
    Example:
        '00 10 0101 1111' + ['m_parity_0','m_parity_1','m_data_0','m_data_1']
        → {'m_parity_0':'00','m_parity_1':'10','m_data_0':'0101','m_data_1':'1111'}
    """
    parts = bitstring.split()
    if len(parts) != len(key_order):
        raise ValueError(f"Bitstring/register count mismatch: {len(parts)} vs {len(key_order)}")
    return dict(zip(key_order, parts))
