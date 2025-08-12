from math import sqrt
from typing import Dict

__all__ = ["bhattacharyya"]

def bhattacharyya(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Classical (Bhattacharyya) fidelity between two probability distributions."""
    keys = set(p) | set(q)
    return sum(sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)