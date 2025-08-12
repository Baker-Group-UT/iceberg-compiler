from __future__ import annotations

from math import acos, pi
from typing import Tuple

import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from iceberg.IcebergCode import IcebergGrid

__all__ = [
    "random_bloch_point",
    "prepare_random_logical_state",
    "prepare_random_basis_state",
]


def random_bloch_point(rng=np.random.default_rng()) -> Tuple[float, float]:
    """Uniform point on the Bloch sphere |ψ⟩ = cosθ|0⟩ + e^{iφ}sinθ|1⟩."""
    u1, u2 = rng.random(2)
    theta = acos(1 - 2 * u1) / 2           # θ ∈ [0, π/2]
    phi = 2 * pi * u2                      # φ ∈ [0, 2π)
    return theta, phi


def prepare_random_logical_state(
    grid: IcebergGrid, rng=np.random.default_rng()
) -> None:
    """Prepare a genuinely arbitrary 2-logical-qubit state on patch 0."""
    for qubit_idx in (0, 1):
        theta, phi = random_bloch_point(rng)
        if theta:
            grid.RY(qubit_idx, 2 * theta)
        if phi:
            grid.RZ(qubit_idx, phi)


def prepare_random_basis_state(
    grid: IcebergGrid, rng=np.random.default_rng()
) -> None:
    """Random |00⟩..|11⟩ logical basis state on patch 0."""
    bits = rng.integers(0, 4)  # 00..11
    if bits & 1:
        grid.X(0)
    if bits & 2:
        grid.X(1)