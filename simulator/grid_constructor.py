"""
Utilities for laying out logical patches on a physical grid.

The distance dependent noise model needs to know where every physical qubit
resides so it can compute pairwise distances. Each logical patch contains six
physical qubits, and (per our simulator assumptions) every qubit in a patch
shares the same `(x, y)` coordinate.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

PHYSICAL_QUBITS_PER_PATCH = 6


def build_physical_qubit_positions(
    num_patches: int, *, column_major: bool = True
) -> Dict[int, Tuple[int, int]]:
    """
    Construct a mapping from physical qubit index to grid coordinates.

    Args:
        num_patches: Number of logical patches that must be embedded.
        column_major: If True (default) patches are filled down columns before
            advancing to the next column, matching the permutation ordering
            described in the simulator docs. Set to False for row-major fills.

    Returns:
        Dictionary mapping each physical qubit index to an (x, y) coordinate.

    Notes:
        - A near-square grid is chosen by setting the column count to
          ceil(sqrt(num_patches)) and deriving the number of rows accordingly.
        - All qubits belonging to the same patch share the same coordinate.
    """
    if num_patches <= 0:
        return {}

    columns = max(1, math.ceil(math.sqrt(num_patches)))
    rows = math.ceil(num_patches / columns)

    coords: Dict[int, Tuple[int, int]] = {}
    for patch_idx in range(num_patches):
        if column_major:
            x = patch_idx // rows
            y = patch_idx % rows
        else:
            x = patch_idx % columns
            y = patch_idx // columns

        base_qubit = patch_idx * PHYSICAL_QUBITS_PER_PATCH
        for offset in range(PHYSICAL_QUBITS_PER_PATCH):
            coords[base_qubit + offset] = (x, y)

    return coords


if __name__ == "__main__":
    print(build_physical_qubit_positions(9))