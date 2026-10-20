"""
Utilities for laying out logical patches on a physical grid.

The distance dependent noise model needs to know where every physical qubit resides so it
can compute pairwise distances. A logical patch of the [[n, n-2, 2]] code contains n + 2
physical qubits — n data plus 2 parity — and **each of them gets its own cell**.

That last part is the thing this module used to get wrong. Every qubit in a patch shared
one `(x, y)`, which made a patch dimensionless: intra-patch gates were charged distance
zero at every code size, and the site pitch stayed 1 whether the patch held 6 atoms or 14.
Both handed large `k` an advantage it had not earned. Now a patch is the most-square
rectangle that fits its atoms, one atom per cell, and patches tile at that rectangle's
size plus a gutter.

Coordinates are in **cell units**, not metres. `simulator.transport.Geometry` owns the
conversion, so a layout can be reasoned about as integers and priced separately.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from iceberg.layout import patch_count_of, patch_of_qubit, qubits_per_patch_of

from .transport import Geometry, most_square_rectangle

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

# Patch size of the [[4,2,2]] code, kept as the default so callers that predate
# the [[n, n-2, 2]] generalization keep working.
PHYSICAL_QUBITS_PER_PATCH = 6


def patch_grid_shape(num_patches: int) -> Tuple[int, int]:
    """`(columns, rows)` of the near-square arrangement of patch sites."""
    if num_patches <= 0:
        return (0, 0)
    columns = max(1, math.ceil(math.sqrt(num_patches)))
    return (columns, math.ceil(num_patches / columns))


def build_physical_qubit_positions(
    num_patches: int,
    *,
    column_major: bool = True,
    qubits_per_patch: int = PHYSICAL_QUBITS_PER_PATCH,
    geometry: Optional[Geometry] = None,
) -> Dict[int, Tuple[int, int]]:
    """
    Construct a mapping from physical qubit index to grid cell.

    Args:
        num_patches: Number of logical patches that must be embedded.
        column_major: If True (default) patch *sites* are filled down columns before
            advancing to the next column, matching the permutation ordering described in
            the simulator docs. Set to False for row-major fills.
        qubits_per_patch: Physical qubits owned by one patch, n + 2 for the
            [[n, n-2, 2]] code. Prefer `positions_for_circuit`, which reads this
            off the circuit instead of being told.
        geometry: Supplies the inter-patch gutter. Defaults to `Geometry()`.

    Returns:
        Dictionary mapping each physical qubit index to an (x, y) cell.

    Notes:
        - Patch sites form a near-square grid: ceil(sqrt(num_patches)) columns.
        - Within a patch, atoms fill the rectangle row-major.
        - Neighbouring patch rectangles are separated by `geometry.gutter` empty cells so
          they do not sit inside each other's blockade radius.
    """
    if num_patches <= 0:
        return {}
    geometry = geometry or Geometry()

    columns, rows = patch_grid_shape(num_patches)
    patch_rows, patch_columns = most_square_rectangle(qubits_per_patch)
    stride_x = patch_columns + geometry.gutter
    stride_y = patch_rows + geometry.gutter

    coords: Dict[int, Tuple[int, int]] = {}
    for patch_idx in range(num_patches):
        if column_major:
            site_x, site_y = patch_idx // rows, patch_idx % rows
        else:
            site_x, site_y = patch_idx % columns, patch_idx // columns

        origin_x, origin_y = site_x * stride_x, site_y * stride_y
        base_qubit = patch_idx * qubits_per_patch
        for offset in range(qubits_per_patch):
            coords[base_qubit + offset] = (
                origin_x + offset % patch_columns,
                origin_y + offset // patch_columns,
            )

    return coords


def build_patch_centres(
    num_patches: int,
    *,
    column_major: bool = True,
    qubits_per_patch: int = PHYSICAL_QUBITS_PER_PATCH,
    geometry: Optional[Geometry] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Centre cell of each patch rectangle. Inter-patch transport distance is measured here.

    Centres rather than nearest edges: an atom being carried to a partner patch ends up
    somewhere inside it, and which lane it is meeting is not a property the movement
    model tracks.
    """
    if num_patches <= 0:
        return {}
    geometry = geometry or Geometry()

    columns, rows = patch_grid_shape(num_patches)
    patch_rows, patch_columns = most_square_rectangle(qubits_per_patch)
    stride_x = patch_columns + geometry.gutter
    stride_y = patch_rows + geometry.gutter

    centres: Dict[int, Tuple[float, float]] = {}
    for patch_idx in range(num_patches):
        if column_major:
            site_x, site_y = patch_idx // rows, patch_idx % rows
        else:
            site_x, site_y = patch_idx % columns, patch_idx // columns
        centres[patch_idx] = (
            site_x * stride_x + (patch_columns - 1) / 2.0,
            site_y * stride_y + (patch_rows - 1) / 2.0,
        )
    return centres


def positions_for_circuit(
    circuit: "QuantumCircuit",
    *,
    column_major: bool = True,
    geometry: Optional[Geometry] = None,
) -> Dict[int, Tuple[int, int]]:
    """
    Grid cells for every physical qubit of an encoded circuit.

    Reads the patch count and patch size off the circuit's registers rather than
    inferring them by division, which is what makes this correct at any code size
    and in fault-tolerant mode — flag ancillas are appended after the patches, so
    `num_qubits // (n + 2)` overcounts. Flag ancillas are placed at the
    coordinate of the patch they interact with when that is unambiguous, and
    otherwise at the first patch's coordinate.
    """
    patch_map = patch_of_qubit(circuit)
    per_patch = qubits_per_patch_of(circuit)
    coords = build_physical_qubit_positions(
        patch_count_of(circuit),
        column_major=column_major,
        qubits_per_patch=per_patch,
        geometry=geometry,
    )

    # Ancillas (flag qubits) own no patch. Give each the coordinate of a patch it
    # actually touches so distance-dependent noise sees a sensible position.
    ancillas = [q for q in range(circuit.num_qubits) if q not in patch_map]
    if not ancillas:
        return coords

    neighbours: Dict[int, int] = {}
    for instruction in circuit.data:
        indices = [circuit.find_bit(q).index for q in instruction.qubits]
        patches = {patch_map[i] for i in indices if i in patch_map}
        if len(patches) != 1:
            continue
        patch = patches.pop()
        for index in indices:
            if index not in patch_map:
                neighbours.setdefault(index, patch)

    fallback = coords.get(0, (0, 0))
    for ancilla in ancillas:
        patch = neighbours.get(ancilla)
        if patch is None:
            coords[ancilla] = fallback
            continue
        coords[ancilla] = coords[patch * per_patch]
    return coords


def centres_for_circuit(
    circuit: "QuantumCircuit",
    *,
    column_major: bool = True,
    geometry: Optional[Geometry] = None,
) -> Dict[int, Tuple[float, float]]:
    """Patch centres for an encoded circuit, read off its registers."""
    return build_patch_centres(
        patch_count_of(circuit),
        column_major=column_major,
        qubits_per_patch=qubits_per_patch_of(circuit),
        geometry=geometry,
    )


if __name__ == "__main__":
    print(build_physical_qubit_positions(9))
