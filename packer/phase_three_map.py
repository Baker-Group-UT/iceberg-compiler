from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .phase_one_graph import GraphHeuristics


@dataclass
class PatchLayoutMetadata:
    """
    Metadata describing how logical patches are embedded into the 2-D grid.

    Attributes:
        grid_size: Length of one side of the square grid.
        old_to_new_patch: Mapping from Phase-Two patch ids to the new grid-aware ids.
        coordinates: Mapping of new patch ids to their (row, col) coordinates.
        ordering: Phase-Two patch ids ordered by their placement on the grid.
    """

    grid_size: int
    old_to_new_patch: Dict[int, int]
    coordinates: Dict[int, Tuple[int, int]]
    ordering: List[int]

    def coordinate_for_patch(self, patch_id: int) -> Tuple[int, int]:
        return self.coordinates[patch_id]


def compute_grid_aligned_mapping(
    graph: GraphHeuristics,
    qubit_to_patch: Dict[int, int],
) -> Tuple[Dict[int, int], PatchLayoutMetadata]:
    """
    Produce a new qubit->patch mapping whose patch ids encode a row-major grid layout.
    """
    if not qubit_to_patch:
        raise ValueError("Expected a non-empty qubit_to_patch mapping for phase three.")

    patch_ids = sorted(set(qubit_to_patch.values()))
    pair_weights = _extract_pair_weights(graph)
    interactions = _build_patch_interactions(patch_ids, qubit_to_patch, pair_weights)
    ordered_patches = _order_patches(interactions)

    num_qubits = len(graph.nodes) or len(qubit_to_patch)
    grid_size = max(1, math.ceil(math.sqrt(num_qubits)))
    max_cells = grid_size * grid_size
    if len(ordered_patches) > max_cells:
        raise ValueError(
            f"Grid of size {grid_size}x{grid_size} cannot host "
            f"{len(ordered_patches)} patches."
        )

    old_to_new: Dict[int, int] = {}
    coordinates: Dict[int, Tuple[int, int]] = {}
    for idx, patch_id in enumerate(ordered_patches):
        row, col = divmod(idx, grid_size)
        old_to_new[patch_id] = idx
        coordinates[idx] = (row, col)

    remapped = {qubit: old_to_new[patch] for qubit, patch in qubit_to_patch.items()}
    metadata = PatchLayoutMetadata(
        grid_size=grid_size,
        old_to_new_patch=old_to_new,
        coordinates=coordinates,
        ordering=ordered_patches,
    )
    return remapped, metadata


def _extract_pair_weights(graph: GraphHeuristics) -> Dict[Tuple[int, int], float]:
    """Return the interaction weight for every logical qubit pair."""
    weights: Dict[Tuple[int, int], float] = {}
    for edge in graph.edges:
        if edge.kind == "solo" or edge.j is None:
            continue
        i, j = sorted((edge.i, edge.j))
        node_i = graph.nodes.get(i)
        node_j = graph.nodes.get(j)
        if node_i is None or node_j is None:
            continue
        interaction = edge.weight - (node_i.base_cost + node_j.base_cost)
        if interaction <= 0:
            continue
        key = (i, j)
        previous = weights.get(key)
        if previous is None or interaction > previous:
            weights[key] = interaction
    return weights


def _build_patch_interactions(
    patch_ids: List[int],
    qubit_to_patch: Dict[int, int],
    pair_weights: Dict[Tuple[int, int], float],
) -> Dict[int, Dict[int, float]]:
    interactions: Dict[int, Dict[int, float]] = {patch: {} for patch in patch_ids}
    for (qi, qj), weight in pair_weights.items():
        patch_i = qubit_to_patch.get(qi)
        patch_j = qubit_to_patch.get(qj)
        if patch_i is None or patch_j is None or patch_i == patch_j:
            continue
        interactions[patch_i][patch_j] = interactions[patch_i].get(patch_j, 0.0) + weight
        interactions[patch_j][patch_i] = interactions[patch_j].get(patch_i, 0.0) + weight

    # Ensure the graph is fully connected (zero-weight edges are allowed).
    for outer in patch_ids:
        for inner in patch_ids:
            if outer == inner:
                continue
            interactions[outer].setdefault(inner, 0.0)
    return interactions


def _order_patches(interactions: Dict[int, Dict[int, float]]) -> List[int]:
    patches = list(interactions.keys())
    if len(patches) <= 1:
        return patches

    total_weights = {patch: sum(neighbors.values()) for patch, neighbors in interactions.items()}
    first = max(patches, key=lambda pid: (total_weights.get(pid, 0.0), -pid))
    order = [first]
    remaining = set(patches)
    remaining.remove(first)

    while remaining:
        def candidate_score(pid: int) -> Tuple[float, float, int]:
            adjacency = sum(interactions[pid].get(prev, 0.0) for prev in order)
            total = total_weights.get(pid, 0.0)
            return (adjacency, total, -pid)

        next_patch = max(remaining, key=candidate_score)
        order.append(next_patch)
        remaining.remove(next_patch)
    return order
