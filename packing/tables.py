"""
tables.py — the cost tables and timeline predicates the packing model reads.

Ported verbatim out of `grouping.analytic` when `packing` became self-contained.
These are the parts that describe *the code and the circuit*, as opposed to the parts
that scored a placement -- `AnalyticCost` itself did not come across, because
`packing.cost` replaced it.

Three groups:

*gate tables* -- `GATE_KEYS` maps a qiskit gate name to a key in `iceberg.costs`, and
    `CROSS_PATCH_COST` overrides the intra-patch entry for the two-qubit gates whose
    cost changes when their operands sit in different patches. `CX` is deliberately
    absent: intra and cross cost identical gates, which is why depth has to be priced
    for a CNOT-heavy program to carry any placement signal at all.

*geometry* -- `mean_patch_hops` reproduces `simulator.grid_constructor`'s own layout
    convention exactly (ceil(sqrt(P)) columns, column-major) so the model predicts the
    geometry the distance-dependent noise model will later charge for.

*timeline predicates* -- `_coincident_layers`, `_fusable_group_sizes` and
    `_complete_bijections` are how merge savings are estimated without running the
    merging pass. All three are conjunctive and windowed: HH needs every occupied lane
    to want a Hadamard within `window` layers, a transversal macro needs k crossings
    coincident in time with distinct endpoints. They model none of the scheduler, which
    is where the model's 2-3% merge over-count comes from.

`translated_view` is here too, and is the single most important function in the file:
the translator rewrites gate *kinds* before it places anything, and every table above
is keyed by kind. Scoring the raw program instead was worth a 13-49% error.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from iceberg.costs import depth_table

__all__ = [
    "CROSS_PATCH_COST",
    "GATE_KEYS",
    "DEFAULT_WINDOW",
    "gate_depths",
    "mean_patch_hops",
    "translated_view",
]


#: Gates whose cross-patch construction differs from the intra-patch one. Everything
#: else costs the same either way, which is why co-location buys less here than it
#: would in a code with expensive inter-block operations. Canonical copy lives in
#: `iceberg.costs` (single source of truth); re-exported here for existing importers.
from iceberg.costs import CROSS_PATCH_COST  # noqa: E402  (re-export)


#: Program gate name -> cost-table key.
GATE_KEYS: Dict[str, str] = {
    "x": "X", "y": "Y", "z": "Z", "h": "H", "s": "S", "sdg": "S",
    "sx": "SX", "sxdg": "SX", "rz": "RZ", "rx": "RX",
    "cx": "CX", "cnot": "CX", "cz": "CZ", "xcx": "XCX", "swap": "SWAP",
}


#: Layer tolerance for calling two gates simultaneous, matching the merging pass's
#: willingness to delay a gate by a few layers to make a macro fire.
DEFAULT_WINDOW = 2


#: Mean patch separation, cached by (patch count, patch size, gutter). Small
#: dictionary, computed once per shape. Kept for the callers that legitimately want a
#: single number -- `packing.resource.noise`, which prices a code size before any
#: placement exists and so has no patch pairs to ask about.
_MEAN_HOPS: Dict[Tuple[int, int, int], float] = {}

#: Full pair-distance matrix, same key. The placement objective reads this one; see
#: `patch_hop_matrix` for why the mean is not usable there. One matrix per shape, and P is
#: at most a few hundred, so the memory is trivial next to the search it serves.
_HOP_MATRIX: Dict[Tuple[int, int, int], Tuple[Tuple[float, ...], ...]] = {}


def gate_depths(n_code: int) -> Dict[str, Tuple[int, int]]:
    """(intra, cross) depth per two-qubit gate key, at one code size."""
    table = depth_table(n_code)
    return {
        key: (table[key], table.get(f"{key}_CROSS", table[key]))
        for key in ("CX", "CZ", "XCX")
    }


def mean_patch_hops(num_patches: int, qubits_per_patch: int) -> float:
    """
    Mean Manhattan distance between two distinct patch centres, in **cells**.

    The layout convention is `simulator.grid_constructor` verbatim -- `ceil(sqrt(P))`
    columns of patch sites, column-major fill, each site a most-square rectangle of
    `qubits_per_patch` atoms tiled at that rectangle's size plus the gutter -- so the
    model predicts exactly the centre-to-centre distances
    `simulator.transport.derive_transport_groups` will charge. Enumerated rather than
    approximated: P is at most a few hundred, and the result is cached per shape.

    This used to price a patch as a dimensionless point at unit pitch (the pre-transport
    geometry). Now one hop in x is `patch_columns + gutter` cells and one hop in y is
    `patch_rows + gutter`, so the distance both grows like `sqrt(P)` *and* carries the
    patch size -- a k=10 grid is coarser than a k=2 grid, which is part of what makes
    spreading a program over patches cost what the simulator says it costs. Multiply by
    `Geometry.pitch` for metres. One patch means no crossings and no distance, hence 0.

    Intra-patch extent is deliberately *not* here: which pairs sit beyond blockade
    inside a patch is fixed by the code size, not by the placement, so it is priced in
    `packing.resource.noise` where code sizes are chosen.
    """
    if num_patches < 2:
        return 0.0
    matrix = patch_hop_matrix(num_patches, qubits_per_patch)
    total = sum(
        matrix[first][second]
        for first in range(num_patches)
        for second in range(first + 1, num_patches)
    )
    pairs = num_patches * (num_patches - 1) // 2
    return total / pairs if pairs else 0.0


def patch_hop_matrix(num_patches: int, qubits_per_patch: int):
    """
    Manhattan centre-to-centre distance between every pair of patch sites, in cells.

    The per-pair truth that `mean_patch_hops` averages away. A placement objective needs
    this and not the mean: at a fixed patch count the mean is a constant, so an objective
    built on it charges every crossing alike and cannot tell a placement that keeps
    interacting qubits in neighbouring patches from one that puts them at opposite corners.
    Measured consequence of that blindness, on `ising_n34` at k = 4 -- a chain, where naive
    index order is already near distance-optimal -- the search cut crossing *count* by 16
    and tripled the real movement, because both placements used 9 patches and were
    therefore priced identically.

    `[a][b]` is symmetric with a zero diagonal. Cached per shape, like the mean: the layout
    is a function of `(num_patches, qubits_per_patch, PATCH_GUTTER)` alone.

    **Row-major within a column, columns first**, matching `simulator.grid_constructor`
    exactly -- `ceil(sqrt(P))` columns, column-major fill, each site a most-square
    rectangle of `qubits_per_patch` atoms at that rectangle's size plus the gutter. Index
    `i` is grid site `i`, which is the same order the translator labels patches in, so a
    group's position in the partition *is* its position on the grid.
    """
    from simulator.transport import PATCH_GUTTER, most_square_rectangle

    key = (num_patches, qubits_per_patch, PATCH_GUTTER)
    cached = _HOP_MATRIX.get(key)
    if cached is not None:
        return cached

    patch_rows, patch_columns = most_square_rectangle(qubits_per_patch)
    stride_x = patch_columns + PATCH_GUTTER
    stride_y = patch_rows + PATCH_GUTTER

    columns = max(1, math.ceil(math.sqrt(num_patches)))
    rows = math.ceil(num_patches / columns)
    coords = [(index // rows, index % rows) for index in range(num_patches)]

    matrix = tuple(
        tuple(
            abs(coords[first][0] - coords[second][0]) * stride_x
            + abs(coords[first][1] - coords[second][1]) * stride_y
            for second in range(num_patches)
        )
        for first in range(num_patches)
    )
    _HOP_MATRIX[key] = matrix
    return matrix


def translated_view(
    program: QuantumCircuit, enable_h_commutation: bool = True
) -> QuantumCircuit:
    """
    The circuit the translator actually places, after its program-level passes.

    These passes run before placement and do not depend on it, so they can be
    applied once and reused for every candidate. Everything that scores a placement
    -- this model, the interaction graph, the partitioner -- must work from this
    view rather than the raw program.
    """
    from translator.logical_optimization import logically_optimized
    from translator.passes import cancel_inverse_gates, commute_hadamards

    # The translator's first pass, the round of logical optimisation, comes first here too.
    processed = cancel_inverse_gates(logically_optimized(program))
    if enable_h_commutation:
        processed = commute_hadamards(processed)
        processed = cancel_inverse_gates(processed)
    return processed


def _coincident_layers(
    layer_sets: Sequence[FrozenSet[int]], window: int
) -> List[int]:
    """
    Layers where *every* set has an entry within `window`.

    Used for the conjunctive HH term: one returned layer is one macro.
    """
    if not layer_sets or any(not s for s in layer_sets):
        return []
    anchor = sorted(layer_sets[0])
    found: List[int] = []
    used: List[set] = [set() for _ in layer_sets]
    for layer in anchor:
        picks = []
        for index, layers in enumerate(layer_sets):
            candidate = next(
                (
                    other
                    for other in sorted(layers)
                    if other not in used[index] and abs(other - layer) <= window
                ),
                None,
            )
            if candidate is None:
                break
            picks.append(candidate)
        if len(picks) == len(layer_sets):
            for index, pick in enumerate(picks):
                used[index].add(pick)
            found.append(layer)
    return found


def _fusable_group_sizes(
    layer_sets: Sequence[FrozenSet[int]], window: int
) -> List[int]:
    """
    Sizes of the groups of coincident Paulis, one entry per fusable group.

    Unlike HH this does not need every lane, so a group of any size >= 2 counts.
    """
    occurrences: List[Tuple[int, int]] = []
    for index, layers in enumerate(layer_sets):
        occurrences.extend((layer, index) for layer in layers)
    occurrences.sort()

    sizes: List[int] = []
    used_lanes: set = set()
    current: List[Tuple[int, int]] = []
    for layer, index in occurrences:
        if current and (layer - current[0][0] > window or index in used_lanes):
            if len(current) >= 2:
                sizes.append(len(current))
            current, used_lanes = [], set()
        current.append((layer, index))
        used_lanes.add(index)
    if len(current) >= 2:
        sizes.append(len(current))
    return sizes


def _complete_bijections(
    events: Sequence[Tuple[int, int, int]], lanes: int, window: int
) -> int:
    """
    How many times `lanes` crossings line up in time with distinct endpoints.

    One such set is one transversal macro. Greedy over layers, which is what the
    detector does too -- it accumulates candidates and fires as soon as it holds a
    full bijection.
    """
    if lanes <= 0:
        return 0
    ordered = sorted(events)
    macros = 0
    sources: set = set()
    partners: set = set()
    start: Optional[int] = None
    for layer, source, partner in ordered:
        if start is not None and layer - start > window:
            sources, partners, start = set(), set(), None
        if source in sources or partner in partners:
            sources, partners, start = set(), set(), None
        if start is None:
            start = layer
        sources.add(source)
        partners.add(partner)
        if len(sources) == lanes and len(partners) == lanes:
            macros += 1
            sources, partners, start = set(), set(), None
    return macros
