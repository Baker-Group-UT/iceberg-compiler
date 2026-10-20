"""
Grid alignment: the patch relabelling that decides where a packing sits on the array.

The invariants matter more than the heuristic here. Alignment is layered on top of a
finished placement in a cumulative ablation, so if it silently moved a qubit between
patches it would be crediting itself with the packer's work.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
from qiskit import QuantumCircuit

from packing.alignment import (
    align_to_grid,
    patch_traffic,
    weighted_patch_distance,
)
from simulator.grid_constructor import patch_grid_shape


def _groups(permutation) -> List[Tuple[int, ...]]:
    """Qubit sets sharing a patch, order-independent, so two layouts can be compared."""
    by_patch: dict = {}
    for qubit, (patch, _) in enumerate(permutation):
        by_patch.setdefault(patch, []).append(qubit)
    return sorted(tuple(sorted(members)) for members in by_patch.values())


def _chain(num_qubits: int, repeats: int = 3) -> QuantumCircuit:
    """Nearest-neighbour CX chain: traffic is local, so layout has something to exploit."""
    circuit = QuantumCircuit(num_qubits)
    for _ in range(repeats):
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
    return circuit


def _hub(num_qubits: int, repeats: int = 4) -> QuantumCircuit:
    """Star on qubit 0 -- one patch carries everything."""
    circuit = QuantumCircuit(num_qubits)
    for _ in range(repeats):
        for qubit in range(1, num_qubits):
            circuit.cx(0, qubit)
    return circuit


def _pairs(num_qubits: int) -> List[Tuple[int, int]]:
    """Two logical qubits per patch, in program order."""
    return [(qubit // 2, qubit % 2) for qubit in range(num_qubits)]


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("program", [_chain(12), _hub(12), _chain(16, 2)])
def test_groups_are_preserved_exactly(program):
    """The whole premise: alignment moves patches, never qubits between patches."""
    permutation = _pairs(program.num_qubits)
    assert _groups(align_to_grid(permutation, program)) == _groups(permutation)


def test_slots_are_preserved(program=_chain(12)):
    permutation = _pairs(program.num_qubits)
    aligned = align_to_grid(permutation, program)
    assert [slot for _, slot in aligned] == [slot for _, slot in permutation]


def test_patch_ids_stay_dense(program=_chain(12)):
    aligned = align_to_grid(_pairs(program.num_qubits), program)
    patches = {patch for patch, _ in aligned}
    assert patches == set(range(len(patches)))


def test_sparse_input_is_compacted():
    """A caller may hand in non-contiguous patch ids; the grid needs dense ones."""
    program = _chain(6)
    sparse = [(0, 0), (0, 1), (5, 0), (5, 1), (9, 0), (9, 1)]
    aligned = align_to_grid(sparse, program)
    assert {patch for patch, _ in aligned} == {0, 1, 2}
    assert _groups(aligned) == _groups(sparse)


def test_alignment_is_deterministic(program=_chain(14)):
    permutation = _pairs(program.num_qubits)
    assert align_to_grid(permutation, program) == align_to_grid(permutation, program)


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "program", [_chain(12), _chain(16), _chain(20), _hub(12), _hub(18)]
)
def test_never_increases_weighted_distance(program):
    """
    The guard that makes the ablation arm safe to read.

    The ordering is greedy and greedy loses sometimes -- on a star graph, centring the hub
    buys nothing the default order was not already doing. Scoring both layouts costs one
    pass over the gate list and no simulation, so the pass keeps whichever is cheaper.
    """
    permutation = _pairs(program.num_qubits)
    before = weighted_patch_distance(permutation, program)
    after = weighted_patch_distance(align_to_grid(permutation, program), program)
    assert after <= before


@pytest.mark.parametrize("program", [_hub(12), _hub(18)])
def test_a_hub_actually_improves(program):
    """
    Hub topologies are what alignment is for: one patch talks to everything, and putting
    it in a corner makes every conversation long. Measured 72 -> 56 at 12 qubits.
    """
    permutation = _pairs(program.num_qubits)
    before = weighted_patch_distance(permutation, program)
    after = weighted_patch_distance(align_to_grid(permutation, program), program)
    assert after < before


@pytest.mark.parametrize("program", [_chain(12), _chain(20)])
def test_a_chain_has_nothing_to_win(program):
    """
    Not a weakness -- program order already lays a chain out well.

    Algorithms get written `q[i]`-`q[i+1]`, packing puts consecutive qubits in consecutive
    patches, and the simulator fills sites column-major, so neighbouring patches are
    already neighbouring sites. `packing.cost.MOVEMENT_SHARE_CEILING` records the same
    effect from the other direction: program order nearly minimises crossings on the chain
    and fan-out families. Alignment ties here, and the guard keeps it from making things
    worse in the attempt.
    """
    permutation = _pairs(program.num_qubits)
    before = weighted_patch_distance(permutation, program)
    after = weighted_patch_distance(align_to_grid(permutation, program), program)
    assert after == before


def test_busiest_patch_lands_near_the_centre():
    program = _hub(14)
    permutation = _pairs(program.num_qubits)
    aligned = align_to_grid(permutation, program)

    traffic = patch_traffic(aligned, program)
    load: dict = {}
    for (left, right), count in traffic.items():
        load[left] = load.get(left, 0) + count
        load[right] = load.get(right, 0) + count
    busiest = max(load, key=lambda patch: load[patch])

    patches = len({patch for patch, _ in aligned})
    columns, rows = patch_grid_shape(patches)
    centre_x, centre_y = (columns - 1) / 2.0, (rows - 1) / 2.0

    def offset(patch: int) -> float:
        return abs(patch // rows - centre_x) + abs(patch % rows - centre_y)

    assert offset(busiest) == min(offset(patch) for patch in range(patches))


# --------------------------------------------------------------------------
# degenerate input
# --------------------------------------------------------------------------


def test_empty_permutation():
    assert align_to_grid([], _chain(4)) == []


def test_single_patch_is_returned_unchanged():
    program = _chain(2)
    assert align_to_grid([(0, 0), (0, 1)], program) == [(0, 0), (0, 1)]


def test_no_cross_patch_traffic_is_a_no_op():
    """Every gate intra-patch: nothing crosses, so no layout is cheaper than another."""
    program = QuantumCircuit(4)
    program.cx(0, 1)
    program.cx(2, 3)
    permutation = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert align_to_grid(permutation, program) == permutation
    assert patch_traffic(permutation, program) == {}


def test_single_qubit_program_has_no_traffic():
    program = QuantumCircuit(4)
    for qubit in range(4):
        program.h(qubit)
    permutation = _pairs(4)
    assert patch_traffic(permutation, program) == {}
    assert weighted_patch_distance(permutation, program) == 0.0
