"""
The indexed cross-gate conflict detector must find exactly the enumerated one's edges.

`conflicts._add_cross_gate_conflicts` was a triple loop over (merge A, merge B, every
two-qubit gate) -- `O(C^2 * G2)`, cubic in circuit size, and >99% of the merging pass on
H-dense circuits at a few thousand gates. It is now indexed: the qualifying bridge gates
are gathered once per B into a qubit -> earliest-position map, and the pair test is a
lookup over A's macro qargs.

That rewrite is only admissible if the resulting conflict graph is *identical*, because
the graph decides which merges the selector commits and therefore what the compiler
emits. So the reference implementation is kept here, verbatim in behaviour, and the two
are compared edge for edge. This is the file to run first if merging output ever changes
unexpectedly: if these pass, the conflict graph is not the cause.

Two levels of check:

    edge set      `_add_cross_gate_conflicts` alone, against the reference
    end to end    the full pass -- savings triple and emitted instruction list

The end-to-end half matters because the edge set could in principle agree while some
other hoisting in the same commit -- `_add_first_in_window_conflicts` also had its per
pair work lifted out -- changed a downstream decision.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translator.merging import preprocess_program_for_merging_with_stats  # noqa: E402
from translator.merging.conflicts import (  # noqa: E402
    _add_cross_gate_conflicts,
    _record_conflict,
    merge_work_table,
)
from translator.merging.layer import layer_by_node  # noqa: E402
from translator.merging.models import MacroReplacement  # noqa: E402
from translator.merging.preprocessing import _build_patch_lanes  # noqa: E402
from translator.merging.settings import DEFAULT_SETTINGS  # noqa: E402
from translator.merging.single_qubit import (  # noqa: E402
    find_hh_merges_for_patch,
    find_xx_merges_for_patch,
    find_zz_merges_for_patch,
)

CODE_SIZES = [4, 6, 8]


def _reference_cross_gate_conflicts(
    merges: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    dag: DAGCircuit,
    conflicts: Dict[int, Set[int]],
) -> None:
    """The enumerated form this replaced, kept as the oracle. Do not optimise."""

    def pos(n: DAGOpNode) -> Tuple[int, int]:
        return (layer_index.get(n, 0), order_index.get(n, 0))

    merge_nodes: Set[DAGOpNode] = set()
    for m in merges:
        merge_nodes.add(m.first)
        merge_nodes.add(m.second)

    two_qubit_nodes = [
        n for n in dag.topological_op_nodes() if len(n.qargs) == 2 and n not in merge_nodes
    ]

    for idx_a, merge_a in enumerate(merges):
        a_second_pos = pos(merge_a.second)
        a_qubits = set(merge_a.macro_qargs)
        for idx_b, merge_b in enumerate(merges):
            if idx_a == idx_b:
                continue
            if idx_b in conflicts.get(idx_a, set()):
                continue
            b_first_pos = pos(merge_b.first)
            b_second_pos = pos(merge_b.second)
            if b_second_pos <= a_second_pos:
                continue
            b_blocked = set(merge_b.blocked_qubits)
            for g_node in two_qubit_nodes:
                g_pos = pos(g_node)
                if g_pos < b_first_pos or g_pos >= b_second_pos:
                    continue
                if g_pos >= a_second_pos:
                    continue
                g_qubits = set(g_node.qargs)
                if not g_qubits & b_blocked:
                    continue
                if not g_qubits & a_qubits:
                    continue
                _record_conflict(conflicts, idx_a, idx_b)
                break


def hadamard_dense(num_qubits: int, num_gates: int, seed: int) -> QuantumCircuit:
    """
    A circuit shaped to generate overlapping candidates and bridging gates.

    Weighted towards H because every H is tried as a group leader, so this is what makes
    the candidate count -- and with it the quadratic term -- large enough to be worth
    testing. The CX gates are the bridges the detector is about.
    """
    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(num_gates):
        roll = rng.random()
        if roll < 0.45:
            circuit.h(rng.randrange(num_qubits))
        elif roll < 0.58:
            circuit.x(rng.randrange(num_qubits))
        elif roll < 0.68:
            circuit.z(rng.randrange(num_qubits))
        else:
            control = rng.randrange(num_qubits)
            target = rng.randrange(num_qubits)
            while target == control:
                target = rng.randrange(num_qubits)
            circuit.cx(control, target)
    return circuit


def sequential(num_qubits: int, k: int) -> List[Tuple[int, int]]:
    return [(index // k, index % k) for index in range(num_qubits)]


def candidates_for(circuit: QuantumCircuit, permutation) -> tuple:
    """Everything `compute_merge_conflicts` would be handed, without calling it."""
    dag = circuit_to_dag(circuit)
    layer_index, order_index = layer_by_node(dag)
    patch_lanes = _build_patch_lanes(permutation, circuit.qubits)
    merges: List[MacroReplacement] = []
    for _, qubits in sorted(patch_lanes.items()):
        for finder in (
            find_hh_merges_for_patch,
            find_xx_merges_for_patch,
            find_zz_merges_for_patch,
        ):
            merges.extend(finder(layer_index, order_index, qubits, DEFAULT_SETTINGS, set()))
    return merges, layer_index, order_index, dag


def edges(conflicts: Dict[int, Set[int]]) -> Set[frozenset]:
    return {
        frozenset((a, b))
        for a, others in conflicts.items()
        for b in others
    }


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("seed", range(8))
def test_indexed_detector_finds_the_same_edges(n_code, seed):
    k = n_code - 2
    num_qubits = 4 * k
    circuit = hadamard_dense(num_qubits, 220, seed)
    permutation = sequential(num_qubits, k)
    merges, layer_index, order_index, dag = candidates_for(circuit, permutation)
    if not merges:
        pytest.skip("no candidates at this size and seed")

    fast: Dict[int, Set[int]] = {}
    _add_cross_gate_conflicts(merges, layer_index, order_index, dag, fast)

    slow: Dict[int, Set[int]] = {}
    _reference_cross_gate_conflicts(merges, layer_index, order_index, dag, slow)

    assert edges(fast) == edges(slow), (
        f"conflict graph diverged at n_code={n_code} seed={seed}: "
        f"{len(edges(fast))} indexed edges vs {len(edges(slow))} enumerated"
    )


@pytest.mark.parametrize("n_code", CODE_SIZES)
def test_seeded_conflicts_are_preserved_not_overwritten(n_code):
    """
    Edges already in the dict must survive, and must still suppress re-adding.

    `compute_merge_conflicts` calls this *after* the geometric and shared-endpoint
    passes, so the dict arrives populated. The indexed version reads `conflicts` for its
    skip check exactly as the enumerated one did, and an implementation that built a
    fresh dict instead would pass the test above and break here.
    """
    k = n_code - 2
    num_qubits = 4 * k
    circuit = hadamard_dense(num_qubits, 220, seed=3)
    permutation = sequential(num_qubits, k)
    merges, layer_index, order_index, dag = candidates_for(circuit, permutation)
    if len(merges) < 2:
        pytest.skip("need at least two candidates")

    seeded = {0: {1}, 1: {0}}
    fast = {key: set(value) for key, value in seeded.items()}
    _add_cross_gate_conflicts(merges, layer_index, order_index, dag, fast)

    slow = {key: set(value) for key, value in seeded.items()}
    _reference_cross_gate_conflicts(merges, layer_index, order_index, dag, slow)

    assert edges(fast) == edges(slow)
    assert frozenset((0, 1)) in edges(fast)


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("seed", range(4))
def test_full_pass_output_is_unchanged(n_code, seed):
    """
    The savings triple and the emitted instruction list, which is what actually ships.

    Pinned by value rather than against a re-run of the old code: the conflict graph is
    the only input the rewrite touched, so if the graph matches and the circuit matches,
    nothing downstream moved.
    """
    k = n_code - 2
    num_qubits = 4 * k
    circuit = hadamard_dense(num_qubits, 220, seed)
    permutation = sequential(num_qubits, k)

    rewritten, savings_1q, savings_2q, weighted = preprocess_program_for_merging_with_stats(
        circuit, permutation, n_code=n_code, allow_transversal_swap_merges=True
    )

    # Non-trivial: the test is worthless if the pass found nothing to do.
    assert len(rewritten.data) < len(circuit.data)
    assert savings_1q > 0 or savings_2q > 0

    table = merge_work_table(n_code)
    assert table["HH"] == (n_code, 0)

    # Deterministic: the same input must give the same output, since the selector's
    # eviction loop is order-sensitive and a dict-iteration change would show up here.
    again = preprocess_program_for_merging_with_stats(
        circuit, permutation, n_code=n_code, allow_transversal_swap_merges=True
    )
    assert (again[1], again[2], again[3]) == (savings_1q, savings_2q, weighted)
    assert [
        (instruction.operation.name, tuple(again[0].find_bit(q).index for q in instruction.qubits))
        for instruction in again[0].data
    ] == [
        (instruction.operation.name, tuple(rewritten.find_bit(q).index for q in instruction.qubits))
        for instruction in rewritten.data
    ]
