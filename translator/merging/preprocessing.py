"""Gate-merging preprocessing.

Given a permutation and a program circuit, scan patch pairs for mergeable HH
pairs and transversal CX/CZ pairs, then rewrite the circuit to insert macro
operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .conflicts import (
    DEFAULT_CODE_SIZE,
    DEFAULT_MERGE_WORK_WEIGHTS,
    MACRO_TO_ORIGINAL,
    filter_single_qubit_conflicting_with_transversal,
    macro_cost,
    merge_work_table,
)
from .single_qubit import find_hh_merges_for_patch, find_xx_merges_for_patch, find_zz_merges_for_patch
from .layer import layer_by_node
from .models import MacroReplacement
from .selection import select_best_group
from .settings import DEFAULT_SETTINGS, MergeSettings
from .transversal import find_transversal_merges
from .rewrite import build_rewritten_circuit


def preprocess_program_for_merging(
    program_circuit: QuantumCircuit,
    permutation: Sequence[Tuple[int, int]],
    *,
    settings: Optional[MergeSettings] = None,
    allow_transversal_swap_merges: bool = False,
    num_merge_passes: int = 2,
    n_code: int = DEFAULT_CODE_SIZE,
) -> QuantumCircuit:
    updated_circuit, _, _, _ = preprocess_program_for_merging_with_stats(
        program_circuit,
        permutation,
        settings=settings,
        allow_transversal_swap_merges=allow_transversal_swap_merges,
        num_merge_passes=num_merge_passes,
        n_code=n_code,
    )
    return updated_circuit


def preprocess_program_for_merging_with_stats(
    program_circuit: QuantumCircuit,
    permutation: Sequence[Tuple[int, int]],
    *,
    settings: Optional[MergeSettings] = None,
    allow_transversal_swap_merges: bool = False,
    num_merge_passes: int = 2,
    n_code: int = DEFAULT_CODE_SIZE,
    two_qubit: bool = True,
) -> Tuple[QuantumCircuit, float, float, float]:
    if program_circuit.num_qubits == 0:
        return program_circuit, 0.0, 0.0, 0.0

    if settings is None:
        settings = DEFAULT_SETTINGS

    work_table = merge_work_table(n_code)
    updated_circuit = program_circuit
    total_savings_1q = 0.0
    total_savings_2q = 0.0

    for _ in range(max(1, num_merge_passes)):
        dag = circuit_to_dag(updated_circuit)
        layer_index, order_index = layer_by_node(dag)
        patch_lanes = _build_patch_lanes(permutation, updated_circuit.qubits)
        if not patch_lanes:
            break

        qubit_indices = {
            qubit: updated_circuit.find_bit(qubit).index for qubit in updated_circuit.qubits
        }

        consumed_nodes: set = set()
        potential_records: List[MacroReplacement] = []
        for _, qubits in patch_lanes.items():
            potential_records.extend(
                find_hh_merges_for_patch(layer_index, order_index, qubits, settings, consumed_nodes)
            )
            potential_records.extend(
                find_xx_merges_for_patch(layer_index, order_index, qubits, settings, consumed_nodes)
            )
            potential_records.extend(
                find_zz_merges_for_patch(layer_index, order_index, qubits, settings, consumed_nodes)
            )

        records: List[MacroReplacement] = []
        # Pass dag and order_index so select_best_group can detect cross-gate
        # conflicts — cases where merge B's blocking window would reorder a
        # 2-qubit gate relative to merge A's macro.
        records.extend(
            select_best_group(
                potential_records,
                layer_index,
                order_index,
                qubit_indices,
                dag=dag,
                work_table=work_table,
            )
        )
        consumed_nodes.update({rec.first for rec in records})
        consumed_nodes.update({rec.second for rec in records})

        # `two_qubit=False` keeps the single-qubit macros only (TranslatorConfig.merge_two_qubit).
        transversal_records = find_transversal_merges(
            layer_index,
            order_index,
            patch_lanes,
            qubit_indices,
            consumed_nodes,
            allow_transversal_swap_merges,
            n_code=n_code,
        ) if two_qubit else []
        # Drop any single-qubit records whose first node falls inside a
        # transversal record's window and would be reordered past its macro.
        # This can't be caught during select_best_group because transversal
        # detection runs after single-qubit selection.
        records = filter_single_qubit_conflicting_with_transversal(
            records, transversal_records, layer_index, order_index
        )
        consumed_nodes = {rec.first for rec in records} | {rec.second for rec in records}
        records.extend(transversal_records)

        if not records:
            break

        savings_1q, savings_2q = _merge_savings_for_records(records, work_table)
        total_savings_1q += savings_1q
        total_savings_2q += savings_2q

        updated_circuit = build_rewritten_circuit(
            updated_circuit,
            dag,
            records,
            layer_index,
            order_index,
        )

    weighted = (
        total_savings_1q * DEFAULT_MERGE_WORK_WEIGHTS[0]
        + total_savings_2q * DEFAULT_MERGE_WORK_WEIGHTS[1]
    )
    return updated_circuit, total_savings_1q, total_savings_2q, weighted


def _build_patch_lanes(
    permutation: Sequence[Tuple[int, int]],
    qubits: Sequence,
) -> Dict[int, Tuple]:
    """
    Map each patch to its occupied lanes, in slot order.

    Every occupied lane is returned, not just the first two: a whole-patch macro at
    k slots needs all of them. Patches holding one qubit are dropped, since there is
    nothing to fuse with -- `apply_logical_hadamard` already promotes a solo H to a
    whole-patch HH without help from this pass.

    Slots left empty by an uneven packing simply do not appear, so a patch with 3 of
    4 slots filled yields three lanes and "full coverage" means those three. That is
    the right reading: idle lanes carry no logical qubit, so a whole-patch operation
    is free to disturb them.
    """
    patch_slots: Dict[int, Dict[int, object]] = {}
    for logical_i, (patch, slot) in enumerate(permutation):
        if logical_i >= len(qubits):
            break
        patch_slots.setdefault(patch, {})[slot] = qubits[logical_i]

    patch_lanes: Dict[int, Tuple] = {}
    for patch, slots in patch_slots.items():
        if len(slots) < 2:
            continue
        patch_lanes[patch] = tuple(q for _, q in sorted(slots.items()))
    return patch_lanes


def _merge_savings_for_records(
    records: Sequence[MacroReplacement],
    work_table: Dict[str, Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Gates saved by each macro: what the originals would have cost, minus the macro.

    The count comes from `len(record.nodes)` rather than being fixed at two -- a
    whole-patch macro at k slots replaces k gates, and at k=10 assuming two would
    understate the saving fivefold.
    """
    savings_1q = 0.0
    savings_2q = 0.0
    for record in records:
        name = getattr(record.macro_gate, "name", "").upper()
        orig = MACRO_TO_ORIGINAL.get(name)
        if orig is None:
            continue
        fused = len(record.nodes)
        base = work_table.get(orig)
        # `macro_cost`, not the raw table: a partial XX/ZZ fusion of m lanes
        # emits m + (m mod 2) gates, and the table's whole-patch entry would
        # overstate the saving for every m < k group.
        macro = macro_cost(name, fused, work_table)
        if base is None or macro is None:
            continue
        # Undo gates (branch v2-tvd-explore) are singles of the same kind, emitted.
        net = fused - len(record.undo_ops)
        savings_1q += net * base[0] - macro[0]
        savings_2q += net * base[1] - macro[1]
    return savings_1q, savings_2q
