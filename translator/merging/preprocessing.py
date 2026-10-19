"""Gate-merging preprocessing.

Given a permutation and a program circuit, scan patch pairs for mergeable HH
pairs and transversal CX/CZ pairs, then rewrite the circuit to insert macro
operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .conflicts import DEFAULT_MERGE_WORK_TABLE, DEFAULT_MERGE_WORK_WEIGHTS, filter_single_qubit_conflicting_with_transversal
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
) -> QuantumCircuit:
    updated_circuit, _, _, _ = preprocess_program_for_merging_with_stats(
        program_circuit,
        permutation,
        settings=settings,
        allow_transversal_swap_merges=allow_transversal_swap_merges,
        num_merge_passes=num_merge_passes,
    )
    return updated_circuit


def preprocess_program_for_merging_with_stats(
    program_circuit: QuantumCircuit,
    permutation: Sequence[Tuple[int, int]],
    *,
    settings: Optional[MergeSettings] = None,
    allow_transversal_swap_merges: bool = False,
    num_merge_passes: int = 2,
) -> Tuple[QuantumCircuit, float, float, float]:
    if program_circuit.num_qubits == 0:
        return program_circuit, 0.0, 0.0, 0.0

    if settings is None:
        settings = DEFAULT_SETTINGS

    updated_circuit = program_circuit
    total_savings_1q = 0.0
    total_savings_2q = 0.0

    for _ in range(max(1, num_merge_passes)):
        dag = circuit_to_dag(updated_circuit)
        layer_index, order_index = layer_by_node(dag)
        patch_pairs = _build_patch_pairs(permutation, updated_circuit.qubits)
        if not patch_pairs:
            break

        qubit_indices = {
            qubit: updated_circuit.find_bit(qubit).index for qubit in updated_circuit.qubits
        }

        consumed_nodes: set = set()
        potential_records: List[MacroReplacement] = []
        for _, qubits in patch_pairs.items():
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
            )
        )
        consumed_nodes.update({rec.first for rec in records})
        consumed_nodes.update({rec.second for rec in records})

        transversal_records = find_transversal_merges(
            layer_index,
            order_index,
            patch_pairs,
            qubit_indices,
            consumed_nodes,
            allow_transversal_swap_merges,
        )
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

        savings_1q, savings_2q = _merge_savings_for_records(records, DEFAULT_MERGE_WORK_TABLE)
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


def _build_patch_pairs(
    permutation: Sequence[Tuple[int, int]],
    qubits: Sequence,
) -> Dict[int, Tuple]:
    patch_slots: Dict[int, Dict[int, object]] = {}
    for logical_i, (patch, slot) in enumerate(permutation):
        if logical_i >= len(qubits):
            break
        patch_slots.setdefault(patch, {})[slot] = qubits[logical_i]

    patch_pairs: Dict[int, Tuple] = {}
    for patch, slots in patch_slots.items():
        if len(slots) < 2:
            continue
        ordered = [q for _, q in sorted(slots.items(), key=lambda item: item[0])]
        patch_pairs[patch] = (ordered[0], ordered[1])
    return patch_pairs


def _merge_savings_for_records(
    records: Sequence[MacroReplacement],
    work_table: Dict[str, Tuple[int, int]],
) -> Tuple[float, float]:
    savings_1q = 0.0
    savings_2q = 0.0
    macro_to_orig = {
        "HH": "H",
        "XX": "X",
        "ZZ": "Z",
        "CXT": "CX",
        "CZT": "CZ",
        "XCXT": "XCX",
    }
    for record in records:
        name = getattr(record.macro_gate, "name", "").upper()
        orig = macro_to_orig.get(name)
        if orig is None:
            continue
        base = work_table.get(orig)
        macro = work_table.get(name)
        if base is None or macro is None:
            continue
        savings_1q += 2 * base[0] - macro[0]
        savings_2q += 2 * base[1] - macro[1]
    return savings_1q, savings_2q
