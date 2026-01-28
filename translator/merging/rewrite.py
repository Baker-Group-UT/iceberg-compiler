"""Circuit rewrite for merge replacements."""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from .models import MacroReplacement


def build_rewritten_circuit(
    original: QuantumCircuit,
    dag: DAGCircuit,
    records: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
) -> QuantumCircuit:
    new_qc = QuantumCircuit(*original.qregs, *original.cregs)
    new_qc.global_phase = getattr(original, "global_phase", 0)
    if hasattr(original, "calibrations"):
        new_qc.calibrations = copy.deepcopy(original.calibrations)
    if hasattr(original, "metadata"):
        new_qc.metadata = copy.deepcopy(original.metadata)

    skip_nodes = {record.first for record in records}
    replacements = {record.second: record for record in records}
    waiting_lookup = {record.first: record for record in records}

    blocked_qubits: Dict[object, List[MacroReplacement]] = {}
    delayed_nodes: Dict[MacroReplacement, List[DAGOpNode]] = {}
    ordered_nodes = sorted(
        dag.topological_op_nodes(),
        key=lambda n: (layer_index.get(n, 0), order_index.get(n, 0)),
    )
    for node in ordered_nodes:
        if node in skip_nodes:
            record = waiting_lookup.get(node)
            if record is not None:
                for qubit in record.blocked_qubits:
                    blocked_qubits.setdefault(qubit, []).append(record)
                delayed_nodes.setdefault(record, [])
            continue
        if node in replacements:
            record = replacements[node]
            for planned in record.pre_ops:
                new_qc.append(planned.gate.copy(), list(planned.qargs), [])
            new_qc.append(record.macro_gate, list(record.macro_qargs), [])
            for planned in record.post_ops:
                new_qc.append(planned.gate.copy(), list(planned.qargs), [])
            for delayed in delayed_nodes.get(record, []):
                new_qc.append(delayed.op.copy(), delayed.qargs, delayed.cargs)
            delayed_nodes.pop(record, None)
            for qubit in record.blocked_qubits:
                stack = blocked_qubits.get(qubit)
                if not stack:
                    continue
                try:
                    stack.remove(record)
                except ValueError:
                    continue
                if not stack:
                    blocked_qubits.pop(qubit, None)
            continue
        blocking_records = []
        for q in node.qargs:
            stack = blocked_qubits.get(q)
            if stack:
                blocking_records.append(stack[-1])
        if blocking_records:
            record = max(
                blocking_records,
                key=lambda r: (layer_index.get(r.second, 0), order_index.get(r.second, 0)),
            )
            delayed_nodes.setdefault(record, []).append(node)
            continue
        new_qc.append(node.op.copy(), node.qargs, node.cargs)

    return new_qc
