"""Circuit rewrite for merge replacements."""
from __future__ import annotations
import copy
from typing import Dict, List, Sequence, Tuple
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

    # Each entry in a qubit's blocking stack is a (record, min_layer, min_order)
    # triple. A node on that qubit is only blocked by `record` if its own
    # (layer, order) >= (min_layer, min_order). This prevents gates that appear
    # before the gate that caused the blocking from being incorrectly delayed.
    #
    # Entries are always appended in topological order, so each qubit's list is
    # monotonically ascending by (min_layer, min_order). Iterating in reverse
    # finds the most-recently-armed applicable entry first.
    blocked_qubits: Dict[object, List[Tuple[MacroReplacement, int, int]]] = {}
    delayed_nodes: Dict[MacroReplacement, List[DAGOpNode]] = {}

    def _key(n: DAGOpNode) -> Tuple[int, int]:
        return (layer_index.get(n, 0), order_index.get(n, 0))

    ordered_nodes = sorted(dag.topological_op_nodes(), key=_key)

    for node in ordered_nodes:
        node_layer = layer_index.get(node, 0)
        node_order = order_index.get(node, 0)

        # ── first: drop it and arm blocking on first's qubit(s) ──────────────
        if node in skip_nodes:
            record = waiting_lookup.get(node)
            if record is not None:
                for qubit in record.blocked_qubits:
                    blocked_qubits.setdefault(qubit, []).append(
                        (record, node_layer, node_order)
                    )
                delayed_nodes.setdefault(record, [])
            continue

        # ── second: emit the macro and flush delayed gates ───────────────────
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
            # Remove this record from every qubit stack — including qubits that
            # were added dynamically by the Part-1 extension below.
            for qubit in list(blocked_qubits.keys()):
                blocked_qubits[qubit] = [
                    e for e in blocked_qubits[qubit] if e[0] is not record
                ]
                if not blocked_qubits[qubit]:
                    del blocked_qubits[qubit]
            continue

        # ── all other nodes: check for positional blocking ───────────────────
        blocking_records = []
        for q in node.qargs:
            entries = blocked_qubits.get(q)
            if not entries:
                continue
            # Walk the stack newest-first; pick the innermost entry that covers
            # this node's position. Entries added before this node's position
            # (min > node) are skipped — those blocks do not apply here.
            for rec, min_layer, min_order in reversed(entries):
                if (node_layer, node_order) >= (min_layer, min_order):
                    blocking_records.append(rec)
                    break

        if not blocking_records:
            new_qc.append(node.op.copy(), node.qargs, node.cargs)
            continue

        # Delay under the record whose second fires latest. This guarantees the
        # node stays held until every relevant macro has been emitted.
        chosen = max(blocking_records, key=lambda r: _key(r.second))
        delayed_nodes.setdefault(chosen, []).append(node)

        # Part-1 extension: block each qubit of the delayed node, but only
        # starting from this node's position. Gates on those qubits that came
        # before this node are unaffected and will be emitted immediately.
        for q in node.qargs:
            entries = blocked_qubits.setdefault(q, [])
            if not any(e[0] is chosen for e in entries):
                entries.append((chosen, node_layer, node_order))

    return new_qc