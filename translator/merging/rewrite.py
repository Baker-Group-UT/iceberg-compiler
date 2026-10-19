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

    # Every node a macro absorbs except its last is dropped here and reconstituted
    # by the macro. At k=2 that is exactly `record.first`; at larger k a macro
    # absorbs up to k nodes, so all of `record.dropped` must be skipped.
    replacements = {record.second: record for record in records}
    waiting_lookup: Dict[DAGOpNode, MacroReplacement] = {}
    for record in records:
        for node in record.dropped:
            waiting_lookup[node] = record
    skip_nodes = set(waiting_lookup)

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

    # Branch v2-tvd-explore (`explore_switches.MERGE_KEEP_PROGRAM_ORDER`): the program
    # position each emitted instruction stands for, so the result can be re-linearized.
    # Macros and their pre/undo/post ops take the position of the node they replace.
    keys: List[Tuple[int, int]] = []

    def _emit(op, qargs, cargs, key) -> None:
        new_qc.append(op, qargs, cargs)
        keys.append(key)

    for node in ordered_nodes:
        node_layer = layer_index.get(node, 0)
        node_order = order_index.get(node, 0)

        # ── a dropped node: remove it and arm blocking on its own qubit(s) ───
        if node in skip_nodes:
            record = waiting_lookup.get(node)
            if record is not None:
                # Arm on *this* node's qubits at *this* node's position. A macro
                # spanning k lanes drops one node per lane at different positions,
                # and each lane's block must start where that lane's gate was:
                # arming every lane at the earliest node's position would delay
                # gates that legitimately precede a later lane's gate, pushing them
                # after the macro and reordering them against it.
                for qubit in node.qargs:
                    blocked_qubits.setdefault(qubit, []).append(
                        (record, node_layer, node_order)
                    )
                delayed_nodes.setdefault(record, [])
            continue

        # ── second: emit the macro and flush delayed gates ───────────────────
        if node in replacements:
            record = replacements[node]
            at = order_index.get(node, 0)
            for planned in record.pre_ops:
                _emit(planned.gate.copy(), list(planned.qargs), [], (at, 0))
            _emit(record.macro_gate, list(record.macro_qargs), [], (at, 1))
            for planned in record.undo_ops:
                _emit(planned.gate.copy(), list(planned.qargs), [], (at, 2))
            for planned in record.post_ops:
                _emit(planned.gate.copy(), list(planned.qargs), [], (at, 3))
            for delayed in delayed_nodes.get(record, []):
                _emit(delayed.op.copy(), delayed.qargs, delayed.cargs,
                      (order_index.get(delayed, 0), 0))
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
            _emit(node.op.copy(), node.qargs, node.cargs, (node_order, 0))
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

    from translator import explore_switches

    if explore_switches.MERGE_KEEP_PROGRAM_ORDER:
        return _relinearize(new_qc, keys)
    return new_qc


def _relinearize(circuit: QuantumCircuit, keys: List[Tuple[int, int]]) -> QuantumCircuit:
    """
    The same circuit, in the topological order closest to program order.

    Lexicographic topological sort on each instruction's program position: whenever
    several instructions are ready, the earliest in the program goes first. Dependencies
    are respected exactly, so this is the same unitary; only adjacency changes.
    """
    from qiskit.converters import circuit_to_dag, dag_to_circuit

    dag = circuit_to_dag(circuit)
    ops = list(dag.topological_op_nodes())
    if len(ops) != len(keys):
        return circuit
    # circuit_to_dag adds op nodes in instruction order, so node ids sort the same way.
    by_id = {node._node_id: key for node, key in zip(sorted(ops, key=lambda n: n._node_id), keys)}
    order = dag.topological_op_nodes(
        key=lambda node: "%010d.%d" % by_id.get(node._node_id, (0, 0))
        if node._node_id in by_id else "0000000000.0"
    )
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    out.global_phase = circuit.global_phase
    if hasattr(circuit, "metadata"):
        out.metadata = copy.deepcopy(circuit.metadata)
    for node in order:
        out.append(node.op, node.qargs, node.cargs)
    return out