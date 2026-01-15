"""
gate_count_stats.py - Extracts gate usage statistics from Qiskit circuits.
"""

from collections import Counter
from typing import Dict, Tuple, List
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def extract_gate_count(circuit: QuantumCircuit) -> Dict[str, int]:
    """
    Count each gate type in a Qiskit circuit.

    Args:
        circuit: A Qiskit QuantumCircuit (usually the translated physical one).

    Returns:
        Dictionary mapping gate names to counts.
    """
    counter = Counter()

    # Each circuit instruction is a tuple: (Instruction, qargs, cargs)
    for inst, qargs, cargs in circuit.data:
        if inst is None:
            name = "UNKNOWN"
        else:
            # Qiskit Instruction objects have a .name attribute
            name = inst.name.upper()
        counter[name] += 1

    return dict(counter)


def total_gates(gate_count: Dict[str, int]) -> int:
    """
    Compute the total number of gates from the gate count dictionary.

    Args:
        gate_count: Dictionary returned by extract_gate_count().

    Returns:
        Sum of all gate occurrences.
    """
    return sum(gate_count.values())


def one_two_qubit_gate_totals(
    circuit: QuantumCircuit,
    *,
    include_measure: bool = False,
    include_barrier: bool = False,
) -> Tuple[int, int]:
    """
    Count 1q and 2q operations in a circuit.

    By default this excludes MEASURE and BARRIER instructions since they are not
    "logical gates" and tend to be constant across permutations.
    """
    one_q = 0
    two_q = 0

    for inst in circuit.data:
        op = getattr(inst, "operation", None)
        if op is None:
            op = inst[0]

        name = getattr(op, "name", "").lower()
        if not include_measure and name == "measure":
            continue
        if not include_barrier and name == "barrier":
            continue

        n_q = getattr(op, "num_qubits", None)
        if n_q is None:
            qargs = getattr(inst, "qubits", None)
            if qargs is None:
                qargs = inst[1]
            n_q = len(qargs)

        if n_q == 1:
            one_q += 1
        elif n_q == 2:
            two_q += 1

    return one_q, two_q


def gate_counts_by_code(
    circuit: QuantumCircuit,
    *,
    qubits_per_code: int = 6,
    include_measure: bool = False,
    include_barrier: bool = False,
) -> Dict[int, Tuple[float, float]]:
    """
    Count 1q and 2q operations per code, grouping qubits by contiguous blocks.

    Two-qubit gates are split evenly across the participating codes.
    """
    if qubits_per_code <= 0:
        raise ValueError("qubits_per_code must be positive.")

    one_q_counts: Dict[int, float] = {}
    two_q_counts: Dict[int, float] = {}

    def code_for_qubit(qindex: int) -> int:
        return qindex // qubits_per_code

    for inst, qargs, _ in circuit.data:
        op = inst
        name = getattr(op, "name", "").lower()
        if not include_measure and name == "measure":
            continue
        if not include_barrier and name == "barrier":
            continue

        qubit_indices = [circuit.find_bit(q).index for q in qargs]
        if len(qubit_indices) == 1:
            code = code_for_qubit(qubit_indices[0])
            one_q_counts[code] = one_q_counts.get(code, 0.0) + 1.0
        elif len(qubit_indices) == 2:
            code_i = code_for_qubit(qubit_indices[0])
            code_j = code_for_qubit(qubit_indices[1])
            if code_i == code_j:
                two_q_counts[code_i] = two_q_counts.get(code_i, 0.0) + 1.0
            else:
                two_q_counts[code_i] = two_q_counts.get(code_i, 0.0) + 0.5
                two_q_counts[code_j] = two_q_counts.get(code_j, 0.0) + 0.5

    codes = set(one_q_counts) | set(two_q_counts)
    return {code: (one_q_counts.get(code, 0.0), two_q_counts.get(code, 0.0)) for code in codes}


def depth_by_code(
    circuit: QuantumCircuit,
    *,
    qubits_per_code: int = 6,
    include_measure: bool = False,
    include_barrier: bool = False,
) -> Dict[int, int]:
    """
    Count per-code depth as the number of circuit layers that touch each code.
    """
    if qubits_per_code <= 0:
        raise ValueError("qubits_per_code must be positive.")

    def code_for_qubit(qindex: int) -> int:
        return qindex // qubits_per_code

    code_depths: Dict[int, int] = {}
    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        ops = layer["graph"].op_nodes()
        codes_in_layer = set()
        for node in ops:
            name = getattr(node.op, "name", "").lower()
            if not include_measure and name == "measure":
                continue
            if not include_barrier and name == "barrier":
                continue
            for q in node.qargs:
                qindex = circuit.find_bit(q).index
                codes_in_layer.add(code_for_qubit(qindex))
        for code in codes_in_layer:
            code_depths[code] = code_depths.get(code, 0) + 1

    return code_depths
