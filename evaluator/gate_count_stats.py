"""
gate_count_stats.py - Extracts gate usage statistics from Qiskit circuits.
"""

from collections import Counter
from typing import Dict, Tuple
from qiskit import QuantumCircuit


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
