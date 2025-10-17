"""
gate_count_stats.py - Extracts gate usage statistics from Qiskit circuits.
"""

from collections import Counter
from typing import Dict
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
