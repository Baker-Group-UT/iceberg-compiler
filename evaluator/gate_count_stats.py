"""
gate_count_stats.py - Extracts gate usage statistics from Cirq circuits.
"""

import cirq
from collections import Counter
from typing import Dict, Union

def extract_gate_count(circuit: cirq.Circuit) -> Dict[str, int]:
    """
    Count each gate type in a Cirq circuit.

    Args:
        circuit: A Cirq circuit (usually the physical translated one).

    Returns:
        Dictionary mapping gate names to counts.
    """
    counter = Counter()
    for moment in circuit:
        for op in moment.operations:
            gate = op.gate
            if gate is None:
                name = "UNKNOWN"
            else:
                name = type(gate).__name__
            counter[name] += 1
    return dict(counter)

def total_gates(gate_count: Dict[str, int]) -> int:
    """
    Compute the total number of gates from the gate count dictionary.

    Args:
        gate_count: Dictionary returned by extract_gate_count.

    Returns:
        Sum of all gate occurrences.
    """
    return sum(gate_count.values())