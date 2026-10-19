"""
pipeline.analysis.window_print.py - Prints out the circuit (Qiskit version).
"""

from qiskit import QuantumCircuit

def print_in_windows(circ: QuantumCircuit, window: int = 8):
    """
    Print a QuantumCircuit in chunks of 'window' operations for easier viewing.
    """
    ops = list(circ.data)
    for t in range(0, len(ops), window):
        sub_circ = QuantumCircuit(circ.num_qubits, circ.num_clbits, name=f"sub_{t//window}")
        for inst, qargs, cargs in ops[t : t + window]:
            sub_circ.append(inst, qargs, cargs)
        print(f"\n# Instructions {t}–{min(t+window-1, len(ops)-1)}")
        print(sub_circ.draw("text"))


def print_distribution(dist: dict[str, float]) -> None:
    """
    Pretty-print a distribution dictionary.

    Args:
        dist: Dictionary mapping bitstrings to probabilities.

    Prints:
        Keys and values, rounded to 4 decimals, 4 per row.
    """
    if not dist:
        print("(empty distribution)")
        return

    items = [f"{k}: {v:.4f}" for k, v in dist.items()]
    for i in range(0, len(items), 4):
        print("   ".join(items[i:i+4]))
