"""
pipeline.analysis.window_print.py - Prints out the circuit.
"""

import cirq

def print_in_windows(circ: cirq.Circuit, window: int = 8):
    for t in range(0, len(circ), window):
        sub = cirq.Circuit(circ[t : t + window])
        print(f"\n# Moments {t}–{t+window-1}")
        print(sub)

def print_distribution(dist: dict[str, float]) -> None:
    """
    Pretty-print a distribution dictionary.

    Args:
        dist: Dictionary mapping bitstrings to probabilities.

    Prints:
        Keys and values, rounded to 4 decimals, 4 per row.
    """
    items = [f"{k}: {v:.4f}" for k, v in dist.items()]
    for i in range(0, len(items), 4):
        print("   ".join(items[i:i+4]))