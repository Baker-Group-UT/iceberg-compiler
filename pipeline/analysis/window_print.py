"""
pipeline.analysis.window_print.py - Prints out the circuit.
"""

import cirq

def print_in_windows(circ: cirq.Circuit, window: int = 8):
    for t in range(0, len(circ), window):
        sub = cirq.Circuit(circ[t : t + window])
        print(f"\n# Moments {t}–{t+window-1}")
        print(sub)
