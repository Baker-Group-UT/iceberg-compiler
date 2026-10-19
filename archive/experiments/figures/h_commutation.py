"""
Simple demo for the H-commutation pass.

Run with:
    python -m experiments.figures.h_commutation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from translator.passes import commute_hadamards


def build_demo_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4, 4, name="h_commutation_demo")
    qc.h(1)
    qc.x(1)
    qc.z(1)
    qc.x(1)
    qc.h(1)

    qc.z(2)
    qc.x(2)
    qc.h(2)
    qc.z(2)
    qc.x(2)

    qc.cx(1,2)

    qc.x(1)
    qc.z(1)
    qc.cx(0,1)
    qc.h(1)

    qc.x(2)
    qc.h(2)
    qc.z(2)
    qc.cx(3,2)

    # qc.h(0)
    # qc.x(0)
    # qc.cx(0, 1)
    # qc.h(1)
    # qc.h(0)
    # qc.z(1)
    # # qc.h(1)
    # qc.cx(0, 1)
    return qc


def main() -> None:
    circuit = build_demo_circuit()
    print("Original circuit:")
    print(circuit.draw())
    print()

    commuted = commute_hadamards(circuit)
    print("After commute_hadamards:")
    print(commuted.draw())

    output_dir = Path("experiments/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1 = circuit.draw(output="mpl")
    fig1.savefig(output_dir / "h_commutation_1.svg", bbox_inches="tight")
    plt.close(fig1)

    fig2 = commuted.draw(output="mpl")
    fig2.savefig(output_dir / "h_commutation_2.svg", bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
