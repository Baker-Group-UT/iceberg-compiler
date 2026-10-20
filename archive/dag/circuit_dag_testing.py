"""
Interaction graph and dependency DAG for a random circuit. A demo, not a test.

Guarded behind `__main__` for the same reason as `circuit_dag_shor_test.py`: it drew two
figures at module level, so importing it opened windows.

**This file did not run as written.** `gen_random_clifford_circ` was imported from a
`random_circuit_gen` module that does not exist anywhere in the repository, so the import
raised as soon as it was reached -- which the module-level layout hid, since nothing
imported the file deliberately. `experiments.shared.random_circuits.random_gate_circuit`
is the surviving equivalent and is used here instead.

    PYTHONPATH=. python -m dag.circuit_dag_testing
"""

from __future__ import annotations

from qiskit import transpile

from archive.dag.circuit_dag_prev import CircuitDAG, draw_detailed_interaction_graph
from experiments.shared.random_circuits import random_gate_circuit


def main(num_qubits: int = 10, num_gates: int = 50, seed: int = 0) -> None:
    circuit = transpile(random_gate_circuit(num_qubits, num_gates, seed=seed))

    # Inter-qubit interactions, detailed multi-edge view. Thin edges and tucked labels so
    # a dense circuit still reads.
    draw_detailed_interaction_graph(
        circuit, linewidths=0.5, base_rad=0.10, label_offset=0.05
    )

    # Dependency DAG, one node per instruction.
    CircuitDAG(circuit).draw(figsize=(14, 6), layout="layered")


if __name__ == "__main__":
    main()
