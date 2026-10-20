"""
Toy 7-qubit circuit rendered four ways: interaction graph, dependency DAG, qubit lanes,
3-D tower. A demo, not a test.

Guarded behind `__main__`, and the reason is worth recording. The filename matches
pytest's default `*_test.py` glob, so bare `pytest` collected this file, *imported* it,
and executed the four `draw_*` calls that used to sit at module level. Under an
interactive backend that opens four windows and `plt.show()` blocks on each, so a test
run stopped and waited -- while the file contributes zero test items, so the collected
count gave no hint where the windows came from.

`pytest.ini` now confines collection to `tests/` and drops the `*_test.py` glob, so the
name no longer matters. The guard is here anyway: a module that plots on import is a
hazard whoever imports it.

Imports were also bare (`from circuit_dag_prev import ...`), which only resolved because
pytest inserts the containing directory on `sys.path`. They are package-qualified now, so
the file actually runs on its own:

    PYTHONPATH=. python -m dag.circuit_dag_shor_test
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit

from archive.dag.circuit_dag import build_qubit_graph, draw_qubit_graph
from archive.dag.circuit_dag_prev import CircuitDAG, draw_detailed_interaction_graph


def shor_toy_circuit() -> QuantumCircuit:
    """The 7-qubit example: a CX fan out of q1, a CSWAP ladder driven by q2, then a
    controlled phase coupling the two. q0 is untouched, which is why it sits alone in
    the interaction graph."""
    qc = QuantumCircuit(7, 2)
    qc.x(6)
    qc.h(1)
    qc.cx(1, 4)
    qc.cx(1, 6)
    qc.h(1)
    qc.h(2)
    qc.cswap(2, 3, 4)
    qc.cswap(2, 4, 5)
    qc.cswap(2, 5, 6)
    qc.crz(math.pi / 4, 1, 2)
    qc.h(2)
    qc.measure(1, 0)
    qc.measure(2, 1)
    return qc


def lane_demo_circuit() -> QuantumCircuit:
    """The same shape on 6 qubits, which is what the lane and tower views were drawn on."""
    qc = QuantumCircuit(6, 2)
    qc.x(5)
    qc.h(0)
    qc.cx(0, 3)
    qc.cx(0, 5)
    qc.h(0)
    qc.h(1)
    qc.cswap(1, 2, 3)
    qc.cswap(1, 3, 4)
    qc.cswap(1, 4, 5)
    qc.crz(math.pi / 4, 0, 1)
    qc.h(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def main() -> None:
    qc = shor_toy_circuit()

    # Inter-qubit interactions, detailed multi-edge view. Thin edges and tucked labels so
    # a dense circuit still reads.
    draw_detailed_interaction_graph(qc, linewidths=0.5, base_rad=0.10, label_offset=0.05)

    # Dependency DAG, one node per instruction.
    CircuitDAG(qc).draw(figsize=(14, 6), layout="layered")

    graph = build_qubit_graph(lane_demo_circuit(), decompose=True)
    draw_qubit_graph(graph, orient="horiz")
    draw_qubit_graph(graph, orient="tower", tower_radius=4.5, tower_zscale=0.7)


if __name__ == "__main__":
    main()
