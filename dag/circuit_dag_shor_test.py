# %matplotlib qt              # Native Qt window

# shor15_demo.py – toy 7-qubit circuit + timeline graph demo
import math
from qiskit import QuantumCircuit, transpile
from circuit_dag_prev import CircuitDAG, draw_detailed_interaction_graph
from circuit_dag import build_qubit_graph, draw_qubit_graph

# ------------------------------------------------------------------
# 1. Generate a random Clifford circuit (50 gates on 10 qubits)
# ------------------------------------------------------------------
# qc_big, qc_big_i = gen_random_clifford_circ(10, 10)   # returns (circuit, seed/idx)
# qc_big = transpile(qc_big)                            # optional: canonicalize

qc = QuantumCircuit(7, 2)
qc.x(6)
qc.h(1); qc.cx(1, 4); qc.cx(1, 6); qc.h(1)
qc.h(2); qc.cswap(2, 3, 4); qc.cswap(2, 4, 5); qc.cswap(2, 5, 6)
qc.crz(math.pi / 4, 1, 2); qc.h(2)
qc.measure(1, 0); qc.measure(2, 1)

# ------------------------------------------------------------------
# 2. Visualise inter-qubit interactions (detailed multi-edge view)
# ------------------------------------------------------------------
draw_detailed_interaction_graph(
    qc,
    linewidths=0.5,      # thin edges so lots can fit
    base_rad=0.10,       # first edge curvature
    label_offset=0.05    # tuck labels closer to arcs
)

# ------------------------------------------------------------------
# 3. Build & show the dependency DAG (one node per instruction)
# ------------------------------------------------------------------
dag = CircuitDAG(qc)
dag.draw(figsize=(14, 6), layout="layered")           # adjust fig-size / prog="dot" if you like

# If you need the raw NetworkX graph for further analysis:
nx_graph = dag.get_graph()
# e.g., print("Depth:", dag.get_graph().size())

# ------------------------------------------------------------------
# Build the minimal 7-qubit example (same logic as before)
# ------------------------------------------------------------------
qc = QuantumCircuit(6, 2)
qc.x(5)
qc.h(0); qc.cx(0, 3); qc.cx(0, 5); qc.h(0)
qc.h(1); qc.cswap(1, 2, 3); qc.cswap(1, 3, 4); qc.cswap(1, 4, 5)
qc.crz(math.pi / 4, 0, 1); qc.h(1)
qc.measure(0, 0); qc.measure(1, 1)

# qc_big, qc_big_i = gen_random_clifford_circ(10, 10)   # returns (circuit, seed/idx)
# qc_big = transpile(qc_big)   

# ------------------------------------------------------------------
# Build & draw the qubit-timeline graph
# ------------------------------------------------------------------
g = build_qubit_graph(qc, decompose=True)

draw_qubit_graph(g, orient="horiz")
# draw_qubit_graph(g, orient="laneless")

draw_qubit_graph(g, orient="tower", tower_radius=4.5, tower_zscale=0.7)