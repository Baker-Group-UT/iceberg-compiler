from qiskit import transpile
from circuit_dag_prev import CircuitDAG, draw_detailed_interaction_graph
from random_circuit_gen import gen_random_clifford_circ     # your utility

# ------------------------------------------------------------------
# 1. Generate a random Clifford circuit (50 gates on 10 qubits)
# ------------------------------------------------------------------
qc_big, qc_big_i = gen_random_clifford_circ(10, 10)   # returns (circuit, seed/idx)
qc_big = transpile(qc_big)                            # optional: canonicalize

# ------------------------------------------------------------------
# 2. Visualise inter-qubit interactions (detailed multi-edge view)
# ------------------------------------------------------------------
draw_detailed_interaction_graph(
    qc_big,
    linewidths=0.5,      # thin edges so lots can fit
    base_rad=0.10,       # first edge curvature
    label_offset=0.05    # tuck labels closer to arcs
)

# ------------------------------------------------------------------
# 3. Build & show the dependency DAG (one node per instruction)
# ------------------------------------------------------------------
dag = CircuitDAG(qc_big)
dag.draw(figsize=(14, 6), layout="layered")           # adjust fig-size / prog="dot" if you like

# If you need the raw NetworkX graph for further analysis:
nx_graph = dag.get_graph()
# e.g., print("Depth:", dag.get_graph().size())