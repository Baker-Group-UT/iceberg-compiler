"""
run_scan.py - high level test file
"""

import qiskit

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan, generate_permutation_set
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan, generate_random_circuit, translate_permutation
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator
from circuits.example_circuits import ghz_3, toffoli_like_snippet
import math
from experiments.shared.random_circuit import random_circuit
from experiments_superstaq.converters.qiskit import cirq_to_qiskit
from dag.circuit_dag import draw_qubit_graph, build_qubit_graph
from cirq.contrib.qasm_import import circuit_from_qasm

program_circuit = qiskit.QuantumCircuit(3)
program_circuit.h(0)
program_circuit.cx(0,1)
program_circuit.cx(0,2)
program_circuit.h(0)


# program_circuit = ghz_3()["logical"]

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=1)\
        # .with_noiseless()
        .with_seed(6),
    verbose=True,
    save=False,
    seed=4,
)\
.with_ft(True)

tcirc = translate_permutation(program_circuit=program_circuit, permutation=[(0, 0), (1, 1), (1, 0)], config=config)

print(program_circuit)
print(tcirc)

# qiskit_translated_circuit = cirq_to_qiskit(program_circuit, qubits=program_circuit.all_qubits())
# # print(qiskit_translated_circuit.draw(output="text"))
# qubit_graph = build_qubit_graph(qiskit_translated_circuit)

# draw_qubit_graph(qubit_graph, orient="tower")

# # # Run the full scan pipeline
# # record: RunRecord = run_specific_scan(program_circuit, 
# #     permutations=[  #[(0, 0), (0, 1), (1, 0)], 
# #     [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)],
# #                     # [(0, 0), (1, 0), (2, 0)], 
# #                     # [(0, 1), (0, 0), (1, 1)],
# #                     ], config=config)
# # # record: RunRecord = run_extremal_scan(program_circuit, config=config)
# # # record: RunRecord = run_full_scan(program_circuit, config=config)
# # # generate_permutation_set(program_circuit, config)
# # translated_circuit = translate_permutation(program_circuit, [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)], config=config).circuit
# # qiskit_translated_circuit = cirq_to_qiskit(translated_circuit, qubits=translated_circuit.all_qubits())
# # print(qiskit_translated_circuit.draw(output="text"))
# # qubit_graph = build_qubit_graph(qiskit_translated_circuit)
# # draw_qubit_graph(qubit_graph, orient="tower")

# import networkx as nx
# import matplotlib.pyplot as plt

# def draw_multidigraph(G: nx.MultiDiGraph, with_labels=True):
#     """
#     Draws a NetworkX MultiDiGraph with arrows and optional labels.
#     """
#     pos = nx.spring_layout(G, seed=42)  # consistent layout
#     plt.figure(figsize=(6, 4))
    
#     # Draw nodes and edges
#     nx.draw(
#         G, pos,
#         with_labels=with_labels,
#         node_color="lightblue",
#         node_size=700,
#         edgecolors="black",
#         arrows=True,
#         connectionstyle='arc3,rad=0.1'  # curved edges for multiple edges
#     )

#     # Optionally draw edge labels
#     edge_labels = {
#         (u, v): f"{len(G[u][v])}" if len(G[u][v]) > 1 else ""
#         for u, v in G.edges()
#     }
#     if any(edge_labels.values()):
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()

# draw_multidigraph(qubit_graph)