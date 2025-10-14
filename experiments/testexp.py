"""
run_scan.py - high level test file
"""

import cirq

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan, generate_permutation_set
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan, generate_random_circuit, translate_permutation
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator, OversampledSimulator
from circuits.example_circuits import ghz_3, toffoli_like_snippet
import math
from experiments.shared.random_circuit import random_circuit
from experiments_superstaq.converters.qiskit import cirq_to_qiskit
from dag.circuit_dag import draw_qubit_graph, build_qubit_graph
from cirq.contrib.qasm_import import circuit_from_qasm

qA, qB, qC, = cirq.LineQubit.range(3)
program_circuit = cirq.Circuit(
    cirq.H(qA),
    cirq.CNOT(qA, qB),
    cirq.CNOT(qA, qC),
    cirq.H(qA),
    # cirq.H(qA),
    # cirq.H(qB),
    # cirq.Z(qB),
    # cirq.X(qB),
    # cirq.CNOT(qA, qB),
    # cirq.CNOT(qA, qC),
    # cirq.H(qA),
    # cirq.rz(-math.pi/4)(qA),
    cirq.X(qA),
    # cirq.X(qB),
    cirq.X(qC),
    cirq.measure(qA),
    cirq.measure(qB),
    cirq.measure(qC),
)
program_circuit = random_circuit(n=5, depth=5, seed=4)

# program_circuit = ghz_3()["logical"]

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=OversampledSimulator(repetitions=1, oversample=1)\
        # .with_noiseless()
        .with_seed(6),
    verbose=True,
    save=False,
    seed=4,
)\
# .with_ft(True)

qiskit_translated_circuit = cirq_to_qiskit(program_circuit, qubits=program_circuit.all_qubits())
print(qiskit_translated_circuit.draw(output="text"))
qubit_graph = build_qubit_graph(qiskit_translated_circuit)
draw_qubit_graph(qubit_graph, orient="tower")

# Run the full scan pipeline
record: RunRecord = run_specific_scan(program_circuit, 
    permutations=[  #[(0, 0), (0, 1), (1, 0)], 
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)],
                    # [(0, 0), (1, 0), (2, 0)], 
                    # [(0, 1), (0, 0), (1, 1)],
                    ], config=config)
# record: RunRecord = run_extremal_scan(program_circuit, config=config)
# record: RunRecord = run_full_scan(program_circuit, config=config)
# generate_permutation_set(program_circuit, config)
translated_circuit = translate_permutation(program_circuit, [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)], config=config).circuit
qiskit_translated_circuit = cirq_to_qiskit(translated_circuit, qubits=translated_circuit.all_qubits())
print(qiskit_translated_circuit.draw(output="text"))
qubit_graph = build_qubit_graph(qiskit_translated_circuit)
draw_qubit_graph(qubit_graph, orient="tower")