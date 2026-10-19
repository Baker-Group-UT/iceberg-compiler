"""
run_scan.py - High-level test file
"""

from qiskit import QuantumCircuit
import qiskit
from qiskit.circuit.random import random_clifford_circuit

from pipeline.config import PipelineConfig
from pipeline import run_full_scan
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator
from experiments.shared.example_circuits import n_ghz_state

# # Build the logical (program) circuit
# program_circuit = QuantumCircuit(3, name="run_scan")
# program_circuit.h(0)
# program_circuit.cx(0, 1)
# program_circuit.cx(0, 2)
# program_circuit.h(0)
# program_circuit.measure_all()

# program_circuit: QuantumCircuit = random_clifford_circuit(4, 10)

# program_circuit.measure_all()

# print(program_circuit)

# target_basis = ['x', 'z', 'h', 'rz', 'cx', 'cz', 'measure']

# tcirc = qiskit.transpile(
#     program_circuit,
#     basis_gates=target_basis,
#     optimization_level=0,
# )
# print(tcirc)

circ = n_ghz_state(4)["program"]
# Create a config for the pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=50_000),
    verbose=True,
    fault_tolerant=False,
)

# Run the full scan pipeline
record: RunRecord = run_full_scan(circ, config)
print(record.dataframe)