"""
permutation_example.py
"""

import math
from qiskit import QuantumCircuit

from pipeline.config import PipelineConfig
from pipeline import generate_circuits
from simulator.config_presets import StandardSimulator

# Build the logical (program) circuit in Qiskit
program_circuit = QuantumCircuit(2, name="perm_example")
program_circuit.h(0)
program_circuit.h(1)
program_circuit.cx(0, 1)
program_circuit.rz(math.pi / 4, 1)
program_circuit.cx(0, 1)
program_circuit.h(0)
program_circuit.h(1)

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=1000),
    verbose=True,
    fault_tolerant=False,
)

# Run the full scan pipeline
translated = generate_circuits(program_circuit, config)

print(program_circuit.draw("text"))
print()