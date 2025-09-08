"""
permutation_example.py
"""

import math
import cirq

from pipeline.config import PipelineConfig
from pipeline import generate_circuits
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator


qA, qB = cirq.LineQubit.range(2)
program_circuit = cirq.Circuit(
    cirq.H(qA),
    cirq.H(qB),
    cirq.CNOT(qA, qB),
    cirq.rz(math.pi / 4)(qB),
    cirq.CNOT(qA, qB),
    cirq.H(qA),
    cirq.H(qB),
)


# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=1000),
    verbose=True,
    fault_tolerant=False
)

# Run the full scan pipeline
translated = generate_circuits(program_circuit, config)

print(program_circuit)
print()