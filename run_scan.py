"""
run_scan.py - high level test file
"""

import cirq

from pipeline.config import PipelineConfig
from pipeline import run_extremal_scan
from simulator.config_presets import StandardSimulator


qA, qB, qC = cirq.LineQubit.range(3)
program_circuit = cirq.Circuit(
    cirq.H(qA),
    cirq.CNOT(qA, qB),
    cirq.CNOT(qA, qC),
    cirq.H(qA),
)

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=1),
    verbose=True,
    fault_tolerant=True
)

# Run the full scan pipeline
record = run_extremal_scan(program_circuit, config)
