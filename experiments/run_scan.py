"""
run_scan.py - high level test file
"""

import cirq

from pipeline.config import PipelineConfig
from pipeline import run_full_scan
from repository.run_builder import RunRecord
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
    sim_config=StandardSimulator(repetitions=1000),
    verbose=True,
    fault_tolerant=False
)

# Run the full scan pipeline
record: RunRecord = run_full_scan(program_circuit, config)
print(record.dataframe)