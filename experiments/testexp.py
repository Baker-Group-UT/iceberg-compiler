"""
run_scan.py - high level test file
"""

import cirq

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator
from circuits.example_circuits import ghz_3, toffoli_like_snippet


qA, qB, qC = cirq.LineQubit.range(3)
program_circuit = cirq.Circuit(
    cirq.H(qA),
    cirq.CNOT(qA, qB),
    cirq.CNOT(qA, qC),
    cirq.H(qA),
)

# program_circuit = ghz_3()["logical"]

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=500),
    verbose=True,
    save=False,
)
# .with_ft(True)

# Run the full scan pipeline
# record: RunRecord = run_specific_scan(program_circuit, permutations=[[(0, 0), (1, 1)]], config=config)
# record: RunRecord = run_extremal_scan(program_circuit, config=config)
record: RunRecord = run_full_scan(program_circuit, config=config)
