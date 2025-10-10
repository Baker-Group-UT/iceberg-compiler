"""
run_scan.py - high level test file
"""

from math import pi

import cirq

from pipeline import run_extremal_scan
from pipeline.config import PipelineConfig
from simulator.config_presets import TestSimulator

qA, = cirq.LineQubit.range(1)
program_circuit = cirq.Circuit(
    cirq.H(qA),
    cirq.rz(pi / 8)(qA),
    cirq.H(qA)
)

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=TestSimulator(noiseless=True),
    verbose=True,
    fault_tolerant=True
)

# Run the full scan pipeline
record = run_extremal_scan(program_circuit, config)
