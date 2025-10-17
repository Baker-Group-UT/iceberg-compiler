"""
run_scan.py - high level test file
"""

import qiskit

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan
from simulator.config_presets import StandardSimulator

program_circuit = qiskit.QuantumCircuit(3)
program_circuit.h(0)
program_circuit.cx(0,1)
program_circuit.cx(0,2)
program_circuit.h(0)
program_circuit.measure_all()


# program_circuit = ghz_3()["logical"]

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=10000)\
        .with_noiseless()
        .with_seed(6),
    verbose=True,
    save=False,
    seed=4,
)\
# .with_ft(False)

record = run_full_scan(program_circuit=program_circuit, config=config)
