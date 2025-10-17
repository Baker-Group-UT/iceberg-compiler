"""
run_scan.py - high level test file
"""

import qiskit

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan
from simulator.config_presets import ScaledSimulator

# Experiment 1
program_circuit = qiskit.QuantumCircuit(2)
program_circuit.x(0)
program_circuit.cx(0,1)
program_circuit.measure_all()
permGroup = [(0,0), (0,1)]
permDiff = [(0,0), (1,0)]

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=1, repetitions=100_000)\
        # .with_noiseless()
        .with_seed(6),
    verbose=True,
    save=False,
    seed=4,
)\
# .with_ft(False)
record = run_specific_scan(program_circuit=program_circuit, permutations=[permGroup, permDiff], config=config)



# Experiment two

print("\n\nAAC")
program_circuit = qiskit.QuantumCircuit(3)
program_circuit.x(0)
program_circuit.cx(0,2)
program_circuit.measure_all()
perm = [(0,0),(0,1),(1,0)]
record = run_specific_scan(program_circuit=program_circuit, permutations=[perm], config=config)

print("\n\nBBC")
program_circuit = qiskit.QuantumCircuit(3)
program_circuit.x(1)
program_circuit.cx(1,2)
program_circuit.measure_all()
record = run_specific_scan(program_circuit=program_circuit, permutations=[perm], config=config)

print("\n\nCCA")
program_circuit = qiskit.QuantumCircuit(3)
program_circuit.x(2)
program_circuit.cx(2,0)
program_circuit.measure_all()
record = run_specific_scan(program_circuit=program_circuit, permutations=[perm], config=config)

print("\n\nCCB")
program_circuit = qiskit.QuantumCircuit(3)
program_circuit.x(2)
program_circuit.cx(2,1)
program_circuit.measure_all()
record = run_specific_scan(program_circuit=program_circuit, permutations=[perm], config=config)
