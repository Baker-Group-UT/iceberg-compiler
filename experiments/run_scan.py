"""
run_scan.py - High-level test file
"""

from qiskit import QuantumCircuit

from pipeline.config import PipelineConfig
from pipeline import run_full_scan
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator

# Build the logical (program) circuit
program_circuit = QuantumCircuit(3, name="run_scan")
program_circuit.h(0)
program_circuit.cx(0, 1)
program_circuit.cx(0, 2)
program_circuit.h(0)

# Create a config for the pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=1000),
    verbose=True,
    fault_tolerant=False,
)

# Run the full scan pipeline
record: RunRecord = run_full_scan(program_circuit, config)
print(record.dataframe)