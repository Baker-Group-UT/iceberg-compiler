"""
run_scan.py - high level test file
"""

import qiskit
from qiskit.circuit.random import random_clifford_circuit

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan, run_specific_scan
from simulator.config_presets import ScaledSimulator, StandardSimulator
from packer.packer import pack


program_circuit: qiskit.QuantumCircuit = random_clifford_circuit(4, 6)

target_basis = ['x', 'z', 'h', 'rz', 'cx', 'cz', 'measure']

tcirc = qiskit.transpile(
    program_circuit,
    basis_gates=target_basis,
    optimization_level=2,
)
tcirc.measure_all()
# print(tcirc)

# res = pack(tcirc)
# print(res.permutation)

pcirc = qiskit.QuantumCircuit(4)
pcirc.x(0)
pcirc.cx(0, 2)
pcirc.h(3)
pcirc.h(2)
# pcirc.cx(3, 2)
pcirc.measure_all()

config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=1, repetitions=5_000).with_seed(6),
    verbose=True,
    save=False,
    seed=4,
)
print(pcirc)
# print(pack(pcirc))

res = run_full_scan(pcirc, config)
