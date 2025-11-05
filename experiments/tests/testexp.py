"""
run_scan.py - high level test file
"""

from translator.core import TranslationResult


from pipeline.tasks.generate_permutations import generate_permutation_set
from repository.run_builder import RunRecord


import qiskit
from qiskit.circuit.random import random_clifford_circuit

from pipeline.config import PipelineConfig
from pipeline import generate_circuit, run_full_scan, run_extremal_scan, run_specific_scan
from simulator.config_presets import ScaledSimulator, StandardSimulator
from packer.packer import pack


program_circuit: qiskit.QuantumCircuit = random_clifford_circuit(4, 10)

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
pcirc.h(0)
pcirc.cx(0, 1)
pcirc.cx(1, 2)
pcirc.cx(2, 3)
pcirc.measure_all()

config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=1, repetitions=5_000).with_seed(6),
    verbose=True,
    save=False,
    seed=4,
)

# print(pack(pcirc))

# res: RunRecord = run_full_scan(pcirc, config)
# df = res.dataframe.df

perms = [
    [(0,0), (0,1), (1,0), (1,1)],
    [(0,0), (1,0), (2,0), (3,0)],
]

from qiskit.qasm3 import dumps
print(pcirc)
translated: TranslationResult = generate_circuit(pcirc, perms[1], config)
print(dumps(translated.circuit))