"""
run_scan.py - high level test file
"""

from pipeline.analysis import print_df
from translator.core import TranslationResult


from pipeline.tasks.generate_permutations import generate_permutation_set
from repository.run_builder import RunRecord


import qiskit
from qiskit.circuit.random import random_clifford_circuit

from pipeline.config import PipelineConfig
from pipeline import generate_circuit, run_full_scan, run_extremal_scan, run_random_scan, run_specific_scan
from simulator.config_presets import ScaledSimulator, StandardSimulator
from packer.packer import pack


# program_circuit: qiskit.QuantumCircuit = random_clifford_circuit(40, 40, gates=['x', 'z', 'h', 's', 'cx', 'cz'])

# target_basis = ['x', 'z', 'h', 's', 'cx', 'cz', 'measure']

# tcirc = qiskit.transpile(
#     program_circuit,
#     basis_gates=target_basis,
#     optimization_level=2,
# )
# tcirc.measure_all()

tcirc = qiskit.QuantumCircuit(4)
tcirc.x(0)
tcirc.z(0)
tcirc.x(0)
tcirc.x(1)
tcirc.cx(0,2)
tcirc.z(1)
tcirc.cx(1,2)
tcirc.cz(0,2)
tcirc.x(2)
tcirc.cz(1,3)
tcirc.h(0)
tcirc.h(1)
tcirc.cx(0,3)
tcirc.cx(1,2)
tcirc.x(3)
tcirc.x(0)

tcirc.x(0)
tcirc.x(1)

tcirc.measure_all()
print(tcirc)

config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=1, repetitions=50_000),
    verbose=True,
    save=False,
)

# print(pack(pcirc))

# res: RunRecord = run_full_scan(pcirc, config)
# df = res.dataframe.df

perms = [
    [(0,0), (0,1), (1,0), (1,1)],
    # [(0,0), (1,0), (2,0), (3,0)],
]

# res: RunRecord = run_random_scan(tcirc, 1, config)
# res: RunRecord = run_specific_scan(tcirc, perms,  config)
res: RunRecord = run_full_scan(tcirc, config)
