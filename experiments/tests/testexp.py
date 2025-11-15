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
from pipeline import generate_circuit, run_full_scan, run_extremal_scan, run_random_scan, run_specific_scan, translate_permutation
from simulator.config_presets import ScaledSimulator, StandardSimulator
from packer.packer import HeuristicPacker, PackerResult

program_circuit: qiskit.QuantumCircuit = random_clifford_circuit(15, 50, gates=['x', 'z', 'h', 's', 'cx', 'cz'])

# target_basis = ['x', 'z', 'h', 's', 'cx', 'cz', 'measure']

# tcirc = qiskit.transpile(
#     program_circuit,
#     basis_gates=target_basis,
#     optimization_level=2,
# )
program_circuit.measure_all()

# orig
# tcirc = qiskit.QuantumCircuit(4)
# tcirc.h([0,1,2,3])
# tcirc.cx(0,2)
# tcirc.cx(0,1)
# tcirc.h([0,1,3])
# tcirc.cz(1,3)
# tcirc.x([0,1,2,3])
# tcirc.measure_all()

# print(tcirc)

config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=1, repetitions=25_000).with_stim(),
    verbose=True,
    save=False,
)
print(program_circuit)
packer = HeuristicPacker(max_patches=8)
packing = packer.pack(program_circuit, approach="balanced")
permutation20 = packer.packing_to_permutation(packing.qubit_to_patch)
print(permutation20)

packer = HeuristicPacker(max_patches=10)
packing = packer.pack(program_circuit, approach="balanced")
permutation28 = packer.packing_to_permutation(packing.qubit_to_patch)
print(permutation28)

packer = HeuristicPacker(max_patches=13)
packing: PackerResult = packer.pack(program_circuit, approach="balanced")
permutation35 = packer.packing_to_permutation(packing.qubit_to_patch)
print(permutation35)
# perms = [
#     [(0,0), (0,1), (1,0), (1,1)], # in this perm, 0 goes 0, 1 goes 1, 2 goes 2, 3 goes 3
#     [(1,0), (0,0), (1,1), (0,1)], # we need to relabel so that this recreates the above
#     # [(1,0), (0,0), (1,1), (0,1)], # we need to relabel so that this recreates the above
#     # [(0,0), (0,1)],
#     # [(0,0), (1,0), (2,0), (3,0)],
# ]

# out_circ = translate_permutation(tcirc, perms[1])
# out_circ.circuit.draw('mpl', filename="output_circ_relabeled.svg", fold=-1)
# # # res: RunRecord = run_random_scan(tcirc, 1, config)
res: RunRecord = run_random_scan(program_circuit,  40, config=config, added_perms=[permutation20, permutation28, permutation35])
# # res: RunRecord = run_full_scan(tcirc, config)
