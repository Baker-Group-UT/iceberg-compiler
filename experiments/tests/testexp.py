"""
run_scan.py - high level test file
"""

from pipeline.analysis import print_df
from simulator import DistanceDependentNoiseModel
from translator.core import TranslationResult


from pipeline.tasks.generate_permutations import generate_permutation_set
from repository.run_builder import RunRecord


import qiskit
from qiskit.circuit.random import random_clifford_circuit

from pipeline.config import PipelineConfig
from pipeline import generate_circuit, run_full_scan, run_extremal_scan, run_random_scan, run_specific_scan, translate_permutation
from simulator.config_presets import ScaledSimulator, StandardSimulator
from packer.packer import HeuristicPacker, PackerResult

# program_circuit: qiskit.QuantumCircuit = random_clifford_circuit(
#     5, 
#     20, 
#     gates=['x', 'z', 'h', 'cx', 'cz'],
#     seed=5
# )
# # program_circuit.measure_all()
# # tcirc=program_circuit
# # orig
# tcirc = qiskit.QuantumCircuit(4)
# tcirc.h(0)
# tcirc.h(3)
# tcirc.x(1)
# tcirc.cx(0,1)
# tcirc.cx(3,2)
# # tcirc.h(2)
# tcirc.z(2)
# tcirc.measure_all()

# print(tcirc)

# config = PipelineConfig(
#     sim_config=StandardSimulator(repetitions=50_000),#.with_stim(),
#     # sim_config=ScaledSimulator(repetitions=10_000, alpha=1),
#     verbose=False
# )

# packer = HeuristicPacker()
# packing2 = packer.pack(tcirc, max_patches=3)
# packing4 = packer.pack(tcirc, max_patches=4)
# perm2 = packer.packing_to_permutation(packing2.qubit_to_patch)
# perm4 = packer.packing_to_permutation(packing4.qubit_to_patch)

# perms = [
#     perm2,
#     perm4,
# ]

# # res = run_specific_scan(tcirc, perms, config)
# res = run_full_scan(program_circuit=tcirc, config=config, added_perms=perms)
# # res = run_random_scan(program_circuit=tcirc, m=20, config=config, added_perms=perms)
# print_df.print_df_table(res.dataframe.df)

qc = qiskit.QuantumCircuit(3)
qc.h(0)
qc.x(1)
qc.z(2)
qc.barrier()
qc.x(1)
qc.barrier()
qc.cx(0,1)
qc.cz(2,1)
qc.h(1)
qc.draw('mpl', filename="testcirc.svg")
