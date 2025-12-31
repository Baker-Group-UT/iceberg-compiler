"""
run_scan.py - high level test file
"""

from numpy import logspace
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
# tcirc.h(2)
# tcirc.z(2)
# tcirc.measure_all()

# tcirc = qiskit.QuantumCircuit(2)
# tcirc.x(0)
# tcirc.cx(0,1)
# tcirc.measure_all()
tcirc: qiskit.QuantumCircuit = random_clifford_circuit(
    num_qubits=10,
    num_gates=100,
    gates=['x', 'z', 'h', 's', 'cx', 'cz'],
    seed=11
)
tcirc.measure_all()

config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=0.5,repetitions=100_000).with_distance_based_noise().with_stim(),
    # sim_config=ScaledSimulator(repetitions=10_000, alpha=1),
    verbose=False,
    seed=5,
)

# packer = HeuristicPacker()
# packing6 = packer.pack(tcirc, max_patches=16)
# packing8 = packer.pack(tcirc, max_patches=20)
# packing10 = packer.pack(tcirc, max_patches=24)
# packing12 = packer.pack(tcirc, max_patches=28)
# perm6 = packer.packing_to_permutation(packing6.qubit_to_patch)
# perm8 = packer.packing_to_permutation(packing8.qubit_to_patch)
# perm10 = packer.packing_to_permutation(packing10.qubit_to_patch)
# perm12 = packer.packing_to_permutation(packing12.qubit_to_patch)

# perms = [
#     perm6,
#     perm8,
#     perm10,
#     perm12,
# ]
# perms = [
#     [(0,0), (0,1)],
#     [(0,0), (1,0)]
# ]

# res = run_specific_scan(tcirc, perms, config)
# # res = run_full_scan(program_circuit=tcirc, config=config, added_perms=perms)
res = run_random_scan(program_circuit=tcirc, m=1, config=config)#, added_perms=perms)
print_df.print_df_table(res.dataframe.df)

# qc = qiskit.QuantumCircuit(3)
# qc.h(0)
# qc.x(1)
# qc.z(2)
# qc.barrier()
# qc.x(1)
# qc.barrier()
# qc.cx(0,1)
# qc.cz(2,1)
# qc.h(1)
# qc.draw('mpl', filename="testcirc.svg")

# ---- Linear distance-scaling specific scan ----
# Sweep a few linear distance scalars to see how the distance-dependent noise impacts the same circuit/permutation set.
# linear_distance_scalars = logspace(0, 8, 10)
# for scalar in linear_distance_scalars:
#     linear_config = PipelineConfig(
#         sim_config=StandardSimulator(repetitions=5_000).with_distance_based_noise(distance_scalar=scalar),
#         verbose=False,
#     )
#     linear_res = run_specific_scan(tcirc, perms, linear_config)
#     print(f"\nLinear distance scaling run (distance_scalar={scalar}):")
#     print_df.print_df_table(linear_res.dataframe.df)
