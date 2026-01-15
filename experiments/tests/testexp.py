"""
run_scan.py - high level test file
"""

from numpy import logspace
from packer.merge_packer import MergeMaximizingPacker
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

def tight_trivial_packing(num_qubits: int, patch_capacity: int = 2):
    """
    Return a tight trivial packing: (0,0),(0,1),(1,0),(1,1),...
    """
    if patch_capacity < 1:
        raise ValueError("patch_capacity must be >= 1")
    return [(i // patch_capacity, i % patch_capacity) for i in range(num_qubits)]

def sparse_trivial_packing(num_qubits: int):
    """
    Return a sparse trivial packing: (0,0),(1,0),(2,0),...
    """
    return [(i, 0) for i in range(num_qubits)]

tcirc: qiskit.QuantumCircuit = random_clifford_circuit(
    num_qubits=20,
    num_gates=200,
    gates=['x', 'z', 'h', 's', 'cx', 'cz'],
    seed=11
)

tcirc.measure_all()

config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=0.25,repetitions=250_000).with_stim().with_distance_based_noise(),
    # sim_config=ScaledSimulator(repetitions=10_000, alpha=1),
    verbose=False,
    seed=5,
)

packer = HeuristicPacker()
packing6g = packer.pack(tcirc, max_patches=10)
packing6d = packer.pack(tcirc, max_patches=12, approach="defensive", explore_all_approaches=True)
packing6b = packer.pack(tcirc, max_patches=14, approach="balanced", explore_all_approaches=True)
packing8g = packer.pack(tcirc, max_patches=16, approach="greedy", explore_all_approaches=True)
# packing8d = packer.pack(tcirc, max_patches=9, approach="defensive", explore_all_approaches=True)
# packing8b = packer.pack(tcirc, max_patches=10, approach="balanced", explore_all_approaches=True)
# packing10 = packer.pack(tcirc, max_patches=24)
# packing12 = packer.pack(tcirc, max_patches=28)
perm6g = packer.packing_to_permutation(packing6g.qubit_to_patch)
perm6d = packer.packing_to_permutation(packing6d.qubit_to_patch)
perm6b = packer.packing_to_permutation(packing6b.qubit_to_patch)
perm8g = packer.packing_to_permutation(packing8g.qubit_to_patch)
# perm8d = packer.packing_to_permutation(packing8d.qubit_to_patch)
# perm8b = packer.packing_to_permutation(packing8b.qubit_to_patch)
# perm10 = packer.packing_to_permutation(packing10.qubit_to_patch)
# perm12 = packer.packing_to_permutation(packing12.qubit_to_patch)

tight_packing = tight_trivial_packing(tcirc.num_qubits)
sparse_packing = sparse_trivial_packing(tcirc.num_qubits)

m_packer_mg = MergeMaximizingPacker(solo_strategy="most_gates")
m_packer_pts = MergeMaximizingPacker(solo_strategy="pair_then_solo")
m_packer_mh = MergeMaximizingPacker(solo_strategy="most_hadamards")

permutation_mg10 =   m_packer_mg.pack(tcirc, num_codes=10).permutation
permutation_pts10 = m_packer_pts.pack(tcirc, num_codes=10).permutation
permutation_mh10 =   m_packer_mh.pack(tcirc, num_codes=10).permutation

permutation_mg12 =   m_packer_mg.pack(tcirc, num_codes=12).permutation
permutation_pts12 = m_packer_pts.pack(tcirc, num_codes=12).permutation
permutation_mh12 =   m_packer_mh.pack(tcirc, num_codes=12).permutation

permutation_mg14 =   m_packer_mg.pack(tcirc, num_codes=14).permutation
permutation_pts14 = m_packer_pts.pack(tcirc, num_codes=14).permutation
permutation_mh14 =   m_packer_mh.pack(tcirc, num_codes=14).permutation

permutation_mg16 =   m_packer_mg.pack(tcirc, num_codes=16).permutation
permutation_pts16 = m_packer_pts.pack(tcirc, num_codes=16).permutation
permutation_mh16 =   m_packer_mh.pack(tcirc, num_codes=16).permutation

permutation_mg18 =   m_packer_mg.pack(tcirc, num_codes=18).permutation
permutation_pts18 = m_packer_pts.pack(tcirc, num_codes=18).permutation
permutation_mh18 =   m_packer_mh.pack(tcirc, num_codes=18).permutation

perms = [
    perm6g,
    perm6d,
    perm6b,
    perm8g,
    # perm8d,
    # perm8b,
    tight_packing,
    sparse_packing,
    permutation_mg10,
    permutation_pts10,
    permutation_mh10,
    permutation_mg12,
    permutation_pts12,
    permutation_mh12,
    permutation_mg14,
    permutation_pts14,
    permutation_mh14,
    permutation_mg16,
    permutation_pts16,
    permutation_mh16,
    permutation_mg18,
    permutation_pts18,
    permutation_mh18,
]

# res = run_specific_scan(tcirc, perms, config)
# # res = run_full_scan(program_circuit=tcirc, config=config, added_perms=perms)
res = run_random_scan(program_circuit=tcirc, m=10, config=config, added_perms=perms)
print_df.print_df_table(res.dataframe.df, sort_by="logical_total_variation_distance", hide_permutation=True)


# What I expect to find:
# If there is no gate merging, a circuit that uses more codes will likely perform worse due to distance based errors
# Unless we can perform some circuit chunking to reduce total depth, 
# My guess is random circuits will benefit from additional codes more than programmatic circuits due to a sequential
# dependence on previous information
# In general, using fewer codes will extend the depth of a circuit. However this depends on the circuit.
# Gate count has the biggets impact on fidelity and TVD (predominantly 2 qubit, then 1 qubit)
# Selection rate is unrelated to TVD mostly because it has too much baked into its value.
#   Don't know if reducing errors or catching noise
# In general, better code and depth equality yields better TVD

# Good for permutation quality:
# - Low depth
# - High code and depth equality
# - Intetional use of codes, more =/= better
# - As much gate merging as possible, lower 1q and 2q gate counts

# Ideas for algo support:
# - Network analysis of the circuit DAGs,
#   - Can we check how critical each qubit is and / or how many sequential ops there aret?
#   - i.e. 4 qubits, 2 receive only single qubit gates
#   - Should long paths be spread across codes? in some way they have to, right?
# - 