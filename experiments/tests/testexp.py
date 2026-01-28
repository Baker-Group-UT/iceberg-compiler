"""
run_scan.py - high level test file
"""

from math import perm
from numpy import logspace
from packer.merge_packer import MergeMaximizingPacker
from pipeline.analysis import print_df
from simulator import DistanceDependentNoiseModel
from translator.core import TranslationResult


from pipeline.tasks.generate_permutations import generate_permutation_set
from repository.run_builder import RunRecord


import qiskit
from experiments.shared.random_circuits import random_gate_circuit

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

if __name__ == "__main__":
    from translator.merging import preprocess_program_for_merging

    qc = random_gate_circuit(num_qubits=2, num_gates=20, gates=["h", "x", "cx"], seed=10)
    qc.measure_all()

    config = PipelineConfig(
        sim_config=ScaledSimulator(alpha=0.25,repetitions=10_000)
        .with_stim()
        .with_distance_noise(1e-10)
        .with_idle_noise(1e-10),
        # sim_config=ScaledSimulator(repetitions=10_000, alpha=1),
        verbose=True,
        seed=5,
    )
    print(qc)
    t_merg: RunRecord = run_specific_scan(program_circuit=qc, permutations=[[(0, 0), (0, 1)]], config=config)
