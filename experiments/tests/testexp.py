"""
run_scan.py - high level test file
"""

from math import perm
import math
from numpy import logspace
from packer.merge_packer import MergeMaximizingPacker
from pipeline.analysis import print_df
from simulator import DistanceDependentNoiseModel
from translator.config import TranslatorConfig
from translator.core import TranslationResult


from pipeline.tasks.generate_permutations import generate_permutation_set
from repository.run_builder import RunRecord


import qiskit
from experiments.shared.benchmark_circuits import create_proxy_from_qasm, find_qasm_files, load_proxy_circuits
from experiments.shared.random_circuits import random_gate_circuit

from pipeline.config import PipelineConfig
from pipeline import generate_circuit, run_full_scan, run_extremal_scan, run_random_scan, run_specific_scan, translate_permutation
from simulator.config_presets import ScaledSimulator, StandardSimulator
from packer.packer import HeuristicPacker, PackerResult
from translator.passes import cancel_inverse_gates, commute_hadamards

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

REPO_ROOT = "/Users/cordputer/Documents/Projects/School/Research/Infleqtion/infleqtion-qed-exploration-main"

# Representative medium circuits: one small-medium (~11q) and one larger (~18q)
PROBE_CIRCUITS = [
    f"{REPO_ROOT}/benchmarks/qasmbench/medium/sat_n11",
    f"{REPO_ROOT}/benchmarks/qasmbench/medium/qft_n18",
    f"{REPO_ROOT}/benchmarks/qasmbench/medium/ising_n26",
]

ALPHA = 0.05
PILOT_REPETITIONS = 4_000
TARGET_KEPT_SHOTS = 20_000
MIN_REPETITIONS   = 20_000
MAX_REPETITIONS   = 500_000

IDLE_SCALARS     = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DISTANCE_SCALARS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def _repetitions_from_lsr(lsr: float) -> int:
    denom = max(lsr, 1e-4)
    reps = int(math.ceil(TARGET_KEPT_SHOTS / denom))
    return int(max(MIN_REPETITIONS, min(MAX_REPETITIONS, reps)))


def _make_config(repetitions: int, idle_scalar: float, distance_scalar: float, seed: int) -> PipelineConfig:
    return PipelineConfig(
        sim_config=(
            ScaledSimulator(alpha=ALPHA, repetitions=repetitions, seed=seed)
            .with_stim()
            .with_distance_noise(distance_scalar=distance_scalar)
            .with_idle_noise(idle_time_scalar=idle_scalar)
        ),
        verbose=False,
        translator_config=TranslatorConfig(enable_h_commutation=False).disable_merging(),
    )


if __name__ == "__main__":
    circuits = []
    for path in PROBE_CIRCUITS:
        loaded = load_proxy_circuits(path, replace_rz_rx_with_s=True)
        if loaded:
            name, qc = loaded[0]
            qc.measure_all()
            n = qc.num_qubits
            perm = tight_trivial_packing(n, patch_capacity=2)
            circuits.append((name, qc, perm))
            print(f"Loaded {name}: {n} qubits, {math.ceil(n/2)} codes, depth {qc.depth()}")

    print()
    header = f"{'circuit':<20} {'idle':>6} {'dist':>6} {'TVD':>8} {'LSR':>8} {'reps':>8}"
    print(header)
    print("-" * len(header))

    for name, qc, perm in circuits:
        for idle_scalar in IDLE_SCALARS:
            for distance_scalar in DISTANCE_SCALARS:
                # Pilot: estimate LSR cheaply
                pilot_cfg = _make_config(PILOT_REPETITIONS, idle_scalar, distance_scalar, seed=7)
                pilot_record = run_specific_scan(program_circuit=qc, permutations=[perm], config=pilot_cfg)
                pilot_lsr = pilot_record.dataframe.df["logical_selection_rate"].iloc[0]

                # Scale repetitions to hit ~TARGET_KEPT_SHOTS kept shots
                final_reps = _repetitions_from_lsr(pilot_lsr)

                final_cfg = _make_config(final_reps, idle_scalar, distance_scalar, seed=8)
                record = run_specific_scan(program_circuit=qc, permutations=[perm], config=final_cfg)
                df = record.dataframe.df
                tvd = df["logical_total_variation_distance"].iloc[0]
                lsr = df["logical_selection_rate"].iloc[0]
                print(f"{name:<20} {idle_scalar:>6.2f} {distance_scalar:>6.2f} {tvd:>8.4f} {lsr:>8.4f} {final_reps:>8}")
        print()
