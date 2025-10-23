"""
run_scan_cnot_sweep.py
"""
import random
import qiskit
import numpy as np
import matplotlib.pyplot as plt

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan
from simulator.config_presets import ScaledSimulator

# Permutations
permGroup = [(0, 0), (0, 1)]  # A,B same code
permDiff  = [(0, 0), (1, 0)]  # A,B different codes
perms = [permGroup, permDiff]

# Fixed config (tweak as desired)
config = PipelineConfig(
    sim_config=ScaledSimulator(alpha=1, repetitions=5_000).with_seed(6),
    verbose=False,
    save=False,
    seed=4,
)

seed = 4
random.seed(4)
RANDOM_BIT = False
def make_circuit(num_cnots: int) -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    for _ in range(num_cnots):
        if RANDOM_BIT:
            bit = random.randint(0,1)
        else:
            bit = 0
        other_bit = 1 - bit
        qc.cx(bit, other_bit)

    qc.measure_all()
    return qc

cnot_counts = list(range(1, 101))
fid_group = []
fid_diff  = []

for n in cnot_counts:
    program_circuit = make_circuit(n)
    record = run_specific_scan(
        program_circuit=program_circuit,
        permutations=perms,
        config=config,
    )
    df = record.dataframe.df  # expects columns: 'permutation', 'fidelity'
    # Pull fidelities in the same order as `perms`
    f_group = df.loc[df["permutation"].apply(tuple) == tuple(permGroup), "fidelity"].iloc[0]
    f_diff  = df.loc[df["permutation"].apply(tuple) == tuple(permDiff),  "fidelity"].iloc[0]
    fid_group.append(f_group)
    fid_diff.append(f_diff)

# Plot
plt.figure(figsize=(7, 4.5))
plt.plot(cnot_counts, fid_group, label="Same-code (A,B)", marker="o", linewidth=1, markersize=3)
plt.plot(cnot_counts, fid_diff,  label="Different-codes (A,B)", marker="s", linewidth=1, markersize=3)
plt.xlabel("Number of CNOTs in circuit")
plt.ylabel("Fidelity")
plt.title("Fidelity vs. CNOT count for two permutations")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()