"""
run_scan.py - high level test file
"""
from qiskit.circuit.quantumcircuit import QuantumCircuit


import numpy as np
import qiskit

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan
from simulator.config_presets import StandardSimulator
from simulator.noise_models import ScaledNoiseModel
import matplotlib.pyplot as plt
from qiskit.circuit.random import random_circuit, random_clifford_circuit

permGroup = [(0, 0), (0, 1), (1, 0), (1, 1)]
# permDiff =  [(1, 0), (0, 0), (1, 1), (0, 1)]
permDiff =  [(0, 0), (1, 0), (2, 0), (3,0)]

# program_circuit = qiskit.QuantumCircuit(2)
# program_circuit.rz(10, 0)
# program_circuit.cx(0, 1)
# program_circuit.measure_all()

# program_circuit = random_circuit(
#     2, depth=20, measure=True, seed=123, num_operand_distribution={1: 0.5, 2: 0.5}
# )

# program_circuit: QuantumCircuit = random_clifford_circuit(4, 4, seed=17) # low fidelity
# program_circuit: QuantumCircuit = random_clifford_circuit(4, 100, seed=seed) # low fidelity
# This circuit is failing most of the time, works sometimes (seed 57, 4 qubits, 6 gates) but the weird thing is 
# even with the same seed it is generating slightly different circuits that are probably equivalent
# program_circuit: QuantumCircuit = random_clifford_circuit(4, 6, seed=57)

program_circuit: QuantumCircuit = random_clifford_circuit(4, 25)

program_circuit.measure_all()

print(program_circuit)

target_basis = ['x', 'z', 'h', 'rz', 'cx', 'cz', 'measure']

tcirc = qiskit.transpile(
    program_circuit,
    basis_gates=target_basis,
    optimization_level=0,
)

print(tcirc)

scalars = np.logspace(-2.5, 2.5, num=20)
results = []
for scalar in scalars:
    # print(f"\n=== Running simulation for p_scalar = {scalar:.3f} ===")

    noise_model = ScaledNoiseModel(alpha=scalar)

    config = PipelineConfig(
        sim_config=StandardSimulator(repetitions=10_000).with_noise_model(noise_model=noise_model),
        verbose=False,
        save=False,
    )

    record = run_specific_scan(
        program_circuit=tcirc,
        permutations=[permGroup, permDiff],
        config=config,
    )

    results.append((scalar, record))

f_group = []
f_diff = []
lsr_group = []
lsr_diff = []

for scalar, record in results:
    df = record.dataframe.df
    f_group.append(1- df.loc[0, "logical_total_variation_distance"])
    f_diff.append(1- df.loc[1, "logical_total_variation_distance"])
    lsr_group.append(df.loc[0, "logical_selection_rate"])
    lsr_diff.append(df.loc[1, "logical_selection_rate"])

plt.figure(figsize=(6, 4))
plt.semilogx(scalars, f_group, "-o", label="Group Permutation F")
plt.semilogx(scalars, f_diff, "-o", label="Different Permutation F")
plt.semilogx(scalars, lsr_group, "-o", label="Group Permutation LSR")
plt.semilogx(scalars, lsr_diff, "-o", label="Different Permutation LSR")
plt.xlabel("p_scalar")
plt.ylabel("Logical Fidelity")
plt.title("Fidelity vs Noise Scale")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
