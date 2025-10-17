"""
run_scan.py - high level test file
"""
import numpy as np
import qiskit

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan
from simulator.config_presets import StandardSimulator
from simulator.noise_models import ScaledNoiseModel
import matplotlib.pyplot as plt


program_circuit = qiskit.QuantumCircuit(2)
program_circuit.x(0)
program_circuit.cx(0, 1)
program_circuit.measure_all()

permGroup = [(0, 0), (0, 1)]
permDiff = [(0, 0), (1, 0)]

scalars = np.logspace(0, 1.5, num=25)

results = []

for scalar in scalars:
    print(f"\n=== Running simulation for p_scalar = {scalar:.3f} ===")

    noise_model = ScaledNoiseModel(alpha=scalar)

    config = PipelineConfig(
        sim_config=StandardSimulator(repetitions=10_000).with_noise_model(ScaledNoiseModel(scalar)),
        verbose=False,
        save=False,
    )

    record = run_specific_scan(
        program_circuit=program_circuit,
        permutations=[permGroup, permDiff],
        config=config,
    )

    print(record.dataframe.df["fidelity"])
    results.append((scalar, record))

f_group = []
f_diff = []

for scalar, record in results:
    df = record.dataframe.df
    f_group.append(df.loc[0, "fidelity"])
    f_diff.append(df.loc[1, "fidelity"])

plt.figure(figsize=(6, 4))
plt.semilogx(scalars, f_group, "-o", label="Group Permutation")
plt.semilogx(scalars, f_diff, "-o", label="Different Permutation")
plt.xlabel("p_scalar")
plt.ylabel("Logical Fidelity")
plt.title("Fidelity vs Noise Scale")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()