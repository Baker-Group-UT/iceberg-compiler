"""
scalar_parameters.py - experiment that creates a 2D plot of permutation fidelities vs alpha parameter scalars
"""

import qiskit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, concat

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan, simulate_circuit
from archive.repository.run_builder import RunRecord
from simulator import ScaledNoiseModel, SimulatorConfig

# Define the program circuit
program_circuit = qiskit.QuantumCircuit(3)
program_circuit.h(0)
program_circuit.cx(0,1)
program_circuit.cx(0,2)
program_circuit.h(0)

perms_to_scan = [
    [(0, 0), (0, 1), (1, 0)],
    [(0, 1), (0, 0), (1, 0)],
    [(1, 0), (0, 0), (0, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1), (0, 0), (1, 1)],
    [(1, 1), (0, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (2, 1), (1, 0)],
]

# Sweep alpha values (logarithmic scale)
alphas = np.logspace(-1, 1, num=7)  # e.g., [0.1, 0.215, ..., 10]
overall_df = []

for alpha in alphas:
    print(f"\nRunning alpha = {alpha:.3f}")
    noise_model = ScaledNoiseModel(alpha=alpha)
    sim_config = SimulatorConfig(noise_model=noise_model, repetitions=1000)
    config = PipelineConfig(
        sim_config=sim_config,
        verbose=False,
        fault_tolerant=False,
    )

    # Run extremal permutations
    perm_record: RunRecord = run_specific_scan(program_circuit=program_circuit, permutations=perms_to_scan, config=config)
    perm_df: DataFrame = perm_record.dataframe.df.copy()
    perm_df["source"] = perm_df["permutation"].apply(str)  # use permutation key as source label
    perm_df["alpha"] = alpha

    # Run bare circuit
    bare_record: RunRecord = simulate_circuit(circuit=program_circuit, config=config)
    bare_df: DataFrame = bare_record.dataframe.df.copy()
    bare_df["source"] = "bare"
    bare_df["alpha"] = alpha

    # Combine
    combined = concat([perm_df, bare_df], ignore_index=True)
    overall_df.append(combined)

# Aggregate into one DataFrame
final_df = concat(overall_df, ignore_index=True)

# plot
fig, ax = plt.subplots(figsize=(14, 6))  # wider canvas

sns.lineplot(
    data=final_df,
    x="alpha",
    y="fidelity",
    hue="source",
    marker="o",
    ax=ax
)

ax.set_xscale("log")
ax.set_xlabel("Noise Scaling Factor (alpha)")
ax.set_ylabel("Fidelity")
ax.set_title("Permutation Fidelity vs Noise Scaling (α)")

# Shrink the main plot to the left to make space for the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# Place the legend fully outside on the right
ax.legend(
    title="source",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),  # push legend farther out
    borderaxespad=0,
    fontsize="x-small"
)

plt.show()