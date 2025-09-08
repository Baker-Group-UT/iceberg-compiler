"""
codes_used_vs_sel_rate_by_alpha.py - experiment to show how logical selection rate varies with
noise scaling (alpha), grouped by number of codes used.
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline import run_full_scan
from simulator import alpha_scale_noise_params, ScaledNoiseModel, SimulatorConfig, DEFAULT_PARAMS

# Define the program circuit
qA, qB, qC, qD, qE, qF = cirq.LineQubit.range(6)
program_circuit = cirq.Circuit(
    cirq.H(qA),
    cirq.CNOT(qA, qB),
    cirq.CNOT(qB, qC),
    cirq.CNOT(qC, qD),
    cirq.CNOT(qD, qE),
    cirq.CNOT(qE, qF),
)

# Sweep over alphas (noise scaling)
alphas = np.logspace(-3, 1, num=10)  # [0.001 ... 10]
all_grouped_dfs = []

for alpha in alphas:
    print(f"Running full scan for alpha = {alpha:.3f}")

    # Build noise model
    scaled_params = alpha_scale_noise_params(base=DEFAULT_PARAMS, alpha=alpha)
    noise_model = ScaledNoiseModel(noise_params=scaled_params)
    sim_config = SimulatorConfig(noise_model=noise_model, repetitions=2, oversample=10)

    # Configure pipeline
    config = PipelineConfig(
        sim_config=sim_config,
        verbose=False,
        fault_tolerant=False
    )

    # Run scan
    df = run_full_scan(program_circuit, config).dataframe.df

    # Group by number of codes used, average the logical selection rate
    grouped = df.groupby("codes_used")["logical_selection_rate"].mean().reset_index()
    grouped["alpha"] = alpha
    all_grouped_dfs.append(grouped)

# Combine all alpha sweeps into a single DataFrame
final_df = pd.concat(all_grouped_dfs, ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    data=final_df,
    x="alpha",
    y="logical_selection_rate",
    hue="codes_used",
    marker="o",
    ax=ax,
    palette="viridis"
)

ax.set_xscale("log")
ax.set_xlabel("Noise Scaling Factor (alpha)", fontsize=12)
ax.set_ylabel("Logical Selection Rate", fontsize=12)
ax.set_title("Logical Selection Rate vs Noise Scaling (Grouped by Codes Used)", fontsize=14)
ax.set_ylim(0, 1.05)
ax.grid(True)

# Optional: Legend formatting
ax.legend(title="Codes Used", bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)

plt.tight_layout()
plt.show()