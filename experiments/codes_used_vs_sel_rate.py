"""
codes_used_vs_sel_rate_by_alpha.py - experiment to show how logical selection rate varies with
noise scaling (alpha), grouped by number of codes used.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline import run_full_scan
from simulator import ScaledNoiseModel, SimulatorConfig
from experiments.shared.example_circuits import n_ghz_state

# Sweep over alphas (noise scaling)
alphas = np.logspace(-3, 1, num=10)  # [0.001 ... 10]
all_grouped_dfs = []

for alpha in alphas:
    print(f"Running full scan for alpha = {alpha:.3f}")

    # Build noise model
    noise_model = ScaledNoiseModel(alpha=alpha)
    sim_config = SimulatorConfig(noise_model=noise_model, repetitions=20)

    # Configure pipeline
    config = PipelineConfig(
        sim_config=sim_config,
        verbose=False,
        fault_tolerant=False
    )

    program_circuit = n_ghz_state(5)["logical"]

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