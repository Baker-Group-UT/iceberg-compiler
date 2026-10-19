"""
codes_used_vs_sel_rate_by_alpha.py - experiment to show how logical selection rate varies with
noise scaling (alpha), grouped by number of codes used.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, run_extremal_scan
from simulator import ScaledNoiseModel, SimulatorConfig
from experiments.shared.example_circuits import n_ghz_state

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 20,
        "figure.titlesize": 20,
    }
)

# Sweep over alphas (noise scaling)
alphas = np.logspace(-1.5, 1, num=20)  # [0.001 ... 10]
all_grouped_dfs = []

for alpha in alphas:
    print(f"Running full scan for alpha = {alpha:.3f}")

    # Build noise model
    noise_model = ScaledNoiseModel(alpha=alpha)
    sim_config = SimulatorConfig(noise_model=noise_model, repetitions=10_000)

    # Configure pipeline
    config = PipelineConfig(
        sim_config=sim_config,
        verbose=False,
        fault_tolerant=False
    )

    program_circuit = n_ghz_state(6)["program"]

    # Run scan
    df = run_extremal_scan(program_circuit, config).dataframe.df

    # Group by number of codes used, average the logical selection rate
    grouped = df.groupby("codes_used")["logical_selection_rate"].mean().reset_index()
    grouped["alpha"] = alpha
    all_grouped_dfs.append(grouped)

# Combine all alpha sweeps into a single DataFrame
final_df = pd.concat(all_grouped_dfs, ignore_index=True)
# clipping for logit scale
final_df["logical_selection_rate"] = final_df["logical_selection_rate"].clip(1e-6, 1 - 1e-6)
# Plot
fig, ax = plt.subplots(figsize=(18, 6))

sns.lineplot(
    data=final_df,
    x="alpha",
    y="logical_selection_rate",
    hue="codes_used",
    linewidth=3.6,
    ax=ax,
    palette="viridis"
)

ax.set_xscale("log")
ax.set_yscale("logit")
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
ax.set_xlabel("Noise Scaling Factor (alpha)")
ax.set_ylabel("Logical Selection Rate")
ax.set_title("Logical Selection Rate vs Noise Scaling (Grouped by Codes Used)")
ax.grid(True, which="both", linestyle="--", linewidth=0.4)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Optional: Legend formatting
ax.legend(title="Codes Used", bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)

plt.tight_layout()

plt.savefig("logical_selection_rate_vs_alpha.svg", format="svg", dpi=300)

plt.show()
