"""
average_fidelity.py - an experiment to find the average fidelity across all permutations
of a circuit as well as the best and worst fidelities. Final result is a plot with this graph 
for multiple different circuits including the bare circuit fidelity.
"""

import matplotlib.pyplot as plt
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline import run_full_scan, simulate_circuit
from simulator.config_presets import StandardSimulator
from experiments.shared.example_circuits import ALL_FACTORIES

# Create a config for Pipeline
config = PipelineConfig(
    sim_config=StandardSimulator(repetitions=100),
    verbose=False,
    fault_tolerant=False
)

# Choose circuits to scan
circuits_to_test = {
    "Shor-3": ALL_FACTORIES["unscalable_shors_3"]()["program"],
    "Square-Cluster-4": ALL_FACTORIES["square_cluster_4"]()["program"],
    # "Grover-3": ALL_FACTORIES["grover_3"]()["program"], # @TODO: FIX THIS, CCX BROKEN
    "GHZ-3": ALL_FACTORIES["ghz_3"]()["program"],
    # "Half-Adder": ALL_FACTORIES["half_adder"]()["program"],
    # "Ripple-Incrementer-3": ALL_FACTORIES["ripple_incrementer_3"]()["program"],
    "Toffoli": ALL_FACTORIES["toffoli_like_snippet"]()["program"],
}

# Collect fidelity summary stats
summary_rows = []

for name, circuit in circuits_to_test.items():
    print(f"Running permutation scan for {name}...")

    # Full permutation scan
    record = run_full_scan(program_circuit=circuit, config=config)
    df = record.dataframe.df

    mean_fidelity = df["fidelity"].mean()
    min_fidelity = df["fidelity"].min()
    max_fidelity = df["fidelity"].max()

    # Bare circuit fidelity
    bare_record = simulate_circuit(circuit=circuit, config=config)
    bare_df = bare_record.dataframe.df.copy()
    bare_fidelity = bare_df["fidelity"].iloc[0] if not bare_df.empty else None

    summary_rows.append({
        "circuit": name,
        "mean": mean_fidelity,
        "min": min_fidelity,
        "max": max_fidelity,
        "bare": bare_fidelity
    })

# Convert to DataFrame
summary_df = pd.DataFrame(summary_rows)

# Plot
plt.figure(figsize=(12, 7))  # larger canvas

# Plot permutation mean fidelity with error bars
plt.errorbar(
    summary_df["circuit"],
    summary_df["mean"],
    yerr=[summary_df["mean"] - summary_df["min"], summary_df["max"] - summary_df["mean"]],
    fmt="o",
    markersize=8,         # bigger mean point
    capsize=6,
    capthick=2,
    linewidth=2,
    color="blue",
    ecolor="gray",
    label="Permutation Fidelity (mean ± range)"
)

# Plot bare circuit fidelity
plt.scatter(
    summary_df["circuit"],
    summary_df["bare"],
    marker="x",
    color="red",
    s=80,                 # bigger X marker
    linewidths=2,
    label="Bare Circuit Fidelity"
)

plt.xticks(rotation=30, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0, 1.05)
plt.ylabel("Fidelity", fontsize=13)
plt.title("Average Permutation Fidelity vs Bare Circuit Fidelity", fontsize=14)
plt.grid(True)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()