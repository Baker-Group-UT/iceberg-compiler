"""
Compare Qiskit Aer vs Stim permutation rankings across noise scales.

For each random circuit and noise scaling factor we run the logical permutation
scan twice (once per backend), sort the permutations by
`logical_total_variation_distance` and `logical_selection_rate`, and measure how
different the Aer and Stim orderings are via the normalized Kendall τ distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit
from scipy.stats import kendalltau

from pipeline import run_full_scan, run_specific_scan
from pipeline.config import PipelineConfig
from simulator.config_presets import StandardSimulator
from simulator.noise_models import ScaledNoiseModel

# Representative permutations (same as other noise scaling experiments)
PERM_GROUP = [(0, 0), (0, 1), (1, 0), (1, 1)]
PERM_INNER_SWAP = [(0, 0), (1, 0), (0, 1), (1, 1)]
PERM_OUTER_SWAP = [(1, 1), (0, 1), (1, 0), (0, 0)]
PERM_DIFF = [(0, 0), (1, 0), (2, 0), (3, 0)]
PERMUTATIONS: List[List[Tuple[int, int]]] = [PERM_GROUP, PERM_INNER_SWAP, PERM_OUTER_SWAP, PERM_DIFF]


@dataclass(frozen=True)
class MetricSpec:
    column: str
    ascending: bool
    label: str


METRIC_SPECS: Sequence[MetricSpec] = (
    MetricSpec(
        column="logical_total_variation_distance",
        ascending=True,
        label="Logical TV distance",
    ),
    MetricSpec(
        column="logical_selection_rate",
        ascending=False,
        label="Logical selection rate",
    ),
)


def build_circuit(seed: int = 17, qubits: int = 4, depth: int = 25) -> QuantumCircuit:
    """Generate and transpile a random Clifford circuit for the experiment."""
    program_circuit = random_clifford_circuit(
        qubits,
        depth,
        seed=seed,
        gates=["x", "h", "z", "s", "cx", "cz"],
    )
    # program_circuit = QuantumCircuit(2)
    # program_circuit.h(0)
    # program_circuit.cx(0, 1)
    # program_circuit.cx(0, 1)
    # program_circuit.h(0)
    program_circuit.measure_all()
    return program_circuit


def run_scan_with_backend(
    tcirc: QuantumCircuit,
    sim_config: PipelineConfig,
):
    """Convenience wrapper for run_specific_scan so we can swap configs."""
    return run_specific_scan(
        program_circuit=tcirc,
        permutations=PERMUTATIONS,
        config=sim_config,
    )


def _ranking_map(
    df,
    metric: str,
    *,
    ascending: bool,
) -> Dict[Tuple[Tuple[int, int], ...], int]:
    """Return {perm_key: rank} for the requested metric."""
    subset = (
        df[["perm_key", metric]]
        .dropna(subset=[metric])
        .set_index("perm_key")
        .sort_index()
    )
    sorted_idx = subset.sort_values(
        by=metric,
        ascending=ascending,
        kind="mergesort",
    ).index
    return {perm_key: rank for rank, perm_key in enumerate(sorted_idx)}


def kendall_distance_for_metric(
    aer_df,
    stim_df,
    metric: MetricSpec,
) -> float:
    """
    Return the normalized Kendall τ distance between Aer/Stim permutation orders.
    """
    aer_rank = _ranking_map(aer_df, metric.column, ascending=metric.ascending)
    stim_rank = _ranking_map(stim_df, metric.column, ascending=metric.ascending)
    common = sorted(set(aer_rank) & set(stim_rank))
    if len(common) < 2:
        return 0.0

    aer_series = [aer_rank[perm_key] for perm_key in common]
    stim_series = [stim_rank[perm_key] for perm_key in common]
    tau, _ = kendalltau(aer_series, stim_series)
    if np.isnan(tau):
        return 0.0
    return 0.5 * (1.0 - tau)


def compute_kendall_distances(
    circuits: List[QuantumCircuit],
    scalars: np.ndarray,
    *,
    repetitions: int,
):
    """
    For each circuit/noise scale return the Aer-vs-Stim Kendall τ distances per metric.
    """
    distances = {
        metric.column: np.zeros((len(circuits), len(scalars)), dtype=float)
        for metric in METRIC_SPECS
    }

    for ci, tcirc in enumerate(circuits):
        for si, scalar in enumerate(scalars):
            noise_model = ScaledNoiseModel(alpha=scalar)
            aer_sim = (
                StandardSimulator(repetitions=repetitions)
                .with_noise_model(noise_model)
                # .with_stim(False)
            )
            stim_sim = (
                StandardSimulator(repetitions=repetitions)
                .with_noise_model(noise_model)
                # .with_stim(True)
                # .with_stim_scaling(None)
            )
            aer_config = PipelineConfig(sim_config=aer_sim, verbose=False, save=False)
            stim_config = PipelineConfig(sim_config=stim_sim, verbose=False, save=False)

            aer_record = run_scan_with_backend(tcirc, aer_config)
            stim_record = run_scan_with_backend(tcirc, stim_config)
            aer_df = aer_record.dataframe.df
            stim_df = stim_record.dataframe.df

            line_items = []
            for metric in METRIC_SPECS:
                distance = kendall_distance_for_metric(aer_df, stim_df, metric)
                distances[metric.column][ci, si] = distance
                line_items.append(f"{metric.label}: {distance:.3f}")
            print(
                f"[circuit {ci + 1}/{len(circuits)}] "
                f"alpha={scalar:.3e} | " + ", ".join(line_items)
            )

    return distances


def plot_kendall_distances(
    scalars: np.ndarray,
    distances: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    Plot the mean ±σ Kendall τ distance per metric as a function of noise scale.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    colors = plt.get_cmap("tab10").colors

    for idx, metric in enumerate(METRIC_SPECS):
        data = distances[metric.column]
        if np.all(np.isnan(data)):
            continue
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        color = colors[idx % len(colors)]
        ax.plot(
            scalars,
            mean,
            color=color,
            linewidth=2.5,
            label=metric.label,
        )
        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)
        ax.fill_between(
            scalars,
            lower,
            upper,
            color=color,
            alpha=0.25,
        )

    ax.set_xlabel("Noise scale (α)")
    ax.set_ylabel("Normalized Kendall τ distance (Aer vs Stim)")
    ax.set_title("Permutation ranking mismatch across noise scales")
    ax.legend(loc="upper left", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg", facecolor=fig.get_facecolor())
    print(f"Saved Kendall τ plot to {output_path}")


def main() -> None:
    num_circuits = 10
    repetitions = 2_000
    scalars = np.logspace(-1.5, 1.5, num=10)

    circuits = [
        build_circuit(seed=1000 + idx, qubits=2, depth=25)
        for idx in range(num_circuits)
    ]

    distances = compute_kendall_distances(
        circuits,
        scalars,
        repetitions=repetitions,
    )

    plot_kendall_distances(
        scalars,
        distances,
        output_path="results/stim_vs_aer_permutation_kendall_tau.svg",
    )


if __name__ == "__main__":
    main()
