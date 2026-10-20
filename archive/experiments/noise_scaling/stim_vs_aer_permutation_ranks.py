"""
Compare Qiskit Aer vs Stim permutation rankings across noise scales.

For each random circuit and noise scaling factor we run a full Stim permutation
scan, pick a handful of permutations that cover a wide performance range,
simulate that subset with Aer, and then compare the Aer vs Stim rankings via
the normalized Kendall τ distance.
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

TVD_COLUMN = "logical_total_variation_distance"
MAX_SELECTED_PERMUTATIONS = 5


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


def _normalize_permutation(perm: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """Ensure permutations are stored as tuples for repeatable downstream use."""
    return [tuple(pair) for pair in perm]


def select_tvd_spread_permutations(
    df,
    *,
    column: str = TVD_COLUMN,
    max_count: int = MAX_SELECTED_PERMUTATIONS,
) -> List[List[Tuple[int, int]]]:
    """
    Select permutations whose logical TVD values are maximally spread apart.
    """
    subset = (
        df[["perm_key", "permutation", column]]
        .dropna(subset=[column])
        .sort_values(by=column, ascending=True, kind="mergesort")
    )
    if subset.empty:
        return []

    values = subset[column].to_numpy()
    target_count = min(max_count, len(subset))
    if target_count == len(subset):
        selected = subset
    else:
        chosen_indices: List[int] = [0]
        if target_count > 1 and len(subset) > 1:
            chosen_indices.append(len(subset) - 1)

        while len(chosen_indices) < target_count:
            remaining = [idx for idx in range(len(subset)) if idx not in chosen_indices]
            best_idx = None
            best_distance = -np.inf
            for idx in remaining:
                distance = min(abs(values[idx] - values[c]) for c in chosen_indices)
                if distance > best_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is None:
                break
            chosen_indices.append(best_idx)

        chosen_indices.sort()
        selected = subset.iloc[chosen_indices]
    return [_normalize_permutation(perm) for perm in selected["permutation"].tolist()]


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
            stim_sim = (
                StandardSimulator(repetitions=repetitions)
                .with_noise_model(noise_model)
                .with_stim(True)
                # .with_stim_scaling(None)
            )
            stim_config = PipelineConfig(sim_config=stim_sim, verbose=False, save=False)

            stim_record = run_full_scan(tcirc, stim_config)
            stim_df = stim_record.dataframe.df
            permutations = select_tvd_spread_permutations(stim_df)
            print(permutations)
            if not permutations:
                for metric in METRIC_SPECS:
                    distances[metric.column][ci, si] = np.nan
                print(
                    f"[circuit {ci + 1}/{len(circuits)}] "
                    f"alpha={scalar:.3e} | no valid permutations selected"
                )
                continue

            aer_sim = (
                StandardSimulator(repetitions=repetitions)
                .with_noise_model(noise_model)
            )
            aer_config = PipelineConfig(sim_config=aer_sim, verbose=False, save=False)
            aer_record = run_specific_scan(tcirc, permutations, aer_config)
            aer_df = aer_record.dataframe.df

            line_items = [f"perms={len(permutations)}"]
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
    ax.set_ylabel("Normalized Kendall τ distance (Stim vs Aer)")
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
    repetitions = 50_000
    scalars = np.logspace(-1.5, 1.5, num=10)

    circuits = [
        build_circuit(seed=1000 + idx, qubits=3, depth=10)
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
