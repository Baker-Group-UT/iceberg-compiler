"""
Compare logical performance between Qiskit Aer and Stim backends across
different noise scaling factors.

This mirrors experiments/noise_scaling/p_scaling.py but runs the same logical
permutation scan twice per noise scale: once with the standard AerSimulator
backend and once with the Stim backend. It then plots the resulting logical
fidelities and selection rates to highlight any mismatch between the two
simulation pipelines.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit

from pipeline import run_specific_scan
from pipeline.config import PipelineConfig
from simulator.config_presets import StandardSimulator
from simulator.noise_models import ScaledNoiseModel

# Representative permutations (same as p_scaling experiment)
PERM_GROUP = [(0, 0), (0, 1), (1, 0), (1, 1)]
PERM_DIFF = [(0, 0), (1, 0), (2, 0), (3, 0)]
PERMUTATIONS: List[List[Tuple[int, int]]] = [PERM_GROUP, PERM_DIFF]

TARGET_BASIS = ["x", "z", "h", "rz", "cx", "cz", "measure"]


def build_circuit(seed: int = 17, qubits: int = 4, depth: int = 25) -> QuantumCircuit:
    """Generate and transpile a random Clifford circuit for the experiment."""
    program_circuit = random_clifford_circuit(
        qubits,
        depth,
        seed=seed,
        gates=["x", "h", "z", "s", "cx", "cz"],
    )
    program_circuit.measure_all()
    # return qiskit.transpile(
    #     program_circuit,
    #     basis_gates=TARGET_BASIS,
    #     optimization_level=0,
    # )
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


def compute_fidelity_mismatch(
    circuits: List[QuantumCircuit],
    scalars: np.ndarray,
    *,
    repetitions: int,
) -> np.ndarray:
    """
    For each circuit and noise scaling factor, return the Stim-vs-Aer fidelity mismatch.
    Returns an array of shape (num_circuits, len(scalars)).
    """
    mismatches = np.zeros((len(circuits), len(scalars)), dtype=float)

    for ci, tcirc in enumerate(circuits):
        for si, scalar in enumerate(scalars):
            noise_model = ScaledNoiseModel(alpha=scalar)

            aer_sim = (
                StandardSimulator(repetitions=repetitions)
                .with_noise_model(noise_model)
                .with_stim(False)
            )
            stim_sim = (
                StandardSimulator(repetitions=repetitions)
                .with_noise_model(noise_model)
                .with_stim(True)
                .with_stim_scaling(None)
            )

            aer_config = PipelineConfig(sim_config=aer_sim, verbose=False, save=False)
            stim_config = PipelineConfig(sim_config=stim_sim, verbose=False, save=False)

            aer_record = run_specific_scan(
                program_circuit=tcirc,
                permutations=PERMUTATIONS,
                config=aer_config,
            )
            stim_record = run_specific_scan(
                program_circuit=tcirc,
                permutations=PERMUTATIONS,
                config=stim_config,
            )

            aer_fidelity = 1 - aer_record.dataframe.df.loc[
                0, "logical_total_variation_distance"
            ]
            stim_fidelity = 1 - stim_record.dataframe.df.loc[
                0, "logical_total_variation_distance"
            ]
            denom = aer_fidelity if aer_fidelity != 0 else 1.0
            mismatch_pct = 100.0 * abs(aer_fidelity - stim_fidelity) / denom
            mismatches[ci, si] = mismatch_pct
            print(
                f"[circuit {ci + 1}/{len(circuits)}] "
                f"alpha={scalar:.3e} | Aer={aer_fidelity:.4f}, "
                f"Stim={stim_fidelity:.4f}, Δ={mismatch_pct:+.4f}%"
            )

    return mismatches



def plot_mismatch(
    scalars: np.ndarray,
    mismatches: np.ndarray,
    output_path: str,
) -> None:
    """
    Plot the mean/σ/min/max fidelity mismatch as a function of noise scale.
    """
    mean = np.mean(mismatches, axis=0)
    std = np.std(mismatches, axis=0)
    min_vals = np.min(mismatches, axis=0)
    max_vals = np.max(mismatches, axis=0)

    # Clip values to avoid issues when switching to log-log plotting
    positive_entries = mismatches[mismatches > 0]
    eps = float(np.min(positive_entries)) * 0.5 if positive_entries.size else 1e-6
    eps = max(eps, 1e-6)

    def _clip(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, eps, None)

    mean_clip = _clip(mean)
    std_lower = _clip(mean - std)
    std_upper = _clip(mean + std)
    min_clip = _clip(min_vals)
    max_clip = _clip(max_vals)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(scalars, mean_clip, color="#1454d0", linewidth=3, label="Mean mismatch")
    ax.fill_between(
        scalars,
        std_lower,
        std_upper,
        color="#1454d0",
        alpha=0.35,
        label="±1σ",
    )
    ax.fill_between(
        scalars,
        min_clip,
        max_clip,
        color="#1454d0",
        alpha=0.25,
        label="Min–max",
    )
    ax.set_xlabel("Noise scale (α)")
    ax.set_ylabel("Fidelity mismatch % (Aer − Stim)")
    ax.set_title("Stim vs Aer fidelity mismatch across random circuits")
    ax.legend(loc="upper left", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    plt.tight_layout()
    plt.savefig(output_path, format="svg", facecolor=fig.get_facecolor())
    print(f"Saved mismatch plot to {output_path}")


def main() -> None:
    num_circuits = 20
    repetitions = 5_000
    scalars = np.logspace(-2.5, 1.5, num=20)

    circuits = [
        build_circuit(seed=1000 + idx, qubits=4, depth=25)
        for idx in range(num_circuits)
    ]

    mismatches = compute_fidelity_mismatch(
        circuits,
        scalars,
        repetitions=repetitions,
    )

    plot_mismatch(
        scalars,
        mismatches,
        output_path="results/stim_vs_aer_fidelity_mismatch.svg",
    )


if __name__ == "__main__":
    main()
