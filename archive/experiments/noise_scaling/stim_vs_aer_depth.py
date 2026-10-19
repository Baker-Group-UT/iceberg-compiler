"""
Compare Stim vs Aer fidelity mismatch as circuit depth scales.

This experiment keeps qubit count and noise level fixed, generates random
Clifford programs at multiple depths, and reports the percent mismatch between
Stim and Qiskit Aer fidelities for each depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

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

PERM_GROUP = [(0, 0), (0, 1), (1, 0), (1, 1)]
PERM_DIFF = [(0, 0), (1, 0), (2, 0), (3, 0)]
PERMUTATIONS: List[List[tuple[int, int]]] = [PERM_GROUP, PERM_DIFF]


def build_circuit(qubits: int, depth: int, seed: int) -> QuantumCircuit:
    circuit = random_clifford_circuit(
        num_qubits=qubits,
        num_gates=depth,
        seed=seed,
        gates=["x", "h", "z", "s", "cx", "cz"],
    )
    circuit.measure_all()
    return circuit
    # qiskit.transpile(
    #     circuit,
    #     basis_gates=["x", "z", "h", "rz", "cx", "cz", "measure"],
    #     optimization_level=0,
    # )


def backend_mismatch(
    tcirc: QuantumCircuit,
    *,
    alpha: float,
    repetitions: int,
) -> float:
    noise_model = ScaledNoiseModel(alpha=alpha)

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
    return mismatch_pct


def plot_mismatch(
    depths: np.ndarray,
    mismatches: np.ndarray,
    output_path: str,
) -> None:
    valid_mask = np.any(mismatches > 0, axis=0)
    valid_depths = depths[valid_mask]
    filtered = mismatches[:, valid_mask]
    if filtered.size == 0:
        print("No positive mismatch values to plot.")
        return

    mean = np.mean(filtered, axis=0)
    std = np.std(filtered, axis=0)
    min_vals = np.min(filtered, axis=0)
    max_vals = np.max(filtered, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")

    # ax.set_yscale("log")

    ax.plot(valid_depths, mean, color="#1454d0", linewidth=3, label="Mean mismatch")
    # ax.fill_between(
    #     valid_depths,
    #     np.clip(mean - std, 1e-12, None),
    #     np.clip(mean + std, 1e-12, None),
    #     color="#1454d0",
    #     alpha=0.35,
    #     label="±1σ",
    # )
    ax.fill_between(
        valid_depths,
        np.clip(min_vals, 1e-12, None),
        np.clip(max_vals, 1e-12, None),
        color="#1454d0",
        alpha=0.35,
        label="Min–max",
    )
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Fidelity mismatch % (Aer − Stim)")
    ax.set_title("Stim vs Aer mismatch vs circuit depth")
    ax.legend(loc="upper left", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg", facecolor=fig.get_facecolor())
    print(f"Saved mismatch plot to {output_path}")


def main() -> None:
    num_circuits = 3
    repetitions = 5_000
    qubits = 4
    alpha = 1.0
    depths = np.array([5, 10, 15, 20, 30, 45, 60, 80])

    mismatches = np.zeros((num_circuits, len(depths)), dtype=float)

    for di, depth in enumerate(depths):
        for ci in range(num_circuits):
            seed = 2000 + depth * 31 + ci
            circuit = build_circuit(qubits=qubits, depth=depth, seed=seed)
            mismatch = backend_mismatch(circuit, alpha=alpha, repetitions=repetitions)
            mismatches[ci, di] = mismatch
            print(
                f"[depth {depth:>3}] circuit {ci + 1}/{num_circuits} "
                f"mismatch={mismatch:.3f}%"
            )

    plot_mismatch(
        depths=depths,
        mismatches=mismatches,
        output_path="results/stim_vs_aer_depth_mismatch.svg",
    )


if __name__ == "__main__":
    main()
