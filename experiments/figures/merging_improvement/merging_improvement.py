"""
merging_improvement.py
----------------------

Compare percent reductions at n/2 codes for three pack-and-merge strategies,
using tight packing with merging disabled as the baseline:
1) tight naive packing with merging enabled
2) merge-maximizing packing with merging enabled

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm import tqdm

from evaluator.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.random_circuits import random_gate_circuit
from packer.merge_packer import MergeMaximizingPacker
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]

GATE_SET = ["x", "z", "h", "cx"]


@dataclass
class ExperimentSettings:
    n_qubits: int = 20
    n_gates: int = 200
    num_circuits: int = 10
    random_seed: int | None = 7
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_boxplot.png"
        )
    )
    show_plot: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2.")
        if self.n_gates < 1:
            raise ValueError("n_gates must be >= 1.")
        if self.num_circuits <= 0:
            raise ValueError("num_circuits must be >= 1.")


def build_random_circuit(
    n_qubits: int,
    n_gates: int,
    seed: int | None,
) -> qiskit.QuantumCircuit:
    circuit = random_gate_circuit(
        num_qubits=n_qubits,
        num_gates=n_gates,
        gates=GATE_SET,
        seed=seed,
    )
    circuit.measure_all()
    return circuit


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    patch_capacity = max(1, math.ceil(num_qubits / n_codes))
    perm: Perm = []
    for i in range(num_qubits):
        code = min(i // patch_capacity, n_codes - 1)
        perm.append((code, i % patch_capacity))
    return perm


def compute_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Tuple[int, int, int]:
    translated = translate_permutation(program_circuit, permutation, config=config)
    circuit = translated.circuit
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    return one_q, two_q, depth


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def run_experiment(settings: ExperimentSettings) -> pd.DataFrame:
    packer = MergeMaximizingPacker(solo_strategy="most_gates")

    merging_enabled = PipelineConfig()
    merging_enabled.translator_config.enable_merging = True
    merging_disabled = PipelineConfig()
    merging_disabled.translator_config.enable_merging = False

    rows: List[Dict[str, int | float | str]] = []
    target_codes = math.ceil(settings.n_qubits / 2)

    for circuit_idx in tqdm(range(settings.num_circuits), desc="circuits"):
        circuit_seed = None
        if settings.random_seed is not None:
            circuit_seed = settings.random_seed * 10_000 + circuit_idx
        program_circuit = build_random_circuit(
            n_qubits=settings.n_qubits,
            n_gates=settings.n_gates,
            seed=circuit_seed,
        )
        tight_perm = tight_trivial_packing(settings.n_qubits, target_codes)
        packed_perm = packer.pack(program_circuit, num_codes=target_codes).permutation

        baseline_one_q, baseline_two_q, baseline_depth = compute_metrics(
            program_circuit, tight_perm, merging_disabled
        )
        tight_merge_one_q, tight_merge_two_q, tight_merge_depth = compute_metrics(
            program_circuit, tight_perm, merging_enabled
        )
        packed_one_q, packed_two_q, packed_depth = compute_metrics(
            program_circuit, packed_perm, merging_enabled
        )

        rows.append(
            {
                "circuit_idx": circuit_idx,
                "target_codes": target_codes,
                "codes_used": len({p for p, _ in tight_perm}),
                "method": "tight_merge",
                "pct_reduction_one_qubit_gates": pct_reduction(
                    baseline_one_q, tight_merge_one_q
                ),
                "pct_reduction_two_qubit_gates": pct_reduction(
                    baseline_two_q, tight_merge_two_q
                ),
                "pct_reduction_circuit_depth": pct_reduction(
                    baseline_depth, tight_merge_depth
                ),
            }
        )
        rows.append(
            {
                "circuit_idx": circuit_idx,
                "target_codes": target_codes,
                "codes_used": len({p for p, _ in packed_perm}),
                "method": "merge_maximizing",
                "pct_reduction_one_qubit_gates": pct_reduction(
                    baseline_one_q, packed_one_q
                ),
                "pct_reduction_two_qubit_gates": pct_reduction(
                    baseline_two_q, packed_two_q
                ),
                "pct_reduction_circuit_depth": pct_reduction(
                    baseline_depth, packed_depth
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    metrics = [
        ("pct_reduction_one_qubit_gates", "Percent reduction in #1q gates"),
        ("pct_reduction_two_qubit_gates", "Percent reduction in #2q gates"),
        ("pct_reduction_circuit_depth", "Percent reduction in depth"),
    ]
    methods = [
        ("tight_merge", "Tight (merge)", "#c7e9c0", "#238b45"),
        ("merge_maximizing", "Merge-maximizing", "#c6dbef", "#2171b5"),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    for ax, (metric, title) in zip(axes, metrics):
        means = []
        errors = []
        for method, _, _, _ in methods:
            values = df.loc[df["method"] == method, metric].values
            means.append(float(np.mean(values)) if len(values) else 0.0)
            errors.append(float(np.std(values)) if len(values) else 0.0)

        x = np.arange(len(methods))
        bars = ax.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=[face for _, _, face, _ in methods],
            edgecolor=[edge for _, _, _, edge in methods],
            alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label, _, _ in methods])
        ax.set_xlabel(f"Codes used = {math.ceil(settings.n_qubits / 2)}")
        ax.set_ylabel(title)
        ax.set_title(title)

        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)

        for bar, edge in zip(bars, [edge for _, _, _, edge in methods]):
            bar.set_linewidth(1.0)
            bar.set_edgecolor(edge)

    fig.suptitle("Percent reduction vs tight packing with merging disabled")
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[merging_improvement] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> ExperimentSettings:
    parser = argparse.ArgumentParser(
        description="Compare tight vs merge-maximizing packings with/without merging."
    )
    parser.add_argument("--n-qubits", type=int, default=50, help="Number of logical qubits.")
    parser.add_argument("--n-gates", type=int, default=500, help="Number of random gates.")
    parser.add_argument("--num-circuits", type=int, default=20, help="Number of random circuits.")
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Base seed for circuit randomness (set negative for stochastic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/figures/merging_improvement/merging_improvement_boxplot.svg"
        ),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    seed_value = args.seed if args.seed >= 0 else None
    return ExperimentSettings(
        n_qubits=args.n_qubits,
        n_gates=args.n_gates,
        num_circuits=args.num_circuits,
        random_seed=seed_value,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    df = run_experiment(settings)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
