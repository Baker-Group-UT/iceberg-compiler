"""
merging_improvement.py
----------------------

Random-circuit variant of the stylized merging improvement benchmark.

For each size bucket, generate random circuits with qubit and gate counts sampled
uniformly from the requested ranges. Each bucket is summarized by one stacked bar
showing the average additive benefit from:
1) H commutation
2) additional merging
3) additional packing

The packing row is planned by the `packing` package through `common.pack_program`;
`packer/merge_packer.py`, which used to do it, is archived. Both place two logical
qubits per patch, so the figure measures the same quantity as before.

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm import tqdm

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 18,
    }
)

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.figures.merging_improvement.common import pack_program
from experiments.shared.random_circuits import random_gate_circuit
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]

GATE_SET = ["h", "x", "z", "s", "rz", "rx", "cx", "cz"]
METRICS = {
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "#1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "#2q gates"),
    "circuit_depth": ("pct_reduction_circuit_depth", "depth"),
}
SIZE_GROUPS = [
    ("small", (5, 10), (10, 50)),
    ("medium", (15, 25), (30, 150)),
    ("large", (30, 100), (60, 600)),
]


@dataclass
class ExperimentSettings:
    num_circuits: int = 20
    random_seed: int | None = 10
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement.svg"
        )
    )
    show_plot: bool = True

    def __post_init__(self) -> None:
        if self.num_circuits <= 0:
            raise ValueError("num_circuits must be >= 1.")


def build_random_circuit(
    num_qubits: int,
    num_gates: int,
    seed: int | None,
) -> qiskit.QuantumCircuit:
    circuit = random_gate_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
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


def sample_bucket_params(
    qubit_range: Tuple[int, int],
    gate_range: Tuple[int, int],
    rng: random.Random,
) -> Tuple[int, int]:
    return (
        rng.randint(*qubit_range),
        rng.randint(*gate_range),
    )


def run_experiment(settings: ExperimentSettings) -> pd.DataFrame:
    seed_rng = random.Random(settings.random_seed)

    baseline_cfg = PipelineConfig()
    baseline_cfg.translator_config.enable_merging = False
    baseline_cfg.translator_config.enable_h_commutation = False

    h_only_cfg = PipelineConfig()
    h_only_cfg.translator_config.enable_merging = False
    h_only_cfg.translator_config.enable_h_commutation = True

    h_merge_cfg = PipelineConfig()
    h_merge_cfg.translator_config.enable_merging = True
    h_merge_cfg.translator_config.enable_h_commutation = True

    rows: List[Dict[str, int | float | str]] = []

    for size_group, qubit_range, gate_range in SIZE_GROUPS:
        for circuit_idx in tqdm(
            range(settings.num_circuits),
            desc=f"circuits ({size_group})",
        ):
            n_qubits, n_gates = sample_bucket_params(qubit_range, gate_range, seed_rng)
            circuit_seed = seed_rng.randint(0, 2**31 - 1)
            program_circuit = build_random_circuit(
                num_qubits=n_qubits,
                num_gates=n_gates,
                seed=circuit_seed,
            )

            target_codes = math.ceil(n_qubits / 2)
            tight_perm = tight_trivial_packing(n_qubits, target_codes)
            packed_perm, commute = pack_program(program_circuit)
            # Honour the planner's per-circuit commutation decision, and translate every
            # row under it. When it comes back False the H-commutation row collapses onto
            # the baseline and its segment is legitimately zero -- the shipped pipeline
            # declined the pass on that circuit -- so the ablation stays cumulative
            # instead of crediting a technique the compiler did not use.
            h_only_cfg.translator_config.enable_h_commutation = commute
            h_merge_cfg.translator_config.enable_h_commutation = commute

            base_one_q, base_two_q, base_depth = compute_metrics(
                program_circuit,
                tight_perm,
                baseline_cfg,
            )
            h_only_one_q, h_only_two_q, h_only_depth = compute_metrics(
                program_circuit,
                tight_perm,
                h_only_cfg,
            )
            h_merge_one_q, h_merge_two_q, h_merge_depth = compute_metrics(
                program_circuit,
                tight_perm,
                h_merge_cfg,
            )
            packed_one_q, packed_two_q, packed_depth = compute_metrics(
                program_circuit,
                packed_perm,
                h_merge_cfg,
            )

            h_cum_pct_one_q = pct_reduction(base_one_q, h_only_one_q)
            h_cum_pct_two_q = pct_reduction(base_two_q, h_only_two_q)
            h_cum_pct_depth = pct_reduction(base_depth, h_only_depth)

            h_merge_cum_pct_one_q = pct_reduction(base_one_q, h_merge_one_q)
            h_merge_cum_pct_two_q = pct_reduction(base_two_q, h_merge_two_q)
            h_merge_cum_pct_depth = pct_reduction(base_depth, h_merge_depth)

            packed_cum_pct_one_q = pct_reduction(base_one_q, packed_one_q)
            packed_cum_pct_two_q = pct_reduction(base_two_q, packed_two_q)
            packed_cum_pct_depth = pct_reduction(base_depth, packed_depth)

            rows.append(
                {
                    "size_group": size_group,
                    "circuit_idx": circuit_idx,
                    "num_qubits": n_qubits,
                    "num_gates": n_gates,
                    "target_codes": target_codes,
                    "component": "h_commutation",
                    "pct_reduction_one_qubit_gates": h_cum_pct_one_q,
                    "pct_reduction_two_qubit_gates": h_cum_pct_two_q,
                    "pct_reduction_circuit_depth": h_cum_pct_depth,
                }
            )
            rows.append(
                {
                    "size_group": size_group,
                    "circuit_idx": circuit_idx,
                    "num_qubits": n_qubits,
                    "num_gates": n_gates,
                    "target_codes": target_codes,
                    "component": "merging",
                    "pct_reduction_one_qubit_gates": (
                        h_merge_cum_pct_one_q - h_cum_pct_one_q
                    ),
                    "pct_reduction_two_qubit_gates": (
                        h_merge_cum_pct_two_q - h_cum_pct_two_q
                    ),
                    "pct_reduction_circuit_depth": (
                        h_merge_cum_pct_depth - h_cum_pct_depth
                    ),
                }
            )
            rows.append(
                {
                    "size_group": size_group,
                    "circuit_idx": circuit_idx,
                    "num_qubits": n_qubits,
                    "num_gates": n_gates,
                    "target_codes": target_codes,
                    "component": "packing",
                    "pct_reduction_one_qubit_gates": (
                        packed_cum_pct_one_q - h_merge_cum_pct_one_q
                    ),
                    "pct_reduction_two_qubit_gates": (
                        packed_cum_pct_two_q - h_merge_cum_pct_two_q
                    ),
                    "pct_reduction_circuit_depth": (
                        packed_cum_pct_depth - h_merge_cum_pct_depth
                    ),
                }
            )

    return pd.DataFrame(rows)


def print_group_averages(df: pd.DataFrame) -> None:
    print("[merging_improvement] Plot averages (matching clipped bar values)")
    for size_group, _, _ in SIZE_GROUPS:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        print(f"  {size_group}:")
        metric_parts = []
        for _, (metric_column, metric_label) in METRICS.items():
            pivot = (
                group_df.pivot_table(
                    index="size_group",
                    columns="component",
                    values=metric_column,
                    aggfunc="mean",
                )
                .reindex([size_group])
                .fillna(0.0)
            )
            total = 0.0
            for component in ["h_commutation", "merging", "packing"]:
                if component in pivot.columns:
                    total += float(
                        pivot[component].astype(float).clip(lower=0.0).iloc[0]
                    )
            metric_parts.append(f"{metric_label}={total:.2f}%")
        print("    plotted total: " + ", ".join(metric_parts))

    overall_parts = []
    for _, (metric_column, metric_label) in METRICS.items():
        pivot = (
            df.pivot_table(
                index="size_group",
                columns="component",
                values=metric_column,
                aggfunc="mean",
            )
            .reindex([group for group, _, _ in SIZE_GROUPS])
            .fillna(0.0)
        )
        total_values = np.zeros(len(SIZE_GROUPS), dtype=float)
        for component in ["h_commutation", "merging", "packing"]:
            if component in pivot.columns:
                total_values += pivot[component].astype(float).clip(lower=0.0).values
        overall_parts.append(f"{metric_label}={float(total_values.mean()):.2f}%")
    print("  plotted overall average: " + ", ".join(overall_parts))


def plot_results(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    components = [
        ("h_commutation", "H commutation benefit", "#44545e"),
        ("merging", "+ Merging benefit", "#cfc5c3"),
        ("packing", "+ Packing benefit", "#0c639b"),
    ]
    group_labels = {
        "small": "Small\n5-10q, 10-50g",
        "medium": "Medium\n15-25q, 30-150g",
        "large": "Large\n30-100q, 60-300g",
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 13), sharex=True)
    positions = np.arange(len(SIZE_GROUPS), dtype=float)
    bar_width = 0.72

    for ax, (_, (metric_column, metric_label)) in zip(axes, METRICS.items()):
        pivot = (
            df.pivot_table(
                index="size_group",
                columns="component",
                values=metric_column,
                aggfunc="mean",
            )
            .reindex([group for group, _, _ in SIZE_GROUPS])
            .fillna(0.0)
        )

        running_bottom = np.zeros(len(SIZE_GROUPS), dtype=float)
        for component, label, color in components:
            values = (
                pivot[component].astype(float).clip(lower=0.0).values
                if component in pivot.columns
                else np.zeros(len(SIZE_GROUPS), dtype=float)
            )
            ax.bar(
                positions,
                values,
                width=bar_width,
                bottom=running_bottom,
                color=color,
                edgecolor="white",
                alpha=0.85,
                label=label,
            )
            running_bottom += values

        ax.set_ylabel(metric_label)
        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(
        [group_labels[group] for group, _, _ in SIZE_GROUPS],
        rotation=0,
        ha="center",
    )
    axes[-1].set_xlabel("Random circuit size bucket")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.suptitle(
        "Benefits of H Commutation, Merging, and Packing vs. a Naive Tight Permutation"
    )
    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

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
        description=(
            "Random-circuit ablation of H commutation, merging, and packing."
        )
    )
    parser.add_argument(
        "--num-circuits",
        type=int,
        default=20,
        help="Number of random circuits to average within each size bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Base seed for size sampling and circuit generation (set negative for stochastic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/figures/merging_improvement/merging_improvement.svg"),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    return ExperimentSettings(
        num_circuits=args.num_circuits,
        random_seed=None if args.seed < 0 else args.seed,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    df = run_experiment(settings)
    print_group_averages(df)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
