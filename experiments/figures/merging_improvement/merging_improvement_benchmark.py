"""
merging_improvement_benchmark.py
--------------------------------

Benchmark variant of merging_improvement.

For each QASMbench proxy circuit, compare percent reductions (vs tight packing
with merging disabled) between:
1) tight naive packing with merging enabled
2) merge-maximizing packing with merging enabled

Grouped bars are shown per circuit at ceil(n/2) codes. Three stacked subplots
show percent reductions for 1q gates, 2q gates, and circuit depth.

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement_benchmark
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

from evaluator.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from packer.merge_packer import MergeMaximizingPacker
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation
from translator.passes import cancel_inverse_gates, commute_hadamards

Perm = List[Tuple[int, int]]

METRICS = {
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "Percent reduction in #1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "Percent reduction in #2q gates"),
    "circuit_depth": ("pct_reduction_circuit_depth", "Percent reduction in depth"),
}


@dataclass
class BenchmarkSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    max_proxy_gates: int | None = 2000
    max_proxy_qubits: int | None = 100
    proxy_timeout_seconds: float | None = 1.0
    drop_equal_packings: bool = False
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark.svg"
        )
    )
    show_plot: bool = True


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


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


def preprocess_for_packing(
    program_circuit: qiskit.QuantumCircuit, config: PipelineConfig
) -> qiskit.QuantumCircuit:
    processed = cancel_inverse_gates(program_circuit)
    if config.translator_config.enable_h_commutation:
        processed = commute_hadamards(processed)
        processed = cancel_inverse_gates(processed)
    return processed


def run_experiment(
    circuits: Sequence[Tuple[str, qiskit.QuantumCircuit]],
    size_group: str,
    settings: BenchmarkSettings,
) -> pd.DataFrame:
    packer = MergeMaximizingPacker()

    merging_enabled = PipelineConfig()
    merging_enabled.translator_config.enable_merging = True
    merging_enabled.translator_config.enable_h_commutation = True
    merging_disabled = PipelineConfig()
    merging_disabled.translator_config.enable_merging = False
    merging_disabled.translator_config.enable_h_commutation = True

    rows: List[Dict[str, int | float | str]] = []

    for circuit_idx, (circuit_name, program_circuit) in tqdm(
        list(enumerate(circuits)),
        desc=f"circuits ({size_group})",
    ):
        n_qubits = program_circuit.num_qubits
        target_codes = math.ceil(n_qubits / 2)
        tight_perm = tight_trivial_packing(n_qubits, target_codes)
        processed_for_packing = preprocess_for_packing(program_circuit, merging_enabled)
        packed_perm = packer.pack(processed_for_packing, num_codes=target_codes).permutation
        
        base_one_q, base_two_q, base_depth = compute_metrics(
            program_circuit, tight_perm, merging_disabled
        )
        tight_one_q, tight_two_q, tight_depth = compute_metrics(
            program_circuit, tight_perm, merging_enabled
        )
        packed_one_q, packed_two_q, packed_depth = compute_metrics(
            program_circuit, packed_perm, merging_enabled
        )

        tight_pct_one_q = pct_reduction(base_one_q, tight_one_q)
        tight_pct_two_q = pct_reduction(base_two_q, tight_two_q)
        tight_pct_depth = pct_reduction(base_depth, tight_depth)
        packed_pct_one_q = pct_reduction(base_one_q, packed_one_q)
        packed_pct_two_q = pct_reduction(base_two_q, packed_two_q)
        packed_pct_depth = pct_reduction(base_depth, packed_depth)

        if (
            tight_pct_one_q < 0
            or tight_pct_two_q < 0
            or tight_pct_depth < 0
            or packed_pct_one_q < 0
            or packed_pct_two_q < 0
            or packed_pct_depth < 0
        ):
            print(
                "[merging_improvement_benchmark] Negative reduction:",
                f"{size_group}/{circuit_name}",
                f"codes={target_codes}",
                f"tight_merge=(1q={tight_pct_one_q:.2f}%, 2q={tight_pct_two_q:.2f}%, depth={tight_pct_depth:.2f}%)",
                f"merge_max=(1q={packed_pct_one_q:.2f}%, 2q={packed_pct_two_q:.2f}%, depth={packed_pct_depth:.2f}%)",
            )

        if settings.drop_equal_packings:
            if (
                tight_one_q == packed_one_q
                and tight_two_q == packed_two_q
                and tight_depth == packed_depth
            ):
                continue

        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "method": "tight_merge",
                "pct_reduction_one_qubit_gates": tight_pct_one_q,
                "pct_reduction_two_qubit_gates": tight_pct_two_q,
                "pct_reduction_circuit_depth": tight_pct_depth,
            }
        )
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "method": "merge_maximizing",
                "pct_reduction_one_qubit_gates": packed_pct_one_q,
                "pct_reduction_two_qubit_gates": packed_pct_two_q,
                "pct_reduction_circuit_depth": packed_pct_depth,
            }
        )

    return pd.DataFrame(rows)


def print_metric_averages(df: pd.DataFrame) -> None:
    print("[merging_improvement_benchmark] Average percent reductions across loaded benchmark circuits")
    for method in ["tight_merge", "merge_maximizing"]:
        method_df = df.loc[df["method"] == method]
        if method_df.empty:
            continue
        metric_parts = []
        for _, (metric_column, metric_label) in METRICS.items():
            avg = method_df[metric_column].astype(float).mean()
            metric_parts.append(f"{metric_label}={avg:.2f}%")
        print(f"  {method}: " + ", ".join(metric_parts))
    aggregate_df = df.loc[df["method"] == "merge_maximizing"]
    if not aggregate_df.empty:
        aggregate_parts = []
        for _, (metric_column, metric_label) in METRICS.items():
            avg = aggregate_df[metric_column].astype(float).mean()
            aggregate_parts.append(f"{metric_label}={avg:.2f}%")
        print("  aggregate total: " + ", ".join(aggregate_parts))


def plot_results(df: pd.DataFrame, settings: BenchmarkSettings) -> None:
    methods = [
        ("tight_merge", "Tight (merge+H)", "#c7e9c0", "#238b45"),
        ("merge_maximizing", "Merge-maximizing+H", "#c6dbef", "#2171b5"),
    ]
    size_groups = ["small", "medium", "large"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 14), sharex=True)
    bar_width = 0.22
    gap = 4.0

    for ax, (_, (metric_column, metric_title)) in zip(axes, METRICS.items()):
        all_positions: List[float] = []
        all_circuit_labels: List[str] = []
        current_x = 0.0

        for size_group in size_groups:
            group_df = df.loc[df["size_group"] == size_group]
            if group_df.empty:
                continue

            pivot = (
                group_df.pivot_table(
                    index="circuit_name",
                    columns="method",
                    values=metric_column,
                    aggfunc="mean",
                )
                .reset_index()
            )
            sort_key = pivot["merge_maximizing"]
            pivot = pivot.assign(_sort=sort_key).sort_values("_sort")
            circuit_names = pivot["circuit_name"].tolist()
            positions = np.arange(len(circuit_names)) + current_x
            offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_width

            for (method, label, face, edge), offset in zip(methods, offsets):
                values = pivot[method].astype(float).values
                ax.bar(
                    positions + offset,
                    values,
                    width=bar_width,
                    color=face,
                    edgecolor=edge,
                    alpha=0.85,
                    label=label,
                )

            all_positions.extend(positions.tolist())
            all_circuit_labels.extend(circuit_names)
            current_x = positions[-1] + gap if len(positions) else current_x + gap

        ax.set_ylabel(metric_title)
        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_xticks(all_positions, minor=True)
        ax.set_xticklabels(all_circuit_labels, rotation=45, ha="right", minor=True)
        ax.tick_params(axis="x", which="major", bottom=False, labelbottom=False)
        ax.tick_params(axis="x", which="minor", pad=5)

    axes[-1].set_xlabel("Benchmark circuit (grouped by size)")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.suptitle(
        "Percent reduction vs tight packing without merging (per circuit at ceil(n/2) codes)"
    )
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[merging_improvement_benchmark] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description="Benchmark comparison of tight vs merge-maximizing packings."
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        default=Path(
            "benchmarks/qasmbench"
        ),
        help="Root directory containing QASMbench size folders (small/medium/large).",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=2000,
        help="Exclude proxy circuits with more than this many gates (post-transpile).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=100,
        help="Exclude proxy circuits with more than this many qubits (from README metadata).",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for proxy circuit creation (set <=0 to disable).",
    )
    parser.add_argument(
        "--drop-equal-packings",
        # default=True,
        action="store_true",
        help="Skip circuits where tight_merge equals merge_maximizing for all metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark.svg"
        ),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    return BenchmarkSettings(
        qasm_root=args.qasm_root,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        drop_equal_packings=args.drop_equal_packings,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def _load_group_circuits(
    qasm_root: Path,
    group: str,
    settings: BenchmarkSettings,
) -> Sequence[Tuple[str, qiskit.QuantumCircuit]]:
    timeout_seconds = settings.proxy_timeout_seconds
    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None
    circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=False,
    )
    if settings.max_proxy_gates is not None:
        circuits = [
            (name, qc)
            for name, qc in circuits
            if qc.size() <= settings.max_proxy_gates
        ]
    circuits = [
        (name, qc)
        for name, qc in circuits
        if not name.lower().endswith("_n2")
    ]
    return circuits


def main() -> None:
    settings = parse_args()
    all_rows: List[pd.DataFrame] = []
    for group in ["small", "medium"]:
        circuits = _load_group_circuits(settings.qasm_root, group, settings)
        if not circuits:
            print(f"[merging_improvement_benchmark] No circuits loaded for {group}.")
            continue
        group_df = run_experiment(circuits, group, settings)
        all_rows.append(group_df)

    if not all_rows:
        raise ValueError(f"No proxy circuits loaded from {settings.qasm_root}")
    df = pd.concat(all_rows, ignore_index=True)
    print_metric_averages(df)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
