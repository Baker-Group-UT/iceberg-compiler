"""
merging_improvement_benchmark_stylized.py
--------------------------------

Stylized benchmark variant of merging_improvement.

For each QASMbench proxy circuit, compare cumulative percent reductions
(vs naive tight packing with merging and H commutation disabled) at
ceil(n/2) codes.

Each circuit has one stacked bar broken into additive benefit components:
1) H commutation benefit
2) additional merging benefit
3) additional packing benefit

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement_benchmark_stylized
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
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "#1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "#2q gates"),
    "circuit_depth": ("pct_reduction_circuit_depth", "depth"),
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
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized.svg"
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

    for circuit_idx, (circuit_name, program_circuit) in tqdm(
        list(enumerate(circuits)),
        desc=f"circuits ({size_group})",
    ):
        n_qubits = program_circuit.num_qubits
        target_codes = math.ceil(n_qubits / 2)
        tight_perm = tight_trivial_packing(n_qubits, target_codes)
        processed_for_packing = preprocess_for_packing(program_circuit, h_merge_cfg)
        packed_perm = packer.pack(processed_for_packing, num_codes=target_codes).permutation
        
        base_one_q, base_two_q, base_depth = compute_metrics(
            program_circuit, tight_perm, baseline_cfg
        )
        h_only_one_q, h_only_two_q, h_only_depth = compute_metrics(
            program_circuit, tight_perm, h_only_cfg
        )
        h_merge_one_q, h_merge_two_q, h_merge_depth = compute_metrics(
            program_circuit, tight_perm, h_merge_cfg
        )
        packed_one_q, packed_two_q, packed_depth = compute_metrics(
            program_circuit, packed_perm, h_merge_cfg
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

        h_only_seg_one_q = h_cum_pct_one_q
        h_only_seg_two_q = h_cum_pct_two_q
        h_only_seg_depth = h_cum_pct_depth
        merge_seg_one_q = h_merge_cum_pct_one_q - h_cum_pct_one_q
        merge_seg_two_q = h_merge_cum_pct_two_q - h_cum_pct_two_q
        merge_seg_depth = h_merge_cum_pct_depth - h_cum_pct_depth
        pack_seg_one_q = packed_cum_pct_one_q - h_merge_cum_pct_one_q
        pack_seg_two_q = packed_cum_pct_two_q - h_merge_cum_pct_two_q
        pack_seg_depth = packed_cum_pct_depth - h_merge_cum_pct_depth

        if (
            h_only_seg_one_q < 0
            or h_only_seg_two_q < 0
            or h_only_seg_depth < 0
            or merge_seg_one_q < 0
            or merge_seg_two_q < 0
            or merge_seg_depth < 0
            or pack_seg_one_q < 0
            or pack_seg_two_q < 0
            or pack_seg_depth < 0
        ):
            print(
                "[merging_improvement_benchmark_stylized] Negative incremental benefit:",
                f"{size_group}/{circuit_name}",
                f"codes={target_codes}",
                f"h=(1q={h_only_seg_one_q:.2f}%, 2q={h_only_seg_two_q:.2f}%, depth={h_only_seg_depth:.2f}%)",
                f"merge=(1q={merge_seg_one_q:.2f}%, 2q={merge_seg_two_q:.2f}%, depth={merge_seg_depth:.2f}%)",
                f"pack=(1q={pack_seg_one_q:.2f}%, 2q={pack_seg_two_q:.2f}%, depth={pack_seg_depth:.2f}%)",
            )

        if settings.drop_equal_packings:
            if (
                h_only_one_q == packed_one_q
                and h_only_two_q == packed_two_q
                and h_only_depth == packed_depth
            ):
                continue

        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "component": "h_commutation",
                "pct_reduction_one_qubit_gates": h_only_seg_one_q,
                "pct_reduction_two_qubit_gates": h_only_seg_two_q,
                "pct_reduction_circuit_depth": h_only_seg_depth,
            }
        )
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "component": "merging",
                "pct_reduction_one_qubit_gates": merge_seg_one_q,
                "pct_reduction_two_qubit_gates": merge_seg_two_q,
                "pct_reduction_circuit_depth": merge_seg_depth,
            }
        )
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "component": "packing",
                "pct_reduction_one_qubit_gates": pack_seg_one_q,
                "pct_reduction_two_qubit_gates": pack_seg_two_q,
                "pct_reduction_circuit_depth": pack_seg_depth,
            }
        )

    return pd.DataFrame(rows)


def print_metric_averages(df: pd.DataFrame) -> None:
    print("[merging_improvement_benchmark_stylized] Plot averages (matching clipped bar values)")
    size_groups = ["small", "medium", "large"]
    for size_group in size_groups:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        print(f"  {size_group}:")
        metric_parts = []
        for _, (metric_column, metric_label) in METRICS.items():
            total = 0.0
            for component in ["h_commutation", "merging", "packing"]:
                component_df = group_df.loc[group_df["component"] == component, metric_column]
                if not component_df.empty:
                    total += float(
                        component_df.astype(float).clip(lower=0.0).mean()
                    )
            metric_parts.append(f"{metric_label}={total:.2f}%")
        print("    plotted total: " + ", ".join(metric_parts))

    overall_parts = []
    for _, (metric_column, metric_label) in METRICS.items():
        total = 0.0
        for component in ["h_commutation", "merging", "packing"]:
            component_df = df.loc[df["component"] == component, metric_column]
            if not component_df.empty:
                total += float(component_df.astype(float).clip(lower=0.0).mean())
        overall_parts.append(f"{metric_label}={total:.2f}%")
    print("  plotted overall average: " + ", ".join(overall_parts))


def plot_results(df: pd.DataFrame, settings: BenchmarkSettings) -> None:
    components = [
        ("h_commutation", "H commutation benefit", "#44545e"),
        ("merging", "+ Merging benefit", "#cfc5c3"),
        ("packing", "+ Packing benefit", "#0c639b"),
    ]
    size_groups = ["small", "medium", "large"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 14), sharex=True)
    bar_width = 0.75
    gap = 4.0
    depth_metric_column = METRICS["two_qubit_gates"][0]

    # Build one canonical ordering (ascending) from depth totals and reuse it
    # for every subplot so the shared x-axis labels remain correct.
    ordered_circuits_by_group: Dict[str, List[str]] = {}
    for size_group in size_groups:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        depth_pivot = (
            group_df.pivot_table(
                index="circuit_name",
                columns="component",
                values=depth_metric_column,
                aggfunc="mean",
            )
            .reset_index()
        )
        for component, _, _ in components:
            if component not in depth_pivot.columns:
                depth_pivot[component] = 0.0
            depth_pivot[component] = (
                depth_pivot[component].fillna(0.0).astype(float).clip(lower=0.0)
            )
        sort_key = depth_pivot[[c for c, _, _ in components]].sum(axis=1)
        depth_pivot = depth_pivot.assign(_sort=sort_key).sort_values(
            by=["_sort", "circuit_name"], ascending=[True, True]
        )
        ordered_circuits_by_group[size_group] = depth_pivot["circuit_name"].tolist()

    for ax, (_, (metric_column, metric_label)) in zip(axes, METRICS.items()):
        all_positions: List[float] = []
        all_circuit_labels: List[str] = []
        current_x = 0.0

        for size_group in size_groups:
            ordered_circuits = ordered_circuits_by_group.get(size_group, [])
            if not ordered_circuits:
                continue
            group_df = df.loc[df["size_group"] == size_group]

            pivot = (
                group_df.pivot_table(
                    index="circuit_name",
                    columns="component",
                    values=metric_column,
                    aggfunc="mean",
                )
                .reset_index()
            )
            for component, _, _ in components:
                if component not in pivot.columns:
                    pivot[component] = 0.0
                pivot[component] = (
                    pivot[component].fillna(0.0).astype(float).clip(lower=0.0)
                )
            pivot = (
                pivot.set_index("circuit_name")
                .reindex(ordered_circuits)
                .fillna(0.0)
                .reset_index()
            )
            circuit_names = ordered_circuits
            positions = np.arange(len(circuit_names)) + current_x

            running_bottom = np.zeros(len(circuit_names), dtype=float)
            for component, label, color in components:
                values = pivot[component].astype(float).values
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

            all_positions.extend(positions.tolist())
            all_circuit_labels.extend(circuit_names)
            current_x = positions[-1] + gap if len(positions) else current_x + gap

        # Append an overall average subgroup (single stacked bar).
        # Center the single average bar where the old 3-bar subgroup midpoint was.
        avg_x = current_x + 1.0
        avg_width = 5 * bar_width
        avg_bottom = 0.0
        for component, label, color in components:
            avg_value = (
                df.loc[df["component"] == component, metric_column]
                .astype(float)
                .clip(lower=0.0)
                .mean()
            )
            if np.isnan(avg_value):
                avg_value = 0.0
            ax.bar(
                avg_x,
                avg_value,
                width=avg_width,
                bottom=avg_bottom,
                color=color,
                edgecolor="white",
                alpha=0.95,
                label=label,
            )
            avg_bottom += avg_value
        ax.axvline(current_x - gap / 2.0, color="#bdbdbd", linestyle=":", linewidth=1.0)
        left_gap_to_avg = (avg_x - avg_width / 2.0) - (current_x - gap / 2.0)
        right_limit = avg_x + avg_width / 2.0 + max(0.0, left_gap_to_avg)
        left_limit, _ = ax.get_xlim()
        ax.set_xlim(left=left_limit, right=right_limit)
        all_positions.append(avg_x)
        all_circuit_labels.append("average")

        ax.set_ylabel(metric_label)
        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_xticks(all_positions, minor=True)
        ax.set_xticklabels(all_circuit_labels, rotation=45, ha="right", minor=True)
        ticklabels = ax.get_xticklabels(minor=True)
        if ticklabels:
            ticklabels[-1].set_rotation(0)
            ticklabels[-1].set_ha("center")
        ax.tick_params(axis="x", which="major", bottom=False, labelbottom=False)
        ax.tick_params(axis="x", which="minor", pad=5)

    axes[-1].set_xlabel("Benchmark circuit (grouped by size)")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.suptitle(
        "Benefits of H Commutation, Merging, and Packing vs. a Naive Permutation Without Techniques Applied"
    )
    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[merging_improvement_benchmark_stylized] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description="Stylized benchmark with stacked benefits from H commutation, merging, and packing."
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
        default=2500,
        help="Exclude proxy circuits with more than this many gates (post-transpile).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=150,
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
        help="Skip circuits where H-only and H+merge+packing are equal for all metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized.svg"
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
    for group in ["small", "medium", "large"]:
        circuits = _load_group_circuits(settings.qasm_root, group, settings)
        if not circuits:
            print(f"[merging_improvement_benchmark_stylized] No circuits loaded for {group}.")
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
