"""
density_ablation_sweep.py
--------------------------

Sweeps code density (fraction of codes used, from 0.5q to 1q) across all
QASMBench circuits (small / medium / large) and measures how the 4-variant
ablation (baseline → +H commutation → +merging → +packing) changes with density.

For each (circuit, n_codes) pair the 4 variants are:
  1. baseline      – no H commutation, no merging, trivial packing
  2. h_only        – H commutation, no merging, trivial packing
  3. h_merge       – H commutation, merging, trivial packing
  4. h_merge_pack  – H commutation, merging, MergeMaximizingPacker

Results are stored incrementally (small → medium → large) so a mid-run crash
does not lose prior work.  Use ``--from-csv`` to replot without resimulating.

Outputs:
  * density_ablation_sweep.csv   (raw per-component rows)
  * density_ablation_sweep.svg   (5-row grouped stacked bar figure)

Run with:

    python -m experiments.figures.density_ablation_sweep.density_ablation_sweep
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from evaluator.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from packer.merge_packer import MergeMaximizingPacker
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.config_presets import ScaledSimulator
from translator.passes import cancel_inverse_gates, commute_hadamards

Perm = List[Tuple[int, int]]

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "figure.titlesize": 16,
    }
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS = {
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "#1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "#2q gates"),
    "circuit_depth":   ("pct_reduction_circuit_depth", "depth"),
    "lsr":             ("pct_improvement_logical_selection_rate", "LSR"),
    "tvd":             ("pct_reduction_logical_total_variation_distance", "TVD"),
}

COMPONENTS = [
    ("h_commutation", "H commutation", "#3b4c56"),
    ("merging",       "+ Merging",     "#cdc2c0"),
    ("packing",       "+ Packing",     "#0097fd"),
]

SIZE_GROUPS = ["small", "medium", "large"]
SIZE_GROUP_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class DensityAblationSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    max_proxy_gates: int | None = 2500
    max_proxy_qubits: int | None = 150
    proxy_timeout_seconds: float | None = 1.0
    n_codes_samples: int = 6
    n_plot_bins: int = 6
    alpha_reference: float = 0.1
    reference_gate_count: float = 200.0
    target_selection_rate: float = 0.12
    pilot_repetitions: int = 4_000
    target_kept_shots: int = 20_000
    min_repetitions: int = 20_000
    max_repetitions: int = 1_000_000
    stim: bool = True
    distance_noise: bool = True
    distance_scalar: float = 1.0
    idle_noise: bool = True
    idle_scalar: float = 1.0
    seed: int = 10
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/density_ablation_sweep/density_ablation_sweep.svg"
        )
    )
    csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/density_ablation_sweep/density_ablation_sweep.csv"
        )
    )
    from_csv: Path | None = None
    show_plot: bool = False


DEFAULT_SETTINGS = DensityAblationSettings()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def pct_improvement(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (challenger - reference) / reference


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    """Produce a permutation with exactly n_codes patches.

    Pairs the first (num_qubits - n_codes) consecutive qubit pairs together,
    then assigns remaining qubits to solo patches.
    """
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    min_codes = math.ceil(num_qubits / 2)
    if n_codes < min_codes or n_codes > num_qubits:
        raise ValueError(
            f"n_codes={n_codes} out of range [{min_codes}, {num_qubits}] "
            f"for {num_qubits} qubits."
        )
    pairs_needed = num_qubits - n_codes
    perm: Perm = [None] * num_qubits  # type: ignore[list-item]
    patch = 0
    qubit = 0
    for _ in range(pairs_needed):
        perm[qubit]     = (patch, 0)
        perm[qubit + 1] = (patch, 1)
        qubit += 2
        patch += 1
    while qubit < num_qubits:
        perm[qubit] = (patch, 0)
        qubit += 1
        patch += 1
    return perm


def ensure_measurements(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    if circuit.num_clbits > 0 and any(op.operation.name == "measure" for op in circuit.data):
        return circuit
    measured = copy.deepcopy(circuit)
    measured.measure_all()
    return measured


def preprocess_for_packing(
    program_circuit: qiskit.QuantumCircuit,
    config: PipelineConfig,
) -> qiskit.QuantumCircuit:
    processed = cancel_inverse_gates(program_circuit)
    if config.translator_config.enable_h_commutation:
        processed = commute_hadamards(processed)
        processed = cancel_inverse_gates(processed)
    return processed


# ---------------------------------------------------------------------------
# Pipeline config builders
# ---------------------------------------------------------------------------

def build_pipeline_config(
    settings: DensityAblationSettings,
    *,
    alpha: float,
    repetitions: int,
    seed: int,
    enable_h_commutation: bool,
    enable_merging: bool,
) -> PipelineConfig:
    sim_cfg = ScaledSimulator(alpha=alpha, repetitions=repetitions, seed=seed)
    if settings.stim:
        sim_cfg = sim_cfg.with_stim()
    if settings.distance_noise:
        sim_cfg = sim_cfg.with_distance_noise(distance_scalar=settings.distance_scalar)
    if settings.idle_noise:
        sim_cfg = sim_cfg.with_idle_noise(idle_time_scalar=settings.idle_scalar)
    config = PipelineConfig(verbose=False, save=False, sim_config=sim_cfg, seed=seed)
    config.translator_config.enable_h_commutation = enable_h_commutation
    config.translator_config.enable_merging = enable_merging
    return config


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_translation_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Tuple[int, int, int]:
    translated = translate_permutation(program_circuit, permutation, config=config)
    circuit = translated.circuit
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    return one_q, two_q, depth


def evaluate_variant(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Dict[str, float]:
    record = run_specific_scan(
        program_circuit=program_circuit,
        permutations=[permutation],
        config=config,
    )
    if record.dataframe is None or record.dataframe.df.empty:
        raise RuntimeError("run_specific_scan returned no dataframe.")
    row = record.dataframe.df.iloc[0]
    return {
        "logical_selection_rate": float(row["logical_selection_rate"]),
        "logical_total_variation_distance": float(row["logical_total_variation_distance"]),
    }


# ---------------------------------------------------------------------------
# Alpha calibration helpers
# ---------------------------------------------------------------------------

def calibrated_alpha_from_gate_count(gate_count: int, settings: DensityAblationSettings) -> float:
    gate_count = max(1, int(gate_count))
    return settings.alpha_reference * (settings.reference_gate_count / float(gate_count))


def corrected_alpha(
    alpha: float,
    observed_selection_rate: float,
    target_selection_rate: float,
) -> float:
    eps = 1e-6
    observed = float(np.clip(observed_selection_rate, eps, 1.0 - eps))
    target = float(np.clip(target_selection_rate, eps, 1.0 - eps))
    return alpha * math.log(target) / math.log(observed)


def repetitions_from_selection_rate(
    selection_rate: float,
    settings: DensityAblationSettings,
) -> int:
    denom = max(selection_rate, 1e-4)
    reps = int(math.ceil(settings.target_kept_shots / denom))
    return int(np.clip(reps, settings.min_repetitions, settings.max_repetitions))


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

def _load_group_circuits(
    qasm_root: Path,
    group: str,
    settings: DensityAblationSettings,
) -> Sequence[Tuple[str, qiskit.QuantumCircuit, qiskit.QuantumCircuit]]:
    timeout_seconds = settings.proxy_timeout_seconds
    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None
    resource_circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2_000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=False,
    )
    simulation_circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2_000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=True,
    )

    resource_map = {
        name: qc
        for name, qc in resource_circuits
        if settings.max_proxy_gates is None or qc.size() <= settings.max_proxy_gates
    }
    simulation_map = {
        name: qc
        for name, qc in simulation_circuits
        if settings.max_proxy_gates is None or qc.size() <= settings.max_proxy_gates
    }

    paired_names = sorted(set(resource_map) & set(simulation_map))
    return [
        (name, resource_map[name], simulation_map[name])
        for name in paired_names
        if not name.lower().endswith("_n2")
    ]


# ---------------------------------------------------------------------------
# Density bins
# ---------------------------------------------------------------------------

def _density_bins(n_qubits: int, n_codes_samples: int) -> List[int]:
    """Return deduplicated n_codes values evenly spaced from min to max."""
    min_codes = math.ceil(n_qubits / 2)
    max_codes = n_qubits
    raw = np.linspace(min_codes, max_codes, n_codes_samples)
    return sorted(set(int(round(v)) for v in raw))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    circuits: Sequence[Tuple[str, qiskit.QuantumCircuit, qiskit.QuantumCircuit]],
    size_group: str,
    settings: DensityAblationSettings,
) -> pd.DataFrame:
    packer = MergeMaximizingPacker()

    baseline_translate_cfg = PipelineConfig()
    baseline_translate_cfg.translator_config.enable_merging = False
    baseline_translate_cfg.translator_config.enable_h_commutation = False

    h_only_translate_cfg = PipelineConfig()
    h_only_translate_cfg.translator_config.enable_merging = False
    h_only_translate_cfg.translator_config.enable_h_commutation = True

    h_merge_translate_cfg = PipelineConfig()
    h_merge_translate_cfg.translator_config.enable_merging = True
    h_merge_translate_cfg.translator_config.enable_h_commutation = True

    rows: List[Dict] = []

    for circuit_idx, (circuit_name, resource_circuit, simulation_circuit) in tqdm(
        list(enumerate(circuits)),
        desc=f"circuits ({size_group})",
    ):
        resource_program = ensure_measurements(resource_circuit)
        simulation_program = ensure_measurements(simulation_circuit)
        n_qubits = resource_program.num_qubits
        min_codes = math.ceil(n_qubits / 2)
        code_counts = _density_bins(n_qubits, settings.n_codes_samples)

        # Preprocess once for packing
        preprocessed = preprocess_for_packing(resource_program, h_merge_translate_cfg)

        # Alpha calibration: baseline at min_codes
        tight_min = tight_trivial_packing(n_qubits, min_codes)
        base_1q, base_2q, _ = compute_translation_metrics(
            resource_program, tight_min, baseline_translate_cfg
        )
        baseline_gate_count = max(1, base_1q + base_2q)
        alpha_initial = calibrated_alpha_from_gate_count(baseline_gate_count, settings)

        pilot1_cfg = build_pipeline_config(
            settings,
            alpha=alpha_initial,
            repetitions=settings.pilot_repetitions,
            seed=settings.seed * 1_000_003 + circuit_idx * 17 + 1,
            enable_h_commutation=False,
            enable_merging=False,
        )
        pilot1 = evaluate_variant(simulation_program, tight_min, pilot1_cfg)

        alpha_final = corrected_alpha(
            alpha_initial,
            pilot1["logical_selection_rate"],
            settings.target_selection_rate,
        )

        pilot2_cfg = build_pipeline_config(
            settings,
            alpha=alpha_final,
            repetitions=settings.pilot_repetitions,
            seed=settings.seed * 1_000_003 + circuit_idx * 17 + 2,
            enable_h_commutation=False,
            enable_merging=False,
        )
        pilot2 = evaluate_variant(simulation_program, tight_min, pilot2_cfg)
        final_repetitions = repetitions_from_selection_rate(
            pilot2["logical_selection_rate"], settings
        )

        for density_bin_idx, n_codes in enumerate(
            tqdm(code_counts, desc=f"  {circuit_name} density", leave=False)
        ):
            fraction = n_codes / n_qubits

            trivial_perm = tight_trivial_packing(n_qubits, n_codes)
            packed_perm = packer.pack(preprocessed, num_codes=n_codes).permutation

            # Translation metrics
            b_1q, b_2q, b_depth = compute_translation_metrics(
                resource_program, trivial_perm, baseline_translate_cfg
            )
            h_1q, h_2q, h_depth = compute_translation_metrics(
                resource_program, trivial_perm, h_only_translate_cfg
            )
            hm_1q, hm_2q, hm_depth = compute_translation_metrics(
                resource_program, trivial_perm, h_merge_translate_cfg
            )
            pk_1q, pk_2q, pk_depth = compute_translation_metrics(
                resource_program, packed_perm, h_merge_translate_cfg
            )

            # Simulation metrics
            seed_off = circuit_idx * 100_000 + density_bin_idx * 10
            base_sim_cfg = build_pipeline_config(
                settings,
                alpha=alpha_final,
                repetitions=final_repetitions,
                seed=settings.seed * 1_000_003 + seed_off + 3,
                enable_h_commutation=False,
                enable_merging=False,
            )
            h_sim_cfg = build_pipeline_config(
                settings,
                alpha=alpha_final,
                repetitions=final_repetitions,
                seed=settings.seed * 1_000_003 + seed_off + 4,
                enable_h_commutation=True,
                enable_merging=False,
            )
            hm_sim_cfg = build_pipeline_config(
                settings,
                alpha=alpha_final,
                repetitions=final_repetitions,
                seed=settings.seed * 1_000_003 + seed_off + 5,
                enable_h_commutation=True,
                enable_merging=True,
            )
            pk_sim_cfg = build_pipeline_config(
                settings,
                alpha=alpha_final,
                repetitions=final_repetitions,
                seed=settings.seed * 1_000_003 + seed_off + 6,
                enable_h_commutation=True,
                enable_merging=True,
            )

            base_sim = evaluate_variant(simulation_program, trivial_perm, base_sim_cfg)
            h_sim    = evaluate_variant(simulation_program, trivial_perm, h_sim_cfg)
            hm_sim   = evaluate_variant(simulation_program, trivial_perm, hm_sim_cfg)
            pk_sim   = evaluate_variant(simulation_program, packed_perm,  pk_sim_cfg)

            # Cumulative from baseline
            h_cum_1q   = pct_reduction(b_1q,   h_1q)
            hm_cum_1q  = pct_reduction(b_1q,   hm_1q)
            pk_cum_1q  = pct_reduction(b_1q,   pk_1q)

            h_cum_2q   = pct_reduction(b_2q,   h_2q)
            hm_cum_2q  = pct_reduction(b_2q,   hm_2q)
            pk_cum_2q  = pct_reduction(b_2q,   pk_2q)

            h_cum_dep  = pct_reduction(b_depth, h_depth)
            hm_cum_dep = pct_reduction(b_depth, hm_depth)
            pk_cum_dep = pct_reduction(b_depth, pk_depth)

            h_cum_lsr  = pct_improvement(base_sim["logical_selection_rate"],
                                         h_sim["logical_selection_rate"])
            hm_cum_lsr = pct_improvement(base_sim["logical_selection_rate"],
                                         hm_sim["logical_selection_rate"])
            pk_cum_lsr = pct_improvement(base_sim["logical_selection_rate"],
                                         pk_sim["logical_selection_rate"])

            h_cum_tvd  = pct_reduction(base_sim["logical_total_variation_distance"],
                                       h_sim["logical_total_variation_distance"])
            hm_cum_tvd = pct_reduction(base_sim["logical_total_variation_distance"],
                                       hm_sim["logical_total_variation_distance"])
            pk_cum_tvd = pct_reduction(base_sim["logical_total_variation_distance"],
                                       pk_sim["logical_total_variation_distance"])

            component_payloads = [
                {
                    "component": "h_commutation",
                    "pct_reduction_one_qubit_gates": h_cum_1q,
                    "pct_reduction_two_qubit_gates": h_cum_2q,
                    "pct_reduction_circuit_depth": h_cum_dep,
                    "pct_improvement_logical_selection_rate": h_cum_lsr,
                    "pct_reduction_logical_total_variation_distance": h_cum_tvd,
                },
                {
                    "component": "merging",
                    "pct_reduction_one_qubit_gates": hm_cum_1q - h_cum_1q,
                    "pct_reduction_two_qubit_gates": hm_cum_2q - h_cum_2q,
                    "pct_reduction_circuit_depth":   hm_cum_dep - h_cum_dep,
                    "pct_improvement_logical_selection_rate": hm_cum_lsr - h_cum_lsr,
                    "pct_reduction_logical_total_variation_distance": hm_cum_tvd - h_cum_tvd,
                },
                {
                    "component": "packing",
                    "pct_reduction_one_qubit_gates": pk_cum_1q - hm_cum_1q,
                    "pct_reduction_two_qubit_gates": pk_cum_2q - hm_cum_2q,
                    "pct_reduction_circuit_depth":   pk_cum_dep - hm_cum_dep,
                    "pct_improvement_logical_selection_rate": pk_cum_lsr - hm_cum_lsr,
                    "pct_reduction_logical_total_variation_distance": pk_cum_tvd - hm_cum_tvd,
                },
            ]

            for payload in component_payloads:
                rows.append(
                    {
                        "size_group": size_group,
                        "circuit_name": circuit_name,
                        "num_qubits": n_qubits,
                        "n_codes": n_codes,
                        "fraction_codes_used": fraction,
                        "alpha_final": alpha_final,
                        "final_repetitions": final_repetitions,
                        **payload,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "size_group", "circuit_name", "num_qubits", "n_codes", "fraction_codes_used",
    "alpha_final", "final_repetitions", "component",
    "pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates",
    "pct_reduction_circuit_depth", "pct_improvement_logical_selection_rate",
    "pct_reduction_logical_total_variation_distance",
]


def _append_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df[_CSV_COLUMNS].to_csv(csv_path, mode="a", index=False, header=write_header)
    print(f"[density_ablation_sweep] Appended {len(df)} rows to {csv_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _assign_plot_bins(
    df: pd.DataFrame,
    n_plot_bins: int,
) -> pd.DataFrame:
    df = df.copy()
    bin_edges = np.linspace(0.5, 1.0, n_plot_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df["plot_bin"] = pd.cut(
        df["fraction_codes_used"],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
    ).astype(float)
    return df


def _aggregate_for_plot(
    df: pd.DataFrame,
    n_plot_bins: int,
) -> pd.DataFrame:
    df = _assign_plot_bins(df, n_plot_bins)
    metric_cols = [
        "pct_reduction_one_qubit_gates",
        "pct_reduction_two_qubit_gates",
        "pct_reduction_circuit_depth",
        "pct_improvement_logical_selection_rate",
        "pct_reduction_logical_total_variation_distance",
    ]
    for col in metric_cols:
        df[col] = df[col].astype(float).clip(lower=0.0)
    grouped = (
        df.groupby(["size_group", "plot_bin", "component"])[metric_cols]
        .mean()
        .reset_index()
    )
    return grouped


def plot_results(df: pd.DataFrame, settings: DensityAblationSettings) -> None:
    agg = _aggregate_for_plot(df, settings.n_plot_bins)
    bin_centers = sorted(agg["plot_bin"].dropna().unique())
    n_bins = len(bin_centers)
    bin_step = 4.0
    bar_width = 0.75
    group_offsets = np.array([-bar_width, 0.0, bar_width])

    fig, axes = plt.subplots(nrows=len(METRICS), ncols=1, figsize=(14, 20), sharex=True)

    for ax, (_, (metric_col, metric_label)) in zip(axes, METRICS.items()):
        for bin_idx, bin_center in enumerate(bin_centers):
            for g_idx, size_group in enumerate(SIZE_GROUPS):
                x = bin_idx * bin_step + group_offsets[g_idx]
                sub = agg.loc[
                    (agg["size_group"] == size_group) & (agg["plot_bin"] == bin_center)
                ]
                if sub.empty:
                    continue
                bottom = 0.0
                for comp_key, _, color in COMPONENTS:
                    comp_row = sub.loc[sub["component"] == comp_key]
                    value = float(comp_row[metric_col].iloc[0]) if not comp_row.empty else 0.0
                    ax.bar(
                        x,
                        value,
                        width=bar_width,
                        bottom=bottom,
                        color=color,
                        edgecolor="white",
                        alpha=0.85,
                    )
                    bottom += value

        ax.set_ylabel(metric_label, fontsize=24)
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # X-axis ticks: one tick group per bin, labelled with fraction
    tick_positions = [
        bin_idx * bin_step + group_offsets[1]
        for bin_idx in range(n_bins)
    ]
    tick_labels = [f"{c:.2f}" for c in bin_centers]
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_xlabel("Fraction of codes used (code density)", fontsize=24)

    # Legend: component colors + size-group hatches
    component_patches = [
        mpatches.Patch(color=color, alpha=0.85, label=label)
        for _, label, color in COMPONENTS
    ]
    size_patches = [
        mpatches.Patch(color="#888888", alpha=0.0, label="")
    ]  # spacer
    axes[0].legend(
        handles=component_patches,
        loc="upper right",
        frameon=False,
        fontsize=13,
    )

    # Secondary size-group legend via text annotations
    for g_idx, size_group in enumerate(SIZE_GROUPS):
        x_annot = (n_bins - 1) * bin_step + group_offsets[g_idx]
        axes[-1].annotate(
            SIZE_GROUP_LABELS[size_group],
            xy=(x_annot, axes[-1].get_ylim()[0]),
            xytext=(x_annot, axes[-1].get_ylim()[0] - 4),
            ha="center",
            fontsize=10,
            color="#444444",
        )

    fig.supylabel("Percent improvement / reduction", fontsize=24)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.output_path, dpi=200)
    print(f"[density_ablation_sweep] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> DensityAblationSettings:
    parser = argparse.ArgumentParser(
        description="Ablation sweep across code density for all QASMBench circuits."
    )
    parser.add_argument("--qasm-root", type=Path, default=DEFAULT_SETTINGS.qasm_root)
    parser.add_argument("--max-gates", type=int, default=DEFAULT_SETTINGS.max_proxy_gates)
    parser.add_argument("--max-qubits", type=int, default=DEFAULT_SETTINGS.max_proxy_qubits)
    parser.add_argument("--proxy-timeout", type=float, default=DEFAULT_SETTINGS.proxy_timeout_seconds)
    parser.add_argument("--n-codes-samples", type=int, default=DEFAULT_SETTINGS.n_codes_samples)
    parser.add_argument("--n-plot-bins", type=int, default=DEFAULT_SETTINGS.n_plot_bins)
    parser.add_argument("--alpha-reference", type=float, default=DEFAULT_SETTINGS.alpha_reference)
    parser.add_argument("--reference-gate-count", type=float, default=DEFAULT_SETTINGS.reference_gate_count)
    parser.add_argument("--target-selection-rate", type=float, default=DEFAULT_SETTINGS.target_selection_rate)
    parser.add_argument("--pilot-repetitions", type=int, default=DEFAULT_SETTINGS.pilot_repetitions)
    parser.add_argument("--target-kept-shots", type=int, default=DEFAULT_SETTINGS.target_kept_shots)
    parser.add_argument("--min-repetitions", type=int, default=DEFAULT_SETTINGS.min_repetitions)
    parser.add_argument("--max-repetitions", type=int, default=DEFAULT_SETTINGS.max_repetitions)
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    parser.add_argument("--no-stim", action="store_true")
    parser.add_argument("--no-distance-noise", action="store_true")
    parser.add_argument("--distance-scalar", type=float, default=DEFAULT_SETTINGS.distance_scalar)
    parser.add_argument("--no-idle-noise", action="store_true")
    parser.add_argument("--idle-scalar", type=float, default=DEFAULT_SETTINGS.idle_scalar)
    parser.add_argument("--output", type=Path, default=DEFAULT_SETTINGS.output_path)
    parser.add_argument("--csv", type=Path, default=DEFAULT_SETTINGS.csv_path)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reload a previously saved CSV and replot without resimulating.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    return DensityAblationSettings(
        qasm_root=args.qasm_root,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        n_codes_samples=args.n_codes_samples,
        n_plot_bins=args.n_plot_bins,
        alpha_reference=args.alpha_reference,
        reference_gate_count=args.reference_gate_count,
        target_selection_rate=args.target_selection_rate,
        pilot_repetitions=args.pilot_repetitions,
        target_kept_shots=args.target_kept_shots,
        min_repetitions=args.min_repetitions,
        max_repetitions=args.max_repetitions,
        stim=not args.no_stim,
        distance_noise=not args.no_distance_noise,
        distance_scalar=args.distance_scalar,
        idle_noise=not args.no_idle_noise,
        idle_scalar=args.idle_scalar,
        seed=args.seed,
        output_path=args.output,
        csv_path=args.csv,
        from_csv=args.from_csv,
        show_plot=not args.no_show,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    settings = parse_args()

    if settings.from_csv is not None:
        df = pd.read_csv(settings.from_csv)
        plot_results(df, settings)
        return

    # Run in order: small → medium → large, appending CSV after each group
    for group in SIZE_GROUPS:
        circuits = _load_group_circuits(settings.qasm_root, group, settings)
        if not circuits:
            print(f"[density_ablation_sweep] No circuits loaded for {group}, skipping.")
            continue
        print(f"[density_ablation_sweep] Loaded {len(circuits)} {group} circuits.")
        group_df = run_experiment(circuits, group, settings)
        if not group_df.empty:
            _append_csv(group_df, settings.csv_path)

    if not settings.csv_path.exists():
        raise RuntimeError("No data was produced — check qasm_root and circuit filters.")

    df = pd.read_csv(settings.csv_path)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
