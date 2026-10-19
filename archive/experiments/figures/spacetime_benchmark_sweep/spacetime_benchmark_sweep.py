"""
spacetime_benchmark_sweep.py
-----------------------------

Sweeps fraction-of-codes-used for three fixed benchmark circuits:
  - hs4_n4        (4 qubits)
  - bigadder_n18  (18 qubits)
  - dnn_n51       (51 qubits)

For each (circuit, code_count) pair, measures how much the optimized variant
(H commutation + merging + MergeMaximizingPacker) improves over the baseline
(tight trivial packing with H commutation only) across gate/depth metrics:

  1. #1q gates reduction
  2. #2q gates reduction
  3. Depth reduction
  4. LSR improvement
  5. TVD reduction

Gate metrics are computed from translation only. LSR and TVD use full
simulation with per-circuit alpha calibration.

Run with:

    python -m experiments.figures.spacetime_benchmark_sweep.spacetime_benchmark_sweep
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import create_proxy_from_qasm
from archive.packer.merge_packer import MergeMaximizingPacker
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.config_presets import ScaledSimulator
from translator.merging import PairMergeAnalyzer
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
# Circuit registry
# ---------------------------------------------------------------------------

BENCHMARK_CIRCUITS: List[Tuple[str, Path]] = [
    ("hs4_n4",       Path("benchmarks/qasmbench/small/hs4_n4/hs4_n4.qasm")),
    ("bigadder_n18", Path("benchmarks/qasmbench/medium/bigadder_n18/bigadder_n18.qasm")),
    ("dnn_n51",      Path("benchmarks/qasmbench/large/dnn_n51/dnn_n51.qasm")),
]

CIRCUIT_COLORS: Dict[str, str] = {
    "hs4_n4":       "#3b4c56",
    "bigadder_n18": "#0097fd",
    "dnn_n51":      "#cdc2c0",
}

METRICS = [
    ("pct_reduction_one_qubit_gates",              "#1q gates"),
    ("pct_reduction_two_qubit_gates",              "#2q gates"),
    ("pct_reduction_circuit_depth",                "depth"),
    ("pct_improvement_lsr",                        "LSR"),
    ("pct_reduction_tvd",                          "TVD"),
]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSweepSettings:
    n_codes_samples: int = 8
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
            "experiments/figures/spacetime_benchmark_sweep/spacetime_benchmark_sweep.svg"
        )
    )
    csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/spacetime_benchmark_sweep/spacetime_benchmark_sweep.csv"
        )
    )
    from_csv: Path | None = None
    show_plot: bool = False


DEFAULT_SETTINGS = BenchmarkSweepSettings()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _perm_pair_cx_merges(
    preprocessed: qiskit.QuantumCircuit,
    permutation: Perm,
) -> Tuple[int, int, int]:
    """Return total (cx_merges, cz_merges, xcx_merges) across all paired patches."""
    analyzer = PairMergeAnalyzer(preprocessed)
    n = preprocessed.num_qubits
    patch_to_qubits: Dict[int, List[int]] = {}
    for qubit_idx, (patch, _slot) in enumerate(permutation):
        patch_to_qubits.setdefault(patch, []).append(qubit_idx)
    cx = cz = xcx = 0
    for qubits in patch_to_qubits.values():
        if len(qubits) < 2:
            continue
        qa, qb = qubits[0], qubits[1]
        stats = analyzer.analyze_pair(qa, qb)
        cx += stats.cx_merges
        cz += stats.cz_merges
        xcx += stats.xcx_merges
    return cx, cz, xcx


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
    then assigns remaining qubits to solo patches.  This matches the merge
    packer's patch count exactly, making the comparison fair.
    """
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    min_codes = math.ceil(num_qubits / 2)
    if n_codes < min_codes or n_codes > num_qubits:
        raise ValueError(
            f"n_codes={n_codes} out of range [{min_codes}, {num_qubits}] "
            f"for {num_qubits} qubits."
        )
    pairs_needed = num_qubits - n_codes   # number of paired patches
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

def _base_sim_cfg(settings: BenchmarkSweepSettings, alpha: float, repetitions: int, seed: int):
    sim_cfg = ScaledSimulator(alpha=alpha, repetitions=repetitions, seed=seed)
    if settings.stim:
        sim_cfg = sim_cfg.with_stim()
    if settings.distance_noise:
        sim_cfg = sim_cfg.with_distance_noise(distance_scalar=settings.distance_scalar)
    if settings.idle_noise:
        sim_cfg = sim_cfg.with_idle_noise(idle_time_scalar=settings.idle_scalar)
    return sim_cfg


def build_pipeline_config(
    settings: BenchmarkSweepSettings,
    *,
    alpha: float,
    repetitions: int,
    seed: int,
    enable_h_commutation: bool,
    enable_merging: bool,
) -> PipelineConfig:
    sim_cfg = _base_sim_cfg(settings, alpha, repetitions, seed)
    config = PipelineConfig(verbose=False, save=False, sim_config=sim_cfg, seed=seed)
    config.translator_config.enable_h_commutation = enable_h_commutation
    config.translator_config.enable_merging = enable_merging
    return config


# ---------------------------------------------------------------------------
# Translation metric helpers
# ---------------------------------------------------------------------------

def compute_translation_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Tuple[int, int, int, float, float]:
    """Returns (1q, 2q, depth, merge_savings_1q, merge_savings_2q)."""
    translated = translate_permutation(program_circuit, permutation, config=config)
    circuit = translated.circuit
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    return one_q, two_q, depth, translated.merge_savings_1q, translated.merge_savings_2q


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

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


def calibrated_alpha_from_gate_count(gate_count: int, settings: BenchmarkSweepSettings) -> float:
    gate_count = max(1, int(gate_count))
    return settings.alpha_reference * (settings.reference_gate_count / float(gate_count))


def corrected_alpha(alpha: float, observed_selection_rate: float, target_selection_rate: float) -> float:
    eps = 1e-6
    observed = float(np.clip(observed_selection_rate, eps, 1.0 - eps))
    target = float(np.clip(target_selection_rate, eps, 1.0 - eps))
    return alpha * math.log(target) / math.log(observed)


def repetitions_from_selection_rate(selection_rate: float, settings: BenchmarkSweepSettings) -> int:
    denom = max(selection_rate, 1e-4)
    reps = int(math.ceil(settings.target_kept_shots / denom))
    return int(np.clip(reps, settings.min_repetitions, settings.max_repetitions))


# ---------------------------------------------------------------------------
# PipelineConfig stubs for translation-only (no sim)
# ---------------------------------------------------------------------------

def _baseline_translate_cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.translator_config.enable_h_commutation = True
    cfg.translator_config.enable_merging = False
    return cfg


def _h_merge_translate_cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.translator_config.enable_h_commutation = True
    cfg.translator_config.enable_merging = True
    return cfg


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(settings: BenchmarkSweepSettings) -> pd.DataFrame:
    packer = MergeMaximizingPacker()
    h_only_cfg = _baseline_translate_cfg()
    h_merge_cfg = _h_merge_translate_cfg()

    rows: List[Dict] = []

    for circuit_idx, (circuit_name, qasm_path) in enumerate(
        tqdm(BENCHMARK_CIRCUITS, desc="circuits")
    ):
        proxy = create_proxy_from_qasm(qasm_path, replace_rz_rx_with_s=True)
        if proxy is None:
            print(f"[spacetime_benchmark_sweep] WARNING: could not load {circuit_name}, skipping.")
            continue

        program_circuit = ensure_measurements(proxy)
        n_qubits = program_circuit.num_qubits
        min_codes = math.ceil(n_qubits / 2)
        max_codes = n_qubits

        # Build evenly-spaced code counts, deduplicated
        raw = np.linspace(min_codes, max_codes, settings.n_codes_samples)
        code_counts = sorted(set(int(round(v)) for v in raw))

        # Preprocess once for packing
        preprocessed = preprocess_for_packing(program_circuit, h_merge_cfg)

        # One-time alpha calibration from baseline at min_codes
        baseline_perm_calib = tight_trivial_packing(n_qubits, min_codes)
        calib_one_q, calib_two_q, _, _, _ = compute_translation_metrics(
            program_circuit, baseline_perm_calib, h_only_cfg
        )
        baseline_gate_count = max(1, calib_one_q + calib_two_q)
        alpha_initial = calibrated_alpha_from_gate_count(baseline_gate_count, settings)

        pilot_baseline_cfg = build_pipeline_config(
            settings,
            alpha=alpha_initial,
            repetitions=settings.pilot_repetitions,
            seed=settings.seed * 1_000_003 + circuit_idx * 17 + 1,
            enable_h_commutation=True,
            enable_merging=False,
        )
        pilot_metrics = evaluate_variant(program_circuit, baseline_perm_calib, pilot_baseline_cfg)

        alpha_final = corrected_alpha(
            alpha_initial,
            pilot_metrics["logical_selection_rate"],
            settings.target_selection_rate,
        )

        pilot_verify_cfg = build_pipeline_config(
            settings,
            alpha=alpha_final,
            repetitions=settings.pilot_repetitions,
            seed=settings.seed * 1_000_003 + circuit_idx * 17 + 2,
            enable_h_commutation=True,
            enable_merging=False,
        )
        pilot_verify = evaluate_variant(program_circuit, baseline_perm_calib, pilot_verify_cfg)
        final_repetitions = repetitions_from_selection_rate(
            pilot_verify["logical_selection_rate"], settings
        )

        for code_idx, n_codes in enumerate(
            tqdm(code_counts, desc=f"  {circuit_name} code sweep", leave=False)
        ):
            fraction = n_codes / n_qubits

            baseline_perm = tight_trivial_packing(n_qubits, n_codes)
            packed_perm = packer.pack(preprocessed, num_codes=n_codes).permutation

            # Gate metrics (translation only)
            base_1q, base_2q, base_depth, _, _ = compute_translation_metrics(
                program_circuit, baseline_perm, h_only_cfg
            )
            packed_1q, packed_2q, packed_depth, packed_ms1q, packed_ms2q = compute_translation_metrics(
                program_circuit, packed_perm, h_merge_cfg
            )

            # Diagnostic: compare merge counts between tight-trivial and merge-packer pairs.
            base_cx, base_cz, base_xcx = _perm_pair_cx_merges(preprocessed, baseline_perm)
            pack_cx, pack_cz, pack_xcx = _perm_pair_cx_merges(preprocessed, packed_perm)
            if packed_2q > base_2q:
                print(
                    f"[DIAG 2q-regression] {circuit_name} n_codes={n_codes}:"
                    f"  base_2q={base_2q}  packed_2q={packed_2q}"
                    f"  actual_ms2q={packed_ms2q:.1f}"
                    f"  | trivial cx/cz/xcx={base_cx}/{base_cz}/{base_xcx}"
                    f"  packed cx/cz/xcx={pack_cx}/{pack_cz}/{pack_xcx}"
                )

            # Simulation metrics
            seed_offset = circuit_idx * 10_000 + code_idx * 10
            baseline_sim_cfg = build_pipeline_config(
                settings,
                alpha=alpha_final,
                repetitions=final_repetitions,
                seed=settings.seed * 1_000_003 + seed_offset + 3,
                enable_h_commutation=True,
                enable_merging=False,
            )
            packed_sim_cfg = build_pipeline_config(
                settings,
                alpha=alpha_final,
                repetitions=final_repetitions,
                seed=settings.seed * 1_000_003 + seed_offset + 4,
                enable_h_commutation=True,
                enable_merging=True,
            )
            baseline_sim = evaluate_variant(program_circuit, baseline_perm, baseline_sim_cfg)
            packed_sim = evaluate_variant(program_circuit, packed_perm, packed_sim_cfg)

            rows.append(
                {
                    "circuit_name": circuit_name,
                    "n_qubits": n_qubits,
                    "n_codes": n_codes,
                    "fraction_codes_used": fraction,
                    "pct_reduction_one_qubit_gates": pct_reduction(base_1q, packed_1q),
                    "pct_reduction_two_qubit_gates": pct_reduction(base_2q, packed_2q),
                    "pct_reduction_circuit_depth": pct_reduction(base_depth, packed_depth),
                    "pct_improvement_lsr": pct_improvement(
                        baseline_sim["logical_selection_rate"],
                        packed_sim["logical_selection_rate"],
                    ),
                    "pct_reduction_tvd": pct_reduction(
                        baseline_sim["logical_total_variation_distance"],
                        packed_sim["logical_total_variation_distance"],
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, settings: BenchmarkSweepSettings) -> None:
    circuit_names = [name for name, _ in BENCHMARK_CIRCUITS if name in df["circuit_name"].unique()]

    fig, axes = plt.subplots(nrows=len(METRICS), ncols=1, figsize=(10, 20), sharex=True)

    for ax, (metric_col, metric_label) in zip(axes, METRICS):
        for circuit_name in circuit_names:
            sub = df.loc[df["circuit_name"] == circuit_name].sort_values("fraction_codes_used")
            if sub.empty:
                continue
            color = CIRCUIT_COLORS.get(circuit_name, None)
            ax.plot(
                sub["fraction_codes_used"].values,
                sub[metric_col].values,
                label=circuit_name,
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=5,
            )

        ax.set_ylabel(metric_label, fontsize=24)
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlim(0.5, 1.0)

    axes[-1].set_xlabel("Fraction of codes used", fontsize=24)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="upper right", frameon=False)

    fig.supylabel("Percent improvement / reduction", fontsize=24)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.output_path, dpi=200)
    print(f"[spacetime_benchmark_sweep] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> BenchmarkSweepSettings:
    parser = argparse.ArgumentParser(
        description="Sweep fraction-of-codes-used for fixed benchmark circuits."
    )
    parser.add_argument(
        "--n-codes-samples",
        type=int,
        default=DEFAULT_SETTINGS.n_codes_samples,
        help="Number of evenly-spaced code-count samples per circuit.",
    )
    parser.add_argument(
        "--alpha-reference",
        type=float,
        default=DEFAULT_SETTINGS.alpha_reference,
    )
    parser.add_argument(
        "--reference-gate-count",
        type=float,
        default=DEFAULT_SETTINGS.reference_gate_count,
    )
    parser.add_argument(
        "--target-selection-rate",
        type=float,
        default=DEFAULT_SETTINGS.target_selection_rate,
    )
    parser.add_argument(
        "--pilot-repetitions",
        type=int,
        default=DEFAULT_SETTINGS.pilot_repetitions,
    )
    parser.add_argument(
        "--target-kept-shots",
        type=int,
        default=DEFAULT_SETTINGS.target_kept_shots,
    )
    parser.add_argument(
        "--min-repetitions",
        type=int,
        default=DEFAULT_SETTINGS.min_repetitions,
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        default=DEFAULT_SETTINGS.max_repetitions,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    parser.add_argument("--no-stim", action="store_true")
    parser.add_argument("--no-distance-noise", action="store_true")
    parser.add_argument(
        "--distance-scalar",
        type=float,
        default=DEFAULT_SETTINGS.distance_scalar,
    )
    parser.add_argument("--no-idle-noise", action="store_true")
    parser.add_argument(
        "--idle-scalar",
        type=float,
        default=DEFAULT_SETTINGS.idle_scalar,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SETTINGS.output_path,
        help="Path to save the SVG plot.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_SETTINGS.csv_path,
        help="Path to save/load the results CSV.",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reload a previously saved CSV and replot without resimulating.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    return BenchmarkSweepSettings(
        n_codes_samples=args.n_codes_samples,
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

    df = run_experiment(settings)

    settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.csv_path, index=False)
    print(f"[spacetime_benchmark_sweep] Saved data to {settings.csv_path}")

    plot_results(df, settings)


if __name__ == "__main__":
    main()
