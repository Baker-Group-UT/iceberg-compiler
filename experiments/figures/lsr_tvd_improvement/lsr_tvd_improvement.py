"""
Measure how H commutation, merging, and packing improve LSR and TVD.

This figure evaluates the ablation through simulation rather than translation
alone. For each circuit, it compares four staged variants at a fixed code
budget of ceil(n/2):

1. Tight packing, no H commutation, no merging
2. Tight packing, H commutation only
3. Tight packing, H commutation + merging
4. Merge-maximizing packing, H commutation + merging

The plotted values are average percent improvements, split into the additive
contributions from H commutation, merging, and packing. Two cohorts are shown:
random circuits and benchmark proxy circuits.

Run with:

    python -m experiments.figures.lsr_tvd_improvement.lsr_tvd_improvement
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
MPL_CACHE_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.random_circuits import random_gate_circuit
from packer.merge_packer import MergeMaximizingPacker
from pipeline import PipelineConfig, run_specific_scan
from simulator.config_presets import ScaledSimulator
from translator.passes import cancel_inverse_gates, commute_hadamards

Perm = List[Tuple[int, int]]

ACCENT = "#34444c"
H_COLOR = "#e0c36e"
MERGE_COLOR = "#7db5a8"
PACK_COLOR = "#4f82b8"
THIS_DIR = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "text.color": ACCENT,
        "axes.labelcolor": ACCENT,
        "axes.edgecolor": ACCENT,
        "xtick.color": ACCENT,
        "ytick.color": ACCENT,
    }
)

GATE_SET = ("h", "x", "z", "s", "cx", "cz")
SIZE_GROUPS = [
    ("small", (5, 10), (10, 50)),
    ("medium", (15, 25), (30, 150)),
    ("large", (30, 100), (60, 600)),
]
COMPONENTS = ("h_commutation", "merging", "packing")
METRICS = (
    ("logical_selection_rate", "LSR improvement"),
    ("logical_total_variation_distance", "TVD improvement"),
)


@dataclass(frozen=True)
class ExperimentSettings:
    random_circuits_per_group: int = 100
    random_seed: int = 10
    repetitions: int = 100_000
    alpha: float = 0.1
    stim: bool = True
    distance_noise: bool = True
    distance_scalar: float = 1.0
    idle_noise: bool = True
    idle_scalar: float = 1.0
    qasm_root: Path = Path("benchmarks/qasmbench")
    random_groups: Tuple[str, ...] = ("small", "medium")
    benchmark_groups: Tuple[str, ...] = ("small", "medium")
    max_proxy_gates: int | None = 500
    max_proxy_qubits: int | None = 100
    proxy_timeout_seconds: float | None = 1.0
    replace_rz_rx_with_s: bool = True
    min_baseline_lsr_for_percent: float = 0.01
    from_csv: Path | None = None
    output_dir: Path = field(default_factory=lambda: THIS_DIR)
    show_plot: bool = False

    @property
    def figure_path(self) -> Path:
        return self.output_dir / "lsr_tvd_improvement.svg"

    @property
    def raw_csv_path(self) -> Path:
        return self.output_dir / "lsr_tvd_improvement_raw.csv"

    @property
    def aggregate_csv_path(self) -> Path:
        return self.output_dir / "lsr_tvd_improvement_aggregate.csv"


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1.")
    patch_capacity = max(1, math.ceil(num_qubits / n_codes))
    permutation: Perm = []
    for i in range(num_qubits):
        code = min(i // patch_capacity, n_codes - 1)
        permutation.append((code, i % patch_capacity))
    return permutation


def preprocess_for_packing(program_circuit: qiskit.QuantumCircuit, enable_h_commutation: bool) -> qiskit.QuantumCircuit:
    processed = cancel_inverse_gates(program_circuit)
    if enable_h_commutation:
        processed = commute_hadamards(processed)
        processed = cancel_inverse_gates(processed)
    return processed


def ensure_measurements(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    if circuit.num_clbits > 0 and any(instruction.operation.name == "measure" for instruction in circuit.data):
        return circuit
    measured = copy.deepcopy(circuit)
    measured.measure_all()
    return measured


def percent_improvement(baseline: float, challenger: float, *, higher_is_better: bool) -> float:
    if abs(baseline) <= 1e-12:
        if abs(challenger - baseline) <= 1e-12:
            return 0.0
        return 0.0
    if higher_is_better:
        return 100.0 * (challenger - baseline) / baseline
    return 100.0 * (baseline - challenger) / baseline


def build_pipeline_config(settings: ExperimentSettings, *, seed: int, enable_h_commutation: bool, enable_merging: bool) -> PipelineConfig:
    sim_cfg = ScaledSimulator(
        alpha=settings.alpha,
        repetitions=settings.repetitions,
        seed=seed,
    )
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


def extract_metrics(record) -> Dict[str, float]:
    if record.dataframe is None or record.dataframe.df.empty:
        raise RuntimeError("run_specific_scan returned no dataframe.")
    row = record.dataframe.df.iloc[0]
    return {
        "logical_selection_rate": float(row["logical_selection_rate"]),
        "logical_total_variation_distance": float(row["logical_total_variation_distance"]),
    }


def evaluate_variant(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Dict[str, float]:
    record = run_specific_scan(program_circuit=program_circuit, permutations=[permutation], config=config)
    return extract_metrics(record)


def evaluate_ablation(
    circuit_name: str,
    cohort: str,
    size_group: str,
    program_circuit: qiskit.QuantumCircuit,
    settings: ExperimentSettings,
    *,
    seed: int,
) -> List[Dict[str, float | int | str]]:
    n_qubits = program_circuit.num_qubits
    target_codes = math.ceil(n_qubits / 2)
    tight_perm = tight_trivial_packing(n_qubits, target_codes)
    packer = MergeMaximizingPacker(solo_strategy="all")
    processed_for_packing = preprocess_for_packing(program_circuit, enable_h_commutation=True)
    packed_perm = packer.pack(processed_for_packing, num_codes=target_codes).permutation

    baseline_cfg = build_pipeline_config(
        settings,
        seed=seed,
        enable_h_commutation=False,
        enable_merging=False,
    )
    h_only_cfg = build_pipeline_config(
        settings,
        seed=seed,
        enable_h_commutation=True,
        enable_merging=False,
    )
    h_merge_cfg = build_pipeline_config(
        settings,
        seed=seed,
        enable_h_commutation=True,
        enable_merging=True,
    )

    baseline_metrics = evaluate_variant(program_circuit, tight_perm, baseline_cfg)
    h_only_metrics = evaluate_variant(program_circuit, tight_perm, h_only_cfg)
    h_merge_metrics = evaluate_variant(program_circuit, tight_perm, h_merge_cfg)
    packed_metrics = evaluate_variant(program_circuit, packed_perm, h_merge_cfg)

    staged_metrics = {
        "baseline": baseline_metrics,
        "h_only": h_only_metrics,
        "h_merge": h_merge_metrics,
        "packed": packed_metrics,
    }

    rows: List[Dict[str, float | int | str]] = []
    for metric_name, _ in METRICS:
        higher_is_better = metric_name == "logical_selection_rate"
        baseline = staged_metrics["baseline"][metric_name]
        h_cumulative = percent_improvement(
            baseline,
            staged_metrics["h_only"][metric_name],
            higher_is_better=higher_is_better,
        )
        h_merge_cumulative = percent_improvement(
            baseline,
            staged_metrics["h_merge"][metric_name],
            higher_is_better=higher_is_better,
        )
        packed_cumulative = percent_improvement(
            baseline,
            staged_metrics["packed"][metric_name],
            higher_is_better=higher_is_better,
        )
        component_values = {
            "h_commutation": h_cumulative,
            "merging": h_merge_cumulative - h_cumulative,
            "packing": packed_cumulative - h_merge_cumulative,
        }
        for component, value in component_values.items():
            rows.append(
                {
                    "cohort": cohort,
                    "size_group": size_group,
                    "circuit_name": circuit_name,
                    "num_qubits": n_qubits,
                    "target_codes": target_codes,
                    "metric": metric_name,
                    "component": component,
                    "pct_improvement": float(value),
                    "baseline_value": float(baseline),
                    "include_in_aggregate": (
                        metric_name != "logical_selection_rate"
                        or baseline >= settings.min_baseline_lsr_for_percent
                    ),
                    "h_only_value": float(staged_metrics["h_only"][metric_name]),
                    "h_merge_value": float(staged_metrics["h_merge"][metric_name]),
                    "packed_value": float(staged_metrics["packed"][metric_name]),
                }
            )
    return rows


def build_random_circuit(
    num_qubits: int,
    num_gates: int,
    *,
    seed: int,
) -> qiskit.QuantumCircuit:
    circuit = random_gate_circuit(
        num_qubits=num_qubits,
        num_gates=num_gates,
        gates=GATE_SET,
        seed=seed,
    )
    circuit.measure_all()
    return circuit


def load_random_circuits(settings: ExperimentSettings) -> List[Tuple[str, str, qiskit.QuantumCircuit, int]]:
    rng = random.Random(settings.random_seed)
    circuits: List[Tuple[str, str, qiskit.QuantumCircuit, int]] = []
    for size_group, qubit_range, gate_range in SIZE_GROUPS:
        if size_group not in settings.random_groups:
            continue
        for circuit_idx in range(settings.random_circuits_per_group):
            num_qubits = rng.randint(*qubit_range)
            num_gates = rng.randint(*gate_range)
            seed = rng.randint(0, 2**31 - 1)
            circuit = build_random_circuit(num_qubits, num_gates, seed=seed)
            name = f"{size_group}_random_{circuit_idx:02d}"
            circuits.append((size_group, name, circuit, seed))
    return circuits


def load_benchmark_proxy_circuits(settings: ExperimentSettings) -> List[Tuple[str, str, qiskit.QuantumCircuit, int]]:
    loaded: List[Tuple[str, str, qiskit.QuantumCircuit, int]] = []

    if settings.qasm_root.is_dir() and any((settings.qasm_root / group).is_dir() for group in settings.benchmark_groups):
        roots = [(group, settings.qasm_root / group) for group in settings.benchmark_groups]
    else:
        roots = [("benchmark", settings.qasm_root)]

    index = 0
    for size_group, root in roots:
        circuits = load_proxy_circuits(
            root=root,
            gate_limit=settings.max_proxy_gates,
            qubit_limit=settings.max_proxy_qubits,
            timeout_seconds=settings.proxy_timeout_seconds,
            replace_rz_rx_with_s=settings.replace_rz_rx_with_s,
        )
        for name, circuit in circuits:
            measured = ensure_measurements(circuit)
            seed = settings.random_seed * 10_000 + index
            loaded.append((size_group, name, measured, seed))
            index += 1
    return loaded


def infer_size_group(name: str) -> str:
    lowered = name.lower()
    for size_group, _, _ in SIZE_GROUPS:
        if size_group in lowered:
            return size_group
    return "benchmark"


def run_cohort(
    *,
    cohort: str,
    circuits: Sequence[Tuple[str, str, qiskit.QuantumCircuit, int]],
    settings: ExperimentSettings,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    iterator = tqdm(circuits, desc=f"{cohort} circuits")
    for size_group, circuit_name, program_circuit, seed in iterator:
        rows.extend(
            evaluate_ablation(
                circuit_name=circuit_name,
                cohort=cohort,
                size_group=size_group,
                program_circuit=program_circuit,
                settings=settings,
                seed=seed,
            )
        )
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["cohort", "metric", "component", "avg_pct_improvement"]
        )
    filtered_df = df.loc[df["include_in_aggregate"]].copy()
    grouped = (
        filtered_df.groupby(["cohort", "metric", "component"], as_index=False)["pct_improvement"]
        .mean()
        .rename(columns={"pct_improvement": "avg_pct_improvement"})
    )
    return grouped


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    if "include_in_aggregate" in df.columns:
        df["include_in_aggregate"] = df["include_in_aggregate"].astype(str).str.lower().map(
            {"true": True, "false": False}
        ).fillna(df["include_in_aggregate"])
        df["include_in_aggregate"] = df["include_in_aggregate"].astype(bool)
    return df


def print_summary(aggregate_df: pd.DataFrame) -> None:
    if aggregate_df.empty:
        print("[lsr_tvd_improvement] No results to summarize.")
        return
    print("[lsr_tvd_improvement] Average additive percent improvements")
    for cohort in ("random", "benchmark"):
        cohort_df = aggregate_df.loc[aggregate_df["cohort"] == cohort]
        if cohort_df.empty:
            continue
        print(f"  {cohort}:")
        for metric_name, metric_label in METRICS:
            metric_df = cohort_df.loc[cohort_df["metric"] == metric_name]
            parts = []
            total = 0.0
            for component in COMPONENTS:
                value = float(
                    metric_df.loc[metric_df["component"] == component, "avg_pct_improvement"].mean()
                )
                total += value
                parts.append(f"{component}={value:.2f}%")
            print(f"    {metric_label}: " + ", ".join(parts) + f", total={total:.2f}%")


def print_filter_summary(raw_df: pd.DataFrame, settings: ExperimentSettings) -> None:
    lsr_df = raw_df.loc[raw_df["metric"] == "logical_selection_rate"]
    if lsr_df.empty:
        return
    circuit_df = (
        lsr_df[["cohort", "circuit_name", "baseline_value", "include_in_aggregate"]]
        .drop_duplicates()
        .copy()
    )
    excluded = circuit_df.loc[~circuit_df["include_in_aggregate"]]
    if excluded.empty:
        print(
            "[lsr_tvd_improvement] LSR aggregate kept all circuits "
            f"(baseline threshold {settings.min_baseline_lsr_for_percent:.4f})."
        )
        return
    print(
        "[lsr_tvd_improvement] Excluded low-baseline LSR circuits from LSR aggregate "
        f"(threshold {settings.min_baseline_lsr_for_percent:.4f}):"
    )
    for cohort in ("random", "benchmark"):
        cohort_excluded = excluded.loc[excluded["cohort"] == cohort]
        if cohort_excluded.empty:
            continue
        print(f"  {cohort}: {len(cohort_excluded)} circuits excluded")


def plot_results(aggregate_df: pd.DataFrame, settings: ExperimentSettings) -> None:
    cohorts = ["random", "benchmark"]
    component_styles = {
        "h_commutation": ("H commutation", H_COLOR),
        "merging": ("Merging", MERGE_COLOR),
        "packing": ("Packing", PACK_COLOR),
    }
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 5.2), sharey=False)
    x = np.arange(len(cohorts))

    for ax, (metric_name, metric_label) in zip(axes, METRICS):
        metric_df = aggregate_df.loc[aggregate_df["metric"] == metric_name]
        bottoms = np.zeros(len(cohorts), dtype=float)
        for component in COMPONENTS:
            values = []
            for cohort in cohorts:
                match = metric_df.loc[
                    (metric_df["cohort"] == cohort) & (metric_df["component"] == component),
                    "avg_pct_improvement",
                ]
                values.append(float(match.iloc[0]) if not match.empty else 0.0)
            label, color = component_styles[component]
            ax.bar(
                x,
                values,
                width=0.58,
                bottom=bottoms,
                color=color,
                edgecolor="white",
                linewidth=1.2,
                label=label,
            )
            bottoms += np.asarray(values, dtype=float)

        ax.axhline(0, color="#7a878d", linewidth=1.0, linestyle="--")
        ax.set_xticks(x, cohorts)
        ax.set_ylabel("Average percent improvement")
        ax.set_title(metric_label)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("AQORN percent improvement from H commutation, merging, and packing")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.figure_path, dpi=200, bbox_inches="tight")
    print(f"[lsr_tvd_improvement] Saved figure to {settings.figure_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> ExperimentSettings:
    parser = argparse.ArgumentParser(
        description="Simulated LSR/TVD ablation for H commutation, merging, and packing."
    )
    parser.add_argument("--random-circuits-per-group", type=int, default=ExperimentSettings.random_circuits_per_group)
    parser.add_argument("--random-seed", type=int, default=ExperimentSettings.random_seed)
    parser.add_argument("--repetitions", type=int, default=ExperimentSettings.repetitions)
    parser.add_argument("--alpha", type=float, default=ExperimentSettings.alpha)
    parser.add_argument("--no-stim", action="store_true", help="Use Aer instead of Stim.")
    parser.add_argument("--no-distance-noise", action="store_true", help="Disable distance-based noise.")
    parser.add_argument("--distance-scalar", type=float, default=ExperimentSettings.distance_scalar)
    parser.add_argument("--no-idle-noise", action="store_true", help="Disable idle noise.")
    parser.add_argument("--idle-scalar", type=float, default=ExperimentSettings.idle_scalar)
    parser.add_argument("--qasm-root", type=Path, default=ExperimentSettings.qasm_root)
    parser.add_argument(
        "--random-groups",
        nargs="+",
        default=list(ExperimentSettings.random_groups),
        choices=["small", "medium", "large"],
        help="Random-circuit size groups to include.",
    )
    parser.add_argument(
        "--benchmark-groups",
        nargs="+",
        default=list(ExperimentSettings.benchmark_groups),
        choices=["small", "medium", "large"],
        help="Benchmark size folders to include when qasm-root points at qasmbench.",
    )
    parser.add_argument("--max-gates", type=int, default=ExperimentSettings.max_proxy_gates)
    parser.add_argument("--max-qubits", type=int, default=ExperimentSettings.max_proxy_qubits)
    parser.add_argument("--proxy-timeout-seconds", type=float, default=ExperimentSettings.proxy_timeout_seconds)
    parser.add_argument(
        "--keep-rz-rx",
        action="store_true",
        help="Keep proxy RZ/RX gates instead of mapping them to S.",
    )
    parser.add_argument(
        "--min-baseline-lsr-for-percent",
        type=float,
        default=ExperimentSettings.min_baseline_lsr_for_percent,
        help="Exclude circuits from the LSR percent aggregate when baseline LSR is below this threshold.",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reuse an existing raw CSV instead of rerunning simulations.",
    )
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    return ExperimentSettings(
        random_circuits_per_group=args.random_circuits_per_group,
        random_seed=args.random_seed,
        repetitions=args.repetitions,
        alpha=args.alpha,
        stim=not args.no_stim,
        distance_noise=not args.no_distance_noise,
        distance_scalar=args.distance_scalar,
        idle_noise=not args.no_idle_noise,
        idle_scalar=args.idle_scalar,
        qasm_root=args.qasm_root,
        random_groups=tuple(args.random_groups),
        benchmark_groups=tuple(args.benchmark_groups),
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout_seconds,
        replace_rz_rx_with_s=not args.keep_rz_rx,
        min_baseline_lsr_for_percent=args.min_baseline_lsr_for_percent,
        from_csv=args.from_csv,
        output_dir=args.output_dir,
        show_plot=args.show_plot,
    )


def main() -> None:
    settings = parse_args()
    if settings.from_csv is not None:
        raw_df = load_raw_dataframe(settings.from_csv)
        print(f"[lsr_tvd_improvement] Loaded raw data from {settings.from_csv}")
    else:
        random_df = run_cohort(
            cohort="random",
            circuits=load_random_circuits(settings),
            settings=settings,
        )
        benchmark_df = run_cohort(
            cohort="benchmark",
            circuits=load_benchmark_proxy_circuits(settings),
            settings=settings,
        )
        raw_df = pd.concat([random_df, benchmark_df], ignore_index=True)

    if "include_in_aggregate" not in raw_df.columns:
        raw_df["include_in_aggregate"] = (
            (raw_df["metric"] != "logical_selection_rate")
            | (pd.to_numeric(raw_df["baseline_value"], errors="coerce") >= settings.min_baseline_lsr_for_percent)
        )
    aggregate_df = aggregate_results(raw_df)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(settings.raw_csv_path, index=False)
    aggregate_df.to_csv(settings.aggregate_csv_path, index=False)
    print(f"[lsr_tvd_improvement] Saved raw data to {settings.raw_csv_path}")
    print(f"[lsr_tvd_improvement] Saved aggregate data to {settings.aggregate_csv_path}")

    print_filter_summary(raw_df, settings)
    print_summary(aggregate_df)
    plot_results(aggregate_df, settings)


if __name__ == "__main__":
    main()
