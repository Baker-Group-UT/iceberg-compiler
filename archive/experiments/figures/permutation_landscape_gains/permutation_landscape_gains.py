"""
Quantify how much room there is to improve over the permutation landscape.

For each random logical circuit, this script runs a full permutation scan using
Stim with distance and idle noise, then restricts to code counts below the
number of logical qubits. For each eligible code count it summarizes each
metric independently by taking the minimum, mean, and maximum value across all
permutations in that bucket.

The output includes:
  * Per-circuit/code-count summary CSV.
  * Aggregate summary CSV.
  * Boxplot figure of percent reductions in TVD, depth, 1q gates, and 2q gates.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Dict, List

from translator.config import TranslatorConfig

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
FONT_CACHE_DIR = THIS_DIR / ".cache"
(FONT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from tqdm.auto import tqdm

from experiments.shared.random_circuits import random_gate_circuit
from pipeline import PipelineConfig, run_full_scan
from simulator.config_presets import StandardSimulator


GATE_SET = ("h", "x", "z", "s", "cz", "cx")
METRICS: List[tuple[str, str]] = [
    ("logical_total_variation_distance", "TVD"),
    ("circuit_depth", "Depth"),
    ("one_qubit_gates", "1q Gates"),
    ("two_qubit_gates", "2q Gates"),
]


@dataclass(frozen=True)
class ExperimentConfig:
    num_qubits: int = 4
    num_gates: int = 20
    num_circuits: int = 3
    repetitions: int = 50_000
    seed: int = 7
    distance_scalar: float = 1.0
    idle_time_scalar: float = 1.0
    metric_column: str = "logical_total_variation_distance"
    output_dir: Path = field(default_factory=lambda: THIS_DIR)
    show_plot: bool = False

    @property
    def plot_path(self) -> Path:
        return self.output_dir / "permutation_landscape_gains_boxplot.png"

    @property
    def detailed_csv_path(self) -> Path:
        return self.output_dir / "permutation_landscape_gains_detailed.csv"

    @property
    def aggregate_csv_path(self) -> Path:
        return self.output_dir / "permutation_landscape_gains_aggregate.csv"


def pct_reduction(reference: float, challenger: float) -> float:
    if pd.isna(reference) or pd.isna(challenger):
        return float("nan")
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def build_random_circuit(cfg: ExperimentConfig, circuit_index: int):
    circuit_seed = cfg.seed * 1_000_003 + circuit_index * 101
    circuit = random_gate_circuit(
        num_qubits=cfg.num_qubits,
        num_gates=cfg.num_gates,
        gates=GATE_SET,
        seed=circuit_seed,
    )
    circuit.measure_all()
    return circuit, circuit_seed


def build_pipeline_config(cfg: ExperimentConfig) -> PipelineConfig:
    sim_config = (
        StandardSimulator(repetitions=cfg.repetitions, seed=cfg.seed)
        .with_stim()
        .with_distance_noise(distance_scalar=cfg.distance_scalar)
        .with_idle_noise(idle_time_scalar=cfg.idle_time_scalar)
    )
    translator_config = TranslatorConfig(enable_h_commutation=False)
    return PipelineConfig(sim_config=sim_config, translator_config=translator_config, verbose=False, save=False)


def summarize_bucket(
    bucket: pd.DataFrame,
    *,
    circuit_index: int,
    circuit_seed: int,
    num_qubits: int,
) -> Dict[str, float | int]:
    summary: Dict[str, float | int] = {
        "circuit_index": circuit_index,
        "circuit_seed": circuit_seed,
        "num_qubits": num_qubits,
        "codes_used": int(bucket["codes_used"].iloc[0]),
        "permutations_in_bucket": int(len(bucket)),
    }

    for metric, _label in METRICS:
        metric_values = pd.to_numeric(bucket[metric], errors="coerce").dropna()
        if metric_values.empty:
            continue

        best_value = float(metric_values.min())
        avg_value = float(bucket[metric].mean())
        worst_value = float(metric_values.max())

        summary[f"best_{metric}"] = best_value
        summary[f"avg_{metric}"] = avg_value
        summary[f"worst_{metric}"] = worst_value
        summary[f"best_vs_avg_{metric}_delta"] = avg_value - best_value
        summary[f"best_vs_worst_{metric}_delta"] = worst_value - best_value
        summary[f"best_vs_avg_{metric}_pct_reduction"] = pct_reduction(avg_value, best_value)
        summary[f"best_vs_worst_{metric}_pct_reduction"] = pct_reduction(
            worst_value, best_value
        )

    return summary


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    pipeline_cfg = build_pipeline_config(cfg)
    rows: List[Dict[str, float | int]] = []

    eligible_codes = range(max(2, (cfg.num_qubits + 1) // 2), cfg.num_qubits)
    progress = tqdm(range(cfg.num_circuits), desc="circuits")
    for circuit_index in progress:
        circuit, circuit_seed = build_random_circuit(cfg, circuit_index)
        with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                record = run_full_scan(circuit, config=pipeline_cfg)
        if record.dataframe is None:
            raise RuntimeError("Run record missing dataframe.")

        df = record.dataframe.df.copy()
        if cfg.metric_column not in df.columns:
            raise KeyError(f"Missing required column '{cfg.metric_column}' in scan dataframe.")

        df = df.dropna(subset=[cfg.metric_column])
        if df.empty:
            continue

        for code_count in eligible_codes:
            bucket = df.loc[df["codes_used"] == code_count].copy()
            if bucket.empty:
                continue
            rows.append(
                summarize_bucket(
                    bucket,
                    circuit_index=circuit_index,
                    circuit_seed=circuit_seed,
                    num_qubits=cfg.num_qubits,
                )
            )

    if not rows:
        raise RuntimeError("No eligible permutation buckets were produced.")

    return pd.DataFrame(rows)


def build_aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["codes_used"]
    agg_spec = {"permutations_in_bucket": ["count", "mean"]}
    for metric, _label in METRICS:
        for comp in ("best_vs_avg", "best_vs_worst"):
            agg_spec[f"{comp}_{metric}_pct_reduction"] = ["mean", "median", "min", "max"]

    aggregated = df.groupby(group_cols).agg(agg_spec)
    aggregated.columns = ["_".join(part for part in col if part) for col in aggregated.columns]
    return aggregated.reset_index()


def plot_results(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    code_counts = sorted(df["codes_used"].unique())
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8), sharex=True)

    colors = {"best_vs_avg": "#1f78b4", "best_vs_worst": "#e31a1c"}
    labels = {"best_vs_avg": "Best vs avg", "best_vs_worst": "Best vs worst"}

    for col_index, (metric, title) in enumerate(METRICS):
        for row_index, comparison in enumerate(("best_vs_avg", "best_vs_worst")):
            ax = axes[row_index, col_index]
            data = [
                df.loc[
                    df["codes_used"] == code_count,
                    f"{comparison}_{metric}_pct_reduction",
                ].dropna().to_numpy()
                for code_count in code_counts
            ]

            bp = ax.boxplot(
                data,
                tick_labels=[str(code_count) for code_count in code_counts],
                patch_artist=True,
                showfliers=True,
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=colors[comparison], alpha=0.65, edgecolor="#222222")
            for median in bp["medians"]:
                median.set(color="#111111", linewidth=1.8)

            ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
            ax.set_title(f"{title} | {labels[comparison]}")
            ax.set_ylabel("Percent reduction")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            if row_index == 1:
                ax.set_xlabel("Codes used")

    fig.suptitle(
        f"Permutation-landscape gains over {cfg.num_circuits} random {cfg.num_qubits}-qubit circuits"
    )
    fig.tight_layout()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.plot_path, dpi=220)
    print(f"[permutation_landscape_gains] Saved plot to {cfg.plot_path}")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Compare per-metric minima against average and maxima buckets."
    )
    parser.add_argument("--num-qubits", type=int, default=ExperimentConfig.num_qubits)
    parser.add_argument("--num-gates", type=int, default=ExperimentConfig.num_gates)
    parser.add_argument("--num-circuits", type=int, default=ExperimentConfig.num_circuits)
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument(
        "--distance-scalar", type=float, default=ExperimentConfig.distance_scalar
    )
    parser.add_argument(
        "--idle-time-scalar", type=float, default=ExperimentConfig.idle_time_scalar
    )
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        num_qubits=args.num_qubits,
        num_gates=args.num_gates,
        num_circuits=args.num_circuits,
        repetitions=args.repetitions,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        idle_time_scalar=args.idle_time_scalar,
        output_dir=args.output_dir,
        show_plot=args.show_plot,
    )


def main() -> None:
    cfg = parse_args()
    df = run_experiment(cfg)
    aggregate_df = build_aggregate_summary(df)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.detailed_csv_path, index=False)
    aggregate_df.to_csv(cfg.aggregate_csv_path, index=False)
    print(f"[permutation_landscape_gains] Saved detailed CSV to {cfg.detailed_csv_path}")
    print(f"[permutation_landscape_gains] Saved aggregate CSV to {cfg.aggregate_csv_path}")

    plot_results(df, cfg)


if __name__ == "__main__":
    main()
