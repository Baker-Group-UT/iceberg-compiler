"""
Quantify how strongly LSR correlates with TVD across random circuits.

For each random circuit and each fixed code count, this script:

1. Generates permutations that all use exactly that number of codes.
2. Runs ``run_specific_scan`` for those permutations.
3. Computes per-circuit Pearson r and Spearman rho between LSR and TVD.

Outputs:
  * Per-circuit correlation CSV.
  * Aggregate-by-code-count CSV.
  * A two-panel trend figure for Pearson and Spearman.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from experiments.shared.random_circuits import random_gate_circuit
from permutations.generator import permutation_with_n_codes
from pipeline import PipelineConfig, run_specific_scan
from simulator.config_presets import ScaledSimulator
from translator.config import TranslatorConfig

ACCENT = "#3b4c56"
BLUE = "#005a96"
LIGHT_BLUE = "#8db7d3"
PERM_REPEAT_LIMIT = 2_500
GATE_SET = ("h", "x", "z", "cz", "cx")
THIS_DIR = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.color": ACCENT,
        "axes.labelcolor": ACCENT,
        "axes.edgecolor": ACCENT,
        "xtick.color": ACCENT,
        "ytick.color": ACCENT,
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    num_qubits: int = 20
    num_gates: int = 100
    num_circuits: int = 12
    permutations_per_circuit: int = 20
    repetitions: int = 100_000
    seed: int = 11
    distance_scalar: float = 1.0
    idle_scalar: float = 1.0
    min_codes: int | None = None
    max_codes: int | None = None
    output_dir: Path = field(default_factory=lambda: THIS_DIR)
    show_plot: bool = False

    @property
    def circuit_csv_path(self) -> Path:
        return self.output_dir / "lsr_tvd_correlation_random_circuit_summary.csv"

    @property
    def aggregate_csv_path(self) -> Path:
        return self.output_dir / "lsr_tvd_correlation_random_aggregate_summary.csv"

    @property
    def figure_path(self) -> Path:
        return self.output_dir / "lsr_tvd_correlation_random_trends.svg"


def _code_counts(cfg: ExperimentConfig) -> list[int]:
    min_codes = cfg.min_codes if cfg.min_codes is not None else math.ceil(cfg.num_qubits / 2)
    max_codes = cfg.max_codes if cfg.max_codes is not None else cfg.num_qubits
    if min_codes < math.ceil(cfg.num_qubits / 2):
        raise ValueError(
            f"min_codes must be at least ceil(num_qubits / 2) = {math.ceil(cfg.num_qubits / 2)}."
        )
    if max_codes < min_codes:
        raise ValueError("max_codes must be >= min_codes.")
    return list(range(min_codes, max_codes + 1))


def _pd_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _corr_summary(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if len(x_arr) < 2:
        return {
            "n": float(len(x_arr)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    if np.isclose(np.std(x_arr), 0.0) or np.isclose(np.std(y_arr), 0.0):
        return {
            "n": float(len(x_arr)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    pear = pearsonr(x_arr, y_arr)
    spear = spearmanr(x_arr, y_arr)
    return {
        "n": float(len(x_arr)),
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_rho": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


def _build_random_circuit(cfg: ExperimentConfig, circuit_index: int):
    circuit_seed = cfg.seed * 1_000_003 + circuit_index * 101
    circuit = random_gate_circuit(
        num_qubits=cfg.num_qubits,
        num_gates=cfg.num_gates,
        gates=GATE_SET,
        seed=circuit_seed,
    )
    circuit.measure_all()
    return circuit, circuit_seed


def _build_pipeline_config(cfg: ExperimentConfig, seed: int) -> PipelineConfig:
    sim_cfg = (
        ScaledSimulator(alpha=0.05, repetitions=cfg.repetitions, seed=seed)
        .with_stim()
        .with_distance_noise(distance_scalar=cfg.distance_scalar)
        .with_idle_noise(idle_time_scalar=cfg.idle_scalar)
    )
    translator_cfg = TranslatorConfig(enable_h_commutation=False)
    return PipelineConfig(
        sim_config=sim_cfg,
        translator_config=translator_cfg,
        verbose=False,
        save=False,
        seed=seed,
    )


def _generate_fixed_code_permutations(
    *,
    num_qubits: int,
    target_codes: int,
    count: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    permutations: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    failures = 0
    next_seed = seed

    while len(permutations) < count and failures < PERM_REPEAT_LIMIT:
        perm = permutation_with_n_codes(q=num_qubits, n_codes=target_codes, seed=next_seed)
        next_seed += 1
        key = tuple(perm)
        if key in seen:
            failures += 1
            continue
        seen.add(key)
        permutations.append(perm)

    if len(permutations) < count:
        raise RuntimeError(
            f"Only generated {len(permutations)} unique permutations for "
            f"q={num_qubits}, target_codes={target_codes}."
        )
    return permutations


def _prep_df(record) -> pd.DataFrame:
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")

    df = record.dataframe.df.copy()
    required = {
        "logical_selection_rate",
        "logical_total_variation_distance",
        "codes_used",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Scan dataframe missing columns: {sorted(missing)}")

    for col in required:
        df[col] = _pd_to_numeric(df[col])

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    return df


def _summarize_circuit_bucket(
    df: pd.DataFrame,
    *,
    circuit_index: int,
    circuit_seed: int,
    target_codes: int,
) -> Dict[str, float | int]:
    summary: Dict[str, float | int] = {
        "circuit_index": circuit_index,
        "circuit_seed": circuit_seed,
        "codes_used": target_codes,
        "n_points": int(len(df)),
        "mean_lsr": float(df["logical_selection_rate"].mean()) if not df.empty else float("nan"),
        "mean_tvd": float(df["logical_total_variation_distance"].mean()) if not df.empty else float("nan"),
    }
    summary.update(
        _corr_summary(df["logical_selection_rate"], df["logical_total_variation_distance"])
    )
    return summary


def _build_aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | int]] = []
    for codes_used, bucket in df.groupby("codes_used"):
        rows.append(
            {
                "codes_used": int(codes_used),
                "n_circuits": int(len(bucket)),
                "pearson_r_median": float(bucket["pearson_r"].median()),
                "pearson_r_mean": float(bucket["pearson_r"].mean()),
                "pearson_r_q25": float(bucket["pearson_r"].quantile(0.25)),
                "pearson_r_q75": float(bucket["pearson_r"].quantile(0.75)),
                "spearman_rho_median": float(bucket["spearman_rho"].median()),
                "spearman_rho_mean": float(bucket["spearman_rho"].mean()),
                "spearman_rho_q25": float(bucket["spearman_rho"].quantile(0.25)),
                "spearman_rho_q75": float(bucket["spearman_rho"].quantile(0.75)),
                "n_points_median": float(bucket["n_points"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("codes_used").reset_index(drop=True)


def _plot_metric_trend(
    ax,
    *,
    circuit_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    for circuit_id, bucket in circuit_df.groupby("circuit_index"):
        bucket = bucket.sort_values("codes_used")
        ax.plot(
            bucket["codes_used"].to_numpy(dtype=float),
            bucket[metric].to_numpy(dtype=float),
            color=LIGHT_BLUE,
            linewidth=1.0,
            alpha=0.35,
        )

    x = aggregate_df["codes_used"].to_numpy(dtype=float)
    median = aggregate_df[f"{metric}_median"].to_numpy(dtype=float)
    q25 = aggregate_df[f"{metric}_q25"].to_numpy(dtype=float)
    q75 = aggregate_df[f"{metric}_q75"].to_numpy(dtype=float)

    ax.fill_between(x, q25, q75, color=BLUE, alpha=0.18)
    ax.plot(x, median, color=BLUE, linewidth=2.5)
    ax.axhline(0.0, color="#7a7a7a", linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Codes used")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_results(
    circuit_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 4.8), sharex=True)

    _plot_metric_trend(
        axes[0],
        circuit_df=circuit_df,
        aggregate_df=aggregate_df.rename(
            columns={
                "pearson_r_median": "pearson_r_median",
                "pearson_r_q25": "pearson_r_q25",
                "pearson_r_q75": "pearson_r_q75",
            }
        ),
        metric="pearson_r",
        ylabel="Pearson r",
    )
    axes[0].set_title("Per-circuit Pearson r")

    _plot_metric_trend(
        axes[1],
        circuit_df=circuit_df,
        aggregate_df=aggregate_df.rename(
            columns={
                "spearman_rho_median": "spearman_rho_median",
                "spearman_rho_q25": "spearman_rho_q25",
                "spearman_rho_q75": "spearman_rho_q75",
            }
        ),
        metric="spearman_rho",
        ylabel="Spearman rho",
    )
    axes[1].set_title("Per-circuit Spearman rho")

    code_counts = sorted(circuit_df["codes_used"].unique())
    fig.suptitle(
        f"LSR-TVD correlation across {cfg.num_circuits} random circuits "
        f"(q={cfg.num_qubits}, gates={cfg.num_gates}, codes={code_counts[0]}-{code_counts[-1]})"
    )
    fig.tight_layout()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.figure_path, dpi=240, bbox_inches="tight")
    print(f"Wrote: {cfg.figure_path}")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_experiment(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float | int]] = []
    code_counts = _code_counts(cfg)

    progress = tqdm(range(cfg.num_circuits), desc="circuits")
    for circuit_index in progress:
        circuit, circuit_seed = _build_random_circuit(cfg, circuit_index)
        pipeline_cfg = _build_pipeline_config(cfg, circuit_seed)

        for code_count in code_counts:
            permutations = _generate_fixed_code_permutations(
                num_qubits=cfg.num_qubits,
                target_codes=code_count,
                count=cfg.permutations_per_circuit,
                seed=cfg.seed * 1_000_003 + circuit_index * 10_000 + code_count * 101,
            )
            record = run_specific_scan(
                program_circuit=circuit,
                permutations=permutations,
                config=pipeline_cfg,
            )
            df = _prep_df(record)
            df = df[df["codes_used"] == code_count].reset_index(drop=True)
            rows.append(
                _summarize_circuit_bucket(
                    df,
                    circuit_index=circuit_index,
                    circuit_seed=circuit_seed,
                    target_codes=code_count,
                )
            )

    if not rows:
        raise RuntimeError("No correlation summaries were produced.")

    circuit_df = pd.DataFrame(rows).sort_values(["codes_used", "circuit_index"]).reset_index(drop=True)
    aggregate_df = _build_aggregate_summary(circuit_df)
    return circuit_df, aggregate_df


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Measure per-circuit LSR/TVD correlations across fixed code counts."
    )
    parser.add_argument("--num-qubits", type=int, default=ExperimentConfig.num_qubits)
    parser.add_argument("--num-gates", type=int, default=ExperimentConfig.num_gates)
    parser.add_argument("--num-circuits", type=int, default=ExperimentConfig.num_circuits)
    parser.add_argument(
        "--permutations-per-circuit",
        type=int,
        default=ExperimentConfig.permutations_per_circuit,
    )
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--idle-scalar", type=float, default=ExperimentConfig.idle_scalar)
    parser.add_argument("--min-codes", type=int, default=None)
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        num_qubits=args.num_qubits,
        num_gates=args.num_gates,
        num_circuits=args.num_circuits,
        permutations_per_circuit=args.permutations_per_circuit,
        repetitions=args.repetitions,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        idle_scalar=args.idle_scalar,
        min_codes=args.min_codes,
        max_codes=args.max_codes,
        output_dir=args.output_dir,
        show_plot=args.show_plot,
    )


def main() -> int:
    cfg = parse_args()
    circuit_df, aggregate_df = run_experiment(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    circuit_df.to_csv(cfg.circuit_csv_path, index=False)
    aggregate_df.to_csv(cfg.aggregate_csv_path, index=False)
    print(f"Wrote: {cfg.circuit_csv_path}")
    print(f"Wrote: {cfg.aggregate_csv_path}")

    _plot_results(circuit_df, aggregate_df, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
