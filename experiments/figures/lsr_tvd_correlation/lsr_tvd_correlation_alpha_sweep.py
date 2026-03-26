"""
Quantify how strongly LSR correlates with TVD as the noise scale alpha changes.

For each alpha value and each random circuit, this script:

1. Builds a random circuit with 12 logical qubits and 10 gates per qubit.
2. Fixes the code count to ceil(0.75 * 12) = 9.
3. Generates permutations using exactly that many codes.
4. Runs ``run_specific_scan`` and computes per-circuit Pearson r and Spearman rho.

Outputs:
  * Per-circuit correlation CSV.
  * Aggregate-by-alpha CSV.
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
    num_qubits: int = 12
    num_gates_per_qubit: int = 10
    num_circuits: int = 12
    permutations_per_circuit: int = 20
    repetitions: int = 100_000
    alpha_min: float = 1e-2
    alpha_max: float = 1e2
    alpha_steps: int = 10
    seed: int = 11
    distance_scalar: float = 1.0
    idle_scalar: float = 1.0
    from_csv: bool = False
    resume: bool = False
    output_dir: Path = field(default_factory=lambda: THIS_DIR)
    show_plot: bool = False

    @property
    def target_codes(self) -> int:
        return math.ceil(0.75 * self.num_qubits)

    @property
    def num_gates(self) -> int:
        return self.num_qubits * self.num_gates_per_qubit

    @property
    def circuit_csv_path(self) -> Path:
        return self.output_dir / "lsr_tvd_correlation_alpha_sweep_circuit_summary.csv"

    @property
    def aggregate_csv_path(self) -> Path:
        return self.output_dir / "lsr_tvd_correlation_alpha_sweep_aggregate_summary.csv"

    @property
    def figure_path(self) -> Path:
        return self.output_dir / "lsr_tvd_correlation_alpha_sweep_trends.svg"


def _alpha_values(cfg: ExperimentConfig) -> list[float]:
    if cfg.alpha_min <= 0 or cfg.alpha_max <= 0:
        raise ValueError("Alpha bounds must be positive.")
    if cfg.alpha_max < cfg.alpha_min:
        raise ValueError("alpha_max must be >= alpha_min.")
    if cfg.alpha_steps < 2:
        raise ValueError("alpha_steps must be at least 2.")
    return [float(x) for x in np.logspace(np.log10(cfg.alpha_min), np.log10(cfg.alpha_max), cfg.alpha_steps)]


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.6g}"


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


def _build_pipeline_config(
    cfg: ExperimentConfig,
    *,
    seed: int,
    alpha: float,
) -> PipelineConfig:
    sim_cfg = (
        ScaledSimulator(alpha=alpha, repetitions=cfg.repetitions, seed=seed)
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


def _summarize_circuit_run(
    df: pd.DataFrame,
    *,
    alpha: float,
    codes_used: int,
    circuit_index: int,
    circuit_seed: int,
    num_qubits: int,
    num_gates: int,
    repetitions: int,
) -> Dict[str, float | int | str]:
    summary: Dict[str, float | int | str] = {
        "alpha": float(alpha),
        "alpha_label": _alpha_label(alpha),
        "repetitions": repetitions,
        "num_qubits": num_qubits,
        "num_gates": num_gates,
        "codes_used": codes_used,
        "circuit_index": circuit_index,
        "circuit_seed": circuit_seed,
        "n_points": int(len(df)),
        "mean_lsr": float(df["logical_selection_rate"].mean()) if not df.empty else float("nan"),
        "mean_tvd": float(df["logical_total_variation_distance"].mean()) if not df.empty else float("nan"),
    }
    summary.update(
        _corr_summary(df["logical_selection_rate"], df["logical_total_variation_distance"])
    )
    return summary


def _build_aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for alpha, bucket in df.groupby("alpha"):
        rows.append(
            {
                "alpha": float(alpha),
                "alpha_label": str(bucket["alpha_label"].iloc[0]),
                "repetitions": int(bucket["repetitions"].iloc[0]),
                "num_qubits": int(bucket["num_qubits"].iloc[0]),
                "num_gates": int(bucket["num_gates"].iloc[0]),
                "codes_used": int(bucket["codes_used"].iloc[0]),
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
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def _plot_metric_trend(
    ax,
    *,
    circuit_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    for _circuit_id, bucket in circuit_df.groupby("circuit_index"):
        bucket = bucket.sort_values("alpha")
        ax.plot(
            bucket["alpha"].to_numpy(dtype=float),
            bucket[metric].to_numpy(dtype=float),
            color=LIGHT_BLUE,
            linewidth=1.0,
            alpha=0.35,
        )

    x = aggregate_df["alpha"].to_numpy(dtype=float)
    median = aggregate_df[f"{metric}_median"].to_numpy(dtype=float)
    q25 = aggregate_df[f"{metric}_q25"].to_numpy(dtype=float)
    q75 = aggregate_df[f"{metric}_q75"].to_numpy(dtype=float)

    ax.fill_between(x, q25, q75, color=BLUE, alpha=0.18)
    ax.plot(x, median, color=BLUE, linewidth=2.5)
    ax.axhline(0.0, color="#7a7a7a", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Noise scale alpha")
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
        aggregate_df=aggregate_df,
        metric="pearson_r",
        ylabel="Pearson r",
    )
    axes[0].set_title("Per-circuit Pearson r")

    _plot_metric_trend(
        axes[1],
        circuit_df=circuit_df,
        aggregate_df=aggregate_df,
        metric="spearman_rho",
        ylabel="Spearman rho",
    )
    axes[1].set_title("Per-circuit Spearman rho")

    fig.suptitle(
        "LSR-TVD correlation vs noise scale alpha "
        f"(q={cfg.num_qubits}, gates={cfg.num_gates}, codes={cfg.target_codes})"
    )
    fig.tight_layout()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.figure_path, dpi=240, bbox_inches="tight")
    print(f"Wrote: {cfg.figure_path}")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)


def _load_results_from_csv(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.circuit_csv_path.exists():
        raise FileNotFoundError(f"Missing circuit summary CSV: {cfg.circuit_csv_path}")
    if not cfg.aggregate_csv_path.exists():
        raise FileNotFoundError(f"Missing aggregate summary CSV: {cfg.aggregate_csv_path}")

    circuit_df = pd.read_csv(cfg.circuit_csv_path)
    aggregate_df = pd.read_csv(cfg.aggregate_csv_path)
    return circuit_df, aggregate_df


def _load_existing_circuit_df(cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.circuit_csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(cfg.circuit_csv_path)


def _is_row_complete(row: pd.Series, expected_points: int) -> bool:
    n_points = pd.to_numeric(row.get("n_points"), errors="coerce")
    pearson_r = pd.to_numeric(row.get("pearson_r"), errors="coerce")
    spearman_rho = pd.to_numeric(row.get("spearman_rho"), errors="coerce")
    return (
        pd.notna(n_points)
        and int(n_points) >= expected_points
        and pd.notna(pearson_r)
        and pd.notna(spearman_rho)
    )


def run_experiment(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float | int | str]] = []
    alpha_values = _alpha_values(cfg)
    existing_df = _load_existing_circuit_df(cfg) if cfg.resume else pd.DataFrame()
    existing_map: dict[tuple[float, int], pd.Series] = {}
    if not existing_df.empty:
        for _, row in existing_df.iterrows():
            alpha = float(pd.to_numeric(row.get("alpha"), errors="coerce"))
            circuit_index = int(pd.to_numeric(row.get("circuit_index"), errors="coerce"))
            existing_map[(alpha, circuit_index)] = row

    progress = tqdm(alpha_values, desc="alpha values")
    for alpha in progress:
        for circuit_index in range(cfg.num_circuits):
            existing_row = existing_map.get((float(alpha), circuit_index))
            if existing_row is not None and _is_row_complete(
                existing_row, cfg.permutations_per_circuit
            ):
                rows.append(existing_row.to_dict())
                continue

            circuit, circuit_seed = _build_random_circuit(cfg, circuit_index)
            pipeline_cfg = _build_pipeline_config(
                cfg,
                seed=circuit_seed + int(round(alpha * 1000)),
                alpha=alpha,
            )
            permutations = _generate_fixed_code_permutations(
                num_qubits=cfg.num_qubits,
                target_codes=cfg.target_codes,
                count=cfg.permutations_per_circuit,
                seed=cfg.seed * 10_000_019 + int(round(alpha * 1_000_000)) + circuit_index * 503,
            )
            record = run_specific_scan(
                program_circuit=circuit,
                permutations=permutations,
                config=pipeline_cfg,
            )
            df = _prep_df(record)
            df = df[df["codes_used"] == cfg.target_codes].reset_index(drop=True)
            rows.append(
                _summarize_circuit_run(
                    df,
                    alpha=alpha,
                    codes_used=cfg.target_codes,
                    circuit_index=circuit_index,
                    circuit_seed=circuit_seed,
                    num_qubits=cfg.num_qubits,
                    num_gates=cfg.num_gates,
                    repetitions=cfg.repetitions,
                )
            )

    if not rows:
        raise RuntimeError("No correlation summaries were produced.")

    circuit_df = pd.DataFrame(rows).sort_values(["alpha", "circuit_index"]).reset_index(drop=True)
    aggregate_df = _build_aggregate_summary(circuit_df)
    return circuit_df, aggregate_df


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Measure per-circuit LSR/TVD correlations across noise scale alpha."
    )
    parser.add_argument("--num-circuits", type=int, default=ExperimentConfig.num_circuits)
    parser.add_argument(
        "--permutations-per-circuit",
        type=int,
        default=ExperimentConfig.permutations_per_circuit,
    )
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--alpha-min", type=float, default=ExperimentConfig.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=ExperimentConfig.alpha_max)
    parser.add_argument("--alpha-steps", type=int, default=ExperimentConfig.alpha_steps)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--idle-scalar", type=float, default=ExperimentConfig.idle_scalar)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing per-circuit CSV rows when they already contain complete "
            "results, and resimulate only missing or invalid rows."
        ),
    )
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        num_circuits=args.num_circuits,
        permutations_per_circuit=args.permutations_per_circuit,
        repetitions=args.repetitions,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_steps=args.alpha_steps,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        idle_scalar=args.idle_scalar,
        from_csv=args.from_csv,
        resume=args.resume,
        output_dir=args.output_dir,
        show_plot=args.show_plot,
    )


def main() -> int:
    cfg = parse_args()
    if cfg.from_csv:
        circuit_df, aggregate_df = _load_results_from_csv(cfg)
    else:
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
