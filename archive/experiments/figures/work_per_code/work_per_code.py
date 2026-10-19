"""
Analyze how consistently work-per-code predicts logical TVD across circuits.

The main question is not whether the metric correlates for one scan, but
whether it remains useful when averaged across many random circuits. For each
circuit this script:

1. Runs a random permutation scan.
2. Computes work_per_code = (w1 * #1q + w2 * #2q) / codes_used.
3. Measures overall TVD-vs-work correlation for that circuit.
4. Repeats the analysis within each codes_used bucket.
5. Checks how good the minimum-work permutation is as a proxy for the
   minimum-TVD permutation.

Outputs:
  * Per-circuit summary CSV.
  * Per-circuit/per-bucket summary CSV.
  * Aggregate summary CSV.
  * A figure summarizing consistency across circuits.
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
from pipeline import PipelineConfig, run_random_scan
from simulator.config_presets import ScaledSimulator
from translator.config import TranslatorConfig

GATE_SET = ("h", "x", "z", "cx", "cz")
ACCENT = "#30444f"
BLUE = "#2c7fb8"
TEAL = "#41b6c4"
GOLD = "#f1a340"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

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
    num_qubits: int = 10
    num_gates: int = 50
    num_circuits: int = 5
    permutations: int = 100
    repetitions: int = 100_000
    seed: int = 11
    distance_scalar: float = 1.0
    idle_time_scalar: float = 1.0
    w1: float = 1.0
    w2: float = 13.0
    min_bucket_size: int = 4
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    show_plot: bool = False

    @property
    def figure_path(self) -> Path:
        return self.output_dir / "work_per_code_consistency.svg"

    @property
    def circuit_csv_path(self) -> Path:
        return self.output_dir / "work_per_code_circuit_summary.csv"

    @property
    def bucket_csv_path(self) -> Path:
        return self.output_dir / "work_per_code_bucket_summary.csv"

    @property
    def aggregate_csv_path(self) -> Path:
        return self.output_dir / "work_per_code_aggregate_summary.csv"


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
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
        }

    if np.isclose(np.std(x_arr), 0.0) or np.isclose(np.std(y_arr), 0.0):
        return {
            "n": float(len(x_arr)),
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
        }

    pear = pearsonr(x_arr, y_arr)
    spear = spearmanr(x_arr, y_arr)
    return {
        "n": float(len(x_arr)),
        "pearson": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


def _rank_percentile(values: pd.Series, candidate: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or not math.isfinite(candidate):
        return float("nan")
    return 100.0 * float(np.mean(arr <= candidate))


def _pct_regret(best_value: float, selected_value: float) -> float:
    if not math.isfinite(best_value) or not math.isfinite(selected_value):
        return float("nan")
    if best_value <= 0:
        return 0.0
    return 100.0 * (selected_value - best_value) / best_value


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


def _build_pipeline_config(cfg: ExperimentConfig, circuit_seed: int) -> PipelineConfig:
    sim_cfg = (
        ScaledSimulator(repetitions=cfg.repetitions, seed=circuit_seed, alpha=1.0)
        .with_stim()
        # .with_distance_noise(distance_scalar=cfg.distance_scalar)
        .with_idle_noise(idle_time_scalar=cfg.idle_time_scalar)
    )
    translator_cfg = TranslatorConfig(enable_h_commutation=False)
    return PipelineConfig(
        sim_config=sim_cfg,
        translator_config=translator_cfg,
        verbose=False,
        save=False,
        seed=circuit_seed,
    )


def _prep_df(record, *, w1: float, w2: float) -> pd.DataFrame:
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")

    df = record.dataframe.df.copy()
    required = {
        "logical_total_variation_distance",
        "codes_used",
        "one_qubit_gates",
        "two_qubit_gates",
        "total_gates",
        "circuit_depth",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Scan dataframe missing columns: {sorted(missing)}")

    for col in required:
        df[col] = _pd_to_numeric(df[col])

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df = df[(df["codes_used"] > 0) & (df["total_gates"] > 0)].reset_index(drop=True)
    df["work_per_code"] = (
        w1 * df["one_qubit_gates"] + w2 * df["two_qubit_gates"]
    ) / df["codes_used"]
    return df


def _summarize_bucket(
    bucket: pd.DataFrame,
    *,
    circuit_index: int,
    circuit_seed: int,
    min_bucket_size: int,
) -> Dict[str, float | int]:
    row: Dict[str, float | int] = {
        "circuit_index": circuit_index,
        "circuit_seed": circuit_seed,
        "codes_used": int(bucket["codes_used"].iloc[0]),
        "n_permutations": int(len(bucket)),
    }

    corr = _corr_summary(
        bucket["work_per_code"],
        bucket["logical_total_variation_distance"],
    )
    row.update(
        {
            "pearson": corr["pearson"],
            "pearson_p": corr["pearson_p"],
            "spearman": corr["spearman"],
            "spearman_p": corr["spearman_p"],
            "meets_min_bucket_size": int(len(bucket) >= min_bucket_size),
        }
    )

    work_pick_idx = bucket["work_per_code"].idxmin()
    tvd_best_idx = bucket["logical_total_variation_distance"].idxmin()
    work_pick_tvd = float(bucket.loc[work_pick_idx, "logical_total_variation_distance"])
    best_tvd = float(bucket.loc[tvd_best_idx, "logical_total_variation_distance"])
    row.update(
        {
            "min_work_per_code": float(bucket.loc[work_pick_idx, "work_per_code"]),
            "selected_tvd_from_min_work": work_pick_tvd,
            "best_tvd": best_tvd,
            "tvd_regret": work_pick_tvd - best_tvd,
            "tvd_regret_pct": _pct_regret(best_tvd, work_pick_tvd),
            "selected_tvd_percentile": _rank_percentile(
                bucket["logical_total_variation_distance"], work_pick_tvd
            ),
        }
    )
    return row


def _summarize_circuit(
    df: pd.DataFrame,
    *,
    circuit_index: int,
    circuit_seed: int,
    cfg: ExperimentConfig,
) -> tuple[Dict[str, float | int], List[Dict[str, float | int]]]:
    corr = _corr_summary(df["work_per_code"], df["logical_total_variation_distance"])
    work_pick_idx = df["work_per_code"].idxmin()
    tvd_best_idx = df["logical_total_variation_distance"].idxmin()
    work_pick_tvd = float(df.loc[work_pick_idx, "logical_total_variation_distance"])
    best_tvd = float(df.loc[tvd_best_idx, "logical_total_variation_distance"])

    bucket_rows: List[Dict[str, float | int]] = []
    for codes_used, bucket in df.groupby("codes_used"):
        if len(bucket) < 2:
            continue
        bucket_rows.append(
            _summarize_bucket(
                bucket,
                circuit_index=circuit_index,
                circuit_seed=circuit_seed,
                min_bucket_size=cfg.min_bucket_size,
            )
        )

    bucket_df = pd.DataFrame(bucket_rows)
    eligible_bucket_df = (
        bucket_df.loc[bucket_df["meets_min_bucket_size"] == 1].copy()
        if not bucket_df.empty
        else pd.DataFrame()
    )

    summary: Dict[str, float | int] = {
        "circuit_index": circuit_index,
        "circuit_seed": circuit_seed,
        "n_permutations": int(len(df)),
        "distinct_codes_used": int(df["codes_used"].nunique()),
        "overall_pearson": corr["pearson"],
        "overall_pearson_p": corr["pearson_p"],
        "overall_spearman": corr["spearman"],
        "overall_spearman_p": corr["spearman_p"],
        "min_work_per_code": float(df.loc[work_pick_idx, "work_per_code"]),
        "selected_tvd_from_min_work": work_pick_tvd,
        "best_tvd": best_tvd,
        "tvd_regret": work_pick_tvd - best_tvd,
        "tvd_regret_pct": _pct_regret(best_tvd, work_pick_tvd),
        "selected_tvd_percentile": _rank_percentile(
            df["logical_total_variation_distance"], work_pick_tvd
        ),
        "median_codes_used": float(df["codes_used"].median()),
    }

    if eligible_bucket_df.empty:
        summary["eligible_bucket_count"] = 0
        summary["bucket_spearman_median"] = float("nan")
        summary["bucket_spearman_mean"] = float("nan")
        summary["bucket_percentile_median"] = float("nan")
        summary["bucket_regret_pct_median"] = float("nan")
    else:
        summary["eligible_bucket_count"] = int(len(eligible_bucket_df))
        summary["bucket_spearman_median"] = float(
            eligible_bucket_df["spearman"].median(skipna=True)
        )
        summary["bucket_spearman_mean"] = float(
            eligible_bucket_df["spearman"].mean(skipna=True)
        )
        summary["bucket_percentile_median"] = float(
            eligible_bucket_df["selected_tvd_percentile"].median(skipna=True)
        )
        summary["bucket_regret_pct_median"] = float(
            eligible_bucket_df["tvd_regret_pct"].median(skipna=True)
        )

    return summary, bucket_rows


def run_experiment(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    circuit_rows: List[Dict[str, float | int]] = []
    bucket_rows: List[Dict[str, float | int]] = []

    progress = tqdm(range(cfg.num_circuits), desc="work_per_code circuits")
    for circuit_index in progress:
        circuit, circuit_seed = _build_random_circuit(cfg, circuit_index)
        pipeline_cfg = _build_pipeline_config(cfg, circuit_seed)
        record = run_random_scan(
            program_circuit=circuit,
            m=cfg.permutations,
            config=pipeline_cfg,
        )
        df = _prep_df(record, w1=cfg.w1, w2=cfg.w2)
        if df.empty:
            continue

        circuit_summary, circuit_bucket_rows = _summarize_circuit(
            df,
            circuit_index=circuit_index,
            circuit_seed=circuit_seed,
            cfg=cfg,
        )
        circuit_rows.append(circuit_summary)
        bucket_rows.extend(circuit_bucket_rows)

        progress.set_postfix(
            overall_spearman=f"{circuit_summary['overall_spearman']:.3f}",
            tvd_pctile=f"{circuit_summary['selected_tvd_percentile']:.1f}",
        )
        print(
            "[work_per_code] "
            f"circuit={circuit_index} "
            f"n={len(df)} "
            f"overall_spearman={circuit_summary['overall_spearman']:.3f} "
            f"selected_tvd_percentile={circuit_summary['selected_tvd_percentile']:.1f}"
        )

    if not circuit_rows:
        raise RuntimeError("No circuit summaries were produced.")

    return pd.DataFrame(circuit_rows), pd.DataFrame(bucket_rows)


def build_aggregate_summary(
    circuit_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
) -> pd.DataFrame:
    bucket_eligible = (
        bucket_df.loc[bucket_df["meets_min_bucket_size"] == 1].copy()
        if not bucket_df.empty
        else pd.DataFrame()
    )
    row: Dict[str, float | int] = {
        "num_circuits": int(len(circuit_df)),
        "overall_spearman_mean": float(circuit_df["overall_spearman"].mean(skipna=True)),
        "overall_spearman_median": float(circuit_df["overall_spearman"].median(skipna=True)),
        "overall_pearson_mean": float(circuit_df["overall_pearson"].mean(skipna=True)),
        "overall_selected_tvd_percentile_mean": float(
            circuit_df["selected_tvd_percentile"].mean(skipna=True)
        ),
        "overall_selected_tvd_percentile_median": float(
            circuit_df["selected_tvd_percentile"].median(skipna=True)
        ),
        "overall_tvd_regret_pct_mean": float(circuit_df["tvd_regret_pct"].mean(skipna=True)),
        "overall_tvd_regret_pct_median": float(
            circuit_df["tvd_regret_pct"].median(skipna=True)
        ),
    }

    if bucket_eligible.empty:
        row["eligible_bucket_rows"] = 0
        row["bucket_spearman_mean"] = float("nan")
        row["bucket_spearman_median"] = float("nan")
        row["bucket_selected_tvd_percentile_mean"] = float("nan")
        row["bucket_selected_tvd_percentile_median"] = float("nan")
        row["bucket_tvd_regret_pct_mean"] = float("nan")
    else:
        row["eligible_bucket_rows"] = int(len(bucket_eligible))
        row["bucket_spearman_mean"] = float(bucket_eligible["spearman"].mean(skipna=True))
        row["bucket_spearman_median"] = float(bucket_eligible["spearman"].median(skipna=True))
        row["bucket_selected_tvd_percentile_mean"] = float(
            bucket_eligible["selected_tvd_percentile"].mean(skipna=True)
        )
        row["bucket_selected_tvd_percentile_median"] = float(
            bucket_eligible["selected_tvd_percentile"].median(skipna=True)
        )
        row["bucket_tvd_regret_pct_mean"] = float(
            bucket_eligible["tvd_regret_pct"].mean(skipna=True)
        )

    return pd.DataFrame([row])


def plot_results(
    circuit_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> None:
    fig, axes = plt.subplots(ncols=3, figsize=(15.5, 4.8), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

    axes[0].boxplot(
        [
            circuit_df["bucket_spearman_median"].dropna().to_numpy(),
            circuit_df["overall_spearman"].dropna().to_numpy(),
        ],
        tick_labels=["Within code bucket", "Overall"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.8},
        boxprops={"facecolor": BLUE, "alpha": 0.72, "edgecolor": ACCENT},
    )
    axes[0].axhline(0.0, color=ACCENT, linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("Spearman rho")
    axes[0].set_title("Prediction strength across circuits")

    axes[1].boxplot(
        [
            circuit_df["bucket_percentile_median"].dropna().to_numpy(),
            bucket_df.loc[
                bucket_df["meets_min_bucket_size"] == 1,
                "selected_tvd_percentile",
            ].dropna().to_numpy(),
        ],
        tick_labels=["Circuit median", "All code buckets"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.8},
        boxprops={"facecolor": TEAL, "alpha": 0.72, "edgecolor": ACCENT},
    )
    axes[1].axhline(50.0, color=ACCENT, linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel("Within-bucket TVD percentile of min-work pick")
    axes[1].set_title("Min-work pick quality within fixed code count")

    bucket_eligible = (
        bucket_df.loc[bucket_df["meets_min_bucket_size"] == 1].copy()
        if not bucket_df.empty
        else pd.DataFrame()
    )
    if not bucket_eligible.empty:
        grouped = (
            bucket_eligible.groupby("codes_used")
            .agg(
                spearman_median=("spearman", "median"),
                spearman_q1=("spearman", lambda s: s.quantile(0.25)),
                spearman_q3=("spearman", lambda s: s.quantile(0.75)),
                percentile_median=("selected_tvd_percentile", "median"),
            )
            .reset_index()
            .sort_values("codes_used")
        )
        x = grouped["codes_used"].to_numpy(dtype=float)
        axes[2].fill_between(
            x,
            grouped["spearman_q1"].to_numpy(dtype=float),
            grouped["spearman_q3"].to_numpy(dtype=float),
            color=GOLD,
            alpha=0.25,
        )
        axes[2].plot(
            x,
            grouped["spearman_median"].to_numpy(dtype=float),
            color=GOLD,
            marker="o",
            linewidth=2.0,
            label="Median bucket Spearman",
        )
        axes[2].axhline(0.0, color=ACCENT, linestyle="--", linewidth=1.0, alpha=0.6)
        axes[2].set_xlabel("Codes used")
        axes[2].set_ylabel("Median bucket Spearman rho")
        axes[2].set_title("Within-code predictiveness by codes_used")

        right_ax = axes[2].twinx()
        right_ax.plot(
            x,
            grouped["percentile_median"].to_numpy(dtype=float),
            color=ACCENT,
            marker="s",
            linewidth=1.5,
            alpha=0.75,
            label="Median selected TVD percentile",
        )
        right_ax.set_ylabel("Median TVD percentile")
        right_ax.tick_params(axis="y", colors=ACCENT)
        right_ax.spines["top"].set_visible(False)

        legend_lines = [axes[2].lines[-1], right_ax.lines[-1]]
        legend_labels = [line.get_label() for line in legend_lines]
        axes[2].legend(
            legend_lines,
            legend_labels,
            loc="upper left",
            frameon=False,
            fontsize=10,
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "No eligible code buckets",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_axis_off()

    fig.suptitle(
        (
            f"Work-per-code is strongest within fixed code counts over {len(circuit_df)} random circuits "
            f"(q={cfg.num_qubits}, gates={cfg.num_gates}, perms={cfg.permutations})"
        ),
        y=1.03,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.figure_path, dpi=240, bbox_inches="tight")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args(argv: List[str] | None = None) -> ExperimentConfig:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Analyze whether work_per_code consistently predicts logical TVD."
    )
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits)
    parser.add_argument("--num-gates", type=int, default=defaults.num_gates)
    parser.add_argument("--num-circuits", type=int, default=defaults.num_circuits)
    parser.add_argument("--permutations", type=int, default=defaults.permutations)
    parser.add_argument("--repetitions", type=int, default=defaults.repetitions)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--distance-scalar", type=float, default=defaults.distance_scalar)
    parser.add_argument("--idle-time-scalar", type=float, default=defaults.idle_time_scalar)
    parser.add_argument("--w1", type=float, default=defaults.w1)
    parser.add_argument("--w2", type=float, default=defaults.w2)
    parser.add_argument("--min-bucket-size", type=int, default=defaults.min_bucket_size)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args(argv)

    return ExperimentConfig(
        num_qubits=args.num_qubits,
        num_gates=args.num_gates,
        num_circuits=args.num_circuits,
        permutations=args.permutations,
        repetitions=args.repetitions,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        idle_time_scalar=args.idle_time_scalar,
        w1=args.w1,
        w2=args.w2,
        min_bucket_size=args.min_bucket_size,
        output_dir=args.output_dir,
        show_plot=args.show_plot,
    )


def main(argv: List[str] | None = None) -> int:
    cfg = parse_args(argv)
    circuit_df, bucket_df = run_experiment(cfg)
    aggregate_df = build_aggregate_summary(circuit_df, bucket_df)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    circuit_df.to_csv(cfg.circuit_csv_path, index=False)
    bucket_df.to_csv(cfg.bucket_csv_path, index=False)
    aggregate_df.to_csv(cfg.aggregate_csv_path, index=False)
    plot_results(circuit_df, bucket_df, cfg)

    print(f"[work_per_code] Wrote {cfg.circuit_csv_path}")
    print(f"[work_per_code] Wrote {cfg.bucket_csv_path}")
    print(f"[work_per_code] Wrote {cfg.aggregate_csv_path}")
    print(f"[work_per_code] Wrote {cfg.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
