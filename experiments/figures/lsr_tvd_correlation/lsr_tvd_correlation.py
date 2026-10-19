"""
experiments/figures/lsr_tvd_correlation/lsr_tvd_correlation.py

Scatter plot of logical TVD vs logical selection rate (LSR) for random
permutations from a single random circuit.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from translator.config import TranslatorConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from experiments.shared.benchmark_circuits import load_proxy_circuits
from pipeline import PipelineConfig, run_random_scan
from simulator.config_presets import ScaledSimulator

POINT_COLOR = "#005a96"
ACCENT_COLOR = "#3b4c56"

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 16,
        "text.color": ACCENT_COLOR,
        "axes.labelcolor": ACCENT_COLOR,
        "axes.edgecolor": ACCENT_COLOR,
        "xtick.color": ACCENT_COLOR,
        "ytick.color": ACCENT_COLOR,
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark_root: Path = REPO_ROOT / "benchmarks" / "qasmbench" / "small" / "qec_en_n5"
    benchmark_name: str = "qec_en_n5"
    proxy_timeout_seconds: float | None = 1.0
    replace_rz_rx_with_s: bool = False
    permutations: int = 100
    repetitions: int = 100_000
    seed: int = 11
    distance_scalar: float = 1.0
    idle_scalar: float = 1.0
    output_dir: Path = REPO_ROOT / "experiments" / "figures" / "lsr_tvd_correlation"


def _load_proxy_circuit(cfg: ExperimentConfig):
    timeout_seconds = cfg.proxy_timeout_seconds
    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None

    circuits = load_proxy_circuits(
        cfg.benchmark_root,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=cfg.replace_rz_rx_with_s,
    )
    if not circuits:
        raise ValueError(f"No proxy circuits loaded from {cfg.benchmark_root}")

    for name, circuit in circuits:
        if name == cfg.benchmark_name or name.lower() == cfg.benchmark_name.lower():
            break
    else:
        name, circuit = circuits[0]

    circuit.measure_all()
    return name, circuit


def _pd_to_numeric(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _prep_df(record):
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")
    df = record.dataframe.df.copy()

    required = {
        "logical_selection_rate",
        "logical_total_variation_distance",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Scan dataframe missing required columns: " + ", ".join(sorted(missing))
        )

    df["logical_selection_rate"] = _pd_to_numeric(df["logical_selection_rate"])
    df["logical_total_variation_distance"] = _pd_to_numeric(
        df["logical_total_variation_distance"]
    )
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    return df


def _scatter_with_fit(
    *,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    ax.set_facecolor("white")

    ax.scatter(
        x,
        y,
        s=28,
        alpha=0.78,
        color=POINT_COLOR,
        edgecolor="none",
    )

    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ys = slope * xs + intercept
        ax.plot(xs, ys, linewidth=2.2, color=ACCENT_COLOR, alpha=0.95)

        if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
            pearson_text = "n/a"
            spearman_text = "n/a"
        else:
            pearson_text = f"{pearsonr(x, y).statistic:.3f}"
            spearman_text = f"{spearmanr(x, y).statistic:.3f}"
        ax.text(
            0.02,
            0.98,
            (
                f"n = {len(x)}\n"
                f"Pearson r = {pearson_text}\n"
                f"Spearman rho = {spearman_text}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=ACCENT_COLOR,
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": ACCENT_COLOR,
                "alpha": 0.92,
            },
        )

    ax.set_xlabel("Logical selection rate (LSR)")
    ax.set_ylabel("Logical total variation distance (TVD)")
    ax.set_title("LSR vs TVD across random permutations")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        output_path,
        format=output_path.suffix.lstrip("."),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlation between LSR and TVD.")
    parser.add_argument("--benchmark-root", type=Path, default=ExperimentConfig.benchmark_root)
    parser.add_argument("--benchmark-name", type=str, default=ExperimentConfig.benchmark_name)
    parser.add_argument("--proxy-timeout", type=float, default=ExperimentConfig.proxy_timeout_seconds)
    parser.add_argument(
        "--keep-rz-rx",
        action="store_true",
        help="Keep rz/rx in the loaded proxy circuit instead of mapping them to s.",
    )
    parser.add_argument("--permutations", type=int, default=ExperimentConfig.permutations)
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        benchmark_root=args.benchmark_root,
        benchmark_name=args.benchmark_name,
        proxy_timeout_seconds=args.proxy_timeout,
        replace_rz_rx_with_s=not args.keep_rz_rx,
        permutations=args.permutations,
        repetitions=args.repetitions,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        output_dir=args.output_dir,
    )

    benchmark_name, circuit = _load_proxy_circuit(cfg)
    print(circuit)
    sim_cfg = (
        ScaledSimulator(alpha=0.1, repetitions=cfg.repetitions, seed=cfg.seed)
        .with_distance_noise(distance_scalar=cfg.distance_scalar)
        .with_idle_noise(idle_time_scalar=cfg.idle_scalar)
        .with_stim()
    )
    pipeline_cfg = PipelineConfig(sim_config=sim_cfg, translator_config=TranslatorConfig(enable_h_commutation=False), verbose=True, save=False, seed=cfg.seed)

    record = run_random_scan(program_circuit=circuit, m=cfg.permutations, config=pipeline_cfg)
    df = _prep_df(record)

    x_lsr = df["logical_selection_rate"].to_numpy(dtype=float)
    y_tvd = df["logical_total_variation_distance"].to_numpy(dtype=float)

    base = (
        f"{benchmark_name}_m{cfg.permutations}_shots{cfg.repetitions}"
        f"_seed{cfg.seed}"
    )
    output_path = cfg.output_dir / f"{base}__lsr_vs_tvd.svg"
    _scatter_with_fit(x=x_lsr, y=y_tvd, output_path=output_path)
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
