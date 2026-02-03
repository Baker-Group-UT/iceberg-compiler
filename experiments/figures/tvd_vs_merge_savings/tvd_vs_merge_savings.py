"""
experiments/figures/tvd_vs_merge_savings/tvd_vs_merge_savings.py

Scatter plot of logical TVD vs merge savings (work units) for random permutations.

Usage:
  ./venv/bin/python -m experiments.figures.tvd_vs_merge_savings.tvd_vs_merge_savings \
    --qubits 10 --num-gates 200 --permutations 250
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from qiskit.circuit.random import random_clifford_circuit

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from pipeline import PipelineConfig, run_random_scan
from simulator.config_presets import ScaledSimulator
from translator.merging.conflicts import DEFAULT_MERGE_WORK_WEIGHTS


@dataclass(frozen=True)
class ExperimentConfig:
    qubits: int = 10
    num_gates: int = 200
    permutations: int = 250
    repetitions: int = 250_000
    seed: int = 11
    distance_scalar: float = 1.0
    output_dir: Path = REPO_ROOT / "experiments" / "figures" / "tvd_vs_merge_savings"


def _build_random_circuit(cfg: ExperimentConfig):
    circuit = random_clifford_circuit(
        cfg.qubits,
        num_gates=cfg.num_gates,
        gates=["h", "x", "z", "s", "cx", "cz"],
        seed=cfg.seed,
    )
    circuit.measure_all()
    return circuit


def _pd_to_numeric(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


def _prep_df(record):
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")
    df = record.dataframe.df.copy()

    required = {
        "logical_total_variation_distance",
        "codes_used",
        "merge_savings_weighted",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Scan dataframe missing columns (did you run after adding merge savings?): "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df["logical_total_variation_distance"] = _pd_to_numeric(
        df["logical_total_variation_distance"]
    )
    df["codes_used"] = _pd_to_numeric(df["codes_used"])
    df["merge_savings_weighted"] = _pd_to_numeric(df["merge_savings_weighted"])

    df = df[(df["codes_used"] > 0)].reset_index(drop=True)
    return df


def _scatter_with_fit(
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    color_values: np.ndarray | None = None,
    color_label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    ax.set_facecolor("white")

    if color_values is None:
        ax.scatter(x, y, s=22, alpha=0.65, edgecolor="none")
    else:
        sc = ax.scatter(
            x,
            y,
            c=color_values,
            s=22,
            alpha=0.75,
            cmap="viridis",
            edgecolor="none",
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(color_label or "value")

    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ys = slope * xs + intercept
        ax.plot(xs, ys, linewidth=2.0, color="black", alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        output_path,
        format=output_path.suffix.lstrip("."),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def _print_corr(label: str, x: np.ndarray, y: np.ndarray, n: int) -> None:
    if n < 2:
        print(f"{label}: n={n} (skipping correlations)")
        return
    pear = pearsonr(x, y)
    spear = spearmanr(x, y)
    print(
        f"{label}: n={n} | pearson={pear[0]:.3f} (p={pear[1]:.2e}) | spearman={spear[0]:.3f} (p={spear[1]:.2e})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlation between TVD and merge savings.")
    parser.add_argument("--qubits", type=int, default=ExperimentConfig.qubits)
    parser.add_argument("--num-gates", type=int, default=ExperimentConfig.num_gates)
    parser.add_argument("--permutations", type=int, default=ExperimentConfig.permutations)
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        qubits=args.qubits,
        num_gates=args.num_gates,
        permutations=args.permutations,
        repetitions=args.repetitions,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        output_dir=args.output_dir,
    )

    circuit = _build_random_circuit(cfg)
    sim_cfg = (
        ScaledSimulator(alpha=0.5, repetitions=cfg.repetitions, seed=cfg.seed)
        .with_distance_noise(distance_scalar=cfg.distance_scalar)
        .with_stim()
    )
    pipeline_cfg = PipelineConfig(sim_config=sim_cfg, verbose=False, save=False, seed=cfg.seed)

    record = run_random_scan(program_circuit=circuit, m=cfg.permutations, config=pipeline_cfg)
    df = _prep_df(record)

    y_tvd = df["logical_total_variation_distance"].to_numpy(dtype=float)
    x_merge = df["merge_savings_weighted"].to_numpy(dtype=float)
    codes_used = df["codes_used"].to_numpy(dtype=float)

    n = len(df)
    _print_corr("tvd vs merge_savings_weighted", x_merge, y_tvd, n)

    w1, w2 = DEFAULT_MERGE_WORK_WEIGHTS
    base = (
        f"q{cfg.qubits}_g{cfg.num_gates}_m{cfg.permutations}_shots{cfg.repetitions}_seed{cfg.seed}"
        f"__w1{w1:g}_w2{w2:g}"
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    out1 = cfg.output_dir / f"{base}__tvd_vs_merge_savings_weighted.svg"
    _scatter_with_fit(
        x=x_merge,
        y=y_tvd,
        xlabel=f"Merge savings (weighted work, w1={w1:g}, w2={w2:g})",
        ylabel="Logical total variation distance (TVD)",
        title="TVD vs merge savings (weighted)",
        output_path=out1,
        color_values=codes_used,
        color_label="codes_used",
    )
    print(f"Wrote: {out1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
