"""
experiments/tvd_vs_work_depth.py

Runs a random permutation scan (Stim + distance-based noise) for a random
Clifford circuit (H, X, Z, S, CX, CZ), then plots TVD against a composite metric:

  score = work_per_code * (depth / total_gates)

where:
  work_per_code = (w1 * one_qubit_gates + w2 * two_qubit_gates) / codes_used

Usage:
  ./venv/bin/python -m experiments.tvd_vs_work_depth --qubits 10 --num-gates 200 --permutations 250
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from pipeline import PipelineConfig, run_random_scan
from simulator.config_presets import ScaledSimulator


@dataclass(frozen=True)
class ExperimentConfig:
    qubits: int = 10
    num_gates: int = 200
    permutations: int = 100
    repetitions: int = 50_000
    seed: int = 11
    distance_scalar: float = 1.0
    w1: float = 1.0
    w2: float = 13.0
    output_dir: Path = REPO_ROOT / "results" / "experiments" / "tvd_vs_work_depth"


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


def _prep_df(record, *, w1: float, w2: float):
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")
    df = record.dataframe.df.copy()

    required = {
        "logical_total_variation_distance",
        "codes_used",
        "one_qubit_gates",
        "two_qubit_gates",
        "circuit_depth",
        "total_gates",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Scan dataframe missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df["logical_total_variation_distance"] = _pd_to_numeric(df["logical_total_variation_distance"])
    df["codes_used"] = _pd_to_numeric(df["codes_used"])
    df["one_qubit_gates"] = _pd_to_numeric(df["one_qubit_gates"])
    df["two_qubit_gates"] = _pd_to_numeric(df["two_qubit_gates"])
    df["circuit_depth"] = _pd_to_numeric(df["circuit_depth"])
    df["total_gates"] = _pd_to_numeric(df["total_gates"])

    df = df[
        (df["codes_used"] > 0)
        & (df["total_gates"] > 0)
        & (df["circuit_depth"] >= 0)
    ].reset_index(drop=True)

    df["work_per_code"] = (w1 * df["one_qubit_gates"] + w2 * df["two_qubit_gates"]) / df["codes_used"]
    df["depth_over_total_gates"] = df["circuit_depth"] / df["total_gates"]
    df["work_depth_score"] = df["work_per_code"] * df["depth_over_total_gates"]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlation between TVD and work_per_code*(depth/total_gates).")
    parser.add_argument("--qubits", type=int, default=ExperimentConfig.qubits)
    parser.add_argument("--num-gates", type=int, default=ExperimentConfig.num_gates)
    parser.add_argument("--permutations", type=int, default=ExperimentConfig.permutations)
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--w1", type=float, default=ExperimentConfig.w1)
    parser.add_argument("--w2", type=float, default=ExperimentConfig.w2)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        qubits=args.qubits,
        num_gates=args.num_gates,
        permutations=args.permutations,
        repetitions=args.repetitions,
        seed=args.seed,
        distance_scalar=args.distance_scalar,
        w1=args.w1,
        w2=args.w2,
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
    df = _prep_df(record, w1=cfg.w1, w2=cfg.w2)

    x = df["work_depth_score"].to_numpy(dtype=float)
    x_work = df["work_per_code"].to_numpy(dtype=float)
    y = df["logical_total_variation_distance"].to_numpy(dtype=float)
    codes_used = df["codes_used"].to_numpy(dtype=float)

    pear = pearsonr(x, y) if len(df) >= 2 else (float("nan"), float("nan"))
    spear = spearmanr(x, y) if len(df) >= 2 else (float("nan"), float("nan"))
    pear_work = pearsonr(x_work, y) if len(df) >= 2 else (float("nan"), float("nan"))
    spear_work = spearmanr(x_work, y) if len(df) >= 2 else (float("nan"), float("nan"))

    print(
        f"Rows: {len(df)} | pearson(work_per_code, tvd)={pear_work[0]:.3f} (p={pear_work[1]:.2e}) | "
        f"spearman(work_per_code, tvd)={spear_work[0]:.3f} (p={spear_work[1]:.2e})"
    )
    print(
        f"Rows: {len(df)} | pearson(work_per_code*(depth/total_gates), tvd)={pear[0]:.3f} (p={pear[1]:.2e}) | "
        f"spearman(...)={spear[0]:.3f} (p={spear[1]:.2e})"
    )

    base = (
        f"q{cfg.qubits}_g{cfg.num_gates}_m{cfg.permutations}_shots{cfg.repetitions}_seed{cfg.seed}"
        f"__w1{cfg.w1:g}_w2{cfg.w2:g}"
    )
    out_work = cfg.output_dir / f"{base}__tvd_vs_work_per_code.svg"
    out = cfg.output_dir / f"{base}__tvd_vs_work_depth_score.svg"

    _scatter_with_fit(
        x=x_work,
        y=y,
        xlabel=f"(w1*#1q + w2*#2q)/codes_used   (w1={cfg.w1:g}, w2={cfg.w2:g})",
        ylabel="Logical total variation distance (TVD)",
        title="TVD vs work-per-code",
        output_path=out_work,
        color_values=codes_used,
        color_label="codes_used",
    )
    _scatter_with_fit(
        x=x,
        y=y,
        xlabel=f"((w1*#1q + w2*#2q)/codes_used) * (depth/total_gates)   (w1={cfg.w1:g}, w2={cfg.w2:g})",
        ylabel="Logical total variation distance (TVD)",
        title="TVD vs work×depth score",
        output_path=out,
        color_values=codes_used,
        color_label="codes_used",
    )
    print(f"Wrote: {out_work}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
