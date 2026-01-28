"""
experiments/tvd_vs_gate_count.py

Scan random permutations for a random Clifford circuit (H, X, Z, S, CX, CZ)
using the Stim + distance-based-noise simulator. Produces two figures:
  1) Logical TVD vs total gate count of the translated (physical) circuit.
  2) Logical TVD vs (total gates / codes_used).

Usage:
  python -m experiments.tvd_vs_gate_count --qubits 10 --num-gates 200 --permutations 200
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
    num_gates: int = 100
    permutations: int = 250
    repetitions: int = 250_000
    seed: int = 11
    distance_scalar: float = 1.0
    output_dir: Path = REPO_ROOT / "results" / "experiments" / "tvd_vs_gate_count"


def _build_random_circuit(cfg: ExperimentConfig):
    circuit = random_clifford_circuit(
        cfg.qubits,
        num_gates=cfg.num_gates,
        gates=["h", "x", "z", "s", "cx", "cz"],
        seed=cfg.seed,
    )
    circuit.measure_all()
    return circuit


def _prep_df(record):
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")
    df = record.dataframe.df.copy()

    required = {"logical_total_variation_distance", "total_gates", "codes_used", "circuit_depth"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Scan dataframe missing columns: {sorted(missing)}")

    df = df.dropna(
        subset=["logical_total_variation_distance", "total_gates", "codes_used", "circuit_depth"]
    ).reset_index(drop=True)

    df["total_gates"] = pd_to_numeric(df["total_gates"])
    df["codes_used"] = pd_to_numeric(df["codes_used"])
    df["circuit_depth"] = pd_to_numeric(df["circuit_depth"])
    df["logical_total_variation_distance"] = pd_to_numeric(
        df["logical_total_variation_distance"]
    )

    df = df[(df["codes_used"] > 0) & (df["total_gates"] >= 0) & (df["circuit_depth"] >= 0)].reset_index(drop=True)
    df["gates_per_code"] = df["total_gates"] / df["codes_used"]
    return df


def pd_to_numeric(series):
    # Avoid importing pandas at module import time (keeps CLI help snappy).
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


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
        sc = ax.scatter(x, y, c=color_values, s=22, alpha=0.75, cmap="viridis", edgecolor="none")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(color_label or "value")

    # Linear least squares fit (visual guide).
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
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlation between TVD and translated gate count.")
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

    x_gates = df["total_gates"].to_numpy(dtype=float)
    x_gates_per_code = df["gates_per_code"].to_numpy(dtype=float)
    x_depth = df["circuit_depth"].to_numpy(dtype=float)
    y_tvd = df["logical_total_variation_distance"].to_numpy(dtype=float)
    codes_used = df["codes_used"].to_numpy(dtype=float)

    pear_gates = pearsonr(x_gates, y_tvd) if len(df) >= 2 else (float("nan"), float("nan"))
    spear_gates = spearmanr(x_gates, y_tvd) if len(df) >= 2 else (float("nan"), float("nan"))
    pear_norm = pearsonr(x_gates_per_code, y_tvd) if len(df) >= 2 else (float("nan"), float("nan"))
    spear_norm = spearmanr(x_gates_per_code, y_tvd) if len(df) >= 2 else (float("nan"), float("nan"))
    pear_depth = pearsonr(x_depth, y_tvd) if len(df) >= 2 else (float("nan"), float("nan"))
    spear_depth = spearmanr(x_depth, y_tvd) if len(df) >= 2 else (float("nan"), float("nan"))

    print(
        f"Rows: {len(df)} | "
        f"pearson(total_gates, tvd)={pear_gates[0]:.3f} (p={pear_gates[1]:.2e}) | "
        f"spearman(total_gates, tvd)={spear_gates[0]:.3f} (p={spear_gates[1]:.2e})"
    )
    print(
        f"Rows: {len(df)} | "
        f"pearson(gates_per_code, tvd)={pear_norm[0]:.3f} (p={pear_norm[1]:.2e}) | "
        f"spearman(gates_per_code, tvd)={spear_norm[0]:.3f} (p={spear_norm[1]:.2e})"
    )
    print(
        f"Rows: {len(df)} | "
        f"pearson(circuit_depth, tvd)={pear_depth[0]:.3f} (p={pear_depth[1]:.2e}) | "
        f"spearman(circuit_depth, tvd)={spear_depth[0]:.3f} (p={spear_depth[1]:.2e})"
    )

    base = (
        f"q{cfg.qubits}_g{cfg.num_gates}_m{cfg.permutations}_shots{cfg.repetitions}_seed{cfg.seed}"
    )
    out1 = cfg.output_dir / f"{base}__tvd_vs_total_gates.svg"
    out2 = cfg.output_dir / f"{base}__tvd_vs_gates_per_code.svg"
    out3 = cfg.output_dir / f"{base}__tvd_vs_circuit_depth.svg"

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    _scatter_with_fit(
        x=x_gates,
        y=y_tvd,
        xlabel="Total gates in translated circuit",
        ylabel="Logical total variation distance (TVD)",
        title="TVD vs translated gate count",
        output_path=out1,
        color_values=codes_used,
        color_label="codes_used",
    )

    _scatter_with_fit(
        x=x_gates_per_code,
        y=y_tvd,
        xlabel="Total gates / codes_used",
        ylabel="Logical total variation distance (TVD)",
        title="TVD vs normalized gate count",
        output_path=out2,
        color_values=codes_used,
        color_label="codes_used",
    )

    _scatter_with_fit(
        x=x_depth,
        y=y_tvd,
        xlabel="Circuit depth (translated circuit)",
        ylabel="Logical total variation distance (TVD)",
        title="TVD vs circuit depth",
        output_path=out3,
        color_values=codes_used,
        color_label="codes_used",
    )

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
