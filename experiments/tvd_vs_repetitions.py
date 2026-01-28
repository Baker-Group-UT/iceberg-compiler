"""
experiments/tvd_vs_repetitions.py

Measure how logical TVD settles as the number of repetitions increases for a
fixed set of permutations.

Usage:
  python -m experiments.tvd_vs_repetitions
  python -m experiments.tvd_vs_repetitions --repetitions 1000,2000,4000
  python -m experiments.tvd_vs_repetitions --perm-index 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from qiskit.circuit.random import random_clifford_circuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

from pipeline import PipelineConfig, run_specific_scan
from simulator.config_presets import ScaledSimulator

DEFAULT_REPETITIONS = [
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
    # 512_000,
    # 1_024_000,
    # 2_048_000,
    # 4_096_000,
]

DEFAULT_PERMS: List[List[Tuple[int, int]]] = [
    [(4, 1), (8, 1), (9, 1), (7, 1), (6, 1), (4, 0), (3, 0), (2, 0), (7, 0), (3, 1)],
]


@dataclass(frozen=True)
class ExperimentConfig:
    qubits: int = 10
    num_gates: int = 200
    repetitions: Sequence[int] = tuple(DEFAULT_REPETITIONS)
    seed: int = 11
    runs_per_rep: int = 8
    distance_scalar: float = 1.0
    output_dir: Path = REPO_ROOT / "results" / "experiments" / "tvd_vs_repetitions"


def _build_random_circuit(cfg: ExperimentConfig):
    circuit = random_clifford_circuit(
        cfg.qubits,
        num_gates=cfg.num_gates,
        gates=["h", "x", "z", "s", "cx", "cz"],
        seed=cfg.seed,
    )
    circuit.measure_all()
    return circuit


def _parse_repetitions(raw: str | None) -> List[int]:
    if raw is None or not raw.strip():
        return list(DEFAULT_REPETITIONS)
    items = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        items.append(int(item.replace("_", "")))
    if not items:
        raise ValueError("No repetitions parsed from --repetitions.")
    return items


def _perm_key(perm: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(tuple(p) for p in perm)


def _normalize_perm_key(value) -> Tuple[Tuple[int, int], ...]:
    if isinstance(value, tuple) and value and isinstance(value[0], tuple):
        return value
    if isinstance(value, list):
        return tuple(tuple(p) for p in value)
    raise TypeError(f"Unsupported perm_key type: {type(value)}")


def _extract_tvd_by_perm(df, perms: Sequence[Sequence[Tuple[int, int]]]) -> List[float]:
    if "logical_total_variation_distance" not in df.columns:
        raise KeyError("Missing logical_total_variation_distance column in dataframe.")
    if "perm_key" not in df.columns:
        raise KeyError("Missing perm_key column in dataframe.")

    df = df.copy()
    df["perm_key"] = df["perm_key"].apply(_normalize_perm_key)
    tvds = []
    for perm in perms:
        key = _perm_key(perm)
        matches = df["perm_key"].apply(lambda value: value == key)
        subset = df.loc[matches, "logical_total_variation_distance"]
        if subset.empty:
            raise KeyError(f"Permutation not found in dataframe: {perm}")
        tvd = float(subset.iloc[0])
        tvds.append(tvd)
    return tvds


def _plot(
    *,
    repetitions: Sequence[int],
    tvd_matrix: np.ndarray,
    perm_labels: Sequence[str],
    output_path: Path,
    include_mean: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    ax.set_facecolor("white")

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(perm_labels))))
    for idx, label in enumerate(perm_labels):
        ax.plot(
            repetitions,
            tvd_matrix[idx],
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            color=colors[idx % len(colors)],
            label=label,
        )

    if include_mean and len(perm_labels) > 1:
        mean = np.mean(tvd_matrix, axis=0)
        std = np.std(tvd_matrix, axis=0)
        ax.plot(repetitions, mean, color="black", linewidth=2.2, label="mean")
        ax.fill_between(repetitions, mean - std, mean + std, color="black", alpha=0.12)

    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)

    ax.set_xlabel("Repetitions (shots)")
    ax.set_ylabel("Logical total variation distance (TVD)")
    ax.set_title("TVD vs repetitions (fixed permutations)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot(
    *,
    repetitions: Sequence[int],
    tvd_runs: np.ndarray,
    perm_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    ax.set_facecolor("white")

    if tvd_runs.shape[0] != len(repetitions):
        raise ValueError("tvd_runs must have shape (len(repetitions), runs_per_rep).")
    data = [tvd_runs[idx] for idx in range(len(repetitions))]
    ax.boxplot(
        data,
        positions=list(repetitions),
        widths=[rep * 0.12 for rep in repetitions],
        showfliers=True,
    )

    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)

    ax.set_xlabel("Repetitions (shots)")
    ax.set_ylabel("Logical total variation distance (TVD)")
    ax.set_title(f"TVD variability vs repetitions ({perm_label})")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, axis="y")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), dpi=250, bbox_inches="tight")
    plt.close(fig)


def _write_csv(
    *,
    repetitions: Sequence[int],
    tvd_runs: np.ndarray,
    perm_labels: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repetitions", "perm_index", "perm_label", "run_index", "tvd"])
        for perm_idx, label in enumerate(perm_labels):
            for rep_idx, repetitions_value in enumerate(repetitions):
                for run_idx in range(tvd_runs.shape[2]):
                    writer.writerow(
                        [
                            repetitions_value,
                            perm_idx,
                            label,
                            run_idx,
                            tvd_runs[perm_idx, rep_idx, run_idx],
                        ]
                    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TVD convergence vs repetitions.")
    parser.add_argument("--qubits", type=int, default=ExperimentConfig.qubits)
    parser.add_argument("--num-gates", type=int, default=ExperimentConfig.num_gates)
    parser.add_argument("--repetitions", type=str, default=None)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--runs-per-rep", type=int, default=ExperimentConfig.runs_per_rep)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--perm-index", type=int, default=None)
    parser.add_argument("--no-mean", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    args = parser.parse_args(argv)

    repetitions = _parse_repetitions(args.repetitions)
    if any(rep <= 0 for rep in repetitions):
        raise ValueError("Repetitions must be positive integers.")

    cfg = ExperimentConfig(
        qubits=args.qubits,
        num_gates=args.num_gates,
        repetitions=repetitions,
        seed=args.seed,
        runs_per_rep=args.runs_per_rep,
        distance_scalar=args.distance_scalar,
        output_dir=args.output_dir,
    )
    if cfg.runs_per_rep < 1:
        raise ValueError("--runs-per-rep must be >= 1.")

    circuit = _build_random_circuit(cfg)
    perms = list(DEFAULT_PERMS)
    if args.perm_index is not None:
        if args.perm_index < 0 or args.perm_index >= len(perms):
            raise ValueError(f"--perm-index must be between 0 and {len(perms) - 1}.")
        perms = [perms[args.perm_index]]

    perm_labels = []
    for idx, perm in enumerate(perms):
        codes_used = len({p for p, _ in perm})
        perm_labels.append(f"perm_{idx} (codes_used={codes_used})")

    tvd_runs = np.zeros((len(perms), len(cfg.repetitions), cfg.runs_per_rep), dtype=float)

    total_runs = len(cfg.repetitions) * cfg.runs_per_rep
    run_counter = 0
    for rep_idx, rep in enumerate(cfg.repetitions):
        for run_idx in range(cfg.runs_per_rep):
            run_counter += 1
            run_seed = cfg.seed + rep_idx * 10_000 + run_idx
            print(
                f"[{run_counter}/{total_runs}] repetitions={rep} run={run_idx + 1}/{cfg.runs_per_rep}"
            )
            sim_cfg = (
                ScaledSimulator(alpha=0.5, repetitions=rep, seed=run_seed)
                .with_distance_noise(distance_scalar=cfg.distance_scalar)
                .with_stim()
            )
            pipeline_cfg = PipelineConfig(sim_config=sim_cfg, verbose=False, save=False, seed=run_seed)
            record = run_specific_scan(program_circuit=circuit, permutations=perms, config=pipeline_cfg)
            tvds = _extract_tvd_by_perm(record.dataframe.df, perms)
            tvd_runs[:, rep_idx, run_idx] = tvds

    base = f"q{cfg.qubits}_g{cfg.num_gates}_m{len(perms)}_seed{cfg.seed}"
    fig_path = cfg.output_dir / f"{base}__tvd_vs_repetitions.svg"
    csv_path = cfg.output_dir / f"{base}__tvd_vs_repetitions.csv"

    tvd_matrix = np.mean(tvd_runs, axis=2)
    _plot(
        repetitions=cfg.repetitions,
        tvd_matrix=tvd_matrix,
        perm_labels=perm_labels,
        output_path=fig_path,
        include_mean=not args.no_mean,
    )
    _write_csv(
        repetitions=cfg.repetitions,
        tvd_runs=tvd_runs,
        perm_labels=perm_labels,
        output_path=csv_path,
    )

    for perm_idx, label in enumerate(perm_labels):
        box_path = cfg.output_dir / f"{base}__tvd_boxplot_perm{perm_idx}.svg"
        _plot_boxplot(
            repetitions=cfg.repetitions,
            tvd_runs=tvd_runs[perm_idx],
            perm_label=label,
            output_path=box_path,
        )
        print(f"Wrote: {box_path}")

    print(f"Wrote: {fig_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
