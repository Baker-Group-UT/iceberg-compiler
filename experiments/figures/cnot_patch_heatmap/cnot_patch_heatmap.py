"""
Compare a single intra-patch vs inter-patch logical CNOT across idle and distance noise.

This script sweeps:
    - idle_time_scalar
    - distance_scalar

for the same 2-logical-qubit program circuit under two packings:
    - intra-patch: [(0, 0), (0, 1)]
    - inter-patch: [(0, 0), (1, 0)]

Outputs:
    - cnot_patch_heatmap.svg
    - cnot_patch_heatmap.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator
from tqdm.auto import tqdm

from pipeline import PipelineConfig, run_specific_scan
from simulator.config_presets import StandardSimulator


INTRA_PATCH_PERMUTATION = [(0, 0), (0, 1)]
INTER_PATCH_PERMUTATION = [(0, 0), (1, 0)]
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "cnot_diff",
    ["#ffffff", "#005b96"],
)


@dataclass(frozen=True)
class ExperimentConfig:
    repetitions: int = 100_000
    seed: int = 7
    idle_min: float = 1e1
    idle_max: float = .3e3
    idle_points: int = 25
    distance_min: float = 1e1
    distance_max: float = 1.5e3
    distance_points: int = 25
    metric: str = "logical_fidelity"
    output_dir: Path = THIS_DIR


def _build_program_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="logical_single_cnot")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _metric_label(metric: str) -> str:
    if metric == "logical_fidelity":
        return "Logical fidelity"
    if metric == "logical_total_variation_distance":
        return "Logical TVD"
    raise ValueError(f"Unsupported metric: {metric}")


def _title_label(metric: str) -> str:
    if metric == "logical_fidelity":
        return "Fidelity"
    if metric == "logical_total_variation_distance":
        return "TVD"
    raise ValueError(f"Unsupported metric: {metric}")


def _build_sim_config(*, repetitions: int, seed: int, idle_scalar: float, distance_scalar: float):
    return (
        StandardSimulator(repetitions=repetitions, seed=seed)
        .with_stim()
        .with_distance_noise(distance_scalar=distance_scalar)
        .with_idle_noise(idle_time_scalar=idle_scalar)
    )


def _run_case(
    *,
    circuit: QuantumCircuit,
    permutation: list[tuple[int, int]],
    repetitions: int,
    seed: int,
    idle_scalar: float,
    distance_scalar: float,
    metric: str,
) -> float:
    sim_cfg = _build_sim_config(
        repetitions=repetitions,
        seed=seed,
        idle_scalar=idle_scalar,
        distance_scalar=distance_scalar,
    )
    pipeline_cfg = PipelineConfig(sim_config=sim_cfg, verbose=False, save=False, seed=seed)
    with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            record = run_specific_scan(circuit, [permutation], pipeline_cfg)
    df = record.dataframe.df
    if metric not in df.columns:
        raise KeyError(f"Missing metric column: {metric}")
    return float(df.loc[0, metric])


def _sweep(cfg: ExperimentConfig):
    circuit = _build_program_circuit()
    idle_values = np.logspace(np.log10(cfg.idle_min), np.log10(cfg.idle_max), cfg.idle_points)
    distance_values = np.logspace(
        np.log10(cfg.distance_min),
        np.log10(cfg.distance_max),
        cfg.distance_points,
    )

    intra = np.zeros((len(idle_values), len(distance_values)), dtype=float)
    inter = np.zeros_like(intra)
    rows: list[dict[str, float | str]] = []
    total_runs = len(idle_values) * len(distance_values) * 2

    with tqdm(total=total_runs, desc="Simulations", unit="run") as progress:
        for idle_index, idle_scalar in enumerate(idle_values):
            for distance_index, distance_scalar in enumerate(distance_values):
                intra_value = _run_case(
                    circuit=circuit,
                    permutation=INTRA_PATCH_PERMUTATION,
                    repetitions=cfg.repetitions,
                    seed=cfg.seed,
                    idle_scalar=float(idle_scalar),
                    distance_scalar=float(distance_scalar),
                    metric=cfg.metric,
                )
                progress.update(1)

                inter_value = _run_case(
                    circuit=circuit,
                    permutation=INTER_PATCH_PERMUTATION,
                    repetitions=cfg.repetitions,
                    seed=cfg.seed,
                    idle_scalar=float(idle_scalar),
                    distance_scalar=float(distance_scalar),
                    metric=cfg.metric,
                )
                progress.update(1)

                intra[idle_index, distance_index] = intra_value
                inter[idle_index, distance_index] = inter_value

                rows.append(
                    {
                        "idle_time_scalar": float(idle_scalar),
                        "distance_scalar": float(distance_scalar),
                        "intra_patch_metric": intra_value,
                        "inter_patch_metric": inter_value,
                        "difference_intra_minus_inter": intra_value - inter_value,
                        "metric": cfg.metric,
                    }
                )

    return idle_values, distance_values, intra, inter, rows


def _log_edges(values: np.ndarray) -> np.ndarray:
    logs = np.log10(values)
    mids = 0.5 * (logs[:-1] + logs[1:])
    first = logs[0] - 0.5 * (logs[1] - logs[0])
    last = logs[-1] + 0.5 * (logs[-1] - logs[-2])
    return np.power(10.0, np.concatenate(([first], mids, [last])))


def _plot_heatmaps(
    *,
    idle_values: np.ndarray,
    distance_values: np.ndarray,
    intra: np.ndarray,
    inter: np.ndarray,
    metric: str,
    output_path: Path,
) -> None:
    diff = intra - inter
    metric_label = _metric_label(metric)
    title_label = _title_label(metric)
    x_edges = _log_edges(distance_values)
    y_edges = _log_edges(idle_values)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)

    vmin = min(float(np.min(intra)), float(np.min(inter)))
    vmax = max(float(np.max(intra)), float(np.max(inter)))

    intra_im = axes[0].pcolormesh(
        x_edges,
        y_edges,
        intra,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        shading="auto",
    )
    inter_im = axes[1].pcolormesh(
        x_edges,
        y_edges,
        inter,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        shading="auto",
    )

    diff_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    diff_im = axes[2].pcolormesh(
        x_edges,
        y_edges,
        diff,
        vmin=-diff_abs,
        vmax=diff_abs,
        cmap=DIFF_CMAP,
        shading="auto",
    )

    axes[0].set_title(f"Intra-patch {title_label}")
    axes[1].set_title(f"Inter-patch {title_label}")
    axes[2].set_title("Difference (Intra - Inter)")

    for ax in axes:
        ax.set_xlabel("Distance scalar")
        ax.set_ylabel("Idle scalar")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())

    fig.colorbar(intra_im, ax=axes[:2], shrink=0.92, label=metric_label)
    fig.colorbar(diff_im, ax=axes[2], shrink=0.92, label=f"{metric_label} difference")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "idle_time_scalar",
                "distance_scalar",
                "intra_patch_metric",
                "inter_patch_metric",
                "difference_intra_minus_inter",
                "metric",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_from_csv(csv_path: Path, metric: str):
    df = pd.read_csv(csv_path)
    required = {
        "idle_time_scalar",
        "distance_scalar",
        "intra_patch_metric",
        "inter_patch_metric",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    if "metric" in df.columns:
        metric_df = df[df["metric"] == metric].copy()
        if metric_df.empty:
            raise ValueError(f"No rows found in CSV for metric '{metric}'.")
        df = metric_df

    idle_values = np.array(sorted(df["idle_time_scalar"].astype(float).unique()))
    distance_values = np.array(sorted(df["distance_scalar"].astype(float).unique()))

    intra_table = (
        df.pivot(
            index="idle_time_scalar",
            columns="distance_scalar",
            values="intra_patch_metric",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    inter_table = (
        df.pivot(
            index="idle_time_scalar",
            columns="distance_scalar",
            values="inter_patch_metric",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    intra = intra_table.to_numpy(dtype=float)
    inter = inter_table.to_numpy(dtype=float)
    return idle_values, distance_values, intra, inter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate intra- vs inter-patch CNOT idle/distance heatmaps."
    )
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--idle-min", type=float, default=ExperimentConfig.idle_min)
    parser.add_argument("--idle-max", type=float, default=ExperimentConfig.idle_max)
    parser.add_argument("--idle-points", type=int, default=ExperimentConfig.idle_points)
    parser.add_argument("--distance-min", type=float, default=ExperimentConfig.distance_min)
    parser.add_argument("--distance-max", type=float, default=ExperimentConfig.distance_max)
    parser.add_argument("--distance-points", type=int, default=ExperimentConfig.distance_points)
    parser.add_argument(
        "--metric",
        choices=("logical_fidelity", "logical_total_variation_distance"),
        default=ExperimentConfig.metric,
    )
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reload a previously generated CSV and re-render the SVG without rerunning simulations.",
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        repetitions=args.repetitions,
        seed=args.seed,
        idle_min=args.idle_min,
        idle_max=args.idle_max,
        idle_points=args.idle_points,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        distance_points=args.distance_points,
        metric=args.metric,
        output_dir=args.output_dir,
    )

    figure_path = cfg.output_dir / "cnot_patch_heatmap.svg"
    csv_path = cfg.output_dir / "cnot_patch_heatmap.csv"
    if args.from_csv is not None:
        idle_values, distance_values, intra, inter = _load_from_csv(args.from_csv, cfg.metric)
    else:
        idle_values, distance_values, intra, inter, rows = _sweep(cfg)

    _plot_heatmaps(
        idle_values=idle_values,
        distance_values=distance_values,
        intra=intra,
        inter=inter,
        metric=cfg.metric,
        output_path=figure_path,
    )
    if args.from_csv is None:
        _write_csv(rows, csv_path)

    print(f"Wrote: {figure_path}")
    if args.from_csv is None:
        print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
