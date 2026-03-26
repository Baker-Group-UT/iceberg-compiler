"""
Patch-density vs idle/distance noise heatmaps.

Sweeps idle_time_scalar × distance_scalar on a 5×5 log-spaced grid for
three code-density regimes:
    - tight    (0.5q codes):  n_codes = ceil(q / 2)
    - balanced (0.75q codes): n_codes = ceil(q * 0.75)
    - sparse   (1q codes):    n_codes = q

Results are averaged across all loadable medium QASMBench circuits using a
trivial sequential packing at each density level (no merging, no
h-commutation, no packer optimisation).

Outputs:
  * density_noise_heatmap.svg  — 2×3 grid: TVD row on top, LSR row on bottom,
                                 columns ordered tight / balanced / sparse
  * density_noise_heatmap.csv  — 75 rows (one per density × idle × distance)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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

from experiments.shared.benchmark_circuits import load_proxy_circuits
from pipeline import PipelineConfig, run_specific_scan
from simulator.config_presets import ScaledSimulator
from translator.config import TranslatorConfig

BENCHMARKS_ROOT = REPO_ROOT / "benchmarks" / "qasmbench"

plt.rcParams["font.family"] = "Helvetica"

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12
COLORBAR_LABEL_FONTSIZE = 15
COLORBAR_TICK_FONTSIZE = 12

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "density_noise_heatmaps",
    ["#ffffff", "#b3cde0", "#6497b1", "#005b96", "#03396c", "#011f4b"],
)

# (label shown in titles/CSV, density ratio used to compute n_codes)
DENSITY_LEVELS: list[tuple[str, float]] = [
    ("tight",    0.50),
    ("balanced", 0.75),
    ("sparse",   1.00),
]

DENSITY_TITLES = {
    "tight":    "0.5q codes (tight)",
    "balanced": "0.75q codes (balanced)",
    "sparse":   "1q codes (sparse)",
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    alpha: float = 0.05
    pilot_repetitions: int = 4_000
    target_kept_shots: int = 20_000
    min_repetitions: int = 20_000
    max_repetitions: int = 500_000
    idle_min: float = 0.1
    idle_max: float = 10.0
    idle_points: int = 5
    distance_min: float = 0.1
    distance_max: float = 10.0
    distance_points: int = 5
    output_dir: Path = THIS_DIR


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def _n_codes(q: int, ratio: float) -> int:
    """Number of code patches for a circuit of q qubits at the given density ratio."""
    return max(1, math.ceil(q * ratio))


def _trivial_packing(q: int, n_codes: int) -> list[tuple[int, int]]:
    """Assign qubits sequentially into codes.

    Qubit i → (code i // capacity, slot i % capacity)
    where capacity = ceil(q / n_codes).  Slot indices are always 0 or 1
    since capacity ≤ 2 for all density levels supported here.
    """
    capacity = math.ceil(q / n_codes)
    return [(i // capacity, i % capacity) for i in range(q)]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _repetitions_from_lsr(lsr: float, cfg: ExperimentConfig) -> int:
    denom = max(lsr, 1e-4)
    reps = int(math.ceil(cfg.target_kept_shots / denom))
    return int(np.clip(reps, cfg.min_repetitions, cfg.max_repetitions))


def _build_pipeline_config(
    cfg: ExperimentConfig,
    *,
    idle_scalar: float,
    distance_scalar: float,
    repetitions: int,
    seed: int,
) -> PipelineConfig:
    sim_cfg = (
        ScaledSimulator(alpha=cfg.alpha, repetitions=repetitions, seed=seed)
        .with_stim()
        .with_distance_noise(distance_scalar=distance_scalar)
        .with_idle_noise(idle_time_scalar=idle_scalar)
    )
    translator_cfg = TranslatorConfig(
        enable_merging=False,
        enable_h_commutation=False,
    )
    return PipelineConfig(
        sim_config=sim_cfg,
        translator_config=translator_cfg,
        verbose=False,
        save=False,
        seed=seed,
    )


def _run_circuit(
    circuit,
    permutation: list[tuple[int, int]],
    pipeline_cfg: PipelineConfig,
) -> tuple[float, float]:
    """Run a single circuit + permutation and return (tvd, lsr)."""
    with StringIO() as out, StringIO() as err:
        with redirect_stdout(out), redirect_stderr(err):
            record = run_specific_scan(circuit, [permutation], pipeline_cfg)
    df = record.dataframe.df
    tvd = float(df["logical_total_variation_distance"].mean())
    lsr = float(df["logical_selection_rate"].mean())
    return tvd, lsr


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _sweep(
    cfg: ExperimentConfig,
    circuits: list[tuple[str, object]],
) -> list[dict]:
    idle_values = np.logspace(
        np.log10(cfg.idle_min), np.log10(cfg.idle_max), cfg.idle_points
    )
    distance_values = np.logspace(
        np.log10(cfg.distance_min), np.log10(cfg.distance_max), cfg.distance_points
    )

    total_runs = len(DENSITY_LEVELS) * len(idle_values) * len(distance_values) * len(circuits)
    rows: list[dict] = []

    with tqdm(total=total_runs, desc="Simulations", unit="run") as progress:
        for density_idx, (density_label, density_ratio) in enumerate(DENSITY_LEVELS):
            for idle_idx, idle_scalar in enumerate(idle_values):
                for dist_idx, distance_scalar in enumerate(distance_values):
                    tvd_vals: list[float] = []
                    lsr_vals: list[float] = []
                    for circuit_idx, (_circuit_name, circuit) in enumerate(circuits):
                        q = circuit.num_qubits
                        n_codes = _n_codes(q, density_ratio)
                        perm = _trivial_packing(q, n_codes)

                        seed_base = cfg.seed * 1_000_003 + density_idx * 10_000 + idle_idx * 100 + dist_idx * 10 + circuit_idx

                        pilot_cfg = _build_pipeline_config(
                            cfg,
                            idle_scalar=float(idle_scalar),
                            distance_scalar=float(distance_scalar),
                            repetitions=cfg.pilot_repetitions,
                            seed=seed_base,
                        )
                        _, pilot_lsr = _run_circuit(circuit, perm, pilot_cfg)
                        final_reps = _repetitions_from_lsr(pilot_lsr, cfg)

                        final_cfg = _build_pipeline_config(
                            cfg,
                            idle_scalar=float(idle_scalar),
                            distance_scalar=float(distance_scalar),
                            repetitions=final_reps,
                            seed=seed_base + 1,
                        )
                        tvd, lsr = _run_circuit(circuit, perm, final_cfg)
                        tvd_vals.append(tvd)
                        lsr_vals.append(lsr)
                        progress.update(1)

                    rows.append(
                        {
                            "density_label": density_label,
                            "density_ratio": density_ratio,
                            "idle_scalar": float(idle_scalar),
                            "distance_scalar": float(distance_scalar),
                            "mean_tvd": float(np.mean(tvd_vals)),
                            "mean_lsr": float(np.mean(lsr_vals)),
                        }
                    )
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _log_edges(values: np.ndarray) -> np.ndarray:
    logs = np.log10(values)
    mids = 0.5 * (logs[:-1] + logs[1:])
    first = logs[0] - 0.5 * (logs[1] - logs[0])
    last = logs[-1] + 0.5 * (logs[-1] - logs[-2])
    return np.power(10.0, np.concatenate(([first], mids, [last])))


def _pivot(rows: list[dict], density_label: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (idle_values, distance_values, matrix) for one density + metric."""
    sub = [r for r in rows if r["density_label"] == density_label]
    idle_vals = sorted({r["idle_scalar"] for r in sub})
    dist_vals = sorted({r["distance_scalar"] for r in sub})
    idle_idx = {v: i for i, v in enumerate(idle_vals)}
    dist_idx = {v: i for i, v in enumerate(dist_vals)}
    mat = np.zeros((len(idle_vals), len(dist_vals)), dtype=float)
    for r in sub:
        mat[idle_idx[r["idle_scalar"]], dist_idx[r["distance_scalar"]]] = r[metric]
    return np.array(idle_vals), np.array(dist_vals), mat


def _plot(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    metric_rows = [
        ("mean_tvd", "Mean TVD"),
        ("mean_lsr", "Mean LSR"),
    ]

    for row_idx, (metric_col, metric_label) in enumerate(metric_rows):
        # Shared colour scale across all three density columns.
        all_vals = [r[metric_col] for r in rows]
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        last_im = None

        for col_idx, (density_label, _) in enumerate(DENSITY_LEVELS):
            ax = axes[row_idx, col_idx]
            idle_vals, dist_vals, mat = _pivot(rows, density_label, metric_col)
            x_edges = _log_edges(dist_vals)
            y_edges = _log_edges(idle_vals)

            im = ax.pcolormesh(
                x_edges,
                y_edges,
                mat,
                vmin=vmin,
                vmax=vmax,
                cmap=HEATMAP_CMAP,
                shading="auto",
            )
            last_im = im

            if row_idx == 0:
                ax.set_title(DENSITY_TITLES[density_label], fontsize=TITLE_FONTSIZE)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        cb = fig.colorbar(last_im, ax=axes[row_idx, :], shrink=0.85)
        cb.set_label(metric_label, fontsize=COLORBAR_LABEL_FONTSIZE)
        cb.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    fig.supxlabel("Distance scalar", fontsize=AXIS_LABEL_FONTSIZE)
    fig.supylabel("Idle scalar", fontsize=AXIS_LABEL_FONTSIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_path}")


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "density_label",
        "density_ratio",
        "idle_scalar",
        "distance_scalar",
        "mean_tvd",
        "mean_lsr",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {output_path}")


def _load_from_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Patch-density vs idle/distance noise heatmaps averaged over "
            "medium QASMBench circuits."
        )
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--alpha", type=float, default=ExperimentConfig.alpha)
    parser.add_argument("--pilot-repetitions", type=int, default=ExperimentConfig.pilot_repetitions)
    parser.add_argument("--target-kept-shots", type=int, default=ExperimentConfig.target_kept_shots)
    parser.add_argument("--min-repetitions", type=int, default=ExperimentConfig.min_repetitions)
    parser.add_argument("--max-repetitions", type=int, default=ExperimentConfig.max_repetitions)
    parser.add_argument("--idle-min", type=float, default=ExperimentConfig.idle_min)
    parser.add_argument("--idle-max", type=float, default=ExperimentConfig.idle_max)
    parser.add_argument("--idle-points", type=int, default=ExperimentConfig.idle_points)
    parser.add_argument("--distance-min", type=float, default=ExperimentConfig.distance_min)
    parser.add_argument("--distance-max", type=float, default=ExperimentConfig.distance_max)
    parser.add_argument("--distance-points", type=int, default=ExperimentConfig.distance_points)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reload a previously generated CSV and re-render the SVG without rerunning simulations.",
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        seed=args.seed,
        alpha=args.alpha,
        pilot_repetitions=args.pilot_repetitions,
        target_kept_shots=args.target_kept_shots,
        min_repetitions=args.min_repetitions,
        max_repetitions=args.max_repetitions,
        idle_min=args.idle_min,
        idle_max=args.idle_max,
        idle_points=args.idle_points,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        distance_points=args.distance_points,
        output_dir=args.output_dir,
    )

    figure_path = cfg.output_dir / "density_noise_heatmap.svg"
    csv_path = cfg.output_dir / "density_noise_heatmap.csv"

    if args.from_csv is not None:
        rows = _load_from_csv(args.from_csv)
    else:
        print("Loading medium QASMBench circuits...")
        circuits = load_proxy_circuits(BENCHMARKS_ROOT / "medium")
        if not circuits:
            print("No circuits loaded — check benchmarks path.", file=sys.stderr)
            return 1
        print(f"Loaded {len(circuits)} circuits.")
        rows = _sweep(cfg, circuits)
        _write_csv(rows, csv_path)

    _plot(rows, figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
