"""
lsr_tvd_correlation_medium_benchmark.py

Scatter plot of logical TVD vs logical selection rate (LSR) across random
permutations of all medium QASMBench circuits, all on a single figure.

Each circuit contributes a cloud of points and its own linear fit line.
No per-circuit labels are shown — the plot shows the aggregate relationship.

Outputs:
  * lsr_tvd_correlation_medium_benchmark.svg
  * lsr_tvd_correlation_medium_benchmark.csv  (circuit_name, lsr, tvd per row)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt

from experiments.shared.benchmark_circuits import load_proxy_circuits
from pipeline import PipelineConfig, run_random_scan
from simulator.config_presets import ScaledSimulator
from translator.config import TranslatorConfig

BENCHMARKS_ROOT = REPO_ROOT / "benchmarks" / "qasmbench"

POINT_COLOR = "#6497b1"
FIT_COLOR   = "#011f4b"
ACCENT      = "#3b4c56"

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
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
    permutations: int = 20
    repetitions: int = 100_000
    seed: int = 11
    alpha: float = 0.05
    distance_scalar: float = 1.0
    idle_scalar: float = 1.0
    output_dir: Path = THIS_DIR


def _run_circuit(
    name: str,
    circuit,
    cfg: ExperimentConfig,
) -> list[dict]:
    sim_cfg = (
        ScaledSimulator(alpha=cfg.alpha, repetitions=cfg.repetitions, seed=cfg.seed)
        .with_stim()
        .with_distance_noise(distance_scalar=cfg.distance_scalar)
        .with_idle_noise(idle_time_scalar=cfg.idle_scalar)
    )
    pipeline_cfg = PipelineConfig(
        sim_config=sim_cfg,
        translator_config=TranslatorConfig(enable_h_commutation=False),
        verbose=False,
        save=False,
        seed=cfg.seed,
    )
    record = run_random_scan(program_circuit=circuit, m=cfg.permutations, config=pipeline_cfg)
    df = record.dataframe.df.copy()
    df["logical_selection_rate"] = df["logical_selection_rate"].apply(
        lambda v: float(v) if v is not None else float("nan")
    )
    df["logical_total_variation_distance"] = df["logical_total_variation_distance"].apply(
        lambda v: float(v) if v is not None else float("nan")
    )
    df = df.dropna(subset=["logical_selection_rate", "logical_total_variation_distance"])
    return [
        {"circuit_name": name, "lsr": row["logical_selection_rate"], "tvd": row["logical_total_variation_distance"]}
        for _, row in df.iterrows()
    ]


def _plot(circuit_data: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
    ax.set_facecolor("white")

    for lsr, tvd in circuit_data.values():
        ax.scatter(lsr, tvd, s=18, alpha=0.45, color=POINT_COLOR, edgecolor="none")
        if len(lsr) >= 2 and np.isfinite(lsr).all() and np.isfinite(tvd).all():
            if not (np.isclose(np.std(lsr), 0.0) or np.isclose(np.std(tvd), 0.0)):
                slope, intercept = np.polyfit(lsr, tvd, deg=1)
                xs = np.linspace(float(np.min(lsr)), float(np.max(lsr)), 200)
                ax.plot(xs, slope * xs + intercept, linewidth=1.4, color=FIT_COLOR, alpha=0.7)

    total_points = sum(len(v[0]) for v in circuit_data.values())
    all_lsr = np.concatenate([v[0] for v in circuit_data.values()])
    all_tvd = np.concatenate([v[1] for v in circuit_data.values()])
    mask = np.isfinite(all_lsr) & np.isfinite(all_tvd)
    if mask.sum() >= 2 and not np.isclose(np.std(all_lsr[mask]), 0.0):
        r_total = f"{pearsonr(all_lsr[mask], all_tvd[mask]).statistic:.3f}"
    else:
        r_total = "n/a"
    ax.text(
        0.02, 0.98,
        f"n circuits = {len(circuit_data)}\nn points = {total_points}\nPearson r = {r_total}",
        transform=ax.transAxes,
        ha="left", va="top",
        color=ACCENT, fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": ACCENT, "alpha": 0.9},
    )

    ax.set_xlabel("Logical selection rate (LSR)")
    ax.set_ylabel("Logical total variation distance (TVD)")
    ax.set_title("LSR vs TVD across random permutations — medium QASMBench")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_path}")


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["circuit_name", "lsr", "tvd"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TVD vs LSR scatter across medium QASMBench circuits."
    )
    parser.add_argument("--permutations", type=int, default=ExperimentConfig.permutations)
    parser.add_argument("--repetitions", type=int, default=ExperimentConfig.repetitions)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--alpha", type=float, default=ExperimentConfig.alpha)
    parser.add_argument("--distance-scalar", type=float, default=ExperimentConfig.distance_scalar)
    parser.add_argument("--idle-scalar", type=float, default=ExperimentConfig.idle_scalar)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        permutations=args.permutations,
        repetitions=args.repetitions,
        seed=args.seed,
        alpha=args.alpha,
        distance_scalar=args.distance_scalar,
        idle_scalar=args.idle_scalar,
        output_dir=args.output_dir,
    )

    print("Loading medium QASMBench circuits...")
    circuits = load_proxy_circuits(BENCHMARKS_ROOT / "medium")
    if not circuits:
        print("No circuits loaded.", file=sys.stderr)
        return 1
    for _, qc in circuits:
        qc.measure_all(inplace=True)
    print(f"Loaded {len(circuits)} circuits.")

    all_rows: list[dict] = []
    circuit_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, circuit in tqdm(circuits, desc="circuits"):
        rows = _run_circuit(name, circuit, cfg)
        all_rows.extend(rows)
        if rows:
            lsr_arr = np.array([r["lsr"] for r in rows], dtype=float)
            tvd_arr = np.array([r["tvd"] for r in rows], dtype=float)
            circuit_data[name] = (lsr_arr, tvd_arr)

    _write_csv(all_rows, cfg.output_dir / "lsr_tvd_correlation_medium_benchmark.csv")
    _plot(circuit_data, cfg.output_dir / "lsr_tvd_correlation_medium_benchmark.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
