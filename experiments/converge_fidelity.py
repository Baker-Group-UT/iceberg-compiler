# experiments/repetition_sweep.py
"""
Sweep simulator repetitions for a fixed program circuit and permutations,
then plot TVD and fidelity vs repetitions.

Outputs:
- results/plots/repetition_sweep_<timestamp>.csv
- results/plots/repetition_sweep_tvd_<timestamp>.png
- results/plots/repetition_sweep_fidelity_<timestamp>.png
"""

from __future__ import annotations
import argparse
import os
from datetime import datetime

import qiskit
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan
from repository.run_builder import RunRecord
from simulator.config_presets import StandardSimulator

# --- Program circuit (same as your run_scan example) ---
PROGRAM_CIRCUIT = qiskit.QuantumCircuit(3)
PROGRAM_CIRCUIT.h(0)
PROGRAM_CIRCUIT.cx(0,1)
PROGRAM_CIRCUIT.cx(0,2)
PROGRAM_CIRCUIT.h(0)

# Default permutations from your snippet
DEFAULT_PERMUTATIONS = [
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (2, 0)],
    # [(0, 1), (0, 0), (1, 1)],  # example you had commented out
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep repetitions and plot TVD/Fidelity.")
    p.add_argument(
        "--reps",
        type=int,
        nargs="+",
        default=[100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000],
        help="List of simulator repetition counts to sweep.",
    )
    p.add_argument(
        "--save-records",
        action="store_true",
        help="If set, enable saving RunRecord artifacts to your repo’s storage.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="If set, enable verbose pipeline logs.",
    )
    p.add_argument(
        "--ft",
        action="store_true",
        help="If set, run fault-tolerant mode via PipelineConfig.with_ft(True).",
    )
    return p.parse_args()

def ensure_outdir() -> str:
    outdir = os.path.join("results", "plots")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def build_config(repetitions: int, save: bool, verbose: bool, ft: bool) -> PipelineConfig:
    sim = StandardSimulator(repetitions=repetitions)
    cfg = PipelineConfig(sim_config=sim, verbose=verbose, save=save)
    if ft:
        cfg = cfg.with_ft(True)
    return cfg

def run_once(repetitions: int, save: bool, verbose: bool, ft: bool) -> pd.DataFrame:
    config = build_config(repetitions, save, verbose, ft)
    record: RunRecord = run_specific_scan(
        PROGRAM_CIRCUIT,
        permutations=DEFAULT_PERMUTATIONS,
        config=config,
    )
    df = record.dataframe.df  # <- your RunRecord dataframe
    # Add sweep metadata
    df = df.assign(repetitions=repetitions)
    return df

def plot_metric(df: pd.DataFrame, metric: str, outpath: str, title: str):
    """
    Plot a metric vs repetitions.
    - One line per permutation (perm_key)
    - Also plot the mean across permutations
    """
    plt.figure()
    # Per-permutation lines
    for perm_key, g in df.groupby("perm_key", sort=False):
        g_sorted = g.sort_values("repetitions")
        plt.semilogx(
            g_sorted["repetitions"],
            g_sorted[metric],
            marker="o",
            label=f"perm={perm_key}",
        )

    # Global mean line
    mean_df = df.groupby("repetitions", as_index=False)[metric].mean().sort_values("repetitions")
    plt.semilogx(
        mean_df["repetitions"],
        mean_df[metric],
        marker="o",
        linewidth=3,
        label="mean across perms",
    )

    plt.xlabel("Repetitions (log scale)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    args = parse_args()
    outdir = ensure_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect across sweep
    frames: list[pd.DataFrame] = []
    for reps in args.reps:
        df = run_once(reps, args.save_records, args.verbose, args.ft)
        # Keep only the columns we need (plus helpful identifiers)
        select_cols = [
            "perm_key",
            "permutation",
            "codes_used",
            "logical_fidelity",
            "logical_total_variation_distance",
            "logical_selection_rate",
            "selection_rate_overall",
            "gate_count",
            "total_gates",
            "contains_faults",
            "fault_tolerant",
            "repetitions",
        ]
        # Columns may vary by config; keep intersection
        cols = [c for c in select_cols if c in df.columns]
        frames.append(df[cols].copy())

    sweep_df = pd.concat(frames, ignore_index=True)

    # Persist CSV artifact
    csv_path = os.path.join(outdir, f"repetition_sweep_{stamp}.csv")
    sweep_df.to_csv(csv_path, index=False)

    # Make plots: TVD and Fidelity
    if "logical_total_variation_distance" in sweep_df.columns:
        tvd_png = os.path.join(outdir, f"repetition_sweep_tvd_{stamp}.png")
        plot_metric(
            sweep_df,
            metric="logical_total_variation_distance",
            outpath=tvd_png,
            title="Logical TVD vs Repetitions",
        )

    if "logical_fidelity" in sweep_df.columns:
        fid_png = os.path.join(outdir, f"repetition_sweep_fidelity_{stamp}.png")
        plot_metric(
            sweep_df,
            metric="logical_fidelity",
            outpath=fid_png,
            title="Logical Fidelity vs Repetitions",
        )

    print(f"[OK] Saved sweep CSV to: {csv_path}")
    if "logical_total_variation_distance" in sweep_df.columns:
        print(f"[OK] Saved TVD plot to: {tvd_png}")
    if "logical_fidelity" in sweep_df.columns:
        print(f"[OK] Saved fidelity plot to: {fid_png}")

if __name__ == "__main__":
    main()
