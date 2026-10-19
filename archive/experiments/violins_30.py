"""
violins_30.py

Generate 10 random 4-qubit, 10 random 5-qubit, and 10 random 6-qubit
random Clifford circuits of gate depth 30–60. For each circuit:
  1) Transpile to the specified gateset.
  2) Run a specific scan with the given permutations.
  3) Graph TVD and Logical Selection Rate as violin plots with circuit info.
  4) Save plots to 30_violins/<combo>.
  5) Save the scan using RunRepository (RunRecord JSON.gz) and a CSV of the DF.

This script assumes your project layout provides:
  - pipeline.{run_specific_scan}, PipelineConfig
  - simulator.config_presets.{ScaledSimulator, StandardSimulator}
  - repository.run_builder.RunRecord
  - recorder.repository.RunRepository
"""

from __future__ import annotations

import os
import sys
sys.path.append("../")

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import qiskit
from qiskit.circuit.random import random_clifford_circuit

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan
from simulator.config_presets import ScaledSimulator  # or StandardSimulator
from archive.repository.run_builder import RunRecord
from archive.repository.repository import RunRepository

# -----------------------
# Permutations (0..60)
# -----------------------
permutations4: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (0, 1)],
    [(0, 0), (1, 1), (0, 1), (1, 0)],
    [(0, 0), (1, 1), (1, 0), (0, 1)],
    [(0, 1), (0, 0), (1, 0), (1, 1)],
    [(0, 1), (0, 0), (1, 1), (1, 0)],
    [(0, 1), (1, 0), (0, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (0, 0)],
    [(0, 1), (1, 1), (0, 0), (1, 0)],
    [(0, 1), (1, 1), (1, 0), (0, 0)],
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (0, 1), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1)],
    [(0, 1), (0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 0), (0, 0), (2, 0)],
    [(0, 1), (1, 0), (2, 0), (0, 0)],
    [(1, 0), (0, 0), (0, 1), (2, 0)],
    [(1, 0), (0, 0), (2, 0), (0, 1)],
    [(1, 0), (0, 1), (0, 0), (2, 0)],
    [(1, 0), (0, 1), (2, 0), (0, 0)],
    [(1, 0), (2, 0), (0, 0), (0, 1)],
    [(1, 0), (2, 0), (0, 1), (0, 0)],
    [(0, 0), (0, 1), (1, 0), (2, 1)],
    [(0, 0), (0, 1), (2, 1), (1, 0)],
    [(0, 0), (1, 0), (0, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 1), (0, 1)],
    [(0, 0), (2, 1), (0, 1), (1, 0)],
    [(0, 0), (2, 1), (1, 0), (0, 1)],
    [(0, 1), (0, 0), (1, 0), (2, 1)],
    [(0, 1), (0, 0), (2, 1), (1, 0)],
    [(0, 1), (1, 0), (0, 0), (2, 1)],
    [(0, 1), (1, 0), (2, 1), (0, 0)],
    [(0, 1), (2, 1), (0, 0), (1, 0)],
    [(0, 1), (2, 1), (1, 0), (0, 0)],
    [(1, 0), (0, 0), (0, 1), (2, 1)],
    [(1, 0), (0, 0), (2, 1), (0, 1)],
    [(1, 0), (0, 1), (0, 0), (2, 1)],
    [(1, 0), (0, 1), (2, 1), (0, 0)],
    [(1, 0), (2, 1), (0, 0), (0, 1)],
    [(1, 0), (2, 1), (0, 1), (0, 0)],
    [(2, 1), (0, 0), (0, 1), (1, 0)],
    [(2, 1), (0, 0), (1, 0), (0, 1)],
    [(2, 1), (0, 1), (0, 0), (1, 0)],
    [(2, 1), (0, 1), (1, 0), (0, 0)],
    [(2, 1), (1, 0), (0, 0), (0, 1)],
    [(2, 1), (1, 0), (0, 1), (0, 0)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (1, 1), (0, 1), (2, 1)],
    [(0, 0), (1, 1), (2, 1), (0, 1)],
    [(0, 1), (0, 0), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (0, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 1), (0, 0)],
    [(1, 1), (0, 0), (0, 1), (2, 1)],
    [(1, 1), (0, 0), (2, 1), (0, 1)],
    [(1, 1), (0, 1), (0, 0), (2, 1)],
    [(1, 1), (0, 1), (2, 1), (0, 0)],
    [(1, 1), (2, 1), (0, 0), (0, 1)],
    [(1, 1), (2, 1), (0, 1), (0, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
]

permutations5: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    [(1, 0), (2, 0), (0, 1), (0, 0), (3, 0)],
    [(2, 1), (3, 1), (1, 0), (0, 1), (0, 0)],
    [(2, 0), (0, 0), (1, 1), (1, 0), (0, 1)],
    [(2, 1), (3, 1), (0, 0), (0, 1), (1, 0)],
    [(0, 0), (3, 1), (1, 0), (2, 0), (0, 1)],
    [(0, 0), (2, 1), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (2, 1), (0, 0), (3, 1)],
    [(1, 1), (0, 0), (2, 1), (3, 1), (0, 1)],
    [(1, 0), (2, 1), (0, 1), (3, 1), (0, 0)],
    [(1, 0), (0, 1), (2, 1), (0, 0), (3, 1)],
    [(1, 0), (0, 0), (2, 0), (3, 0), (0, 1)],
    [(0, 1), (1, 0), (2, 0), (0, 0), (3, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 1), (0, 1)],
    [(0, 1), (2, 0), (0, 0), (1, 1), (1, 0)],
    [(0, 1), (1, 0), (0, 0), (2, 0), (3, 1)],
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (2, 1), (3, 1)],
    [(1, 0), (0, 0), (0, 1), (2, 0), (3, 0)],
    [(1, 1), (0, 0), (0, 1), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (2, 1), (1, 0), (1, 1)],
    [(0, 1), (2, 0), (1, 0), (1, 1), (0, 0)],
    [(0, 0), (0, 1), (2, 1), (3, 1), (1, 0)],
    [(2, 1), (0, 1), (1, 0), (3, 1), (0, 0)],
    [(3, 1), (1, 0), (0, 0), (0, 1), (2, 0)],
    [(2, 0), (0, 1), (1, 0), (1, 1), (0, 0)],
    [(0, 0), (1, 0), (2, 1), (0, 1), (1, 1)],
    [(0, 0), (2, 1), (1, 0), (0, 1), (3, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 1), (2, 1)],
    [(0, 1), (1, 0), (2, 1), (3, 1), (0, 0)],
    [(0, 1), (2, 1), (0, 0), (1, 1), (1, 0)],
    [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0)],
    [(2, 1), (1, 0), (3, 1), (0, 0), (0, 1)],
    [(2, 1), (1, 0), (0, 0), (0, 1), (3, 1)],
    [(1, 0), (2, 1), (3, 1), (0, 1), (0, 0)],
    [(1, 0), (2, 0), (3, 1), (0, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 1), (1, 1), (0, 1)],
    [(2, 1), (0, 0), (3, 1), (0, 1), (1, 0)],
    [(0, 0), (3, 1), (1, 0), (1, 1), (2, 0)],
    [(2, 1), (0, 1), (0, 0), (3, 1), (1, 0)],
    [(2, 1), (1, 0), (0, 1), (3, 1), (0, 0)],
    [(0, 0), (0, 1), (1, 0), (3, 1), (2, 0)],
    [(0, 0), (2, 0), (1, 0), (1, 1), (0, 1)],
    [(1, 0), (2, 0), (3, 0), (0, 0), (0, 1)],
    [(0, 1), (3, 1), (0, 0), (1, 0), (2, 0)],
    [(3, 1), (0, 1), (0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (0, 1), (3, 1), (2, 0)],
    [(1, 0), (3, 1), (2, 0), (0, 1), (0, 0)],
    [(0, 0), (2, 1), (0, 1), (3, 1), (1, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
]

permutations6 = [
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    [(3, 1), (2, 0), (0, 1), (1, 1), (0, 0), (1, 0)],
    [(0, 1), (1, 0), (4, 1), (2, 0), (0, 0), (3, 0)],
    [(0, 0), (3, 1), (0, 1), (1, 1), (2, 0), (1, 0)],
    [(0, 1), (1, 0), (0, 0), (2, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (1, 0), (2, 1), (1, 1), (2, 0)],
    [(0, 0), (2, 1), (3, 1), (1, 1), (1, 0), (0, 1)],
    [(0, 1), (3, 1), (0, 0), (2, 0), (1, 1), (1, 0)],
    [(0, 1), (2, 1), (1, 0), (3, 1), (0, 0), (4, 1)],
    [(0, 1), (1, 1), (0, 0), (2, 0), (3, 1), (1, 0)],
    [(2, 1), (3, 1), (0, 1), (1, 0), (0, 0), (4, 1)],
    [(0, 1), (1, 0), (3, 1), (2, 0), (4, 1), (0, 0)],
    [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
    [(0, 1), (0, 0), (1, 1), (3, 1), (2, 0), (1, 0)],
    [(2, 1), (0, 0), (1, 0), (3, 1), (4, 1), (0, 1)],
    [(2, 1), (0, 1), (3, 1), (1, 0), (0, 0), (4, 1)],
    [(3, 1), (1, 0), (2, 0), (0, 1), (4, 1), (0, 0)],
    [(1, 1), (2, 1), (3, 1), (0, 0), (0, 1), (4, 1)],
    [(2, 1), (0, 0), (0, 1), (3, 1), (1, 1), (1, 0)],
    [(0, 1), (0, 0), (2, 1), (3, 1), (1, 1), (1, 0)],
    [(0, 0), (1, 0), (3, 1), (2, 0), (1, 1), (0, 1)],
    [(2, 0), (0, 1), (1, 1), (1, 0), (3, 1), (0, 0)],
    [(0, 1), (1, 0), (1, 1), (2, 0), (0, 0), (3, 1)],
    [(0, 0), (3, 1), (4, 1), (0, 1), (1, 0), (2, 0)],
    [(0, 1), (0, 0), (1, 0), (3, 1), (4, 1), (2, 0)],
    [(1, 0), (2, 0), (0, 1), (3, 0), (4, 0), (0, 0)],
    [(0, 0), (2, 0), (1, 0), (0, 1), (3, 1), (1, 1)],
    [(0, 1), (1, 1), (2, 1), (0, 0), (3, 1), (4, 1)],
    [(0, 1), (0, 0), (1, 1), (1, 0), (2, 0), (3, 0)],
    [(1, 0), (0, 0), (2, 1), (3, 1), (4, 1), (0, 1)],
    [(1, 0), (0, 1), (2, 0), (3, 0), (0, 0), (4, 1)],
    [(2, 0), (0, 0), (1, 0), (3, 0), (1, 1), (0, 1)],
    [(0, 0), (1, 0), (0, 1), (1, 1), (3, 1), (2, 0)],
    [(1, 0), (0, 1), (3, 1), (4, 1), (0, 0), (2, 0)],
    [(1, 0), (3, 1), (2, 0), (4, 1), (0, 0), (0, 1)],
    [(0, 0), (1, 1), (3, 1), (0, 1), (1, 0), (2, 0)],
    [(0, 0), (1, 1), (0, 1), (1, 0), (2, 1), (2, 0)],
    [(0, 0), (3, 1), (0, 1), (4, 1), (1, 0), (2, 0)],
    [(0, 1), (3, 1), (4, 1), (1, 0), (0, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (3, 1), (4, 1)],
    [(4, 1), (0, 0), (1, 0), (2, 0), (2, 1), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)],
    [(2, 1), (1, 0), (3, 1), (0, 1), (4, 1), (0, 0)],
    [(2, 1), (0, 0), (3, 1), (1, 1), (1, 0), (0, 1)],
    [(0, 1), (0, 0), (1, 1), (2, 1), (1, 0), (3, 1)],
    [(3, 1), (4, 1), (0, 0), (1, 0), (4, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 1), (0, 1), (3, 1), (4, 1)],
    [(1, 0), (3, 1), (0, 1), (2, 0), (0, 0), (4, 1)],
    [(0, 1), (1, 0), (1, 1), (2, 1), (0, 0), (3, 1)],
    [(2, 0), (0, 0), (1, 0), (3, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (0, 0), (2, 0), (1, 1), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
]

permutationsN = [permutations4, permutations5, permutations6]

# -----------------------
# Plotting helpers
# -----------------------
def _get_first_present_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found. Looked for: {candidates}")

def plot_violin_by_codes(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "codes_used",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:

    missing = [c for c in (value_col, group_col) if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    clean = df[[group_col, value_col]].dropna()
    if clean.empty:
        raise ValueError("No data available after dropping NA values.")

    groups: Sequence[int] = sorted(clean[group_col].unique())
    datasets = [clean.loc[clean[group_col] == g, value_col].to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.violinplot(
        datasets,
        positions=np.arange(len(groups)) + 1,
        showmeans=True,
        showmedians=False,
        showextrema=True,
    )

    # overlay jittered points
    rng = np.random.default_rng(0)
    for i, data in enumerate(datasets, start=1):
        if len(data) == 0:
            continue
        x = np.full_like(data, i, dtype=float) + rng.normal(0, 0.03, size=len(data))
        ax.plot(x, data, linestyle="", marker="o", markersize=3, alpha=0.6)

    ax.set_xlabel("# of codes used")
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} by # codes used (Violin)")
    ax.set_xticks(np.arange(len(groups)) + 1)
    ax.set_xticklabels([str(g) for g in groups])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax

# -----------------------
# Run one circuit helper
# -----------------------
@dataclass
class CircuitSpec:
    n_qubits: int
    depth: int
    seed: int

def run_one_combo(
    spec: CircuitSpec,
    repo: RunRepository,
    out_root: str = "30_violins",
    target_basis: Sequence[str] = ('x', 'z', 'h', 'rz', 'cx', 'cz', 'measure'),
    sim_repetitions: int = 5_000,
    sim_alpha: float = 1.0,
    save_df_csv: bool = True,
) -> Tuple[RunRecord, str]:
    """
    Build random Clifford circuit, transpile, scan permutations, plot/save, and persist run.

    Returns:
        (record, combo_dir)
    """
    # --- 1) make circuit
    program_circuit: qiskit.QuantumCircuit = random_clifford_circuit(
        spec.n_qubits, spec.depth, seed=spec.seed
    )
    tcirc = qiskit.transpile(
        program_circuit,
        basis_gates=list(target_basis),
        optimization_level=0,
    )
    tcirc.measure_all()

    # --- 2) run specific scan
    config = PipelineConfig(
        sim_config=ScaledSimulator(alpha=sim_alpha, repetitions=sim_repetitions),
        verbose=False,
        save=False,
        seed=spec.seed,  # keep seed traceable
    )

    permutations = permutationsN[spec.n_qubits - 4]

    record: RunRecord = run_specific_scan(tcirc, permutations, config)
    df = record.dataframe.df

    # --- 3) plots
    combo_tag = f"n{spec.n_qubits}_d{spec.depth}_seed{spec.seed}"
    combo_dir = os.path.join(out_root, combo_tag)
    os.makedirs(combo_dir, exist_ok=True)

    # Columns: TVD and Logical Selection Rate (with fallbacks)
    tvd_col = _get_first_present_col(
        df,
        candidates=["logical_total_variation_distance", "TVD", "tvd"],
    )
    sel_col = _get_first_present_col(
        df,
        candidates=["logical_selection_rate", "selection_rate", "selection rate"],
    )

    common_title = f"{combo_tag} | perms={len(permutations)} | reps={record.manifest.repetitions}"

    # TVD violin
    plot_violin_by_codes(
        df=df,
        value_col=tvd_col,
        group_col="codes_used",
        title=f"TVD by Codes Used — {common_title}",
        save_path=os.path.join(combo_dir, "tvd_violin.png"),
    )
    plt.close()  # free memory in batch runs

    # Selection Rate violin
    plot_violin_by_codes(
        df=df,
        value_col=sel_col,
        group_col="codes_used",
        title=f"Logical Selection Rate by Codes Used — {common_title}",
        save_path=os.path.join(combo_dir, "selection_rate_violin.png"),
    )
    plt.close()

    # --- 4) save scan (RunRepository) and CSV
    saved_path = repo.save_run(record, compress=True, path=combo_dir)
    # Also drop a small metadata file in combo_dir for quick lookup
    with open(os.path.join(combo_dir, "saved_run_path.txt"), "w", encoding="utf-8") as f:
        f.write(saved_path + "\n")

    if save_df_csv:
        df.to_csv(os.path.join(combo_dir, "scan_dataframe.csv"), index=False)

    return record, combo_dir

# -----------------------
# Batch driver
# -----------------------
def main():
    out_root = "30_violins"
    os.makedirs(out_root, exist_ok=True)

    repo = RunRepository(base_dir="30_violins/")

    rng = np.random.default_rng(42)  # reproducible top-level RNG

    tasks: List[CircuitSpec] = []
    for nq in [4, 5, 6]:
        # 10 circuits each; depth uniform in [30, 60]
        depths = rng.integers(low=30, high=61, size=10)  # 61 is exclusive upper bound
        seeds = rng.integers(low=1, high=2**31 - 1, size=10)
        tasks.extend(CircuitSpec(n_qubits=nq, depth=int(d), seed=int(s)) for d, s in zip(depths, seeds))

    # Run them
    for spec in tasks:
        try:
            _, combo_dir = run_one_combo(
                spec=spec,
                repo=repo,
                out_root=out_root,
                target_basis=('x', 'z', 'h', 'rz', 'cx', 'cz', 'measure'),
                sim_repetitions=50_000,
                sim_alpha=1.0,
                save_df_csv=True,
            )
            print(f"Completed: {combo_dir}")
        except Exception as e:
            print(f"Failed for {spec}: {e}")

if __name__ == "__main__":
    main()