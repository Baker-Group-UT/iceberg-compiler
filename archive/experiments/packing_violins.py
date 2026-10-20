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
import ast
sys.path.append("../")

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List, Dict, Any

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
from archive.packer.merge_packer import MergeMaximizingPacker

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


def _perm_key(perm: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(p), int(s)) for p, s in perm)


def _parse_perm_value(value: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return None
        return _perm_key(parsed)
    if isinstance(value, Sequence):
        return _perm_key(value)
    return None


def _complete_qubit_mapping(
    mapping: Dict[int, int],
    num_qubits: int,
    patch_capacity: int,
    max_patches: int,
) -> Dict[int, int]:
    """
    Ensure every logical qubit has a patch assignment by reusing or allocating
    additional patches (up to max_patches).
    """
    completed: Dict[int, int] = dict(mapping)
    if len(completed) >= num_qubits:
        # Already have full coverage.
        return completed

    if max_patches is None:
        max_patches = num_qubits

    patch_usage: Dict[int, int] = {}
    for patch in completed.values():
        patch_usage[patch] = patch_usage.get(patch, 0) + 1
    used_patch_ids = set(patch_usage.keys())
    next_patch_id = 0

    def allocate_new_patch() -> int:
        nonlocal next_patch_id
        if len(used_patch_ids) >= max_patches:
            raise ValueError(
                f"Cannot allocate more than {max_patches} patches for {num_qubits} qubits."
            )
        while next_patch_id in used_patch_ids:
            next_patch_id += 1
        patch_id = next_patch_id
        used_patch_ids.add(patch_id)
        patch_usage[patch_id] = 0
        next_patch_id += 1
        return patch_id

    for qubit in range(num_qubits):
        if qubit in completed:
            continue
        target_patch = None
        for patch_id in sorted(used_patch_ids):
            if patch_usage.get(patch_id, 0) < patch_capacity:
                target_patch = patch_id
                break
        if target_patch is None:
            target_patch = allocate_new_patch()
        completed[qubit] = target_patch
        patch_usage[target_patch] = patch_usage.get(target_patch, 0) + 1

    return completed


def _highlight_packer_choices(
    ax: plt.Axes,
    group_positions: Dict[int, float],
    choices: List[Dict[str, Any]],
    value_field: str,
    label: str,
) -> None:
    if not choices:
        return
    added_label = {False: False, True: False}
    for choice in choices:
        codes_used = choice.get("codes_used")
        value = choice.get(value_field)
        if codes_used is None or pd.isna(codes_used):
            continue
        xpos = group_positions.get(int(codes_used))
        if xpos is None or value is None or pd.isna(value):
            continue
        is_new = bool(choice.get("is_new"))
        color = "orange" if is_new else "crimson"
        marker = "D" if is_new else "x"
        plot_label = f"{label} (new)" if is_new else label
        ax.scatter(
            xpos,
            value,
            color=color,
            marker=marker,
            s=70,
            linewidths=2,
            zorder=5,
            label=plot_label if not added_label[is_new] else None,
        )
        added_label[is_new] = True
    if any(added_label.values()):
        ax.legend()

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
) -> Tuple[plt.Figure, plt.Axes, Dict[int, float]]:

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
    positions = np.arange(len(groups)) + 1
    ax.set_xticks(positions)
    ax.set_xticklabels([str(g) for g in groups])
    group_positions = {int(group): float(pos) for group, pos in zip(groups, positions)}
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax, group_positions

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
    tcirc: qiskit.QuantumCircuit = random_clifford_circuit(
        spec.n_qubits, spec.depth, seed=spec.seed, gates=['h', 'z', 's', 'x', 'cz', 'cx']
    )
    # tcirc = qiskit.transpile(
    #     program_circuit,
    #     basis_gates=list(target_basis),
    #     optimization_level=0,
    # )
    tcirc.measure_all()

    # --- 2) run specific scan
    config = PipelineConfig(
        sim_config=ScaledSimulator(alpha=sim_alpha, repetitions=sim_repetitions),
        verbose=False,
        save=False,
        seed=spec.seed,  # keep seed traceable
    )

    permutations = list(permutationsN[spec.n_qubits - 4])
    existing_keys = {_perm_key(perm) for perm in permutations}

    packer_plan: List[Dict[str, Any]] = []
    codes_values = sorted(
        {len({patch for patch, _ in perm}) for perm in permutations}
    )
    print("code values:")
    print(codes_values)
    for max_codes in codes_values:
        try:
            packer = MergeMaximizingPacker()
            packing = packer.pack(tcirc, num_codes=int(max_codes))
            permutation = packing.permutation
            perm_key = _perm_key(permutation)
        except Exception as exc:
            print(f"[packer] Failed to produce permutation for codes={max_codes}: {exc}")
            continue
        is_new = perm_key not in existing_keys
        if is_new:
            permutations.append(list(permutation))
            existing_keys.add(perm_key)
            print(f"[packer] Added heuristic permutation for codes={max_codes}: {permutation}")
        packer_plan.append(
            {
                "codes_used": max_codes,
                "perm_key": perm_key,
                "is_new": is_new,
            }
        )

    record: RunRecord = run_specific_scan(tcirc, permutations, config)
    df = record.dataframe.df.copy()

    perm_index_map: Dict[Tuple[Tuple[int, int], ...], Any] = {}
    perm_keys = [_perm_key(p) for p in permutations]
    if len(df) == len(perm_keys):
        df["permutation_key"] = perm_keys
        for pos, key in enumerate(perm_keys):
            perm_index_map[key] = df.index[pos]
    elif "permutation_key" in df.columns:
        for idx, value in df["permutation_key"].items():
            key = _parse_perm_value(value)
            if key is not None:
                perm_index_map[key] = idx
    else:
        df["permutation_key"] = [None] * len(df)

    df["packer_choice"] = False

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

    packer_choices: List[Dict[str, Any]] = []
    for plan in packer_plan:
        perm_key = plan.get("perm_key")
        row_idx = perm_index_map.get(perm_key) if perm_key is not None else None
        if row_idx is None:
            print(f"[packer] Permutation not found in scan for codes={plan.get('codes_used')}: {perm_key}")
            continue
        df.at[row_idx, "packer_choice"] = True
        packer_choices.append(
            {
                "codes_used": df.at[row_idx, "codes_used"],
                "tvd": df.at[row_idx, tvd_col] if tvd_col in df.columns else None,
                "selection": df.at[row_idx, sel_col] if sel_col in df.columns else None,
                "permutation_key": perm_key,
                "max_patches": plan.get("codes_used"),
                "is_new": plan.get("is_new", False),
            }
        )

    common_title = f"{combo_tag} | perms={len(permutations)} | reps={record.manifest.repetitions}"

    # TVD violin
    fig, ax, group_pos = plot_violin_by_codes(
        df=df,
        value_col=tvd_col,
        group_col="codes_used",
        title=f"TVD by Codes Used — {common_title}",
        save_path=None,
    )
    _highlight_packer_choices(ax, group_pos, packer_choices, "tvd", label="Packer choice")
    fig.savefig(os.path.join(combo_dir, "tvd_violin.svg"), dpi=200, bbox_inches="tight")
    plt.close()  # free memory in batch runs

    # Selection Rate violin
    fig, ax, group_pos = plot_violin_by_codes(
        df=df,
        value_col=sel_col,
        group_col="codes_used",
        title=f"Logical Selection Rate by Codes Used — {common_title}",
        save_path=None,
    )
    _highlight_packer_choices(ax, group_pos, packer_choices, "selection", label="Packer choice")
    fig.savefig(os.path.join(combo_dir, "selection_rate_violin.svg"), dpi=200, bbox_inches="tight")
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
    out_root = "packing_violins"
    os.makedirs(out_root, exist_ok=True)

    repo = RunRepository(base_dir="packing_violins/")

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
                sim_repetitions=50,
                sim_alpha=1.0,
                save_df_csv=True,
            )
            print(f"Completed: {combo_dir}")
        except Exception as e:
            print(f"Failed for {spec}: {e}")

if __name__ == "__main__":
    main()
