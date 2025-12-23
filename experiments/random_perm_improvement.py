"""
Random permutation scan experiment that estimates how much better the best
permutation performs compared to the average and worst permutations sampled
from a random subset.

The experiment runs several random permutations scans (using
``pipeline.run_random_scan``) for logical qubit counts in a configurable range
and reports:
  * Percent improvement of the best permutation vs. the average permutation.
  * Percent improvement of the best permutation vs. the worst permutation.
  * Whether the winning permutation matches the trivial sparse packing
    (each qubit on its own code, slot 0).

After all scans finish, aggregate averages and the fraction of runs whose best
permutation was *not* the trivial sparse packing are printed.

Usage (defaults to q=4..6, 4 runs each):

    python -m experiments.random_perm_improvement
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import qiskit
from qiskit.circuit.random import random_clifford_circuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline import PipelineConfig, run_full_scan, run_random_scan
from simulator.config_presets import StandardSimulator

Perm = List[Tuple[int, int]]
PermKey = Tuple[Tuple[int, int], ...]


def generate_sparse_perm(q: int) -> Perm:
    """
    Sparse packing = each logical qubit occupies its own patch in slot 0.
    """
    return [(i, 0) for i in range(q)]


def perm_to_tuple(perm_like) -> PermKey:
    """
    Normalize permutations to a tuple-of-tuples for consistent comparison.
    """
    return tuple((int(a), int(b)) for a, b in list(perm_like))


def error_reduction_metrics(best: float, ref: float) -> Tuple[float, float]:
    """
    Compute percent error reduction and improvement factor using logical errors
    (1 - fidelity) instead of fidelities directly.
    """
    err_best = max(0.0, 1.0 - best)
    err_ref = max(0.0, 1.0 - ref)

    if err_ref == 0:
        pct = float("nan")
    else:
        pct = 100.0 * (err_ref - err_best) / err_ref

    if err_best == 0:
        factor = float("inf") if err_ref > 0 else float("nan")
    else:
        factor = err_ref / err_best

    return pct, factor


@dataclass
class ExperimentConfig:
    qubit_counts: Sequence[int] = (4, 5, 6)
    scans_per_qubit: int = 4
    circuit_depth: int = 20
    permutation_samples: int = 40
    repetitions: int = 5000
    target_basis: Sequence[str] = ("x", "z", "h", "rz", "cx", "cz", "measure")
    random_seed: Optional[int] = 17
    metric_column: str = "logical_fidelity"


@dataclass
class ScanMetrics:
    qubits: int
    scan_index: int
    permutations_evaluated: int
    best_fidelity: float
    avg_fidelity: float
    worst_fidelity: float
    best_over_avg_pct: float
    best_over_worst_pct: float
    best_over_avg_factor: float
    best_over_worst_factor: float
    best_perm: PermKey
    best_codes_used: int
    best_is_trivial_sparse: bool
    circuit_seed: Optional[int]


def build_random_circuit(
    q: int,
    depth: int,
    target_basis: Iterable[str],
    seed: Optional[int] = None,
) -> qiskit.QuantumCircuit:
    """
    Random logical circuit helper used for each scan.
    """
    circuit = random_clifford_circuit(q, depth, seed=seed, gates=["cx", 'cz', 's', 'z', 'h', 'x'])
    circuit.measure_all()
    # transpiled = qiskit.transpile(
    #     circuit,
    #     basis_gates=list(target_basis),
    #     optimization_level=0,
    # )
    return circuit


def compute_scan_metrics(
    df: pd.DataFrame,
    q: int,
    metric_col: str,
    scan_index: int,
    circuit_seed: Optional[int],
) -> ScanMetrics:
    """
    Given a dataframe from a permutation scan, extract best/avg/worst metrics.
    """
    if metric_col not in df.columns:
        raise KeyError(f"Metric column '{metric_col}' missing from dataframe.")

    df = df.dropna(subset=[metric_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Permutation scan produced no valid rows.")

    best_idx = df[metric_col].idxmax()
    worst_idx = df[metric_col].idxmin()

    best_val = float(df.loc[best_idx, metric_col])
    worst_val = float(df.loc[worst_idx, metric_col])
    avg_val = float(df[metric_col].mean())

    best_perm = perm_to_tuple(df.loc[best_idx, "perm_key"])
    sparse_perm = perm_to_tuple(generate_sparse_perm(q))
    codes_used = int(df.loc[best_idx, "codes_used"])

    avg_pct, avg_factor = error_reduction_metrics(best_val, avg_val)
    worst_pct, worst_factor = error_reduction_metrics(best_val, worst_val)

    return ScanMetrics(
        qubits=q,
        scan_index=scan_index,
        permutations_evaluated=len(df),
        best_fidelity=best_val,
        avg_fidelity=avg_val,
        worst_fidelity=worst_val,
        best_over_avg_pct=avg_pct,
        best_over_worst_pct=worst_pct,
        best_over_avg_factor=avg_factor,
        best_over_worst_factor=worst_factor,
        best_perm=best_perm,
        best_codes_used=codes_used,
        best_is_trivial_sparse=(best_perm == sparse_perm),
        circuit_seed=circuit_seed,
    )


def run_single_scan(
    q: int,
    scan_index: int,
    exp_cfg: ExperimentConfig,
    pipeline_config: PipelineConfig,
) -> ScanMetrics:
    """
    Execute a random permutation scan for a single qubit count.
    """
    seed = None
    if exp_cfg.random_seed is not None:
        seed = exp_cfg.random_seed + q * 1000 + scan_index

    circuit = build_random_circuit(
        q=q,
        depth=exp_cfg.circuit_depth,
        target_basis=exp_cfg.target_basis,
        seed=seed,
    )
    record = run_random_scan(
        program_circuit=circuit,
        m=exp_cfg.permutation_samples,
        config=pipeline_config,
    )
    if record.dataframe is None:
        raise RuntimeError("Run record missing dataframe.")
    df = record.dataframe.df.copy()
    return compute_scan_metrics(
        df=df,
        q=q,
        metric_col=exp_cfg.metric_column,
        scan_index=scan_index,
        circuit_seed=seed,
    )


def summarize_metrics(metrics: Sequence[ScanMetrics]) -> dict:
    """
    Aggregate summary of per-scan metrics.
    """
    best_avg = np.array([m.best_over_avg_pct for m in metrics], dtype=float)
    best_worst = np.array([m.best_over_worst_pct for m in metrics], dtype=float)
    best_avg_factors = np.array([m.best_over_avg_factor for m in metrics], dtype=float)
    best_worst_factors = np.array([m.best_over_worst_factor for m in metrics], dtype=float)

    trivial_count = sum(1 for m in metrics if m.best_is_trivial_sparse)
    total = len(metrics)

    return {
        "avg_best_over_avg_pct": float(np.nanmean(best_avg)),
        "avg_best_over_worst_pct": float(np.nanmean(best_worst)),
        "avg_best_over_avg_factor": float(np.nanmean(best_avg_factors)),
        "avg_best_over_worst_factor": float(np.nanmean(best_worst_factors)),
        "pct_trivial_best_perm": 100.0 * trivial_count / total if total else float("nan"),
        "pct_non_trivial_best_perm": 100.0 * (total - trivial_count) / total if total else float("nan"),
    }


def metrics_dataframe(metrics: Sequence[ScanMetrics]) -> pd.DataFrame:
    """
    Convert ScanMetrics objects into a pandas DataFrame for optional analysis.
    """
    rows = []
    for m in metrics:
        rows.append(
            {
                "q": m.qubits,
                "scan_index": m.scan_index,
                "perms": m.permutations_evaluated,
                "best_fidelity": m.best_fidelity,
                "avg_fidelity": m.avg_fidelity,
                "worst_fidelity": m.worst_fidelity,
                "best_over_avg_pct": m.best_over_avg_pct,
                "best_over_worst_pct": m.best_over_worst_pct,
                "best_over_avg_factor": m.best_over_avg_factor,
                "best_over_worst_factor": m.best_over_worst_factor,
                "best_codes_used": m.best_codes_used,
                "best_is_trivial_sparse": m.best_is_trivial_sparse,
                "circuit_seed": m.circuit_seed,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    exp_cfg = ExperimentConfig()
    pipeline_config = PipelineConfig(
        sim_config=StandardSimulator(repetitions=exp_cfg.repetitions).with_stim().with_distance_based_noise(),
        verbose=False,
        fault_tolerant=False,
        save=False,
    )

    all_metrics: List[ScanMetrics] = []
    for q in exp_cfg.qubit_counts:
        for scan_idx in range(exp_cfg.scans_per_qubit):
            metrics = run_single_scan(
                q=q,
                scan_index=scan_idx,
                exp_cfg=exp_cfg,
                pipeline_config=pipeline_config,
            )
            all_metrics.append(metrics)
            avg_factor = (
                f"{metrics.best_over_avg_factor:.2f}×"
                if np.isfinite(metrics.best_over_avg_factor)
                else "inf"
            )
            worst_factor = (
                f"{metrics.best_over_worst_factor:.2f}×"
                if np.isfinite(metrics.best_over_worst_factor)
                else "inf"
            )
            print(
                f"[q={q} scan={scan_idx}] "
                f"error best-avg: {metrics.best_over_avg_pct:.2f}% ({avg_factor}) | "
                f"error best-worst: {metrics.best_over_worst_pct:.2f}% ({worst_factor}) | "
                f"perms={metrics.permutations_evaluated} | "
                f"codes_used={metrics.best_codes_used} | "
                f"is_trivial_sparse={metrics.best_is_trivial_sparse}"
            )

    if not all_metrics:
        print("No scans were executed.")
        return

    summary = summarize_metrics(all_metrics)
    print("\nAggregate summary:")
    print(f"Average best-over-average error reduction: {summary['avg_best_over_avg_pct']:.2f}% "
          f"({summary['avg_best_over_avg_factor']:.2f}×)")
    print(f"Average best-over-worst error reduction:   {summary['avg_best_over_worst_pct']:.2f}% "
          f"({summary['avg_best_over_worst_factor']:.2f}×)")
    print(f"Best perm was trivial sparse in:        {summary['pct_trivial_best_perm']:.1f}% of scans")
    print(f"Best perm was non-trivial in:           {summary['pct_non_trivial_best_perm']:.1f}% of scans")

    df = metrics_dataframe(all_metrics)
    print("\nPer-scan metrics:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
