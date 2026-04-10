"""
spacetime_reduction_benchmark_naive.py
--------------------------------------

Structured-circuit variant that compares merge-maximizing permutations against
two naive baselines (tight and sparse packings) instead of random permutations.

Plots percent reductions vs fraction of codes used (n_codes / n_qubits).
Run with:

    python -m experiments.figures.spacetime_reduction.spacetime_reduction_benchmark_naive
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import qiskit
from tqdm import tqdm

from evaluator.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from packer.merge_packer import MergeMaximizingPacker
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]


METRICS = [
    ("pct_reduction_one_qubit_gates", "Percent reduction in #1q gates"),
    ("pct_reduction_two_qubit_gates", "Percent reduction in #2q gates"),
    ("pct_reduction_circuit_depth", "Percent reduction in depth"),
]


@dataclass
class BenchmarkSettings:
    random_seed: int | None = 7
    max_proxy_gates: int | None = 500
    max_proxy_qubits: int | None = None
    proxy_timeout_seconds: float | None = 1.0
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_benchmark_naive.svg"
        )
    )
    show_plot: bool = True


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    delta = reference - challenger
    return 100.0 * delta / reference


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    patch_capacity = max(1, math.ceil(num_qubits / n_codes))
    perm: Perm = []
    for i in range(num_qubits):
        code = min(i // patch_capacity, n_codes - 1)
        perm.append((code, i % patch_capacity))
    return perm


def sparse_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    perm: Perm = []
    for i in range(num_qubits):
        code = i % n_codes
        offset = i // n_codes
        perm.append((code, offset))
    return perm


def compute_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Tuple[int, int, int]:
    translated = translate_permutation(program_circuit, permutation, config=config)
    circuit = translated.circuit
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    return one_q, two_q, depth


def run_experiment(
    circuits: Sequence[Tuple[str, qiskit.QuantumCircuit]],
    settings: BenchmarkSettings,
) -> pd.DataFrame:
    config = PipelineConfig()
    packer = MergeMaximizingPacker()

    rows: List[Dict[str, int | float | str]] = []

    for circuit_idx, (circuit_name, program_circuit) in tqdm(
        list(enumerate(circuits)),
        desc="circuits",
    ):
        n_qubits = program_circuit.num_qubits
        min_codes = math.ceil(n_qubits / 2)
        max_codes = n_qubits

        for n_codes in tqdm(
            range(min_codes, max_codes + 1),
            desc="codes",
            leave=False,
        ):
            packed_perm = packer.pack(program_circuit, num_codes=n_codes).permutation
            packed_one_q, packed_two_q, packed_depth = compute_metrics(
                program_circuit, packed_perm, config
            )

            baselines = {
                "tight": tight_trivial_packing(n_qubits, n_codes),
                "sparse": sparse_trivial_packing(n_qubits, n_codes),
            }

            for baseline_name, baseline_perm in baselines.items():
                base_one_q, base_two_q, base_depth = compute_metrics(
                    program_circuit, baseline_perm, config
                )

                rows.append(
                    {
                        "circuit_idx": circuit_idx,
                        "circuit_name": circuit_name,
                        "baseline": baseline_name,
                        "n_qubits": n_qubits,
                        "target_codes": n_codes,
                        "fraction_codes_used": n_codes / n_qubits,
                        "packed_one_qubit_gates": packed_one_q,
                        "baseline_one_qubit_gates": base_one_q,
                        "pct_reduction_one_qubit_gates": pct_reduction(
                            reference=base_one_q, challenger=packed_one_q
                        ),
                        "packed_two_qubit_gates": packed_two_q,
                        "baseline_two_qubit_gates": base_two_q,
                        "pct_reduction_two_qubit_gates": pct_reduction(
                            reference=base_two_q, challenger=packed_two_q
                        ),
                        "packed_circuit_depth": packed_depth,
                        "baseline_circuit_depth": base_depth,
                        "pct_reduction_circuit_depth": pct_reduction(
                            reference=base_depth, challenger=packed_depth
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _plot_panel(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    circuit_ids = sorted(df["circuit_idx"].unique())
    circuit_labels = {
        circuit_id: df.loc[df["circuit_idx"] == circuit_id, "circuit_name"].iloc[0]
        for circuit_id in circuit_ids
    }
    cmap = plt.get_cmap("tab20", max(len(circuit_ids), 2))

    tick_fracs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(tick_fracs)
    ax.set_xticklabels([f"{frac:.0%}" for frac in tick_fracs], rotation=0, ha="center")
    ax.set_xlabel("Fraction of codes used")
    ax.set_ylabel("Percent reduction")
    ax.set_title(title)
    ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
    ax.set_xlim(0.5, 1.0)

    for idx, circuit_id in enumerate(circuit_ids):
        circuit_df = df.loc[df["circuit_idx"] == circuit_id].sort_values("fraction_codes_used")
        ax.plot(
            circuit_df["fraction_codes_used"].values,
            circuit_df[metric].values,
            color=cmap(idx),
            linewidth=1.5,
            alpha=0.9,
            label=circuit_labels.get(circuit_id, f"circuit {circuit_id}"),
        )


def plot_results(df: pd.DataFrame, settings: BenchmarkSettings) -> None:
    circuit_ids = sorted(df["circuit_idx"].unique())
    circuit_labels = {
        circuit_id: df.loc[df["circuit_idx"] == circuit_id, "circuit_name"].iloc[0]
        for circuit_id in circuit_ids
    }

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 11))

    sparse_df = df.loc[df["baseline"] == "sparse"]
    tight_df = df.loc[df["baseline"] == "tight"]

    for ax, (metric, title) in zip(axes[0], METRICS):
        _plot_panel(ax, sparse_df, metric, f"{title} (sparse baseline)")

    for ax, (metric, title) in zip(axes[1], METRICS):
        _plot_panel(ax, tight_df, metric, f"{title} (tight baseline)")

    fig.suptitle("Gate/depth reductions: merge-maximizing vs naive permutations")
    axes[0][0].legend(loc="best", fontsize=7)
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[spacetime_reduction_benchmark_naive] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> tuple[Path, BenchmarkSettings]:
    parser = argparse.ArgumentParser(
        description="Spacetime reduction benchmark using naive baselines."
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        default=Path(
            "/Users/cordputer/Documents/Projects/School/Research/Infleqtion/infleqtion-qed-exploration-main/benchmarks/qasmbench/medium"
        ),
        help="Root directory containing benchmark QASM circuits.",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=500,
        help="Exclude proxy circuits with more than this many gates (post-transpile).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Exclude proxy circuits with more than this many qubits (from README metadata).",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for proxy circuit creation (set <=0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Base seed for permutation randomness (set negative for stochastic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_benchmark_naive.svg"
        ),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    seed_value = args.seed if args.seed >= 0 else None
    settings = BenchmarkSettings(
        random_seed=seed_value,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        output_path=args.output,
        show_plot=not args.no_show,
    )
    return args.qasm_root, settings


def main() -> None:
    qasm_root, settings = parse_args()
    timeout_seconds = settings.proxy_timeout_seconds
    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None
    circuits = load_proxy_circuits(
        qasm_root,
        gate_limit=2000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
    )
    if settings.max_proxy_gates is not None:
        circuits = [
            (name, qc)
            for name, qc in circuits
            if qc.size() <= settings.max_proxy_gates
        ]
    if not circuits:
        raise ValueError(f"No proxy circuits loaded from {qasm_root}")
    df = run_experiment(circuits, settings)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
