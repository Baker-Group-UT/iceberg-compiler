"""
spacetime_reduction_benchmark.py
--------------------------------

Structured-circuit variant of spacetime reduction (no simulation).
Compares merge-maximizing permutation vs the average of random permutations
for the same code count, across a provided set of circuits.

Plots percent reductions vs fraction of codes used (n_codes / n_qubits).
Run with:

    python -m experiments.figures.spacetime_reduction.spacetime_reduction_benchmark --qasm-root <path>
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import qiskit
from tqdm import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from archive.packer.merge_packer import MergeMaximizingPacker
from permutations.generator import permutation_with_n_codes
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
    random_perms_per_code: int = 10
    random_seed: int | None = 7
    max_proxy_gates: int | None = 500
    max_proxy_qubits: int | None = None
    proxy_timeout_seconds: float | None = 1.0
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_benchmark.svg"
        )
    )
    show_plot: bool = True

    def __post_init__(self) -> None:
        if self.random_perms_per_code <= 0:
            raise ValueError("random_perms_per_code must be >= 1.")


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    delta = reference - challenger
    return 100.0 * delta / reference


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
    rng = random.Random(settings.random_seed)
    packer = MergeMaximizingPacker()

    rows: List[Dict[str, int | float]] = []

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

            random_one_q_values: List[int] = []
            random_two_q_values: List[int] = []
            random_depth_values: List[int] = []
            for _ in range(settings.random_perms_per_code):
                random_perm_seed = (
                    rng.randint(0, 2**31 - 1) if settings.random_seed is not None else None
                )
                random_perm = permutation_with_n_codes(
                    q=n_qubits,
                    n_codes=n_codes,
                    seed=random_perm_seed,
                )
                random_one_q, random_two_q, random_depth = compute_metrics(
                    program_circuit, random_perm, config
                )
                random_one_q_values.append(random_one_q)
                random_two_q_values.append(random_two_q)
                random_depth_values.append(random_depth)

            avg_random_one_q = sum(random_one_q_values) / len(random_one_q_values)
            avg_random_two_q = sum(random_two_q_values) / len(random_two_q_values)
            avg_random_depth = sum(random_depth_values) / len(random_depth_values)

            rows.append(
                {
                    "circuit_idx": circuit_idx,
                    "circuit_name": circuit_name,
                    "n_qubits": n_qubits,
                    "target_codes": n_codes,
                    "fraction_codes_used": n_codes / n_qubits,
                    "packed_one_qubit_gates": packed_one_q,
                    "avg_random_one_qubit_gates": avg_random_one_q,
                    "pct_reduction_one_qubit_gates": pct_reduction(
                        reference=avg_random_one_q, challenger=packed_one_q
                    ),
                    "packed_two_qubit_gates": packed_two_q,
                    "avg_random_two_qubit_gates": avg_random_two_q,
                    "pct_reduction_two_qubit_gates": pct_reduction(
                        reference=avg_random_two_q, challenger=packed_two_q
                    ),
                    "packed_circuit_depth": packed_depth,
                    "avg_random_circuit_depth": avg_random_depth,
                    "pct_reduction_circuit_depth": pct_reduction(
                        reference=avg_random_depth, challenger=packed_depth
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, settings: BenchmarkSettings) -> None:
    fractions = sorted(df["fraction_codes_used"].unique())
    circuit_ids = sorted(df["circuit_idx"].unique())
    circuit_labels = {
        circuit_id: df.loc[df["circuit_idx"] == circuit_id, "circuit_name"].iloc[0]
        for circuit_id in circuit_ids
    }

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
    cmap = plt.get_cmap("tab20", max(len(circuit_ids), 2))
    for ax, (metric, title) in zip(axes, METRICS):
        # Only plot per-circuit lines for clarity (no violin/box plots).

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

    fig.suptitle("Gate/depth reductions: merge-maximizing vs avg random permutations")
    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[spacetime_reduction_benchmark] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> tuple[Path, BenchmarkSettings]:
    parser = argparse.ArgumentParser(
        description="Spacetime reduction benchmark over structured circuits."
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        default=Path(
            "/Users/cordputer/Documents/Projects/School/Research/Infleqtion/infleqtion-qed-exploration-main/benchmarks/qasmbench/small"
        ),
        help="Root directory containing benchmark QASM circuits.",
    )
    parser.add_argument(
        "--random-perms",
        type=int,
        default=100,
        help="Number of random permutations per codes value.",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=2000,
        help="Exclude proxy circuits with more than this many gates (post-transpile).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=100,
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
            "experiments/figures/spacetime_reduction/spacetime_reduction_benchmark.svg"
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
        random_perms_per_code=args.random_perms,
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
