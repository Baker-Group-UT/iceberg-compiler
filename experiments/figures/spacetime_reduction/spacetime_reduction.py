"""
spacetime_reduction.py
----------------------

Analyze 1q/2q gate count and depth reductions without simulation.

For K random circuits of the same size, evaluate code counts from
ceil(n/2) to n-1. For each code count, compare the merge-maximizing
packer (pair_then_solo) against M random permutations of the same size.

Run with:

    python -m experiments.figures.spacetime_reduction.spacetime_reduction

Use --help for overrides.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import qiskit
from qiskit.circuit.random import random_clifford_circuit
from tqdm import tqdm

from evaluator.gate_count_stats import one_two_qubit_gate_totals
from packer.merge_packer import MergeMaximizingPacker
from permutations.generator import permutation_with_n_codes
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]

GATE_SET = ["x", "z", "h", "cx"]


@dataclass
class ExperimentSettings:
    n_qubits: int = 20
    n_gates: int = 200
    num_circuits: int = 10
    random_perms_per_code: int = 10
    random_seed: int | None = 7
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_boxplot.png"
        )
    )
    show_plot: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2.")
        if self.n_gates < 1:
            raise ValueError("n_gates must be >= 1.")
        if self.num_circuits <= 0:
            raise ValueError("num_circuits must be >= 1.")
        if self.random_perms_per_code <= 0:
            raise ValueError("random_perms_per_code must be >= 1.")


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    delta = reference - challenger
    return 100.0 * delta / reference


def build_random_circuit(
    n_qubits: int,
    n_gates: int,
    seed: int | None,
) -> qiskit.QuantumCircuit:
    circuit = random_clifford_circuit(
        n_qubits,
        n_gates,
        # seed=seed,
        gates=GATE_SET,
    )
    circuit.measure_all()
    return circuit


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


def run_experiment(settings: ExperimentSettings) -> pd.DataFrame:
    config = PipelineConfig()
    rng = random.Random(settings.random_seed)
    packer = MergeMaximizingPacker(solo_strategy="pair_then_solo")

    rows: List[Dict[str, int | float]] = []
    min_codes = math.ceil(settings.n_qubits / 2)
    max_codes = settings.n_qubits - 1

    for circuit_idx in tqdm(range(settings.num_circuits), desc="circuits"):
        circuit_seed = None
        if settings.random_seed is not None:
            circuit_seed = settings.random_seed * 10_000 + circuit_idx
        program_circuit = build_random_circuit(
            n_qubits=settings.n_qubits,
            n_gates=settings.n_gates,
            seed=circuit_seed,
        )

        for n_codes in tqdm(
            range(min_codes, max_codes + 1),
            desc="codes",
            leave=False,
        ):
            packed_perm = packer.pack(program_circuit, num_codes=n_codes).permutation
            packed_one_q, packed_two_q, packed_depth = compute_metrics(
                program_circuit, packed_perm, config
            )

            for _ in range(settings.random_perms_per_code):
                random_perm_seed = rng.randint(0, 2**31 - 1) if settings.random_seed is not None else None
                random_perm = permutation_with_n_codes(
                    q=settings.n_qubits,
                    n_codes=n_codes,
                    seed=random_perm_seed,
                )
                random_one_q, random_two_q, random_depth = compute_metrics(
                    program_circuit, random_perm, config
                )

                rows.append(
                    {
                        "circuit_idx": circuit_idx,
                        "target_codes": n_codes,
                        "codes_used": len({p for p, _ in packed_perm}),
                        "packed_one_qubit_gates": packed_one_q,
                        "random_one_qubit_gates": random_one_q,
                        "pct_reduction_one_qubit_gates": pct_reduction(
                            reference=random_one_q, challenger=packed_one_q
                        ),
                        "packed_two_qubit_gates": packed_two_q,
                        "random_two_qubit_gates": random_two_q,
                        "pct_reduction_two_qubit_gates": pct_reduction(
                            reference=random_two_q, challenger=packed_two_q
                        ),
                        "packed_circuit_depth": packed_depth,
                        "random_circuit_depth": random_depth,
                        "pct_reduction_circuit_depth": pct_reduction(
                            reference=random_depth, challenger=packed_depth
                        ),
                    }
                )

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    metrics = [
        ("pct_reduction_one_qubit_gates", "Percent reduction in #1q gates"),
        ("pct_reduction_two_qubit_gates", "Percent reduction in #2q gates"),
        ("pct_reduction_circuit_depth", "Percent reduction in depth"),
    ]
    codes = sorted(df["codes_used"].unique())

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    for ax, (metric, title) in zip(axes, metrics):
        data = [df.loc[df["codes_used"] == code, metric].values for code in codes]
        bp = ax.boxplot(data, labels=[str(code) for code in codes], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#a6bddb", edgecolor="#2b8cbe", alpha=0.8)
        for whisker in bp["whiskers"]:
            whisker.set(color="#2b8cbe")
        for cap in bp["caps"]:
            cap.set(color="#2b8cbe")
        for median in bp["medians"]:
            median.set(color="#045a8d", linewidth=2)
        ax.set_xlabel("Codes used")
        ax.set_ylabel("Percent reduction")
        ax.set_title(title)
        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)

    fig.suptitle("Gate/depth reductions: merge-maximizing vs random permutations")
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[spacetime_reduction] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> ExperimentSettings:
    parser = argparse.ArgumentParser(
        description="Gate/depth reduction analysis without simulation."
    )
    parser.add_argument("--n-qubits", type=int, default=100, help="Number of logical qubits.")
    parser.add_argument("--n-gates", type=int, default=1000, help="Depth passed to random_clifford_circuit.")
    parser.add_argument("--num-circuits", type=int, default=10, help="Number of random circuits.")
    parser.add_argument(
        "--random-perms",
        type=int,
        default=10,
        help="Number of random permutations per codes value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Base seed for circuit and permutation randomness (set negative for stochastic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/figures/spacetime_reduction/spacetime_reduction_boxplot.svg"),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    seed_value = args.seed if args.seed >= 0 else None
    return ExperimentSettings(
        n_qubits=args.n_qubits,
        n_gates=args.n_gates,
        num_circuits=args.num_circuits,
        random_perms_per_code=args.random_perms,
        random_seed=seed_value,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    df = run_experiment(settings)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
