"""
n_heuristic_single_circuit.py
-----------------------------

Experiment driver that compares the merge-maximizing packer's assignment against
random permutations using a single fixed program circuit. For each requested
``n_codes`` value in the range ceil(n_qubits/2) to n_qubits-1 we:

1. Build one random Clifford circuit.
2. Pack it with ``MergeMaximizingPacker`` restricted to ``n_codes`` patches.
3. Build ``k`` random permutations using the same ``q`` and ``n_codes``.
4. Run a specific scan for the packed permutation vs. all random ones.
5. Measure percent reduction in logical TVD relative to each random sample.

Run with:

    python -m experiments.figures.n_heuristic.n_heuristic_single_circuit

Use ``--help`` for optional overrides (repetitions, k, etc.).
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

from packer.merge_packer import MergeMaximizingPacker, SoloStrategy
from permutations.generator import permutation_with_n_codes
from pipeline.analysis.print_df import print_df_table
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from simulator.config_presets import ScaledSimulator, StandardSimulator

Perm = List[Tuple[int, int]]
PipelineMetric = str

GATE_SET = ["x", "z", "h", "cx"]


@dataclass
class ExperimentSettings:
    """
    Tunable knobs for the single-circuit n-codes experiment.
    """

    n_qubits: int = 10
    n_gates: int = 200
    random_perms_per_code: int = 10
    simulator_repetitions: int = 100_000
    metric_column: PipelineMetric = "logical_total_variation_distance"
    random_seed: int | None = 7
    solo_strategies: Tuple[SoloStrategy, ...] = (
        "most_gates",
        "pair_then_solo",
        "most_hadamards",
    )
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/n_heuristic/n_heuristic_single_circuit_boxplot.png"
        )
    )
    show_plot: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2.")
        if self.n_gates < 1:
            raise ValueError("n_gates must be >= 1.")
        if self.random_perms_per_code <= 0:
            raise ValueError("random_perms_per_code must be >= 1.")


def build_pipeline_config(repetitions: int) -> PipelineConfig:
    """
    Construct a quiet pipeline config with the requested simulator repetitions.
    """
    return PipelineConfig(
        sim_config=ScaledSimulator(repetitions=repetitions, alpha=0.5).with_stim().with_distance_based_noise(),
        verbose=False,
        save=False,
    )


def perm_key(perm: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """
    Canonical key for dataframe lookups.
    """
    return tuple(tuple(p) for p in perm)


def pct_reduction(reference: float, challenger: float) -> float:
    """
    Percent reduction of challenger relative to the reference error.
    Works with TVD (lower-is-better).
    """
    if reference <= 0:
        return 0.0
    delta = reference - challenger
    return 100.0 * delta / reference


def build_random_circuit(
    n_qubits: int,
    n_gates: int,
    seed: int | None,
) -> qiskit.QuantumCircuit:
    """
    Generate a random Clifford circuit with the requested gate set and depth.
    """
    circuit = random_clifford_circuit(
        n_qubits,
        n_gates,
        # seed=seed,
        gates=GATE_SET,
    )
    circuit.measure_all()
    return circuit


def tight_trivial_packing(num_qubits: int, patch_capacity: int = 2) -> Perm:
    """
    Return a tight trivial packing: (0,0),(0,1),(1,0),(1,1),...
    """
    if patch_capacity < 1:
        raise ValueError("patch_capacity must be >= 1")
    return [(i // patch_capacity, i % patch_capacity) for i in range(num_qubits)]


def sparse_trivial_packing(num_qubits: int) -> Perm:
    """
    Return a sparse trivial packing: (0,0),(1,0),(2,0),...
    """
    return [(i, 0) for i in range(num_qubits)]


def extract_metric(df: pd.DataFrame, permutation: Perm, metric_column: str) -> float:
    """
    Fetch the requested metric for the given permutation from a scan dataframe.
    """
    if metric_column not in df.columns:
        raise KeyError(f"Metric column '{metric_column}' missing from dataframe.")
    key = perm_key(permutation)
    subset = df.loc[df["perm_key"] == key]
    if subset.empty:
        raise KeyError(f"Permutation {permutation} missing from dataframe.")
    value = float(subset.iloc[0][metric_column])
    if math.isnan(value):
        raise ValueError(f"Metric '{metric_column}' for permutation {permutation} is NaN.")
    return value


def run_single_circuit_trial(
    config: PipelineConfig,
    settings: ExperimentSettings,
    program_circuit: qiskit.QuantumCircuit,
    target_codes: int,
    rng: random.Random,
    solo_strategies: Sequence[SoloStrategy],
) -> List[Dict[str, float | int | Perm]]:
    """
    Execute packed-vs-random comparisons for a specific n_codes value.
    """
    random_perms: List[Perm] = []
    for _ in range(settings.random_perms_per_code):
        random_perm_seed = rng.randint(0, 2**31 - 1) if settings.random_seed is not None else None
        random_perms.append(
            permutation_with_n_codes(
                q=program_circuit.num_qubits,
                n_codes=target_codes,
                seed=random_perm_seed,
            )
        )

    strategy_perms: Dict[SoloStrategy, Perm] = {}
    for solo_strategy in solo_strategies:
        packer = MergeMaximizingPacker(solo_strategy=solo_strategy)
        strategy_perms[solo_strategy] = packer.pack(
            program_circuit,
            num_codes=target_codes,
            solo_strategy=solo_strategy,
        ).permutation

    tight_perm = tight_trivial_packing(program_circuit.num_qubits)
    sparse_perm = sparse_trivial_packing(program_circuit.num_qubits)

    perms = list(strategy_perms.values()) + random_perms + [tight_perm, sparse_perm]
    record = run_specific_scan(program_circuit, perms, config)
    if record.dataframe is None:
        raise RuntimeError("run_specific_scan returned no dataframe.")
    df = record.dataframe.df
    print_df_table(
        df,
        header=f"[n_heuristic_single_circuit] Runs for n_codes={target_codes}",
        hide_permutation=False,
    )

    tight_metric = extract_metric(df, tight_perm, settings.metric_column)
    sparse_metric = extract_metric(df, sparse_perm, settings.metric_column)
    tight_one_q = extract_metric(df, tight_perm, "one_qubit_gates")
    tight_two_q = extract_metric(df, tight_perm, "two_qubit_gates")
    tight_depth = extract_metric(df, tight_perm, "circuit_depth")
    sparse_one_q = extract_metric(df, sparse_perm, "one_qubit_gates")
    sparse_two_q = extract_metric(df, sparse_perm, "two_qubit_gates")
    sparse_depth = extract_metric(df, sparse_perm, "circuit_depth")

    rows: List[Dict[str, float | int | Perm]] = []
    for solo_strategy, packed_perm in strategy_perms.items():
        packed_metric = extract_metric(df, packed_perm, settings.metric_column)
        packed_one_q = extract_metric(df, packed_perm, "one_qubit_gates")
        packed_two_q = extract_metric(df, packed_perm, "two_qubit_gates")
        packed_depth = extract_metric(df, packed_perm, "circuit_depth")
        codes_used = len({p for p, _ in packed_perm})

        for random_perm in random_perms:
            random_metric = extract_metric(df, random_perm, settings.metric_column)
            random_one_q = extract_metric(df, random_perm, "one_qubit_gates")
            random_two_q = extract_metric(df, random_perm, "two_qubit_gates")
            random_depth = extract_metric(df, random_perm, "circuit_depth")
            improvement_pct = pct_reduction(reference=random_metric, challenger=packed_metric)
            one_q_reduction_pct = pct_reduction(reference=random_one_q, challenger=packed_one_q)
            two_q_reduction_pct = pct_reduction(reference=random_two_q, challenger=packed_two_q)
            depth_reduction_pct = pct_reduction(reference=random_depth, challenger=packed_depth)
            tight_improvement_pct = pct_reduction(reference=tight_metric, challenger=packed_metric)
            sparse_improvement_pct = pct_reduction(reference=sparse_metric, challenger=packed_metric)
            tight_one_q_reduction = pct_reduction(reference=tight_one_q, challenger=packed_one_q)
            tight_two_q_reduction = pct_reduction(reference=tight_two_q, challenger=packed_two_q)
            tight_depth_reduction = pct_reduction(reference=tight_depth, challenger=packed_depth)
            sparse_one_q_reduction = pct_reduction(reference=sparse_one_q, challenger=packed_one_q)
            sparse_two_q_reduction = pct_reduction(reference=sparse_two_q, challenger=packed_two_q)
            sparse_depth_reduction = pct_reduction(reference=sparse_depth, challenger=packed_depth)
            rows.append(
                {
                    "target_codes": target_codes,
                    "codes_used": codes_used,
                    "solo_strategy": solo_strategy,
                    "packed_metric": packed_metric,
                    "random_metric": random_metric,
                    "metric_delta": random_metric - packed_metric,
                    "pct_reduction": improvement_pct,
                    "packed_one_qubit_gates": packed_one_q,
                    "random_one_qubit_gates": random_one_q,
                    "pct_reduction_one_qubit_gates": one_q_reduction_pct,
                    "packed_two_qubit_gates": packed_two_q,
                    "random_two_qubit_gates": random_two_q,
                    "pct_reduction_two_qubit_gates": two_q_reduction_pct,
                    "packed_circuit_depth": packed_depth,
                    "random_circuit_depth": random_depth,
                    "pct_reduction_circuit_depth": depth_reduction_pct,
                    "pct_reduction_vs_tight": tight_improvement_pct,
                    "pct_reduction_vs_sparse": sparse_improvement_pct,
                    "pct_reduction_one_qubit_vs_tight": tight_one_q_reduction,
                    "pct_reduction_two_qubit_vs_tight": tight_two_q_reduction,
                    "pct_reduction_depth_vs_tight": tight_depth_reduction,
                    "pct_reduction_one_qubit_vs_sparse": sparse_one_q_reduction,
                    "pct_reduction_two_qubit_vs_sparse": sparse_two_q_reduction,
                    "pct_reduction_depth_vs_sparse": sparse_depth_reduction,
                }
            )
    return rows


def run_experiment(settings: ExperimentSettings) -> pd.DataFrame:
    """
    Run a single-circuit experiment over the n_codes range.
    """
    config = build_pipeline_config(settings.simulator_repetitions)
    rng = random.Random(settings.random_seed)

    circuit_seed: int | None = None
    if settings.random_seed is not None:
        circuit_seed = settings.random_seed * 10_000 + settings.n_qubits * 100 + settings.n_gates
    program_circuit = build_random_circuit(
        n_qubits=settings.n_qubits,
        n_gates=settings.n_gates,
        seed=circuit_seed,
    )

    min_codes = math.ceil(settings.n_qubits / 2)
    max_codes = settings.n_qubits - 1

    rows: List[Dict[str, float | int | Perm]] = []
    for n_codes in range(min_codes, max_codes + 1):
        print(f"[n_heuristic_single_circuit] Evaluating n_codes={n_codes} ...")
        rows.extend(
            run_single_circuit_trial(
                config=config,
                settings=settings,
                program_circuit=program_circuit,
                target_codes=n_codes,
                rng=rng,
                solo_strategies=settings.solo_strategies,
            )
        )

    df = pd.DataFrame(rows)
    return df


def plot_results(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    """
    Produce and optionally save the box-and-whisker plot of percent reductions.
    """
    strategies = sorted(df["solo_strategy"].unique())
    codes = sorted(df["codes_used"].unique())

    fig, axes = plt.subplots(nrows=1, ncols=len(strategies), figsize=(6 * len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies):
        subset = df.loc[df["solo_strategy"] == strategy]
        data = [subset.loc[subset["codes_used"] == code, "pct_reduction"].values for code in codes]
        bp = ax.boxplot(data, labels=[str(code) for code in codes], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#7fcdbb", edgecolor="#1d91c0", alpha=0.8)
        for whisker in bp["whiskers"]:
            whisker.set(color="#1d91c0")
        for cap in bp["caps"]:
            cap.set(color="#1d91c0")
        for median in bp["medians"]:
            median.set(color="#0c2c84", linewidth=2)
        ax.set_xlabel("Codes used")
        ax.set_ylabel("Percent reduction in logical TVD")
        ax.set_title(f"Strategy: {strategy}")
        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)

    fig.suptitle("Merge-maximizing packing vs. random (single circuit)")
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[n_heuristic_single_circuit] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_resource_results(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    """
    Plot reduction for #1q gates, #2q gates, and depth.
    """
    metrics = [
        ("pct_reduction_one_qubit_gates", "Percent reduction in #1q gates"),
        ("pct_reduction_two_qubit_gates", "Percent reduction in #2q gates"),
        ("pct_reduction_circuit_depth", "Percent reduction in depth"),
    ]
    strategies = sorted(df["solo_strategy"].unique())
    codes = sorted(df["codes_used"].unique())

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(strategies),
        figsize=(6 * len(strategies), 4 * len(metrics)),
    )
    if len(metrics) == 1:
        axes = [axes]
    for row_idx, (metric, title) in enumerate(metrics):
        row_axes = axes[row_idx] if len(strategies) > 1 else [axes[row_idx]]
        for ax, strategy in zip(row_axes, strategies):
            subset = df.loc[df["solo_strategy"] == strategy]
            data = [subset.loc[subset["codes_used"] == code, metric].values for code in codes]
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
            ax.set_title(f"{title} (strategy={strategy})")
            ax.axhline(0, color="#636363", linestyle="--", linewidth=1)

    fig.tight_layout()

    if settings.output_path is not None:
        output_path = settings.output_path
        resource_path = output_path.with_name(
            f"{output_path.stem}_gate_depth{output_path.suffix}"
        )
        resource_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resource_path, dpi=200)
        print(f"[n_heuristic_single_circuit] Saved plot to {resource_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_trivial_baselines(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    """
    Plot reduction versus tight and sparse trivial packings.
    """
    metrics = [
        ("pct_reduction_vs_tight", "TVD reduction vs tight packing"),
        ("pct_reduction_vs_sparse", "TVD reduction vs sparse packing"),
    ]
    strategies = sorted(df["solo_strategy"].unique())
    codes = sorted(df["codes_used"].unique())

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(strategies),
        figsize=(6 * len(strategies), 4 * len(metrics)),
    )
    if len(metrics) == 1:
        axes = [axes]
    for row_idx, (metric, title) in enumerate(metrics):
        row_axes = axes[row_idx] if len(strategies) > 1 else [axes[row_idx]]
        for ax, strategy in zip(row_axes, strategies):
            subset = df.loc[df["solo_strategy"] == strategy]
            data = [subset.loc[subset["codes_used"] == code, metric].values for code in codes]
            bp = ax.boxplot(data, labels=[str(code) for code in codes], patch_artist=True)
            for patch in bp["boxes"]:
                patch.set(facecolor="#c7e9c0", edgecolor="#238b45", alpha=0.8)
            for whisker in bp["whiskers"]:
                whisker.set(color="#238b45")
            for cap in bp["caps"]:
                cap.set(color="#238b45")
            for median in bp["medians"]:
                median.set(color="#005a32", linewidth=2)
            ax.set_xlabel("Codes used")
            ax.set_ylabel("Percent reduction")
            ax.set_title(f"{title} (strategy={strategy})")
            ax.axhline(0, color="#636363", linestyle="--", linewidth=1)

    fig.tight_layout()

    if settings.output_path is not None:
        output_path = settings.output_path
        baseline_path = output_path.with_name(
            f"{output_path.stem}_tight_sparse{output_path.suffix}"
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(baseline_path, dpi=200)
        print(f"[n_heuristic_single_circuit] Saved plot to {baseline_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)
def summarize_results(df: pd.DataFrame) -> None:
    """
    Print aggregated summary statistics for quick inspection.
    """
    grouped = df.groupby(["solo_strategy", "codes_used"])["pct_reduction"].agg(
        ["count", "mean", "median", "std"]
    )
    print("[n_heuristic_single_circuit] Percent reduction summary (by strategy, codes used):")
    print(grouped.to_string())


def parse_args() -> ExperimentSettings:
    parser = argparse.ArgumentParser(
        description="n-codes merge-maximizing packer vs random permutations (single circuit)"
    )
    parser.add_argument("--n-qubits", type=int, default=16, help="Number of logical qubits.")
    parser.add_argument("--n-gates", type=int, default=200, help="Depth passed to random_clifford_circuit.")
    parser.add_argument(
        "--random-perms",
        type=int,
        default=10,
        help="Number of random permutations per codes value.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100_000,
        help="Simulator repetitions per run.",
    )
    parser.add_argument(
        "--metric-column",
        type=str,
        default="logical_total_variation_distance",
        help="Dataframe metric to compare (default: logical_total_variation_distance).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="Base seed for circuit and permutation randomness (set negative for stochastic).",
    )
    parser.add_argument(
        "--solo-strategies",
        type=str,
        default="most_gates,pair_then_solo,most_hadamards",
        help=(
            "Comma-separated strategies for selecting solo qubits "
            "(choices: most_gates, pair_then_solo, most_hadamards)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/figures/n_heuristic/n_heuristic_single_circuit_boxplot.svg"),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    seed_value = args.seed if args.seed >= 0 else None
    strategies = tuple(
        s.strip()
        for s in args.solo_strategies.split(",")
        if s.strip()
    )
    allowed = {"most_gates", "pair_then_solo", "most_hadamards"}
    unknown = [s for s in strategies if s not in allowed]
    if unknown:
        raise ValueError(f"Unknown solo strategies: {unknown}")
    return ExperimentSettings(
        n_qubits=args.n_qubits,
        n_gates=args.n_gates,
        random_perms_per_code=args.random_perms,
        simulator_repetitions=args.repetitions,
        metric_column=args.metric_column,
        random_seed=seed_value,
        solo_strategies=strategies,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    df = run_experiment(settings)
    summarize_results(df)
    plot_results(df, settings)
    plot_resource_results(df, settings)
    plot_trivial_baselines(df, settings)


if __name__ == "__main__":
    main()
