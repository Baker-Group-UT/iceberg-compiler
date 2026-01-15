"""
n_heuristic.py
---------------

Experiment driver that compares the merge-maximizing packer's assignment against a
random permutation that uses the same number of codes. For each requested
``n_codes`` value we:

1. Build a random Clifford circuit.
2. Pack it with ``MergeMaximizingPacker`` restricted to ``n_codes`` patches.
3. Build a random permutation with ``permutation_with_n_codes`` using the
   same ``q`` and ``n_codes``.
4. Run a specific scan for the packed permutation vs. the random one.
5. Measure the percent reduction in logical total variation distance (TVD)
   and collect those samples for a box-and-whisker plot.

Run with:

    python -m experiments.figures.n_heuristic.n_heuristic

Use ``--help`` for optional overrides (sample counts, repetitions, etc.).
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

from packer.merge_packer import MergeMaximizingPacker
from permutations.generator import permutation_with_n_codes
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from simulator.config_presets import StandardSimulator

Perm = List[Tuple[int, int]]
PipelineMetric = str

GATE_SET = ["x", "z", "h", "cx"]


@dataclass
class ExperimentSettings:
    """
    Tunable knobs for the n-codes heuristic experiment.
    """

    min_codes: int = 2
    max_codes: int = 6
    samples_per_code: int = 10
    qubits_per_code: int = 2
    n_gates: int = 25
    simulator_repetitions: int = 25_000
    metric_column: PipelineMetric = "logical_total_variation_distance"
    random_seed: int | None = 7
    output_path: Path = field(
        default_factory=lambda: Path("experiments/figures/n_heuristic/n_heuristic_boxplot.png")
    )
    show_plot: bool = True

    def __post_init__(self) -> None:
        if self.min_codes < 1:
            raise ValueError("min_codes must be >= 1.")
        if self.max_codes < self.min_codes:
            raise ValueError("max_codes must be >= min_codes.")
        if self.samples_per_code <= 0 or self.samples_per_code % 2 != 0:
            raise ValueError("samples_per_code must be a positive even integer.")
        if self.qubits_per_code < 1:
            raise ValueError("qubits_per_code must be >= 1.")
        if self.n_gates < 1:
            raise ValueError("n_gates must be >= 1.")


def build_pipeline_config(repetitions: int) -> PipelineConfig:
    """
    Construct a quiet pipeline config with the requested simulator repetitions.
    """
    return PipelineConfig(
        sim_config=StandardSimulator(repetitions=repetitions).with_stim(),#.with_distance_based_noise(),
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


def extract_metric(df: pd.DataFrame, permutation: Perm, metric_column: str) -> float:
    """
    Fetch the requested metric for the given permutation from a scan dataframe.
    """
    key = perm_key(permutation)
    subset = df.loc[df["perm_key"] == key]
    if subset.empty:
        raise KeyError(f"Permutation {permutation} missing from dataframe.")
    value = float(subset.iloc[0][metric_column])
    if math.isnan(value):
        raise ValueError(f"Metric '{metric_column}' for permutation {permutation} is NaN.")
    return value


def run_single_trial(
    packer: MergeMaximizingPacker,
    config: PipelineConfig,
    settings: ExperimentSettings,
    target_codes: int,
    rng: random.Random,
    sample_idx: int,
) -> Dict[str, float | int | Perm]:
    """
    Execute a single packed-vs-random comparison for a specific n_codes value.
    """
    n_qubits = target_codes * settings.qubits_per_code
    circuit_seed: int | None = None
    if settings.random_seed is not None:
        circuit_seed = settings.random_seed * 10_000 + target_codes * 100 + sample_idx

    program_circuit = build_random_circuit(
        n_qubits=n_qubits,
        n_gates=settings.n_gates,
        seed=circuit_seed,
    )
    tcirc = program_circuit

    packing = packer.pack(tcirc, num_codes=target_codes)
    packed_perm = packing.permutation
    codes_used = len({p for p, _ in packed_perm})

    random_perm_seed = None
    if settings.random_seed is not None:
        random_perm_seed = rng.randint(0, 2**31 - 1)
    random_perm = permutation_with_n_codes(
        q=tcirc.num_qubits,
        n_codes=codes_used,
        seed=random_perm_seed,
    )

    perms = [packed_perm, random_perm]
    record = run_specific_scan(tcirc, perms, config)
    if record.dataframe is None:
        raise RuntimeError("run_specific_scan returned no dataframe.")
    df = record.dataframe.df

    packed_metric = extract_metric(df, packed_perm, settings.metric_column)
    random_metric = extract_metric(df, random_perm, settings.metric_column)

    improvement_pct = pct_reduction(reference=random_metric, challenger=packed_metric)
    return {
        "target_codes": target_codes,
        "codes_used": codes_used,
        "packed_metric": packed_metric,
        "random_metric": random_metric,
        "metric_delta": random_metric - packed_metric,
        "pct_reduction": improvement_pct,
        "circuit_seed": circuit_seed if circuit_seed is not None else -1,
    }


def run_experiment(settings: ExperimentSettings) -> pd.DataFrame:
    """
    Run the configured number of trials for each n_codes value and collect results.
    """
    config = build_pipeline_config(settings.simulator_repetitions)
    packer = MergeMaximizingPacker()
    rng = random.Random(settings.random_seed)

    rows: List[Dict[str, float | int | Perm]] = []
    for n_codes in range(settings.min_codes, settings.max_codes + 1):
        print(f"[n_heuristic] Evaluating n_codes={n_codes} ...")
        for sample_idx in range(settings.samples_per_code):
            result_row = run_single_trial(
                packer=packer,
                config=config,
                settings=settings,
                target_codes=n_codes,
                rng=rng,
                sample_idx=sample_idx,
            )
            rows.append(result_row)

    df = pd.DataFrame(rows)
    return df


def plot_results(df: pd.DataFrame, settings: ExperimentSettings) -> None:
    """
    Produce and optionally save the box-and-whisker plot of percent reductions.
    """
    codes = sorted(df["codes_used"].unique())
    data = [df.loc[df["codes_used"] == code, "pct_reduction"].values for code in codes]

    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_title("Merge-maximizing packing vs. random permutation")
    ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
    fig.tight_layout()

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[n_heuristic] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def summarize_results(df: pd.DataFrame) -> None:
    """
    Print aggregated summary statistics for quick inspection.
    """
    grouped = df.groupby("codes_used")["pct_reduction"].agg(["count", "mean", "median", "std"])
    print("[n_heuristic] Percent reduction summary (by codes used):")
    print(grouped.to_string())


def parse_args() -> ExperimentSettings:
    parser = argparse.ArgumentParser(
        description="n-codes merge-maximizing packer vs random permutation experiment"
    )
    parser.add_argument("--min-codes", type=int, default=20, help="Minimum number of codes to evaluate.")
    parser.add_argument("--max-codes", type=int, default=24, help="Maximum number of codes to evaluate.")
    parser.add_argument(
        "--samples-per-code",
        type=int,
        default=10,
        help="Number of trials per codes value (must be even).",
    )
    parser.add_argument(
        "--qubits-per-code",
        type=int,
        default=2,
        help="Logical qubits simulated per code (determines circuit width).",
    )
    parser.add_argument("--n-gates", type=int, default=200, help="Depth passed to random_clifford_circuit.")
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
        default=7,
        help="Base seed for circuit and permutation randomness (set negative for stochastic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/figures/n_heuristic/n_heuristic_boxplot.svg"),
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
        min_codes=args.min_codes,
        max_codes=args.max_codes,
        samples_per_code=args.samples_per_code,
        qubits_per_code=args.qubits_per_code,
        n_gates=args.n_gates,
        simulator_repetitions=args.repetitions,
        metric_column=args.metric_column,
        random_seed=seed_value,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    df = run_experiment(settings)
    summarize_results(df)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
