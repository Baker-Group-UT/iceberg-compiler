"""
spacetime_reduction_sweep.py
----------------------------

Sweep spacetime reduction over:
  1) multiple numbers of qubits with fixed gates-per-qubit, and
  2) multiple gates-per-qubit with fixed number of qubits.

Plots percent reductions vs fraction of codes used.

Run with:

    python -m experiments.figures.spacetime_reduction.spacetime_reduction_sweep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from experiments.figures.spacetime_reduction.spacetime_reduction import (
    ExperimentSettings,
    run_experiment,
)


METRICS = [
    ("pct_reduction_one_qubit_gates", "Percent reduction in #1q gates"),
    ("pct_reduction_two_qubit_gates", "Percent reduction in #2q gates"),
    ("pct_reduction_circuit_depth", "Percent reduction in depth"),
]


@dataclass
class SweepSettings:
    qubits: List[int]
    fixed_gates_per_qubit: int
    fixed_qubits: int
    gates_per_qubit: List[int]
    num_circuits: int
    random_perms_per_code: int
    random_seed: int | None
    output_qubits: Path
    output_gates: Path
    show_plot: bool


def _parse_int_list(value: str) -> List[int]:
    value = value.strip()
    if ":" in value:
        parts = [part.strip() for part in value.split(":") if part.strip()]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid range format: {value}")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("Range step cannot be 0.")
        return list(range(start, stop, step))
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(value)]


def _summarize_by_fraction(df: pd.DataFrame, n_qubits: int) -> pd.DataFrame:
    summary = (
        df.groupby("codes_used")[list(dict(METRICS).keys())]
        .mean()
        .reset_index()
    )
    summary["fraction_codes_used"] = summary["codes_used"] / n_qubits
    summary = summary.loc[
        (summary["fraction_codes_used"] >= 0.5)
        & (summary["fraction_codes_used"] <= 1.0)
    ]
    return summary.sort_values("fraction_codes_used")


def _run_single_experiment(
    n_qubits: int,
    gates_per_qubit: int,
    settings: SweepSettings,
) -> pd.DataFrame:
    experiment_settings = ExperimentSettings(
        n_qubits=n_qubits,
        n_gates=n_qubits * gates_per_qubit,
        num_circuits=settings.num_circuits,
        random_perms_per_code=settings.random_perms_per_code,
        random_seed=settings.random_seed,
        output_path=None,
        show_plot=False,
    )
    df = run_experiment(experiment_settings)
    return _summarize_by_fraction(df, n_qubits)


def _plot_sweep(
    summaries: Dict[int, pd.DataFrame],
    sweep_label: str,
    title: str,
    output_path: Path,
    cmap_name: str,
    show_plot: bool,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    cmap = plt.get_cmap(cmap_name, max(len(summaries), 2))

    for idx, (sweep_value, summary) in enumerate(summaries.items()):
        color = cmap(idx)
        x = summary["fraction_codes_used"].values
        for ax, (metric, metric_title) in zip(axes, METRICS):
            ax.plot(
                x,
                summary[metric].values,
                label=f"{sweep_label}={sweep_value}",
                color=color,
                linewidth=2,
            )
            ax.set_xlabel("Fraction of codes used")
            ax.set_ylabel("Percent reduction")
            ax.set_title(metric_title)
            ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
            ax.set_xlim(0.5, 1.0)

    fig.suptitle(title)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[spacetime_reduction_sweep] Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_sweeps(settings: SweepSettings) -> None:
    qubit_summaries: Dict[int, pd.DataFrame] = {}
    for n_qubits in settings.qubits:
        qubit_summaries[n_qubits] = _run_single_experiment(
            n_qubits=n_qubits,
            gates_per_qubit=settings.fixed_gates_per_qubit,
            settings=settings,
        )

    _plot_sweep(
        summaries=qubit_summaries,
        sweep_label="n",
        title=(
            "Percent reductions vs fraction of codes used "
            f"(gates/qubit={settings.fixed_gates_per_qubit})"
        ),
        output_path=settings.output_qubits,
        cmap_name="viridis",
        show_plot=settings.show_plot,
    )

    gate_summaries: Dict[int, pd.DataFrame] = {}
    for gates_per_qubit in settings.gates_per_qubit:
        gate_summaries[gates_per_qubit] = _run_single_experiment(
            n_qubits=settings.fixed_qubits,
            gates_per_qubit=gates_per_qubit,
            settings=settings,
        )

    _plot_sweep(
        summaries=gate_summaries,
        sweep_label="g/q",
        title=(
            "Percent reductions vs fraction of codes used "
            f"(n_qubits={settings.fixed_qubits})"
        ),
        output_path=settings.output_gates,
        cmap_name="plasma",
        show_plot=settings.show_plot,
    )


def parse_args() -> SweepSettings:
    parser = argparse.ArgumentParser(
        description="Sweep spacetime reduction over qubits and gates-per-qubit."
    )
    parser.add_argument(
        "--qubits",
        type=str,
        default="20:101:20",
        help="Comma list or range start:stop:step for qubit sweep.",
    )
    parser.add_argument(
        "--fixed-gates-per-qubit",
        type=int,
        default=10,
        help="Gates per qubit for the qubit sweep.",
    )
    parser.add_argument(
        "--fixed-qubits",
        type=int,
        default=50,
        help="Fixed qubits for the gates-per-qubit sweep.",
    )
    parser.add_argument(
        "--gates-per-qubit",
        type=str,
        default="3:21:1",
        help="Comma list or range start:stop:step for gates-per-qubit sweep.",
    )
    parser.add_argument(
        "--num-circuits",
        type=int,
        default=10,
        help="Number of random circuits per sweep point.",
    )
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
        "--output-qubits",
        type=Path,
        default=Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_sweep_qubits.png"
        ),
        help="Path to save the qubit sweep plot.",
    )
    parser.add_argument(
        "--output-gates",
        type=Path,
        default=Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_sweep_gates.png"
        ),
        help="Path to save the gates-per-qubit sweep plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot windows (useful for headless runs).",
    )
    args = parser.parse_args()

    seed_value = args.seed if args.seed >= 0 else None
    return SweepSettings(
        qubits=_parse_int_list(args.qubits),
        fixed_gates_per_qubit=args.fixed_gates_per_qubit,
        fixed_qubits=args.fixed_qubits,
        gates_per_qubit=_parse_int_list(args.gates_per_qubit),
        num_circuits=args.num_circuits,
        random_perms_per_code=args.random_perms,
        random_seed=seed_value,
        output_qubits=args.output_qubits,
        output_gates=args.output_gates,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    run_sweeps(settings)


if __name__ == "__main__":
    main()
