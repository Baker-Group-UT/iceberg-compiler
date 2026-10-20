"""
spacetime_reduction_sweep.py
----------------------------

Sweep spacetime reduction over:
  1) multiple numbers of qubits with fixed gates-per-qubit, and
  2) multiple gates-per-qubit with fixed number of qubits.

Plots percent reductions vs fraction of codes used for:
  - random-permutation average with H commutation, versus
  - merge-maximizing packing with H commutation and merging.

Run with:

    python -m experiments.figures.spacetime_reduction.spacetime_reduction_sweep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd

from experiments.figures.spacetime_reduction.spacetime_reduction import (
    ExperimentSettings,
    run_experiment,
)

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 18,
    }
)


METRICS = [
    ("pct_reduction_one_qubit_gates", "#1q gates"),
    ("pct_reduction_two_qubit_gates", "#2q gates"),
    ("pct_reduction_circuit_depth", "depth"),
]
SWEEP_LABELS = {
    "n": "Number of qubits",
    "g/q": "Gates per qubit",
}


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
    from_csv: bool
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


def _csv_path_for_plot(output_path: Path) -> Path:
    return output_path.with_suffix(".csv")


def _save_summary_csv(
    summaries: Dict[int, pd.DataFrame],
    sweep_label: str,
    output_path: Path,
) -> None:
    rows: List[pd.DataFrame] = []
    for sweep_value, summary in sorted(summaries.items()):
        rows.append(
            summary.assign(
                sweep_parameter=sweep_label,
                sweep_value=sweep_value,
            )
        )
    csv_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    csv_path = _csv_path_for_plot(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(csv_path, index=False)
    print(f"[spacetime_reduction_sweep] Saved plot data to {csv_path}")


def _load_summary_csv(csv_path: Path, expected_sweep_label: str) -> Dict[int, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {
        "codes_used",
        "fraction_codes_used",
        "pct_reduction_one_qubit_gates",
        "pct_reduction_two_qubit_gates",
        "pct_reduction_circuit_depth",
        "sweep_parameter",
        "sweep_value",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    df = df.loc[df["sweep_parameter"] == expected_sweep_label].copy()
    if df.empty:
        raise ValueError(
            f"No rows found in {csv_path} for sweep_parameter='{expected_sweep_label}'."
        )

    summaries: Dict[int, pd.DataFrame] = {}
    for sweep_value, group_df in df.groupby("sweep_value", sort=True):
        summaries[int(sweep_value)] = (
            group_df[
                [
                    "codes_used",
                    "fraction_codes_used",
                    "pct_reduction_one_qubit_gates",
                    "pct_reduction_two_qubit_gates",
                    "pct_reduction_circuit_depth",
                ]
            ]
            .sort_values("fraction_codes_used")
            .reset_index(drop=True)
        )
    return summaries


def _plot_sweep(
    summaries: Dict[int, pd.DataFrame],
    sweep_label: str,
    title: str,
    output_path: Path,
    cmap_name: str,
    show_plot: bool,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), sharex=True)
    sweep_values = sorted(summaries)
    base_cmap = plt.get_cmap(cmap_name)
    if not sweep_values:
        raise ValueError("Expected at least one sweep summary to plot.")
    vmin = float(min(sweep_values))
    vmax = float(max(sweep_values))
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_truncated",
        base_cmap([i / 255 * 0.875 for i in range(256)]),
    )

    for sweep_value, summary in sorted(summaries.items()):
        color = cmap(norm(float(sweep_value)))
        x = summary["fraction_codes_used"].values
        for ax, (metric, metric_title) in zip(axes, METRICS):
            ax.plot(
                x,
                summary[metric].values,
                label=f"{sweep_label}={sweep_value}",
                color=color,
                linewidth=2.8,
                marker="o",
                markersize=5,
            )
            ax.set_ylabel(metric_title)
            ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.set_xlim(0.5, 1.0)

    for ax in axes:
        ax.set_xlabel("Fraction of codes used")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=axes, shrink=0.92, pad=0.4)
    colorbar.set_label(SWEEP_LABELS.get(sweep_label, sweep_label))
    fig.suptitle(title)
    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.03, 0.0, 0.96, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[spacetime_reduction_sweep] Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_sweeps(settings: SweepSettings) -> None:
    if settings.from_csv:
        qubit_summaries = _load_summary_csv(
            _csv_path_for_plot(settings.output_qubits),
            expected_sweep_label="n",
        )
    else:
        qubit_summaries = {}
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
            "Packed (merge+H) vs random baseline with H "
            f"(gates/qubit={settings.fixed_gates_per_qubit})"
        ),
        output_path=settings.output_qubits,
        cmap_name="viridis",
        show_plot=settings.show_plot,
    )
    if not settings.from_csv:
        _save_summary_csv(
            summaries=qubit_summaries,
            sweep_label="n",
            output_path=settings.output_qubits,
        )

    if settings.from_csv:
        gate_summaries = _load_summary_csv(
            _csv_path_for_plot(settings.output_gates),
            expected_sweep_label="g/q",
        )
    else:
        gate_summaries = {}
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
            "Packed (merge+H) vs random baseline with H "
            f"(n_qubits={settings.fixed_qubits})"
        ),
        output_path=settings.output_gates,
        cmap_name="plasma",
        show_plot=settings.show_plot,
    )
    if not settings.from_csv:
        _save_summary_csv(
            summaries=gate_summaries,
            sweep_label="g/q",
            output_path=settings.output_gates,
        )


def parse_args() -> SweepSettings:
    parser = argparse.ArgumentParser(
        description="Sweep spacetime reduction over qubits and gates-per-qubit."
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "Reload previously saved sweep CSVs next to the output paths and re-render "
            "the SVGs without rerunning the experiments."
        ),
    )
    parser.add_argument(
        "--qubits",
        type=str,
        default="20:101:10",
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
        default="2:22:2",
        help="Comma list or range start:stop:step for gates-per-qubit sweep.",
    )
    parser.add_argument(
        "--num-circuits",
        type=int,
        default=20,
        help="Number of random circuits per sweep point.",
    )
    parser.add_argument(
        "--random-perms",
        type=int,
        default=40,
        help="Number of random permutations per codes value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=104,
        help="Base seed for circuit and permutation randomness (set negative for stochastic).",
    )
    parser.add_argument(
        "--output-qubits",
        type=Path,
        default=Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_sweep_qubits.svg"
        ),
        help="Path to save the qubit sweep plot.",
    )
    parser.add_argument(
        "--output-gates",
        type=Path,
        default=Path(
            "experiments/figures/spacetime_reduction/spacetime_reduction_sweep_gates.svg"
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
        from_csv=args.from_csv,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()
    run_sweeps(settings)


if __name__ == "__main__":
    main()
