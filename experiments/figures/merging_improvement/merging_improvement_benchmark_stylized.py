"""
merging_improvement_benchmark_stylized.py
--------------------------------

Stylized benchmark variant of merging_improvement.

For each QASMbench proxy circuit, compare cumulative percent reductions
(vs naive tight packing with merging and H commutation disabled) at
ceil(n/2) codes.

Each circuit has one stacked bar broken into additive benefit components:
1) H commutation benefit
2) additional merging benefit
3) additional packing benefit

The packing row is planned by the `packing` package through `common.pack_program`;
`packer/merge_packer.py`, which used to do it, is archived. Both place two logical
qubits per patch, so the figure measures the same quantity as before. Head to head on
circuit counts at k=2 the two agree closely -- on `dnn_n33` the archived pairwise packer
returns 53.5% of depth and this one 53.0% -- so a bar that changed against the published
version changed for one of the two reasons below, not because the packer got worse.

**Segments are clipped cumulatively, not one at a time.** `stack_heights` owns the rule
and explains it; the short version is that clipping each segment at zero and summing
overstates the bar whenever one technique hurts and a later one recovers. On `ising_n42`
H commutation costs 91% of depth and merging returns 171%; the honest total is 80%, and
the per-segment rule this script used to apply drew 171% on an axis where 100% is all of
it.

**Each stage keeps the better of H commutation on and off.** `H_COMMUTATION_MODES`
explains the three settings and their trade-offs. The default is `best`: a stage is never
charged for a pass the compiler would have declined, so no segment goes negative because
a technique hurt -- which is the right reading of an ablation, since
`enable_h_commutation` is a flag and not applying it is always available. Forcing the
pass on, as the published figure did, is what produced the >100% bars: on `ising_n98` it
takes naive depth from 335 to 1013, and merging's return to 55 then reads as +286%.

Circuits are ordered alphabetically within each size group. Ordering by summed depth
benefit, as this script used to, made every panel look monotonic in a quantity only one
of them plotted.

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement_benchmark_stylized
    python -m ... --h-commutation auto     # the planner's own per-circuit decision
    python -m ... --h-commutation on       # reproduce the published figure
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm import tqdm

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

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.figures.merging_improvement.common import pack_program
from experiments.shared.figures import stack_heights
from experiments.shared.benchmark_circuits import load_proxy_circuits
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]

METRICS = {
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "#1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "#2q gates"),
    "circuit_depth": ("pct_reduction_circuit_depth", "depth"),
}

#: How the H commutation pass is chosen. This is a *figure* decision, not a compiler one,
#: and the distinction matters:
#:
#: `best`   -- measure every stage with the pass on and off and keep the better result,
#:     per metric. The default, and what the ablation should show: `enable_h_commutation`
#:     is a config flag, so declining a pass that costs you is always available, and a
#:     stage can never be charged for a technique it would not have used. It is also why
#:     no segment here can go negative for the reason a *pass* made things worse.
#:
#:     The caveat is that "per metric" means the 2q panel and the depth panel may be
#:     describing two different compiled circuits. On `dnn_n33` at k=2 the packed circuit
#:     with the pass on has 1q 1155 / 2q 1503 / depth 535 and with it off 1q 1045 /
#:     2q 1565 / depth 472 -- off wins 1q and depth, on wins 2q. That is honest as "what
#:     the compiler achieves on each objective when it is free to decline the pass", which
#:     is what an ablation is asking, but it is not one circuit.
#:
#: `auto`   -- the planner's own per-circuit decision, from `commutation_pays`. One
#:     circuit, so all three panels describe the same compilation, at the cost of showing
#:     less than the compiler can do on any single metric: the decision minimises the
#:     weighted scalar `1q + 13*2q + 12*depth`, which no individual panel plots.
#:
#: `on`     -- forced on everywhere. Reproduces the published figure. Not recommended:
#:     forcing the pass where it does not pay inflates the row the next stage is
#:     differenced against, which is what drew the >100% depth bars. On `ising_n98` it
#:     takes naive depth from 335 to 1013, merging returns it to 55, and the segment reads
#:     +286% on an axis where 100% is all of it.
H_COMMUTATION_MODES = ("best", "auto", "on")


@dataclass
class BenchmarkSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    max_proxy_gates: int | None = 2000
    max_proxy_qubits: int | None = 100
    proxy_timeout_seconds: float | None = 1.0
    drop_equal_packings: bool = False
    #: One of `H_COMMUTATION_MODES`.
    h_commutation: str = "best"
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized.svg"
        )
    )
    show_plot: bool = True


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    patch_capacity = max(1, math.ceil(num_qubits / n_codes))
    perm: Perm = []
    for i in range(num_qubits):
        code = min(i // patch_capacity, n_codes - 1)
        perm.append((code, i % patch_capacity))
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
    size_group: str,
    settings: BenchmarkSettings,
) -> pd.DataFrame:
    baseline_cfg = PipelineConfig()
    baseline_cfg.translator_config.enable_merging = False
    baseline_cfg.translator_config.enable_h_commutation = False

    h_only_cfg = PipelineConfig()
    h_only_cfg.translator_config.enable_merging = False
    h_only_cfg.translator_config.enable_h_commutation = True

    h_merge_cfg = PipelineConfig()
    h_merge_cfg.translator_config.enable_merging = True
    h_merge_cfg.translator_config.enable_h_commutation = True

    rows: List[Dict[str, int | float | str]] = []

    for circuit_idx, (circuit_name, program_circuit) in tqdm(
        list(enumerate(circuits)),
        desc=f"circuits ({size_group})",
    ):
        n_qubits = program_circuit.num_qubits
        target_codes = math.ceil(n_qubits / 2)
        tight_perm = tight_trivial_packing(n_qubits, target_codes)
        base_one_q, base_two_q, base_depth = compute_metrics(
            program_circuit, tight_perm, baseline_cfg
        )
        base = (base_one_q, base_two_q, base_depth)

        # Every stage is measured under both commutation settings and the better one
        # wins, per metric. `H_COMMUTATION_MODES` documents why. Note the H-off arm of
        # stage 1 *is* the baseline -- declining the pass leaves the circuit untouched --
        # so the H segment is non-negative by construction rather than by clipping.
        arms = [True, False] if settings.h_commutation == "best" else [None]
        stages: Dict[str, List[Tuple[int, int, int]]] = {"h": [], "hm": [], "pack": []}
        for arm in arms:
            if arm is None:
                packed_perm, commute = pack_program(
                    program_circuit,
                    h_commutation=True if settings.h_commutation == "on" else "auto",
                )
            else:
                commute = arm
                packed_perm, _ = pack_program(program_circuit, h_commutation=commute)
            h_only_cfg.translator_config.enable_h_commutation = commute
            h_merge_cfg.translator_config.enable_h_commutation = commute
            stages["h"].append(compute_metrics(program_circuit, tight_perm, h_only_cfg))
            stages["hm"].append(compute_metrics(program_circuit, tight_perm, h_merge_cfg))
            stages["pack"].append(
                compute_metrics(program_circuit, packed_perm, h_merge_cfg)
            )

        def best(stage: str, index: int) -> float:
            """Largest reduction any arm achieves for one metric at one stage."""
            return max(
                pct_reduction(base[index], measured[index])
                for measured in stages[stage]
            )

        h_cum_pct_one_q, h_cum_pct_two_q, h_cum_pct_depth = (
            best("h", 0), best("h", 1), best("h", 2)
        )
        h_merge_cum_pct_one_q, h_merge_cum_pct_two_q, h_merge_cum_pct_depth = (
            best("hm", 0), best("hm", 1), best("hm", 2)
        )
        packed_cum_pct_one_q, packed_cum_pct_two_q, packed_cum_pct_depth = (
            best("pack", 0), best("pack", 1), best("pack", 2)
        )

        h_only_seg_one_q = h_cum_pct_one_q
        h_only_seg_two_q = h_cum_pct_two_q
        h_only_seg_depth = h_cum_pct_depth
        merge_seg_one_q = h_merge_cum_pct_one_q - h_cum_pct_one_q
        merge_seg_two_q = h_merge_cum_pct_two_q - h_cum_pct_two_q
        merge_seg_depth = h_merge_cum_pct_depth - h_cum_pct_depth
        pack_seg_one_q = packed_cum_pct_one_q - h_merge_cum_pct_one_q
        pack_seg_two_q = packed_cum_pct_two_q - h_merge_cum_pct_two_q
        pack_seg_depth = packed_cum_pct_depth - h_merge_cum_pct_depth

        if (
            h_only_seg_one_q < 0
            or h_only_seg_two_q < 0
            or h_only_seg_depth < 0
            or merge_seg_one_q < 0
            or merge_seg_two_q < 0
            or merge_seg_depth < 0
            or pack_seg_one_q < 0
            or pack_seg_two_q < 0
            or pack_seg_depth < 0
        ):
            print(
                "[merging_improvement_benchmark_stylized] Negative incremental benefit:",
                f"{size_group}/{circuit_name}",
                f"codes={target_codes}",
                f"h=(1q={h_only_seg_one_q:.2f}%, 2q={h_only_seg_two_q:.2f}%, depth={h_only_seg_depth:.2f}%)",
                f"merge=(1q={merge_seg_one_q:.2f}%, 2q={merge_seg_two_q:.2f}%, depth={merge_seg_depth:.2f}%)",
                f"pack=(1q={pack_seg_one_q:.2f}%, 2q={pack_seg_two_q:.2f}%, depth={pack_seg_depth:.2f}%)",
            )

        if settings.drop_equal_packings:
            if (
                h_only_one_q == packed_one_q
                and h_only_two_q == packed_two_q
                and h_only_depth == packed_depth
            ):
                continue

        # Both the segment and the cumulative reduction it lands at. `stack_heights`
        # needs the cumulative series: clipping each segment at zero and summing
        # inflates the bar whenever one technique hurts and a later one recovers, and
        # on this suite that is not a corner case. Forcing H commutation on `dnn_n33`
        # costs 6.7% of depth and packing then returns 53.0%; the true full-pipeline
        # reduction is 46.3%, but dropping the negative segment draws 53.0%.
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "component": "h_commutation",
                "pct_reduction_one_qubit_gates": h_only_seg_one_q,
                "pct_reduction_two_qubit_gates": h_only_seg_two_q,
                "pct_reduction_circuit_depth": h_only_seg_depth,
                "cumulative_pct_reduction_one_qubit_gates": h_cum_pct_one_q,
                "cumulative_pct_reduction_two_qubit_gates": h_cum_pct_two_q,
                "cumulative_pct_reduction_circuit_depth": h_cum_pct_depth,
            }
        )
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "component": "merging",
                "pct_reduction_one_qubit_gates": merge_seg_one_q,
                "pct_reduction_two_qubit_gates": merge_seg_two_q,
                "pct_reduction_circuit_depth": merge_seg_depth,
                "cumulative_pct_reduction_one_qubit_gates": h_merge_cum_pct_one_q,
                "cumulative_pct_reduction_two_qubit_gates": h_merge_cum_pct_two_q,
                "cumulative_pct_reduction_circuit_depth": h_merge_cum_pct_depth,
            }
        )
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "target_codes": target_codes,
                "component": "packing",
                "cumulative_pct_reduction_one_qubit_gates": packed_cum_pct_one_q,
                "cumulative_pct_reduction_two_qubit_gates": packed_cum_pct_two_q,
                "cumulative_pct_reduction_circuit_depth": packed_cum_pct_depth,
                "pct_reduction_one_qubit_gates": pack_seg_one_q,
                "pct_reduction_two_qubit_gates": pack_seg_two_q,
                "pct_reduction_circuit_depth": pack_seg_depth,
            }
        )

    return pd.DataFrame(rows)


def _plotted_total(frame: pd.DataFrame, metric_column: str) -> float:
    """Mean stack height over the circuits in `frame`, drawn exactly as the bars are."""
    components = ["h_commutation", "merging", "packing"]
    names = frame["circuit_name"].unique()
    if not len(names):
        return 0.0
    return float(
        np.mean(
            [
                sum(
                    stack_heights(
                        frame.loc[frame["circuit_name"] == name],
                        metric_column,
                        components,
                    )
                )
                for name in names
            ]
        )
    )


def print_metric_averages(df: pd.DataFrame) -> None:
    print("[merging_improvement_benchmark_stylized] Plot averages (matching bar heights)")
    for size_group in ["small", "medium", "large"]:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        print(f"  {size_group}:")
        print(
            "    plotted total: "
            + ", ".join(
                f"{label}={_plotted_total(group_df, column):.2f}%"
                for _, (column, label) in METRICS.items()
            )
        )

    print(
        "  plotted overall average: "
        + ", ".join(
            f"{label}={_plotted_total(df, column):.2f}%"
            for _, (column, label) in METRICS.items()
        )
    )


def plot_results(df: pd.DataFrame, settings: BenchmarkSettings) -> None:
    components = [
        ("h_commutation", "H commutation benefit", "#44545e"),
        ("merging", "+ Merging benefit", "#cfc5c3"),
        ("packing", "+ Packing benefit", "#0c639b"),
    ]
    size_groups = ["small", "medium", "large"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 14), sharex=True)
    bar_width = 0.75
    gap = 4.0

    # Alphabetical within each size group, and the same order in every subplot because
    # the x axis is shared. This replaces sorting by summed depth benefit: that put the
    # same circuit at a different x in a reader's memory of the previous version of the
    # figure, and it made the three panels' bar heights look monotonic in a panel whose
    # own metric was not the sort key.
    ordered_circuits_by_group: Dict[str, List[str]] = {}
    for size_group in size_groups:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        ordered_circuits_by_group[size_group] = sorted(
            group_df["circuit_name"].unique(), key=str.lower
        )

    for ax, (_, (metric_column, metric_label)) in zip(axes, METRICS.items()):
        all_positions: List[float] = []
        all_circuit_labels: List[str] = []
        current_x = 0.0

        for size_group in size_groups:
            ordered_circuits = ordered_circuits_by_group.get(size_group, [])
            if not ordered_circuits:
                continue
            group_df = df.loc[df["size_group"] == size_group]

            # One `stack_heights` call per circuit rather than a pivot-and-clip: the
            # rule is cumulative, so a segment cannot be computed from the segment
            # column alone.
            segments = {
                name: stack_heights(
                    group_df.loc[group_df["circuit_name"] == name],
                    metric_column,
                    [component for component, _, _ in components],
                )
                for name in ordered_circuits
            }
            circuit_names = ordered_circuits
            positions = np.arange(len(circuit_names)) + current_x

            running_bottom = np.zeros(len(circuit_names), dtype=float)
            for index, (component, label, color) in enumerate(components):
                values = np.array(
                    [segments[name][index] for name in circuit_names], dtype=float
                )
                ax.bar(
                    positions,
                    values,
                    width=bar_width,
                    bottom=running_bottom,
                    color=color,
                    edgecolor="white",
                    alpha=0.85,
                    label=label,
                )
                running_bottom += values

            all_positions.extend(positions.tolist())
            all_circuit_labels.extend(circuit_names)
            current_x = positions[-1] + gap if len(positions) else current_x + gap

        # Append an overall average subgroup (single stacked bar).
        # Center the single average bar where the old 3-bar subgroup midpoint was.
        avg_x = current_x + 1.0
        avg_width = 5 * bar_width
        avg_bottom = 0.0
        # Mean of each circuit's *clipped* segment, so the average bar is the mean of the
        # bars drawn beside it. Clipping the pooled column instead lets one circuit's
        # inflated segment raise the average above every bar it summarises.
        average_segments = np.mean(
            [
                stack_heights(
                    df.loc[df["circuit_name"] == name],
                    metric_column,
                    [component for component, _, _ in components],
                )
                for name in df["circuit_name"].unique()
            ],
            axis=0,
        )
        for index, (component, label, color) in enumerate(components):
            avg_value = float(average_segments[index])
            if np.isnan(avg_value):
                avg_value = 0.0
            ax.bar(
                avg_x,
                avg_value,
                width=avg_width,
                bottom=avg_bottom,
                color=color,
                edgecolor="white",
                alpha=0.95,
                label=label,
            )
            avg_bottom += avg_value
        ax.axvline(current_x - gap / 2.0, color="#bdbdbd", linestyle=":", linewidth=1.0)
        left_gap_to_avg = (avg_x - avg_width / 2.0) - (current_x - gap / 2.0)
        right_limit = avg_x + avg_width / 2.0 + max(0.0, left_gap_to_avg)
        left_limit, _ = ax.get_xlim()
        ax.set_xlim(left=left_limit, right=right_limit)
        all_positions.append(avg_x)
        all_circuit_labels.append("average")

        ax.set_ylabel(metric_label)
        ax.axhline(0, color="#636363", linestyle="--", linewidth=1)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_xticks(all_positions, minor=True)
        ax.set_xticklabels(all_circuit_labels, rotation=45, ha="right", minor=True)
        ticklabels = ax.get_xticklabels(minor=True)
        if ticklabels:
            ticklabels[-1].set_rotation(0)
            ticklabels[-1].set_ha("center")
        ax.tick_params(axis="x", which="major", bottom=False, labelbottom=False)
        ax.tick_params(axis="x", which="minor", pad=5)

    axes[-1].set_xlabel("Benchmark circuit (grouped by size)")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.suptitle(
        "Benefits of H Commutation, Merging, and Packing vs. a Naive Permutation Without Techniques Applied"
    )
    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    if settings.output_path is not None:
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(settings.output_path, dpi=200)
        print(f"[merging_improvement_benchmark_stylized] Saved plot to {settings.output_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description="Stylized benchmark with stacked benefits from H commutation, merging, and packing."
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        default=Path(
            "benchmarks/qasmbench"
        ),
        help="Root directory containing QASMbench size folders (small/medium/large).",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=2500,
        help="Exclude proxy circuits with more than this many gates (post-transpile).",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=150,
        help="Exclude proxy circuits with more than this many qubits (from README metadata).",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for proxy circuit creation (set <=0 to disable).",
    )
    parser.add_argument(
        "--drop-equal-packings",
        # default=True,
        action="store_true",
        help="Skip circuits where H-only and H+merge+packing are equal for all metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized.svg"
        ),
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--h-commutation",
        choices=H_COMMUTATION_MODES,
        default="best",
        help="best: keep whichever of on/off wins, per metric per stage (default). "
             "auto: the planner's per-circuit decision. on: forced everywhere, "
             "reproducing the published figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (useful for headless runs).",
    )
    args = parser.parse_args()

    return BenchmarkSettings(
        qasm_root=args.qasm_root,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        drop_equal_packings=args.drop_equal_packings,
        h_commutation=args.h_commutation,
        output_path=args.output,
        show_plot=not args.no_show,
    )


def _load_group_circuits(
    qasm_root: Path,
    group: str,
    settings: BenchmarkSettings,
) -> Sequence[Tuple[str, qiskit.QuantumCircuit]]:
    timeout_seconds = settings.proxy_timeout_seconds
    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None
    circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=False,
    )
    if settings.max_proxy_gates is not None:
        circuits = [
            (name, qc)
            for name, qc in circuits
            if qc.size() <= settings.max_proxy_gates
        ]
    circuits = [
        (name, qc)
        for name, qc in circuits
        if not name.lower().endswith("_n2")
    ]
    return circuits


def main() -> None:
    settings = parse_args()
    all_rows: List[pd.DataFrame] = []
    for group in ["small", "medium", "large"]:
        circuits = _load_group_circuits(settings.qasm_root, group, settings)
        if not circuits:
            print(f"[merging_improvement_benchmark_stylized] No circuits loaded for {group}.")
            continue
        group_df = run_experiment(circuits, group, settings)
        all_rows.append(group_df)

    if not all_rows:
        raise ValueError(f"No proxy circuits loaded from {settings.qasm_root}")
    df = pd.concat(all_rows, ignore_index=True)
    print_metric_averages(df)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
