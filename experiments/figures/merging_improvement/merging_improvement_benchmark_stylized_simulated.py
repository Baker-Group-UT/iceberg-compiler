"""
merging_improvement_benchmark_stylized_simulated.py
--------------------------------------------------

Benchmark-only stylized ablation figure for H commutation, merging, and packing.

This is a simulated variant of merging_improvement_benchmark_stylized. It keeps
the original benchmark-bar layout, but adds two extra rows:

1. 1q gate reduction
2. 2q gate reduction
3. Depth reduction
4. LSR improvement
5. TVD improvement

Simulation uses a per-circuit baseline calibration:
  * initial alpha scales as 1 / m using the baseline translated circuit gate count
  * a cheap baseline pilot measures selection rate
  * alpha is corrected toward a target selection rate
  * a second pilot at corrected alpha estimates the final selection rate
  * final repetitions are chosen from target_kept_shots / estimated_sel_rate

The packing row comes from the `packing` package via `common.pack_program`, not from
`packer/merge_packer.py`, which is archived. Both place two logical qubits per patch, so
the figure measures the same thing it always did; see `common.pack_program` for what the
call had to change to say.

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.figures.merging_improvement.common import pack_program
from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import (
    factor_from_pct,
    stacked_percent,
    stamp_provenance,
)
from experiments.shared.simulation import SimSettings, noise_profile_for
from experiments.shared.parallel import add_jobs_argument, parallel_map
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.config_presets import ScaledSimulator

Perm = List[Tuple[int, int]]

#: This figure's size ladder is its own, and deliberately *not* the document's 10/9/8pt.
#: The canvas is 22 inches and `evaluation.tex` places it at `.95\linewidth` in a `figure*`,
#: so LaTeX shrinks everything to 30.9% -- 24pt here prints at about 7.4pt. Scaling the
#: ladder up to compensate was tried and reverted: 71 circuits share the x axis at roughly
#: 16.5pt per slot, so at 45 degrees those labels cannot exceed about 12pt whatever the rest
#: of the figure does, and raising only the labels around them left the figure visibly out
#: of proportion. The ladder below is internally consistent, which matters more here than
#: agreeing with the body text; the real fix is a wider figure, not a bigger number.
AXIS_PT = 24

plt.rcParams.update(
    {
        # Keep text as real `<text>` elements in the SVG instead of outlining it into
        # paths. matplotlib's default `svg.fonttype` is "path", which is why this figure's
        # SVG contained zero text elements and could not be edited in Inkscape or searched
        # for a label. Every other figure gets this from `experiments/shared/aqorn.mplstyle`;
        # this one keeps its own canvas and size ladder, so it needs the key spelled out.
        # `pdf.fonttype: 42` embeds TrueType rather than Type 3, which IEEE PDF eXpress
        # rejects in the final version.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        # Times, matching the paper's body font and `aqorn.mplstyle`. This figure drew in
        # Helvetica while every other one drew in Times. Same fallback chain as the shared
        # style, so a machine without Times New Roman degrades to a metric-compatible clone
        # rather than substituting something arbitrary. Sizes stay on this figure's own
        # ladder: it draws on a 22-inch canvas rather than at final size, so the shared
        # 10/9/8pt tiers would not mean what they mean elsewhere.
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman", "Times",
            "DejaVu Serif",
        ],
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "figure.titlesize": 16,
    }
)

METRICS = {
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "#1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "#2q gates"),
    "circuit_depth": ("pct_reduction_circuit_depth", "depth"),
    "logical_selection_rate": ("pct_improvement_logical_selection_rate", "LSR"),
    "logical_total_variation_distance": (
        "pct_reduction_logical_total_variation_distance",
        "TVD",
    ),
}


def as_factor(metric_column: str, cumulative_pct: float) -> float:
    """
    Cumulative percentage -> the multiplier the *prose* quotes, not the axis.

    The axis is in percent (see `plot_results`), but the paper reports headline numbers as
    factors -- "1.5x shallower" rather than "34% shallower" -- so `print_metric_averages`
    reports both. Direction comes from the column name, which is where this project's sign
    convention already lives: `pct_reduction_*` is lower-is-better, `pct_improvement_*` is
    not. The arithmetic is in `factor_from_pct`.
    """
    if not np.isfinite(cumulative_pct):
        return 1.0
    return factor_from_pct(
        cumulative_pct, reduction=metric_column.startswith("pct_reduction_")
    )


#: Stage order, bottom of the stack upwards. Each is cumulative on the one before it.
COMPONENT_ORDER = ("h_commutation", "merging", "packing")


#: How the summary bar averages: "bars" (mean of the clipped per-circuit bars, the default)
#: or "mean" (unclipped per-stage means, the numbers the prose quotes).
AVERAGE_BAR = "bars"


def corpus_boundaries(frame: pd.DataFrame, metric_column: str) -> np.ndarray:
    """One row of cumulative percentage boundaries per circuit, in `COMPONENT_ORDER`."""
    if frame.empty:
        return np.empty((0, len(COMPONENT_ORDER) + 1))
    pivot = frame.pivot_table(
        index="circuit_name", columns="component", values=metric_column, aggfunc="mean",
    )
    for component in COMPONENT_ORDER:
        if component not in pivot.columns:
            pivot[component] = 0.0
    pivot = pivot[list(COMPONENT_ORDER)].fillna(0.0).astype(float)
    if pivot.empty:
        return np.empty((0, len(COMPONENT_ORDER) + 1))
    return np.array([stacked_percent(row.tolist()) for _, row in pivot.iterrows()])


@dataclass
class BenchmarkSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    max_proxy_gates: int | None = 2500
    max_proxy_qubits: int | None = 150
    proxy_timeout_seconds: float | None = 1.0
    drop_equal_packings: bool = False
    alpha_reference: float = 0.1
    reference_gate_count: float = 200.0
    #: Kept in step with `SimSettings.target_selection_rate`; see the note there.
    target_selection_rate: float = 0.06
    pilot_repetitions: int = 4_000
    target_kept_shots: int = 20_000
    min_repetitions: int = 20_000
    max_repetitions: int = 1_000_000
    stim: bool = True
    distance_noise: bool = True
    distance_scalar: float = 1.0
    idle_noise: bool = True
    idle_scalar: float = 1.0
    seed: int = 10
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized_simulated.svg"
        )
    )
    raw_csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized_simulated.csv"
        )
    )
    plot_csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized_simulated_plot_data.csv"
        )
    )
    from_csv: Path | None = None
    show_plot: bool = False
    #: Worker processes for the circuit sweep. None takes `parallel.DEFAULT_JOBS`.
    jobs: int | None = None


DEFAULT_SETTINGS = BenchmarkSettings()


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def pct_improvement(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (challenger - reference) / reference


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    patch_capacity = max(1, math.ceil(num_qubits / n_codes))
    perm: Perm = []
    for i in range(num_qubits):
        code = min(i // patch_capacity, n_codes - 1)
        perm.append((code, i % patch_capacity))
    return perm


def ensure_measurements(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    if circuit.num_clbits > 0 and any(op.operation.name == "measure" for op in circuit.data):
        return circuit
    measured = copy.deepcopy(circuit)
    measured.measure_all()
    return measured


def build_pipeline_config(
    settings: BenchmarkSettings,
    *,
    alpha: float,
    repetitions: int,
    seed: int,
    enable_h_commutation: bool,
    enable_merging: bool,
) -> PipelineConfig:
    sim_cfg = ScaledSimulator(alpha=alpha, repetitions=repetitions, seed=seed)
    if settings.stim:
        sim_cfg = sim_cfg.with_stim()
    if settings.distance_noise:
        sim_cfg = sim_cfg.with_distance_noise(distance_scalar=settings.distance_scalar)
    if settings.idle_noise:
        sim_cfg = sim_cfg.with_idle_noise(idle_time_scalar=settings.idle_scalar)

    config = PipelineConfig(verbose=False, save=False, sim_config=sim_cfg, seed=seed)
    config.translator_config.enable_h_commutation = enable_h_commutation
    config.translator_config.enable_merging = enable_merging
    return config


def compute_translated_circuit(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> qiskit.QuantumCircuit:
    translated = translate_permutation(program_circuit, permutation, config=config)
    return translated.circuit


def compute_translation_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Tuple[int, int, int]:
    circuit = compute_translated_circuit(program_circuit, permutation, config)
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    return one_q, two_q, depth


def evaluate_variant(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Dict[str, float]:
    record = run_specific_scan(program_circuit=program_circuit, permutations=[permutation], config=config)
    if record.dataframe is None or record.dataframe.df.empty:
        raise RuntimeError("run_specific_scan returned no dataframe.")
    row = record.dataframe.df.iloc[0]
    return {
        "logical_selection_rate": float(row["logical_selection_rate"]),
        "logical_total_variation_distance": float(row["logical_total_variation_distance"]),
    }


def calibrated_alpha_from_gate_count(
    gate_count: int,
    settings: BenchmarkSettings,
) -> float:
    gate_count = max(1, int(gate_count))
    return settings.alpha_reference * (settings.reference_gate_count / float(gate_count))


def corrected_alpha(
    alpha: float,
    observed_selection_rate: float,
    target_selection_rate: float,
) -> float:
    eps = 1e-6
    observed = float(np.clip(observed_selection_rate, eps, 1.0 - eps))
    target = float(np.clip(target_selection_rate, eps, 1.0 - eps))
    return alpha * math.log(target) / math.log(observed)


def repetitions_from_selection_rate(
    selection_rate: float,
    settings: BenchmarkSettings,
) -> int:
    denom = max(selection_rate, 1e-4)
    reps = int(math.ceil(settings.target_kept_shots / denom))
    return int(np.clip(reps, settings.min_repetitions, settings.max_repetitions))


def _run_circuit(job) -> List[Dict[str, int | float | str]]:
    """
    One benchmark circuit: four translated arms, a two-pilot calibration, four simulations.

    Module-level, and taking everything it needs as an argument, because `parallel_map`
    spawns rather than forks and a closure cannot cross that boundary. Returns this
    circuit's rows -- three, one per pipeline stage -- or an empty list where the serial
    loop hit `continue` under `--drop-equal-packings`.

    **The three translate configs are built here rather than once for the whole sweep.**
    The serial version hoisted them above the loop and then mutated
    `enable_h_commutation` on two of them inside it, which is correct only because every
    iteration wrote the field before reading it. That is exactly the shared mutable state
    that does not survive concurrency, and rebuilding three dataclasses per circuit costs
    nothing next to the shots below.

    `circuit_idx` stays the index *within its size group*, because it feeds the seed
    derivation (`seed * 1_000_003 + circuit_idx * 17 + n`). Numbering circuits globally
    instead would silently reseed every simulation and stop this reproducing the CSV
    already on disk.
    """
    (
        settings,
        size_group,
        circuit_idx,
        circuit_name,
        resource_circuit,
        simulation_circuit,
    ) = job

    baseline_translate_cfg = PipelineConfig()
    baseline_translate_cfg.translator_config.enable_merging = False
    baseline_translate_cfg.translator_config.enable_h_commutation = False

    h_only_translate_cfg = PipelineConfig()
    h_only_translate_cfg.translator_config.enable_merging = False
    h_only_translate_cfg.translator_config.enable_h_commutation = True

    h_merge_translate_cfg = PipelineConfig()
    h_merge_translate_cfg.translator_config.enable_merging = True
    h_merge_translate_cfg.translator_config.enable_h_commutation = True

    rows: List[Dict[str, int | float | str]] = []

    resource_program_circuit = ensure_measurements(resource_circuit)
    simulation_program_circuit = ensure_measurements(simulation_circuit)
    n_qubits = resource_program_circuit.num_qubits
    target_codes = math.ceil(n_qubits / 2)
    tight_perm = tight_trivial_packing(n_qubits, target_codes)
    # Bug B6: hand the planner the machine the simulator will actually charge for. Without
    # this the placement is chosen against movement_cost = idle_cost = 0 while `_sim_config`
    # below switches both channels on, so the packing arm was scored on a placement
    # optimised for different hardware. The profile mirrors the same four flags.
    packed_perm, commute = pack_program(
        resource_program_circuit,
        noise=noise_profile_for(
            SimSettings(
                distance_noise=settings.distance_noise,
                distance_scalar=settings.distance_scalar,
                idle_noise=settings.idle_noise,
                idle_time_scalar=settings.idle_scalar,
                stim=settings.stim,
                shot_progress=False,
            )
        ),
    )
    # Honour the planner's per-circuit commutation decision, and translate every
    # row under it. When it comes back False the H-commutation row collapses onto
    # the baseline and its segment is legitimately zero -- the shipped pipeline
    # declined the pass on that circuit -- so the ablation stays cumulative
    # instead of crediting a technique the compiler did not use.
    h_only_translate_cfg.translator_config.enable_h_commutation = commute
    h_merge_translate_cfg.translator_config.enable_h_commutation = commute

    base_one_q, base_two_q, base_depth = compute_translation_metrics(
        resource_program_circuit, tight_perm, baseline_translate_cfg
    )
    h_only_one_q, h_only_two_q, h_only_depth = compute_translation_metrics(
        resource_program_circuit, tight_perm, h_only_translate_cfg
    )
    h_merge_one_q, h_merge_two_q, h_merge_depth = compute_translation_metrics(
        resource_program_circuit, tight_perm, h_merge_translate_cfg
    )
    packed_one_q, packed_two_q, packed_depth = compute_translation_metrics(
        resource_program_circuit, packed_perm, h_merge_translate_cfg
    )

    baseline_gate_count = max(1, base_one_q + base_two_q)
    alpha_initial = calibrated_alpha_from_gate_count(baseline_gate_count, settings)

    pilot_baseline_cfg = build_pipeline_config(
        settings,
        alpha=alpha_initial,
        repetitions=settings.pilot_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 1,
        enable_h_commutation=False,
        enable_merging=False,
    )
    pilot_metrics_initial = evaluate_variant(
        simulation_program_circuit, tight_perm, pilot_baseline_cfg
    )

    alpha_final = corrected_alpha(
        alpha_initial,
        pilot_metrics_initial["logical_selection_rate"],
        settings.target_selection_rate,
    )

    pilot_verify_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=settings.pilot_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 2,
        enable_h_commutation=False,
        enable_merging=False,
    )
    pilot_metrics_final = evaluate_variant(
        simulation_program_circuit, tight_perm, pilot_verify_cfg
    )
    final_repetitions = repetitions_from_selection_rate(
        pilot_metrics_final["logical_selection_rate"],
        settings,
    )

    baseline_sim_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=final_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 3,
        enable_h_commutation=False,
        enable_merging=False,
    )
    h_only_sim_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=final_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 4,
        enable_h_commutation=commute,
        enable_merging=False,
    )
    h_merge_sim_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=final_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 5,
        enable_h_commutation=commute,
        enable_merging=True,
    )
    packed_sim_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=final_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 6,
        enable_h_commutation=commute,
        enable_merging=True,
    )

    baseline_sim = evaluate_variant(simulation_program_circuit, tight_perm, baseline_sim_cfg)
    h_only_sim = evaluate_variant(simulation_program_circuit, tight_perm, h_only_sim_cfg)
    h_merge_sim = evaluate_variant(simulation_program_circuit, tight_perm, h_merge_sim_cfg)
    packed_sim = evaluate_variant(simulation_program_circuit, packed_perm, packed_sim_cfg)

    h_cum_pct_one_q = pct_reduction(base_one_q, h_only_one_q)
    h_cum_pct_two_q = pct_reduction(base_two_q, h_only_two_q)
    h_cum_pct_depth = pct_reduction(base_depth, h_only_depth)

    h_merge_cum_pct_one_q = pct_reduction(base_one_q, h_merge_one_q)
    h_merge_cum_pct_two_q = pct_reduction(base_two_q, h_merge_two_q)
    h_merge_cum_pct_depth = pct_reduction(base_depth, h_merge_depth)

    packed_cum_pct_one_q = pct_reduction(base_one_q, packed_one_q)
    packed_cum_pct_two_q = pct_reduction(base_two_q, packed_two_q)
    packed_cum_pct_depth = pct_reduction(base_depth, packed_depth)

    h_cum_lsr = pct_improvement(
        baseline_sim["logical_selection_rate"],
        h_only_sim["logical_selection_rate"],
    )
    h_merge_cum_lsr = pct_improvement(
        baseline_sim["logical_selection_rate"],
        h_merge_sim["logical_selection_rate"],
    )
    packed_cum_lsr = pct_improvement(
        baseline_sim["logical_selection_rate"],
        packed_sim["logical_selection_rate"],
    )

    h_cum_tvd = pct_reduction(
        baseline_sim["logical_total_variation_distance"],
        h_only_sim["logical_total_variation_distance"],
    )
    h_merge_cum_tvd = pct_reduction(
        baseline_sim["logical_total_variation_distance"],
        h_merge_sim["logical_total_variation_distance"],
    )
    packed_cum_tvd = pct_reduction(
        baseline_sim["logical_total_variation_distance"],
        packed_sim["logical_total_variation_distance"],
    )

    component_payloads = [
        {
            "component": "h_commutation",
            "pct_reduction_one_qubit_gates": h_cum_pct_one_q,
            "pct_reduction_two_qubit_gates": h_cum_pct_two_q,
            "pct_reduction_circuit_depth": h_cum_pct_depth,
            "pct_improvement_logical_selection_rate": h_cum_lsr,
            "pct_reduction_logical_total_variation_distance": h_cum_tvd,
        },
        {
            "component": "merging",
            "pct_reduction_one_qubit_gates": h_merge_cum_pct_one_q - h_cum_pct_one_q,
            "pct_reduction_two_qubit_gates": h_merge_cum_pct_two_q - h_cum_pct_two_q,
            "pct_reduction_circuit_depth": h_merge_cum_pct_depth - h_cum_pct_depth,
            "pct_improvement_logical_selection_rate": h_merge_cum_lsr - h_cum_lsr,
            "pct_reduction_logical_total_variation_distance": h_merge_cum_tvd - h_cum_tvd,
        },
        {
            "component": "packing",
            "pct_reduction_one_qubit_gates": packed_cum_pct_one_q - h_merge_cum_pct_one_q,
            "pct_reduction_two_qubit_gates": packed_cum_pct_two_q - h_merge_cum_pct_two_q,
            "pct_reduction_circuit_depth": packed_cum_pct_depth - h_merge_cum_pct_depth,
            "pct_improvement_logical_selection_rate": packed_cum_lsr - h_merge_cum_lsr,
            "pct_reduction_logical_total_variation_distance": packed_cum_tvd - h_merge_cum_tvd,
        },
    ]

    if settings.drop_equal_packings:
        if (
            h_only_one_q == packed_one_q
            and h_only_two_q == packed_two_q
            and h_only_depth == packed_depth
            and abs(h_only_sim["logical_selection_rate"] - packed_sim["logical_selection_rate"]) <= 1e-12
            and abs(
                h_only_sim["logical_total_variation_distance"]
                - packed_sim["logical_total_variation_distance"]
            )
            <= 1e-12
        ):
            return []

    for payload in component_payloads:
        rows.append(
            {
                "size_group": size_group,
                "circuit_idx": circuit_idx,
                "circuit_name": circuit_name,
                "num_qubits": n_qubits,
                "target_codes": target_codes,
                "baseline_gate_count": baseline_gate_count,
                "alpha_initial": alpha_initial,
                "alpha_final": alpha_final,
                "pilot_initial_lsr": pilot_metrics_initial["logical_selection_rate"],
                "pilot_final_lsr": pilot_metrics_final["logical_selection_rate"],
                "final_repetitions": final_repetitions,
                "baseline_lsr": baseline_sim["logical_selection_rate"],
                "baseline_tvd": baseline_sim["logical_total_variation_distance"],
                **payload,
            }
        )

    return rows


#: Size groups in the order the figure draws them, which is also the canonical row order
#: of the CSV. Scheduling order is different -- see `run_experiment`.
GROUP_ORDER: Tuple[str, ...] = ("small", "medium", "large")


def run_experiment(settings: BenchmarkSettings) -> pd.DataFrame:
    """
    Every size group in one pool.

    **The three groups are flattened into a single work list** rather than parallelised
    one group at a time. Circuit cost here is extremely uneven -- the large group runs to
    140 qubits and several thousand gates while small finishes almost immediately -- so a
    barrier between groups would leave most of the machine idle behind the last few
    circuits of `large`. One pool lets the small circuits backfill slots that the large
    ones have not finished holding.

    **Work is scheduled longest-first**, by descending qubit count. That is the standard
    heuristic for a pool with a long tail: start the expensive items while there are still
    cheap ones left to fill the gaps with. Scheduling last would leave a 140-qubit circuit
    starting on an otherwise-empty machine at the very end of the run.

    Rows are restored to canonical (group, index) order afterwards, so the CSV is
    directly comparable against a serial run rather than ordered by whichever worker
    happened to finish first.
    """
    jobs: List[Tuple] = []
    for size_group in GROUP_ORDER:
        circuits = _load_group_circuits(settings.qasm_root, size_group, settings)
        if not circuits:
            print(
                "[merging_improvement_benchmark_stylized_simulated] "
                f"No circuits loaded for {size_group}."
            )
            continue
        for circuit_idx, (name, resource, simulation) in enumerate(circuits):
            jobs.append((settings, size_group, circuit_idx, name, resource, simulation))

    if not jobs:
        raise ValueError(f"No proxy circuits loaded from {settings.qasm_root}")

    print(
        "[merging_improvement_benchmark_stylized_simulated] "
        f"{len(jobs)} circuits across {len(GROUP_ORDER)} size groups"
    )

    schedule = sorted(range(len(jobs)), key=lambda index: -jobs[index][4].num_qubits)
    produced = parallel_map(
        _run_circuit,
        [jobs[index] for index in schedule],
        jobs=settings.jobs,
        label="circuits",
    )

    batches: List[object] = [None] * len(jobs)
    for slot, canonical in enumerate(schedule):
        batches[canonical] = produced[slot]
    rows = [row for batch in batches if batch for row in batch]
    return pd.DataFrame(rows)


def print_metric_averages(df: pd.DataFrame) -> None:
    """
    Console summary of exactly what the figure draws, per size group and overall.

    Percentages first, matching the axis: each is the mean over circuits of the total the
    bar draws, which is the height of the summary bar. The factor in brackets is the same
    number in the convention the paper's prose uses -- "1.5x shallower" rather than "34%
    shallower" -- converted from the mean rather than averaged separately, so the two lines
    can never disagree.

    The corpus line is a direct mean over every circuit rather than a size-weighted blend
    of the three group means; the two agree to three decimals and the direct one is the
    number to quote.
    """
    print("[merging_improvement_benchmark_stylized_simulated] Plot averages (matching the bars)")
    size_groups = ["small", "medium", "large"]
    group_counts: Dict[str, int] = {}

    def summarise(frame: pd.DataFrame) -> Tuple[List[str], int]:
        parts: List[str] = []
        for _, (metric_column, metric_label) in METRICS.items():
            # Same guard, third aggregation path. See `TVD_SATURATION`.
            scoped = frame
            if metric_column == "pct_reduction_logical_total_variation_distance":
                scoped = frame.loc[_tvd_is_resolved(frame)]
            rows = corpus_boundaries(scoped, metric_column)
            if rows.size == 0:
                parts.append(f"{metric_label}=n/a")
                continue
            mean = float(rows.mean(axis=0)[-1])
            parts.append(
                f"{metric_label}={mean:.2f}% ({as_factor(metric_column, mean):.2f}×)"
            )
        return parts, int(frame["circuit_name"].nunique())

    for size_group in size_groups:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        parts, count = summarise(group_df)
        group_counts[size_group] = count
        print(f"  {size_group}:")
        print("    plotted total: " + ", ".join(parts))

    if group_counts:
        parts, _ = summarise(df)
        count_parts = [f"{group}={group_counts[group]}" for group in size_groups if group in group_counts]
        print("  corpus overall: " + ", ".join(parts))
        print("    circuits: " + ", ".join(count_parts) + f", total={sum(group_counts.values())}")

    if not df.empty:
        print(
            "  calibration averages: "
            f"alpha={df['alpha_final'].mean():.4f}, "
            f"pilot_lsr={df['pilot_final_lsr'].mean():.4f}, "
            f"repetitions={df['final_repetitions'].mean():.0f}"
        )


#: Above this baseline TVD the measurement carries no information, and the row must be
#: withheld from the TVD metric.
#:
#: TVD is bounded by the sampling floor `1 - N/S`, where `S` is the output distribution's
#: support and `N` the accepted shots. On a wide circuit the support outruns any affordable
#: shot count and *both* arms score ~1.0, so their difference is exactly zero -- not "no
#: improvement", but "no measurement". Five circuits here do it, all ising at n >= 26:
#: `ising_n26` at 0.9997 and `n34`, `n42`, `n66`, `n98` at 1.000000. Averaging those zeros
#: into the TVD row pulled every component down by about 7% relative (h_commutation
#: +13.77 -> +14.82, merging +8.61 -> +9.26, packing +2.98 -> +3.21).
#:
#: Withheld from **TVD only**. Their gate counts and depth are exact translation properties
#: and their selection rate is a binomial over ~150k shots, so those rows stay.
TVD_SATURATION = 0.95


def _tvd_is_resolved(df: pd.DataFrame) -> pd.Series:
    """Per-row mask: is this circuit's TVD measurable at all? See `TVD_SATURATION`."""
    if "baseline_tvd" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["baseline_tvd"].astype(float) <= TVD_SATURATION


def build_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for size_group in ["small", "medium", "large"]:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        for metric_key, (metric_column, metric_label) in METRICS.items():
            # Saturated circuits contribute a hard zero to TVD that is an absence of
            # measurement, not an absence of improvement. Drop them from that metric only.
            scoped = group_df
            if metric_column == "pct_reduction_logical_total_variation_distance":
                scoped = group_df.loc[_tvd_is_resolved(group_df)]
                withheld = sorted(
                    set(group_df["circuit_name"]) - set(scoped["circuit_name"])
                )
                if withheld:
                    print(f"[merging_improvement_benchmark_stylized_simulated] TVD "
                          f"withheld for {len(withheld)} saturated {size_group} circuit(s) "
                          f"(baseline TVD > {TVD_SATURATION}): {', '.join(withheld)}")
                if scoped.empty:
                    continue
            metric_group = scoped[["circuit_name", "component", metric_column]].copy()
            pivot = (
                metric_group.pivot_table(
                    index="circuit_name",
                    columns="component",
                    values=metric_column,
                    aggfunc="mean",
                )
                .reset_index()
            )
            for component in ["h_commutation", "merging", "packing"]:
                if component not in pivot.columns:
                    pivot[component] = 0.0
                pivot[component] = pivot[component].fillna(0.0).astype(float)
            pivot["sort_total"] = pivot[["h_commutation", "merging", "packing"]].sum(axis=1)
            pivot = pivot.sort_values(by="circuit_name", ascending=True)
            for plot_order, row in enumerate(pivot.to_dict(orient="records"), start=1):
                rows.append(
                    {
                        "size_group": size_group,
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "circuit_name": str(row["circuit_name"]),
                        "plot_order": plot_order,
                        "h_commutation": float(row["h_commutation"]),
                        "merging": float(row["merging"]),
                        "packing": float(row["packing"]),
                        "total": float(row["sort_total"]),
                    }
                )
            rows.append(
                {
                    "size_group": "all",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "circuit_name": "average",
                    "plot_order": 0,
                    "h_commutation": float(
                        df.loc[df["component"] == "h_commutation", metric_column].astype(float).mean()
                    ),
                    "merging": float(
                        df.loc[df["component"] == "merging", metric_column].astype(float).mean()
                    ),
                    "packing": float(
                        df.loc[df["component"] == "packing", metric_column].astype(float).mean()
                    ),
                    "total": float(
                        df.loc[df["component"] == "h_commutation", metric_column].astype(float).mean()
                        + df.loc[df["component"] == "merging", metric_column].astype(float).mean()
                        + df.loc[df["component"] == "packing", metric_column].astype(float).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, settings: BenchmarkSettings) -> None:
    components = [
        ("h_commutation", "H commutation benefit", "#3b4c56"),
        ("merging", "+ Merging benefit", "#cdc2c0"),
        ("packing", "+ Packing benefit", "#0097fd"),
    ]
    size_groups = ["small", "medium", "large"]

    fig, axes = plt.subplots(nrows=len(METRICS), ncols=1, figsize=(22, 20), sharex=True)
    bar_width = 0.825
    gap = 4.0
    bar_step = 1.1

    ordered_circuits_by_group: Dict[str, List[str]] = {}
    for size_group in size_groups:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        circuit_names = sorted(group_df["circuit_name"].unique().tolist())
        ordered_circuits_by_group[size_group] = circuit_names

    for ax, (_, (metric_column, metric_label)) in zip(axes, METRICS.items()):
        all_positions: List[float] = []
        all_circuit_labels: List[str] = []
        current_x = 0.0
        highest = 0.0

        for size_group in size_groups:
            ordered_circuits = ordered_circuits_by_group.get(size_group, [])
            if not ordered_circuits:
                continue
            group_df = df.loc[df["size_group"] == size_group]
            # Saturated circuits carry a hard zero that is an absence of measurement, not
            # an absence of improvement. See `TVD_SATURATION`. This path re-aggregates
            # independently of `build_plot_dataframe`, so the guard has to be repeated --
            # fixing only that one left the rendered figure unchanged.
            if metric_column == "pct_reduction_logical_total_variation_distance":
                # Filter the *data*, never `ordered_circuits`: the x positions come from
                # that list, and with `sharex=True` the last axis drawn -- TVD, being last
                # in METRICS -- sets the shared x limit for every row. Shrinking it here
                # pulled the limit inwards and pushed the other four rows' average bars off
                # the canvas. A withheld circuit keeps its slot and simply draws no bar,
                # which is also the honest rendering: a gap, not a zero.
                group_df = group_df.loc[_tvd_is_resolved(group_df)]
            pivot = (
                group_df.pivot_table(
                    index="circuit_name",
                    columns="component",
                    values=metric_column,
                    aggfunc="mean",
                )
                .reset_index()
            )
            for component, _, _ in components:
                if component not in pivot.columns:
                    pivot[component] = 0.0
                # Deliberately *not* clipped per stage. `stacked_percent` clips the running
                # total instead, which is what keeps each bar's height equal to what the
                # circuit actually reached; see that function on what the old per-stage
                # clip did to `ising_n98`.
                pivot[component] = pivot[component].fillna(0.0).astype(float)
            pivot = (
                pivot.set_index("circuit_name")
                .reindex(ordered_circuits)
                .fillna(0.0)
                .reset_index()
            )
            boundaries = np.array(
                [
                    stacked_percent([row[component] for component, _, _ in components])
                    for _, row in pivot.iterrows()
                ]
            )

            positions = np.arange(len(ordered_circuits)) * bar_step + current_x
            for index, (component, label, color) in enumerate(components):
                ax.bar(
                    positions,
                    boundaries[:, index + 1] - boundaries[:, index],
                    width=bar_width,
                    bottom=boundaries[:, index],
                    color=color,
                    edgecolor="white",
                    alpha=0.85,
                    label=label,
                )
            highest = max(highest, float(boundaries[:, -1].max()))

            all_positions.extend(positions.tolist())
            all_circuit_labels.extend(ordered_circuits)
            current_x = positions[-1] + gap if len(positions) else current_x + gap

        avg_x = current_x + 1.0
        avg_width = 5 * bar_width
        avg_source = df
        if metric_column == "pct_reduction_logical_total_variation_distance":
            avg_source = df.loc[_tvd_is_resolved(df)]
        # Arithmetic mean of the per-circuit boundaries. Boundaries are cumulative and
        # the mean is linear, so each segment is the mean of that stage's segments and the
        # top is the mean of the totals -- the summary bar is exactly the average of the
        # bars behind it, with no separate aggregation path to drift out of step.
        if AVERAGE_BAR == "mean":
            # The corpus mean the prose quotes (10-11): each stage's per-circuit delta
            # averaged unclipped, then stacked once, so the bar's top is the mean net change
            # including the circuits that got worse. The default ("bars") is the mean of the
            # clipped per-circuit bars, which reads higher wherever some circuits regress.
            stage = avg_source.pivot_table(index="circuit_name", columns="component",
                                           values=metric_column, aggfunc="mean")
            means = [float(stage[c].fillna(0.0).mean()) if c in stage.columns else 0.0
                     for c in COMPONENT_ORDER]
            avg_boundaries = np.array(stacked_percent(means))
        else:
            avg_boundaries = corpus_boundaries(avg_source, metric_column).mean(axis=0)
        for index, (component, label, color) in enumerate(components):
            ax.bar(
                avg_x,
                avg_boundaries[index + 1] - avg_boundaries[index],
                width=avg_width,
                bottom=avg_boundaries[index],
                color=color,
                edgecolor="white",
                alpha=0.95,
                label=label,
            )
        highest = max(highest, float(avg_boundaries[-1]))

        ax.axvline(current_x - gap / 2.0, color="#bdbdbd", linestyle=":", linewidth=1.0)
        left_gap_to_avg = (avg_x - avg_width / 2.0) - (current_x - gap / 2.0)
        right_limit = avg_x + avg_width / 2.0 + max(0.0, left_gap_to_avg)
        left_limit, _ = ax.get_xlim()
        ax.set_xlim(left=left_limit, right=right_limit)
        all_positions.append(avg_x)
        all_circuit_labels.append("average")

        ax.set_ylabel(metric_label, fontsize=AXIS_PT)
        # Percent, linear, and that is deliberate for *this* figure. A factor axis was
        # tried: percent compresses the spread and factors expand it, and here the spread
        # is the problem. `ising_n98` depth is 90.1% against a 27% median -- 3.3x on a
        # linear percent axis, but 10.1x against 1.37x as factors, which needs a log axis,
        # which cannot carry evenly spaced round gridlines. The paper still quotes factors
        # in prose; `print_metric_averages` reports both.
        ax.set_ylim(bottom=0.0, top=highest * 1.06)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
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

    axes[-1].set_xlabel("Benchmark circuit (grouped by size)", fontsize=AXIS_PT)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", frameon=False)
    fig.supylabel("Percent improvement / reduction", fontsize=AXIS_PT)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.output_path, dpi=200)
    print(
        "[merging_improvement_benchmark_stylized_simulated] "
        f"Saved plot to {settings.output_path}"
    )

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> BenchmarkSettings:
    parser = argparse.ArgumentParser(
        description=(
            "Stylized benchmark with stacked H commutation, merging, and packing "
            "benefits, including simulated LSR/TVD columns."
        )
    )
    parser.add_argument(
        "--qasm-root",
        type=Path,
        default=Path("benchmarks/qasmbench"),
        help="Root directory containing QASMbench size folders (small/medium/large).",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=DEFAULT_SETTINGS.max_proxy_gates,
        help="Exclude proxy circuits with more than this many gates.",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=DEFAULT_SETTINGS.max_proxy_qubits,
        help="Exclude proxy circuits with more than this many qubits.",
    )
    parser.add_argument(
        "--proxy-timeout",
        type=float,
        default=DEFAULT_SETTINGS.proxy_timeout_seconds,
        help="Timeout in seconds for proxy circuit creation (set <=0 to disable).",
    )
    parser.add_argument(
        "--drop-equal-packings",
        action="store_true",
        help="Skip circuits where H-only and H+merge+packing are equal for all plotted metrics.",
    )
    parser.add_argument(
        "--alpha-reference",
        type=float,
        default=DEFAULT_SETTINGS.alpha_reference,
        help="Reference alpha used for the reference gate count.",
    )
    parser.add_argument(
        "--reference-gate-count",
        type=float,
        default=DEFAULT_SETTINGS.reference_gate_count,
        help="Gate count corresponding to alpha-reference.",
    )
    parser.add_argument(
        "--target-selection-rate",
        type=float,
        default=DEFAULT_SETTINGS.target_selection_rate,
        help="Baseline target logical selection rate used for alpha calibration.",
    )
    parser.add_argument(
        "--pilot-repetitions",
        type=int,
        default=DEFAULT_SETTINGS.pilot_repetitions,
        help="Pilot repetitions for each baseline calibration pass.",
    )
    parser.add_argument(
        "--target-kept-shots",
        type=int,
        default=DEFAULT_SETTINGS.target_kept_shots,
        help="Approximate kept postselected shots desired for the final run.",
    )
    parser.add_argument(
        "--min-repetitions",
        type=int,
        default=DEFAULT_SETTINGS.min_repetitions,
        help="Lower clamp for final repetitions.",
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        default=DEFAULT_SETTINGS.max_repetitions,
        help="Upper clamp for final repetitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SETTINGS.seed,
        help="Base seed for calibration and final simulations.",
    )
    add_jobs_argument(parser)
    parser.add_argument(
        "--no-stim",
        action="store_true",
        help="Disable Stim backend.",
    )
    parser.add_argument(
        "--no-distance-noise",
        action="store_true",
        help="Disable distance-dependent noise scaling.",
    )
    parser.add_argument(
        "--distance-scalar",
        type=float,
        default=DEFAULT_SETTINGS.distance_scalar,
        help="Distance-noise scalar.",
    )
    parser.add_argument(
        "--no-idle-noise",
        action="store_true",
        help="Disable idle-noise scaling.",
    )
    parser.add_argument(
        "--idle-scalar",
        type=float,
        default=DEFAULT_SETTINGS.idle_scalar,
        help="Idle-noise scalar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SETTINGS.output_path,
        help="Path to save the resulting plot.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_SETTINGS.raw_csv_path,
        help="Path to save raw per-component results.",
    )
    parser.add_argument(
        "--plot-csv",
        type=Path,
        default=DEFAULT_SETTINGS.plot_csv_path,
        help="Path to save plot-ready aggregated data for later replotting.",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Load previously saved raw CSV and replot without resimulating.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window.",
    )
    args = parser.parse_args()

    return BenchmarkSettings(
        qasm_root=args.qasm_root,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        drop_equal_packings=args.drop_equal_packings,
        alpha_reference=args.alpha_reference,
        reference_gate_count=args.reference_gate_count,
        target_selection_rate=args.target_selection_rate,
        pilot_repetitions=args.pilot_repetitions,
        target_kept_shots=args.target_kept_shots,
        min_repetitions=args.min_repetitions,
        max_repetitions=args.max_repetitions,
        stim=not args.no_stim,
        distance_noise=not args.no_distance_noise,
        distance_scalar=args.distance_scalar,
        idle_noise=not args.no_idle_noise,
        idle_scalar=args.idle_scalar,
        seed=args.seed,
        output_path=args.output,
        raw_csv_path=args.csv,
        plot_csv_path=args.plot_csv,
        from_csv=args.from_csv,
        show_plot=not args.no_show,
        jobs=args.jobs,
    )


def _load_group_circuits(
    qasm_root: Path,
    group: str,
    settings: BenchmarkSettings,
) -> Sequence[Tuple[str, qiskit.QuantumCircuit, qiskit.QuantumCircuit]]:
    timeout_seconds = settings.proxy_timeout_seconds
    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None
    resource_circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2_000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=False,
    )
    simulation_circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2_000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout_seconds,
        replace_rz_rx_with_s=True,
    )

    resource_map = {
        name: qc
        for name, qc in resource_circuits
        if settings.max_proxy_gates is None or qc.size() <= settings.max_proxy_gates
    }
    simulation_map = {
        name: qc
        for name, qc in simulation_circuits
        if settings.max_proxy_gates is None or qc.size() <= settings.max_proxy_gates
    }

    paired_names = sorted(set(resource_map) & set(simulation_map))
    paired = [
        (name, resource_map[name], simulation_map[name])
        for name in paired_names
        if not name.lower().endswith("_n2")
    ]
    return paired


def main() -> None:
    settings = parse_args()
    if settings.from_csv is not None:
        df = pd.read_csv(settings.from_csv)
        settings.plot_csv_path.parent.mkdir(parents=True, exist_ok=True)
        build_plot_dataframe(df).to_csv(settings.plot_csv_path, index=False)
        print(
            "[merging_improvement_benchmark_stylized_simulated] "
            f"Saved plot data to {settings.plot_csv_path}"
        )
        print_metric_averages(df)
        plot_results(df, settings)
        return

    df = run_experiment(settings)
    if df.empty:
        raise ValueError("every circuit failed -- no rows produced")
    settings.raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the cost model. This figure now passes a noise profile to the planner
    # (Bug B6), so its placements DO move with the cost model version.
    df = stamp_provenance(df)
    df.to_csv(settings.raw_csv_path, index=False)
    print(
        "[merging_improvement_benchmark_stylized_simulated] "
        f"Saved raw data to {settings.raw_csv_path}"
    )
    settings.plot_csv_path.parent.mkdir(parents=True, exist_ok=True)
    build_plot_dataframe(df).to_csv(settings.plot_csv_path, index=False)
    print(
        "[merging_improvement_benchmark_stylized_simulated] "
        f"Saved plot data to {settings.plot_csv_path}"
    )
    print_metric_averages(df)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
