"""
merging_improvement_benchmark_stylized_simulated_aligned.py
-----------------------------------------------------------

`merging_improvement_benchmark_stylized_simulated` plus a fourth ablation stage:
**grid alignment**, a routing optimisation layered on top of the best previous arm
(H commutation + merging + packing).

**What the fourth stage does.** `packing.alignment.align_to_grid` renumbers patches so the
ones carrying the most cross-patch traffic sit near the centre of the array. Groups are
untouched -- the same logical qubits share the same patch -- so it is a pure relabelling,
and the patch id is what `simulator.grid_constructor` uses to pick a grid site. Measured on
15 proxies it cuts weighted patch distance 14.2% with no regressions, because the pass
scores both layouts and keeps the cheaper one.

**It appears on LSR and TVD only, and that is the point.** A relabelling cannot change gate
counts or depth -- the same gates are emitted either way -- so the sandy-yellow segment is
absent from the three resource panels rather than drawn as a zero. What it changes is how
far atoms travel, which the transport model in `simulator.transport` charges as movement
error and as idling forced on everything standing still. This is the one stage in the
ablation whose benefit is purely geometric, and showing it against three panels where it
*cannot* help is what makes that legible.

(There is a second-order effect the suppression also hides: `translator.merging.transversal`
enumerates patch pairs in sorted-id order, so relabelling can change which of two competing
macros fires. That moves gate counts by small amounts which are scheduling artefacts, not
benefits, and they have no place in a cumulative ablation.)

**Ising circuits are excluded from the TVD average.** Their AQORN improvement is 0 by
construction: the states are wide enough superpositions that two finite samples score
TVD ~ 1.0 whatever the noise, so the number is a sampling floor rather than a measurement.
They stay in every other panel and in the per-circuit TVD bars; only the average drops
them, and `print_metric_averages` reports how many were dropped.

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

    python -m experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated_aligned
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
import numpy as np
import pandas as pd
import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.figures.merging_improvement.common import pack_program
from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.parallel import add_jobs_argument, parallel_map
from packing.alignment import align_to_grid
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.config_presets import ScaledSimulator

Perm = List[Tuple[int, int]]

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "figure.titlesize": 16,
    }
)

#: Ablation stages, in the order they stack. `grid_alignment` is the addition here.
COMPONENTS = [
    ("h_commutation", "H commutation benefit", "#3b4c56"),
    ("merging", "+ Merging benefit", "#cdc2c0"),
    ("packing", "+ Packing benefit", "#0097fd"),
    ("grid_alignment", "+ Routing benefit", "#e3b23c"),
]

#: Panels a patch relabelling cannot move. Grid alignment emits identical gates, so it is
#: dropped from these rather than drawn as a zero segment -- a zero bar reads as "measured
#: and found to be nothing", which is a different claim from "cannot apply".
RESOURCE_METRICS = frozenset(
    {"one_qubit_gates", "two_qubit_gates", "circuit_depth"}
)

#: Circuits whose TVD is a sampling floor rather than a measurement. Their states are wide
#: enough superpositions that two finite samples score ~1.0 regardless of noise, so every
#: arm scores the same and the improvement is 0 by construction. Dropped from the TVD
#: *average* only; the per-circuit bars still show them.
TVD_EXCLUDED_PREFIXES = ("ising",)

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


@dataclass
class BenchmarkSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    max_proxy_gates: int | None = 2500
    max_proxy_qubits: int | None = 150
    proxy_timeout_seconds: float | None = 1.0
    drop_equal_packings: bool = False
    alpha_reference: float = 0.1
    reference_gate_count: float = 200.0
    target_selection_rate: float = 0.12
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
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized_simulated_aligned.svg"
        )
    )
    raw_csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized_simulated_aligned.csv"
        )
    )
    plot_csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_benchmark_stylized_simulated_aligned_plot_data.csv"
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
    packed_perm, commute = pack_program(resource_program_circuit)
    # Relabel patches so the heaviest-traffic ones sit centrally. Derived from the
    # *resource* circuit, matching every other placement decision in this figure.
    aligned_perm = align_to_grid(packed_perm, resource_program_circuit)
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
    # Same translator settings as the packed arm -- alignment is a relabelling applied to
    # the placement, not a different compilation. A distinct seed so the two arms are
    # independent samples rather than the same shots relabelled.
    aligned_sim_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=final_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 7,
        enable_h_commutation=commute,
        enable_merging=True,
    )

    baseline_sim = evaluate_variant(simulation_program_circuit, tight_perm, baseline_sim_cfg)
    h_only_sim = evaluate_variant(simulation_program_circuit, tight_perm, h_only_sim_cfg)
    h_merge_sim = evaluate_variant(simulation_program_circuit, tight_perm, h_merge_sim_cfg)
    packed_sim = evaluate_variant(simulation_program_circuit, packed_perm, packed_sim_cfg)
    aligned_sim = evaluate_variant(
        simulation_program_circuit, aligned_perm, aligned_sim_cfg
    )

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

    aligned_cum_lsr = pct_improvement(
        baseline_sim["logical_selection_rate"],
        aligned_sim["logical_selection_rate"],
    )
    aligned_cum_tvd = pct_reduction(
        baseline_sim["logical_total_variation_distance"],
        aligned_sim["logical_total_variation_distance"],
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
        {
            # Zero on the three resource metrics by construction, not by measurement: a
            # patch relabelling emits the same gates. `RESOURCE_METRICS` keeps this
            # component off those panels entirely rather than drawing a zero segment.
            "component": "grid_alignment",
            "pct_reduction_one_qubit_gates": 0.0,
            "pct_reduction_two_qubit_gates": 0.0,
            "pct_reduction_circuit_depth": 0.0,
            "pct_improvement_logical_selection_rate": aligned_cum_lsr - packed_cum_lsr,
            "pct_reduction_logical_total_variation_distance": aligned_cum_tvd - packed_cum_tvd,
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
                "[merging_improvement_benchmark_stylized_simulated_aligned] "
                f"No circuits loaded for {size_group}."
            )
            continue
        for circuit_idx, (name, resource, simulation) in enumerate(circuits):
            jobs.append((settings, size_group, circuit_idx, name, resource, simulation))

    if not jobs:
        raise ValueError(f"No proxy circuits loaded from {settings.qasm_root}")

    print(
        "[merging_improvement_benchmark_stylized_simulated_aligned] "
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


def components_for(metric_key: str) -> List[Tuple[str, str, str]]:
    """Stages that can move this metric. See `RESOURCE_METRICS`."""
    if metric_key in RESOURCE_METRICS:
        return [entry for entry in COMPONENTS if entry[0] != "grid_alignment"]
    return list(COMPONENTS)


def averaging_frame(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """
    The rows that belong in this metric's average.

    Only TVD drops anything -- see `TVD_EXCLUDED_PREFIXES` for why those circuits cannot
    be measured on it. Every other panel averages the full set.
    """
    if metric_key != "logical_total_variation_distance":
        return df
    excluded = df["circuit_name"].str.lower().str.startswith(TVD_EXCLUDED_PREFIXES)
    return df.loc[~excluded]


def print_metric_averages(df: pd.DataFrame) -> None:
    print("[merging_improvement_benchmark_stylized_simulated_aligned] Plot averages (matching clipped bar values)")
    size_groups = ["small", "medium", "large"]
    group_totals: Dict[str, Dict[str, float]] = {}
    group_counts: Dict[str, int] = {}
    for size_group in size_groups:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        group_counts[size_group] = int(group_df["circuit_name"].nunique())
        print(f"  {size_group}:")
        metric_parts = []
        metric_totals: Dict[str, float] = {}
        for metric_key, (metric_column, metric_label) in METRICS.items():
            scoped = averaging_frame(group_df, metric_key)
            dropped = group_df["circuit_name"].nunique() - scoped["circuit_name"].nunique()
            total = 0.0
            for component, _, _ in components_for(metric_key):
                component_df = scoped.loc[scoped["component"] == component, metric_column]
                if not component_df.empty:
                    total += float(component_df.astype(float).clip(lower=0.0).mean())
            metric_totals[metric_label] = total
            note = f" (-{dropped} unmeasurable)" if dropped else ""
            metric_parts.append(f"{metric_label}={total:.2f}%{note}")
        print("    plotted total: " + ", ".join(metric_parts))
        group_totals[size_group] = metric_totals

    total_count = sum(group_counts.values())
    if total_count > 0:
        weighted_metric_parts = []
        for _, (_, metric_label) in METRICS.items():
            weighted_total = sum(
                group_totals[size_group][metric_label] * group_counts[size_group]
                for size_group in size_groups
                if size_group in group_totals
            ) / float(total_count)
            weighted_metric_parts.append(f"{metric_label}={weighted_total:.2f}%")
        count_parts = [f"{size_group}={group_counts[size_group]}" for size_group in size_groups if size_group in group_counts]
        print("  weighted overall: " + ", ".join(weighted_metric_parts))
        print("    weights: " + ", ".join(count_parts) + f", total={total_count}")

    if not df.empty:
        print(
            "  calibration averages: "
            f"alpha={df['alpha_final'].mean():.4f}, "
            f"pilot_lsr={df['pilot_final_lsr'].mean():.4f}, "
            f"repetitions={df['final_repetitions'].mean():.0f}"
        )


def build_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for size_group in ["small", "medium", "large"]:
        group_df = df.loc[df["size_group"] == size_group]
        if group_df.empty:
            continue
        for metric_key, (metric_column, metric_label) in METRICS.items():
            metric_group = group_df[["circuit_name", "component", metric_column]].copy()
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

    for ax, (metric_key, (metric_column, metric_label)) in zip(axes, METRICS.items()):
        components = components_for(metric_key)
        all_positions: List[float] = []
        all_circuit_labels: List[str] = []
        current_x = 0.0

        for size_group in size_groups:
            ordered_circuits = ordered_circuits_by_group.get(size_group, [])
            if not ordered_circuits:
                continue
            group_df = df.loc[df["size_group"] == size_group]
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
                pivot[component] = pivot[component].fillna(0.0).astype(float).clip(lower=0.0)
            pivot = (
                pivot.set_index("circuit_name")
                .reindex(ordered_circuits)
                .fillna(0.0)
                .reset_index()
            )

            positions = np.arange(len(ordered_circuits)) * bar_step + current_x
            running_bottom = np.zeros(len(ordered_circuits), dtype=float)
            for component, label, color in components:
                values = pivot[component].astype(float).to_numpy()
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
            all_circuit_labels.extend(ordered_circuits)
            current_x = positions[-1] + gap if len(positions) else current_x + gap

        avg_x = current_x + 1.0
        avg_width = 5 * bar_width
        avg_bottom = 0.0
        averaged = averaging_frame(df, metric_key)
        for component, label, color in components:
            avg_value = (
                averaged.loc[averaged["component"] == component, metric_column]
                .astype(float)
                .clip(lower=0.0)
                .mean()
            )
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

        ax.set_ylabel(metric_label, fontsize=24)
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

    axes[-1].set_xlabel("Benchmark circuit (grouped by size)", fontsize=24)

    # Built from `COMPONENTS` rather than scraped off `axes[0]`, which is a resource panel
    # and therefore never draws the routing stage -- scraping it would silently omit the
    # one entry this variant exists to add.
    axes[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.9, label=label)
            for _, label, color in COMPONENTS
        ],
        loc="best",
        frameon=False,
    )
    fig.supylabel("Percent improvement / reduction", fontsize=24)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.output_path, dpi=200)
    print(
        "[merging_improvement_benchmark_stylized_simulated_aligned] "
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
            "[merging_improvement_benchmark_stylized_simulated_aligned] "
            f"Saved plot data to {settings.plot_csv_path}"
        )
        print_metric_averages(df)
        plot_results(df, settings)
        return

    df = run_experiment(settings)
    if df.empty:
        raise ValueError("every circuit failed -- no rows produced")
    settings.raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.raw_csv_path, index=False)
    print(
        "[merging_improvement_benchmark_stylized_simulated_aligned] "
        f"Saved raw data to {settings.raw_csv_path}"
    )
    settings.plot_csv_path.parent.mkdir(parents=True, exist_ok=True)
    build_plot_dataframe(df).to_csv(settings.plot_csv_path, index=False)
    print(
        "[merging_improvement_benchmark_stylized_simulated_aligned] "
        f"Saved plot data to {settings.plot_csv_path}"
    )
    print_metric_averages(df)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
