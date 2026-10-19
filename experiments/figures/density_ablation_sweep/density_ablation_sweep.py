"""
density_ablation_sweep.py
--------------------------

Sweeps code density (fraction of codes used, from 0.5q to 1q) across all
QASMBench circuits (small / medium / large) and measures how the 4-variant
ablation (baseline → +H commutation → +merging → +packing) changes with density.

For each (circuit, n_codes) pair the 4 variants are:
  1. baseline      – no H commutation, no merging, trivial packing
  2. h_only        – H commutation, no merging, trivial packing
  3. h_merge       – H commutation, merging, trivial packing
  4. h_merge_pack  – H commutation, merging, MergeMaximizingPacker

Results are stored incrementally (small → medium → large) so a mid-run crash
does not lose prior work.  Use ``--from-csv`` to replot without resimulating.

Outputs:
  * density_ablation_sweep.csv   (raw per-component rows)
  * density_ablation_sweep.svg   (5-row grouped stacked bar figure)

Run with:

    python -m experiments.figures.density_ablation_sweep.density_ablation_sweep

``--standard`` reruns the sweep under AQORN's standard configuration
(`packing.standard.STANDARD`, "w10T"), following `ablation_standard` (Fig 6 under the
standard):

  * every translation -- gate-count metrics, calibration pilots, simulated arms -- is
    layer-sorted (`TranslatorConfig.asap_order`), the baseline included; the transversal
    in-patch CZ / XCX is on for the arms with merging on (+merge, packed), off for the others;
  * the packed arm is `plan_standard` on the raw resource program (n_code 4, the same
    `noise_profile_for` profile the submitted packed arm gets), planned once per circuit.
    It does not follow the density -- it packs into ceil(n/2) patches at every point, so
    below packing density 2 "+ Packing" also carries the patch-count difference -- and its
    patch count is recorded (`packed_patches`); the plan's commutation decision is honoured
    on the +H, +merge and packed arms;
  * alpha and the shot count are recalibrated on the layer-sorted baseline with the same
    pilots and pilot seeds, and every arm keeps the seed the submitted run gives it;
  * every simulated arm also reports the silent error per kept shot from a 100k-shot
    Pauli-frame run on the noisy Stim circuit the simulation converted (TVD stand-in at any
    width): per-arm `<arm>_silent_per_kept` / `_p_silent` / `_frame_lsr`, and the stacked
    `pct_reduction_silent_per_kept` per component, beside per-arm gate counts, depth, LSR, TVD.

It never replaces anything: the CSV is `density_ablation_sweep_standard.csv` beside this
script (`_v2`, `_v3`, ... when that stem is taken by a CSV or a figure), created by the first
size group and appended to by the others, and the figure takes the CSV's stem through
`save_figure(..., versioned=True)`. `--names` runs a subset (each circuit keeps its group
index, hence its seeds); `--output-dir` puts the figure there and nowhere else.

    python -m experiments.figures.density_ablation_sweep.density_ablation_sweep --standard
    ... --from-csv experiments/figures/density_ablation_sweep/density_ablation_sweep_standard.csv

``--standard --matched-patches`` makes the packed arm follow the density, so every arm at a
point has the same patch count and "+ Packing" is placement alone:

  * at each density point the packed arm is `plan_standard(..., patches=n_codes)`, which holds
    every planner stage at exactly that count and verifies against the naive packing at it
    (the trivial arm's own packing, `tight_trivial_packing`); at the tightest point that is
    the per-circuit plan above, unchanged;
  * the commutation decision is the per-circuit plan's at every point, so the baseline, +H
    and +merge arms are exactly the ``--standard`` ones and only the packed arm moves;
  * `packed_patches == n_codes` on every row; `plan_fell_back` and `placement` are the plan
    used at that row's density, and `matched_patches` marks the rows.

Its CSV is `density_ablation_sweep_matched.csv` (versioned the same way). Since only the packed
arm differs from ``--standard``, ``--reuse-arms <standard CSV>`` copies the baseline, +H and
+merge arms -- and the calibration -- from that run instead of re-simulating them, after checking
the circuit, its index, its commutation decision and its tightest plan agree; the packed arm keeps
the seed a full run gives it. About a quarter of the simulation of a full run:

    python -m experiments.figures.density_ablation_sweep.density_ablation_sweep --standard \\
        --matched-patches --reuse-arms \\
        experiments/figures/density_ablation_sweep/density_ablation_sweep_standard.csv

``--split-merging`` (with ``--matched-patches --reuse-arms <matched CSV>``) answers what merging
does as patches empty, on a baseline that has none of it. The Hadamard handler turns a solo
qubit's H into a whole-patch HH in every arm, merging on or off, so at one qubit per patch the
``--standard`` baseline already had single-qubit merging (33% fewer 1q gates than without it,
on the small group). Here the baseline and +H arms translate with
`TranslatorConfig.solo_hh_promotion=False`, and merging comes back in two arms: `h_merge_1q`
(HH / XX / ZZ macros and the solo promotion; `merge_two_qubit=False`, no transversal in-patch
CZ / XCX) and the existing +merge arm (everything). Components h_commutation, merging_1q,
merging_2q, packing; arms SPLIT_ARMS. The +merge and packed arms and the calibration are copied
from the matched CSV (they do not change), so three arms are simulated. `--groups` limits the
run to size groups. The CSV is `density_ablation_sweep_split.csv`; `density_references.py`
draws it (`density_ablation_split`), this script does not. Without ``--reuse-arms`` the split
run simulates every arm itself -- the only way under the logical optimisation pass, whose
programs no earlier CSV has: rows record the pass (`logical_optimization`), and ``--reuse-arms``
refuses a CSV made under another setting.

``--share-identical`` (2026-10-10): an arm that compiles to exactly the circuit an earlier arm at
the same density point runs -- the same physical circuit, placement, and lane each logical state
is read from (`simulated_circuit_key`) -- takes that arm's simulated metrics instead of a second
simulation with its own seed, so a step that changes nothing contributes exactly zero rather
than seed noise. At one qubit per patch two-qubit merging leaves 74 of 76 points' circuits as
they were and the packed arm is the +merge arm by construction; elsewhere a third to two-thirds
of the points repeat the step before. Rows name the shared arms in `shared_arms`
("packed=h_merge;..."), and `density_references.py`'s recount reproduces them from their source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import (
    COLUMN_WIDTH,
    FULL_WIDTH,
    output_dir as figure_dir,
    save_figure,
    stacked_percent,
    stamp_provenance,
    use_style,
)
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import SimSettings, noise_profile_for
from experiments.figures.merging_improvement.common import pack_program
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.config_presets import ScaledSimulator
from translator.passes import cancel_inverse_gates, commute_hadamards

Perm = List[Tuple[int, int]]

# Style comes from `experiments.shared.figures.use_style` at plot time, not from an
# rcParams block here. This script predates the shared style and carried Helvetica at
# 16pt with fontsize=24 axis labels, which is a different figure from every other one
# in the paper.

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS = {
    "one_qubit_gates": ("pct_reduction_one_qubit_gates", "#1q gates"),
    "two_qubit_gates": ("pct_reduction_two_qubit_gates", "#2q gates"),
    "circuit_depth":   ("pct_reduction_circuit_depth", "depth"),
    "lsr":             ("pct_improvement_logical_selection_rate", "LSR"),
    "tvd":             ("pct_reduction_logical_total_variation_distance", "TVD"),
}

COMPONENTS = [
    ("h_commutation", "H commutation", "#3b4c56"),
    ("merging",       "+ Merging",     "#cdc2c0"),
    ("packing",       "+ Packing",     "#0097fd"),
]

#: The 2x2 panel layout. Each entry is `(metric columns, label)`; more than one column
#: means average them element-wise.
#:
#: 1q and 2q gates share a panel because they are the same statement twice -- both are "how
#: much smaller is the translated circuit" and their curves track each other at every
#: density. Averaging them frees the fourth slot, which is what lets five stacked rows
#: become a 2x2 at half the height. Averaging is valid on the boundaries directly: they are
#: cumulative percentages, so the mean of two boundary rows has segments equal to the mean
#: of the segments and a top equal to the mean of the totals.
PANELS: Sequence[Tuple[Tuple[str, ...], str]] = (
    (("pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates"), "#gates (1q, 2q avg)"),
    (("pct_reduction_circuit_depth",), "depth"),
    (("pct_improvement_logical_selection_rate",), "LSR"),
    (("pct_reduction_logical_total_variation_distance",), "TVD"),
)


def _panel_boundaries(agg, size_group: str, bin_center: float, metric_cols):
    """Stacked boundaries for one panel: the mean over its metric columns, or None."""
    rows = [
        agg.get((size_group, bin_center, column))
        for column in metric_cols
    ]
    rows = [row for row in rows if row is not None]
    if not rows:
        return None
    return np.mean(np.asarray(rows, dtype=float), axis=0)


SIZE_GROUPS = ["small", "medium", "large"]
SIZE_GROUP_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class DensityAblationSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    max_proxy_gates: int | None = 2500
    max_proxy_qubits: int | None = 150
    proxy_timeout_seconds: float | None = 1.0
    n_codes_samples: int = 6
    n_plot_bins: int = 6
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
            "experiments/figures/density_ablation_sweep/density_ablation_sweep.svg"
        )
    )
    csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/density_ablation_sweep/density_ablation_sweep.csv"
        )
    )
    from_csv: Path | None = None
    show_plot: bool = False
    #: Worker processes for the per-circuit sweep. None takes `parallel.DEFAULT_JOBS`.
    jobs: int | None = None
    #: The standard configuration (`_run_one_circuit_standard`) instead of the submitted one.
    standard: bool = False
    #: Pauli-frame shots per simulated arm for the silent error per kept shot (standard only).
    frame_shots: int = 100_000
    #: Only these circuits (standard only). Each keeps its index in its size group, and with
    #: it the seeds a full run gives it.
    names: Tuple[str, ...] = ()
    #: Where the standard figure goes instead of results/figures/; nothing is written beside
    #: the script then.
    output_dir: Path | None = None
    #: Standard only: plan the packed arm at each density point's patch count.
    matched_patches: bool = False
    #: Matched only: a standard CSV whose baseline, +H and +merge arms (and calibration) to
    #: copy instead of re-simulating. With `split_merging`, a matched CSV whose +merge and
    #: packed arms to copy.
    reuse_arms: Path | None = None
    #: Matched only: split merging into single- and two-qubit steps on a merging-free baseline
    #: (`SPLIT_ARMS`).
    split_merging: bool = False
    #: Standard only: run just these size groups (each circuit keeps its index and seeds).
    groups: Tuple[str, ...] = ()
    #: With `reuse_arms`: the arms to simulate, every other arm copied. Empty is the mode's
    #: default: the packed arm (matched), or baseline, +H and +1q merging (split).
    simulate: Tuple[str, ...] = ()
    #: Matched only: plan the packed arm with a solo patch's Hadamards priced as the HH the
    #: translator emits (`solo_hh`), at every density point. The standard has done this by
    #: default since 2026-10-09, so this only matters with `legacy_solo_pricing`.
    solo_aware_planner: bool = False
    #: Plan with the solo pricing every run before 2026-10-09 used
    #: (`STANDARD.with_(solo_hh_aware=False)`), to reproduce those CSVs.
    legacy_solo_pricing: bool = False
    #: Standard only: an arm whose simulated circuit is an earlier arm's at the same density
    #: point (`simulated_circuit_key`) takes that arm's metrics instead of a second simulation
    #: with its own seed; rows name such arms in `shared_arms`.
    share_identical: bool = False


DEFAULT_SETTINGS = DensityAblationSettings()

#: The standard run's CSV: beside this script whatever the working directory, never replaced.
STANDARD_CSV = Path(__file__).resolve().parent / "density_ablation_sweep_standard.csv"
#: The matched-patches run's CSV, likewise.
MATCHED_CSV = Path(__file__).resolve().parent / "density_ablation_sweep_matched.csv"
#: The split-merging run's CSV, likewise.
SPLIT_CSV = Path(__file__).resolve().parent / "density_ablation_sweep_split.csv"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def pct_improvement(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (challenger - reference) / reference


def tight_trivial_packing(num_qubits: int, n_codes: int) -> Perm:
    """Produce a permutation with exactly n_codes patches.

    Pairs the first (num_qubits - n_codes) consecutive qubit pairs together,
    then assigns remaining qubits to solo patches.
    """
    if n_codes < 1:
        raise ValueError("n_codes must be >= 1")
    min_codes = math.ceil(num_qubits / 2)
    if n_codes < min_codes or n_codes > num_qubits:
        raise ValueError(
            f"n_codes={n_codes} out of range [{min_codes}, {num_qubits}] "
            f"for {num_qubits} qubits."
        )
    pairs_needed = num_qubits - n_codes
    perm: Perm = [None] * num_qubits  # type: ignore[list-item]
    patch = 0
    qubit = 0
    for _ in range(pairs_needed):
        perm[qubit]     = (patch, 0)
        perm[qubit + 1] = (patch, 1)
        qubit += 2
        patch += 1
    while qubit < num_qubits:
        perm[qubit] = (patch, 0)
        qubit += 1
        patch += 1
    return perm


def ensure_measurements(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    if circuit.num_clbits > 0 and any(op.operation.name == "measure" for op in circuit.data):
        return circuit
    measured = copy.deepcopy(circuit)
    measured.measure_all()
    return measured


def preprocess_for_packing(
    program_circuit: qiskit.QuantumCircuit,
    config: PipelineConfig,
) -> qiskit.QuantumCircuit:
    processed = cancel_inverse_gates(program_circuit)
    if config.translator_config.enable_h_commutation:
        processed = commute_hadamards(processed)
        processed = cancel_inverse_gates(processed)
    return processed


# ---------------------------------------------------------------------------
# Pipeline config builders
# ---------------------------------------------------------------------------

def build_pipeline_config(
    settings: DensityAblationSettings,
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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_translation_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Tuple[int, int, int]:
    translated = translate_permutation(program_circuit, permutation, config=config)
    circuit = translated.circuit
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    return one_q, two_q, depth


def evaluate_variant(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Dict[str, float]:
    record = run_specific_scan(
        program_circuit=program_circuit,
        permutations=[permutation],
        config=config,
    )
    if record.dataframe is None or record.dataframe.df.empty:
        raise RuntimeError("run_specific_scan returned no dataframe.")
    row = record.dataframe.df.iloc[0]
    return {
        "logical_selection_rate": float(row["logical_selection_rate"]),
        "logical_total_variation_distance": float(row["logical_total_variation_distance"]),
    }


def evaluate_variant_frames(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
    *,
    frames: int,
    frame_seed: int,
) -> Dict[str, float]:
    """
    `evaluate_variant`, plus the silent logical error per kept shot.

    Measured as `experiments.shared.simulation.evaluate(frames=...)` measures it: a
    Pauli-frame run of `frames` shots on the noisy Stim circuit this very simulation
    converted (`capture_noisy_stim`), so it costs no second conversion. It is the accuracy
    measure that still works where TVD saturates (above ~18 qubits). NaN when the backend
    is not Stim. Wraps this script's `evaluate_variant` rather than calling `evaluate`, so a
    failed simulation raises here exactly as it does on the submitted path.
    """
    from experiments.revision.exp8_undetected.frames import capture_noisy_stim, split_from_stim

    with capture_noisy_stim() as captured:
        out = evaluate_variant(program_circuit, permutation, config)
    out.update(silent_per_kept=float("nan"), p_silent=float("nan"), frame_lsr=float("nan"))
    if captured:
        encoded, noisy = captured[-1]
        split = split_from_stim(encoded, noisy, shots=frames, seed=frame_seed)
        out.update(silent_per_kept=split["p_silent_given_kept"], p_silent=split["p_silent"],
                   frame_lsr=1.0 - split["p_detected"])
    return out


def simulated_circuit_key(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> str:
    """
    Everything `evaluate_variant_frames` simulates and reads for one arm, as a digest: the
    physical circuit `run_specific_scan` translates (`Translator.translate_batch`, the same
    call), the placement, and the (patch, lane) each logical state ends on, which post-selection
    decodes by (`reduce_distribution` reads `context.state_positions`). Arms of one density
    point share alpha, shots and noise, so two arms with one key differ only in their seeds.
    """
    from translator.core import Translator

    translated = Translator(config=config.translator_config).translate_batch(
        program_circuit, [permutation])[0]
    circuit, context = translated.circuit, translated.context
    digest = hashlib.sha256(repr((
        circuit.num_qubits, circuit.num_clbits, context.n_code, context.k_log,
        [tuple(p) for p in permutation], [tuple(p) for p in context.state_positions],
    )).encode())
    for inst in circuit.data:
        digest.update(repr((
            inst.operation.name,
            tuple(circuit.find_bit(q).index for q in inst.qubits),
            tuple(circuit.find_bit(c).index for c in inst.clbits),
            tuple(repr(p) for p in inst.operation.params),
        )).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Alpha calibration helpers
# ---------------------------------------------------------------------------

def calibrated_alpha_from_gate_count(gate_count: int, settings: DensityAblationSettings) -> float:
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
    settings: DensityAblationSettings,
) -> int:
    denom = max(selection_rate, 1e-4)
    reps = int(math.ceil(settings.target_kept_shots / denom))
    return int(np.clip(reps, settings.min_repetitions, settings.max_repetitions))


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

def _load_group_circuits(
    qasm_root: Path,
    group: str,
    settings: DensityAblationSettings,
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
    return [
        (name, resource_map[name], simulation_map[name])
        for name in paired_names
        if not name.lower().endswith("_n2")
    ]


# ---------------------------------------------------------------------------
# Density bins
# ---------------------------------------------------------------------------

def _density_bins(n_qubits: int, n_codes_samples: int) -> List[int]:
    """Return deduplicated n_codes values evenly spaced from min to max."""
    min_codes = math.ceil(n_qubits / 2)
    max_codes = n_qubits
    raw = np.linspace(min_codes, max_codes, n_codes_samples)
    return sorted(set(int(round(v)) for v in raw))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def _sim_settings(settings: "DensityAblationSettings") -> SimSettings:
    """The `SimSettings` matching this script's own noise flags, for `noise_profile_for`."""
    return SimSettings(
        stim=getattr(settings, "stim", True),
        distance_noise=getattr(settings, "distance_noise", True),
        distance_scalar=getattr(settings, "distance_scalar", 1.0),
        idle_noise=getattr(settings, "idle_noise", True),
        idle_time_scalar=getattr(settings, "idle_scalar", 1.0),
        shot_progress=False,
    )


#: One work item: everything `_run_one_circuit` needs, and nothing that cannot be pickled.
#: `parallel_map` spawns rather than forks, so each worker re-imports this module and
#: receives its item over a pipe -- qiskit circuits and the settings dataclass both survive
#: that, but an open `PipelineConfig` does not, which is why the three translate configs are
#: rebuilt inside the worker rather than passed in.
CircuitJob = Tuple[
    "DensityAblationSettings", str, int, str, qiskit.QuantumCircuit, qiskit.QuantumCircuit
]


def _run_one_circuit(job: CircuitJob) -> List[Dict]:
    """
    Every density point for one circuit. Module scope, for `spawn`.

    Was the body of `run_experiment`'s per-circuit loop. Nothing about the computation
    changed -- the seeds still derive from `circuit_idx`, so a parallel run reproduces the
    serial one exactly -- but it had to become a top-level function before `parallel_map`
    could reach it. This is the last figure in the regeneration queue and the only one that
    was still single-threaded.
    """
    settings, size_group, circuit_idx, circuit_name, resource_circuit, simulation_circuit = job

    baseline_translate_cfg = PipelineConfig()
    baseline_translate_cfg.translator_config.enable_merging = False
    baseline_translate_cfg.translator_config.enable_h_commutation = False

    h_only_translate_cfg = PipelineConfig()
    h_only_translate_cfg.translator_config.enable_merging = False
    h_only_translate_cfg.translator_config.enable_h_commutation = True

    h_merge_translate_cfg = PipelineConfig()
    h_merge_translate_cfg.translator_config.enable_merging = True
    h_merge_translate_cfg.translator_config.enable_h_commutation = True

    rows: List[Dict] = []

    resource_program = ensure_measurements(resource_circuit)
    simulation_program = ensure_measurements(simulation_circuit)
    n_qubits = resource_program.num_qubits
    min_codes = math.ceil(n_qubits / 2)
    code_counts = _density_bins(n_qubits, settings.n_codes_samples)

    # Preprocess once for packing
    preprocessed = preprocess_for_packing(resource_program, h_merge_translate_cfg)

    # Alpha calibration: baseline at min_codes
    tight_min = tight_trivial_packing(n_qubits, min_codes)
    base_1q, base_2q, _ = compute_translation_metrics(
        resource_program, tight_min, baseline_translate_cfg
    )
    baseline_gate_count = max(1, base_1q + base_2q)
    alpha_initial = calibrated_alpha_from_gate_count(baseline_gate_count, settings)

    pilot1_cfg = build_pipeline_config(
        settings,
        alpha=alpha_initial,
        repetitions=settings.pilot_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 1,
        enable_h_commutation=False,
        enable_merging=False,
    )
    pilot1 = evaluate_variant(simulation_program, tight_min, pilot1_cfg)

    alpha_final = corrected_alpha(
        alpha_initial,
        pilot1["logical_selection_rate"],
        settings.target_selection_rate,
    )

    pilot2_cfg = build_pipeline_config(
        settings,
        alpha=alpha_final,
        repetitions=settings.pilot_repetitions,
        seed=settings.seed * 1_000_003 + circuit_idx * 17 + 2,
        enable_h_commutation=False,
        enable_merging=False,
    )
    pilot2 = evaluate_variant(simulation_program, tight_min, pilot2_cfg)
    final_repetitions = repetitions_from_selection_rate(
        pilot2["logical_selection_rate"], settings
    )

    # No tqdm here. Every worker inherits the parent's stderr, so N bars serialise their
    # writes onto one pipe -- measured on `improvement_vs_noise_and_k` at 148 KB of log per
    # item with bars on against 15 KB with them off, and the items completed faster because
    # the workers had been blocked on writes rather than computing. `parallel_map` prints
    # one aggregate progress line for the whole group instead.
    for density_bin_idx, n_codes in enumerate(code_counts):
        fraction = n_codes / n_qubits

        trivial_perm = tight_trivial_packing(n_qubits, n_codes)
        # `MergeMaximizingPacker` is archived; this is the shipped planner. It is
        # handed a noise profile because the simulator below switches distance and idle
        # noise on -- Bug B6, the same mismatch `merging_improvement_benchmark_stylized_simulated`
        # had. `n_code` follows the density point: `n_codes` patches over `n_qubits`
        # logical qubits means ceil(n_qubits / n_codes) per patch.
        per_patch = max(1, math.ceil(n_qubits / n_codes))
        try:
            packed_perm, _ = pack_program(
                preprocessed, per_patch + 2,
                noise=noise_profile_for(_sim_settings(settings)),
            )
        except Exception:  # noqa: BLE001 - a refused plan falls back to trivial
            packed_perm = trivial_perm

        # Translation metrics
        b_1q, b_2q, b_depth = compute_translation_metrics(
            resource_program, trivial_perm, baseline_translate_cfg
        )
        h_1q, h_2q, h_depth = compute_translation_metrics(
            resource_program, trivial_perm, h_only_translate_cfg
        )
        hm_1q, hm_2q, hm_depth = compute_translation_metrics(
            resource_program, trivial_perm, h_merge_translate_cfg
        )
        pk_1q, pk_2q, pk_depth = compute_translation_metrics(
            resource_program, packed_perm, h_merge_translate_cfg
        )

        # Simulation metrics
        seed_off = circuit_idx * 100_000 + density_bin_idx * 10
        base_sim_cfg = build_pipeline_config(
            settings,
            alpha=alpha_final,
            repetitions=final_repetitions,
            seed=settings.seed * 1_000_003 + seed_off + 3,
            enable_h_commutation=False,
            enable_merging=False,
        )
        h_sim_cfg = build_pipeline_config(
            settings,
            alpha=alpha_final,
            repetitions=final_repetitions,
            seed=settings.seed * 1_000_003 + seed_off + 4,
            enable_h_commutation=True,
            enable_merging=False,
        )
        hm_sim_cfg = build_pipeline_config(
            settings,
            alpha=alpha_final,
            repetitions=final_repetitions,
            seed=settings.seed * 1_000_003 + seed_off + 5,
            enable_h_commutation=True,
            enable_merging=True,
        )
        pk_sim_cfg = build_pipeline_config(
            settings,
            alpha=alpha_final,
            repetitions=final_repetitions,
            seed=settings.seed * 1_000_003 + seed_off + 6,
            enable_h_commutation=True,
            enable_merging=True,
        )

        base_sim = evaluate_variant(simulation_program, trivial_perm, base_sim_cfg)
        h_sim    = evaluate_variant(simulation_program, trivial_perm, h_sim_cfg)
        hm_sim   = evaluate_variant(simulation_program, trivial_perm, hm_sim_cfg)
        pk_sim   = evaluate_variant(simulation_program, packed_perm,  pk_sim_cfg)

        # Cumulative from baseline
        h_cum_1q   = pct_reduction(b_1q,   h_1q)
        hm_cum_1q  = pct_reduction(b_1q,   hm_1q)
        pk_cum_1q  = pct_reduction(b_1q,   pk_1q)

        h_cum_2q   = pct_reduction(b_2q,   h_2q)
        hm_cum_2q  = pct_reduction(b_2q,   hm_2q)
        pk_cum_2q  = pct_reduction(b_2q,   pk_2q)

        h_cum_dep  = pct_reduction(b_depth, h_depth)
        hm_cum_dep = pct_reduction(b_depth, hm_depth)
        pk_cum_dep = pct_reduction(b_depth, pk_depth)

        h_cum_lsr  = pct_improvement(base_sim["logical_selection_rate"],
                                     h_sim["logical_selection_rate"])
        hm_cum_lsr = pct_improvement(base_sim["logical_selection_rate"],
                                     hm_sim["logical_selection_rate"])
        pk_cum_lsr = pct_improvement(base_sim["logical_selection_rate"],
                                     pk_sim["logical_selection_rate"])

        h_cum_tvd  = pct_reduction(base_sim["logical_total_variation_distance"],
                                   h_sim["logical_total_variation_distance"])
        hm_cum_tvd = pct_reduction(base_sim["logical_total_variation_distance"],
                                   hm_sim["logical_total_variation_distance"])
        pk_cum_tvd = pct_reduction(base_sim["logical_total_variation_distance"],
                                   pk_sim["logical_total_variation_distance"])

        component_payloads = [
            {
                "component": "h_commutation",
                "pct_reduction_one_qubit_gates": h_cum_1q,
                "pct_reduction_two_qubit_gates": h_cum_2q,
                "pct_reduction_circuit_depth": h_cum_dep,
                "pct_improvement_logical_selection_rate": h_cum_lsr,
                "pct_reduction_logical_total_variation_distance": h_cum_tvd,
            },
            {
                "component": "merging",
                "pct_reduction_one_qubit_gates": hm_cum_1q - h_cum_1q,
                "pct_reduction_two_qubit_gates": hm_cum_2q - h_cum_2q,
                "pct_reduction_circuit_depth":   hm_cum_dep - h_cum_dep,
                "pct_improvement_logical_selection_rate": hm_cum_lsr - h_cum_lsr,
                "pct_reduction_logical_total_variation_distance": hm_cum_tvd - h_cum_tvd,
            },
            {
                "component": "packing",
                "pct_reduction_one_qubit_gates": pk_cum_1q - hm_cum_1q,
                "pct_reduction_two_qubit_gates": pk_cum_2q - hm_cum_2q,
                "pct_reduction_circuit_depth":   pk_cum_dep - hm_cum_dep,
                "pct_improvement_logical_selection_rate": pk_cum_lsr - hm_cum_lsr,
                "pct_reduction_logical_total_variation_distance": pk_cum_tvd - hm_cum_tvd,
            },
        ]

        for payload in component_payloads:
            rows.append(
                {
                    "size_group": size_group,
                    "circuit_name": circuit_name,
                    "num_qubits": n_qubits,
                    "n_codes": n_codes,
                    "fraction_codes_used": fraction,
                    "alpha_final": alpha_final,
                    "final_repetitions": final_repetitions,
                    **payload,
                }
            )

    return rows


#: The standard path's arms, bottom of the stack upwards, and the arm each component's
#: cumulative value ends at.
STANDARD_ARMS = ("baseline", "h_only", "h_merge", "packed")
_COMPONENT_ARM = (("h_commutation", "h_only"), ("merging", "h_merge"), ("packing", "packed"))
#: `--split-merging`: merging in two steps, single-qubit macros then two-qubit ones, on a
#: baseline (and +H arm) with no merging at all -- not even a solo qubit's H promoted to HH.
SPLIT_ARMS = ("baseline", "h_only", "h_merge_1q", "h_merge", "packed")
_COMPONENT_ARM_SPLIT = (("h_commutation", "h_only"), ("merging_1q", "h_merge_1q"),
                        ("merging_2q", "h_merge"), ("packing", "packed"))
#: What the standard CSV records for every arm, as `<arm>_<quantity>` columns.
_ARM_QUANTITIES = ("one_qubit_gates", "two_qubit_gates", "depth", "lsr", "tvd",
                   "silent_per_kept", "p_silent", "frame_lsr")


def _standard_cumulative(statics, sims, arm: str) -> Dict[str, float]:
    """`arm`'s cumulative improvement over the baseline arm, in the CSV's metric columns."""
    (b_1q, b_2q, b_depth), (one_q, two_q, depth) = statics["baseline"], statics[arm]
    base, sim = sims["baseline"], sims[arm]
    return {
        "pct_reduction_one_qubit_gates": pct_reduction(b_1q, one_q),
        "pct_reduction_two_qubit_gates": pct_reduction(b_2q, two_q),
        "pct_reduction_circuit_depth": pct_reduction(b_depth, depth),
        "pct_improvement_logical_selection_rate": pct_improvement(
            base["logical_selection_rate"], sim["logical_selection_rate"]),
        "pct_reduction_logical_total_variation_distance": pct_reduction(
            base["logical_total_variation_distance"], sim["logical_total_variation_distance"]),
        "pct_reduction_silent_per_kept": pct_reduction(
            base["silent_per_kept"], sim["silent_per_kept"]),
    }


def _run_one_circuit_standard(job: CircuitJob) -> List[Dict]:
    """
    `_run_one_circuit` under the standard configuration. Module scope, for `spawn`.

    Same circuits, density points, calibration procedure and seed derivation; what changes
    is what `ablation_standard` changes for Fig 6 (see the module docstring), plus one thing
    particular to a density sweep: the packed arm is one `plan_standard` plan per circuit,
    so it sits at the plan's patch count (`packed_patches`) at every density point -- unless
    `settings.matched_patches`, which plans it at each point's own patch count instead, and
    `settings.reuse_arms`, which copies the other arms from the run that already has them.

    Kept apart from `_run_one_circuit` rather than threaded through it with flags, so the
    submitted configuration's code -- and the figure it produced -- stays exactly as it was.
    """
    from packing.standard import STANDARD, plan_standard
    from translator.logical_optimization import state as logical_optimization_state

    settings, size_group, circuit_idx, circuit_name, resource_circuit, simulation_circuit = job
    n_code = 4
    #: The logical optimisation pass this run's programs went through ("off", or its version).
    logical_optimization = logical_optimization_state()

    resource_program = ensure_measurements(resource_circuit)
    simulation_program = ensure_measurements(simulation_circuit)
    n_qubits = resource_program.num_qubits
    min_codes = math.ceil(n_qubits / 2)
    code_counts = _density_bins(n_qubits, settings.n_codes_samples)

    # The raw program, not `preprocess_for_packing`'s output: the planner runs cancel-inverse
    # and H commutation itself (`translated_view`), and running them first prices a circuit
    # the translator never emits. No fallback to the trivial packing either: every circuit
    # here can be planned at n_code 4, so a planner error drops the circuit, loudly, rather
    # than scoring a trivial placement as "+ Packing".
    planner_noise = noise_profile_for(_sim_settings(settings))
    #: The planner's standard: the live one, or the solo pricing every run before 2026-10-09 used.
    planner_standard = (STANDARD.with_(solo_hh_aware=False) if settings.legacy_solo_pricing
                        else STANDARD)
    plan = plan_standard(
        resource_program, n_code,
        noise=planner_noise,
        h_commutation="auto", verify=True, standard=planner_standard,
    )
    placement: Perm = [(int(patch), int(lane)) for patch, lane in plan.placement]
    commute = bool(plan.h_commutation)
    reused = (_reused_rows(settings.reuse_arms, circuit_name, code_counts)
              if settings.reuse_arms is not None else None)
    #: Whether the packed arm's planner prices solo patches (its own standard, or the flag).
    solo_pricing = settings.solo_aware_planner or planner_standard.solo_hh_aware
    if reused is not None:
        # The same circuit at the same index (hence seeds) is required to reuse anything --
        # even the calibration. Each arm is then copied only if this run would produce it
        # again: arms that read the commutation decision need the same decision, and the
        # packed arm the same tightest plan from a planner that prices solo patches alike.
        first = reused[code_counts[0]]
        identity = {"circuit_idx": circuit_idx, "num_qubits": n_qubits}
        differs = {key: (first[key], value) for key, value in identity.items()
                   if first[key] != value}
        # Another logical optimisation setting simulated other programs (older CSVs: off).
        made_under = first.get("logical_optimization")
        made_under = made_under if isinstance(made_under, str) else "off"
        if made_under != logical_optimization:
            differs["logical_optimization"] = (made_under, logical_optimization)
        if differs:
            raise RuntimeError(f"{circuit_name}: {settings.reuse_arms.name} disagrees with "
                               f"this run on {differs}; nothing in it can be reused")
        commute_agrees = bool(first["commute"]) == commute
        packed_agrees = (first["placement"] == json.dumps([list(p) for p in placement])
                         and bool(first.get("solo_aware_planner", False)) == solo_pricing
                         and bool(first.get("matched_patches", False)))

    split = settings.split_merging
    #: arm -> (H commutation, merging, seed offset). The offsets are `_run_one_circuit`'s; the
    #: split run's single-qubit merging arm takes the next free one.
    arms = {
        "baseline": (False, False, 3),
        "h_only": (commute, False, 4),
        "h_merge": (commute, True, 5),
        "packed": (commute, True, 6),
    }
    if split:
        arms = {"baseline": arms["baseline"], "h_only": arms["h_only"],
                "h_merge_1q": (commute, True, 7), "h_merge": arms["h_merge"],
                "packed": arms["packed"]}
    arm_names = SPLIT_ARMS if split else STANDARD_ARMS
    naive_arms = tuple(a for a in arm_names if a != "packed")
    #: Arms copied from `settings.reuse_arms` rather than simulated.
    simulated = settings.simulate or (("baseline", "h_only", "h_merge_1q") if split
                                      else ("packed",))
    copied = (tuple(a for a in arm_names if a not in simulated)
              if settings.reuse_arms is not None else ())
    if reused is not None:
        absent = [a for a in copied if f"{a}_lsr" not in reused[code_counts[0]]]
        if absent:
            raise RuntimeError(f"{settings.reuse_arms.name} has no {absent} arm to copy")
        # An arm this run would not reproduce is simulated instead of copied.
        copied = tuple(a for a in copied
                       if (a == "baseline" or commute_agrees)
                       and (a != "packed" or packed_agrees))

    def configured(config: PipelineConfig, arm: str) -> PipelineConfig:
        h_commutation, merging, _ = arms[arm]
        single_qubit_only = split and arm == "h_merge_1q"
        translator = config.translator_config
        translator.n_code = n_code
        translator.enable_h_commutation = h_commutation
        translator.enable_merging = merging
        translator.asap_order = STANDARD.asap_order
        # The transversal in-patch CZ / XCX is two-qubit merging's: off on the 1q-only arm.
        translator.transversal_intra_two_qubit = (
            STANDARD.transversal_intra_two_qubit and merging and not single_qubit_only
        )
        if split:
            # Merging off means all of it, including a solo qubit's H as a whole-patch HH.
            translator.solo_hh_promotion = merging
            translator.merge_two_qubit = not single_qubit_only
        return config

    def static(arm: str, permutation: Perm) -> Tuple[int, int, int]:
        return compute_translation_metrics(
            resource_program, permutation, configured(PipelineConfig(), arm)
        )

    def simulation_config(arm: str, *, alpha: float, repetitions: int, seed: int):
        h_commutation, merging, _ = arms[arm]
        return configured(build_pipeline_config(
            settings, alpha=alpha, repetitions=repetitions, seed=seed,
            enable_h_commutation=h_commutation, enable_merging=merging,
        ), arm)

    if reused is not None:
        # The calibration is the reused run's: its pilots are seeded per circuit, so a rerun
        # would only re-derive the same alpha and shot count.
        calibration = {key: first[key] for key in (
            "alpha_final", "final_repetitions", "baseline_gate_count", "alpha_initial",
            "pilot_initial_lsr", "pilot_final_lsr")}
        alpha_final = float(calibration["alpha_final"])
        final_repetitions = int(calibration["final_repetitions"])
        baseline_gate_count = int(calibration["baseline_gate_count"])
        alpha_initial = float(calibration["alpha_initial"])
        pilot_initial_lsr = float(calibration["pilot_initial_lsr"])
        pilot_final_lsr = float(calibration["pilot_final_lsr"])
    else:
        # Calibration exactly as `_run_one_circuit` does it -- baseline arm at min_codes, the
        # same two pilots and pilot seeds -- but on the layer-sorted baseline, so alpha is
        # re-solved for the baseline the arms are now measured against.
        tight_min = tight_trivial_packing(n_qubits, min_codes)
        base_1q, base_2q, _ = static("baseline", tight_min)
        baseline_gate_count = max(1, base_1q + base_2q)
        alpha_initial = calibrated_alpha_from_gate_count(baseline_gate_count, settings)
        pilot1 = evaluate_variant(simulation_program, tight_min, simulation_config(
            "baseline", alpha=alpha_initial, repetitions=settings.pilot_repetitions,
            seed=settings.seed * 1_000_003 + circuit_idx * 17 + 1,
        ))
        alpha_final = corrected_alpha(
            alpha_initial, pilot1["logical_selection_rate"], settings.target_selection_rate,
        )
        pilot2 = evaluate_variant(simulation_program, tight_min, simulation_config(
            "baseline", alpha=alpha_final, repetitions=settings.pilot_repetitions,
            seed=settings.seed * 1_000_003 + circuit_idx * 17 + 2,
        ))
        final_repetitions = repetitions_from_selection_rate(
            pilot2["logical_selection_rate"], settings
        )
        pilot_initial_lsr = pilot1["logical_selection_rate"]
        pilot_final_lsr = pilot2["logical_selection_rate"]

    per_circuit = {
        "size_group": size_group,
        "circuit_name": circuit_name,
        "num_qubits": n_qubits,
        "alpha_final": alpha_final,
        "final_repetitions": final_repetitions,
        "configuration": "standard",
        "asap_order": STANDARD.asap_order,
        "transversal_merging": STANDARD.transversal_intra_two_qubit,
        "commute": commute,
        "plan_fell_back": bool(plan.fell_back),
        "circuit_idx": circuit_idx,
        "baseline_gate_count": baseline_gate_count,
        "alpha_initial": alpha_initial,
        "pilot_initial_lsr": pilot_initial_lsr,
        "pilot_final_lsr": pilot_final_lsr,
        "placement": json.dumps([list(p) for p in placement]),
        "matched_patches": settings.matched_patches,
        "reused_arms": settings.reuse_arms.name if settings.reuse_arms is not None else "",
        "split_merging": settings.split_merging,
        "solo_aware_planner": solo_pricing,
        "copied_arms": ",".join(copied),
        "share_identical": settings.share_identical,
        "logical_optimization": logical_optimization,
    }
    packed_static = static("packed", placement)  # the plan is the same at every density
    plan_patches = len({patch for patch, _ in placement})

    rows: List[Dict] = []
    for density_bin_idx, n_codes in enumerate(code_counts):
        trivial_perm = tight_trivial_packing(n_qubits, n_codes)
        perms = {arm: trivial_perm for arm in naive_arms}
        statics = {arm: static(arm, trivial_perm) for arm in naive_arms}
        # One logical qubit per patch leaves nothing to pack. The submitted path reached the
        # same place by accident -- `pack_program` at n_code 3 raised and fell back to the
        # trivial packing -- so this bin's "+ Packing" is zero by construction in both, rather
        # than the plan's tighter patch count read as a packing gain.
        one_per_patch = math.ceil(n_qubits / n_codes) < 2
        row_fell_back = bool(plan.fell_back)
        if one_per_patch:
            perms["packed"] = trivial_perm
            statics["packed"] = static("packed", trivial_perm)
            row_fell_back = False
        elif "packed" in copied:
            # The reused run planned this point already; its placement is the copied arm's.
            perms["packed"] = [tuple(p) for p in json.loads(reused[n_codes]["placement"])]
            statics["packed"] = static("packed", perms["packed"])
            row_fell_back = bool(reused[n_codes]["plan_fell_back"])
        elif settings.matched_patches and (settings.solo_aware_planner
                                           or n_codes != plan_patches):
            # Placement at this point's patch count, commutation held at the circuit's
            # decision so the other arms are the unmatched run's exactly. The solo-aware
            # planner re-plans the tightest point too: an odd width has a solo patch there.
            dense = plan_standard(
                resource_program, n_code, noise=planner_noise,
                h_commutation=commute, verify=True, patches=n_codes,
                standard=planner_standard,
                **({"solo_hh": True} if settings.solo_aware_planner else {}),
            )
            perms["packed"] = [(int(patch), int(lane)) for patch, lane in dense.placement]
            statics["packed"] = static("packed", perms["packed"])
            row_fell_back = bool(dense.fell_back)
        else:
            perms["packed"] = placement
            statics["packed"] = packed_static

        seed_off = circuit_idx * 100_000 + density_bin_idx * 10
        sims: Dict[str, Dict[str, float]] = {}
        #: arm -> the earlier arm whose simulation it takes (`settings.share_identical`).
        shared: Dict[str, str] = {}
        simulated_by_key: Dict[str, str] = {}
        for arm, (_, _, offset) in arms.items():
            if reused is not None and arm in copied:
                sims[arm] = _reused_arm(reused[n_codes], arm, statics[arm], circuit_name)
                continue
            seed = settings.seed * 1_000_003 + seed_off + offset
            config = simulation_config(arm, alpha=alpha_final, repetitions=final_repetitions,
                                       seed=seed)
            if settings.share_identical:
                key = simulated_circuit_key(simulation_program, perms[arm], config)
                if key in simulated_by_key:
                    shared[arm] = simulated_by_key[key]
                    sims[arm] = dict(sims[shared[arm]])
                    continue
                simulated_by_key[key] = arm
            sims[arm] = evaluate_variant_frames(
                simulation_program, perms[arm], config,
                frames=settings.frame_shots, frame_seed=seed,
            )

        per_arm: Dict[str, float] = {}
        for arm in arm_names:
            one_q, two_q, depth = statics[arm]
            sim = sims[arm]
            per_arm.update({
                f"{arm}_one_qubit_gates": one_q,
                f"{arm}_two_qubit_gates": two_q,
                f"{arm}_depth": depth,
                f"{arm}_lsr": sim["logical_selection_rate"],
                f"{arm}_tvd": sim["logical_total_variation_distance"],
                f"{arm}_silent_per_kept": sim["silent_per_kept"],
                f"{arm}_p_silent": sim["p_silent"],
                f"{arm}_frame_lsr": sim["frame_lsr"],
            })

        # Stacked as `_run_one_circuit` stacks them: a component is its arm's cumulative
        # value over the baseline minus the previous arm's.
        previous: Dict[str, float] = {}
        matched_fields = ({"plan_fell_back": row_fell_back,
                           "placement": json.dumps([list(p) for p in perms["packed"]])}
                          if settings.matched_patches else {})
        for component, arm in (_COMPONENT_ARM_SPLIT if split else _COMPONENT_ARM):
            now = _standard_cumulative(statics, sims, arm)
            rows.append({
                **per_circuit,
                **matched_fields,
                "n_codes": n_codes,
                "fraction_codes_used": n_codes / n_qubits,
                "packed_patches": len({patch for patch, _ in perms["packed"]}),
                "packed_source": "trivial" if one_per_patch else "plan",
                "shared_arms": ";".join(f"{arm}={source}" for arm, source in shared.items()),
                "component": component,
                **{column: value - previous.get(column, 0.0) for column, value in now.items()},
                **per_arm,
            })
            previous = now

    return rows


@lru_cache(maxsize=2)
def _read_reused_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _reused_rows(path: Path, circuit_name: str, code_counts: Sequence[int]) -> Dict[int, Dict]:
    """One circuit's rows of a standard CSV, by density point. Every point must be there."""
    frame = _read_reused_csv(str(path))
    block = frame[(frame["circuit_name"] == circuit_name) & (frame["component"] == "packing")]
    rows = {int(row["n_codes"]): row for row in block.to_dict("records")}
    missing = [n for n in code_counts if n not in rows]
    if missing:
        raise RuntimeError(f"{circuit_name}: {path.name} has no rows at n_codes {missing}")
    return rows


def _reused_arm(row: Dict, arm: str, static_now: Tuple[int, int, int],
                circuit_name: str) -> Dict[str, float]:
    """
    A reused run's simulation of `arm`, as `evaluate_variant_frames` returns it.

    The arm is re-translated here (cheap) and must match the recorded gate counts and depth:
    the copy stands in for a re-simulation of the same circuit, which a changed translation
    would not be.
    """
    recorded = tuple(int(row[f"{arm}_{q}"]) for q in ("one_qubit_gates", "two_qubit_gates",
                                                      "depth"))
    if recorded != tuple(int(v) for v in static_now):
        raise RuntimeError(f"{circuit_name} n_codes={row['n_codes']}: {arm} translates to "
                           f"{static_now}, the reused run recorded {recorded}")
    return {
        "logical_selection_rate": float(row[f"{arm}_lsr"]),
        "logical_total_variation_distance": float(row[f"{arm}_tvd"]),
        "silent_per_kept": float(row[f"{arm}_silent_per_kept"]),
        "p_silent": float(row[f"{arm}_p_silent"]),
        "frame_lsr": float(row[f"{arm}_frame_lsr"]),
    }


def run_experiment(
    circuits: Sequence[Tuple[str, qiskit.QuantumCircuit, qiskit.QuantumCircuit]],
    size_group: str,
    settings: DensityAblationSettings,
) -> pd.DataFrame:
    """
    One size group, one worker per circuit.

    Circuits within a group differ in width by several fold and every density point on a
    wide one costs more than a whole narrow circuit, so the work is lumpy; `parallel_map`
    hands out items as workers free up rather than splitting the list up front.

    A circuit that raises comes back as `None` and is dropped here rather than taking the
    group down -- see `parallel.parallel_map` on why a failure is a value.

    `settings.names` filters after enumerating, so a circuit run alone keeps the index --
    and the seeds -- it has in a full run.
    """
    jobs: List[CircuitJob] = [
        (settings, size_group, index, name, resource, simulation)
        for index, (name, resource, simulation) in enumerate(circuits)
        if not settings.names or name in settings.names
    ]
    worker = _run_one_circuit_standard if settings.standard else _run_one_circuit
    produced = parallel_map(
        worker, jobs, jobs=settings.jobs, label=f"circuits ({size_group})",
    )
    failed = sum(1 for result in produced if result is None)
    if failed:
        print(f"[density_ablation_sweep] {failed}/{len(jobs)} {size_group} circuits "
              f"failed and were dropped")
    return pd.DataFrame([row for result in produced if result for row in result])


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "size_group", "circuit_name", "num_qubits", "n_codes", "fraction_codes_used",
    "alpha_final", "final_repetitions", "component",
    "pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates",
    "pct_reduction_circuit_depth", "pct_improvement_logical_selection_rate",
    "pct_reduction_logical_total_variation_distance",
]


def _append_csv(df: pd.DataFrame, csv_path: Path, *, first: bool = False) -> None:
    """
    Append one size group's rows, so a kill keeps the groups that finished.

    `first=True` truncates. Without it a second run appended to the previous run's file and
    the plot silently averaged two runs together -- the header check alone cannot tell a
    resumed run from a fresh one, because both see a file that already exists.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if first and csv_path.exists():
        csv_path.unlink()
    write_header = not csv_path.exists()
    df[_CSV_COLUMNS].pipe(stamp_provenance).to_csv(
        csv_path, mode="a", index=False, header=write_header
    )
    print(f"[density_ablation_sweep] Appended {len(df)} rows to {csv_path}")


#: The standard run's schema: the submitted one, then what the standard path adds. The
#: per-component `pct_reduction_silent_per_kept` is stacked like the LSR and TVD columns.
def _standard_csv_columns(split: bool = False) -> List[str]:
    return _CSV_COLUMNS + [
        "pct_reduction_silent_per_kept",
        "configuration", "asap_order", "transversal_merging", "commute", "packed_patches",
        "packed_source", "plan_fell_back", "circuit_idx", "baseline_gate_count",
        "alpha_initial", "pilot_initial_lsr", "pilot_final_lsr",
        *(f"{arm}_{quantity}" for arm in (SPLIT_ARMS if split else STANDARD_ARMS)
          for quantity in _ARM_QUANTITIES),
        "placement", "matched_patches", "reused_arms", "split_merging", "solo_aware_planner",
        "copied_arms", "share_identical", "shared_arms", "logical_optimization",
    ]


_STANDARD_CSV_COLUMNS = _standard_csv_columns()


def _is_standard_csv(df: pd.DataFrame) -> bool:
    return "configuration" in df.columns and bool((df["configuration"] == "standard").any())


def _figure_target(output_dir: Optional[Path]) -> Tuple[Path, Optional[Path]]:
    """
    `(script, override)` to hand `save_figure` for a standard figure.

    `save_figure` writes into results/figures/ (or `override`) and copies the SVG beside
    `script`. With an output directory, `script` is pointed into it, so that copy lands on the
    file just written and nothing is written outside the directory -- in particular no
    `*_standard.svg` beside this script, which would push the real run's stem to `_v2`.
    """
    if output_dir is None:
        return Path(__file__), None
    override = Path(output_dir).resolve()
    return override / Path(__file__).name, override


def _free_standard_csv(base: Path, output_dir: Optional[Path]) -> Path:
    """`base`, else `<stem>_v2.csv`, `_v3`, ...: the first stem with no CSV and no figure."""
    script, override = _figure_target(output_dir)
    figure_dirs = (figure_dir(script, override), script.resolve().parent)

    def taken(stem: str) -> bool:
        return (base.parent / f"{stem}.csv").exists() or any(
            (directory / f"{stem}.{suffix}").exists()
            for directory in figure_dirs for suffix in ("svg", "pdf")
        )

    stem, version = base.stem, 2
    while taken(stem):
        stem, version = f"{base.stem}_v{version}", version + 1
    return base.parent / f"{stem}.csv"


def _write_standard_csv(
    df: pd.DataFrame, settings: DensityAblationSettings, csv_path: Optional[Path]
) -> Path:
    """
    `_append_csv` for the standard path, which never deletes or replaces a file.

    The first size group (`csv_path` None) creates a new file at the first free stem,
    exclusively -- a file appearing there in the meantime is another run's, so the next stem
    is taken instead -- and every later group appends to the path returned.
    """
    frame = df[_standard_csv_columns(settings.split_merging)].pipe(stamp_provenance)
    if csv_path is not None:
        frame.to_csv(csv_path, mode="a", index=False, header=False)
        print(f"[density_ablation_sweep] Appended {len(df)} rows to {csv_path}")
        return csv_path
    while True:
        csv_path = _free_standard_csv(settings.csv_path, settings.output_dir)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(csv_path, "x", newline="") as handle:
                frame.to_csv(handle, index=False)
        except FileExistsError:
            continue
        print(f"[density_ablation_sweep] Wrote {len(df)} rows to new {csv_path}")
        return csv_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _assign_plot_bins(
    df: pd.DataFrame,
    n_plot_bins: int,
) -> pd.DataFrame:
    df = df.copy()
    bin_edges = np.linspace(0.5, 1.0, n_plot_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df["plot_bin"] = pd.cut(
        df["fraction_codes_used"],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
    ).astype(float)
    return df


#: Stage order, bottom of the stack upwards. Each is cumulative on the one before it.
COMPONENT_ORDER = tuple(key for key, _, _ in COMPONENTS)


def _aggregate_for_plot(
    df: pd.DataFrame,
    n_plot_bins: int,
) -> Dict[Tuple[str, float, str], np.ndarray]:
    """
    `(size_group, plot_bin, metric column)` -> that cell's stacked factor boundaries.

    One change from the version this replaces: it no longer clips each component at zero
    before averaging. `stacked_percent` clips the running total instead, which is what
    makes a bar's height equal the improvement the circuits in it actually reached; see
    that function on what per-stage clipping did to `ising_n98`.

    Boundaries are cumulative and the mean is linear, so averaging them gives a bar whose
    segments are the mean of that stage's segments and whose top is the mean of the totals.
    """
    df = _assign_plot_bins(df, n_plot_bins)
    metric_cols = [column for column, _ in METRICS.values()]
    out: Dict[Tuple[str, float, str], np.ndarray] = {}
    for (size_group, plot_bin), block in df.groupby(["size_group", "plot_bin"]):
        for column in metric_cols:
            pivot = block.pivot_table(
                index="circuit_name", columns="component", values=column, aggfunc="mean",
            )
            for component in COMPONENT_ORDER:
                if component not in pivot.columns:
                    pivot[component] = 0.0
            pivot = pivot[list(COMPONENT_ORDER)].fillna(0.0).astype(float)
            rows = [stacked_percent(row.tolist()) for _, row in pivot.iterrows()]
            out[(size_group, float(plot_bin), column)] = np.asarray(rows).mean(axis=0)
    return out


def plot_results(
    df: pd.DataFrame, settings: DensityAblationSettings, stem: Optional[str] = None
) -> None:
    """
    Draw and save the figure. `stem` None is the submitted figure, `density_ablation_sweep`,
    saved as it always was; a stem (the standard path's) is saved versioned, never replacing
    a figure, and into `settings.output_dir` alone when that is set.
    """
    agg = _aggregate_for_plot(df, settings.n_plot_bins)
    bin_centers = sorted({bin_center for _, bin_center, _ in agg if bin_center == bin_center})
    n_bins = len(bin_centers)
    # Bar thickness is set by the *gaps*, not by `bar_width` alone. At `bin_step = 4.0` a
    # group spanned 2.06 units and the gap to the next spanned 1.94, so half the axis was
    # empty and 18 bars over one column width came out around 10pt each -- thin enough that
    # the stack segments were hard to separate. Closing the inter-group gap to 0.5 is what
    # buys the width back; `bar_width` only rises from 0.62 to 0.70.
    bin_step = 2.6
    # A gap between the three bars of a group, which the original did not have: touching
    # bars make the S/M/L labels read as one word. `bar_width < offset spacing` is what
    # separates them, and the margin is deliberately small now that space is tight.
    bar_width = 0.82
    group_offsets = np.array([-0.84, 0.0, 0.84])

    use_style()
    # 2x2 at full text width rather than five rows down one column. The two gate metrics
    # move as one -- they are both "how much smaller is the translated circuit" and their
    # curves track each other across every density -- so averaging them costs nothing and
    # frees the fourth slot. Each panel then gets a full column's width, so the bars stay
    # the size they were while the figure loses half its height.
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(FULL_WIDTH, 1.55 * 2 + 1.25),
        sharex=True,
    )
    axes = axes.ravel()

    for ax, (metric_cols, metric_label) in zip(axes, PANELS):
        highest = 0.0
        for bin_idx, bin_center in enumerate(bin_centers):
            for g_idx, size_group in enumerate(SIZE_GROUPS):
                x = bin_idx * bin_step + group_offsets[g_idx]
                boundaries = _panel_boundaries(agg, size_group, bin_center, metric_cols)
                if boundaries is None:
                    continue
                for index, (_, _, color) in enumerate(COMPONENTS):
                    ax.bar(
                        x,
                        boundaries[index + 1] - boundaries[index],
                        width=bar_width,
                        bottom=boundaries[index],
                        color=color,
                        edgecolor="white",
                        alpha=0.85,
                    )
                highest = max(highest, float(boundaries[-1]))

        ax.set_ylabel(metric_label)
        # Percent and linear, matching `merging_improvement_..._simulated`, which stacks the
        # same three components. A factor axis was tried on both and reverted: percent
        # compresses the spread where factors expand it, and on a stacked ablation bar the
        # spread is what has to fit.
        ax.set_ylim(bottom=0.0, top=highest * 1.06)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # X-axis ticks: one tick group per bin, labelled with fraction
    tick_positions = [
        bin_idx * bin_step + group_offsets[1]
        for bin_idx in range(n_bins)
    ]
    # The sweep is indexed by `fraction_codes_used = n_codes / n_qubits`; the axis the paper
    # reads is **packing density = n_qubits / n_codes**, its reciprocal. Density k means every
    # patch is full and density 1 means every qubit sits alone, which is the quantity a reader
    # is choosing along. Same bins, relabelled: 0.54 -> 1.85, 0.96 -> 1.04.
    tick_labels = [f"{1.0 / c:.2f}" if c else "" for c in bin_centers]
    # Both panels on the bottom row carry ticks now, not just the last axis. `sharex` hides
    # the top row's labels automatically.
    for ax in axes[2:]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    fig.supxlabel("packing density")
    # Reclaim matplotlib's default 5% x margins. With 18 bars per panel the bars are the
    # whole figure, and 10% of the width spent on empty ends is 10% not spent on them.
    # `sharex` propagates this to all four panels. The pad clears the outermost bar's
    # half-width (0.84 + 0.82/2 = 1.25) by a hair.
    axes[-1].set_xlim(-1.32, (n_bins - 1) * bin_step + 1.32)

    # Legend: component colors + size-group hatches
    component_patches = [
        mpatches.Patch(color=color, alpha=0.85, label=label)
        for _, label, color in COMPONENTS
    ]
    # One row above the top panel rather than inside it. In-panel it sat on the 1q row's
    # tallest bars, and now that the groups are closer together there is no free corner in
    # any of the five rows to move it to.
    fig.legend(
        handles=component_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(COMPONENTS),
        frameon=False,
        fontsize=plt.rcParams["font.size"] * 0.9,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.0,
    )

    # S / M / L inside every bar, in every row.
    #
    # The archived version annotated only the *last* density bin, below the axis, which is
    # why the published figure's per-bar labels had to be added by hand. Three bars per
    # group with nothing distinguishing them is unreadable at column width, and the
    # alternative -- a second legend keyed by hatch or shade -- costs a legend entry per
    # size group in all five rows.
    #
    # Drawn at the base of the bar, and the colour follows what is actually underneath it
    # rather than being fixed. The bottom segment is always `H commutation` at `#3b4c56`,
    # dark enough to carry white type -- but only where that segment is tall enough to
    # contain the glyph. Where H commutation contributes almost nothing the glyph straddles
    # onto `+ Merging` (`#cdc2c0`) or `+ Packing` (`#0097fd`), both light, and white on
    # those is unreadable. So the test is the height of the *first* segment, not of the
    # whole bar, and the fallback is near-black rather than a different position.
    #
    # A bar whose total is too short to hold the glyph at all still gets it above the bar,
    # which is why the `1.04` group shows labels floating.
    #
    # Sized below the style's 8pt minor tier: bars are about 10pt wide at column width and
    # narrower once the `large` group restores the third bar per group, so an 8pt `M`
    # overhangs the bar it belongs to and reads as if it belonged to its neighbour.
    at_6pt = plt.rcParams["font.size"] * 0.62
    for ax, (metric_cols, _) in zip(axes, PANELS):
        low, high = ax.get_ylim()
        span = high - low
        for bin_idx, bin_center in enumerate(bin_centers):
            for g_idx, size_group in enumerate(SIZE_GROUPS):
                boundaries = _panel_boundaries(agg, size_group, bin_center, metric_cols)
                if boundaries is None:
                    continue
                total = float(boundaries[-1])
                # Height of the `H commutation` block, which is what sits under the glyph.
                dark_segment = float(boundaries[1]) - float(boundaries[0])
                x = bin_idx * bin_step + group_offsets[g_idx]
                initial = SIZE_GROUP_LABELS[size_group][0].upper()
                if dark_segment >= 0.06 * span:
                    ax.text(x, low + 0.012 * span, initial, ha="center", va="bottom",
                            fontsize=at_6pt, color="white", zorder=5)
                else:
                    # No dark block to sit on. Black type on the light segments was legible
                    # but read as crowded -- a glyph wedged into a thin bar looks like part
                    # of the bar. Clear of the bar entirely is cleaner, and it is where the
                    # too-short-for-any-glyph case already put it, so the two collapse into
                    # one rule: sit on the dark block, or sit above everything.
                    ax.text(x, total + 0.012 * span, initial, ha="center", va="bottom",
                            fontsize=at_6pt, color="#444444", zorder=5)

    fig.supylabel("percent improvement / reduction")
    # No `tight_layout` here: the shared style sets `figure.constrained_layout.use`, and
    # calling both makes matplotlib warn and discard one of them.

    # `save_figure` writes SVG and PDF at the canvas size, into `results/figures/` plus a
    # copy beside this script. The old `savefig(dpi=200)` wrote one SVG to a hardcoded path.
    # `constrained_layout` sizes the axes to fill the canvas and ignores `fig.legend`, so
    # adding figure height just made the panels taller and the legend still landed on them.
    # Shrinking the layout rect is what actually reserves the strip.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.90))
    if stem is None:
        for path in save_figure(fig, "density_ablation_sweep", Path(__file__)):
            print(f"[density_ablation_sweep] wrote {path}")
    else:
        script, override = _figure_target(settings.output_dir)
        for path in save_figure(fig, stem, script, override, versioned=True):
            print(f"[density_ablation_sweep] wrote {path}")
    plt.close(fig)

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> DensityAblationSettings:
    parser = argparse.ArgumentParser(
        description="Ablation sweep across code density for all QASMBench circuits."
    )
    parser.add_argument("--qasm-root", type=Path, default=DEFAULT_SETTINGS.qasm_root)
    parser.add_argument("--max-gates", type=int, default=DEFAULT_SETTINGS.max_proxy_gates)
    parser.add_argument("--max-qubits", type=int, default=DEFAULT_SETTINGS.max_proxy_qubits)
    parser.add_argument("--proxy-timeout", type=float, default=DEFAULT_SETTINGS.proxy_timeout_seconds)
    parser.add_argument("--n-codes-samples", type=int, default=DEFAULT_SETTINGS.n_codes_samples)
    parser.add_argument("--n-plot-bins", type=int, default=DEFAULT_SETTINGS.n_plot_bins)
    parser.add_argument("--alpha-reference", type=float, default=DEFAULT_SETTINGS.alpha_reference)
    parser.add_argument("--reference-gate-count", type=float, default=DEFAULT_SETTINGS.reference_gate_count)
    add_jobs_argument(parser)
    parser.add_argument("--target-selection-rate", type=float, default=DEFAULT_SETTINGS.target_selection_rate)
    parser.add_argument("--pilot-repetitions", type=int, default=DEFAULT_SETTINGS.pilot_repetitions)
    parser.add_argument("--target-kept-shots", type=int, default=DEFAULT_SETTINGS.target_kept_shots)
    parser.add_argument("--min-repetitions", type=int, default=DEFAULT_SETTINGS.min_repetitions)
    parser.add_argument("--max-repetitions", type=int, default=DEFAULT_SETTINGS.max_repetitions)
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS.seed)
    parser.add_argument("--no-stim", action="store_true")
    parser.add_argument("--no-distance-noise", action="store_true")
    parser.add_argument("--distance-scalar", type=float, default=DEFAULT_SETTINGS.distance_scalar)
    parser.add_argument("--no-idle-noise", action="store_true")
    parser.add_argument("--idle-scalar", type=float, default=DEFAULT_SETTINGS.idle_scalar)
    parser.add_argument("--output", type=Path, default=DEFAULT_SETTINGS.output_path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Default {DEFAULT_SETTINGS.csv_path}, which a fresh run replaces. With "
             f"--standard: the base name of a new CSV (default {STANDARD_CSV.name} beside "
             f"this script), versioned _v2, _v3, ... rather than replaced.",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Reload a previously saved CSV and replot without resimulating.",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--standard",
        action="store_true",
        help="The standard configuration (packing.standard.STANDARD): every arm layer-sorted, "
             "transversal in-patch CZ/XCX on the merging arms, plan_standard for the packed "
             "arm with its commutation decision honoured, alpha recalibrated, silent error "
             "per kept shot on every arm. Never replaces a CSV or figure. A CSV it wrote "
             "replots under its own name with --from-csv, with or without this flag.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=[],
        help="--standard only: run just these circuits (each keeps its full-run seeds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="--standard only: write the figure here, and nothing beside this script.",
    )
    parser.add_argument(
        "--matched-patches",
        action="store_true",
        help="--standard only: plan the packed arm at each density point's own patch count "
             f"(plan_standard(patches=n_codes)). CSV {MATCHED_CSV.name}, versioned.",
    )
    parser.add_argument(
        "--reuse-arms",
        type=Path,
        default=None,
        help="--matched-patches only: a --standard CSV whose baseline, +H and +merge arms and "
             "calibration to copy rather than re-simulate (checked circuit by circuit); with "
             "--split-merging, a --matched-patches CSV whose +merge and packed arms to copy.",
    )
    parser.add_argument(
        "--split-merging",
        action="store_true",
        help="--matched-patches with --reuse-arms only: merging in two steps, single-qubit "
             "macros then two-qubit ones, on a baseline and +H arm with no merging at all (no "
             f"solo-H promotion either). CSV {SPLIT_CSV.name}, versioned.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=SIZE_GROUPS,
        default=[],
        help="--standard only: run just these size groups.",
    )
    parser.add_argument(
        "--simulate",
        nargs="+",
        choices=SPLIT_ARMS,
        default=[],
        help="--reuse-arms only: simulate these arms and copy every other one (default: packed, "
             "or with --split-merging baseline, h_only and h_merge_1q).",
    )
    parser.add_argument(
        "--solo-aware-planner",
        action="store_true",
        help="--matched-patches only: plan the packed arm with solo patches' Hadamards priced "
             "as the HH the translator emits (plan_standard(solo_hh=True)), at every point. The "
             "standard's default since 2026-10-09; matters with --legacy-solo-pricing.",
    )
    parser.add_argument(
        "--legacy-solo-pricing",
        action="store_true",
        help="--standard only: plan with the pre-2026-10-09 pricing of solo patches "
             "(STANDARD.with_(solo_hh_aware=False)), to reproduce earlier CSVs.",
    )
    parser.add_argument(
        "--share-identical",
        action="store_true",
        help="--standard only: an arm that compiles to exactly an earlier arm's circuit at the "
             "same density point takes that arm's simulated metrics instead of a second "
             "simulation with its own seed (rows: shared_arms).",
    )
    args = parser.parse_args()
    if (args.names or args.output_dir) and not args.standard:
        # A fresh submitted run truncates density_ablation_sweep.csv and replaces its figure,
        # so a subset run there would destroy the full sweep's outputs.
        parser.error("--names and --output-dir are --standard options")
    if args.matched_patches and not args.standard:
        parser.error("--matched-patches is a --standard option")
    if args.groups and not args.standard:
        parser.error("--groups is a --standard option")
    if args.split_merging and not args.matched_patches:
        parser.error("--split-merging is a --matched-patches option")
    if args.share_identical and not args.standard:
        parser.error("--share-identical is a --standard option")
    if args.simulate and args.reuse_arms is None:
        parser.error("--simulate is a --reuse-arms option")
    if args.solo_aware_planner and not args.matched_patches:
        parser.error("--solo-aware-planner is a --matched-patches option")
    if args.legacy_solo_pricing and not args.standard:
        parser.error("--legacy-solo-pricing is a --standard option")
    if (args.solo_aware_planner and args.reuse_arms is not None
            and "packed" not in (args.simulate or (["baseline", "h_only", "h_merge_1q"]
                                                   if args.split_merging else ["packed"]))):
        parser.error("--solo-aware-planner re-plans the packed arm, so it must be simulated")
    if args.reuse_arms is not None:
        if not args.matched_patches:
            parser.error("--reuse-arms is a --matched-patches option")
        if not args.reuse_arms.is_file():
            parser.error(f"--reuse-arms: no such file {args.reuse_arms}")
        # The copied arms were simulated under the reused run's settings; the packed arm is
        # simulated under these. Only the defaults are known to be that run's.
        changed = [flag for flag, value, default in _simulation_flags(args) if value != default]
        if changed:
            parser.error(f"--reuse-arms needs the default simulation settings; changed: "
                         f"{', '.join(changed)}")
    csv_path = args.csv
    if csv_path is None:
        if args.split_merging:
            csv_path = SPLIT_CSV
        elif args.matched_patches:
            csv_path = MATCHED_CSV
        else:
            csv_path = STANDARD_CSV if args.standard else DEFAULT_SETTINGS.csv_path

    return DensityAblationSettings(
        qasm_root=args.qasm_root,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        n_codes_samples=args.n_codes_samples,
        n_plot_bins=args.n_plot_bins,
        alpha_reference=args.alpha_reference,
        reference_gate_count=args.reference_gate_count,
        jobs=args.jobs,
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
        csv_path=csv_path,
        from_csv=args.from_csv,
        show_plot=not args.no_show,
        standard=args.standard,
        names=tuple(args.names),
        output_dir=args.output_dir,
        matched_patches=args.matched_patches,
        reuse_arms=args.reuse_arms.resolve() if args.reuse_arms is not None else None,
        split_merging=args.split_merging,
        groups=tuple(args.groups),
        simulate=tuple(args.simulate),
        solo_aware_planner=args.solo_aware_planner,
        legacy_solo_pricing=args.legacy_solo_pricing,
        share_identical=args.share_identical,
    )


def _simulation_flags(args) -> List[Tuple[str, object, object]]:
    """(flag, value, default) for every setting the simulated arms depend on."""
    d = DEFAULT_SETTINGS
    return [
        ("--qasm-root", args.qasm_root, d.qasm_root),
        ("--max-gates", args.max_gates, d.max_proxy_gates),
        ("--max-qubits", args.max_qubits, d.max_proxy_qubits),
        ("--proxy-timeout", args.proxy_timeout, d.proxy_timeout_seconds),
        ("--n-codes-samples", args.n_codes_samples, d.n_codes_samples),
        ("--alpha-reference", args.alpha_reference, d.alpha_reference),
        ("--reference-gate-count", args.reference_gate_count, d.reference_gate_count),
        ("--target-selection-rate", args.target_selection_rate, d.target_selection_rate),
        ("--pilot-repetitions", args.pilot_repetitions, d.pilot_repetitions),
        ("--target-kept-shots", args.target_kept_shots, d.target_kept_shots),
        ("--min-repetitions", args.min_repetitions, d.min_repetitions),
        ("--max-repetitions", args.max_repetitions, d.max_repetitions),
        ("--seed", args.seed, d.seed),
        ("--no-stim", args.no_stim, not d.stim),
        ("--no-distance-noise", args.no_distance_noise, not d.distance_noise),
        ("--distance-scalar", args.distance_scalar, d.distance_scalar),
        ("--no-idle-noise", args.no_idle_noise, not d.idle_noise),
        ("--idle-scalar", args.idle_scalar, d.idle_scalar),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_standard(settings: DensityAblationSettings) -> None:
    """
    `main`'s sweep under the standard configuration, appending one size group at a time.

    The CSV is created by the first group that produces rows, at the first free stem, and
    only ever appended to after that; the figure takes the CSV's stem, versioned.
    """
    csv_path: Optional[Path] = None
    for group in (settings.groups or SIZE_GROUPS):
        circuits = _load_group_circuits(settings.qasm_root, group, settings)
        if not circuits:
            print(f"[density_ablation_sweep] No circuits loaded for {group}, skipping.")
            continue
        mode = "standard configuration"
        if settings.matched_patches:
            mode += ", matched patch counts"
        if settings.split_merging:
            mode += ", merging split into single- and two-qubit steps"
        if settings.solo_aware_planner:
            mode += ", solo-aware planner"
        if settings.reuse_arms is not None:
            mode += f", arms reused from {settings.reuse_arms.name}"
        print(f"[density_ablation_sweep] Loaded {len(circuits)} {group} circuits ({mode}).")
        group_df = run_experiment(circuits, group, settings)
        if not group_df.empty:
            csv_path = _write_standard_csv(group_df, settings, csv_path)

    if csv_path is None:
        raise RuntimeError("No data was produced — check qasm_root, circuit filters and --names.")

    if settings.split_merging:
        # Four stacked stages; this script's figure draws three. density_references plots it.
        print(f"[density_ablation_sweep] split-merging CSV written; plot it with "
              f"density_references (recount, floors, plot --matched {csv_path.name})")
        return
    df = pd.read_csv(csv_path)
    plot_results(df, settings, stem=csv_path.stem)


def main() -> None:
    settings = parse_args()

    if settings.from_csv is not None:
        df = pd.read_csv(settings.from_csv)
        if settings.standard or _is_standard_csv(df):
            # Under the CSV's own name, versioned: a standard CSV must never replot onto the
            # submitted figure's name.
            plot_results(df, settings, stem=settings.from_csv.stem)
        else:
            plot_results(df, settings)
        return

    if settings.standard:
        run_standard(settings)
        return

    # Run in order: small → medium → large, appending CSV after each group, so a kill
    # leaves the finished groups on disk and `--from-csv` replots them.
    written = False
    for group in SIZE_GROUPS:
        circuits = _load_group_circuits(settings.qasm_root, group, settings)
        if not circuits:
            print(f"[density_ablation_sweep] No circuits loaded for {group}, skipping.")
            continue
        print(f"[density_ablation_sweep] Loaded {len(circuits)} {group} circuits.")
        group_df = run_experiment(circuits, group, settings)
        if not group_df.empty:
            _append_csv(group_df, settings.csv_path, first=not written)
            written = True

    if not settings.csv_path.exists():
        raise RuntimeError("No data was produced — check qasm_root and circuit filters.")

    df = pd.read_csv(settings.csv_path)
    plot_results(df, settings)


if __name__ == "__main__":
    main()
