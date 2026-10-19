"""
simulation.py -- shared LSR/TVD measurement for the figures that need it.

Gate counts come from translation alone and are fast. Logical selection rate and total
variation distance need the circuit actually simulated, postselected and compared
against an unencoded reference, which is orders of magnitude slower and is why the
figures that use it have smaller defaults.

Promoted from `experiments/figures/updated/simulation.py`, which is archived. Two things
were added: `distance_scalar` and `idle_time_scalar` are settings rather than pinned at
1.0, because several figures sweep the ratio between them; and `evaluate` takes an
explicit permutation so a caller can compare placements at fixed noise.

Two things to know before reading any number this produces:

**Noise has to be calibrated per circuit.** Postselection is an exponentially strict
filter, so a fixed noise scale either keeps everything (and measures nothing) or keeps
nothing (and measures noise). Each circuit gets a pilot run, and alpha is solved so the
baseline selection rate lands near `target_selection_rate`. The shot count then follows
from the selection rate that calibration found, so a circuit that keeps 3% of its shots
is run harder than one that keeps 30% and both land near `target_kept_shots` accepted
samples. That is what makes error bars comparable across circuit sizes.

**The program circuit must carry measurements.** Nothing in `pipeline.tasks.scan_*` adds
them, and a circuit without them does not fail -- it returns a perfectly plausible
selection rate next to a TVD of exactly 1.0, because the ideal reference distribution it
is compared against is empty. Measured on an 8-qubit random Clifford: TVD 1.0 without,
0.057 with, at an unchanged LSR of 0.83. `evaluate` therefore calls `ensure_measurements`
itself rather than trusting callers, since the failure is silent and looks like a result.

**The Stim backend used to be wrong above k = 2, and is now fixed.** It returned a
logical distribution with every classical register bit-reversed:
`stim_backend._format_key_by_cregs` emitted Qiskit's *printed* little-endian order,
while `iceberg_postselect` reads position `d` of a data register as data qubit `d` --
the convention Aer reaches via `Simulator._invert_endianness`. `Simulator._stimulate`
then inverted the ideal call only, which fixed the unencoded reference (one creg, so
the register reordering was a no-op) and left the encoded distribution reversed.

k = 2 could not see it. `CodeLayout.decode_logical` is reversal-invariant on a 4-bit
even-parity codeword, so the [[4,2,2]] numbers were right and nothing complained. At
n = 6 the same reversal permutes the lanes and mixes in the wrong anchor. At
alpha = 1e-4, where TVD must sit at the sampling floor, an 8-qubit random Clifford
measured:

                    k=2      k=4      k=6
    stim (before)   0.056    0.775    0.771
    stim (after)    0.063    0.058    0.073
    aer             0.059    0.050    0.055

`tests/test_pipeline_roundtrip.py` parametrized CODE_SIZES=[4, 6] and passed, but ran
Aer noiseless and never touched the Stim path, which is why this went unnoticed.
`test_stim_matches_aer_noiseless` now runs both backends at n_code 4, 6 and 8 and
asserts they agree, and fails at 6 and 8 if the reversal comes back.

**Selection rate still differs between the backends, and that is a separate issue.**
It was never the reversal -- both the syndrome test (`"1" in register`) and the
codeword test (even parity) are reversal-invariant, so neither can move with bit
order. The two backends build different channels for the same alpha, and the gap grows
with noise: at alpha = 1e-4 they agree to 4e-4, while at 12 qubits, k=4, alpha = 0.02,
20k shots they read lsr 0.956 against 0.924 with the distance and idle mechanisms off,
and 0.761 against 0.939 with them on. TVD agrees to ~0.003 in all of those. So a
selection *rate* taken from Stim under appreciable noise is still not interchangeable
with Aer's; a logical distribution now is.

`stim` defaults to True, because Stim is the backend this project actually simulates
on and the logical distribution now agrees at every code size. Aer costs about 128x more
wall clock (1.06s against 135.7s for one 20k-shot point at 12 qubits) and is kept as the
reference rather than the default. The one place to think twice is a figure whose y axis
is selection rate under appreciable noise: pass `--backend aer` there, or read the LSR as
Stim's own number rather than as interchangeable with Aer's.

**Seeded runs used to sample less than they claimed.** `Simulator._shot_chunk_size_for`
caps a chunk at `repetitions // 100`, and a single `seed_simulator` on the Aer backend
made every chunk a replay of the first -- `repetitions / 100` independent samples, so
roughly ten times the sampling error the shot count implied. Fixed: chunks now draw
their own seeds from the configured one, pinned by `tests/test_simulator_seeding.py`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.config_presets import ScaledSimulator

__all__ = [
    "SimSettings",
    "measured_regime",
    "MOVEMENT_BALANCE",
    "Perm",
    "METRICS",
    "build_config",
    "evaluate",
    "baseline_gate_count",
    "calibrate",
    "Calibration",
    "naive_packing",
    "ensure_measurements",
    "measured_noise_split",
    "noise_profile_for",
    "knobs_for",
    "noiseless_tvd",
    "check_compile",
    "miscompiled",
    "tvd_resolved",
    "CORRECTNESS_RATIO",
    "CORRECTNESS_MARGIN",
    "CORRECTNESS_REPETITIONS",
    "TVD_RESOLUTION_LIMIT",
]

Perm = List[Tuple[int, int]]

#: **Deprecated as a plotted axis. There is no correct value, and the axis it served is
#: confounded.** `with_noise_ratio` remains legitimate as a *coverage generator*: sweep
#: it to spread points across regimes, then plot each point at the ratio the schedule
#: actually charged (`measured_regime` / `measured_noise_split`) -- which is what every
#: ratio figure now does.
#:
#: The intent was an exchange rate: the `distance_scalar` at which movement contributes as
#: much error as idling does at `idle_time_scalar = 1`, so `ratio = 1` could mean "balanced"
#: rather than "equal scalars". Under the old per-physical-gate distance charge that was a
#: fitted constant (64, geometric mean over 13 QASMBench proxies, bimodal). Under the
#: transport model it is not a constant at all.
#:
#: **Why it cannot be one.** How much transport a circuit does is set by how many patches
#: it has, which is set by `k`. Solving for the balance point on random Clifford circuits
#: at 12-20 logical qubits (measured before the split's attribution and rate conventions
#: were corrected -- treat the spread as the result, not the digits):
#:
#:     k = 2   balance 5.6e-05 to 7.4e-04
#:     k = 4   balance 0.22 to 0.87
#:     k = 6   balance 3.2
#:     k = 10  balance 6.2
#:
#: Five orders of magnitude, monotone in `k`. A single number cannot serve them.
#:
#: **And the axis is circular where it is used.** `1_best_k_phase_diagram` puts this ratio
#: on x and colours by best `k` -- but the ratio's meaning *moves with* `k`, because `k`
#: sets the patch count that sets the transport. The x-axis is not independent of the
#: colour variable: "movement-heavy" and "small k" are close to the same statement, and a
#: figure whose x-axis is one and whose colour is the other is partly drawing its own axis.
#:
#: **A structural problem on top.** `distance_scalar` slows the atoms, so transport
#: seconds grow with it, while `idle_time_scalar` multiplies what those seconds cost --
#: and `with_noise_ratio` moves the two scalars reciprocally. Their product is constant,
#: so the idling that transport forces on the array (`idle_transport`) is *invariant
#: along the entire ratio sweep*: the axis actually trades gate/readout idle against
#: move handling and flight, and leaves that term fixed.
#:
#: **Attribution note.** ``movement`` in `measured_noise_split` now means distance-based
#: noise only -- move handling plus flight dephasing, the charges on the atoms actually
#: carried. The idling transport causes stays in the ``idle`` roll-up as
#: ``idle_transport``. Movement-share numbers quoted before this convention landed
#: counted ``idle_transport`` as movement and priced idle at 1/T2 rather than the
#: twirl rate, so they are not comparable digit for digit.
#:
#: **What to sweep instead.** A real hardware parameter that does not depend on the code
#: size -- `T2` is the natural one, and it spans the whole range on its own (movement is
#: 7% of the budget at T2 = 1.5 s and 88% at T2 = 10 ms at k = 2). Report the measured
#: split from `measured_noise_split` as an annotation rather than as the axis.
#:
#: Kept at 64 so existing figures still run and reproduce their cached CSVs. Nothing new
#: should use it.
MOVEMENT_BALANCE = 64.0


#: Moved to `simulator.attribution`, where it can be reached by `packing` as well --
#: the placement objective needs to price the real schedule, not a mean. Re-exported
#: here so every existing `from experiments.shared.simulation import ...` still works.
from simulator.attribution import measured_noise_split  # noqa: E402


def measured_regime(
    program: qiskit.QuantumCircuit,
    permutation: Perm,
    *,
    n_code: int,
    idle_time_scalar: float = 1.0,
    distance_scalar: float = 1.0,
    h_commutation: bool = False,
    merging: bool = False,
    asap_order: bool = False,
) -> Dict[str, float]:
    """
    The noise regime a workload *actually* experiences at these scalars.

    `asap_order` layer-sorts the translation, as `SimSettings.asap_order` does for the arms.

    Translates `program` under `permutation` (baseline passes by default, matching how
    `calibrate` defines the reference configuration) and reads the schedule with
    `measured_noise_split`. This is the honest x-axis for every figure that used to
    plot the nominal idle/distance ratio: sweep the scalars for coverage, then plot
    each point at the ratio the simulator really charged, because the mapping from
    scalars to regime depends on the circuit, the placement and `k` -- a nominal
    "ratio = 1" is a different physical point for every column.

    **Alpha-invariant by construction.** Since `move_error` scales with alpha like
    every other error rate, idle and movement scale together and their ratio does not
    move with the calibrated noise scale -- so the regime can be measured once at the
    NEUTRAL_ATOM baseline and labels every alpha the calibration might pick.

    Returns:
        ``measured_ratio``  idle / movement, the axis value (``inf`` when the workload
        has no movement at all -- single patch, nothing beyond blockade);
        ``movement_share``  movement / total, a bounded alternative;
        plus the raw ``idle``, ``movement`` and ``total`` for annotation.
    """
    from simulator.noise_models import StimNoiseParameters

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    config.translator_config.asap_order = asap_order
    translated = translate_permutation(
        ensure_measurements(program), permutation, config=config
    ).circuit
    split = measured_noise_split(
        translated,
        StimNoiseParameters.neutral_atom(),
        idle_time_scalar=idle_time_scalar,
        distance_scalar=distance_scalar,
    )
    idle, movement, total = split["idle"], split["movement"], split["total"]
    return {
        "measured_ratio": (idle / movement) if movement > 0 else math.inf,
        "movement_share": (movement / total) if total > 0 else 0.0,
        "idle": idle,
        "movement": movement,
        "total": total,
    }


@dataclass(frozen=True)
class SimSettings:
    """Noise calibration and shot budget."""

    alpha_reference: float = 0.1
    reference_gate_count: float = 200.0
    #: Where alpha is solved to put the *baseline* selection rate. Lowering it moves the
    #: whole comparison into a harder noise regime, which is where post-selection has
    #: something to win: a 72-point grid over this and `readout_error` measured LSR
    #: improvement rising 15.3% -> 21.7% as this fell 0.12 -> 0.06, with TVD improvement
    #: flat (30.2% -> 30.0%, inside shot noise). 0.03 was rejected on both counts: TVD
    #: fell to 26.9%, and the implied budget (20000/0.0372 = 537k) exceeds
    #: `max_repetitions` below, which clips silently rather than erroring.
    target_selection_rate: float = 0.06
    pilot_repetitions: int = 4_000
    target_kept_shots: int = 20_000
    min_repetitions: int = 20_000
    max_repetitions: int = 400_000
    #: True: Stim is what this project simulates on, and its logical distribution now
    #: agrees with Aer at every k. See the module docstring on the selection-rate gap.
    stim: bool = True
    #: Per-shot tqdm bars from the simulator. **Turn this off for anything that runs under
    #: `parallel_map`.** Every worker inherits the parent's stdout, so N workers serialise
    #: their progress writes on one pipe, and the bar updates thousands of times per
    #: simulation. Measured on `improvement_vs_noise_and_k` at 12 workers: 148 KB of log
    #: per work item with the bars on against 15 KB with them throttled, and the items
    #: completed 2.6x faster -- the workers were blocked on writes rather than computing,
    #: which reads as low per-core utilisation and is easy to misdiagnose as too few
    #: workers. Left True so a serial run still shows progress on a long simulation.
    shot_progress: bool = True
    #: Movement noise: error charged for the distance a cross-patch gate travels.
    distance_noise: bool = True
    distance_scalar: float = 1.0
    #: Dephasing charged to a qubit sitting idle through a layer.
    idle_noise: bool = True
    idle_time_scalar: float = 1.0
    #: Left at None deliberately -- see the module docstring on seeded runs.
    seed: Optional[int] = None
    #: The standard configuration's translation settings (`packing.standard`), off by
    #: default so every existing figure is unchanged. `asap_order` layer-sorts every
    #: translation this module builds -- arms, pilots, guards, regime and gate-count
    #: references alike; `transversal_merging` builds in-patch CZ / XCX transversally at
    #: k = 2 on translations with merging on, crediting the gadget to merging as Fig 6 does.
    asap_order: bool = False
    transversal_merging: bool = False

    def with_standard(self) -> "SimSettings":
        """These settings under the standard configuration's translation settings."""
        from packing.standard import STANDARD

        return replace(self, asap_order=STANDARD.asap_order,
                       transversal_merging=STANDARD.transversal_intra_two_qubit)

    def with_noise_ratio(
        self, ratio: float, *, balance: float = MOVEMENT_BALANCE
    ) -> "SimSettings":
        """
        Set idle and distance scalars to a given idle/distance ratio of *contributions*.

        Sweeping the ratio by moving one scalar and pinning the other confounds two
        things: the balance between the mechanisms, and how much total noise there is.
        Pinning the geometric mean moves only the balance, so a crossover in the result is
        a crossover in *which mechanism dominates* rather than an artefact of one arm of
        the sweep being noisier overall.

        **`balance` is what makes `ratio = 1` mean what it says.** This used to pin
        `sqrt(idle * distance) == 1`, which equalises the two *scalars* -- and the two
        scalars do not buy comparable amounts of error. Measured on the k=2 naive baseline
        over 13 QASMBench proxies, the error budget grows by about 0.41 per unit of
        `idle_time_scalar` and 0.017 per unit of `distance_scalar`, a factor near 30. So
        `ratio = 1` under the old convention was not the balance point; it sat deep in the
        idle-dominated regime, and a sweep symmetric about it never reached a regime where
        movement dominated anything. `code_size_crossover` found this the expensive way:
        its "movement-dominated" end was bunched because movement was barely participating.

        `balance` scales the distance arm so that at `ratio = 1` the two mechanisms
        contribute equally. `MOVEMENT_BALANCE` documents how the default was measured.

        **Changing this changed every figure that sweeps the ratio.** Any cached CSV or
        quoted number produced before it -- `depth_vs_gates_tradeoff` in particular -- is on
        the old axis and has to be regenerated rather than compared against.
        """
        if ratio <= 0:
            raise ValueError("ratio must be > 0")
        if balance <= 0:
            raise ValueError("balance must be > 0")
        root = math.sqrt(ratio) / math.sqrt(balance)
        return replace(self, idle_time_scalar=root, distance_scalar=1.0 / root)


@dataclass(frozen=True)
class Calibration:
    """What `calibrate` decided, and what it saw on the way."""

    alpha: float
    repetitions: int
    pilot_selection_rate: float
    #: True when a pilot failed and the settings' floors were used instead.
    fell_back: bool = False


def ensure_measurements(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """
    Return a circuit that measures, deep-copying only when one has to be added.

    A circuit with no measurements produces an empty ideal reference, against which
    every post-selected distribution scores TVD 1.0. That is not an error anywhere in
    the pipeline, so it arrives as a plausible-looking column of ones.
    """
    if circuit.num_clbits > 0 and any(op.operation.name == "measure" for op in circuit.data):
        return circuit
    measured = copy.deepcopy(circuit)
    measured.measure_all()
    return measured


def naive_packing(num_qubits: int, k: int) -> Perm:
    """Fill patches in program order. The baseline every figure measures against."""
    return [(index // k, index % k) for index in range(num_qubits)]


def build_config(
    settings: SimSettings,
    *,
    n_code: int,
    alpha: float,
    repetitions: int,
    h_commutation: bool = True,
    merging: bool = True,
) -> PipelineConfig:
    sim = ScaledSimulator(alpha=alpha, repetitions=repetitions, seed=settings.seed)
    # `ScaledSimulator` returns a `SimulatorConfig`, not a `Simulator`, so this is a field
    # assignment rather than the fluent `with_shot_progress`. Both set the same flag.
    sim.show_shot_progress = settings.shot_progress
    if settings.stim:
        sim = sim.with_stim()
    if settings.distance_noise:
        sim = sim.with_distance_noise(distance_scalar=settings.distance_scalar)
    if settings.idle_noise:
        sim = sim.with_idle_noise(idle_time_scalar=settings.idle_time_scalar)
    config = PipelineConfig(verbose=False, save=False, sim_config=sim)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    config.translator_config.asap_order = settings.asap_order
    config.translator_config.transversal_intra_two_qubit = settings.transversal_merging and merging
    return config


#: What `evaluate` reports, as `{returned name: scan dataframe column}`.
#:
#: The gate and depth entries are the *translated* circuit's, measured by
#: `result_to_row` on the same object the simulator ran, so they come free with the
#: shots rather than needing a second translation. A figure comparing placements on
#: both outcome and mechanism therefore pays for one simulation per placement, not
#: one simulation plus a translation that might have been configured differently.
METRICS: Dict[str, str] = {
    "lsr": "logical_selection_rate",
    "tvd": "logical_total_variation_distance",
    "one_qubit_gates": "one_qubit_gates",
    "two_qubit_gates": "two_qubit_gates",
    "circuit_depth": "circuit_depth",
}


def evaluate(
    circuit: qiskit.QuantumCircuit, permutation: Perm, config: PipelineConfig, *,
    frames: int = 0, frame_seed: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Simulate one placement and return `METRICS`, or None if it failed.

    `ensure_measurements` runs here rather than being left to callers: the no-measurement
    failure is silent and returns a number, so a figure that forgot would plot a flat TVD
    of 1.0 and look merely uninteresting. The deep copy it may make is negligible next to
    the shots below.

    `frames > 0` also reports the silent logical error per kept shot (`silent_per_kept`,
    with `p_silent` per shot and the frames' own `frame_lsr`): a Pauli-frame run of that many
    shots on the noisy Stim circuit the simulation itself converted, so it costs no second
    conversion (`exp8_undetected/frames.py`). It is the accuracy measure that works at any
    width -- TVD ~ 0.95 x silent per kept, rank correlation 0.91-0.98 -- where TVD saturates
    above ~18 qubits. NaN when the backend is not Stim.
    """
    captured: List = []
    try:
        if frames:
            from experiments.figures.density_ablation_sweep.density_references import (
                _capture_translations,
            )
            from experiments.revision.exp8_undetected.frames import capture_noisy_stim

            with capture_noisy_stim() as captured, _capture_translations() as translations:
                record = run_specific_scan(
                    program_circuit=ensure_measurements(circuit),
                    permutations=[permutation],
                    config=config,
                )
        else:
            record = run_specific_scan(
                program_circuit=ensure_measurements(circuit),
                permutations=[permutation],
                config=config,
            )
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    if record.dataframe is None or record.dataframe.df.empty:
        return None
    row = record.dataframe.df.iloc[0]
    out = {name: float(row[column]) for name, column in METRICS.items()}
    if frames:
        out.update({"silent_per_kept": float("nan"), "p_silent": float("nan"),
                    "frame_lsr": float("nan"), "silent_per_kept_occupied": float("nan")})
        if captured:
            from experiments.revision.exp8_undetected.frames import split_from_stim

            encoded, noisy = captured[-1]
            split = split_from_stim(encoded, noisy, shots=frames, seed=frame_seed)
            out.update({"silent_per_kept": split["p_silent_given_kept"],
                        "p_silent": split["p_silent"], "frame_lsr": 1.0 - split["p_detected"]})
            if translations:
                # The same frames counted on the lanes that hold a program qubit only: at k > 2
                # with fewer qubits than slots, a flip on an empty lane is no logical error of
                # the program (`density_references.split_by_lanes`).
                from experiments.figures.density_ablation_sweep.density_references import (
                    split_by_lanes,
                )

                occupied = {tuple(p) for p in translations[-1].context.state_positions}
                lanes = split_by_lanes(encoded, noisy, occupied, shots=frames, seed=frame_seed)
                out["silent_per_kept_occupied"] = lanes["silent_used"]
    return out


#: A miscompile is judged *differentially*, against the naive arm of the same circuit,
#: never against a fixed number.
#:
#: An absolute threshold cannot work, because the noiseless TVD floor scales with the
#: circuit's output support: 8k shots over an 18-qubit distribution leaves roughly
#: `1 - N/S` on the table, and `qft_n18` duly returns 0.97 for *both* placements with
#: nothing wrong. A fixed 0.10 cutoff flagged it, and `ising_n10` at 0.19, as broken --
#: nine good pairs discarded for being wide. The two arms of a pair sample the same
#: distribution with the same shots, so they share that floor exactly, and a real
#: miscompilation shows up as one arm departing from the other. `qpe_n9` at k = 2 is
#: naive 0.01 against AQORN 0.875 and fails on any sane margin.
CORRECTNESS_RATIO = 3.0
CORRECTNESS_MARGIN = 0.10
CORRECTNESS_REPETITIONS = 8_000

#: Above this, the noiseless reference is itself unresolved at `CORRECTNESS_REPETITIONS`,
#: so TVD carries no information for that circuit -- but the gate counts and depth are
#: exact translation properties and the selection rate is a binomial over ~30k shots, so
#: those stay perfectly good. Such rows are kept and reported, not dropped.
TVD_RESOLUTION_LIMIT = 0.15


def noiseless_tvd(
    circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    settings: SimSettings,
    *,
    n_code: int,
    h_commutation: bool,
    merging: bool = True,
) -> float:
    """
    Simulate at zero noise and return the TVD. A correct compile returns its sampling floor.

    **The pipeline can silently miscompile, and nothing else catches it.** On `qpe_n9` at
    k = 2 the combination (AQORN placement, merging on, H commutation off) returns a
    noiseless TVD of 0.87 against naive's 0.01 -- the circuit computes the wrong
    distribution -- while the selection rate stays at 1.0, so post-selection does not flag
    it. `plan_packing(verify=True)` does not either: it compares cost, not correctness.
    Stim and Aer agree on the wrong answer, so it is translation rather than simulation.

    Without this a miscompiled point enters a figure as a very large apparent regression,
    and quieter instances of the same fault add noise nobody can account for. On the
    QASMBench corpus, 22% of AQORN rows run that exact configuration.

    Lives here rather than in one figure because it was in one figure -- and 8 of the 10
    that pass a noise profile had no guard at all, including `code_size_crossover`,
    `lsr_tvd_vs_k` and `1_best_k_phase_diagram`. `improvement_vs_benchmark_locality`'s
    docstring had said for some time that it belonged here.

    Takes a `SimSettings` rather than any figure's own settings object, so every caller can
    reach it. Pair with `miscompiled` to do the comparing.
    """
    config = build_config(
        replace(settings, distance_noise=False, idle_noise=False),
        n_code=n_code, alpha=0.0, repetitions=CORRECTNESS_REPETITIONS,
        h_commutation=h_commutation, merging=merging,
    )
    result = evaluate(circuit, permutation, config)
    return float("nan") if result is None else float(result["tvd"])


def miscompiled(naive_clean: float, aqorn_clean: float) -> bool:
    """
    Did the AQORN arm depart from naive's noiseless floor? True means discard the pair.

    Differential, per the constants above: a departure is `CORRECTNESS_RATIO` times the
    naive floor *and* `CORRECTNESS_MARGIN` above it, so neither a wide circuit's high
    shared floor nor a tiny absolute wobble trips it. NaN on either side is not a
    miscompile -- it is a rejected placement, which callers already handle.
    """
    if naive_clean != naive_clean or aqorn_clean != aqorn_clean:
        return False
    return (
        aqorn_clean > CORRECTNESS_RATIO * naive_clean
        and aqorn_clean > naive_clean + CORRECTNESS_MARGIN
    )


def check_compile(
    circuit: qiskit.QuantumCircuit,
    naive: Perm,
    aqorn: Perm,
    settings: SimSettings,
    *,
    n_code: int,
    h_commutation: bool,
    merging: bool = True,
) -> Optional[str]:
    """
    `None` if the AQORN arm compiles correctly, else a one-line reason to skip the point.

    The whole guard in one call, for figures whose loop has no natural place to carry two
    noiseless columns to an aggregation step. Two zero-noise simulations at
    `CORRECTNESS_REPETITIONS` -- against the two pilots and two measured runs a point
    already pays, roughly 4% overhead, which is the correct price for not publishing a
    circuit that computes the wrong distribution.

    Prefer recording `noiseless_tvd` per arm in the CSV where the loop allows it, as
    `improvement_vs_benchmark_locality` does: that keeps the judgement re-derivable from the
    data instead of baked in at measurement time. Use this where it does not.

    Skip, do not fall back to naive. A miscompiled pair is a missing measurement, and
    silently substituting the baseline for the arm under test would pull every affected
    cell towards zero improvement and read as a result.
    """
    shared = dict(n_code=n_code, h_commutation=h_commutation, merging=merging)
    clean_naive = noiseless_tvd(circuit, naive, settings, **shared)
    clean_aqorn = noiseless_tvd(circuit, aqorn, settings, **shared)
    if miscompiled(clean_naive, clean_aqorn):
        return (
            f"miscompiled: noiseless TVD {clean_aqorn:.3f} against naive's "
            f"{clean_naive:.3f} (merging={merging}, h_commutation={h_commutation})"
        )
    return None


def tvd_resolved(naive_clean: float) -> bool:
    """
    Is TVD meaningful for this circuit, or is its noiseless floor already saturated?

    Unresolved rows keep their gate counts, depth and selection rate -- those are exact
    translation properties and a binomial -- and are withheld from TVD panels only.
    """
    return naive_clean == naive_clean and naive_clean <= TVD_RESOLUTION_LIMIT


def baseline_gate_count(
    circuit: qiskit.QuantumCircuit, permutation: Perm, n_code: int, *,
    asap_order: bool = False,
) -> int:
    """
    Weighted-free gate total of the baseline translation, which sets the initial alpha.

    `asap_order` layer-sorts it, as `SimSettings.asap_order` does for the arms.

    Calibration starts from `alpha ~ 1/m` because selection rate falls exponentially in
    the number of noisy operations, so the gate count is the right first guess at what
    noise scale puts this circuit in a measurable regime.

    Lives here rather than in each figure because `calibrate` requires it and three
    scripts had already grown byte-identical copies.
    """
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = False
    config.translator_config.enable_merging = False
    config.translator_config.asap_order = asap_order
    try:
        translated = translate_permutation(circuit, permutation, config=config)
    except Exception:  # noqa: BLE001
        return circuit.size()
    one_q, two_q = one_two_qubit_gate_totals(translated.circuit)
    return one_q + two_q


def calibrate(
    circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    settings: SimSettings,
    *,
    n_code: int,
    baseline_gate_count: int,
) -> Calibration:
    """
    Pick a noise scale and shot count that put this circuit in a measurable regime.

    Two pilots: one to see where the default alpha lands, one to confirm the correction.
    Selection rate falls exponentially in noise, so the correction is solved in log
    space -- `alpha * log(target) / log(observed)` rather than a linear rescale.

    Both pilots run the *baseline* configuration, passes off. Calibrating on the
    configuration under test would move the noise scale with the thing being measured,
    and every improvement would be partly the calibration chasing it.
    """
    alpha = settings.alpha_reference * (
        settings.reference_gate_count / max(1.0, float(baseline_gate_count))
    )
    pilot = evaluate(
        circuit,
        permutation,
        build_config(
            settings, n_code=n_code, alpha=alpha,
            repetitions=settings.pilot_repetitions,
            h_commutation=False, merging=False,
        ),
    )
    if pilot is None:
        return Calibration(alpha, settings.min_repetitions, float("nan"), fell_back=True)

    observed = float(np.clip(pilot["lsr"], 1e-6, 1 - 1e-6))
    target = float(np.clip(settings.target_selection_rate, 1e-6, 1 - 1e-6))
    alpha = alpha * math.log(target) / math.log(observed)

    confirm = evaluate(
        circuit,
        permutation,
        build_config(
            settings, n_code=n_code, alpha=alpha,
            repetitions=settings.pilot_repetitions,
            h_commutation=False, merging=False,
        ),
    )
    rate = confirm["lsr"] if confirm else settings.target_selection_rate
    repetitions = int(
        np.clip(
            math.ceil(settings.target_kept_shots / max(rate, 1e-4)),
            settings.min_repetitions,
            settings.max_repetitions,
        )
    )
    return Calibration(alpha, repetitions, float(rate), fell_back=confirm is None)


def add_sim_arguments(parser, defaults: SimSettings = SimSettings()) -> None:
    """The shot-budget flags, so every simulated figure spells them the same way."""
    parser.add_argument(
        "--target-selection-rate", type=float, default=defaults.target_selection_rate,
        help="Baseline selection rate the per-circuit alpha calibration aims for.",
    )
    parser.add_argument(
        "--pilot-repetitions", type=int, default=defaults.pilot_repetitions,
        help="Shots per calibration pilot. Two pilots run per circuit.",
    )
    parser.add_argument(
        "--target-kept-shots", type=int, default=defaults.target_kept_shots,
        help="Post-selected shots wanted per point. Repetitions follow from this and "
             "the calibrated selection rate, so wide and narrow circuits get "
             "comparable error bars.",
    )
    parser.add_argument("--min-repetitions", type=int, default=defaults.min_repetitions)
    parser.add_argument("--max-repetitions", type=int, default=defaults.max_repetitions)
    parser.add_argument(
        "--backend", choices=["aer", "stim"],
        default="stim" if defaults.stim else "aer",
        help="stim (default) is ~128x faster and agrees with aer on the logical "
             "distribution at every code size. aer is the slower reference; prefer it "
             "when the y axis is selection rate under appreciable noise, where the two "
             "backends still build different channels for the same alpha.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Simulator seed. Circuit generation is seeded separately and always.",
    )


def settings_from_args(args, base: SimSettings = SimSettings()) -> SimSettings:
    """Fold `add_sim_arguments` results back into a `SimSettings`."""
    return replace(
        base,
        target_selection_rate=args.target_selection_rate,
        pilot_repetitions=args.pilot_repetitions,
        target_kept_shots=args.target_kept_shots,
        min_repetitions=args.min_repetitions,
        max_repetitions=args.max_repetitions,
        stim=args.backend == "stim",
        seed=args.seed,
    )


def knobs_for(settings: "SimSettings", movement_error: float = 1e-3):
    """
    `NoiseKnobs` carrying the same idle/distance balance the simulator will charge.

    Moved here from `code_size_crossover` when that figure was archived. It is a
    `SimSettings` -> `NoiseKnobs` adapter and belongs to neither figure in particular;
    `resource_selection_headroom` imported it across the figure boundary, which is what
    made archiving the other one break a keeper.

    `SimSettings.with_noise_ratio` moves two scalars: `idle_time_scalar` multiplies the
    dephasing a qubit accrues sitting out a layer, and `distance_scalar` scales the
    movement channel -- in the simulator it slows the atoms *and* multiplies `move_error`,
    so the selector mirrors both. `layer_time` carries the idle scalar. This gives the
    resource selector the same regime the measurement is taken in; anything else and the
    selector is being asked about a machine that was never simulated.

    `movement_error` scales `move_error` relative to its hardware baseline, so the CLI axis
    stays meaningful -- the figures' default 3.5e-3 maps to 3.5x the baseline handling
    error. Defaults to the baseline itself, which is a no-op scale.
    """
    from packing.resource.noise import NoiseKnobs

    base = NoiseKnobs()
    handling_scale = movement_error / 1e-3
    speed = base.movement_speed
    if settings.distance_scalar > 0:
        speed = base.movement_speed / settings.distance_scalar
    return replace(
        base,
        layer_time=base.layer_time * settings.idle_time_scalar,
        move_error=min(1.0, base.move_error * handling_scale * settings.distance_scalar),
        movement_speed=speed,
    )


def noise_profile_for(settings: SimSettings, *, movement_scale: float = 1.0):
    """
    A `packing.NoiseProfile` matching what this `SimSettings` will actually charge.

    Bug B6: `plan_packing` accepts a noise profile but every cost in it defaults to zero,
    so unless a caller passes one the planner optimises weighted gates plus depth with
    movement and idling priced at *nothing* -- while the simulator, with
    `distance_noise` and `idle_noise` on by default, charges both. The planner was
    answering a different question from the one being measured.

    The conversion is grounded in `packing.resource.noise`, whose whole content is the
    expected number of detectable errors in a shot:

        E = n1*p1 + n2*p2 + V*(t_layer/T2),   V = patches * (n + 2) * depth

    so one qubit idling through one layer costs `t_layer/T2` where a two-qubit gate costs
    `p2`. `DEFAULT_WORK_WEIGHTS` prices a two-qubit gate at 13 units, which fixes the
    exchange rate: idling one qubit for one layer is worth `13 * (t_layer/T2) / p2` units.
    The simulator's `idle_time_scalar` then scales it, because that is exactly what that
    scalar does to the physical error.

    `movement_cost` is now derived the same way `idle_cost` always was, from the
    grouped-transport charges in `packing.resource.noise`: one crossing is one round
    trip of two atoms, paying `move_error` each plus the idling the flight forces on
    the array. Two conversion choices are pinned to the reference the rest of the
    constants were calibrated on -- a k=2, 12-logical-qubit workload (48 physical
    qubits over 8 patches, mean separation ~4.6 cells) -- because the profile is built
    before a candidate's patch count exists: the array size for the forced-idle term,
    and the mean distance the distance-free handling is amortised over. Probabilities
    convert to weighted-gate units at `w2 / p2`, one two-qubit gate's worth of error
    per 13 units.

    `distance_scalar` enters **once**, through `movement_cost`. It used to be applied
    twice -- once here and once as `distance_scale`, which `packing.cost.price`
    multiplies against the hop count -- so the movement term scaled quadratically in
    the knob that claims to be linear. `distance_scale` is now 1.0 whenever distance
    noise is on (true cell distances), 0.0 when it is off.

    `movement_scale` stays as an extra knob on top for sweeps that want one.

    The arithmetic itself now lives in `packing.resource.noise.packing_profile`, which is
    where it belongs: it is a statement about the machine, not about a sweep, and
    `packing.compiler` needs the same conversion to hand the planner a profile matching
    the knobs the selector ranked on. This function is the `SimSettings` adapter over it
    -- which flags map to which scale -- and nothing more.
    """
    from packing.resource.noise import NoiseKnobs, packing_profile

    return packing_profile(
        NoiseKnobs(),
        movement_scale=movement_scale * settings.distance_scalar,
        idle_scale=settings.idle_time_scalar if settings.idle_noise else 0.0,
        distance=settings.distance_noise,
    )
