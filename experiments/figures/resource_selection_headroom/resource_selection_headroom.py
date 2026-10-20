"""
resource_selection_headroom.py -- where AQORN sits between naive fixed-k and the ceiling.

**Claim.** Against a static code size compiled naively, AQORN's resource selector keeps
most of the shots that are available to keep. The figure is a ladder: the naive static
baseline at the bottom, a per-circuit oracle at the top, and AQORN's two stages in
between, so the reader can see both how much of the headroom the selector captures and
which stage of the compiler captured it.

Five levels, all plotted as shots kept relative to the same reference:

    worst static k, naive     -- what picking the wrong constant costs
    static k=2, naive         -- the reference, the 1.0 rule, the paper's code
    AQORN, static k=2         -- the planner alone: placement search, no resource choice
    AQORN selector            -- the full resource decision, code size and patch count
    oracle                    -- the best of the selector's own candidates, by ground truth

**This figure exists because `code_size_crossover` cannot answer the question.** That
figure's upper line is a max over five `plan_packing` arms while its AQORN line is
`select_resources`, a different code path with a stronger placement search and its own
commutation decision. The two are not comparable at fixed resources, so the selector
exceeds the "oracle" there -- by 1.24x on the cached run -- and the excess is not a
resource-selection result at all. Decomposed on that data the selector is *worse* than
its own oracle at picking a code size (0.81x) and ahead only on placement (1.53x). The
line is honestly labelled "best *fixed* k per point" and the docstring says what it is,
but a reader reasonably takes a topmost line as a ceiling, and that one is not.

So the oracle here is built to actually be one.

**The oracle is the selector's own decision, resolved by measurement instead of by the
model.** `select_resources` enumerates a grid of (code size, patch count) candidates,
places each one, scores them all on `packing.resource.noise.estimate`, and returns the
winner. Its `frontier` carries every candidate it considered. This figure simulates a
covering subset of that frontier and takes the best *measured* selection rate. Same
candidate generator, same placements, same translation -- the only thing replaced is the
ranking, which goes from the noise model's prediction to ground truth.

That makes the gap mean one specific thing: **how much the selector loses by ranking its
candidates with a model rather than knowing the answer.** It is a true ceiling on the
selector as built, because the selector's returned plan is itself included in the max
(`_oracle_set`), so the oracle dominates it row by row by construction. What it is *not*
is a bound over all possible compilers -- a better candidate generator could beat it, and
that is the honest reading of any headroom this figure leaves unclaimed.

**One commutation rule everywhere, which is the fairness fix.** Every arm at a given code
size is translated under `commutation_pays(circuit, n_code)` -- the predicate
`select_resources` already uses internally (`select.py:228`). `plan_packing` is passed
that decision explicitly rather than its usual `h_commutation="auto"`, which plans both
ways and keeps the better one. That costs the planner arm a little realism and buys the
comparison its meaning: on `code_size_crossover`'s cached data the selector arm and its
matched fixed-k arm disagreed about commutation in 23% of cells, and those cells showed a
1.94x gap against 1.42x where they agreed. A ladder whose rungs are translated
differently is measuring the translator.

**The baseline is naive packing at a static code size**, `naive_packing` -- program order
into patches, no planner, no search. `worst static k` is the worst of the swept code
sizes at that point, which is what picking a constant badly costs; the reference is k=2,
the paper's code and the size the calibration is solved at. Both are drawn because "a
static k" is ambiguous between them and the difference is the reader's risk, not ours.

**One noise calibration per (circuit, ratio), at the reference.** Inherited from
`code_size_crossover` along with its reasoning: an alpha per arm would pin selection rate
to a constant along the axis the comparison lives on and flatten every line, while a
single alpha for the whole sweep lets total noise vary 5x and puts both ends outside the
band where anything can separate. Solving at the naive k=2 reference per column puts each
column in the discriminating band and leaves the within-column comparison untouched.

**Fixed raw shots.** `--repetitions` is a raw budget, not scaled by the calibrated rate,
because the shots post-selection discards are the quantity plotted. Selection rate is a
binomial proportion and stays precise where acceptance is low.

**The oracle's max is over noisy estimates, so it is biased upward** -- the winner's curse.
The bias is small here and worth stating rather than correcting: at lsr ~ 0.1 on 40k raw
shots the standard error is ~1.5% relative, against candidate separations of 10-50%. It
inflates the ceiling slightly, which makes the captured-headroom number this figure prints
slightly *conservative*.

**The oracle is capped by default, and the cap is announced.** The frontier runs to ~25
candidates per circuit and simulating all of them is roughly three times the cost of this
whole sweep. `--oracle-top` takes the model's best candidate at *each* code size -- so the
oracle spans the same k range as the naive baseline -- plus its top few overall, plus the
selector's own plan. Taking the model's top N does bias the oracle toward what the model
already likes, which understates the ceiling wherever the model is wrong; the per-code-size
covering set is there to limit that, and `--oracle-all` removes the cap. Every run prints
how many candidates were dropped.

Log-y, which is the style guide's stated exception: these are ratios, equal factors up and
down should read as equal distances, and 1.0 is the meaningful centre. Bands are +-1
standard *error* over circuits, not the house default of +-1 SD, because every series is a
ratio taken within a circuit -- the circuit is already cancelled, so the SD across
benchmarks would describe how much they differ from each other rather than the uncertainty
on the plotted mean. The caption has to say so. Spread is taken in log space.

**`--standard`: the same ladder under the standard configuration**
(`packing.standard.STANDARD`, since 2026-10-07). Opt-in: without the flag none of this
applies and the script is the one that drew the submitted figure. The rungs become

    naive_k{2..10}   the same placements and the same rule, `commutation_pays` -- asked of
                     the gadget set the arm is translated with, which its cache keys on
    aqorn_fixed      `plan_standard` at the reference code size, `h_commutation="auto"`,
                     simulated under the plan's own commutation decision
    selector         `compile_circuit(planner="standard")`, the standard resource selector:
                     every code size planned with `plan_standard`, planned and naive placement
                     priced on `standard_objective`, the lowest kept; simulated under its own
                     commutation decision (`select_resources` and its frontier are not used)
    oracle           every arm `compile_circuit` priced -- planned and naive at each code
                     size, at most ten -- each at its own code size and commutation setting,
                     so `--oracle-top` / `--oracle-all` do not apply: there is no cap

Every translation -- arms, pilots, the miscompile guard, the regime that sets x -- is
layer-sorted, and merging arms build in-patch CZ / XCX transversally at k = 2
(`SimSettings.with_standard`); alpha is therefore recalibrated on the layer-sorted
baseline. Commutation is no longer one function of the code size, so the guard compares an
arm against naive at the same code size *and* commutation setting. An arm whose (placement,
code size, commutation) was already simulated in its column reuses that measurement
(`duplicate_of` names it) instead of drawing a second sample: identical configurations then
compare at exactly 1.0, and the oracle has no extra draws to be lucky with. Each candidate
row carries its model value (`objective_total` = detected + `objective_silent_weight` x
silent, `objective_detected`, `objective_silent`, best first by `candidate_rank`), so the
oracle can be re-taken post hoc on shots, on silent error or on the objective. The weights
are read from `STANDARD` when the run starts and recorded on every row
(`planner_silent_weight` is `silent_weight_for(n_code)`), since they are still being tuned
and a CSV must say which standard it measured. Every simulated arm also
runs `--frames` Pauli-frame shots on the circuit the simulation already converted
(`evaluate(frames=...)`): `silent_per_kept` is the accuracy measure that works at these
widths, where TVD does not, and is drawn as a second panel on the same levels -- relative to
naive [[4,2,2]], lower is better, the oracle the candidates' minimum. Nothing is replaced:
the CSV is `resource_selection_headroom_standard[_vN].csv` beside this script, `--persist`
copies it to results/data/ under a free name, and the figure takes the CSV's stem through
`save_figure(versioned=True)`. `--from-csv` on a standard CSV switches to the standard plot
by itself, so a replot cannot land on the submitted figure.

Run with:

    python -m experiments.figures.resource_selection_headroom.resource_selection_headroom
    python -m ...resource_selection_headroom --from-csv <path>
    python -m ...resource_selection_headroom --circuits ising_n26 knn_n25 --ratios 1 --jobs 2
    python -m ...resource_selection_headroom --standard --jobs 12 --persist
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import (
    knobs_for,
    miscompiled,
    noiseless_tvd,
    SimSettings,
    add_sim_arguments,
    build_config,
    calibrate,
    evaluate,
    measured_regime,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from experiments.shared.figures import stamp_provenance
from packing.passes import commutation_pays
from packing.plan import plan_packing
from packing.resource.select import DEFAULT_CODE_SIZES, select_resources

# The selector has to be told the same regime the simulator will charge, and
# `code_size_crossover` already works out that mapping -- two scalars into `NoiseKnobs`,
# with `distance_scalar` hitting both the atom speed and `move_error`. Imported rather
# than copied so the two figures cannot drift into telling the selector about different
# machines; that would be invisible in both and would make them silently incomparable.

NAME = "resource_selection_headroom"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: `--standard`'s output stem, CSV and figure alike. Never `NAME`: that figure is the
#: submitted one, and the SVG beside this script under it has been finished by hand.
STANDARD_NAME = f"{NAME}_standard"

Perm = List[Tuple[int, int]]

#: One measured arm: `(series name, n_code, placement)`. Commutation is not carried
#: because it is a pure function of `n_code` here -- that is the point of the uniform rule.
Arm = Tuple[str, int, Perm]

#: Five points on the ratio axis rather than `code_size_crossover`'s nine. This figure
#: pays for a whole oracle set per column, so the axis is thinned to the powers of
#: sixteen -- which are exactly the ticks the sibling figure labels, so the two read
#: against each other -- and the saving goes into candidate coverage instead. The
#: question here is where AQORN sits, not where a crossover is, and locating a level
#: needs fewer columns than locating a crossing.
DEFAULT_RATIOS: Tuple[float, ...] = (1 / 256, 1 / 16, 1.0, 16.0, 256.0)

#: The same family as every other k figure in the suite.
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: Program-qubit window. The floor is what gives every code size a distinct patch count;
#: below ~24 the ladder saturates and the static baselines collapse onto each other.
DEFAULT_QUBIT_RANGE: Tuple[int, int] = (24, 40)

#: Program gates above this are dropped. `load_proxy_circuits`' own limit filters the
#: pre-proxy circuit, which lets a 9800-gate benchmark through at 22s per selector call.
DEFAULT_GATE_LIMIT = 1500

#: Model-ranked candidates added to the oracle set beyond the per-code-size covering
#: choice. Three is the knee: the covering set already spans every code size, so these
#: only buy a better patch count within one, and the frontier's model score is monotone
#: enough that the fourth rarely wins.
DEFAULT_ORACLE_TOP = 3

#: Pauli-frame shots per simulated arm under `--standard`. At the calibrated alpha about 6%
#: of them are kept, so the silent rate per kept shot rests on ~6k shots per arm. The frame
#: simulation reuses the conversion the shots already paid for and costs seconds.
DEFAULT_FRAMES = 100_000

#: Series keys that are not per-k naive arms.
SELECTOR = "selector"
ORACLE_CANDIDATE = "oracle_cand"
AQORN_FIXED = "aqorn_fixed"

#: Panel lines, in draw order: `(key, legend label, palette slot)`.
#:
#: The selector takes slot 0 -- blue, circle, solid -- which is the AQORN series' position
#: in every figure in this suite. Slots are consecutive rather than skipping slot 3, for
#: the reason `code_size_crossover` documents: the style's linestyle cycler repeats solid
#: at position 5, so a set like (0, 1, 2, 4) hands two series the same line and leaves only
#: the marker between them in greyscale. Slot 3 is `#636363`, which the style reserves for
#: reference lines and which the 1.0 rule also uses -- fine, because the rule carries no
#: marker and `naive_worst` does.
LINES: Sequence[Tuple[str, str, int]] = (
    ("oracle", "oracle over AQORN's candidates", 1),
    (SELECTOR, "AQORN selector", 0),
    (AQORN_FIXED, "AQORN, static $k=2$", 2),
    ("naive_worst", "worst static $k$, naive", 3),
)

#: Terminal-safe names for the same series. The legend labels carry mathtext (`$k=2$`),
#: which renders in the figure and reads as noise in a printed table.
_PRINT_LABELS: Dict[str, str] = {
    "oracle": "oracle",
    SELECTOR: "AQORN selector",
    AQORN_FIXED: "AQORN, static k=2",
    "naive_worst": "worst static k, naive",
}

#: How a code size is named in a label -- `docs/notation.md`'s form, matching the prose.
#:
#: Plain text, not `$[[{n},{k},2]]$`. Under the shared style's `mathtext.fontset: cm` with
#: `svg.fonttype: none`, matplotlib writes one `<tspan>` per glyph whose character is the
#: glyph's *position in the Computer Modern font*. Letters and digits sit at their own
#: codepoints and survive; the comma does not -- it lives where `;` does in cmmi10, so this
#: rendered as `[[4;2;2]]` in any SVG viewer without the CM fonts, and extracted that way
#: from the PDF. The PDF *displays* correctly, since `pdf.fonttype: 42` embeds the glyphs,
#: so this only ever hurt Inkscape editing and text extraction -- but the SVG is what gets
#: hand-finished. See `9_code_size_tradeoff` for the same bug with ceiling brackets.
_CODE = "[[{n},{k},2]]"

#: Opaque white behind the legend, no edge drawn. The house style asks for frameless
#: legends and this still reads as one; the backing exists because the 1.0 rule spans the
#: full axis by construction and would otherwise print through the legend text.
_LEGEND_BACKING = {
    "frameon": True,
    "framealpha": 1.0,
    "facecolor": "white",
    "edgecolor": "none",
}


@dataclass
class Settings:
    ratios: Sequence[float] = DEFAULT_RATIOS
    k_values: Sequence[int] = DEFAULT_K_VALUES
    circuits: Optional[Sequence[str]] = None
    qubit_range: Tuple[int, int] = DEFAULT_QUBIT_RANGE
    gate_limit: int = DEFAULT_GATE_LIMIT
    #: Fixed *raw* shot budget per point -- see the module docstring.
    repetitions: int = 40_000
    #: Code size the alpha calibration is solved at, and the 1.0 reference. k=2 naive is
    #: the baseline the whole ladder is measured against.
    reference_k: int = 2
    #: Model-ranked oracle candidates beyond the per-code-size covering set.
    oracle_top: int = DEFAULT_ORACLE_TOP
    #: Simulate the whole frontier. Roughly 3x the cost; removes the cap's bias.
    oracle_all: bool = False
    #: What the selector charges per crossing per hop. Uncalibrated -- `noise_profile_for`
    #: calls the exchange rate underivable. One two-qubit gate error by default.
    movement_error: float = 3.5e-3
    benchmark_root: Path = field(default_factory=lambda: Path("benchmarks/qasmbench"))
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None
    #: The standard configuration -- see the module docstring's `--standard` section.
    standard: bool = False
    #: Pauli-frame shots per simulated arm, `--standard` only. 0 skips them.
    frames: int = DEFAULT_FRAMES


def _load_named(
    root: Path, wanted: Sequence[str]
) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """
    Proxy only the benchmarks actually asked for, resolving names before doing any work.

    The obvious spelling -- `load_proxy_circuits(root, qubit_limit=None)` then filter by
    name -- is what a named run used to do, and it is close to unusable. Dropping the
    width limit so an explicitly requested circuit is honoured whatever its size also
    means every one of the ~132 QASMBench files gets parsed, transpiled and rebuilt gate
    by gate first. `create_proxy_from_qasm`'s one-second timeout does not save it either:
    the timeout wraps the parse and transpile, while the proxy reconstruction loop after
    it is unbounded, so a single large benchmark stalls the run before the sweep starts.
    A `--circuits ising_n26 knn_n25` smoke test would sit on that for minutes.

    Names are cheap to resolve -- they come from a README line, or the file stem -- so
    resolve first and proxy the two files that matched. The naming rule is duplicated
    from `load_proxy_circuits` rather than inferred, so the two cannot disagree about
    what a benchmark is called.
    """
    from experiments.shared.benchmark_circuits import (
        _read_circuit_name,
        create_proxy_from_qasm,
        find_qasm_files,
    )

    targets = set(wanted)
    chosen: List[Tuple[str, qiskit.QuantumCircuit]] = []
    failed: List[str] = []
    for path in find_qasm_files(root):
        name = _read_circuit_name(path.parent / "README.md") or path.stem
        if name not in targets:
            continue
        proxy = create_proxy_from_qasm(path)
        if proxy is None:
            failed.append(name)  # non-Clifford gate, parse error, or the 1s timeout
            continue
        chosen.append((name, proxy))

    missing = targets - {name for name, _ in chosen} - set(failed)
    if missing:
        raise ValueError(f"no such benchmark circuit: {sorted(missing)}")
    if failed:
        raise ValueError(
            f"named circuit(s) could not be proxied: {sorted(failed)} -- "
            f"`create_proxy_from_qasm` refused them (non-Clifford gate, parse error, or "
            f"its one-second transpile timeout)"
        )
    return chosen


def load_circuits(settings: Settings) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """The benchmark set, filtered to the width window that gives every k a patch count."""
    root = settings.benchmark_root
    if not root.exists():
        raise FileNotFoundError(
            f"benchmark root {root} not found. Run from the repository root, pass "
            f"--benchmark-root, or initialise the submodule with "
            f"`git submodule update --init --recursive`."
        )
    # An explicitly named circuit is honoured whatever its width, so a smoke test is not
    # silently filtered out by the default window -- but only its own file is proxied.
    if settings.circuits:
        return _load_named(root, settings.circuits)

    low, high = settings.qubit_range
    loaded = load_proxy_circuits(root, qubit_limit=high)
    return [
        (name, circuit)
        for name, circuit in loaded
        if low <= circuit.num_qubits <= high and circuit.size() <= settings.gate_limit
    ]


def _oracle_set(plan, settings: Settings) -> Tuple[List[Tuple[int, Perm]], int]:
    """
    Candidates to simulate for the ceiling, and how many were dropped.

    Three parts, unioned and deduplicated:

    * the model's best candidate at **each** code size, so the oracle spans the same k
      range the naive baseline does and the two ends of the ladder are comparable;
    * the model's top `oracle_top` overall, which is where a better patch count within
      one code size gets found;
    * the selector's own returned plan, which is what makes the result a true ceiling --
      without it the max is over the frontier's greedy placements, and the plan the
      selector actually returns may be the descended or naive arm that
      `_refine_and_verify` substituted, which is not in the frontier at all.

    The frontier arrives sorted by model score with the exactly-translated candidates
    first, so "top N" is a prefix. Ranking by the model is the cap's known bias: it
    concentrates the oracle where the model is already confident and therefore understates
    the ceiling wherever the model is wrong. The per-code-size covering set exists to
    bound that, and `--oracle-all` removes it entirely.
    """
    seen = set()
    chosen: List[Tuple[int, Perm]] = []

    def add(n_code: int, placement: Perm) -> bool:
        key = (n_code, tuple(placement))
        if key in seen:
            return False
        seen.add(key)
        chosen.append((n_code, list(placement)))
        return True

    # The returned plan first, so it survives any cap.
    add(plan.n_code, plan.placement)

    frontier = list(plan.frontier)
    if settings.oracle_all:
        for candidate in frontier:
            add(candidate.n_code, candidate.placement)
        return chosen, 0

    best_at: Dict[int, object] = {}
    for candidate in frontier:  # already best-score-first, so first wins
        best_at.setdefault(candidate.n_code, candidate)
    for candidate in best_at.values():
        add(candidate.n_code, candidate.placement)
    for candidate in frontier[: max(0, settings.oracle_top)]:
        add(candidate.n_code, candidate.placement)

    considered = {(c.n_code, tuple(c.placement)) for c in frontier}
    return chosen, len(considered - seen)


def compute(settings: Settings) -> pd.DataFrame:
    circuits = load_circuits(settings)
    if not circuits:
        raise RuntimeError(
            f"no benchmark circuit has {settings.qubit_range[0]}-"
            f"{settings.qubit_range[1]} qubits and <= {settings.gate_limit} gates"
        )
    cells = len(circuits) * len(settings.ratios)
    # Naive at every k, the planner arm, the selector, then the oracle set: one covering
    # candidate per code size plus `oracle_top` plus the selector's own plan, before
    # deduplication. An estimate, printed so a long run announces its size up front.
    if settings.standard:
        arms = (f"<= {len(settings.k_values) + 2 + 2 * len(DEFAULT_CODE_SIZES)} (the "
                f"{len(settings.k_values) + 2} arms, then every priced candidate not "
                f"already among them)")
    elif settings.oracle_all:
        arms = "the whole frontier (~25) plus 7"
    else:
        arms = f"~{2 * len(settings.k_values) + settings.oracle_top + 3}"
    print(
        f"[{NAME}] {len(circuits)} circuits x {len(settings.ratios)} ratios = {cells} "
        f"columns, {arms} simulations each, {settings.repetitions:,} raw shots/point"
    )

    jobs = [
        (settings, name, circuit, ratio)
        for name, circuit in circuits
        for ratio in settings.ratios
    ]
    worker = _run_column
    if settings.standard:
        print(f"[{NAME}] standard configuration: every translation layer-sorted, in-patch "
              f"CZ / XCX transversal at k = 2, {settings.frames:,} frame shots per "
              f"simulated arm")
        # Heaviest columns first, so the longest one is not left running alone at the end.
        jobs.sort(key=lambda job: (-job[2].size(), -job[2].num_qubits))
        worker = _run_column_standard
    produced = parallel_map(worker, jobs, jobs=settings.jobs, label="circuit x ratio")
    rows = [row for batch in produced if batch for row in batch]
    dropped = sum(
        row.get("oracle_dropped", 0) for row in rows if row["series"] == SELECTOR
    )
    if dropped:
        print(
            f"[{NAME}] oracle capped: {dropped} frontier candidates left unsimulated "
            f"across the sweep (--oracle-all removes the cap; see `_oracle_set`)"
        )
    return pd.DataFrame(rows)


def _run_column(job) -> List[Dict]:
    """
    One (circuit, ratio) column: calibrate once at the reference, then measure every arm.

    Module-level and self-contained because `parallel_map` spawns rather than forks, so a
    closure cannot cross the process boundary.
    """
    settings, name, circuit, ratio = job
    rows: List[Dict] = []
    reference = settings.reference_k
    sim = settings.sim.with_noise_ratio(ratio)
    profile = noise_profile_for(sim)

    # One commutation decision per code size, shared by every arm at that size. This is
    # the fairness fix -- see the module docstring.
    commutes: Dict[int, bool] = {}

    def commute_for(n_code: int) -> bool:
        if n_code not in commutes:
            commutes[n_code] = commutation_pays(circuit, n_code)
        return commutes[n_code]

    ref_placement = naive_packing(circuit.num_qubits, reference)
    calibration = calibrate(
        circuit,
        ref_placement,
        sim,
        n_code=reference + 2,
        baseline_gate_count=circuit.size() * 3,
    )
    if not math.isfinite(calibration.alpha):
        return rows

    # The x value: what the reference configuration's schedule actually charges at these
    # scalars. Fixed at the calibration reference so the axis is independent of every
    # series, and alpha-invariant so the calibrated alpha cannot move it.
    try:
        regime = measured_regime(
            circuit,
            ref_placement,
            n_code=reference + 2,
            idle_time_scalar=sim.idle_time_scalar,
            distance_scalar=sim.distance_scalar,
        )
    except Exception:  # noqa: BLE001 - a rejected reference costs the column
        return rows

    arms: List[Arm] = [
        (f"naive_k{k}", k + 2, naive_packing(circuit.num_qubits, k))
        for k in settings.k_values
    ]

    # The planner alone, at the reference code size: placement search, no resource choice.
    # Given the commutation decision explicitly rather than "auto", which would plan both
    # ways and keep the better -- a freedom no other arm gets.
    try:
        plan = plan_packing(
            circuit,
            reference + 2,
            h_commutation=commute_for(reference + 2),
            verify=True,
            noise=profile,
        )
        arms.append((AQORN_FIXED, reference + 2, plan.placement))
    except Exception:  # noqa: BLE001 - a refused plan costs one arm, and the circuit
        pass

    # The selector, and the ceiling built from its own frontier.
    dropped = 0
    try:
        resource_plan = select_resources(
            circuit, noise=knobs_for(sim, settings.movement_error)
        )
        arms.append((SELECTOR, resource_plan.n_code, resource_plan.placement))
        candidates, dropped = _oracle_set(resource_plan, settings)
        for n_code, placement in candidates:
            arms.append((ORACLE_CANDIDATE, n_code, placement))
    except Exception:  # noqa: BLE001 - a refused selection costs the column's top two
        pass

    # The pipeline can silently miscompile: on `qpe_n9` at k = 2 the combination (packed
    # placement, merging on, H commutation off) computes the wrong distribution while the
    # selection rate stays at 1.0, so neither post-selection nor `plan_packing(verify=True)`
    # catches it. This figure's whole claim is a selection-rate gap *between arms*, so one
    # miscompiled arm would manufacture headroom. `noiseless_tvd` and `miscompiled` are used
    # directly rather than `check_compile` so the naive reference is computed once per code
    # size and shared across the arms that need it.
    clean_reference: Dict[int, float] = {}

    for series, n_code, placement in arms:
        commute = commute_for(n_code)
        if not series.startswith("naive_"):
            if n_code not in clean_reference:
                clean_reference[n_code] = noiseless_tvd(
                    circuit, naive_packing(circuit.num_qubits, n_code - 2), sim,
                    n_code=n_code, h_commutation=commute,
                )
            clean_arm = noiseless_tvd(
                circuit, placement, sim, n_code=n_code, h_commutation=commute,
            )
            if miscompiled(clean_reference[n_code], clean_arm):
                print(f"[{NAME}] skipping {name} {series} n_code={n_code}: "
                      f"miscompiled, noiseless TVD {clean_arm:.3f} against naive's "
                      f"{clean_reference[n_code]:.3f}")
                continue
        config = build_config(
            sim,
            n_code=n_code,
            alpha=calibration.alpha,
            repetitions=settings.repetitions,
            h_commutation=commute,
            merging=True,
        )
        measured = evaluate(circuit, placement, config)
        if measured is None:
            continue
        rows.append(
            {
                "circuit": name,
                "program_qubits": circuit.num_qubits,
                "program_gates": circuit.size(),
                "ratio": ratio,
                "measured_ratio_ref": regime["measured_ratio"],
                "movement_share_ref": regime["movement_share"],
                "idle_scalar": sim.idle_time_scalar,
                "distance_scalar": sim.distance_scalar,
                "series": series,
                "n_code": n_code,
                "k": n_code - 2,
                "patches": len({patch for patch, _ in placement}),
                "alpha": calibration.alpha,
                "repetitions": settings.repetitions,
                "h_commutation": commute,
                "movement_error": settings.movement_error,
                "oracle_dropped": dropped if series == SELECTOR else 0,
                "lsr": measured["lsr"],
                # Recorded, never plotted: at 24-40 qubits two finite samples score
                # TVD ~ 1.0 regardless of noise. See `experiments/shared/simulation.py`.
                "tvd": measured["tvd"],
                "one_qubit_gates": measured["one_qubit_gates"],
                "two_qubit_gates": measured["two_qubit_gates"],
                "circuit_depth": measured["circuit_depth"],
            }
        )
    return rows


#: What identifies one simulation within a column: (n_code, placement, h_commutation).
ArmKey = Tuple[int, Tuple[Tuple[int, int], ...], bool]


def _arm_key(n_code: int, placement, commute: bool) -> ArmKey:
    return int(n_code), tuple((int(p), int(lane)) for p, lane in placement), bool(commute)


def _objective_columns(value) -> Dict[str, object]:
    """A `packing.standard.StandardValue` as CSV columns, NaN where an arm has none."""
    if value is None:
        return {"objective_total": math.nan, "objective_detected": math.nan,
                "objective_silent": math.nan, "objective_currency": ""}
    return {"objective_total": float(value.total), "objective_detected": float(value.detected),
            "objective_silent": float(value.silent), "objective_currency": value.currency}


def _run_column_standard(job) -> List[Dict]:
    """
    `_run_column` under the standard configuration (module docstring, `--standard`).

    The same column -- calibrate once at the reference, then measure every arm -- with the
    standard planner on the static-k rung, `compile_circuit` as the selector, and every arm
    it priced as the oracle set. Each distinct (code size, placement, commutation) is
    simulated once per column; a repeat reuses that measurement and says so in
    `duplicate_of`. Module-level for `parallel_map`, like `_run_column`.
    """
    import time

    from iceberg.gadget_options import transversal_intra
    from packing.compiler import compile_circuit
    from packing.standard import STANDARD, plan_standard

    settings, name, circuit, ratio = job
    rows: List[Dict] = []
    reference = settings.reference_k
    # `with_standard` again, so a `Settings(standard=True)` built by hand gets it too.
    sim = settings.sim.with_standard().with_noise_ratio(ratio)
    profile = noise_profile_for(sim)
    num_qubits = circuit.num_qubits

    # The naive ladder keeps the submitted rule. `commutation_pays` keys its answer on the
    # live gadget set, so it is asked under the one its arms are translated with.
    commutes: Dict[int, bool] = {}

    def commute_for(n_code: int) -> bool:
        if n_code not in commutes:
            with transversal_intra(sim.transversal_merging):
                commutes[n_code] = commutation_pays(circuit, n_code)
        return commutes[n_code]

    ref_placement = naive_packing(num_qubits, reference)
    calibration = calibrate(
        circuit,
        ref_placement,
        sim,
        n_code=reference + 2,
        baseline_gate_count=circuit.size() * 3,
    )
    if not math.isfinite(calibration.alpha):
        return rows
    try:
        regime = measured_regime(
            circuit,
            ref_placement,
            n_code=reference + 2,
            idle_time_scalar=sim.idle_time_scalar,
            distance_scalar=sim.distance_scalar,
            asap_order=sim.asap_order,
        )
    except Exception:  # noqa: BLE001 - a rejected reference costs the column
        return rows

    # `(series, n_code, placement, commute, extra columns)`, in measurement order: the
    # naive ladder, the planner, the selector, then the candidates -- so a candidate that
    # repeats a rung reuses the rung's measurement, never the other way round.
    arms: List[Tuple[str, int, Perm, bool, Dict]] = [
        (f"naive_k{k}", k + 2, naive_packing(num_qubits, k), commute_for(k + 2),
         {"source": "naive"})
        for k in settings.k_values
    ]

    # The planner alone at the reference code size, deciding commutation itself ("auto")
    # and simulated under that decision -- its own, as every standard arm is.
    began = time.time()
    try:
        plan = plan_standard(
            circuit, reference + 2, noise=profile, h_commutation="auto", verify=True
        )
        fell_back = bool(getattr(plan, "fell_back", False))
        arms.append((
            AQORN_FIXED, reference + 2, [tuple(p) for p in plan.placement],
            bool(plan.h_commutation),
            {"source": "naive" if fell_back else "packer", "plan_fell_back": fell_back,
             "plan_seconds": time.time() - began},
        ))
    except Exception:  # noqa: BLE001 - a refused plan costs one arm, and the circuit
        pass

    # The standard resource selector, and the ceiling built from the arms it priced.
    model: Dict[ArmKey, Tuple[int, object]] = {}
    began = time.time()
    try:
        built = compile_circuit(
            circuit,
            noise=knobs_for(sim, settings.movement_error),
            packing_noise=profile,
            planner="standard",
        )
        seconds = time.time() - began
        for rank, (arm, value) in enumerate(built.priced):
            model.setdefault(_arm_key(arm.n_code, arm.placement, arm.h_commutation),
                             (rank, value))
        arms.append((
            SELECTOR, built.n_code, [tuple(p) for p in built.placement],
            bool(built.h_commutation),
            {"source": built.source, "candidate": f"{built.source} n={built.n_code}",
             "objective": built.objective, "plan_seconds": seconds},
        ))
        for rank, (arm, value) in enumerate(built.priced):
            arms.append((
                ORACLE_CANDIDATE, arm.n_code, [tuple(p) for p in arm.placement],
                bool(arm.h_commutation),
                {"source": arm.name, "candidate": f"{arm.name} n={arm.n_code}",
                 "candidate_rank": rank, "objective": value},
            ))
    except Exception:  # noqa: BLE001 - a refused selection costs the column's top two
        pass

    context = {
        "circuit": name,
        "program_qubits": num_qubits,
        "program_gates": circuit.size(),
        "ratio": ratio,
        "measured_ratio_ref": regime["measured_ratio"],
        "movement_share_ref": regime["movement_share"],
        "idle_scalar": sim.idle_time_scalar,
        "distance_scalar": sim.distance_scalar,
        "alpha": calibration.alpha,
        "repetitions": settings.repetitions,
        "movement_error": settings.movement_error,
    }

    # Keyed on (n_code, commute) as well as code size: commutation is the plan's decision
    # now, not one function of n_code, and the guard must compare like with like.
    clean_reference: Dict[Tuple[int, bool], float] = {}

    def measure(label: str, n_code: int, placement: Perm, commute: bool) -> Optional[Dict]:
        """Guard, then simulate with frames. None when miscompiled or rejected."""
        began = time.time()
        clean_arm = clean_naive = math.nan
        naive = naive_packing(num_qubits, n_code - 2)
        # Naive is the guard's own reference, so naive placements skip it -- as the
        # `naive_` series do in `_run_column`.
        if _arm_key(n_code, placement, commute) != _arm_key(n_code, naive, commute):
            if (n_code, commute) not in clean_reference:
                clean_reference[(n_code, commute)] = noiseless_tvd(
                    circuit, naive, sim, n_code=n_code, h_commutation=commute,
                )
            clean_naive = clean_reference[(n_code, commute)]
            clean_arm = noiseless_tvd(
                circuit, placement, sim, n_code=n_code, h_commutation=commute,
            )
            if miscompiled(clean_naive, clean_arm):
                print(f"[{NAME}] skipping {name} {label} n_code={n_code} "
                      f"h_commutation={commute}: miscompiled, noiseless TVD "
                      f"{clean_arm:.3f} against naive's {clean_naive:.3f}")
                return None
        guard_seconds = time.time() - began
        config = build_config(
            sim,
            n_code=n_code,
            alpha=calibration.alpha,
            repetitions=settings.repetitions,
            h_commutation=commute,
            merging=True,
        )
        began = time.time()
        try:
            measured = evaluate(circuit, placement, config, frames=settings.frames,
                                frame_seed=sim.seed)
        except Exception as error:  # noqa: BLE001 - `evaluate` splits frames outside its try
            print(f"[{NAME}] {name} {label}: frame measurement failed ({error!r}); "
                  f"shots only for this arm")
            measured = evaluate(circuit, placement, config)
        if measured is None:
            return None
        return {
            **measured,
            # Read back off the config the simulation ran, not off `sim`.
            "asap_order": bool(config.translator_config.asap_order),
            "transversal_merging": bool(config.translator_config.transversal_intra_two_qubit),
            "noiseless_tvd": clean_arm,
            "noiseless_tvd_naive": clean_naive,
            "guard_seconds": guard_seconds,
            "sim_seconds": time.time() - began,
        }

    done: Dict[ArmKey, Optional[Tuple[str, Dict]]] = {}
    for series, n_code, placement, commute, extra in arms:
        key = _arm_key(n_code, placement, commute)
        label = extra["candidate"] if series == ORACLE_CANDIDATE else series
        duplicate_of = ""
        if key in done:
            if done[key] is None:
                continue  # its first occurrence was miscompiled or rejected
            duplicate_of, measured = done[key]
            measured = {**measured, "guard_seconds": 0.0, "sim_seconds": 0.0}
        else:
            measured = measure(label, n_code, placement, commute)
            done[key] = None if measured is None else (label, measured)
            if measured is None:
                continue
        # The model value is a property of the configuration, so a rung identical to a
        # priced arm carries it too; the rank only belongs to the selector's own rows.
        rank, value = model.get(key, (math.nan, None))
        if "objective" in extra:
            value = extra["objective"]
        if series not in (SELECTOR, ORACLE_CANDIDATE):
            rank = math.nan
        rows.append({
            **context,
            "series": series,
            "n_code": n_code,
            "k": n_code - 2,
            "patches": len({patch for patch, _ in placement}),
            "h_commutation": commute,
            "oracle_dropped": 0,  # no cap under the standard: every priced arm is measured
            "lsr": measured["lsr"],
            "tvd": measured["tvd"],  # recorded, never plotted -- as in `_run_column`
            "one_qubit_gates": measured["one_qubit_gates"],
            "two_qubit_gates": measured["two_qubit_gates"],
            "circuit_depth": measured["circuit_depth"],
            "configuration": "standard",
            "asap_order": measured["asap_order"],
            "transversal_merging": measured["transversal_merging"],
            "planner_silent_weight": float(STANDARD.silent_weight_for(n_code)),
            "objective_silent_weight": float(STANDARD.ranking_weight()),
            "frames": settings.frames,
            "silent_per_kept": measured.get("silent_per_kept", math.nan),
            "p_silent": measured.get("p_silent", math.nan),
            "frame_lsr": measured.get("frame_lsr", math.nan),
            "source": extra.get("source", ""),
            "candidate": extra.get("candidate", ""),
            "candidate_rank": extra.get("candidate_rank", rank),
            "duplicate_of": duplicate_of,
            **_objective_columns(value),
            "plan_fell_back": extra.get("plan_fell_back", math.nan),
            "noiseless_tvd": measured["noiseless_tvd"],
            "noiseless_tvd_naive": measured["noiseless_tvd_naive"],
            "plan_seconds": extra.get("plan_seconds", math.nan),
            "guard_seconds": measured["guard_seconds"],
            "sim_seconds": measured["sim_seconds"],
        })
    return rows


def complete_circuits(frame: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    """
    Drop any circuit missing a rung of the ladder at any ratio, so all lines share a
    denominator.

    Without this the oracle is a max over whichever arms happened to survive and the
    naive floor is a min over the same, which moves both edges of the headroom in the
    same direction and makes the captured fraction meaningless. `code_size_crossover`
    records the version of this that reversed a conclusion.
    """
    required = {f"naive_k{k}" for k in k_values} | {SELECTOR, AQORN_FIXED, ORACLE_CANDIDATE}
    ratios = frame["ratio"].nunique()
    keep = []
    for name, group in frame.groupby("circuit"):
        present = group[group["series"].isin(required)]
        counts = present.groupby("series")["ratio"].nunique()
        if len(counts) == len(required) and (counts == ratios).all():
            keep.append(name)
    dropped = sorted(set(frame["circuit"]) - set(keep))
    if dropped:
        print(f"[{NAME}] dropped for incomplete coverage: {', '.join(dropped)}")
    return frame[frame["circuit"].isin(keep)].copy()


def _log_band(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Geometric mean over circuits and its standard error, computed in log space.

    Selection rate compounds -- it is `exp(-E)` for a sum of independent error terms -- so
    an arithmetic mean is pulled around by whichever circuit sits highest and an arithmetic
    SD can put the lower band edge below zero on a log axis.

    Standard error rather than the house default SD, following `depth_vs_gates_tradeoff`
    and `code_size_crossover`: every series here is a ratio taken *within* a circuit, so
    the circuit is already cancelled and the SD across benchmarks would describe how much
    they differ from each other, not the uncertainty on the mean the line draws. The
    caption has to say which one it is.

    Non-positive and non-finite values are dropped rather than clamped: a selection rate
    of exactly zero is a point where post-selection kept nothing, which has no logarithm
    and no honest substitute.
    """
    finite = np.asarray(
        [v for v in values if v == v and v > 0.0 and math.isfinite(v)], dtype=float
    )
    if finite.size == 0:
        return math.nan, math.nan, math.nan
    logs = np.log(finite)
    centre, spread = float(logs.mean()), float(logs.std(ddof=0))
    if finite.size > 1:
        spread /= math.sqrt(finite.size)
    return math.exp(centre), math.exp(centre - spread), math.exp(centre + spread)


def relative_curves(
    frame: pd.DataFrame, k_values: Sequence[int], ratios: Sequence[float], reference_k: int,
    *, metric: str = "lsr", lower_is_better: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Every series as shots kept relative to naive at the reference code size, paired
    within circuit.

    Paired, not a ratio of means. Circuit-to-circuit spread in absolute selection rate is
    far larger than anything the compiler does, so an unpaired ratio would mostly report
    which benchmarks landed in which arm.

    The oracle is a max over the simulated candidates *and* the selector's own row, which
    is what makes it dominate the selector on every row rather than merely usually.

    `metric` re-takes the same ladder on another per-arm column; with `lower_is_better`
    (`--standard`'s `silent_per_kept` panel) the oracle is the candidates' minimum and the
    worst static k the naive maximum. The defaults are the shots ladder.
    """
    per_row = frame.pivot_table(
        index=["circuit", "ratio"], columns="series", values=metric, aggfunc="mean"
    )
    # `oracle_cand` collapses to one column under `pivot_table`'s mean, which is the wrong
    # reduction -- the ceiling is a max. Taken separately off the raw rows.
    candidates = (
        frame[frame["series"].isin([ORACLE_CANDIDATE, SELECTOR])]
        .groupby(["circuit", "ratio"])[metric]
    )
    candidate_max = candidates.min() if lower_is_better else candidates.max()

    reference = f"naive_k{reference_k}"
    if reference not in per_row.columns:
        raise RuntimeError(f"reference series {reference} absent from the data")
    naive_columns = [f"naive_k{k}" for k in k_values if f"naive_k{k}" in per_row.columns]

    # A reference of exactly zero makes every ratio on that row infinite. Dropped to NaN
    # here so every series skips the row together rather than one of them returning inf.
    denominator = per_row[reference].replace(0.0, np.nan)

    relative = pd.DataFrame(index=per_row.index)
    for column in (SELECTOR, AQORN_FIXED):
        if column in per_row.columns:
            relative[column] = per_row[column].div(denominator)
    naive = per_row[naive_columns]
    worst, best = (naive.max(axis=1), naive.min(axis=1)) if lower_is_better else (
        naive.min(axis=1), naive.max(axis=1))
    relative["naive_worst"] = worst.div(denominator)
    relative["naive_best"] = best.div(denominator)
    relative["oracle"] = candidate_max.reindex(per_row.index).div(denominator)

    return {
        key: np.array(
            [
                _log_band(
                    relative.loc[
                        relative.index.get_level_values("ratio") == r, key
                    ].to_numpy()
                )
                for r in ratios
            ]
        )
        for key in relative.columns
    }


def _headroom(curves: Dict[str, np.ndarray], key: str = SELECTOR) -> np.ndarray:
    """
    Fraction of the oracle's headroom that `key` captures, per ratio.

    In log space, because every quantity here compounds: the baseline sits at
    `log 1 = 0`, so the fraction is simply `log(series) / log(oracle)`. Taken on the
    aggregated curves rather than per row -- a row whose oracle barely clears 1.0 has a
    near-zero denominator and would dominate any average of per-row fractions.

    NaN where the oracle finds no headroom at all, which is the honest answer: there is
    no fraction of nothing.
    """
    oracle = np.log(curves["oracle"][:, 0])
    series = np.log(curves[key][:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(oracle) > 1e-9, series / oracle, math.nan)


def print_summary(frame: pd.DataFrame, settings: Settings) -> None:
    ratios = sorted(frame["ratio"].unique())
    k_values = sorted(
        frame.loc[frame["series"].str.startswith("naive_k"), "k"].unique()
    )
    circuits = frame["circuit"].nunique()
    reference = f"naive_k{settings.reference_k}"

    print(
        f"\n[{NAME}] shots kept relative to naive [[{settings.reference_k + 2},"
        f"{settings.reference_k},2]], geometric mean of {circuits} circuits"
        f"   [{settings.repetitions:,} raw shots/point, alpha per (circuit, ratio)]"
    )
    baseline = frame[frame["series"] == reference]
    if not baseline.empty:
        span = baseline.groupby("ratio")["lsr"].mean()
        print(
            f"  baseline acceptance: {span.min():.3f} to {span.max():.3f} across the "
            f"axis (flat is the goal -- see the docstring)"
        )
    candidates = frame[frame["series"] == ORACLE_CANDIDATE]
    if not candidates.empty:
        per_column = candidates.groupby(["circuit", "ratio"]).size()
        print(
            f"  oracle set: {per_column.mean():.1f} candidates per column on average "
            f"({per_column.min()}-{per_column.max()}), spanning code sizes "
            f"{sorted(int(v) for v in candidates['k'].unique())}"
        )
    chosen = frame[frame["series"] == SELECTOR]
    if not chosen.empty:
        counts = {
            int(k): int(v)
            for k, v in chosen["k"].value_counts().sort_index().items()
        }
        print(f"  selector's chosen k: {counts}")

    curves = relative_curves(frame, k_values, ratios, settings.reference_k)
    order = [key for key, _, _ in LINES if key in curves]
    labels = _PRINT_LABELS

    print("\n  the ladder      rows: measured ratio at the reference, cols: series")
    print("           " + "".join(f"{labels[key]:>24}" for key in order))
    for index, ratio in enumerate(ratios):
        measured = frame[frame["ratio"] == ratio]["measured_ratio_ref"]
        measured = measured[np.isfinite(measured)]
        position = float(np.exp(np.log(measured).mean())) if len(measured) else math.nan
        print(
            f"   {_plain_decimal(position, 3):>7} "
            + "".join(f"{curves[key][index, 0]:>24.3f}" for key in order)
        )

    for key in order:
        column = curves[key][:, 0]
        print(
            f"    {labels[key]:>24}: {np.nanmin(column):.2f}x to "
            f"{np.nanmax(column):.2f}x, geometric mean "
            f"{math.exp(float(np.nanmean(np.log(column)))):.2f}x"
        )

    # The number the caption quotes.
    if SELECTOR in curves and "oracle" in curves:
        captured = _headroom(curves, SELECTOR)
        planner = _headroom(curves, AQORN_FIXED) if AQORN_FIXED in curves else None
        print("\n  fraction of the oracle's headroom captured, per ratio:")
        print("    selector: " + "  ".join(f"{v:.0%}" for v in captured))
        if planner is not None:
            print("    planner : " + "  ".join(f"{v:.0%}" for v in planner))
        print(f"\n  selector captures {np.nanmean(captured):.0%} of the available "
              f"headroom on average")
        if planner is not None:
            print(f"  of which the planner alone (static k=2) accounts for "
                  f"{np.nanmean(planner):.0%}; resource selection adds the rest")

        # A sanity check the figure would otherwise hide: the oracle must dominate the
        # selector on every row by construction, since the selector's own measurement is
        # in the max. A violation here means the pairing broke, not that the selector won.
        breach = float(np.nanmax(curves[SELECTOR][:, 0] - curves["oracle"][:, 0]))
        if breach > 1e-9:
            print(f"\n  WARNING: selector exceeds its own ceiling by {breach:.4f} -- the "
                  f"oracle max is not covering the selector's row. This is a bug, not a "
                  f"result; check `relative_curves`.")


def _has_frames(frame: pd.DataFrame) -> bool:
    return "silent_per_kept" in frame.columns and bool(frame["silent_per_kept"].notna().any())


def print_standard_summary(frame: pd.DataFrame, settings: Settings) -> None:
    """
    What `--standard` adds to `print_summary`: the configuration as the simulated configs
    recorded it, how the oracle set was measured, the accuracy ladder, and how often the
    model's pick was the measured best of the candidates.
    """
    ratios = sorted(frame["ratio"].unique())
    k_values = sorted(
        frame.loc[frame["series"].str.startswith("naive_k"), "k"].unique()
    )
    labels = _PRINT_LABELS

    def recorded(column: str):
        if column not in frame.columns:
            return "absent"
        return sorted({bool(v) for v in frame[column].dropna()})

    print(f"\n[{NAME}] standard configuration, as the simulated configs recorded it: "
          f"asap_order {recorded('asap_order')}, transversal in-patch CZ / XCX "
          f"{recorded('transversal_merging')}")
    if "planner_silent_weight" in frame.columns:
        weights = frame.groupby("n_code")["planner_silent_weight"].unique()
        objective = sorted(float(v) for v in frame["objective_silent_weight"].dropna().unique())
        print(f"  silent weights: objective {objective}, planner by code size "
              f"{({int(n): sorted(float(v) for v in w) for n, w in weights.items()})}")
    candidates = frame[frame["series"] == ORACLE_CANDIDATE]
    if not candidates.empty:
        reused = candidates["duplicate_of"].fillna("").astype(str).ne("")
        per_column = candidates.groupby(["circuit", "ratio"]).size()
        print(f"  oracle set: every arm compile_circuit priced, {per_column.mean():.1f} per "
              f"column ({per_column.min()}-{per_column.max()}); {int((~reused).sum())} "
              f"simulated as new arms, {int(reused.sum())} identical to an arm already "
              f"measured in their column and reused")
    chosen = frame[frame["series"] == SELECTOR]
    if not chosen.empty:
        print(f"  selector kept: "
              f"{chosen['source'].fillna('').value_counts().to_dict()}")

    if not _has_frames(frame):
        print("  no frame columns -- silent error per kept shot not measured")
        return
    curves = relative_curves(frame, k_values, ratios, settings.reference_k,
                             metric="silent_per_kept", lower_is_better=True)
    order = [key for key, _, _ in LINES if key in curves]
    print(f"\n  accuracy: silent error per kept shot relative to naive "
          f"[[{settings.reference_k + 2},{settings.reference_k},2]], lower is better "
          f"({int(frame['frames'].max()):,} frame shots per arm)")
    print("           " + "".join(f"{labels[key]:>24}" for key in order))
    for index, position in enumerate(_measured_positions(frame, ratios)):
        print(
            f"   {_plain_decimal(position, 3):>7} "
            + "".join(f"{curves[key][index, 0]:>24.3f}" for key in order)
        )
    for key in order:
        column = curves[key][:, 0]
        print(
            f"    {labels[key]:>24}: {np.nanmin(column):.2f}x to "
            f"{np.nanmax(column):.2f}x, geometric mean "
            f"{math.exp(float(np.nanmean(np.log(column)))):.2f}x"
        )
    if SELECTOR in curves and "oracle" in curves:
        captured = _headroom(curves, SELECTOR)
        print("  fraction of the oracle's accuracy headroom captured, per ratio:")
        print("    selector: " + "  ".join(f"{v:.0%}" for v in captured))
        if AQORN_FIXED in curves:
            planner = _headroom(curves, AQORN_FIXED)
            print("    planner : " + "  ".join(f"{v:.0%}" for v in planner))
        # Lower is better here, so the selector may not sit *below* its own ceiling.
        breach = float(np.nanmax(curves["oracle"][:, 0] - curves[SELECTOR][:, 0]))
        if breach > 1e-9:
            print(f"\n  WARNING: selector beats its own accuracy ceiling by {breach:.4f} "
                  f"-- the oracle min is not covering the selector's row. A bug, not a "
                  f"result; check `relative_curves`.")

    # The gap the oracle measures, per column: was the model's pick the measured best?
    hits = {"lsr": 0, "silent_per_kept": 0}
    columns = 0
    for _, group in candidates.groupby(["circuit", "ratio"]):
        pick = group[group["candidate_rank"] == 0]
        if pick.empty:
            continue
        columns += 1
        hits["lsr"] += int(pick["lsr"].iloc[0] >= group["lsr"].max() - 1e-12)
        hits["silent_per_kept"] += int(
            pick["silent_per_kept"].iloc[0] <= group["silent_per_kept"].min() + 1e-12)
    if columns:
        print(f"  the model's pick is the measured best candidate in {hits['lsr']}/"
              f"{columns} columns on shots kept, {hits['silent_per_kept']}/{columns} on "
              f"silent error per kept shot")


def _plain(value: float, _position=None) -> str:
    """
    `0.05x`, `1.5x`, `10x` -- never `5 x 10^-2`, which the style guide rules out.

    The `x` suffix is the paper's convention for a ratio, and this axis was already one:
    the y values are shots kept divided by the reference's, on a log scale. Suffixing them
    is what makes that readable without the axis label, and matches the factor axes the
    other figures now draw. Literal U+00D7, not mathtext -- `svg.fonttype: none` would turn
    mathtext back into glyph paths.
    """
    if value >= 1:
        return f"{value:g}×"
    text = f"{value:.4f}".rstrip("0")
    return (text.rstrip(".") if text.endswith(".") else text) + "×"


def _plain_decimal(value: float, digits: int = 2) -> str:
    """
    `value` to about `digits` significant figures, always as a plain decimal.

    The measured x positions span five orders of magnitude, so the obvious `f"{x:.2g}"`
    prints the idle-dominated end as `2.9e+02`. The style guide rules out scientific
    notation on an axis, and a reader comparing a tick against a number in the prose
    should not have to decode an exponent to do it.
    """
    if not math.isfinite(value) or value == 0.0:
        return ""
    exponent = math.floor(math.log10(abs(value)))
    return f"{value:.{max(0, digits - 1 - exponent)}f}"


#: Log-axis tick candidates for a ratio panel, in decimal. A ratio axis clustered around
#: 1.0 gets one decade tick and matplotlib labels the rest `2 x 10^0`.
_RATIO_TICKS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

#: At most this many ticks. Beyond it the 9pt labels collide at column width.
_MAX_TICKS = 7


def _log_ticks(low: float, high: float) -> List[float]:
    """Ticks covering `[low, high]` on a log axis, as plain decimals, thinned to fit."""
    inside = [value for value in _RATIO_TICKS if low <= value <= high]
    if not inside:
        return [low, high]
    while len(inside) > _MAX_TICKS:
        anchor = inside.index(1.0) if 1.0 in inside else 0
        inside = [v for index, v in enumerate(inside) if (index - anchor) % 2 == 0]
    return inside


def _measured_positions(frame: pd.DataFrame, ratios) -> List[float]:
    """Geometric-mean measured reference ratio per nominal column -- the plotted x."""
    positions = []
    for ratio in ratios:
        values = frame[frame["ratio"] == ratio]["measured_ratio_ref"].to_numpy()
        values = values[np.isfinite(values)]
        positions.append(
            float(np.exp(np.log(values).mean())) if len(values) else float("nan")
        )
    return positions


def _draw(axis, x, curve: np.ndarray, colour, marker, dashes, label: str) -> None:
    """One series: mean line plus its spread band at the house alpha."""
    mean, low, high = curve[:, 0], curve[:, 1], curve[:, 2]
    axis.plot(x, mean, color=colour, marker=marker, linestyle=dashes,
              label=label, markersize=3.2, linewidth=1.2)
    axis.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0)


def plot(frame: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    ratios = sorted(frame["ratio"].unique())
    x_measured = _measured_positions(frame, ratios)
    k_values = sorted(
        frame.loc[frame["series"].str.startswith("naive_k"), "k"].unique()
    )
    curves = relative_curves(frame, k_values, ratios, settings.reference_k)

    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    palette, markers, dashes = cycle["color"], cycle["marker"], cycle["linestyle"]

    # One column, a little taller than square: five levels need vertical room to separate,
    # and the width is untouched, which is what keeps type at 10pt on the page.
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.9))

    # The baseline the whole ladder is measured against.
    axis.axhline(1.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)

    for key, label, slot in LINES:
        if key in curves:
            _draw(axis, x_measured, curves[key], palette[slot % len(palette)],
                  markers[slot % len(markers)], dashes[slot % len(dashes)], label)

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(axis="y", which="major")
    low, high = axis.get_ylim()
    axis.set_yticks(_log_ticks(low, high))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(_plain))
    axis.yaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_ylim(low, high)

    reference = _CODE.format(k=settings.reference_k, n=settings.reference_k + 2)
    axis.set_ylabel(f"shots kept, vs. naive {reference}")
    axis.set_xlabel("measured idling / movement noise ratio")

    # `best` rather than a pinned corner, deliberately. Five levels plus a floor line
    # below 1.0 leave no corner that is free by construction: the sibling figure pins its
    # corners and its comments record getting that wrong twice when the data moved. Once
    # the real sweep exists and the shape is known, pin the free corner here so the figure
    # stops moving between runs.
    axis.legend(ncol=1, handlelength=1.5, handletextpad=0.4, borderpad=0.2,
                labelspacing=0.3, loc="best", fontsize=8, **_LEGEND_BACKING)

    # Which end of the axis is which, hung under the extreme ticks rather than annotated
    # inside the panel, where it would collide with whichever corner the legend takes.
    labels = [_plain_decimal(x) for x in x_measured]
    labels[0] += "\nmovement"
    labels[-1] += "\nidling"
    axis.set_xticks(x_measured)
    axis.set_xticklabels(labels)
    axis.xaxis.set_minor_locator(plt.NullLocator())

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override)
    plt.close(fig)
    return paths


def plot_standard(frame: pd.DataFrame, settings: Settings, name: str) -> List[Path]:
    """
    `plot` under `--standard`: the same shots ladder and, beneath it on a shared x, the
    same levels on silent error per kept shot -- the accuracy the standard's silent-error
    weight is there to buy, so the panel that shows what the shots were traded for.

    One column wide with the panels stacked, which keeps type at 10pt on the page. Saved
    through `save_figure(versioned=True)` under `name`, so nothing already on disk -- the
    submitted figure least of all -- is replaced.
    """
    use_style()
    ratios = sorted(frame["ratio"].unique())
    x_measured = _measured_positions(frame, ratios)
    k_values = sorted(
        frame.loc[frame["series"].str.startswith("naive_k"), "k"].unique()
    )
    reference = _CODE.format(k=settings.reference_k, n=settings.reference_k + 2)
    panels = [("lsr", False, f"shots kept, vs. naive {reference}")]
    if _has_frames(frame):
        panels.append(("silent_per_kept", True,
                       f"silent errors per kept shot,\nvs. naive {reference} (lower: better)"))

    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    palette, markers, dashes = cycle["color"], cycle["marker"], cycle["linestyle"]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                             figsize=(COLUMN_WIDTH, 2.9 + 2.5 * (len(panels) - 1)))
    axes = list(axes[:, 0])

    for axis, (metric, lower, ylabel) in zip(axes, panels):
        curves = relative_curves(frame, k_values, ratios, settings.reference_k,
                                 metric=metric, lower_is_better=lower)
        axis.axhline(1.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)
        for key, label, slot in LINES:
            if key in curves:
                _draw(axis, x_measured, curves[key], palette[slot % len(palette)],
                      markers[slot % len(markers)], dashes[slot % len(dashes)], label)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(axis="y", which="major")
        low, high = axis.get_ylim()
        axis.set_yticks(_log_ticks(low, high))
        axis.yaxis.set_major_formatter(plt.FuncFormatter(_plain))
        axis.yaxis.set_minor_formatter(plt.NullFormatter())
        axis.set_ylim(low, high)
        axis.set_ylabel(ylabel)

    # One legend: both panels draw the same series in the same styles. `best`, for the
    # reason `plot` gives.
    axes[0].legend(ncol=1, handlelength=1.5, handletextpad=0.4, borderpad=0.2,
                   labelspacing=0.3, loc="best", fontsize=8, **_LEGEND_BACKING)

    bottom = axes[-1]
    bottom.set_xlabel("measured idling / movement noise ratio")
    labels = [_plain_decimal(x) for x in x_measured]
    labels[0] += "\nmovement"
    labels[-1] += "\nidling"
    bottom.set_xticks(x_measured)
    bottom.set_xticklabels(labels)
    bottom.xaxis.set_minor_locator(plt.NullLocator())

    paths = save_figure(fig, name, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(defaults.ratios),
                        help="Nominal sweep for coverage (geometric mean pinned at one). "
                             "The plotted x is the measured idle/movement ratio of the "
                             "reference configuration at each swept point.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values),
                        help="Code sizes the naive static baseline is drawn from.")
    parser.add_argument("--circuits", nargs="+", default=None,
                        help="Benchmark names. Default: every proxy inside "
                             "--qubit-range under --gate-limit gates.")
    parser.add_argument("--qubit-range", nargs=2, type=int,
                        default=list(defaults.qubit_range), metavar=("LOW", "HIGH"),
                        help="Program-qubit window. Below ~24 the patch ladder saturates "
                             "and the static baselines collapse onto each other.")
    parser.add_argument("--gate-limit", type=int, default=defaults.gate_limit,
                        help="Drop proxies above this many gates.")
    parser.add_argument("--repetitions", type=int, default=defaults.repetitions,
                        help="Raw shots per point, held fixed rather than scaled by the "
                             "calibrated selection rate -- the discarded shots are the "
                             "result here.")
    parser.add_argument("--reference-k", type=int, default=defaults.reference_k,
                        help="Code size alpha is calibrated at, and the naive baseline "
                             "every line is drawn against.")
    parser.add_argument("--oracle-top", type=int, default=defaults.oracle_top,
                        help="Model-ranked frontier candidates simulated for the ceiling, "
                             "on top of the best at each code size and the selector's own "
                             "plan. Higher is a tighter ceiling and a longer run. Not used "
                             "under --standard, whose oracle set is uncapped.")
    parser.add_argument("--oracle-all", action="store_true",
                        help="Simulate the whole frontier. Roughly 3x the cost; removes "
                             "the cap's bias toward candidates the model already likes. "
                             "Not used under --standard.")
    parser.add_argument("--movement-error", type=float, default=defaults.movement_error,
                        help="What the selector charges per crossing per patch hop. "
                             "Uncalibrated; affects the selector and oracle lines.")
    parser.add_argument("--benchmark-root", type=Path, default=defaults.benchmark_root)
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--standard", action="store_true",
                        help="Measure the ladder under the standard configuration "
                             "(packing.standard.STANDARD): plan_standard for the static-k "
                             "planner, compile_circuit(planner='standard') for the "
                             "selector, every arm it priced as the oracle, layer-sorted "
                             "translations with transversal in-patch CZ/XCX, and "
                             "frame-measured silent error per kept shot. Writes "
                             f"{STANDARD_NAME}[_vN].csv and a versioned figure; replaces "
                             "nothing. See the module docstring.")
    parser.add_argument("--frames", type=int, default=defaults.frames,
                        help="Pauli-frame shots per simulated arm under --standard, for "
                             "silent error per kept shot. 0 skips them.")
    parser.add_argument("--csv", type=Path, default=None,
                        help=f"Default {defaults.csv_path.name} beside this script, or "
                             f"{STANDARD_NAME}.csv under --standard (which never replaces "
                             f"a file: it takes the next free _vN name).")
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it here: a full "
                             "run is hours of simulation.")
    args = parser.parse_args()
    sim = settings_from_args(args, defaults.sim)
    csv_path = args.csv if args.csv is not None else defaults.csv_path
    if args.standard:
        sim = sim.with_standard()
        if args.csv is None:
            csv_path = HERE / f"{STANDARD_NAME}.csv"
    return Settings(
        ratios=args.ratios,
        k_values=args.k_values,
        circuits=args.circuits,
        qubit_range=(args.qubit_range[0], args.qubit_range[1]),
        gate_limit=args.gate_limit,
        repetitions=args.repetitions,
        reference_k=args.reference_k,
        oracle_top=args.oracle_top,
        oracle_all=args.oracle_all,
        movement_error=args.movement_error,
        benchmark_root=args.benchmark_root,
        sim=sim,
        csv_path=csv_path,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        jobs=args.jobs,
        standard=args.standard,
        frames=args.frames,
    )


def _free_path(path: Path, *also: Path) -> Path:
    """
    `path`, or `<stem>_v2<suffix>`, `_v3`... -- the first name free beside `path` and in
    every directory in `also`, so a CSV and its `--persist` copy can share one name.
    """
    def taken(candidate: Path) -> bool:
        return candidate.exists() or any((d / candidate.name).exists() for d in also)

    if not taken(path):
        return path
    version = 2
    while taken(path.with_name(f"{path.stem}_v{version}{path.suffix}")):
        version += 1
    return path.with_name(f"{path.stem}_v{version}{path.suffix}")


def _write_new(frame: pd.DataFrame, path: Path) -> Path:
    """Write `frame` as CSV at `path` or the next free `_vN` -- never over a file (mode x)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        path = _free_path(path)
        try:
            frame.to_csv(path, index=False, mode="x")
            return path
        except FileExistsError:  # created between the check and the write
            continue


def _compute_standard(settings: Settings) -> Tuple[pd.DataFrame, str]:
    """
    `--standard`'s compute-and-write: the CSV under a free name beside the script (or
    `--csv`), the `--persist` copy under the same free name in results/data/. Returns the
    rows and the stem the figure is to take.
    """
    from experiments.shared.figures import data_dir

    if settings.reference_k not in settings.k_values:
        raise ValueError(
            f"--reference-k {settings.reference_k} must be one of --k-values "
            f"{list(settings.k_values)}; every line is drawn against it."
        )
    frame = compute(settings)
    if frame.empty:
        raise RuntimeError("no data produced -- every simulation failed")
    frame = stamp_provenance(frame)
    base = settings.csv_path
    if base == HERE / f"{NAME}.csv":  # a hand-built `Settings(standard=True)`
        base = HERE / f"{STANDARD_NAME}.csv"
    persist = data_dir(SCRIPT) if settings.persist else None
    path = _write_new(frame, _free_path(base, *([persist] if persist else [])))
    print(f"[{NAME}] wrote {len(frame)} rows to {path}")
    if persist is not None:
        print(f"[{NAME}] persisted to {_write_new(frame, persist / path.name)}")
    return frame, path.stem


def _standard_csv(frame: pd.DataFrame, settings: Settings) -> bool:
    """Whether a loaded CSV was written by `--standard` -- refusing to mix the two up."""
    found = (
        set(frame["configuration"].dropna().astype(str))
        if "configuration" in frame.columns else set()
    )
    if "standard" in found and len(found) > 1:
        raise SystemExit(f"[{NAME}] {settings.from_csv} mixes configurations "
                         f"{sorted(found)} -- split it before replotting.")
    standard = found == {"standard"}
    if settings.standard and not standard:
        raise SystemExit(f"[{NAME}] --standard replots only a CSV written by --standard, "
                         f"and {settings.from_csv} is not one.")
    if standard and not settings.standard:
        print(f"[{NAME}] {settings.from_csv} was written by --standard: plotting it as "
              f"such, under a versioned {STANDARD_NAME} name")
    return standard


def main() -> None:
    settings = parse_args()
    standard, stem = settings.standard, STANDARD_NAME

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
        if "measured_ratio_ref" not in frame.columns:
            raise SystemExit(
                f"[{NAME}] {settings.from_csv} has no measured_ratio_ref column -- "
                "regenerate rather than replot."
            )
        standard = _standard_csv(frame, settings)
    elif settings.standard:
        frame, stem = _compute_standard(settings)
    else:
        if settings.reference_k not in settings.k_values:
            raise ValueError(
                f"--reference-k {settings.reference_k} must be one of --k-values "
                f"{list(settings.k_values)}; every line is drawn against it."
            )
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Stamp the cost model that produced these rows, so a future run can tell
        # whether this cache still describes the current planner. See packing/version.py.
        frame = stamp_provenance(frame)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            from experiments.shared.figures import data_dir

            target = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(target, index=False)
            print(f"[{NAME}] persisted to {target}")

    frame = complete_circuits(frame, settings.k_values)
    if frame.empty:
        raise RuntimeError("no circuit produced a complete ladder")
    print_summary(frame, settings)
    if standard:
        print_standard_summary(frame, settings)
    paths = plot_standard(frame, settings, stem) if standard else plot(frame, settings)
    for path in paths:
        print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
