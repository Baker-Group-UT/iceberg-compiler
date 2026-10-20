"""
code_size_crossover.py -- the best code size is a property of the hardware, not the circuit.

**Claim.** There is no single best `k` in the [[k+2, k, 2]] family. Which code size keeps
the most shots depends on which noise mechanism the machine is limited by, it depends on
the circuit as well, and AQORN's resource selector tracks the winner without being told
where it is. That is the argument for `packing/resource/select.py` existing, and this is
the only figure that makes it.

Two panels sharing the noise-ratio axis, both plotting selection rate relative to k = 2:

    (a) each larger fixed code size against k = 2 -- the mechanism. Every one of them sits
        above the line where movement dominates, crosses below it around ratio 4, and
        k = 10 alone climbs back to 1.33x at the idle-dominated end.
    (b) the resource selector, the best *fixed* k at each point, and the best single fixed k
        chosen with hindsight for the whole sweep -- the payoff. The selector beats both:
        2.10x against 1.71x for one fixed code size everywhere, and it exceeds even the
        per-point fixed-k line, because that line is an oracle over code size alone while the
        selector also chooses patch count and searches placements.

**Both panels are relative, and panel (a) started out absolute.** That version did not work:
absolute selection rate falls 25x across this sweep (0.44 to 0.018 at k = 2) while the code
sizes separate by about 10%, so five lines arrived on top of each other and the panel read
as "k does not matter" -- the exact opposite of the finding. Dividing out the common fall
leaves the re-ordering, which is what the figure is about. The absolute scale it gives up
belongs in the caption. `print_summary` still prints the absolute table, so the numbers
behind that sentence come from the run that made the figure.

**Why the crossover exists.** Two terms in `packing.resource.noise.estimate` have opposite
slopes in `k`, and the ratio axis reweights them against each other:

    idle       = patches * (n_code + 2) * depth * (t_layer / T2)
    movement   = crossings * mean_patch_hops(patches)

Translated depth grows hard with `k` -- measured 105 -> 226 across k = 2..10 on a 24-qubit
structured circuit -- and that swamps the falling physical-qubit count per logical qubit, so
idling punishes large `k`. Movement runs the other way: `mean_patch_hops` grows like
`sqrt(patches)` (`packing/tables.py`), so spreading a program thin costs distance, and the
movement term falls 0.76 -> 0.21 expected detectable errors over that same range.
Reweighting the two mechanisms is therefore reweighting `k`.

(Both formulas are quoted as `estimate` computes them, including its `n_code + 2` physical
count per patch, which is two more than the code block itself holds. That convention is the
model's, not this figure's, and panel (a) does not depend on it -- it is simulated.)

**The x axis is the measured idle/movement noise ratio** of the reference configuration
-- the same naive k=2 baseline the calibration is solved on, read off `measured_regime`
per (circuit, ratio) and plotted at each column's geometric mean. The nominal ratios
(`SimSettings.with_noise_ratio`, geometric mean pinned at one, same convention as
`depth_vs_gates_tradeoff`) are only the sweep that generates coverage: the mapping from
scalars to regime depends on the circuit and the placement, so plotting the nominal
value asserted a balance the model does not have. Measuring on a fixed reference keeps
the axis independent of every series. Low is movement-dominated, high is idle-dominated,
and a crossover still means "which mechanism dominates" -- now in the schedule's own
units. Alpha-invariance (every error rate scales with alpha, `move_error` included)
is what lets one measurement label all the calibrated columns.

**The answer is circuit-dependent, and the figure says so rather than averaging it away.**
On `ising_n26` the winner moves k = 10 -> k = 2 as idling takes over; on `knn_n25` it moves
the other way, k = 4 -> k = 10. A 1-D chain spreads over many patches almost for free
because its interaction graph is local, so k = 2 buys parallelism without paying crossings;
a denser graph does pay. That heterogeneity is why panel (b) is the point of the figure: it
does not matter which `k` wins on any one circuit, only whether a per-circuit selector
finds it. The oracle line is the ceiling that heterogeneity creates, and the gap between
the best *fixed* `k` and that ceiling is what the selector is being asked to close.

**One noise calibration per (circuit, ratio), at the reference code size.** There is a trap
on each side of this and the figure has to thread between them.

Calibrating per `(circuit, ratio, k)` would delete the finding: it would pin selection rate
to a constant along the exact axis the comparison lives on, and every line would come out
flat. So `k` never gets its own alpha -- all code sizes in a column share the reference's.

Calibrating *once* per circuit is the mistake the first version made, and it cost most of the
effect. Separations scale with total noise, and holding one alpha across the whole ratio axis
let total noise vary 5x along it: baseline acceptance ran from 44% at the movement-dominated
end down to 1.8% at the idle-dominated end. The low-ratio columns were measured where nothing
can separate and the high-ratio columns at the post-selection floor, where the ordering was
erratic enough to flip sign between circuits. Solving alpha per `(circuit, ratio)` at
`--reference-k` puts every column in the discriminating band while leaving the within-column
comparison -- which is the whole figure -- untouched.

**Fixed raw shots, not fixed accepted shots.** `calibrate` normally sets repetitions to
`target_kept_shots / lsr`, which equalises accepted shots and is right for a placement
comparison. Here the whole quantity of interest is how many shots post-selection throws
away, so the raw budget is fixed at `--repetitions` and acceptance is allowed to fall where
it lands. Selection rate is a binomial proportion and stays precise even when acceptance is
low -- at the sweep's worst point, lsr = 0.01 on 40k raw shots, the standard error is 5%
relative, against differences of 10-50% -- so this costs nothing in precision on the
quantity actually plotted. It is the reason this figure claims selection rate and not TVD:
the same budget that pins a proportion to 5% leaves a 26-qubit distribution unmeasurable.

**TVD is recorded and never plotted.** At these widths it is not estimable: an entangled
26-qubit state spreads over enough outcomes that two finite samples score TVD ~ 1.0
regardless of noise (measured 0.9997 on `ising_n26`), which is the documented failure mode
in `experiments/shared/simulation.py` rather than a result. Selection rate is what this
figure claims, and `lsr_tvd_vs_k` owns the accuracy half of the k story.

**Real benchmarks, and wide enough to have a patch ladder.** The three existing k figures
run at 12 qubits, where patch count goes 6, 3, 2, 2, 2 -- the ladder saturates and the top
three code sizes are the same resource shape, so there is nothing for the ratio to trade
against. The default set is QASMBench proxies from 24 to 40 qubits, where the ladder is
clean (`ising_n26`: 13, 7, 5, 4, 3). Circuits that fail to place at any swept `k` are
dropped from every series, so the panels share a denominator.

**The planner sees the noise profile by default.** Each fixed-k line is the full compiler at
that code size, `plan_packing` included, priced under the same ratio the simulator charges
(`noise_profile_for`). That is the deployment-relevant reading and it builds on
`depth_vs_gates_tradeoff`, which establishes separately that the planner responds to the
ratio at all. `--planner-blind` prices every plan at ratio 1 instead, isolating the code
size from the planner's response to the regime.

**The selector line is the whole resource decision, placement included.** A fixed-k line is
`plan_packing` at that code size; the selector line is `select_resources`, which chooses code
size *and* patch count and then runs its own seed, lane assignment and descent. Those can
differ at an identical resource shape -- on `ising_n26` at ratio 1 both land on k = 4 across
7 patches and the selector still reads 0.246 against 0.174, because it found a better
placement there, not a better shape. So read panel (b) as "picking resources per circuit
beats fixing them", which is the claim, and not as "the code size alone is worth that much".
`--planner-blind` does not separate these either; only measuring `select_resources`' shape
under `plan_packing`'s placement would, and that is a different figure.

**One uncalibrated coefficient, stated because it moves the axis.** `NoiseKnobs.movement_error`
defaults to zero and has no measured value -- `noise_profile_for` calls the exchange rate
underivable, since movement noise in the simulator runs through a layout the planner does
not have. `--movement-error` sets what the *selector* charges per crossing per hop,
defaulting to one two-qubit gate error. Panel (a) is unaffected: it is simulated, and the
simulator's own `distance_scalar` is what the ratio moves there. Panel (b)'s selector line
does depend on it, so treat that line as "the selector, told this exchange rate" rather
than as a calibrated prediction.

Both panels are log-y, which is the stated exception to the zero-baseline rule: these are
ratios, where equal factors up and down should read as equal distances, and 1.0 is the
meaningful centre rather than zero. Bands are +-1 standard *error* over circuits rather than
the house default of +-1 SD -- both panels are paired within circuit, so the SD would
describe how much benchmarks differ from each other rather than the uncertainty on the
plotted mean. The caption says so. Spread is taken in log space, because selection rate
compounds multiplicatively.

Run with:

    python -m experiments.figures.code_size_crossover.code_size_crossover
    python -m experiments.figures.code_size_crossover.code_size_crossover --from-csv <path>
    python -m ... --circuits ising_n26 knn_n25 --ratios 0.25 1 4      # a fast smoke test
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from experiments.shared.parallel import add_jobs_argument, parallel_map

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style
from experiments.shared.simulation import (
    SimSettings,
    measured_regime,
    add_sim_arguments,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from packing.passes import commutation_pays
from packing.plan import plan_packing
from packing.resource.noise import NoiseKnobs
from packing.resource.select import select_resources

NAME = "code_size_crossover"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

Perm = List[Tuple[int, int]]

#: One measured arm: `(series name, k, placement, the commutation it was priced under)`.
Arm = Tuple[str, int, Perm, bool]

#: Idle-to-movement contribution ratios, log-spaced in powers of four and symmetric about
#: 1.0, so the movement-dominated and idle-dominated halves get equal resolution -- the
#: question is where the ordering flips, and an asymmetric sweep would bias the answer.
#:
#: Wider than the 1/16..16 this figure first used, and the width is the point. Under
#: `MOVEMENT_BALANCE` the two mechanisms' contributions here run from 300:1 movement to
#: 96:1 idling, so both ends are genuinely dominated by what the axis label claims. The
#: first version spanned a factor of 256 in the *scalars* and, because the scalars buy
#: unequal amounts of error, never left the idle-dominated half at all -- which is why its
#: movement end came out bunched.
#:
#: `MOVEMENT_BALANCE` is a workload average over a bimodal distribution, so ratio = 1 is the
#: balance point on average rather than for any given circuit: `ising_n26` balances near
#: ratio 3, the near-chain benchmarks nearer ratio 20. The span covers both.
DEFAULT_RATIOS: Tuple[float, ...] = (
    1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0, 256.0,
)

#: Matches `lsr_tvd_vs_k`, `ablation_vs_k` and `improvement_vs_locality_and_k`, so the four
#: k figures sweep the same family. Five values also exactly fills the house palette, which
#: is what keeps panel (a) distinguishable in greyscale.
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: Program-qubit window for the default circuit set. The floor is what gives every code
#: size a distinct patch count; the ceiling keeps the sweep inside an hour.
DEFAULT_QUBIT_RANGE: Tuple[int, int] = (24, 40)

#: Program gates above this are dropped from the default set. `load_proxy_circuits`'
#: own `gate_limit` filters the pre-proxy circuit, which lets a 9800-gate benchmark
#: through and costs 22s per selector call on its own.
DEFAULT_GATE_LIMIT = 1500

#: Series name for the resource selector's per-circuit choice.
SELECTOR = "selector"

#: Program gates below which a benchmark has essentially no packing decision in it. At
#: 24-40 qubits a circuit this small is a chain -- `ghz_n40` is 40 gates on 40 qubits,
#: `cat_n35` 35 on 35 -- so every placement is nearly as good as every other and the
#: circuit contributes a ratio near 1.0 to every cell.
#:
#: These are kept in the figure rather than filtered out, and `print_summary` reports the
#: sweep with and without them. Excluding them would sharpen the result while narrowing the
#: claim, and the dilution is worth seeing: it is the difference between "on our benchmark
#: set" and "on the benchmarks where the decision exists", and a reader is entitled to both.
TRIVIAL_GATE_COUNT = 100

#: Panel (b) lines, in draw order. `(key, legend label, palette index)`.
#:
#: The selector takes palette position 1 -- blue, circle, solid -- which is the AQORN
#: series' position in every other figure in this suite. The oracle takes position 2 as
#: the bound it is. The best fixed code size takes position 4, `#636363`, which the style
#: reserves for reference lines: one code size for the whole sweep is exactly the reference
#: this figure argues against.
#: "best *fixed* k per point", not "best k per point". The distinction is load-bearing: this
#: line is the best of the fixed-k arms, every one of which is a tightest packing placed by
#: `plan_packing`, so it is an oracle over *code size only*. The selector chooses code size,
#: patch count and placement, and measurably exceeds it -- 5.94x against 4.42x at the
#: movement-dominated end. Labelled "best k per point" it read as a ceiling being violated,
#: which looks like a bug in the figure rather than the result it is.
RELATIVE_LINES: Sequence[Tuple[str, str, int]] = (
    ("oracle", "best fixed $k$ per point", 1),
    (SELECTOR, "AQORN selector", 0),
    ("best_fixed", "one fixed $k$ throughout", 3),
)


@dataclass
class Settings:
    ratios: Sequence[float] = DEFAULT_RATIOS
    k_values: Sequence[int] = DEFAULT_K_VALUES
    circuits: Optional[Sequence[str]] = None
    qubit_range: Tuple[int, int] = DEFAULT_QUBIT_RANGE
    gate_limit: int = DEFAULT_GATE_LIMIT
    #: Fixed *raw* shot budget per point -- see the module docstring. Not scaled by the
    #: calibrated selection rate, because the shots post-selection discards are the result.
    repetitions: int = 40_000
    #: Code size the single per-circuit alpha calibration is solved at. k = 2 is the
    #: paper's code and panel (b)'s reference, so the sweep is centred on the choice it
    #: is arguing against.
    reference_k: int = 2
    #: What the *selector* charges per crossing per unit patch separation. Uncalibrated --
    #: see the module docstring. One two-qubit gate error by default.
    movement_error: float = 3.5e-3
    #: False: each plan is priced under the ratio the simulator will charge, so a line is
    #: the whole compiler at that code size. True isolates the code size from the
    #: planner's response to the regime.
    planner_blind: bool = False
    benchmark_root: Path = field(default_factory=lambda: Path("benchmarks/qasmbench"))
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None


def knobs_for(settings: SimSettings, movement_error: float) -> NoiseKnobs:
    """
    `NoiseKnobs` carrying the same idle/distance balance the simulator will charge.

    `SimSettings.with_noise_ratio` moves two scalars: `idle_time_scalar` multiplies the
    dephasing a qubit accrues sitting out a layer, and `distance_scalar` scales the
    movement channel -- in the simulator it slows the atoms *and* multiplies
    `move_error`, so the selector mirrors both. `layer_time` carries the idle scalar.
    This gives the selector the same regime the measurement is taken in; anything else
    and panel (b) would be asking the selector about a machine panel (a) never
    simulated.

    `movement_error` (the settings knob) used to be an opaque per-crossing-per-hop
    coefficient; under the grouped-transport `NoiseKnobs` it scales `move_error`
    relative to its hardware baseline instead, keeping the CLI axis meaningful --
    the default 3.5e-3 maps to 3.5x the baseline handling error.
    """
    base = NoiseKnobs()
    handling_scale = movement_error / 1e-3  # the settings knob, as a multiple of baseline
    speed = base.movement_speed
    if settings.distance_scalar > 0:
        speed = base.movement_speed / settings.distance_scalar
    return replace(
        base,
        layer_time=base.layer_time * settings.idle_time_scalar,
        move_error=min(1.0, base.move_error * handling_scale * settings.distance_scalar),
        movement_speed=speed,
    )


def load_circuits(settings: Settings) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """The benchmark set, filtered to the width window that gives every k a patch count."""
    root = settings.benchmark_root
    if not root.exists():
        raise FileNotFoundError(
            f"benchmark root {root} not found. Run from the repository root, pass "
            f"--benchmark-root, or initialise the submodule with "
            f"`git submodule update --init --recursive`."
        )
    low, high = settings.qubit_range
    # An explicitly named circuit is honoured whatever its width, so a smoke test or a
    # one-off check is not silently filtered out by the default window.
    loaded = load_proxy_circuits(root, qubit_limit=None if settings.circuits else high)
    if settings.circuits:
        wanted = set(settings.circuits)
        chosen = [(name, c) for name, c in loaded if name in wanted]
        missing = wanted - {name for name, _ in chosen}
        if missing:
            raise ValueError(f"no such benchmark circuit: {sorted(missing)}")
        return chosen
    return [
        (name, circuit)
        for name, circuit in loaded
        if low <= circuit.num_qubits <= high and circuit.size() <= settings.gate_limit
    ]


def _placement(
    circuit: qiskit.QuantumCircuit, k: int, profile
) -> Tuple[Perm, bool]:
    """
    The compiler's placement at this code size, and the commutation setting it was priced
    under. Falls back to naive so a rejected plan costs a data point rather than the run.

    Translating with commutation in a state the plan was not priced under would describe a
    circuit the pipeline would never build, so the caller has to honour the second return
    value.
    """
    try:
        plan = plan_packing(
            circuit, k + 2, h_commutation="auto", verify=True, noise=profile
        )
        return plan.placement, plan.h_commutation
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return naive_packing(circuit.num_qubits, k), True


def compute(settings: Settings) -> pd.DataFrame:
    circuits = load_circuits(settings)
    if not circuits:
        raise RuntimeError(
            f"no benchmark circuit has {settings.qubit_range[0]}-"
            f"{settings.qubit_range[1]} qubits and <= {settings.gate_limit} gates"
        )
    print(
        f"[{NAME}] {len(circuits)} circuits, {len(settings.ratios)} ratios, "
        f"{len(settings.k_values)} code sizes + selector, "
        f"{settings.repetitions:,} raw shots/point"
    )

    # Priced at ratio 1 and reused for every cell when --planner-blind, so the placement
    # is the same object at every ratio and only the code size varies.
    blind_profile = noise_profile_for(SimSettings())
    jobs = [
        (settings, name, circuit, ratio, blind_profile)
        for name, circuit in circuits
        for ratio in settings.ratios
    ]
    produced = parallel_map(_run_column, jobs, jobs=settings.jobs, label="circuit x ratio")
    rows = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def _run_column(job) -> List[Dict]:
    """
    One (circuit, ratio) column: calibrate once at the reference k, measure every arm.

    Module-level and self-contained because `parallel_map` spawns rather than forks --
    the same shape as `1_best_k_phase_diagram._run_cell`. This loop body was serial and
    single-core; the sims inside it are the whole cost of the figure, and columns are
    independent. `commutation_pays` is memoised per column rather than per circuit now,
    which re-runs a translated view per ratio -- noise next to the simulations.
    """
    settings, name, circuit, ratio, blind_profile = job
    rows: List[Dict] = []
    if True:
        commutes: Dict[int, bool] = {}
        reference = settings.reference_k

        if True:
            sim = settings.sim.with_noise_ratio(ratio)
            profile = blind_profile if settings.planner_blind else noise_profile_for(sim)

            # Per (circuit, ratio), at the reference code size only -- see the module
            # docstring. Every k in this column is then measured at this one alpha, so the
            # column is a fair comparison placed where the measurement has power.
            calibration = calibrate(
                circuit,
                naive_packing(circuit.num_qubits, reference),
                sim,
                n_code=reference + 2,
                baseline_gate_count=circuit.size() * 3,
            )
            if not math.isfinite(calibration.alpha):
                return rows

            # The x value: what the reference configuration's schedule actually
            # charges at these scalars -- fixed at the calibration reference so the
            # axis is independent of every series, and alpha-invariant so the
            # calibrated alpha does not move it.
            try:
                regime = measured_regime(
                    circuit, naive_packing(circuit.num_qubits, reference),
                    n_code=reference + 2,
                    idle_time_scalar=sim.idle_time_scalar,
                    distance_scalar=sim.distance_scalar,
                )
            except Exception:  # noqa: BLE001 - a rejected reference costs the column
                return rows

            arms: List[Arm] = []
            for k in settings.k_values:
                placement, commute = _placement(circuit, k, profile)
                arms.append((f"k{k}", k, placement, commute))

            # The selector picks code size *and* patch count, so its arm is not one of
            # the fixed-k arms above even when the code size coincides.
            #
            # `ResourcePlan` does not carry a commutation decision, so it is recomputed
            # with the same predicate `select_resources` used internally rather than
            # assumed. Forcing the pass on instead would mistranslate the selector's arm
            # in half of these cells -- `commutation_pays` comes out False for 33 of the
            # 65 (circuit, code size) pairs in the default set -- and biasing the one
            # series the figure is an argument for is the last place to guess.
            try:
                plan = select_resources(
                    circuit, noise=knobs_for(sim, settings.movement_error)
                )
                if plan.n_code not in commutes:
                    commutes[plan.n_code] = commutation_pays(circuit, plan.n_code)
                arms.append((SELECTOR, plan.k, plan.placement, commutes[plan.n_code]))
            except Exception:  # noqa: BLE001 - a refused selection costs one arm
                pass

            for series, k, placement, commute in arms:
                config = build_config(
                    sim,
                    n_code=k + 2,
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
                        "k": k,
                        "n_code": k + 2,
                        "patches": len({patch for patch, _ in placement}),
                        "alpha": calibration.alpha,
                        "repetitions": settings.repetitions,
                        "h_commutation": commute,
                        "planner_blind": settings.planner_blind,
                        "movement_error": settings.movement_error,
                        # Recorded, never plotted -- not estimable at these widths.
                        # See the module docstring.
                        "lsr": measured["lsr"],
                        "tvd": measured["tvd"],
                        "one_qubit_gates": measured["one_qubit_gates"],
                        "two_qubit_gates": measured["two_qubit_gates"],
                        "circuit_depth": measured["circuit_depth"],
                    }
                )

    return rows


def complete_circuits(frame: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    """
    Drop any circuit missing a fixed-k arm at any ratio, so the panels share a denominator.

    §3.3 of `STATUS.md` records what happens without this: a third of the benchmarks
    refused to place at k > 2, and the surviving columns covered a smaller and easier
    subset, which reversed the conclusion. A partial circuit here would also break panel
    (b), whose oracle is a max over `k` and would silently be a max over whichever arms
    happened to survive.
    """
    wanted = {f"k{k}" for k in k_values}
    ratios = frame["ratio"].nunique()
    keep = []
    for name, group in frame.groupby("circuit"):
        present = group[group["series"].isin(wanted)]
        counts = present.groupby("series")["ratio"].nunique()
        if len(counts) == len(wanted) and (counts == ratios).all():
            keep.append(name)
    dropped = sorted(set(frame["circuit"]) - set(keep))
    if dropped:
        print(f"[{NAME}] dropped for incomplete k coverage: {', '.join(dropped)}")
    return frame[frame["circuit"].isin(keep)].copy()


def _log_band(values: np.ndarray, *, standard_error: bool = False) -> Tuple[float, float, float]:
    """
    Geometric mean and its spread, computed in log space.

    Selection rate compounds -- it is `exp(-E)` for a sum of independent error terms -- so
    the arithmetic mean over circuits is pulled around by whichever circuit happens to sit
    highest, and an arithmetic SD can put the lower edge of the band below zero on a log
    axis. In log space the spread is symmetric and the band edges stay positive.

    `standard_error` divides by `sqrt(n)`, and every band the figure draws uses it, following
    `depth_vs_gates_tradeoff`, which made the same call for the same reason. Both panels plot
    a ratio taken *within* each circuit, so the circuit has already been cancelled; the SD
    across circuits then describes how much benchmarks differ in how much they benefit, which
    is a real quantity but is not the uncertainty on the mean the line draws. The house
    default is +-1 SD, so the caption has to say which one this is.

    `absolute_curves` leaves it off, because the absolute rate across benchmarks is genuinely
    a spread rather than an estimate of one number. Nothing plots that -- it feeds the printed
    table only -- so the two never appear in the same panel.

    Non-positive and non-finite values are dropped rather than clamped. A selection rate of
    exactly zero is a point where post-selection kept nothing, which has no logarithm and no
    honest substitute; an infinity can only arrive from a ratio whose denominator was such a
    point. Either one would take the whole geometric mean with it.
    """
    finite = np.asarray(
        [v for v in values if v == v and v > 0.0 and math.isfinite(v)], dtype=float
    )
    if finite.size == 0:
        return math.nan, math.nan, math.nan
    logs = np.log(finite)
    centre, spread = float(logs.mean()), float(logs.std(ddof=0))
    if standard_error and finite.size > 1:
        spread /= math.sqrt(finite.size)
    return math.exp(centre), math.exp(centre - spread), math.exp(centre + spread)


def absolute_curves(
    frame: pd.DataFrame, k_values: Sequence[int], ratios: Sequence[float]
) -> Dict[int, np.ndarray]:
    """`{k: array of (mean, low, high) per ratio}` -- panel (a)."""
    curves: Dict[int, np.ndarray] = {}
    for k in k_values:
        rows = frame[frame["series"] == f"k{k}"]
        curves[k] = np.array(
            [_log_band(rows.loc[rows["ratio"] == r, "lsr"].to_numpy()) for r in ratios]
        )
    return curves


def relative_curves(
    frame: pd.DataFrame, k_values: Sequence[int], ratios: Sequence[float], reference_k: int
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Selection rate relative to the reference code size, paired within circuit -- panel (b).

    Paired, not a ratio of means. Circuit-to-circuit spread in absolute selection rate is
    far larger than the code-size effect, so an unpaired ratio would mostly report which
    benchmarks happened to land in which arm. Dividing inside a circuit cancels the
    circuit and leaves the resource choice.

    `best_fixed` is one code size for the whole sweep, picked with hindsight on this very
    data -- the strongest fixed choice a paper could make, and therefore the honest thing
    for a per-circuit selector to be measured against.
    """
    wide = frame.pivot_table(
        index=["circuit", "ratio"], columns="series", values="lsr", aggfunc="mean"
    )
    reference = f"k{reference_k}"
    fixed = [f"k{k}" for k in k_values if f"k{k}" in wide.columns]
    if reference not in wide.columns:
        raise RuntimeError(f"reference series {reference} absent from the data")

    # A reference of exactly zero -- post-selection kept nothing at that point -- makes every
    # ratio on that row infinite. Dropped to NaN here rather than at the aggregation, so the
    # oracle's max and the hindsight scoring below both skip the row instead of one of them
    # returning inf and the other silently disagreeing.
    denominator = wide[reference].replace(0.0, np.nan)
    ratio_of = wide[fixed + ([SELECTOR] if SELECTOR in wide.columns else [])].div(
        denominator, axis=0
    )
    ratio_of["oracle"] = wide[fixed].max(axis=1).div(denominator)

    # One code size for the whole sweep: the fixed arm with the best geometric mean
    # relative rate across every circuit and every ratio.
    scores = {
        column: float(
            np.exp(
                np.log(
                    ratio_of[column].replace([np.inf, -np.inf, 0.0], np.nan).dropna()
                ).mean()
            )
        )
        for column in fixed
    }
    scores = {name: value for name, value in scores.items() if math.isfinite(value)}
    if not scores:
        raise RuntimeError("no fixed code size has a usable selection rate")
    best_name = max(scores, key=lambda column: scores[column])
    ratio_of["best_fixed"] = ratio_of[best_name]

    # Every column, not just panel (b)'s three: panel (a) draws the per-k lines from the
    # same normalisation, so the two panels cannot disagree about what "relative" means.
    curves = {
        key: np.array(
            [
                _log_band(
                    ratio_of.loc[
                        ratio_of.index.get_level_values("ratio") == r, key
                    ].to_numpy(),
                    standard_error=True,
                )
                for r in ratios
            ]
        )
        for key in ratio_of.columns
    }
    return curves, int(best_name[1:])


def print_summary(frame: pd.DataFrame, settings: Settings) -> int:
    ratios = sorted(frame["ratio"].unique())
    k_values = sorted(frame.loc[frame["series"].str.startswith("k"), "k"].unique())
    circuits = frame["circuit"].nunique()

    print(
        f"\n[{NAME}] logical selection rate, geometric mean of {circuits} circuits"
        f"   [{settings.repetitions:,} raw shots/point, alpha per (circuit, ratio) at "
        f"k={settings.reference_k}]"
    )
    patches = frame.groupby("k")["patches"].mean()
    print("  patches by k: " + ", ".join(f"k={k} {v:.1f}" for k, v in patches.items()))
    # Baseline acceptance across the axis. The first version let this run 44% -> 1.8%,
    # which is what put both ends of the sweep outside the measurable band, so it is
    # printed on every run rather than rediscovered.
    reference_rows = frame[frame["series"] == f"k{settings.reference_k}"]
    if not reference_rows.empty:
        span = reference_rows.groupby("ratio")["lsr"].mean()
        print(f"  baseline acceptance: {span.min():.3f} to {span.max():.3f} "
              f"across the ratio axis (flat is the goal -- see the docstring)")

    absolute = absolute_curves(frame, k_values, ratios)
    print("\n  (a) selection rate      rows: ratio, cols: k")
    print("           " + "".join(f"{'k='+str(k):>9}" for k in k_values) + "    argmax")
    for index, ratio in enumerate(ratios):
        cells = [absolute[k][index, 0] for k in k_values]
        best = k_values[int(np.nanargmax(cells))]
        print(
            f"   {ratio:>6.4g} " + "".join(f"{v:>9.4f}" for v in cells) + f"    k={best}"
        )

    relative, best_fixed = relative_curves(frame, k_values, ratios, settings.reference_k)
    labels = {key: label for key, label, _ in RELATIVE_LINES}
    print(f"\n  (b) rate relative to k={settings.reference_k}   "
          f"(best single k over the whole sweep: k={best_fixed})")
    present = [key for key, _, _ in RELATIVE_LINES if key in relative]
    print("           " + "".join(f"{labels[key]:>22}" for key in present))
    for index, ratio in enumerate(ratios):
        print(
            f"   {ratio:>6.4g} "
            + "".join(f"{relative[key][index, 0]:>22.3f}" for key in present)
        )
    for key in present:
        column = relative[key][:, 0]
        print(f"    {labels[key]:>22}: {np.nanmin(column):.2f}x to "
              f"{np.nanmax(column):.2f}x, geometric mean "
              f"{math.exp(float(np.nanmean(np.log(column)))):.2f}x")

    # What the caption quotes: the selector against each of the two fixed-resource
    # alternatives, stated as direct ratios.
    #
    # An earlier version reported this as "% of the oracle's headroom recovered", which was
    # wrong in a way the data made obvious -- it printed 171%. That framing treats the
    # per-point fixed-k line as a ceiling, and it is not one: it is an oracle over code size
    # alone, while the selector also chooses patch count and searches placements, so
    # exceeding it is an available outcome rather than a contradiction. Two plain ratios say
    # what happened without implying a bound that does not exist.
    if SELECTOR in relative and "oracle" in relative and "best_fixed" in relative:
        def geo(key: str) -> float:
            return math.exp(float(np.nanmean(np.log(relative[key][:, 0]))))

        chosen, oracle, fixed = geo(SELECTOR), geo("oracle"), geo("best_fixed")
        print(f"\n  selector vs one fixed k throughout: {chosen / fixed:.2f}x")
        print(f"  selector vs best fixed k per point:  {chosen / oracle:.2f}x"
              f"   ({'ahead' if chosen > oracle else 'behind'} -- the selector also "
              f"chooses\n{'':39}patch count and placement, so this can exceed 1.00x)")

    _print_trivial_split(frame, settings, k_values, ratios)
    return best_fixed


def _print_trivial_split(
    frame: pd.DataFrame,
    settings: Settings,
    k_values: Sequence[int],
    ratios: Sequence[float],
) -> None:
    """
    The same headline numbers with the near-chain benchmarks removed.

    They stay in the figure -- see `TRIVIAL_GATE_COUNT` -- so this is the honest way to show
    what they cost: a circuit with no placement decision in it contributes a ratio near 1.0
    to every cell and drags the mean toward "nothing happens". Printing both means the
    figure's number and the number for circuits where the decision exists are both quotable,
    and neither can be presented as the other.
    """
    gates = frame.groupby("circuit")["program_gates"].first()
    trivial = sorted(gates[gates < TRIVIAL_GATE_COUNT].index)
    if not trivial:
        print(f"\n  no circuit is below {TRIVIAL_GATE_COUNT} gates; nothing to split out")
        return

    substantive = frame[~frame["circuit"].isin(trivial)]
    print(f"\n  circuits under {TRIVIAL_GATE_COUNT} gates (no packing decision to make): "
          f"{', '.join(trivial)}")
    if substantive["circuit"].nunique() < 2:
        print("  too few circuits remain to report a split")
        return

    curves, best = relative_curves(
        substantive, k_values, ratios, settings.reference_k
    )
    labels = {key: label for key, label, _ in RELATIVE_LINES}
    print(f"  excluding them ({substantive['circuit'].nunique()} circuits, "
          f"best single k={best}):")
    for key, label, _ in RELATIVE_LINES:
        if key not in curves:
            continue
        column = curves[key][:, 0]
        print(f"    {labels[key]:>22}: {np.nanmin(column):.2f}x to "
              f"{np.nanmax(column):.2f}x, geometric mean "
              f"{math.exp(float(np.nanmean(np.log(column)))):.2f}x")


def _tick_label(ratio: float) -> str:
    """`1/16` rather than `0.0625`: the axis is a ratio and reads as one."""
    if ratio >= 1:
        return f"{ratio:g}"
    inverse = 1.0 / ratio
    return f"1/{inverse:.0f}" if abs(inverse - round(inverse)) < 1e-6 else f"{ratio:g}"


#: Log-axis tick positions for a ratio panel, in decimal rather than as powers of ten. A
#: ratio axis clustered around 1.0 gets exactly one decade tick, and matplotlib labels the
#: rest `2 x 10^0` -- which the style guide rules out, and which a reader has to decode to
#: compare against a number in the prose.
#:
#: Written out symmetrically rather than generated as `1/x` from an ascending ladder: the
#: reciprocal of a ladder containing 1.5 and 3.0 is 0.6667 and 0.3333, and a tick reading
#: `0.6667` on a figure whose quantities are quoted to two significant figures is worse than
#: the scientific notation it replaced.
_RATIO_TICKS = (0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)

#: At most this many ticks per log axis. Beyond it the 9pt labels collide at column width,
#: which is how `0.8` and `0.75` ended up overprinting each other.
_MAX_TICKS = 7


#: Span above which the fine ratio ladder is dropped for the 1/2/5 decade ladder. A panel
#: covering 0.5 to 5 gets ten fine candidates, thins to `0.5, 1, 1.5, 2.5, 4`, and reads as
#: an arbitrary set; over that much range the decade ladder gives `0.5, 1, 2, 5` instead.
_FINE_LADDER_SPAN = 4.0


def _log_ticks(low: float, high: float, *, decades: bool) -> List[float]:
    """Ticks covering `[low, high]` on a log axis, as plain decimals, thinned to fit."""
    if not decades and high / max(low, 1e-9) > _FINE_LADDER_SPAN:
        decades = True
    candidates = (
        [mantissa * 10.0**power for power in range(-4, 1) for mantissa in (1.0, 2.0, 5.0)]
        if decades
        else list(_RATIO_TICKS)
    )
    inside = [value for value in candidates if low <= value <= high]
    if not inside:
        return [low, high]
    # Thin by dropping every other tick, keeping 1.0 if it is present, until it fits.
    while len(inside) > _MAX_TICKS:
        anchor = inside.index(1.0) if 1.0 in inside else 0
        inside = [
            value for index, value in enumerate(inside) if (index - anchor) % 2 == 0
        ]
    return inside


def _plain(value: float, _position=None) -> str:
    """`0.05`, `1.5`, `10` -- never `5 x 10^-2`."""
    if value >= 1:
        return f"{value:g}"
    text = f"{value:.4f}".rstrip("0")
    return text.rstrip(".") if text.endswith(".") else text


def _draw(axis, x, curve: np.ndarray, colour: str, marker: str, dashes, label: str) -> None:
    """One series: mean line plus its spread band at the house alpha."""
    mean, low, high = curve[:, 0], curve[:, 1], curve[:, 2]
    axis.plot(x, mean, color=colour, marker=marker, linestyle=dashes,
              label=label, markersize=3.2, linewidth=1.2)
    axis.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0)


#: Palette positions panel (a)'s per-k lines take, one per code size above the reference.
#:
#: Consecutive, which took two tries. Skipping index 3 -- `#636363`, the grey the style
#: reserves for reference lines and which the 1.0 rule uses -- looked like the careful choice
#: and was worse: the style's linestyle cycler is (solid, dashed, dash-dot, dotted, solid),
#: so positions 1 and 5 are *both solid*, and slots (0, 1, 2, 4) gave `k=4` and `k=10` the
#: same line with only a circle-versus-triangle marker between them. That is exactly the
#: greyscale failure the redundant encoding exists to prevent, and it is visible in print.
#:
#: Consecutive slots keep all four series mutually unique in colour, marker *and* line style.
#: `k=8` then shares `#636363` with the 1.0 rule, which is fine because the rule carries no
#: marker and the series does: dotted-with-diamonds against plain dashed reads cleanly, and
#: the rule is not data competing to be told apart from data.
_SERIES_SLOTS: Tuple[int, ...] = (0, 1, 2, 3)

#: How a code size is named in a label. `[[n, k, 2]]` is what the prose and `docs/notation.md`
#: use; a bare `k` in an axis label makes the reader decode a parameter before they can read
#: the axis, which is the complaint that produced this constant.
_CODE = "$[[{n},{k},2]]$"

#: Opaque white behind a legend, with no border drawn.
#:
#: The house style asks for frameless legends, and a frameless legend is what this still
#: looks like -- no edge is rendered. The backing is here because both panels carry a
#: full-width dashed rule at 1.0 that otherwise prints straight through the legend text, and
#: no legend position avoids it: the rule spans the whole axis by construction. Transparency
#: would keep the collision; a visible frame would be the larger departure.
_LEGEND_BACKING = {
    "frameon": True,
    "framealpha": 1.0,
    "facecolor": "white",
    "edgecolor": "none",
}


def _measured_positions(frame: pd.DataFrame, ratios) -> list:
    """Geometric-mean measured reference ratio per nominal column -- the plotted x."""
    positions = []
    for ratio in ratios:
        values = frame[frame["ratio"] == ratio]["measured_ratio_ref"].to_numpy()
        values = values[np.isfinite(values)]
        positions.append(float(np.exp(np.log(values).mean())) if len(values) else float("nan"))
    return positions


def plot(frame: pd.DataFrame, settings: Settings, best_fixed: int) -> List[Path]:
    use_style()
    ratios = sorted(frame["ratio"].unique())
    x_measured = _measured_positions(frame, ratios)
    k_values = sorted(frame.loc[frame["series"].str.startswith("k"), "k"].unique())
    relative, _ = relative_curves(frame, k_values, ratios, settings.reference_k)

    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    palette, markers, dashes = cycle["color"], cycle["marker"], cycle["linestyle"]

    # Two stacked panels at exactly one column. Taller than the style default because two
    # panels share one x axis; the width is untouched, which is what keeps type at 10pt.
    fig, axes = plt.subplots(
        nrows=2, figsize=(COLUMN_WIDTH, 4.15), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.82]},
    )
    top, bottom = axes

    # Both panels are normalised to the reference code size, and the 1.0 rule is the thing
    # they are normalised to.
    #
    # Panel (a) plots the *relative* rate rather than the absolute one, which was the first
    # version and did not work: absolute selection rate falls 25x across this sweep while
    # the code sizes separate by about 10%, so five lines arrived on top of each other and
    # the panel read as "k does not matter" -- the opposite of the finding. Dividing out the
    # common fall leaves exactly the re-ordering the figure is about. The absolute scale it
    # gives up belongs in the caption, which is where a reader looks for it anyway.
    for axis in axes:
        axis.axhline(1.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)

    for slot, k in zip(_SERIES_SLOTS, [k for k in k_values if k != settings.reference_k]):
        key = f"k{k}"
        if key in relative:
            _draw(top, x_measured, relative[key], palette[slot], markers[slot],
                  dashes[slot], f"$k={k}$")

    for key, label, index in RELATIVE_LINES:
        if key not in relative:
            continue
        text = f"{label} ($k={best_fixed}$)" if key == "best_fixed" else label
        _draw(bottom, x_measured, relative[key], palette[index % len(palette)],
              markers[index % len(markers)], dashes[index % len(dashes)], text)

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(axis="y", which="major")
        # Plain decimal labels: both panels are now ratios clustered around 1.0, where
        # matplotlib's default log formatter prints `2 x 10^0`.
        low, high = axis.get_ylim()
        axis.set_yticks(_log_ticks(low, high, decades=False))
        axis.yaxis.set_major_formatter(plt.FuncFormatter(_plain))
        axis.yaxis.set_minor_formatter(plt.NullFormatter())
        axis.set_ylim(low, high)

    # The code the paper already uses, spelled as a code rather than as a bare parameter.
    # "rate relative to k = 2" was the first label and made a reader decode `k` before they
    # could read the axis at all; `[[4,2,2]]` is the object the prose names.
    reference = _CODE.format(k=settings.reference_k, n=settings.reference_k + 2)
    top.set_ylabel(f"shots kept, vs. {reference}")
    # Lower left, which is the corner the data cannot reach: every larger code sits at or
    # above the reference where movement dominates, so the bottom of the movement-dominated
    # end is empty by construction.
    # Titled, so `k` is defined inside the figure. Without it the entries read as a bare
    # parameter and the reader has to go to the caption to learn what is being varied.
    #
    # Upper right in (a), lower left in (b). Every series descends monotonically across this
    # axis, so those are the two corners no line reaches -- and which one is free differs
    # between the panels because (b) starts higher. Both legends were on the wrong side when
    # the axis was narrow and the lines were flat.
    legend = top.legend(ncol=2, columnspacing=0.9, handlelength=1.5, handletextpad=0.4,
                        borderpad=0.2, labelspacing=0.3, loc="upper right",
                        title="logical qubits per patch, $k$", **_LEGEND_BACKING)
    legend.get_title().set_fontsize(8)
    top.set_title("(a)", loc="left", fontweight="bold")

    bottom.set_ylabel(f"shots kept, vs. {reference}")
    bottom.set_xlabel("measured idling / movement noise ratio")
    bottom.legend(ncol=1, handlelength=1.5, handletextpad=0.4, borderpad=0.2,
                  labelspacing=0.3, loc="upper right", **_LEGEND_BACKING)
    bottom.set_title("(b)", loc="left", fontweight="bold")
    # Which end of the axis is which, hung under the two extreme ticks rather than written
    # inside the panel. In-plot corner annotations were the first attempt and they collide
    # with whichever legend is in the free corner -- and which corner is free changes with
    # the data. Tick labels are outside the data area, so they cannot.
    # Every other ratio is labelled, and all of them keep a tick mark. Nine labels do not fit
    # across one column: `1/256` is about 0.35in at 9pt against 0.29in of room per tick, so
    # the two movement-dominated labels overprinted each other. Labelling alternate ticks
    # lands on the powers of sixteen -- 1/256, 1/16, 1, 16, 256 -- which is a readable ladder
    # and keeps the two ends, where the regime annotations hang.
    labels = [
        f"{x:.2g}" if index % 2 == 0 and np.isfinite(x) else ""
        for index, x in enumerate(x_measured)
    ]
    labels[0] += "\nmovement"
    labels[-1] += "\nidling"
    bottom.set_xticks(x_measured)
    bottom.set_xticklabels(labels)
    bottom.xaxis.set_minor_locator(plt.NullLocator())

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override)
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
                        help="Code sizes drawn in panel (a), in the [[k+2, k, 2]] family.")
    parser.add_argument("--circuits", nargs="+", default=None,
                        help="Benchmark names. Default: every proxy inside "
                             "--qubit-range under --gate-limit gates.")
    parser.add_argument("--qubit-range", nargs=2, type=int,
                        default=list(defaults.qubit_range), metavar=("LOW", "HIGH"),
                        help="Program-qubit window. The floor is what gives every code "
                             "size a distinct patch count; below ~24 the ladder "
                             "saturates and there is no crossover to find.")
    parser.add_argument("--gate-limit", type=int, default=defaults.gate_limit,
                        help="Drop proxies above this many gates.")
    parser.add_argument("--repetitions", type=int, default=defaults.repetitions,
                        help="Raw shots per point, held fixed rather than scaled by the "
                             "calibrated selection rate -- the discarded shots are the "
                             "result here.")
    parser.add_argument("--reference-k", type=int, default=defaults.reference_k,
                        help="Code size alpha is calibrated at and panel (b) is drawn "
                             "against.")
    parser.add_argument("--movement-error", type=float, default=defaults.movement_error,
                        help="What the selector charges per crossing per patch hop. "
                             "Uncalibrated; affects panel (b)'s selector line only.")
    parser.add_argument("--planner-blind", action="store_true",
                        help="Price every plan at ratio 1, isolating the code size from "
                             "the planner's own response to the noise regime.")
    parser.add_argument("--benchmark-root", type=Path, default=defaults.benchmark_root)
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it here: a full "
                             "run is roughly an hour of simulation.")
    args = parser.parse_args()
    return Settings(
        ratios=args.ratios,
        k_values=args.k_values,
        circuits=args.circuits,
        qubit_range=(args.qubit_range[0], args.qubit_range[1]),
        gate_limit=args.gate_limit,
        repetitions=args.repetitions,
        reference_k=args.reference_k,
        movement_error=args.movement_error,
        planner_blind=args.planner_blind,
        benchmark_root=args.benchmark_root,
        sim=settings_from_args(args, defaults.sim),
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        jobs=args.jobs,
    )


def main() -> None:
    settings = parse_args()

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
        if "measured_ratio_ref" not in frame.columns:
            raise SystemExit(
                f"[{NAME}] {settings.from_csv} predates the measured-regime axis "
                "(no measured_ratio_ref column) -- regenerate rather than replot."
            )
    else:
        if settings.reference_k not in settings.k_values:
            raise ValueError(
                f"--reference-k {settings.reference_k} must be one of --k-values "
                f"{list(settings.k_values)}; panel (b) is drawn against it."
            )
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            from experiments.shared.figures import data_dir

            target = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(target, index=False)
            print(f"[{NAME}] persisted to {target}")

    frame = complete_circuits(frame, settings.k_values)
    if frame.empty:
        raise RuntimeError("no circuit placed at every swept code size")
    best_fixed = print_summary(frame, settings)
    for path in plot(frame, settings, best_fixed):
        print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
