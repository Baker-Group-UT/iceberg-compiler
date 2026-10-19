"""
improvement_vs_noise_and_k.py -- what packing is worth, against the machine it runs on.

**Claim.** AQORN's advantage over naive packing is a property of the hardware as much as
of the compiler. Two facts about a machine move it, and they move it for different
reasons: *which* noise mechanism dominates -- qubits idling, or gates travelling between
patches -- and *how good* the machine is overall. This figure measures both, on real
QASMBench programs, across the `[[k+2, k, 2]]` code family.

It is the benchmark answer to `improvement_vs_locality_and_k`, which asked the same
question of synthetic circuits over a circuit-structure axis. `improvement_vs_benchmark_locality`
already tried carrying that axis to real programs and found it does not transfer: locality
predicts nothing about which benchmark benefits (r = 0.04 on selection rate against 0.91
synthetic). So the second axis here is the machine rather than the circuit, which is the
axis a hardware reader is actually choosing along.

Two sweeps, one run, one CSV:

    regime  x = idle / movement noise balance, movement-dominated on the left.
            `SimSettings.with_noise_ratio` pins the geometric mean, so a column differs
            from its neighbour in which mechanism dominates and *not* in how much total
            noise there is. Five columns at `distance_scalar` 8, 4, 2, 1, 0.5 -- the span
            the cost model is calibrated for, sweeping through `distance_scalar = 1`, which
            is the machine as calibrated. See `DEFAULT_RATIOS`.
    yield   x = overall machine quality, expressed as the selection rate the *naive*
            baseline achieves. Five columns from 5% to 51%, worse hardware on the left.

**Its first claim was written off one circuit and the corpus reversed it. The cause was a
cost-model bug, now fixed -- read to the end before quoting anything here.** An earlier version of this docstring read "both gradients are strong and
monotone ... selection-rate improvement runs +57.6% at measured ratio 0.003 down to +7.4%
at 40.7", measured at k = 4 on three circuits under `--max-circuits 3`. Those three are the
*narrowest* three, and the number came almost entirely from one of them: `seca_n11` at
+172.1%, against `multiply_n13` at +1.4% and `bv_n14` at -2.5%.

Over 28 circuits the same cell is mean **-29.5%**, median -32.4%, sd 55.2 -- and -39.9%
with those three removed. The structured families sit at the floor: `ising_n26`, `ising_n34`
and `ising_n42` all near -98%, `ghz_n40` -98%, the `cat` family -48% to -80%. A -98%
selection-rate improvement means AQORN keeps almost no shots where naive keeps some, which
is far outside sampling error.

**The other three metrics stay positive everywhere, and that combination is the diagnosis.**
Two-qubit gates run +1.3% to +4.8% and depth +3% to +13% across the whole grid, so the
planner is returning a shorter, cheaper circuit that post-selects far worse in exactly the
columns that charge for movement. That is the signature of the planner's movement price
disagreeing with what the simulator charges, not of packing being harmful: `noise_profile_for`
applies `distance_scalar` once, and at the leftmost column `distance_scalar` is 128, so any
under-pricing there is amplified by that factor. The `na-packing-costs` work already flagged
ratio-axis figures as needing regeneration against the new pricing; this figure is the one
that shows why.

**Diagnosed: the planner's movement term is placement-blind, and this axis is what exposes
it.** `packing.cost.price` charges movement as

    hops = 1 + distance_scale * (mean_patch_hops(patches, n_code + 2) - 1)

and `packing.tables.mean_patch_hops(num_patches, qubits_per_patch)` takes **no placement
argument** -- it is the *mean* centre-to-centre distance over all distinct patch pairs. So at
a fixed patch count every cross-patch gate is priced identically, whatever layout produced
it, and the objective reduces to crossing count plus depth. The simulator meanwhile charges
the *actual* per-pair distance, via `positions_for_circuit` and
`transport.derive_transport_groups`.

Measured on `ising_n34` at k = 4, both placements using 9 patches, so the planner's movement
term is literally equal for the two:

    distance_scalar   arm     movement    idle    total    2q   depth
    128               naive      7.64   0.023     9.55   542     103
    128               aqorn     25.65   0.155    27.66   526      81
    1                 naive      0.06   0.026     1.98   542     103
    1                 aqorn      0.13   0.067     2.09   542      99

AQORN buys 16 fewer two-qubit gates and 22 less depth with 3.4x the real movement. The
planner is not ignoring the knob -- its `movement_cost` is 185.07 at `distance_scalar = 128`
against 1.45 at 1, correctly scaled -- it simply cannot see that its own placement lengthened
the surviving crossings. `ising_n34` is a chain, so naive index order puts qubit `i` next to
`i+1` and its crossings are short; the community placement cuts their *count* and scatters
the remainder across the grid.

That is why the worst cells are the regular families (`ising`, `ghz`, `cat`): index order is
already near distance-optimal on a chain or a star, so there is nothing to win and a great
deal to lose. And it is why `merging_improvement_benchmark_stylized_simulated` looks healthy
on the same corpus -- it runs at `distance_scalar = 1`, where this error is worth about 5% of
total error rather than 3x, at k = 2 only, and its claim is about merging, which the bug does
not touch.

**Fixed, at cost model version 3. Everything above is the record of how it was found, not
the current state.** Four changes, all in `packing`, and `packing.version` is the changelog:

    plan.measured_cost      prices movement and idling on the translated circuit's real
                            schedule when handed a profile, where it counted gates and depth
                            only -- so `verify`'s "never worse than naive" stopped being
                            decided in a currency that omitted the dominant cost. It ranks
                            on expected error alone, which makes `depth_weight` inert under
                            a profile: depth is now a reported outcome, not something the
                            planner will trade error for.
    cost.price              no longer clamps movement to `MOVEMENT_SHARE_CEILING * work`.
                            The clamp removed the term's *gradient* wherever it bound.
    cost.price              charges per-pair distance via `tables.patch_hop_matrix` instead
                            of the placement-blind `mean_patch_hops`.
    plan_packing            verifies against the profile the caller handed it, not the
                            `NoiseProfile()` it defaults to, so search and check agree.

**The regime axis is narrowed to the span the model is calibrated for**, `distance_scalar`
0.5 to 8 -- see `DEFAULT_RATIOS`. The old outer columns were not a machine; they were the
model outside its envelope.

**What replaces the retracted claim is a different and better one, and it is about circuit
structure rather than about the noise axis.** `experiments/tests/placement_regression.py`
sweeps k x `distance_scalar` x structure shot-free and finds the gain concentrated by
interaction-graph shape:

    structure     mean error improvement     fell back to naive
    chain                  +0.04%                  15/16
    star                  +12.42%                   0/8
    irregular             +12.71%                   0/8

On a chain, program order already places `q[i]` next to `q[i+1]`, so naive is near
distance-optimal and the planner correctly declines to act -- which is why the old objective
lost so badly there, and why `ising`, `cat` and `ghz` were the -98% cells. The mean is
therefore bimodal, and a single average over the corpus understates the win on the circuits
packing can help and overstates it on the ones it cannot. Quote the split, not the mean.

**Every cached CSV predating version 3 is invalid**, because these changes alter what the
planner *chooses*. `experiments/check_stale_data.py` lists them. Only figures that pass a
noise profile are affected -- the fixes all gate on `movement_cost > 0` -- so the
`merging_improvement` family, which goes through `common.pack_program` without one, is
provably unchanged.

**Why the balance and not alpha.** Alpha is the calibration variable, not a free one.
Every simulated figure in this directory solves it per point so the naive baseline lands
near a target selection rate, because post-selection is an exponentially strict filter:
one fixed alpha across a corpus spanning 20 to 1283 gates leaves the short circuits
keeping nearly every shot and the long ones keeping none, and a cell average would then be
reporting which circuits still had shots. Sweeping the *target* instead keeps the
calibration and stays comparable across circuits, which is what the `yield` axis is.

**Why not distance or idle alone.** Moving one scalar with the other pinned confounds the
balance between the mechanisms with the total amount of noise, so a crossover could not be
attributed to either. `with_noise_ratio`'s docstring works through this; the geometric-mean
pin is the repaired version, and `MOVEMENT_BALANCE` is why `ratio = 1` means what it says.

**Columns are labelled by measurement, never by the knob.** The nominal ratio is a sweep
that generates coverage; the mapping from scalars to physical regime depends on the
circuit, so a nominal "ratio = 1" is a different point for every benchmark. Each regime
column is therefore labelled with the geometric mean of what `measured_regime` says the
schedule actually charged, read on a fixed naive k = 2 reference so the x axis cannot
restate the y axis. It is alpha-invariant -- `move_error` scales with alpha like every
other rate -- so calibration does not move the label. Each yield column is labelled with
the naive selection rate actually achieved there, which is what the column *is*.

**The two sweeps do not carry the same metrics, and that is forced.**
`noise_profile_for` reads `distance_scalar` and `idle_time_scalar` and nothing else, so:

    regime  the planner is handed a different profile per column, re-plans, and returns a
            different placement. Depth, gate counts, selection rate and TVD all move. This
            is the axis where `depth_vs_gates_tradeoff`'s finding lives -- packing trading
            depth for two-qubit gates, and reversing the trade when told movement is cheap.
    yield   the profile is identical in every column, so the placement is *bit-identical*
            and depth and gate counts are constant along the row by construction. Only the
            two stochastic outcomes, selection rate and TVD, carry information.

So the regime layouts plot selection rate against depth, and the yield layouts plot
selection rate against TVD. A depth panel on the yield axis would be five identical
columns rendered as a claim. `print_summary` measures that constancy rather than asserting
it, and prints the residual spread; anything but zero means the placement moved when it
should not have.

**Corpus: QASMBench medium and large, loaded exactly as
`merging_improvement_benchmark_stylized_simulated`** -- README gate limit 2000, qubit limit
150, proxy rebuilt with `replace_rz_rx_with_s=True`, and dropped above 2500 gates on the
loaded proxy. That is 42 circuits, 11 to 140 qubits (median 35), 30 to 1283 gates.

The small suite is deliberately excluded. It is 2 to 10 qubits wide, and a program
narrower than one patch has no placement to make: naive and AQORN return the same
permutation and the row is a tautology reading exactly 0%. Including it would pull the
large-k rows toward zero for a structural reason and the reader would see it as a result.
Every circuit kept here needs at least two patches at every k in the sweep, so **every
cell of both grids averages over the same 42 circuits** and none is empty.

**Miscompilation guard, ported from `improvement_vs_benchmark_locality`.** The pipeline can
silently produce a circuit that computes the wrong distribution: on `qpe_n9` at k = 2 the
combination (AQORN placement, merging on, H commutation off) returns a noiseless TVD of
0.87 against naive's 0.01 while the selection rate stays at 1.0, so post-selection does not
catch it and neither does `plan_packing(verify=True)`, which compares modelled cost rather
than correctness. Every point is therefore simulated once at zero noise first, and pairs
where AQORN departs from naive are dropped and named.

The judgement is *differential*. An absolute threshold cannot work, because the noiseless
TVD floor scales with the output support and this corpus reaches 140 qubits, where 8k shots
leave both arms at 1.0 with nothing wrong. Both arms of a pair share that floor exactly, so
only a gap between them means anything.

**TVD survives on most of this corpus, but not all of it, and the code says which rows.**
Above `TVD_RESOLUTION_LIMIT` the noiseless reference is already saturated, so the measured
TVD is sampling and not logical error. Measured at k = 4 over all 42 circuits, 36 resolve
and 6 do not: `qft_n18` at 0.972 and the whole ising family at ~1.000. Width is not the
predictor -- `bv_n140` resolves at 0.008 and `cat_n130` at 0.001, because those programs
concentrate their output on a handful of strings while ising spreads it. Saturated rows are
kept, since their gate counts are exact translation properties and their selection rate is
a binomial over ~150k shots, and they are withheld from the TVD panel only. `print_summary`
prints the surviving fraction; read it before quoting panel (b) of the yield layout.

**Parallel.** 42 circuits x 5 k x 2 axes is 420 work items, each sweeping its axis'
columns, and the points share nothing. `experiments.shared.parallel` fans them across
cores; `--jobs 1` runs in-process for a real traceback. `--max-circuits` trims it for a
smoke run.

**Budget roughly 14 hours on 12 workers, and do not try to buy it back with shots.** An
early estimate of 5 hours came from timing 18-26 qubit circuits and scaling by gate count.
That is the wrong cost model twice over.

First, the cost is not in the shots. Timing `evaluate` at 2k / 20k / 200k repetitions on
`knn_n25`:

    k = 2    20.2 s   33.4 s   56.2 s     fixed 19.9 s = 35% of the 200k run
    k = 10   25.0 s   26.4 s   39.3 s     fixed 24.9 s = 63% of the 200k run

A hundredfold change in shots moves the wall clock by 1.6-2.8x, because what dominates is
the fixed cost of translating and compiling the encoded circuit, paid once per call
whatever the shot count. So `--target-kept-shots` is close to a free parameter for
runtime, and cutting it only degrades TVD. The lever is the *number of calls*, which is
why the noiseless guard is cached -- see `run_job`.

Second, the scaling is in the width, not the gate count. `bv_n140` at k = 2 is 70 patches
x 4 = 280 physical qubits to translate against `seca_n11`'s 24. `load_benchmarks` sorts
narrowest-first so a `--max-circuits` smoke run finishes quickly, and the side effect is
that a full run decelerates as it climbs: measured 0.96 min/item over items 63-84, and
4.58 min/item over items 168-189. Do not extrapolate an ETA from the early items; the
runner's own linear estimate is optimistic for the same reason.

**Checkpointing is on by default and the run is long enough to need it.** Each item writes
its own rows on completion, so a killed run resumes instead of restarting -- the first full
run of this figure had nothing on disk at hour eight, because `parallel_map` returns only
once every item is done. The directory refuses to be reused under settings that would
change a computed row; `--fresh` discards it and `--no-checkpoint` opts out.

Run with:

    python -m experiments.figures.improvement_vs_noise_and_k.improvement_vs_noise_and_k
    python -m experiments.figures.improvement_vs_noise_and_k.improvement_vs_noise_and_k --from-csv <path>
    python -m experiments.figures.improvement_vs_noise_and_k.improvement_vs_noise_and_k --max-circuits 4 --k-values 2 6
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import qiskit

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import (
    COLUMN_WIDTH,
    check_provenance,
    FULL_WIDTH,
    annotate_cells,
    cell_edges,
    improvement_scale,
    save_figure,
    stamp_provenance,
    use_style,
)
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import (
    METRICS,
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    measured_regime,
    naive_packing,
    noise_profile_for,
    noiseless_tvd,
    settings_from_args,
)
from packing.plan import plan_packing

NAME = "improvement_vs_noise_and_k"
HERE = Path(__file__).resolve().parent

Perm = List[Tuple[int, int]]

#: Same metrics, same signs, same panel names as `improvement_vs_locality_and_k` and
#: `improvement_vs_benchmark_locality`, so the three figures read against each other
#: without translating.
PANELS: Dict[str, Tuple[str, bool]] = {
    "lsr": ("selection rate", True),
    "tvd": ("TVD", False),
    "one_qubit_gates": ("1q gates", False),
    "two_qubit_gates": ("2q gates", False),
    "circuit_depth": ("depth", False),
}

_UNKNOWN = set(PANELS) - set(METRICS)
if _UNKNOWN:
    raise ImportError(
        f"PANELS names metrics evaluate() does not return: {sorted(_UNKNOWN)}. "
        f"Available: {sorted(METRICS)}"
    )

#: The two sweeps. `REGIME` moves which mechanism dominates at fixed total noise; `YIELD`
#: moves how good the machine is at fixed mechanism balance.
REGIME = "regime"
YIELD = "yield"

#: Nominal idle/movement ratios. Coverage only -- the column is *labelled* with what the
#: schedule charged.
#:
#: **These are not the span the sibling ratio figures use, and the difference is measured
#: rather than stylistic.** `with_noise_ratio` sets both scalars to `sqrt(ratio)`, so the
#: regime the schedule actually charges grows as the *square root* of the knob. On this
#: corpus the fit is `measured ~ sqrt(nominal) / 5.1`, checked at eleven nominal values
#: against the geometric mean over six benchmarks:
#:
#:     nominal      1/64    1/4      1      16     64     256    4096    16384
#:     measured    0.024  0.097  0.195   0.799  1.652   3.515  18.873   49.849
#:
#: So `1/64 .. 64`, which is what `1_best_k_phase_diagram` sweeps, spans measured 0.024
#: to 1.65 -- it reaches the balance point and stops, and six of its seven columns sit in
#: the movement-dominated half. The exponent is a property of `with_noise_ratio`; the 5.1
#: is a property of this corpus, so re-measure before carrying these values to a different
#: circuit window.
#:
#: **Narrowed to the span the cost model is calibrated for.** This used to run
#: `1/256 .. 65536`, chosen to put nearly four decades of measured ratio on the page. The
#: outer columns of that span are not a machine, they are the model leaving its envelope:
#: `with_noise_ratio` pins the geometric mean, so nominal `1/256` means
#: `distance_scalar = 128` *and* `idle_time_scalar = 1/128`, and at that distance scalar the
#: movement term ran 108x the retired `MOVEMENT_SHARE_CEILING` -- every `price` call during
#: a plan came back clamped, the search went blind to movement, and the figure measured the
#: cost model rather than the hardware.
#:
#: The span here is `distance_scalar` 0.5 to 8, which is nominal 1 to 256 since
#: `distance_scalar = 8 / sqrt(nominal)`. Five columns at factor-4 steps in the nominal knob,
#: which is factor-2 in `distance_scalar`:
#:
#:     nominal            1      4     16     64    256
#:     distance_scalar    8      4      2      1    0.5
#:     idle_time_scalar   1/8   1/4    1/2      1      2
#:
#: `distance_scalar = 1` is the machine as calibrated, and the operating point
#: `merging_improvement_benchmark_stylized_simulated` runs at, so this axis sweeps *through*
#: the point the rest of the paper is measured at. Verified over 8 QASMBench proxies at
#: k = 2 and 4 across all five columns: zero regressions against naive and a positive mean
#: in every cell.
DEFAULT_RATIOS: Tuple[float, ...] = (1.0, 4.0, 16.0, 64.0, 256.0)

#: Target selection rates for the naive baseline, worst machine first so quality increases
#: rightwards. 0.12 is the house default and sits in the middle, which makes this axis a
#: sweep *through* the operating point every other figure is measured at rather than away
#: from it.
#:
#: The calibration lands high of its target -- measured 5, 11, 17, 33 and 51 percent for
#: these five -- because it solves alpha against a pilot and the pilot is not the final
#: measurement. That is why the columns are labelled with the achieved rate rather than the
#: requested one. The five stay well separated, which is the only thing the axis needs.
DEFAULT_TARGET_YIELDS: Tuple[float, ...] = (0.02, 0.05, 0.12, 0.25, 0.45)

#: Matches `lsr_tvd_vs_k`, `ablation_vs_k`, `code_size_crossover` and
#: `improvement_vs_locality_and_k`, so every k figure sweeps the same family.
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: The regime label is read on this k, fixed, for every row. Reading it at the row's own k
#: would make the x axis partly restate the y axis: patch count sets crossings and
#: crossings are movement, so the measured ratio moves with k on a fixed machine.
REFERENCE_K = 2

#: Shots for the zero-noise correctness reference. Cheap next to the ~150k a measured point
#: runs, and it only has to resolve a gross departure, not a small one.
CORRECTNESS_REPETITIONS = 8_000

#: A miscompile is judged differentially, against the naive arm of the same pair, never
#: against a fixed number -- see the module docstring on why an absolute cutoff throws away
#: wide circuits that are perfectly fine.
CORRECTNESS_RATIO = 3.0
CORRECTNESS_MARGIN = 0.10

#: Above this the zero-noise reference is itself saturated at `CORRECTNESS_REPETITIONS`, so
#: TVD carries no information for that point. Gate counts and depth are exact translation
#: properties and the selection rate is a binomial over the full shot count, so the row is
#: kept and only its TVD is withheld.
TVD_RESOLUTION_LIMIT = 0.15


@dataclass(frozen=True)
class Layout:
    """One output file: a grid of panels on one of the two sweeps, at a stated width."""

    axis: str
    nrows: int
    ncols: int
    metrics: Sequence[str]
    width: float = FULL_WIDTH
    #: Explicit height, or None to derive one that leaves the cells square-ish. Stated on
    #: the narrow canvas, where the fixed chrome allowance is too large a share of the
    #: width for the derivation to give sensible cells.
    height: Optional[float] = None
    #: None follows `--no-annotate`; False overrides it where a cell cannot hold a number.
    annotate: Optional[bool] = None
    bar_ticks: Optional[int] = None


#: `{output suffix: layout}`. The bare-suffix pair are the paper figures; `_column` are the
#: single-column IEEE versions of the same content with the cell numbers dropped, and
#: `_regime_quad` adds the mechanism metric behind the regime result.
#:
#: No depth or gate panel appears on a `YIELD` layout: the placement does not move along
#: that axis, so such a panel would be identical columns. See the module docstring.
LAYOUTS: Dict[str, Layout] = {
    "_regime": Layout(REGIME, 1, 2, ("lsr", "circuit_depth")),
    "_regime_quad": Layout(REGIME, 2, 2, ("lsr", "tvd", "two_qubit_gates", "circuit_depth")),
    "_regime_column": Layout(
        REGIME, 1, 2, ("lsr", "circuit_depth"),
        width=COLUMN_WIDTH, height=2.32, annotate=False, bar_ticks=4,
    ),
    "_yield": Layout(YIELD, 1, 2, ("lsr", "tvd")),
    "_yield_column": Layout(
        YIELD, 1, 2, ("lsr", "tvd"),
        width=COLUMN_WIDTH, height=2.32, annotate=False, bar_ticks=4,
    ),
}

#: `{axis: (shared x label, column-label formatter)}`.
AXIS_LABELS: Dict[str, str] = {
    REGIME: "measured idle / movement noise ratio",
    YIELD: "baseline selection rate (%)",
}


@dataclass
class Settings:
    benchmark_root: Path = field(default_factory=lambda: Path("benchmarks/qasmbench"))
    #: Medium and large. Small is 2-10 qubits and would be a tautology at large k.
    suites: Sequence[str] = ("medium", "large")
    #: The loader bounds `merging_improvement_benchmark_stylized_simulated` uses, so the
    #: two figures describe the same corpus.
    readme_gate_limit: int = 2_000
    proxy_qubit_limit: int = 150
    max_proxy_gates: int = 2_500
    proxy_timeout_seconds: float = 1.0
    #: Trims the corpus for a smoke run. Takes the narrowest circuits first, which are also
    #: the fastest, so a smoke run finishes rather than merely being smaller.
    max_circuits: Optional[int] = None
    k_values: Sequence[int] = DEFAULT_K_VALUES
    ratios: Sequence[float] = DEFAULT_RATIOS
    target_yields: Sequence[float] = DEFAULT_TARGET_YIELDS
    axes: Sequence[str] = (REGIME, YIELD)
    #: `shot_progress=False` because this figure is always parallel: 12 workers writing
    #: per-shot bars to one inherited stdout blocked on the pipe rather than computing,
    #: costing 2.6x throughput. See the field's comment in `SimSettings`.
    sim: SimSettings = field(default_factory=lambda: SimSettings(shot_progress=False))
    jobs: Optional[int] = None
    annotate: bool = True
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    #: Per-item results, so a killed run resumes instead of starting over. None disables.
    checkpoint_dir: Optional[Path] = field(
        default_factory=lambda: HERE / f"{NAME}.checkpoint"
    )
    #: Discard the checkpoint directory before running.
    fresh: bool = False
    #: Plot from whatever the checkpoint directory holds, without running anything.
    from_checkpoint: bool = False


@dataclass(frozen=True)
class Job:
    """One work item: a circuit at one k, swept across one axis' columns.

    Grouped at (circuit, k, axis) rather than at the individual point for two reasons. It
    pickles each circuit once per group instead of once per column, and on the `YIELD`
    axis the noise profile does not move, so the placement can be planned once and reused
    across the whole row -- which is also what guarantees the constancy the summary checks.
    """

    axis: str
    name: str
    circuit: qiskit.QuantumCircuit
    k: int
    levels: Tuple[float, ...]
    sim: SimSettings
    #: Where this item writes its own result on completion. None disables checkpointing.
    #: See `checkpoint_path` on why the worker writes rather than the parent.
    checkpoint_dir: Optional[Path] = None


def checkpoint_path(directory: Path, axis: str, name: str, k: int) -> Path:
    """
    The file one work item owns. Named from the item's identity, not its position.

    Position would break resume the moment the corpus changes: dropping one circuit
    shifts every later index and a resumed run would silently pair item 40's cached
    result with item 41's job. Identity survives `--max-circuits`, a reordered corpus,
    and a `--k-values` change.

    Names come from each benchmark's README `# Application` line and contain spaces and
    punctuation, so they are slugged. Collisions are theoretically possible after
    slugging and would show up as a wrong row count, which `compute` prints.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "-", name).strip("-").lower() or "unnamed"
    return directory / f"{axis}__{slug}__k{k}.json"


#: Settings that change a computed row. A resumed run whose fingerprint differs from the
#: checkpoint directory's must not reuse it -- mixing two shot budgets or two noise
#: profiles into one CSV is the silent-corruption failure the whole mechanism exists to
#: avoid. Deliberately excludes the corpus and the axis levels: those select *which* items
#: run, and a partial checkpoint of a wider sweep is still valid for the items it holds.
def fingerprint(settings: Settings) -> Dict[str, object]:
    from packing.version import COST_MODEL_VERSION

    sim = settings.sim
    return {
        "figure": NAME,
        # The one that was missing. Without it a rerun after a planner fix reuses
        # checkpoints computed under the objective the fix replaced, and the CSV silently
        # mixes two code versions. See `packing.version`.
        "cost_model_version": COST_MODEL_VERSION,
        "alpha_reference": sim.alpha_reference,
        "reference_gate_count": sim.reference_gate_count,
        # Load-bearing on the REGIME axis, where every column takes this value from
        # `settings.sim`. It is *not* load-bearing on YIELD, where `run_job` overrides it
        # per column with the swept level -- but including it there only over-invalidates,
        # and leaving it out let a REGIME resume mix two selection-rate targets into one
        # CSV. Over-conservative beats silently blended.
        "target_selection_rate": sim.target_selection_rate,
        "pilot_repetitions": sim.pilot_repetitions,
        "target_kept_shots": sim.target_kept_shots,
        "min_repetitions": sim.min_repetitions,
        "max_repetitions": sim.max_repetitions,
        "stim": sim.stim,
        "distance_noise": sim.distance_noise,
        "idle_noise": sim.idle_noise,
        "seed": sim.seed,
        "correctness_repetitions": CORRECTNESS_REPETITIONS,
        "reference_k": REFERENCE_K,
    }


def load_checkpoints(
    directory: Path, jobs: Sequence[Job]
) -> Tuple[List[Dict], List[Job]]:
    """Split `jobs` into rows already on disk and items still to run."""
    done: List[Dict] = []
    pending: List[Job] = []
    for job in jobs:
        path = checkpoint_path(directory, job.axis, job.name, job.k)
        if not path.exists():
            pending.append(job)
            continue
        try:
            done.extend(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001 - a truncated file is a rerun, not a crash
            print(f"[{NAME}] unreadable checkpoint, recomputing: {path.name}")
            pending.append(job)
    return done, pending


def frame_from_checkpoints(directory: Path) -> pd.DataFrame:
    """
    Every row currently on disk, whether or not the run that is writing them has finished.

    A 14-hour sweep is worth reading before it ends. Items complete in corpus order, so a
    partial frame is a prefix of the corpus rather than a random sample of it -- draw a
    figure from it to check the pipeline and the colour scales, and read the cell counts
    `print_summary` prints before believing any number in it.
    """
    rows: List[Dict] = []
    files = sorted(path for path in directory.glob("*.json") if path.name != "settings.json")
    for path in files:
        try:
            rows.extend(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001 - the run may be mid-write on this file
            continue
    print(f"[{NAME}] assembled {len(rows)} rows from {len(files)} checkpointed items")
    return pd.DataFrame(rows)


def prepare_checkpoint_dir(settings: Settings) -> Optional[Path]:
    """
    Create or validate the checkpoint directory, returning None when disabled.

    Refuses to reuse a directory written under different result-affecting settings rather
    than mixing them, and says how to clear it.
    """
    directory = settings.checkpoint_dir
    if directory is None:
        return None
    marker = directory / "settings.json"
    current = fingerprint(settings)
    if settings.fresh and directory.exists():
        shutil.rmtree(directory)
        print(f"[{NAME}] cleared {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous != current:
            differing = sorted(
                key for key in set(previous) | set(current)
                if previous.get(key) != current.get(key)
            )
            raise RuntimeError(
                f"{directory} holds results computed under different settings "
                f"({', '.join(differing)} changed). Reusing them would mix two "
                f"configurations into one CSV. Pass --fresh to discard them, or "
                f"--no-checkpoint to ignore the directory entirely."
            )
    else:
        marker.write_text(json.dumps(current, indent=2, sort_keys=True))
    return directory


def load_benchmarks(settings: Settings) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """
    Proxy the QASMBench suites this figure runs on, with the merging figure's bounds.

    Two filters, doing different jobs. `load_proxy_circuits`' `gate_limit` reads the
    pre-proxy gate count out of each benchmark's README and treats a missing or
    unparseable one as no limit; `max_proxy_gates` is then applied to the transpiled proxy,
    which is the object that actually gets simulated. Both are needed -- the first keeps
    the transpile cheap, the second is the one that binds.
    """
    circuits: List[Tuple[str, qiskit.QuantumCircuit]] = []
    for suite in settings.suites:
        loaded = load_proxy_circuits(
            settings.benchmark_root / suite,
            gate_limit=settings.readme_gate_limit,
            qubit_limit=settings.proxy_qubit_limit,
            timeout_seconds=settings.proxy_timeout_seconds,
            replace_rz_rx_with_s=True,
        )
        circuits += [
            (name, circuit) for name, circuit in loaded
            if circuit.size() <= settings.max_proxy_gates
        ]

    # A program narrower than one patch has no placement to make: naive and AQORN return
    # the same permutation and the pair reads exactly 0% for a structural reason. Dropping
    # those here rather than per cell is what keeps every cell averaging the same corpus.
    widest_k = max(settings.k_values)
    usable = [(name, circuit) for name, circuit in circuits if circuit.num_qubits > widest_k]
    dropped = len(circuits) - len(usable)
    if dropped:
        print(f"[{NAME}] dropped {dropped} circuits no wider than k={widest_k}: "
              "one patch, so no placement to make")

    usable.sort(key=lambda item: (item[1].num_qubits, item[1].size(), item[0]))
    if settings.max_circuits is not None:
        usable = usable[: settings.max_circuits]
    return usable


def _noiseless_tvd(
    circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    sim: SimSettings,
    *,
    n_code: int,
    h_commutation: bool,
) -> float:
    """
    Shared guard, kept behind this name because `run_job` caches on top of it.

    **Independent of everything this figure sweeps, which is what makes it cacheable.**
    `build_config` reads `distance_scalar` only under `if distance_noise` and
    `idle_time_scalar` only under `if idle_noise`; the shared implementation switches both
    off and passes alpha zero, so neither regime scalar is consulted.
    `target_selection_rate` is a `calibrate` input that `build_config` never reads. So the
    result depends only on the circuit, the placement, `n_code` and `h_commutation` -- see
    `run_job`, which keys a cache on exactly that.
    """
    return noiseless_tvd(
        circuit, permutation, sim, n_code=n_code, h_commutation=h_commutation,
    )


def run_job(job: Job) -> List[Dict]:
    """
    Measure one circuit at one k across one axis' columns. Module scope, for `spawn`.

    Returns two rows per column -- naive and AQORN -- or fewer where a column failed. A
    failed column is a normal outcome, not an error: `evaluate` returns None for a
    rejected placement and `calibrate` can fail to find a finite alpha on a circuit whose
    baseline never reaches the target selection rate.
    """
    n_code = job.k + 2
    width = job.circuit.num_qubits
    naive = naive_packing(width, job.k)
    reference = naive_packing(width, REFERENCE_K)
    gate_count = baseline_gate_count(job.circuit, naive, n_code)

    # The miscompile guard is the same simulation for every column that shares a placement,
    # and translating an encoded circuit is the dominant fixed cost of a point -- measured
    # at 35-63% of even a 200k-shot evaluate, so a 100x change in shots moves the runtime
    # only 1.6-2.8x. Naive never moves, and adjacent regimes usually replan to the same
    # placement, so this collapses 14 calls per regime row to about 4 and 10 per yield row
    # to 2. See `_noiseless_tvd` for why the swept level is absent from the key.
    clean_cache: Dict[Tuple[Tuple[Tuple[int, int], ...], bool], float] = {}

    def clean_tvd(placement: Perm, h_commutation: bool) -> float:
        key = (tuple(placement), h_commutation)
        if key not in clean_cache:
            clean_cache[key] = _noiseless_tvd(
                job.circuit, placement, job.sim,
                n_code=n_code, h_commutation=h_commutation,
            )
        return clean_cache[key]

    rows: List[Dict] = []
    planned: Optional[Tuple[Perm, bool]] = None
    for level in job.levels:
        sim = (
            job.sim.with_noise_ratio(level) if job.axis == REGIME
            else replace(job.sim, target_selection_rate=level)
        )

        # Along YIELD the profile is identical in every column, so the plan is reused --
        # both to save the call and because reusing it is what makes the placement, and
        # therefore the gate counts, provably constant along the row.
        if planned is None or job.axis == REGIME:
            try:
                # Bug B6: the planner's noise costs default to zero, so without a profile
                # it optimises gates and depth while the simulator charges movement and
                # idling -- answering a different question from the one measured.
                plan = plan_packing(
                    job.circuit, n_code, h_commutation="auto", verify=True,
                    noise=noise_profile_for(sim),
                )
                planned = (plan.placement, plan.h_commutation)
            except Exception:  # noqa: BLE001 - a rejected plan falls back to naive
                planned = (naive, True)
        aqorn, commute = planned

        calibration = calibrate(
            job.circuit, naive, sim, n_code=n_code, baseline_gate_count=gate_count,
        )
        if not math.isfinite(calibration.alpha):
            continue

        # One config, both placements: same alpha, same shots, same passes, so the
        # difference between the two rows is the placement and nothing else.
        config = build_config(
            sim, n_code=n_code, alpha=calibration.alpha,
            repetitions=calibration.repetitions, h_commutation=commute, merging=True,
        )
        measured = {
            "naive": evaluate(job.circuit, naive, config),
            "aqorn": evaluate(job.circuit, aqorn, config),
        }
        if any(value is None for value in measured.values()):
            continue

        # Check the compile before trusting the measurement, not after.
        clean = {
            series: clean_tvd(naive if series == "naive" else aqorn, commute)
            for series in measured
        }

        # The x label: what the schedule really charges at these scalars, on a fixed
        # reference so the column cannot restate k.
        try:
            regime = measured_regime(
                job.circuit, reference, n_code=REFERENCE_K + 2,
                idle_time_scalar=sim.idle_time_scalar,
                distance_scalar=sim.distance_scalar,
            )["measured_ratio"]
        except Exception:  # noqa: BLE001 - a rejected reference only costs the label
            regime = float("nan")

        for series, result in measured.items():
            placement = naive if series == "naive" else aqorn
            rows.append({
                "axis": job.axis,
                "level": level,
                "circuit": job.name,
                "num_qubits": width,
                "program_gates": job.circuit.size(),
                "k": job.k,
                "n_code": n_code,
                "series": series,
                "patches": len({patch for patch, _ in placement}),
                "measured_ratio": regime,
                "idle_time_scalar": sim.idle_time_scalar,
                "distance_scalar": sim.distance_scalar,
                "target_selection_rate": sim.target_selection_rate,
                "alpha": calibration.alpha,
                "repetitions": calibration.repetitions,
                "pilot_selection_rate": calibration.pilot_selection_rate,
                "h_commutation_chosen": commute,
                "noiseless_tvd": clean[series],
                **{metric: result[metric] for metric in PANELS},
            })

    # The worker persists its own result rather than the parent collecting them, because
    # `parallel_map` returns only once every item is done: a run killed at hour eight had
    # nothing on disk and lost all of it. Written even when empty, so a row whose every
    # column legitimately failed is not retried on every resume.
    #
    # Atomic: a kill between `open` and the last byte would otherwise leave a half-written
    # file that reads as a valid-but-short result. Write beside the target on the same
    # filesystem, then rename, which is atomic on POSIX.
    if job.checkpoint_dir is not None:
        path = checkpoint_path(job.checkpoint_dir, job.axis, job.name, job.k)
        temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(rows))
        os.replace(temporary, path)
    return rows


def compute(settings: Settings) -> pd.DataFrame:
    benchmarks = load_benchmarks(settings)
    if not benchmarks:
        raise RuntimeError(f"no usable circuits under {settings.benchmark_root}")
    widths = [circuit.num_qubits for _, circuit in benchmarks]
    gates = [circuit.size() for _, circuit in benchmarks]
    print(f"[{NAME}] {len(benchmarks)} circuits from {'+'.join(settings.suites)}: "
          f"{min(widths)}-{max(widths)} qubits (median {int(np.median(widths))}), "
          f"{min(gates)}-{max(gates)} gates")

    directory = prepare_checkpoint_dir(settings)
    levels = {REGIME: tuple(settings.ratios), YIELD: tuple(settings.target_yields)}
    jobs = [
        Job(axis=axis, name=name, circuit=circuit, k=k,
            levels=levels[axis], sim=settings.sim, checkpoint_dir=directory)
        for axis in settings.axes
        for name, circuit in benchmarks
        for k in settings.k_values
    ]
    points = sum(len(job.levels) for job in jobs)
    print(f"[{NAME}] {len(jobs)} work items, {points} points, {2 * points} simulations")

    cached: List[Dict] = []
    if directory is not None:
        cached, jobs = load_checkpoints(directory, jobs)
        if cached or not jobs:
            print(f"[{NAME}] resuming: {len(cached)} rows cached in {directory}, "
                  f"{len(jobs)} items left to run")

    results = parallel_map(run_job, jobs, jobs=settings.jobs, label=NAME) if jobs else []
    failed = sum(1 for result in results if result is None)
    if failed:
        print(f"[{NAME}] {failed} items failed and were not checkpointed; "
              f"rerunning resumes just those")
    rows = cached + [row for result in results if result for row in result]
    return pd.DataFrame(rows)


def paired_improvement(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per-circuit percent improvement of AQORN over naive, every metric signed so up is better.

    Paired, not a difference of the two series' means: the circuits in this corpus differ
    from each other by orders of magnitude in width and length, far more than the two
    placements differ on any one of them, so an unpaired cell would be mostly corpus draw.

    Two exclusions, both differential and both re-derivable from the CSV rather than baked
    in at measurement time:

      - a pair whose AQORN arm departs from naive at *zero* noise is a miscompile and the
        whole pair is dropped, since none of its metrics can be trusted;
      - a pair whose zero-noise floor is already saturated has its TVD withheld and its
        other four metrics kept, because those do not depend on resolving the distribution.
    """
    wide = frame.pivot_table(
        index=["axis", "level", "circuit", "k"],
        columns="series",
        values=list(PANELS) + ["noiseless_tvd"],
    )
    rows: List[Dict] = []
    miscompiled: List[Tuple[str, str, float, int]] = []
    for (axis, level, circuit, k), row in wide.iterrows():
        floor_naive = row[("noiseless_tvd", "naive")]
        floor_aqorn = row[("noiseless_tvd", "aqorn")]
        broken = (
            floor_naive == floor_naive and floor_aqorn == floor_aqorn
            and floor_aqorn > CORRECTNESS_RATIO * floor_naive + CORRECTNESS_MARGIN
        )
        if broken:
            miscompiled.append((circuit, axis, level, k))
            continue
        resolved = floor_naive == floor_naive and floor_naive <= TVD_RESOLUTION_LIMIT

        record = {
            "axis": axis, "level": level, "circuit": circuit, "k": k,
            "tvd_resolved": bool(resolved),
        }
        for metric, (_, higher_is_better) in PANELS.items():
            naive, aqorn = row[(metric, "naive")], row[(metric, "aqorn")]
            if not naive or (metric == "tvd" and not resolved):
                record[metric] = np.nan
                continue
            gain = (aqorn - naive) if higher_is_better else (naive - aqorn)
            record[metric] = 100.0 * gain / naive
        rows.append(record)

    if miscompiled:
        print(f"\n[{NAME}] dropped {len(miscompiled)} miscompiled pairs "
              f"(AQORN departs from naive at zero noise):")
        for circuit, axis, level, k in sorted(miscompiled)[:12]:
            print(f"    {circuit:<18s} {axis:<7s} level {level:<7.4g} k={k}")
        if len(miscompiled) > 12:
            print(f"    ... and {len(miscompiled) - 12} more")
    return pd.DataFrame(rows)


def column_labels(frame: pd.DataFrame, axis: str, levels: Sequence[float]) -> List[str]:
    """
    Tick label per column: what was measured there, not the knob that was turned.

    A regime column carries the geometric mean of the measured ratio -- geometric because
    the quantity is a ratio and the sweep is geometric, so an arithmetic mean would be
    pulled to the idle-dominated end. A yield column carries the naive baseline's achieved
    selection rate as a percent, which is the definition of that axis.
    """
    labels: List[str] = []
    for level in levels:
        cell = frame[(frame["axis"] == axis) & (frame["level"] == level)]
        if axis == REGIME:
            values = cell["measured_ratio"].dropna()
            values = values[values > 0]
            if not len(values):
                labels.append("--")
                continue
            mean = float(np.exp(np.log(values).mean()))
            labels.append(f"{mean:.3g}" if mean < 100 else f"{mean:.0f}")
        else:
            values = cell.loc[cell["series"] == "naive", "lsr"].dropna()
            labels.append(f"{100 * float(values.mean()):.0f}" if len(values) else "--")
    return labels


def cell_means(
    improvement: pd.DataFrame,
    metric: str,
    axis: str,
    levels: Sequence[float],
    k_values: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Grids of mean improvement and of circuit count, both indexed `[k, level]`."""
    grid = np.full((len(k_values), len(levels)), np.nan)
    counts = np.zeros((len(k_values), len(levels)), dtype=int)
    for row, k in enumerate(k_values):
        for column, level in enumerate(levels):
            cell = improvement.loc[
                (improvement["axis"] == axis)
                & (improvement["k"] == k)
                & (np.isclose(improvement["level"], level)),
                metric,
            ].dropna()
            counts[row, column] = len(cell)
            if len(cell):
                grid[row, column] = float(cell.mean())
    return grid, counts


def print_summary(frame: pd.DataFrame, improvement: pd.DataFrame) -> None:
    circuits = frame["circuit"].nunique()
    print(f"\n[{NAME}] {len(frame)} rows, {circuits} circuits, "
          f"{frame['num_qubits'].min()}-{frame['num_qubits'].max()} qubits")

    resolved = improvement["tvd_resolved"].mean() if len(improvement) else 0.0
    print(f"  TVD resolved on {100 * resolved:.0f}% of pairs "
          f"(noiseless floor <= {TVD_RESOLUTION_LIMIT}); the rest keep their other four "
          "metrics and are withheld from the TVD panel only")

    for axis in sorted(frame["axis"].unique()):
        levels = sorted(frame.loc[frame["axis"] == axis, "level"].unique())
        k_values = sorted(frame.loc[frame["axis"] == axis, "k"].unique())
        labels = column_labels(frame, axis, levels)
        print(f"\n  === {axis} axis ===  columns labelled "
              f"{AXIS_LABELS[axis]}: {', '.join(labels)}")

        for metric, (title, _) in PANELS.items():
            grid, counts = cell_means(improvement, metric, axis, levels, k_values)
            if np.all(np.isnan(grid)):
                continue
            print(f"\n  {title}    rows: k, cols: {axis} level")
            print("        " + "".join(f"{label:>8s}" for label in labels))
            for row, k in enumerate(k_values):
                cells = "".join(
                    "      --" if value != value else f"{value:>8.2f}"
                    for value in grid[row]
                )
                print(f"   {k:>4d} " + cells)
            print(f"    overall {np.nanmean(grid):+.2f}%   "
                  f"range {np.nanmin(grid):+.2f}% to {np.nanmax(grid):+.2f}%   "
                  f"n per cell {counts.min()}-{counts.max()}")

    _check_yield_constancy(improvement)


def _check_yield_constancy(improvement: pd.DataFrame) -> None:
    """
    Verify the claim the yield layouts are built on rather than asserting it.

    `noise_profile_for` ignores alpha and the selection-rate target, so along the yield
    axis every column hands the planner the same profile and gets the same placement back.
    Depth and gate counts are properties of that placement, so they must not move. If they
    do, either the profile picked up a dependency it should not have or the plan cache in
    `run_job` is not doing what it claims, and the yield panels' metric choice is wrong.
    """
    rows = improvement[improvement["axis"] == YIELD]
    if rows.empty:
        return
    print("\n  yield-axis constancy check (placement must not move along this axis):")
    for metric in ("circuit_depth", "two_qubit_gates", "one_qubit_gates"):
        spread = (
            rows.groupby(["circuit", "k"])[metric]
            .agg(lambda values: values.max() - values.min())
            .dropna()
        )
        worst = float(spread.max())
        verdict = "constant" if worst <= 1e-9 else "MOVED -- yield panel choice is wrong"
        print(f"    {PANELS[metric][0]:<14s} max spread across columns "
              f"{worst:.3e}  {verdict}")


def _cap_ticks(bar, limit: Optional[int]) -> None:
    """Thin a colourbar's ticks to at most `limit`, or leave matplotlib's choice alone."""
    if limit is None:
        return
    bar.locator = MaxNLocator(nbins=limit)
    bar.update_ticks()


def plot_layout(
    frame: pd.DataFrame,
    improvement: pd.DataFrame,
    settings: Settings,
    suffix: str,
    layout: Layout,
) -> List[Path]:
    axis = layout.axis
    rows = improvement[improvement["axis"] == axis]
    if rows.empty:
        return []
    levels = sorted(frame.loc[frame["axis"] == axis, "level"].unique())
    k_values = sorted(rows["k"].unique())
    annotate = settings.annotate if layout.annotate is None else layout.annotate
    labels = column_labels(frame, axis, levels)

    # Cell positions are the column *index*, not the level. The regime sweep is geometric
    # over four orders of magnitude, so plotting at the level would make one cell wider
    # than the other six put together; the sweep is a set of conditions rather than a
    # continuum, so equal cells are the honest rendering and the measured value goes on
    # the tick instead.
    positions = list(range(len(levels)))
    x_edges, y_edges = cell_edges(positions), cell_edges(k_values)
    grids = {
        metric: cell_means(improvement, metric, axis, levels, k_values)[0]
        for metric in layout.metrics
    }

    panel = (layout.width - 0.9) / layout.ncols
    height = (
        layout.height if layout.height is not None
        else layout.nrows * panel * 0.92 + 0.55
    )
    fig, axes = plt.subplots(
        nrows=layout.nrows, ncols=layout.ncols,
        figsize=(layout.width, height),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for index, (panel_axis, metric) in enumerate(zip(axes, layout.metrics)):
        label, _ = PANELS[metric]
        # Per-panel scales, as in the sibling figures: the metrics' ranges differ
        # several-fold and one scale would flatten the smaller panel to near-uniform
        # white. White stays pinned at zero in every panel, which is the part of the
        # encoding that does have to be shared.
        cmap, norm = improvement_scale(grids[metric])
        mesh = panel_axis.pcolormesh(
            x_edges, y_edges, grids[metric],
            cmap=cmap, norm=norm,
            shading="flat", edgecolors="white", linewidth=0.4,
        )
        if annotate:
            annotate_cells(panel_axis, grids[metric], positions, k_values, cmap, norm)
        panel_axis.set_title(f"({chr(ord('a') + index)})", loc="left", fontweight="bold")
        panel_axis.set_title(label, loc="center")
        panel_axis.set_xticks(positions, labels)
        panel_axis.set_yticks(k_values)
        panel_axis.grid(False)
        for spine in panel_axis.spines.values():
            spine.set_visible(False)
        panel_axis.tick_params(length=0)

        bar = fig.colorbar(mesh, ax=panel_axis, pad=0.03, aspect=18)
        bar.ax.tick_params(length=2, width=0.6)
        bar.outline.set_visible(False)
        _cap_ticks(bar, layout.bar_ticks)

    for unused in axes[len(layout.metrics):]:
        unused.set_visible(False)

    fig.suptitle("AQORN improvement over naive packing (%)")
    fig.supxlabel(AXIS_LABELS[axis])
    fig.supylabel("logical qubits per patch, $k$")

    paths = save_figure(fig, f"{NAME}{suffix}", Path(__file__), settings.output_override)
    plt.close(fig)
    return paths


def plot(frame: pd.DataFrame, improvement: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    paths: List[Path] = []
    for suffix, layout in LAYOUTS.items():
        paths.extend(plot_layout(frame, improvement, settings, suffix, layout))
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--benchmark-root", type=Path, default=defaults.benchmark_root)
    parser.add_argument("--suites", nargs="+", default=list(defaults.suites),
                        help="QASMBench size groups. Small is excluded by default: at "
                             "2-10 qubits it needs one patch at large k, so the pair is a "
                             "tautology reading exactly 0%%.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values),
                        help="y axis. Logical qubits per patch, in the [[k+2, k, 2]] family.")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(defaults.ratios),
                        help="Nominal idle/movement ratios. Coverage only -- the column is "
                             "labelled with the ratio the schedule actually charged.")
    parser.add_argument("--target-yields", nargs="+", type=float,
                        default=list(defaults.target_yields),
                        help="Naive-baseline selection rates to calibrate to. This is the "
                             "machine-quality axis.")
    parser.add_argument("--axes", nargs="+", choices=[REGIME, YIELD], default=list(defaults.axes),
                        help="Which sweeps to run. Each is an independent set of points.")
    parser.add_argument("--max-circuits", type=int, default=defaults.max_circuits,
                        help="Trim the corpus to the N narrowest circuits, for a smoke run.")
    parser.add_argument("--max-proxy-gates", type=int, default=defaults.max_proxy_gates,
                        help="Applied to the loaded proxy, which is what gets simulated.")
    parser.add_argument("--no-annotate", dest="annotate", action="store_false",
                        help="Drop the per-cell numbers. On by default because each panel "
                             "carries its own colour scale.")
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it here: this run "
                             "is hours of simulation.")
    parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir,
                        help="Where each work item persists its own rows, so a killed run "
                             "resumes rather than restarting. Default is beside the CSV.")
    parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false",
                        help="Do not read or write per-item results. The run then has to "
                             "complete in one go, which over a sweep this long is how "
                             "eight hours of simulation get thrown away.")
    parser.add_argument("--from-checkpoint", action="store_true",
                        help="Plot from whatever the checkpoint directory holds right now, "
                             "running nothing. Use it to read a sweep that is still going: "
                             "items land in corpus order, so a partial frame is a prefix of "
                             "the corpus, and print_summary reports the cell counts.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete the checkpoint directory first. Needed after changing "
                             "a setting that alters a computed row, which is otherwise "
                             "refused rather than silently mixed.")
    args = parser.parse_args()
    return Settings(
        benchmark_root=args.benchmark_root,
        suites=args.suites,
        k_values=args.k_values,
        ratios=args.ratios,
        target_yields=args.target_yields,
        axes=args.axes,
        max_circuits=args.max_circuits,
        max_proxy_gates=args.max_proxy_gates,
        sim=settings_from_args(args, defaults.sim),
        jobs=args.jobs,
        annotate=args.annotate,
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
        checkpoint_dir=args.checkpoint_dir if args.checkpoint else None,
        fresh=args.fresh,
        from_checkpoint=args.from_checkpoint,
    )


def main() -> None:
    settings = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if settings.from_checkpoint:
        if settings.checkpoint_dir is None:
            raise RuntimeError("--from-checkpoint needs a checkpoint directory")
        frame = frame_from_checkpoints(settings.checkpoint_dir)
        if frame.empty:
            raise RuntimeError(f"no checkpointed rows in {settings.checkpoint_dir}")
    elif settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
        check_provenance(frame, settings.from_csv)
    else:
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        frame = stamp_provenance(frame)
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            from experiments.shared.figures import data_dir

            target = data_dir(Path(__file__)) / f"{NAME}.csv"
            frame.to_csv(target, index=False)
            print(f"[{NAME}] persisted to {target}")

    improvement = paired_improvement(frame)
    print_summary(frame, improvement)
    for path in plot(frame, improvement, settings):
        print(f"[{NAME}] wrote {path}")


# Required, not tidiness: `parallel.py` spawns rather than forks, so the child re-imports
# this module and would re-run the sweep on import without the guard.
if __name__ == "__main__":
    main()
