"""
improvement_vs_locality_and_k.py -- where in the design space packing actually pays.

**Claim.** AQORN's gain over naive packing is not a single number: it is a surface over
how much community structure the circuit has and how large the code's patches are.
Locality is what packing has to find, so the gain grows with it; `k` sets how many
placement decisions there are to get right.

This figure is the two-dimensional version of `lsr_tvd_vs_k`, and it exists because that
figure found the two placements sitting inside each other's spread at every `k` and could
not say why. It swept `k` at a single locality of 0.85. Reading the same comparison across
both axes at once shows whether that null result is a fact about the packer or a fact
about the one column it was measured in.

**The y axis used to be immutable fraction, and was dropped after measuring it.** The
generator pins the program at `num_qubits * gates_per_qubit` gates with
`two_qubit_ratio = 0.40`, so raising the immutable share swaps single-qubit H/X/Z for S
and never touches the interaction graph -- which means the *packing problem is identical*
along that axis. It is not inert: blocking merging inflates the naive baseline exactly as
intended (1q 169 -> 226, 2q 254 -> 299, depth 125 -> 169 across 0.0 -> 0.4, with the
realised fraction tracking the requested one to three decimals). But it modulates how much
the baseline bloats rather than what the placer decides, and it carried about a third of
locality's leverage on every metric and none at all on TVD (r = -0.09). That sweep is kept
at `results/data/improvement_vs_structure.csv` and is worth quoting in text. `k` replaces
it because patch capacity changes the placement problem itself.

**Do not plot the TVD panel as a claim at the default shot budget. It currently says the
opposite of what the data says.**

Measured TVD falls as 1/sqrt(accepted shots) almost the whole way: on one cell, 0.0770 at
43k accepted shots, 0.0443 at 173k, 0.0274 at 691k. Fitting `T = a/sqrt(N) + b` puts the
true TVD near 0.011 against a measured 0.077, so roughly 85% of what this sweep reads as
TVD is sampling error.

That would only make the panel imprecise. What makes it wrong is that the two placements
do not sit on the same floor. AQORN accepts more shots than naive -- 52k against 45k here,
which *is* the LSR result -- so it lands on a lower floor and posts a TVD gain it did not
earn. On the 20-qubit grid:

    measured TVD improvement        +3.91%
    shot-count artefact alone       +6.81%   <- what zero real effect looks like
    residual, the possibly-real part -2.89%  +/- 0.23  (t = -12.4)

Every cell of the panel is positive and the trend rises with locality, so it reads as a
clean win; corrected for the floor it is a small, highly consistent *loss*, negative at
all five k values. A reader who takes panel (b) at face value gets the sign backwards.

The correction assumes TVD is pure floor, which is the regime being warned about, so it is
an order-of-magnitude check rather than a result to quote. The fix is shots, not algebra:
`--target-kept-shots 320000`, roughly 16x, about 11 hours against one. `print_summary`
prints this decomposition on every run below `TVD_FLOOR_SHOTS`, so it cannot go unnoticed.

Until that is run, quote LSR, 2q and depth, and ship the quad rather than the 1x2.

Two axes:

    x   locality -- probability a two-qubit gate stays inside its planted block. 1.0 is
        perfectly separable; around 0.5 the interaction graph collapses to Erdos-Renyi,
        every partition is as good as any other, and there is nothing for a packer to
        find. This is the separability axis, and it is a fraction rather than a count so
        it stays comparable if the circuit width or depth ever changes.
    y   k -- logical qubits per patch, in the [[k+2, k, 2]] family, running the patch
        count 10, 5, 4, 3, 2. `patches` is recorded per row and the summary prints it by
        k, so that is checkable rather than assumed.

**The program is 20 qubits wide, not the 12 the sibling figures use, and it has to be.**
`k` only means something here through the patch count, and `ceil(n/k)` for k = 2..10 is
6, 3, 2, 2, 2 at n = 12 -- so three of the five rows would be the *same* two-patch problem
and the top of the axis would be measuring nothing. At n = 20 it is 10, 5, 4, 3, 2, five
distinct problems. n = 24 is both slower and worse (12, 6, 4, 3, 3).

This is worth carrying back: `lsr_tvd_vs_k` sweeps the same k values at n = 12, so the
top three points of its x axis are all two-patch, and that is a live candidate for why it
found no difference between the placements at high k. Its CSV records `patches`, and it
reads 6, 3, 2, 2, 2.

`block_size` is held at 4 rather than tracked to `k`, following `lsr_tvd_vs_k`. Two
reasons: the circuit is then identical along a row, so a cell differs from its neighbour
only in the code and the y axis means one thing; and block/patch alignment is allowed to
vary with `k`, which is the honest case, because a circuit's community structure does not
know what code size you picked. At k = 4 the blocks and patches coincide and a perfect
packer recovers the planted structure exactly; away from it they do not, and the figure
shows what that costs.

**Five metrics per cell, one simulation per placement.** `experiments.shared.simulation`
reads LSR, TVD, 1q, 2q and depth off the same scan row -- the gate and depth numbers are
the *translated* circuit's, measured on the object the simulator ran -- so all five come
free and all five go to the CSV. Only some get plotted, because page space is the
constraint rather than compute: five colour panels either do not fit a column or fit by
starving the grid of cells, and a heatmap with too few cells shows no trend.

Four layouts from one run, so the paper can take whichever fits the page:

    improvement_vs_locality_and_k         1x2  7.14in  LSR, TVD    -- the outcome metrics
    improvement_vs_locality_and_k_depth   1x3  7.14in  LSR, TVD, depth
    improvement_vs_locality_and_k_quad    2x2  7.14in  LSR, TVD, 2q, depth  -- mechanism
    improvement_vs_locality_and_k_column  1x2  3.49in  LSR, depth  -- the resolved pair

On the numbers above, the quad is the one that currently carries a claim end to end: it
shows the mechanism (2q) next to the two outcomes it drives, and three of its four panels
are resolved. The 1x2 is the intended paper figure and becomes the right one once TVD is
run at a shot budget that resolves it.

**The column layout is the one that needs no caveat.** It drops TVD rather than shipping
the panel this docstring spends fifty lines warning about, and shows the two metrics that
are resolved at the default shot budget: selection rate, the outcome, and depth, the cost
that buys it. Its cells carry no numbers -- at 1.1in of axis per panel there is no room
for them -- so the colourbars are the whole reading and get their own scale per panel.

1q is in the CSV and the printed summary but never plotted. It is the one metric packing
does not improve -- a placement that removes a cross-patch CNOT pays for it in the
single-qubit gates that set the transversal operation up -- so it belongs in a sentence
about the tradeoff rather than in a panel that would read as a claim about the outcome.

**Colour.** The navy ramp of the published CNOT heatmaps, extended downwards into red.
White always means exactly zero -- "no better than naive" -- so the sign of a cell reads
without consulting the bar. Each panel's ramp is sampled so that a given number of percent
is the same depth of colour whichever side of zero it falls on, rather than each side
getting half the colours regardless of how much data is on it. That matters in both
directions and this figure hit both: a lone -0.84% cell rendering as full-strength red,
and an all-positive panel printing a red band on its bar. See `_scale`.

Panels get their own bars by default. Their ranges are not comparable -- LSR spans 5.6% to
38.3% across this grid while 2q spans 1.7% to 6.3% -- so one scale would flatten the
smaller panel to near-uniform white and destroy the trend the heatmap exists to show.
`--shared-colorbar` puts every panel on one scale for the case where they are comparable.

Greyscale is the known weak point, and it is inherent to a diverging map rather than
something this figure does wrong: both arms darken away from white, so a strongly negative
cell and a strongly positive one print at similar density. The style's usual fallback,
redundant encoding by marker and line style, has nothing to attach to in a heatmap. Panels
(a), (c) and (d) are one-signed and therefore read correctly in greyscale; only the TVD
panel is ambiguous there.

**One noise calibration per cell, solved on the naive baseline.** Post-selection is an
exponentially strict filter, so a fixed noise scale either keeps every shot or none;
alpha is solved so the *baseline* lands near `--target-selection-rate` and AQORN is then
measured at that same alpha and shot count. Calibrating each placement separately would
move the noise with the thing being measured. Shot counts follow from the calibrated
selection rate, so cells that keep 3% of their shots are run harder than cells that keep
30% and the spreads are comparable across the grid.

**Improvements are paired within a circuit.** Circuit-to-circuit spread is roughly twice
the placement effect, so a difference of means across circuits hides a consistent
within-circuit gain inside sampling noise. Each cell averages per-circuit percent
improvements, which cancels the circuit and leaves the placement.

`clifford_only=True`: the Stim backend rejects Rx and Rz outright, so the immutable gates
here are all S. That is the same reason the paper simulates rotation-bearing benchmarks
through Clifford proxies rather than directly.

`k` is pinned across the whole grid -- the figure asks what circuit structure does, and a
moving code size would confound that with what `k` does (which is `lsr_tvd_vs_k`). At the
default width of 12 and `k = 4` the program needs three patches, so every cell is a real
placement problem, and the planted blocks are the same size as a patch so a perfect packer
recovers them exactly. `plant_blocks` shuffles the labels, so recovering them is an
achievement rather than something naive index-order gets for free.

Run with:

    python -m experiments.figures.improvement_vs_locality_and_k.improvement_vs_locality_and_k
    python -m experiments.figures.improvement_vs_locality_and_k.improvement_vs_locality_and_k --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.figures import (
    COLUMN_WIDTH,
    FULL_WIDTH,
    annotate_cells,
    cell_edges,
    improvement_scale,
    save_figure,
    use_style,
)
from experiments.shared.random_circuits import (
    immutable_fraction,
    intra_block_fraction,
    plant_blocks,
    structured_program_circuit,
)
from experiments.shared.simulation import (
    check_compile,
    METRICS,
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    settings_from_args,
)
from packing.plan import plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

NAME = "improvement_vs_locality_and_k"
HERE = Path(__file__).resolve().parent

Perm = List[Tuple[int, int]]

#: `{metric: (panel title, higher_is_better)}`, in the order the CSV columns are written.
#: Every one of these is computed for every cell; which ones get drawn is a separate
#: decision made by LAYOUTS below. LSR, TVD and the two gate counts are dimensionless or
#: counted, and all five are reported as a percent improvement, so no panel carries a unit.
PANELS: Dict[str, Tuple[str, bool]] = {
    "lsr": ("selection rate", True),
    "tvd": ("TVD", False),
    "one_qubit_gates": ("1q gates", False),
    "two_qubit_gates": ("2q gates", False),
    "circuit_depth": ("depth", False),
}

# Fail at import rather than with a KeyError a few thousand shots into the sweep if
# someone adds a metric here that the shared evaluator does not measure.
_UNKNOWN = set(PANELS) - set(METRICS)
if _UNKNOWN:
    raise ImportError(
        f"PANELS names metrics evaluate() does not return: {sorted(_UNKNOWN)}. "
        f"Available: {sorted(METRICS)}"
    )

@dataclass(frozen=True)
class Layout:
    """One output file: a grid of panels drawn on a stated canvas width."""

    nrows: int
    ncols: int
    metrics: Sequence[str]
    #: Inches. `FULL_WIDTH` is a LaTeX `figure*`, `COLUMN_WIDTH` a `figure`. Both are the
    #: final size -- the canvas is never drawn large and scaled down, because that shrinks
    #: every label with it.
    width: float = FULL_WIDTH
    #: Explicit canvas height, or None to derive one that leaves the cells square-ish.
    #: The derivation subtracts a *fixed* chrome allowance and then divides, which is fine
    #: at 7.14in where chrome is an eighth of the canvas and wrong at 3.487in where it is
    #: over a third -- the derived height then makes the cells noticeably tall for their
    #: width. Stating it beats generalising the formula for one layout.
    height: Optional[float] = None
    #: None follows `--no-annotate`. False overrides it, for a canvas whose cells cannot
    #: hold a number however the caller asked.
    annotate: Optional[bool] = None
    #: Cap on colourbar ticks. Left to matplotlib at full width; held down on the narrow
    #: canvas, where the default set overlaps itself along a 1.5in bar.
    bar_ticks: Optional[int] = None
    #: Write the locality ticks as `.5` rather than `0.5`, and `1` rather than `1.0`. Six
    #: three-character labels want about 1.0in of glyph on a 1.1in axis, so they collide;
    #: locality is a fraction that cannot exceed 1, so the leading digit is redundant and
    #: dropping it buys back a third of the width without touching the type size.
    compact_xticks: bool = False
    #: Shared x label. The long form spans the narrow canvas margin to margin, which reads
    #: as an overrun rather than a label, so that layout drops the word "fraction" -- the
    #: tick values already say it is one.
    xlabel: str = "locality   (fraction of 2q gates inside a block)"


#: `{output suffix: layout}`. Four files per run: the paper picks one, and the others exist
#: so that choice does not mean re-running the sweep. The bare name is the full-width
#: two-panel version, which is the one the claim needs once TVD resolves; `_column` is the
#: one that fits a single column and needs no caveat today.
LAYOUTS: Dict[str, Layout] = {
    "": Layout(1, 2, ("lsr", "tvd")),
    "_depth": Layout(1, 3, ("lsr", "tvd", "circuit_depth")),
    "_quad": Layout(2, 2, ("lsr", "tvd", "two_qubit_gates", "circuit_depth")),
    "_column": Layout(
        1, 2, ("lsr", "circuit_depth"),
        width=COLUMN_WIDTH, height=2.32, annotate=False, bar_ticks=4,
        compact_xticks=True, xlabel="locality (2q gates inside a block)",
    ),
}

#: Locality below ~0.5 is the Erdos-Renyi case with no structure to recover, so the grid
#: starts there: the bottom of the axis is the control that shows the gain going to zero
#: for a structural reason rather than an algorithmic one.
DEFAULT_LOCALITY = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Matches `lsr_tvd_vs_k` and `ablation_vs_k` so the three figures describe one sweep.
DEFAULT_K_VALUES = (2, 4, 6, 8, 10)


@dataclass
class Settings:
    locality_values: Sequence[float] = DEFAULT_LOCALITY
    k_values: Sequence[int] = DEFAULT_K_VALUES
    #: 20, not the 12 the sibling figures use, because the y axis has to actually vary.
    #: `ceil(n/k)` for k = 2..10 gives 6,3,2,2,2 at n=12 -- three of five rows are the same
    #: two-patch problem -- against 10,5,4,3,2 at n=20, which is five distinct ones. n=24
    #: is both slower and worse (12,6,4,3,3). Measured: ~60 min for the grid at 20 against
    #: ~48 at 12 and ~105 at 24.
    num_qubits: int = 20
    gates_per_qubit: int = 10
    #: Zero, as in both sibling figures: S neither merges nor moves with the placement, so
    #: on a figure about placement it only adds variance. The immutable axis was measured
    #: and dropped -- see the module docstring.
    immutable_ratio: float = 0.0
    #: Held fixed rather than tracked to k, following `lsr_tvd_vs_k`. Fixing it means the
    #: circuit is identical along a row and only the code changes, which is what makes the
    #: k axis mean one thing; it also lets block/patch alignment vary with k, which is the
    #: honest case, since a circuit's community structure does not know your code size.
    block_size: int = 4
    #: Higher than the line figures use. A line figure draws its spread as a band, so a
    #: noisy point is visibly noisy; a heatmap cell is one number with nothing to say how
    #: well determined it is, and cell-to-cell scatter reads as structure. Re-run with
    #: `--circuit-seed` changed to check that a feature is the sweep rather than the draw.
    circuits: int = 8
    circuit_seed: int = 20260729
    sim: SimSettings = field(default_factory=SimSettings)
    shared_colorbar: bool = False
    #: Print each cell's value on it. On by default: the panels carry different colour
    #: scales, so the number is what makes them safe to read side by side.
    annotate: bool = True
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False



def compute(settings: Settings) -> pd.DataFrame:
    rows: List[Dict] = []
    # The circuit depends on (seed, locality) and not on k, so it is built once per row
    # and reused across the whole k axis. That is not just a saving: it is what makes the
    # y axis mean one thing, because every cell in a row is the *same program* placed into
    # a different code rather than a fresh draw that happens to share a seed.
    rows_of_grid = [
        (index, locality)
        for index in range(settings.circuits)
        for locality in settings.locality_values
    ]
    points = [(index, locality, k) for index, locality in rows_of_grid
              for k in settings.k_values]

    circuits = {}
    for index, locality in rows_of_grid:
        seed = settings.circuit_seed + index
        blocks = plant_blocks(settings.num_qubits, settings.block_size, seed=seed)
        circuits[(index, locality)] = (
            structured_program_circuit(
                settings.num_qubits,
                settings.gates_per_qubit,
                seed=seed,
                blocks=blocks,
                locality=locality,
                immutable_ratio=settings.immutable_ratio,
                clifford_only=True,
            ),
            blocks,
        )

    for index, locality, k in tqdm(points, desc="circuit x locality x k"):
        seed = settings.circuit_seed + index
        circuit, blocks = circuits[(index, locality)]
        n_code = k + 2
        naive = naive_packing(settings.num_qubits, k)

        calibration = calibrate(
            circuit, naive, settings.sim,
            n_code=n_code,
            baseline_gate_count=baseline_gate_count(circuit, naive, n_code),
        )

        # Let the planner decide whether H commutation pays on this circuit, then honour
        # its answer for *both* rows. `plan.h_commutation` is what the plan was priced
        # under, and translating the placement with the pass in the other state gives a
        # circuit the plan never costed.
        try:
            # Bug B6: the planner's noise costs default to zero, so without this it
            # optimises gates and depth while the simulator charges movement and idling
            # -- answering a different question from the one measured.
            plan = plan_packing(
                circuit, n_code, h_commutation="auto", verify=True,
                noise=noise_profile_for(settings.sim),
            )
            aqorn, commute = plan.placement, plan.h_commutation
        except Exception:  # noqa: BLE001
            aqorn, commute = naive, True

        # The pipeline can silently miscompile. On `qpe_n9` at k = 2 the combination
        # (packed placement, merging on, H commutation off) computes the wrong distribution
        # while the selection rate stays at 1.0, so neither post-selection nor
        # `plan_packing(verify=True)` -- which compares cost, not correctness -- catches it.
        # Skip rather than fall back to naive: substituting the baseline for the arm under
        # test pulls the cell towards zero improvement and reads as a result.
        broken = check_compile(
            circuit, naive, aqorn, settings.sim, n_code=n_code, h_commutation=commute,
        )
        if broken:
            print(f"[{NAME}] skipping seed {seed} k {k} locality {locality}: {broken}")
            continue

        # One config, both placements: same alpha, same shot count, same passes, so the
        # difference between the two rows is the placement and nothing else.
        config = build_config(
            settings.sim, n_code=n_code,
            alpha=calibration.alpha, repetitions=calibration.repetitions,
            h_commutation=commute, merging=True,
        )
        measured = {
            "naive": evaluate(circuit, naive, config),
            "aqorn": evaluate(circuit, aqorn, config),
        }
        if any(value is None for value in measured.values()):
            continue

        for series, result in measured.items():
            placement = naive if series == "naive" else aqorn
            rows.append({
                "seed": seed,
                "num_qubits": settings.num_qubits,
                "program_gates": circuit.size(),
                "k": k,
                "n_code": n_code,
                # Requested values index the grid; realised values record what the
                # generator actually drew, so a reader can check the axis is honest.
                "locality": locality,
                "immutable_ratio": settings.immutable_ratio,
                "intra_block_fraction": intra_block_fraction(circuit, blocks),
                "immutable_fraction": immutable_fraction(circuit),
                "block_size": settings.block_size,
                "series": series,
                "patches": len({patch for patch, _ in placement}),
                "alpha": calibration.alpha,
                "repetitions": calibration.repetitions,
                "pilot_selection_rate": calibration.pilot_selection_rate,
                "h_commutation_chosen": commute,
                **{metric: result[metric] for metric in PANELS},
            })

    return pd.DataFrame(rows)


def paired_improvement(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per-circuit percent improvement of AQORN over naive, every metric signed so up is better.

    Paired, not a difference of the two series' means. The circuits differ from each other
    far more than the placements differ on any one circuit, so unpaired cells would be
    mostly circuit draw. Pairing cancels the circuit and leaves the placement, which is
    the only thing the figure is asking about.
    """
    wide = frame.pivot_table(
        index=["seed", "locality", "k"],
        columns="series",
        values=list(PANELS),
    )
    rows: List[Dict] = []
    for (seed, locality, k), row in wide.iterrows():
        record = {"seed": seed, "locality": locality, "k": k}
        for metric, (_, higher_is_better) in PANELS.items():
            naive, aqorn = row[(metric, "naive")], row[(metric, "aqorn")]
            if not naive:
                record[metric] = np.nan
                continue
            gain = (aqorn - naive) if higher_is_better else (naive - aqorn)
            record[metric] = 100.0 * gain / naive
        rows.append(record)
    return pd.DataFrame(rows)


def cell_means(
    improvement: pd.DataFrame,
    metric: str,
    locality_values: Sequence[float],
    k_values: Sequence[int],
) -> np.ndarray:
    """Grid of mean improvement, indexed `[k, locality]` to match `pcolormesh`."""
    grid = np.full((len(k_values), len(locality_values)), np.nan)
    for row, k in enumerate(k_values):
        for column, locality in enumerate(locality_values):
            cell = improvement.loc[
                (improvement["k"] == k) & (improvement["locality"] == locality),
                metric,
            ].dropna()
            if len(cell):
                grid[row, column] = float(cell.mean())
    return grid


#: Accepted shots below which the measured TVD is mostly sampling error rather than
#: logical error. Calibrated on this figure's own circuits: fitting `T = a/sqrt(N) + b` to
#: 43k/173k/691k accepted shots gave `a` near 13.7 and a true TVD near 0.011, so the floor
#: only drops under the signal past roughly 1.5M accepted shots. See the module docstring.
TVD_FLOOR_SHOTS = 1_500_000


def _warn_if_tvd_is_floor(frame: pd.DataFrame) -> None:
    """
    Say so, on every run, when the shot budget cannot resolve a TVD difference -- and by
    how much the differing floors are moving the answer.

    The two placements accept different numbers of shots, and that difference *is* the LSR
    result, so the better placement automatically sits on a lower sampling floor. Where
    TVD is floor-dominated it therefore posts a TVD "improvement" with no real effect
    behind it at all. On the 20-qubit sweep that artefact is +6.8% against a measured
    +3.9%, so the panel reads as a clean win while the floor-corrected effect is -2.9%
    (t = -12.4) -- the figure would claim the opposite of what the data says.

    So print the decomposition rather than a bare caution. Under `T ~ a/sqrt(N)` the
    shot-count advantage alone buys `1 - sqrt(N_naive/N_aqorn)`; whatever is left over is
    the part that might be real. This is an order-of-magnitude correction, not a
    substitute for running enough shots -- it assumes TVD is pure floor, which is the
    regime it is warning about.
    """
    accepted = frame["lsr"] * frame["repetitions"]
    by_series = frame.assign(accepted=accepted).groupby("series")["accepted"].mean()
    print(
        "  accepted shots/point: "
        + ", ".join(f"{series} {value:,.0f}" for series, value in by_series.items())
    )
    if float(accepted.mean()) >= TVD_FLOOR_SHOTS:
        return

    wide = frame.assign(accepted=accepted).pivot_table(
        index=["seed", "locality", "k"], columns="series", values=["tvd", "accepted"]
    )
    measured = 100.0 * (wide[("tvd", "naive")] - wide[("tvd", "aqorn")]) / wide[("tvd", "naive")]
    artefact = 100.0 * (1.0 - np.sqrt(wide[("accepted", "naive")] / wide[("accepted", "aqorn")]))
    residual = (measured - artefact).dropna()
    spread = residual.std(ddof=1) / max(np.sqrt(len(residual)), 1.0)

    print(
        f"  WARNING: TVD is sampling-floor dominated below ~{TVD_FLOOR_SHOTS:,} accepted "
        f"shots. The panel is not merely imprecise -- the two series sit on\n"
        f"           different floors, so the better placement posts a TVD gain it did "
        f"not earn. Decomposition over {len(residual)} paired points:\n"
        f"             measured           {measured.mean():+6.2f}%\n"
        f"             shot-count artefact{artefact.mean():+6.2f}%   "
        f"(what zero real effect looks like)\n"
        f"             residual           {residual.mean():+6.2f}%   "
        f"+/- {spread:.2f} (t = {residual.mean() / spread if spread else float('nan'):+.1f})\n"
        f"           Do not plot TVD as a claim at this budget. Rerun with "
        f"--target-kept-shots {16 * 20_000:,} to resolve it."
    )


def print_summary(frame: pd.DataFrame, improvement: pd.DataFrame) -> None:
    """
    Grid means for all five metrics, including the ones no layout draws.

    1q never reaches a panel and 2q only reaches the quad, but both get quoted in the
    prose, so the numbers behind those sentences have to come out of the same run that
    made the figure rather than from a second sweep that might not match it.
    """
    localities = sorted(improvement["locality"].unique())
    k_values = sorted(improvement["k"].unique())
    circuits = improvement["seed"].nunique()
    reps = frame["repetitions"].mean()
    commuted = frame.loc[frame["series"] == "aqorn", "h_commutation_chosen"].mean()

    print(
        f"\n[{NAME}] AQORN improvement over naive (%), mean of {circuits} circuits per cell"
        f"   [{reps:.0f} reps avg, commutation chosen {100 * commuted:.0f}% of cells]"
    )
    # Patch count is per-k rather than one grid number, because it is the whole content of
    # the y axis: k reaches the placement only through how many patches it implies. Two k
    # values with the same patch count are two rows of the same experiment, which looks
    # like a measured flat trend rather than the duplicate it is -- so say it out loud.
    patches = frame.loc[frame["series"] == "aqorn"].groupby("k")["patches"].mean()
    print("  patches by k: " + ", ".join(f"k={k} {v:.1f}" for k, v in patches.items()))
    duplicated = len(patches) - patches.round().nunique()
    if duplicated:
        print(
            f"  WARNING: {duplicated} of {len(patches)} k values share a patch count with "
            f"another, so those rows are the same placement problem drawn twice.\n"
            f"           Widen --num-qubits until ceil(n/k) is distinct across the k values."
        )
    _warn_if_tvd_is_floor(frame)
    for metric, (label, _) in PANELS.items():
        grid = cell_means(improvement, metric, localities, k_values)
        print(f"\n  {label}    rows: k, cols: locality")
        print("        " + "".join(f"{value:>8.2f}" for value in localities))
        for row, k in enumerate(k_values):
            print(f"   {k:>4d} " + "".join(f"{v:>8.2f}" for v in grid[row]))
        print(f"    overall {np.nanmean(grid):+.2f}%   "
              f"range {np.nanmin(grid):+.2f}% to {np.nanmax(grid):+.2f}%")


#: The colour machinery -- equal colour rate either side of zero, per-cell numbers picked
#: for contrast, cell edges midway between sample points -- lives in
#: `experiments.shared.figures` so this figure and `improvement_vs_noise_and_k` draw the
#: same heatmap. The reasoning that fixed each of them is in those docstrings.
_scale = improvement_scale
_annotate = annotate_cells
_edges = cell_edges


def _cap_ticks(bar, limit: Optional[int]) -> None:
    """Thin a colourbar's ticks to at most `limit`, or leave matplotlib's choice alone."""
    if limit is None:
        return
    bar.locator = MaxNLocator(nbins=limit)
    bar.update_ticks()


def _xtick_labels(values: Sequence[float], compact: bool) -> List[str]:
    """Locality tick labels, optionally without the redundant leading zero."""
    if not compact:
        return [f"{value:g}" for value in values]
    return [f"{value:g}".lstrip("0") or "0" for value in values]


def plot_layout(
    improvement: pd.DataFrame,
    settings: Settings,
    suffix: str,
    layout: Layout,
) -> List[Path]:
    nrows, ncols, metrics = layout.nrows, layout.ncols, layout.metrics
    annotate = settings.annotate if layout.annotate is None else layout.annotate
    localities = sorted(improvement["locality"].unique())
    k_values = sorted(improvement["k"].unique())
    x_edges, y_edges = _edges(localities), _edges(k_values)
    grids = {
        metric: cell_means(improvement, metric, localities, k_values)
        for metric in metrics
    }

    # A shared bar needs the panels to be on comparable scales. They are not: LSR
    # improvement spans 2% to 64% across the grid while 2q spans 1% to 9%, so one scale
    # draws the 2q panel as a near-uniform white field and destroys exactly the trend the
    # heatmap exists to show. Per-panel bars keep white pinned at zero in every panel,
    # which is the part of the encoding that actually has to be shared.
    shared_scale = (
        _scale(np.concatenate([grid.ravel() for grid in grids.values()]))
        if settings.shared_colorbar else None
    )

    # Panels are square-ish so the cells read as cells: canvas width divided by the column
    # count, plus a fixed allowance for the colourbar and the shared axis labels. The
    # allowance is absolute rather than proportional, so on a canvas narrow enough for it
    # to dominate the layout states its height outright instead.
    panel = (layout.width - 0.9) / ncols
    height = layout.height if layout.height is not None else nrows * panel * 0.92 + 0.55
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(layout.width, height),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    meshes = []
    for index, (axis, metric) in enumerate(zip(axes, metrics)):
        label, _ = PANELS[metric]
        cmap, norm = shared_scale if shared_scale is not None else _scale(grids[metric])
        mesh = axis.pcolormesh(
            x_edges, y_edges, grids[metric],
            cmap=cmap, norm=norm,
            shading="flat", edgecolors="white", linewidth=0.4,
        )
        meshes.append(mesh)
        if annotate:
            _annotate(axis, grids[metric], localities, k_values, cmap, norm)
        # Panel letter bold and outside the axes, metric name centred above it -- two
        # titles on one axis rather than a fourth type size for the hierarchy.
        axis.set_title(f"({chr(ord('a') + index)})", loc="left", fontweight="bold")
        axis.set_title(label, loc="center")
        axis.set_xticks(localities, _xtick_labels(localities, layout.compact_xticks))
        axis.set_yticks(k_values)
        # A heatmap draws its own cells; gridlines from the house style would land on top
        # of them. The white cell edges above are the separator instead.
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)

    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    if shared_scale is not None:
        bar = fig.colorbar(meshes[0], ax=list(axes[:len(metrics)]), pad=0.02, aspect=30)
        bar.set_label("improvement over naive (%)")
        bar.ax.tick_params(length=2, width=0.6)
        bar.outline.set_visible(False)
        _cap_ticks(bar, layout.bar_ticks)
    else:
        # Each panel carries its own bar and therefore its own scale, so the quantity is
        # stated once for the figure rather than repeated on every bar.
        for axis, mesh in zip(axes, meshes):
            bar = fig.colorbar(mesh, ax=axis, pad=0.03, aspect=18)
            bar.ax.tick_params(length=2, width=0.6)
            bar.outline.set_visible(False)
            _cap_ticks(bar, layout.bar_ticks)
        fig.suptitle("AQORN improvement over naive packing (%)")

    fig.supxlabel(layout.xlabel)
    # The code family goes in the caption rather than the label: spelling [[k+2, k, 2]]
    # here, as the k line figures do on their x axis, is too long for a rotated y label
    # beside a heatmap and would force the panels narrower.
    fig.supylabel("logical qubits per patch, $k$")

    paths = save_figure(fig, f"{NAME}{suffix}", Path(__file__), settings.output_override)
    plt.close(fig)
    return paths


def plot(improvement: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    paths: List[Path] = []
    for suffix, layout in LAYOUTS.items():
        paths.extend(plot_layout(improvement, settings, suffix, layout))
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--locality-values", nargs="+", type=float,
                        default=list(defaults.locality_values),
                        help="x axis. Below ~0.5 the interaction graph has no communities.")
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=list(defaults.k_values),
                        help="y axis. Logical qubits per patch, in the [[k+2, k, 2]] family.")
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits,
                        help="Program width. Must exceed max(k) so even the largest code "
                             "needs more than one patch. Wide enough that ceil(n/k) is "
                             "distinct across the k values, or rows of the grid duplicate "
                             "each other; the summary prints patches by k to check.")
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--immutable-ratio", type=float, default=defaults.immutable_ratio,
                        help="Share of the gate stream that merging cannot fuse. Zero by "
                             "default: it was the previous y axis and proved second-order.")
    parser.add_argument("--block-size", type=int, default=defaults.block_size,
                        help="Planted community size, held fixed across the k axis so the "
                             "circuit is the same along a row and only the code changes.")
    parser.add_argument("--circuits", type=int, default=defaults.circuits,
                        help="Random circuits averaged per cell. Each costs two "
                             "calibration pilots plus two measured runs.")
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    parser.add_argument("--no-annotate", dest="annotate", action="store_false",
                        help="Drop the per-cell numbers. They are on by default because "
                             "each panel has its own colour scale, so the same navy means "
                             "different things across panels.")
    parser.add_argument("--shared-colorbar", action="store_true",
                        help="One colour scale across panels. Off by default: the panels' "
                             "ranges differ several-fold, and one scale flattens the "
                             "smaller ones to near-uniform white.")
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it here: this "
                             "run is hours of simulation.")
    args = parser.parse_args()
    return Settings(
        locality_values=args.locality_values,
        k_values=args.k_values,
        num_qubits=args.num_qubits,
        gates_per_qubit=args.gates_per_qubit,
        immutable_ratio=args.immutable_ratio,
        block_size=args.block_size,
        circuits=args.circuits,
        circuit_seed=args.circuit_seed,
        sim=settings_from_args(args, defaults.sim),
        shared_colorbar=args.shared_colorbar,
        annotate=args.annotate,
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
    )


def main() -> None:
    settings = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
    else:
        if settings.num_qubits <= max(settings.k_values):
            raise ValueError(
                f"--num-qubits {settings.num_qubits} is not larger than max k "
                f"{max(settings.k_values)}; at the largest code the whole program would "
                "fit one patch and there would be no placement to make."
            )
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
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
    for path in plot(improvement, settings):
        print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
