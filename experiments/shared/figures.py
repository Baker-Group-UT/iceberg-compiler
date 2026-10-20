"""
figures.py -- the plumbing every figure script repeats.

Style loading, where output goes, and writing both formats. Three things, in one place,
because eleven copies of `plt.style.use(...)` is eleven chances for one of them to drift.

Output resolution deserves a note. `docs/figure-style.md` says generated plots land in
`results/figures/` at the *paper* project root, but this repo is its own git repo with
its own remote and has to stay runnable when cloned on its own. So walk up looking for a
marker; if there is no paper around this checkout, fall back to a directory beside the
script and the figure still builds.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Sequence

# Import-time backend selection. A figure script run over ssh or from a cron job has no
# display, and matplotlib picking an interactive backend there fails at import rather
# than at plot time, which is a confusing place to find out.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "STYLE",
    "COLUMN_WIDTH",
    "FULL_WIDTH",
    "use_style",
    "project_root",
    "output_dir",
    "data_dir",
    "save_figure",
    "next_version",
    "stack_heights",
    "MAX_FACTOR",
    "FACTOR_TICKS",
    "factor_from_pct",
    "factor_ticks",
    "format_factor",
    "stacked_percent",
    "stacked_factors",
    "geometric_mean_bar",
    "apply_factor_axis",
    "HEATMAP_CMAP",
    "heatmap_norm",
    "improvement_scale",
    "annotate_cells",
    "cell_edges",
    "stamp_provenance",
    "check_provenance",
]

STYLE = Path(__file__).resolve().parent / "aqorn.mplstyle"

#: IEEEtran at 10pt: 252pt and 516pt. Draw at final size so `width=\linewidth` scales
#: 1:1 and 10pt type in the figure renders as 10pt on the page.
COLUMN_WIDTH = 3.487
FULL_WIDTH = 7.14

#: Files that mark the paper project root. `CLAUDE.md` is the conventional marker;
#: `docs/figure-style.md` is checked too because this project keeps the style guide
#: rather than a CLAUDE.md at that level, and the style guide is the file that actually
#: specifies `results/figures/`.
_ROOT_MARKERS = ("CLAUDE.md", "docs/figure-style.md")


def use_style() -> None:
    """Load the house style. Call before creating any figure, and before qiskit's drawer."""
    plt.style.use(STYLE)


def project_root() -> Optional[Path]:
    """Nearest ancestor that looks like the paper project, or None when the repo stands alone."""
    for parent in Path(__file__).resolve().parents:
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    return None


def _resolve(kind: str, script: Path, override: Optional[Path]) -> Path:
    if override:
        target = override
    else:
        root = project_root()
        target = (root / "results" / kind) if root else script.parent / "out"
    target.mkdir(parents=True, exist_ok=True)
    return target


def output_dir(script: Path, override: Optional[Path] = None) -> Path:
    """Where the SVG and PDF go. Pass `Path(__file__)` as `script`."""
    return _resolve("figures", script, override)


def data_dir(script: Path, override: Optional[Path] = None) -> Path:
    """
    Where a `--persist`ed CSV goes.

    The default cache lives next to the script; this is only for runs expensive enough
    that losing the numbers would hurt.
    """
    return _resolve("data", script, override)


def next_version(name: str, script: Path, override: Optional[Path] = None) -> str:
    """
    `name`, or `name_v2`, `name_v3`... -- whichever does not exist yet.

    A figure gets regenerated many times in the week before a deadline, often overnight
    with nobody watching, and the second run silently replacing the first destroys the
    only copy of a plot that may have taken hours to simulate. Versioning makes a re-run
    additive, so the comparison between two runs is still possible in the morning.

    Both candidate locations are checked -- `results/figures/` and the script-local copy
    `save_figure` writes -- so a name is only reused when it is free in both.
    """
    target = output_dir(script, override)
    local = script.resolve().parent

    def taken(candidate: str) -> bool:
        return any(
            (directory / f"{candidate}.{suffix}").exists()
            for directory in (target, local)
            for suffix in ("svg", "pdf")
        )

    if not taken(name):
        return name
    version = 2
    while taken(f"{name}_v{version}"):
        version += 1
    return f"{name}_v{version}"


def save_figure(
    fig,
    name: str,
    script: Path,
    override: Optional[Path] = None,
    versioned: bool = False,
) -> List[Path]:
    """
    Write `<name>.svg` and `<name>.pdf`, plus a copy of the SVG beside the script.

    Both formats from one run: the paper includes the PDF, and the SVG is what gets
    promoted to `figures-wip/` when a figure needs hand work in Inkscape.

    The script-local SVG is a convenience copy, so the figure sits next to the code and
    the cached CSV that produced it and can be eyeballed without leaving the directory.
    `results/figures/` remains the canonical output that the paper pipeline reads.

    `versioned=True` writes to the next free `_vN` name rather than replacing what is
    there. See `next_version`.
    """
    if versioned:
        name = next_version(name, script, override)
    target = output_dir(script, override)
    paths = []
    for suffix in ("svg", "pdf"):
        path = target / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(path)

    local = script.resolve().parent / f"{name}.svg"
    if local != paths[0]:
        fig.savefig(local)
        paths.append(local)
    return paths


def stack_heights(
    frame,
    column: str,
    components: Sequence[str],
    cumulative_column: Optional[str] = None,
) -> List[float]:
    """
    Segment heights for a stacked bar, in `components` order, guaranteed sane.

    Clipping each *segment* at zero -- which the draft's scripts do, and which this
    folder inherited -- breaks when one technique hurts and a later one more than
    recovers. On `ising_n98` H commutation costs 88% of depth and merging returns 179%;
    the true total is 90%, but dropping the negative segment leaves a 179% bar on a
    percent-reduction axis.

    So clip the *cumulative* series instead and difference that. Each segment stays
    non-negative, the stack height is the real full-pipeline reduction, and a technique
    that hurts simply contributes nothing rather than inflating the one after it.

    Falls back to per-segment means when the cumulative column is absent, so older CSVs
    still plot.

    `stacked_factors` is the factor-axis equivalent and the one to reach for in new code.
    It differs in more than units: this one clips a running max, which over-reports
    whenever a *later* stage hurts (22 of 66 TVD circuits, by 1.1 points on average),
    while that one pins the bar height to the true full-pipeline total.
    """
    cumulative_column = cumulative_column or f"cumulative_{column}"
    if cumulative_column not in getattr(frame, "columns", ()):
        return [
            max(0.0, float(frame[frame.component == c][column].mean() or 0.0))
            for c in components
        ]

    running = 0.0
    heights: List[float] = []
    for component in components:
        values = frame[frame.component == component][cumulative_column]
        mean = float(values.mean()) if len(values) else running
        mean = 0.0 if mean != mean else mean  # NaN guard
        clipped = max(running, mean)
        heights.append(clipped - running)
        running = clipped
    return heights


# --- the factor convention ----------------------------------------------------------
#
# The paper reports improvements as multipliers rather than percentages. The two carry the
# same information, but percentages compress exactly where the results are strongest --
# a 67% depth reduction and a 75% one are 3x and 4x, four percentage points apart on a
# scale that has already spent two thirds of its range on the first doubling -- and they
# are ambiguous about direction, since "67% better" reads as 1.67x for a rate and 3x for a
# cost. A multiplier is unambiguous once you know which way the metric points, which is
# what `lower_is_better` names.
#
# Everything here is display-only. Nothing rewrites a stored CSV: the columns stay in
# percent, and the conversion happens at the point of drawing.

#: Ceiling on a plotted factor. A 100% reduction means a challenger of exactly zero and an
#: unbounded factor; anything that close is a measurement artefact rather than an infinite
#: win. The merging corpus tops out at 10.1x, so this only stops a degenerate row from
#: blowing an axis out.
MAX_FACTOR = 100.0

#: Tick ladders, finest to coarsest. Each is `(mantissas, octaves per step)`: the ticks are
#: every `mantissa * 2 ** (period * k)`, so the first entry gives 1x, 1.5x, 2x, 3x, 4x, 6x,
#: 8x and the second gives plain octaves.
#:
#: Round numbers and perfectly even spacing are in tension, and this is where the tension
#: gets resolved. Within an octave the evenly-spaced-in-log values are irrational -- two per
#: octave means 1.41x, four means 1.19x, 1.41x, 1.68x -- so a ladder that is exactly even is
#: also a ladder nobody wants to read. Halving each octave instead alternates a x1.5 step
#: with a x1.33 one: gaps differ by 41%, against 84% for a fixed list of round numbers like
#: 1, 1.1, 1.3, 1.5, 2, 2.5, 3. Regular enough to read as a scale, round enough to quote.
#:
#: The coarser entries are all single-mantissa, so they *are* exactly even -- a wide axis
#: gets pure powers of 2, 4, 8 or 16 and loses nothing.
FACTOR_TICKS = (
    ((1.0, 1.5), 1),
    ((1.0,), 1),
    ((1.0,), 2),
    ((1.0,), 3),
    ((1.0,), 4),
)


def factor_from_pct(pct: float, *, reduction: bool, floor: float = 1.0) -> float:
    """
    A stored percentage -> the multiplier to draw. `reduction` names the input convention.

    This project writes percentages two ways, and they are not the same quantity:

      `reduction=True`   the column is a *reduction*, positive meaning smaller.
                         `pct = 100 (ref - chal) / ref`, so the factor is
                         `ref / chal = 1 / (1 - pct/100)`. A 67% reduction is 3.03x.
      `reduction=False`  the column is a plain *change*, positive meaning larger.
                         `pct = 100 (chal - ref) / ref`, so the factor is
                         `chal / ref = 1 + pct/100`. A +121% change is 2.21x, and a -55%
                         change is 0.45x.

    Feeding a change column through the reduction branch is the mistake this parameter
    exists to prevent: `+121` would come out clamped at `MAX_FACTOR` instead of 2.21x,
    because `1 - 1.21` is negative.

    Whether *higher is better* is a separate question, and deliberately not asked here.
    An axis that wants "up is worse" for a higher-is-better series inverts the factor at
    the call site, which keeps the two directions exact reciprocals.

    `floor` is the smallest factor returned: 1.0 suits a bar growing from a baseline, and
    0.0 suits a two-sided axis where "0.45x" is a reading rather than something to clip.
    """
    if pct != pct:  # NaN
        return float("nan")
    if reduction:
        remaining = 1.0 - pct / 100.0
        if remaining <= 1.0 / MAX_FACTOR:
            return MAX_FACTOR
        return min(MAX_FACTOR, max(floor, 1.0 / remaining))
    return min(MAX_FACTOR, max(floor, 1.0 + pct / 100.0))


def factor_ticks(top: float, bottom: float = 1.0, max_ticks: int = 7) -> List[float]:
    """
    Round, near-evenly spaced ticks for a log factor axis. See `FACTOR_TICKS`.

    Walks the ladders from finest to coarsest and takes the first that fits inside
    `max_ticks`, so a narrow panel gets 1x, 1.5x, 2x, 3x and a wide one gets plain octaves
    without either crowding.

    Every ladder is anchored on exact powers of two, so 1x is always a tick when it is in
    range -- which matters for the two-sided tradeoff axes, where 1x is the reference
    every series is measured against.
    """
    if top <= bottom:
        return [bottom]

    def ladder(mantissas, period, limit):
        step = 2.0 ** period
        values, exponent = [], math.floor(math.log(bottom, step)) - 1
        while step ** exponent <= limit * 1.001:
            values.extend(m * step ** exponent for m in mantissas)
            exponent += 1
        return sorted(values)

    for mantissas, period in FACTOR_TICKS:
        values = [
            v for v in ladder(mantissas, period, top)
            if bottom * 0.999 <= v <= top * 1.001
        ]
        if 2 <= len(values) <= max_ticks:
            return values
        if len(values) < 2:
            # Range too narrow to hold two ticks -- a row topping out at 1.4x on the
            # 1/1.5 ladder sees only 1x. Reach one rung past the data and let
            # `apply_factor_axis` open the limit to match, which is better than falling
            # through to a coarser ladder that would hold even fewer.
            above = [v for v in ladder(mantissas, period, top * step_guess(period))
                     if v > top * 1.001]
            if values and above:
                return values + above[:1]
    return [bottom, top]


def step_guess(period: int) -> float:
    """How far past `top` to look for one more rung. One full period is always enough."""
    return 2.0 ** period


def format_factor(value: float) -> str:
    """
    A factor as it appears on an axis or in a cell.

    Literal U+00D7 rather than mathtext `$\\times$`: the shared style keeps SVG text as
    editable `<text>` elements, and mathtext defeats that by splitting the string into
    glyph paths. Two significant figures below 10x, none above -- `1.15x` earns its
    decimal, `10.1x` does not.
    """
    if value != value:
        return ""
    return f"{value:.2f}×" if value < 10.0 else f"{value:.0f}×"


def stacked_percent(deltas: Sequence[float]) -> List[float]:
    """
    Stage deltas -> the cumulative percentage boundaries of one stacked bar, `0.0` first.

    The fix that matters, independent of whether the axis is percent or factors. `deltas`
    are differences of cumulative percentages, so their running sum is the real effect of
    enabling stages 1..i and the full sum is what the pipeline delivers. The original
    version of these figures clipped each delta at zero *independently* and stacked the
    clipped values, which sums to nothing in particular: on `ising_n98` the depth stages are
    -108.8 then +198.8, a true reduction of 90.1%, and the figure drew a 198.8% bar -- more
    than double, and past the 100% a reduction cannot exceed. Across the merging corpus that
    inflated the plotted depth average from 28.9% to 35.8% and TVD from 27.3% to 33.1%.

    So the total is pinned to the truth here, and only the split between stages is a display
    choice. Where the cumulative path never dips -- 62/71 circuits on 1q gates, 65/71 on
    depth -- the boundaries are the honest per-stage values and nothing is approximated.
    Where it does dip, a stage that hurt would need a downward segment the stack cannot
    draw, so its share drops to zero and the rest is split among the stages that helped in
    proportion to what they contributed. Either way the top of the bar is exactly what that
    circuit achieved.
    """
    values = [float(d) for d in deltas]
    total = max(0.0, sum(values))
    if total <= 0.0:
        return [0.0] * (len(values) + 1)

    running = 0.0
    path: List[float] = []
    for value in values:
        running += value
        path.append(running)
    if all(b >= a - 1e-9 for a, b in zip([0.0] + path, path)):
        return [0.0] + path

    positive = [max(v, 0.0) for v in values]
    share = sum(positive)
    if share <= 0.0:
        return [0.0] * (len(values) + 1)
    boundaries, carried = [0.0], 0.0
    for value in positive:
        carried += value
        boundaries.append(total * carried / share)
    return boundaries


def stacked_factors(deltas: Sequence[float], *, reduction: bool) -> List[float]:
    """
    Stage deltas -> the cumulative factor boundaries of one stacked bar, `1.0` first.

    `deltas` are differences of cumulative percentages, so their running sum is the real
    effect of enabling stages 1..i and the full sum is what the pipeline actually delivers.
    The percentage version of these figures clipped each delta at zero *independently* and
    stacked the clipped values, which sums to nothing in particular: on `ising_n98` the
    depth stages are -108.8 then +198.8, a true reduction of 90.1%, and the figure drew a
    198.8% bar -- more than double, and past the 100% a reduction cannot exceed. Across the
    merging corpus that inflated the plotted depth average from 28.9% to 35.8% and TVD from
    27.3% to 33.1%. The factor axis forces the issue, since a 198.8% "reduction" has no
    finite multiplier.

    So the total is pinned to the truth here, and only the split between stages is a
    display choice. Where the cumulative path never dips -- 62/71 circuits on 1q gates,
    65/71 on depth -- the boundaries are the honest per-stage factors and nothing is
    approximated. Where it does dip, a stage that hurt would need a downward segment the
    stack cannot draw, so its share drops to zero and the remaining span is split among the
    stages that helped in proportion to what they contributed. Either way the top of the
    bar is exactly the factor that circuit achieved.
    """
    values = [float(d) for d in deltas]
    top = factor_from_pct(sum(values), reduction=reduction)
    span = math.log(top)
    if span <= 0.0:
        return [1.0] * (len(values) + 1)

    running = 0.0
    path: List[float] = []
    for value in values:
        running += value
        path.append(factor_from_pct(running, reduction=reduction))
    if all(b >= a - 1e-9 for a, b in zip([1.0] + path, path)):
        return [1.0] + path

    positive = [max(v, 0.0) for v in values]
    share = sum(positive)
    if share <= 0.0:
        return [1.0] * (len(values) + 1)
    boundaries = [1.0]
    carried = 0.0
    for value in positive:
        carried += value
        boundaries.append(math.exp(span * carried / share))
    return boundaries


def geometric_mean_bar(rows, stages: int = 3):
    """
    Average boundaries for a summary bar: geometric mean of each stage's factor.

    The arithmetic mean of ratios neither composes nor survives a change of direction --
    averaging `ref/chal` and averaging `chal/ref` disagree -- whereas the geometric mean
    does both, and on a log axis it is simply the mean bar height. Composition is what
    makes the summary bar readable as a bar: the product of the per-stage geometric means
    is the geometric mean of the per-circuit totals, so the stack still adds up.

    `rows` is (circuits, stages + 1) of cumulative boundaries; the result is one such row.
    """
    import numpy as np

    rows = np.asarray(rows, dtype=float)
    if rows.size == 0:
        return np.ones(stages + 1)
    ratios = np.log(rows[:, 1:] / rows[:, :-1]).mean(axis=0)
    return np.concatenate([[1.0], np.exp(np.cumsum(ratios))])


def apply_factor_axis(axis, top: float, *, bottom: float = 1.0, max_ticks: int = 7,
                      baseline: bool = True) -> None:
    """
    Put one axis into factor space: log scale, `x`-suffixed ticks, baseline rule at 1x.

    Log rather than linear because factor rows routinely disagree about scale by an order
    of magnitude -- merging's depth row reaches 10.1x against a median of 1.37x, and on a
    linear axis that one bar flattens every other. Log is also what makes a stacked bar
    honest in this space: segments compose by multiplication, so equal visual lengths are
    equal relative gains and a mean of bar heights is the geometric mean of the factors.
    """
    import matplotlib.ticker as mticker

    ticks = factor_ticks(top, bottom=bottom, max_ticks=max_ticks)
    axis.set_yscale("log")
    # A one-sided axis sits exactly on its baseline -- bars grow from 1x and must not float
    # above the floor. A two-sided one needs room under its lowest tick.
    floor = bottom if bottom >= 1.0 else min(ticks[0], bottom) / 1.03
    axis.set_ylim(bottom=floor, top=max(ticks[-1], top) * 1.03)
    axis.set_yticks(ticks)
    # Three significant figures with trailing zeros dropped. The ladder is geometric, so
    # most ticks are irrational (`2 ** (1/3)` = 1.2599...) and `:g` would spell them out in
    # full; rounding harder than this would print two adjacent ticks as the same number.
    axis.set_yticklabels([f"{float(f'{t:.3g}'):g}×" for t in ticks])
    axis.yaxis.set_minor_locator(mticker.NullLocator())
    if baseline:
        axis.axhline(1.0, color="#636363", linestyle="--", linewidth=1.0)


# --- heatmaps ---------------------------------------------------------------------
#
# The navy ramp is the one the published CNOT heatmaps use, kept exactly so the two read
# as the same family. It is extended downwards rather than replaced: a cell here is a
# signed improvement over naive packing, and a purely sequential ramp would draw "worse
# than naive" in the same pale tone as "barely better".
#
# White is pinned to zero by `heatmap_norm`, so the neutral colour means exactly "no
# better than naive" and the sign is readable without consulting the colourbar.
_NEGATIVE_ARM = ["#4a0d1f", "#7d1a30", "#b04a5a", "#e0b3bb"]
_POSITIVE_ARM = ["#b3cde0", "#6497b1", "#005b96", "#03396c", "#011f4b"]

#: Stops are positioned explicitly rather than spaced evenly, so #ffffff sits at exactly
#: 0.5 -- the point `TwoSlopeNorm` maps zero to. Letting `from_list` space them evenly put
#: white at 4/9 and drew a zero cell in pale blue, which is the one thing this scale is
#: supposed to make unambiguous.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "aqorn_improvement",
    [
        *(
            (0.5 * index / len(_NEGATIVE_ARM), colour)
            for index, colour in enumerate(_NEGATIVE_ARM)
        ),
        (0.5, "#ffffff"),
        *(
            (0.5 + 0.5 * (index + 1) / len(_POSITIVE_ARM), colour)
            for index, colour in enumerate(_POSITIVE_ARM)
        ),
    ],
)


def heatmap_norm(values, *, negative_headroom: float = 0.35):
    """
    Colour scale for signed improvements, with white exactly at zero.

    `negative_headroom` caps how much of the scale the red arm can claim when the data
    barely goes negative -- without it, a single -0.2% cell would stretch the red half
    across the whole lower range and make noise look like a regression.
    """
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    finite = np.asarray([v for v in np.ravel(values) if v == v], dtype=float)
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    top = max(float(finite.max()), 1e-9)
    floor = min(float(finite.min()), -1e-9)
    return TwoSlopeNorm(
        vmin=max(floor, -negative_headroom * top), vcenter=0.0, vmax=top
    )


def _half_cmap(start: float, stop: float):
    """One arm of `HEATMAP_CMAP`, resampled to a colormap of its own."""
    import numpy as np

    return LinearSegmentedColormap.from_list(
        f"aqorn_improvement_{start}_{stop}",
        HEATMAP_CMAP(np.linspace(start, stop, 256)),
    )


def improvement_scale(values):
    """
    Colormap and norm for one improvement panel: white at zero, equal colour per unit.

    `TwoSlopeNorm`, which is what `heatmap_norm` returns, splits the colormap at zero and
    gives each side half the colours *regardless of how much data is on it*. That is wrong
    in both directions and the locality figure hit both:

      - A panel that barely dips below zero spends the whole red arm on that sliver. In
        the locality x k sweep the 2q panel has one cell at -0.84% among 29 running to
        +8.8%, and it renders as full-strength darkest red -- a cell that is zero within
        noise, drawn as the most alarming thing on the page.
      - A panel that is entirely one-signed still reserves half the bar for a range of
        zero width. Invisible in the cells, but the colourbar is drawn in colormap space,
        so it prints a red band beside data that never goes negative.

    `heatmap_norm`'s `negative_headroom` is a partial guard against the first, by clipping
    the arm at a fraction of the positive range. Clipping loses magnitude instead: on an
    earlier immutable sweep the default cap put 11 of 30 TVD cells below `vmin`, so -3% and
    -25% arrived in the same red.

    So keep the colour *rate* equal on both arms instead of the colour *budget*. With
    `span = max(|vmin|, vmax)`, sampling `HEATMAP_CMAP` from `0.5 - 0.5|vmin|/span` to
    `0.5 + 0.5*vmax/span` under a plain linear norm puts white exactly at zero and makes a
    given number of percent the same depth of colour whichever side it falls. A tiny
    negative then reads as the faint pink it is, a large one still reaches deep red, and an
    all-positive panel collapses to white-to-navy over [0, max] with no phantom arm --
    the one-signed case drops out of the same formula rather than needing its own branch.
    """
    import numpy as np
    from matplotlib.colors import Normalize

    finite = np.asarray([v for v in np.ravel(values) if v == v], dtype=float)
    if finite.size == 0:
        return _half_cmap(0.5, 1.0), Normalize(vmin=0.0, vmax=1.0)
    # Anchor the range at zero even when the data does not reach it: white has to mean
    # "no better than naive" whether or not any cell landed there.
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    span = max(abs(low), high, 1e-9)
    return (
        _half_cmap(0.5 - 0.5 * abs(low) / span, 0.5 + 0.5 * high / span),
        Normalize(vmin=low if low < 0 else 0.0, vmax=high if high > 0 else 1e-9),
    )


def annotate_cells(axis, grid, x_values, y_values, cmap, norm, formatter=None) -> None:
    """
    Print each cell's value on it, in whichever of black or white reads on that cell.

    This is the answer to the one thing per-panel colour scales get wrong: the same navy
    is +38% in one panel and +7.6% in the next, and a reader skimming across four panels
    will not stop to check each bar. Giving every panel its own hue would signal the
    incomparability, but each panel needs white pinned at zero with a negative arm, so
    four hues means four *diverging* ramps whose arms then have to stay distinct from each
    other's -- and it would not help greyscale, where matched-luminance hues collapse to
    the same ramp anyway. A number on the cell fixes all of it at once, including the
    diverging map's real weakness in print, where a strong negative and a strong positive
    reach similar density.

    Contrast is chosen from the cell's own luminance rather than from a threshold on the
    data, because where the switch has to happen depends on the norm, which differs per
    panel.
    """
    import numpy as np

    at_8pt = plt.rcParams["font.size"] * 0.8
    if formatter is None:
        # Precision follows the panel's range rather than being fixed. A panel spanning
        # 1.7% to 6.3% collapses visibly different shades onto the same digit at whole
        # numbers, while one reaching 38% gains only noise from a decimal. Keying on the
        # largest value also bounds the string to three characters, which is what has to
        # fit in a cell.
        largest = np.nanmax(np.abs(grid)) if np.any(~np.isnan(grid)) else 0.0
        digits = 1 if largest < 10.0 else 0
        formatter = lambda value: f"{value:.{digits}f}"  # noqa: E731
    for row, y in enumerate(y_values):
        for column, x in enumerate(x_values):
            value = grid[row, column]
            if value != value:
                continue
            red, green, blue = cmap(float(norm(value)))[:3]
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            axis.text(
                x, y, formatter(value),
                ha="center", va="center", fontsize=at_8pt,
                color="white" if luminance < 0.55 else "#1a1a1a",
            )


def cell_edges(values):
    """Cell boundaries midway between sample points, so ticks land on cell centres."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 1:
        return np.array([array[0] - 0.5, array[0] + 0.5])
    mids = 0.5 * (array[:-1] + array[1:])
    first = array[0] - 0.5 * (array[1] - array[0])
    last = array[-1] + 0.5 * (array[-1] - array[-2])
    return np.concatenate(([first], mids, [last]))


#: Column every cached CSV carries, so a stale one is detectable rather than merely
#: suspected. See `packing.version`.
PROVENANCE_COLUMN = "cost_model_version"
#: Written beside it when the logical optimisation pass was on (its version); absent means
#: off, which is every CSV made before 2026-10-09.
LOGICAL_COLUMN = "logical_optimization"


def stamp_provenance(frame):
    """
    Record which cost model produced these rows. Call before writing any cached CSV.

    A figure's CSV is the expensive artefact in this repo -- hours of simulation -- and it
    is only reusable if a change to the planner invalidates it. Nothing enforced that: the
    only cache guard that existed covered the shot budget and the noise flags, so a rerun
    after a cost-model fix reused placements the fix was meant to change. One column fixes
    it, and costs nothing.
    """
    from packing.version import COST_MODEL_VERSION
    from translator.logical_optimization import enabled, state

    frame = frame.copy()
    frame[PROVENANCE_COLUMN] = COST_MODEL_VERSION
    if enabled():
        frame[LOGICAL_COLUMN] = state()
    return frame


def check_provenance(frame, source=None, *, strict: bool = False) -> Optional[int]:
    """
    Compare a loaded CSV's cost model against the current one. Returns its version.

    Warns rather than raises by default, because a stale CSV is still worth *looking* at --
    it is how you compare before and after a fix. Pass `strict=True` on a path that
    publishes a number, where quoting a figure computed under a superseded objective is the
    failure this exists to prevent.

    A missing column means the CSV predates provenance entirely, which given when this was
    added means version 2 or earlier -- reported as None rather than guessed at.
    """
    from packing.version import COST_MODEL_VERSION

    where = f" in {source}" if source else ""
    if PROVENANCE_COLUMN not in getattr(frame, "columns", ()):
        message = (
            f"cached data{where} carries no {PROVENANCE_COLUMN}, so it predates cost-model "
            f"provenance and was computed under an objective older than version "
            f"{COST_MODEL_VERSION}. Regenerate before quoting it."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[stale] {message}")
        return None

    versions = sorted({int(v) for v in frame[PROVENANCE_COLUMN].dropna().unique()})
    if len(versions) > 1:
        raise RuntimeError(
            f"cached data{where} mixes cost model versions {versions}. That should be "
            f"impossible -- it means two runs wrote one CSV. Regenerate it."
        )
    found = versions[0] if versions else None
    if found != COST_MODEL_VERSION:
        message = (
            f"cached data{where} was computed under cost model version {found}, current is "
            f"{COST_MODEL_VERSION}. The placements differ; regenerate before quoting it."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[stale] {message}")

    from translator.logical_optimization import state

    recorded = "off"
    if LOGICAL_COLUMN in frame.columns:
        values = {str(v) for v in frame[LOGICAL_COLUMN].dropna().unique()}
        recorded = values.pop() if len(values) == 1 else f"mixed {sorted(values)}"
    if recorded != state():
        message = (
            f"cached data{where} was computed with logical optimisation {recorded}, current "
            f"is {state()} (AQORN_LOGICAL_OPT). The programs differ; regenerate before "
            f"quoting it."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[stale] {message}")
    return found
