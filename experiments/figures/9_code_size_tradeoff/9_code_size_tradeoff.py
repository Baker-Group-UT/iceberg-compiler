"""
9_code_size_tradeoff.py -- the same tradeoff, parameterised by code size instead of density.

**Claim.** Raising `k` at full density is the mirror image of spreading a program thin.
Fewer, larger patches cost depth and cost accuracy, and they buy back two-qubit gates and
spacetime volume. Density and code size are two handles on one resource space, and neither
handle has a setting that wins on every metric.

x = logical qubits per patch `k`, with the patch count following from it
(`patches = ceil(n/k)`, so 6, 3, 2, 2, 2 at n = 12). Panels and sign convention are exactly
`6_packing_density_tradeoff`'s, so the two figures can be read side by side: (a) resource
cost, (b) measured outcome, both as percent change from `k = 2`, both drawn so **up is
worse**.

**Reads the density sweep's CSV rather than simulating again.** That sweep already covers
every `(k, patch count)` pair; this figure is its full-density slice. One simulation, two
readings, and they cannot disagree about the underlying numbers. Re-running it is free.

**Referenced to k = 2**, which is the paper's published code and the baseline the code-size
argument is made against, rather than to the largest code.

**Do not read the selection-rate series as the cost of a larger code.** The density sweep
solves alpha per `(circuit, k)` *at full density*, and this figure is exactly that slice, so
the baseline selection rate is pinned to the calibration target at every `k` by
construction. What is left in the series is only the residual -- how much the pipeline
passes move selection rate above that pinned baseline -- and it is correspondingly small
(under 10 points across the whole family). It is plotted because omitting a series from a
tradeoff panel is its own kind of misleading, and because the residual is genuine, but the
figure that answers "which code size keeps more shots" is `1_best_k_phase_diagram`, which
calibrates once per circuit at a single reference `k` so that the code sizes in a column
share one alpha and are comparable. TVD carries no such caveat: calibration targets
selection rate alone, so nothing pins TVD, and its +121 percent across the family is a
measurement.

**Why this is a separate figure from `3_lsr_tvd_vs_k`.** That one plots absolute selection
rate and TVD against `k`, which is the cost side of the code-size decision on its own terms.
This one puts those two outcomes next to what they buy -- gates, depth, volume -- on a shared
percent axis, which is the tradeoff. The first answers "what does a bigger code cost?"; this
answers "and what does it buy?".

Bands are +-1 standard deviation over the circuits in the sweep.

Run with:

    PYTHONPATH=. python experiments/figures/9_code_size_tradeoff/9_code_size_tradeoff.py
    PYTHONPATH=. python .../9_code_size_tradeoff.py --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.figures import (
    COLUMN_WIDTH,
    MAX_FACTOR,
    apply_factor_axis,
    factor_from_pct,
    format_factor,
    save_figure,
    use_style,
)

NAME = "9_code_size_tradeoff"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: The density sweep this slices. See the module docstring: no simulation of its own.
DEFAULT_SOURCE = (
    SCRIPT.parents[1] / "6_packing_density_tradeoff" / "6_packing_density_tradeoff.csv"
)

#: Identical to `6_packing_density_tradeoff`'s, so the two figures read as a pair.
COST_SERIES: Sequence[Tuple[str, str]] = (
    ("two_qubit_gates", "2q gates"),
    ("circuit_depth", "depth"),
    ("volume", "volume"),
)

OUTCOME_SERIES: Sequence[Tuple[str, str, bool]] = (
    ("lsr", "selection rate (inverted)", True),
    ("tvd", "TVD", False),
)

PLAIN = {"lsr": "selection rate", "tvd": "TVD"}


@dataclass
class Settings:
    source: Path = field(default_factory=lambda: DEFAULT_SOURCE)
    reference_k: int = 2
    output_override: Optional[Path] = None


def full_density(frame: pd.DataFrame) -> pd.DataFrame:
    """The densest packing available at each code size -- the fewest patches that fit."""
    keep = frame.groupby(["circuit_index", "k"])["patches"].transform("min")
    return frame[frame["patches"] == keep].copy()


def relative_to_reference(frame: pd.DataFrame, reference_k: int) -> pd.DataFrame:
    """
    Percent change from `reference_k`, referenced within circuit.

    Within circuit rather than against a pooled mean: circuits differ in absolute gate
    count by more than the code-size effect being measured, so a pooled reference would
    report the spread between circuits rather than the tradeoff.
    """
    columns = [name for name, _ in COST_SERIES] + [name for name, _, _ in OUTCOME_SERIES]
    pieces: List[pd.DataFrame] = []
    for _, circuit in frame.groupby("circuit_index"):
        base_rows = circuit[circuit["k"] == reference_k]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        scaled = circuit.copy()
        for column in columns:
            value = base[column]
            scaled[f"{column}_pct"] = (
                (circuit[column] / value - 1.0) * 100.0 if value else float("nan")
            )
        pieces.append(scaled)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def curves(scaled: pd.DataFrame) -> pd.DataFrame:
    """Mean and +-1 SD per code size, over circuits."""
    columns = [name for name, _ in COST_SERIES] + [name for name, _, _ in OUTCOME_SERIES]
    rows: List[Dict] = []
    for k in sorted(scaled["k"].unique()):
        block = scaled[scaled["k"] == k]
        row: Dict = {
            "k": int(k),
            "patches": int(block["patches"].iloc[0]),
            "circuits": int(block["circuit_index"].nunique()),
        }
        for column in columns:
            values = block[f"{column}_pct"]
            row[f"{column}_pct"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(table: pd.DataFrame, settings: Settings) -> None:
    print(f"\n[{NAME}] percent change from k = {settings.reference_k}, at full density")
    print(f"[{NAME}] raw signs (the figure inverts selection rate so up is worse there)")
    header = (
        "    k  patches" + "".join(f"{label:>16}" for _, label in COST_SERIES)
        + "".join(f"{PLAIN.get(c, label):>16}" for c, label, _ in OUTCOME_SERIES)
    )
    print(header)
    for _, row in table.sort_values("k").iterrows():
        cells = "".join(f"{row[f'{c}_pct']:>16.1f}" for c, _ in COST_SERIES)
        cells += "".join(f"{row[f'{c}_pct']:>16.1f}" for c, _, _ in OUTCOME_SERIES)
        print(f"  {int(row['k']):>3}  {int(row['patches']):>7}" + cells)

    if len(table) >= 2:
        largest = table.loc[table["k"].idxmax()]
        print(
            f"\n[{NAME}] going from k = {settings.reference_k} to k = {int(largest['k'])} "
            f"({int(table.loc[table['k'].idxmin(), 'patches'])} patches down to "
            f"{int(largest['patches'])}):"
        )
        # Factors, matching the axis. Reported uninverted -- the figure flips the
        # higher-is-better series so up reads as worse, but the console is easier to check
        # against the raw table when every line means "this many times the reference".
        for column, label in COST_SERIES:
            value = factor_from_pct(largest[f"{column}_pct"], reduction=False, floor=0.0)
            print(f"    {label:<16} {format_factor(value):>8}")
        for column, label, _ in OUTCOME_SERIES:
            value = factor_from_pct(largest[f"{column}_pct"], reduction=False, floor=0.0)
            print(f"    {PLAIN.get(column, label):<16} {format_factor(value):>8}")

    print(
        f"\n[{NAME}] CAVEAT: alpha is calibrated per (circuit, k) at full density, which "
        f"is this slice, so the baseline selection rate is pinned at every k and the "
        f"selection-rate series shows only the residual above it. Read "
        f"1_best_k_phase_diagram for which code size actually keeps more shots. TVD is "
        f"not pinned -- calibration targets selection rate alone -- so its trend stands."
    )


def plot(table: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.05), sharex=True)
    top, bottom = axes

    table = table.sort_values("k")
    x = table["k"].to_numpy()

    # Both panels read in factors of the reference code size, still drawn so **up is
    # worse**. Two things about the conversion:
    #
    # `_pct` here is a plain change -- `(value/base - 1) * 100` -- so `reduction=False`.
    # The reduction branch would read `+121` as "121% smaller" and clamp the volume series
    # at `MAX_FACTOR` instead of drawing 2.21x.
    #
    # A higher-is-better series is *inverted*, not negated. On a ratio axis the mirror of
    # 1.4x is 1/1.4 = 0.71x; a minus sign has nothing to mean, and the factor form is what
    # makes the two panels comparable -- +121% volume and -55% selection rate look like
    # different magnitudes in percent and are both about 2.2x.
    def factors(column: str, shift: int = 0, invert: bool = False) -> np.ndarray:
        pct = table[f"{column}_pct"].to_numpy()
        if shift:
            pct = pct + shift * table[f"{column}_sd"].to_numpy()
        values = np.array(
            [factor_from_pct(p, reduction=False, floor=1.0 / MAX_FACTOR) for p in pct]
        )
        return 1.0 / values if invert else values

    spans: Dict[str, List[float]] = {"top": [], "bottom": []}

    def draw(axis, key: str, column: str, label: str, invert: bool) -> None:
        centre = factors(column, invert=invert)
        # The band is converted at its endpoints rather than as centre +- something. A
        # symmetric +-1 SD in percent is not symmetric in factors, and taking the midpoint
        # for both would understate one side.
        edges = (factors(column, -1, invert), factors(column, +1, invert))
        low, high = np.minimum(*edges), np.maximum(*edges)
        line, = axis.plot(x, centre, label=label)
        axis.fill_between(x, low, high, color=line.get_color(), alpha=0.15, linewidth=0)
        spans[key].extend([float(np.nanmin(low)), float(np.nanmax(high))])

    for column, label in COST_SERIES:
        draw(top, "top", column, label, False)
    for column, label, negate in OUTCOME_SERIES:
        draw(bottom, "bottom", column, label, negate)

    for axis, key in ((top, "top"), (bottom, "bottom")):
        # 1x is the reference code size, which every series is measured against.
        # `apply_factor_axis` draws it with the style's reserved reference-line treatment.
        reach = spans[key] + [1.0]
        apply_factor_axis(axis, max(reach), bottom=min(reach))
        axis.legend(frameon=False, fontsize=8, loc="best")

    # Literal U+00D7 rather than mathtext, which the shared style's `svg.fonttype: none`
    # would turn back into glyph paths.
    top.set_ylabel("change from $k=2$ (×)")
    bottom.set_ylabel("change from $k=2$ (×)")
    bottom.set_xticks(list(x))
    # No mathtext in the parenthetical. The shared style sets `mathtext.fontset: cm` with
    # `svg.fonttype: none`, and matplotlib then writes one `<tspan>` per glyph whose
    # character is the glyph's *position in the Computer Modern font*. Only letters and
    # digits sit at their own codepoints. This label was `$\lceil n/k \rceil$` and rendered
    # as `den=k` -- `\lceil` is `d` in cmsy10, `\rceil` is `e`, and `/` is `=` in cmmi10.
    # Rewriting it as `$= n/k$` did not help: the slash is ASCII but still lands on `=`,
    # so it read `n=k`, which is a plausible-looking wrong statement rather than obvious
    # garbage. The rule is not "avoid symbols outside Latin-1" -- it is that anything but a
    # letter or a digit is suspect.
    #
    # A literal U+2308 is not available either: Times New Roman has no ceiling glyph, where
    # it does have U+00D7 and U+03B1 (which is why the factor axes and `4_single_patch`'s
    # alpha can use literals). So: words.
    #
    # `$k$` alone is verified safe and kept, so the axis variable stays italic and matches
    # the prose.
    bottom.set_xlabel("logical qubits per patch, $k$   (patches = n/k, rounded up)")

    for axis, tag in zip(axes, "ab"):
        axis.annotate(
            f"({tag})", xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-32, 2), textcoords="offset points",
            fontsize=10, fontweight="bold", va="bottom",
        )

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--from-csv", type=Path, default=defaults.source,
        help="The density sweep's CSV. This figure runs no simulation of its own.",
    )
    parser.add_argument("--reference-k", type=int, default=defaults.reference_k)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    return Settings(
        source=args.from_csv,
        reference_k=args.reference_k,
        output_override=args.output_dir,
    )


def main() -> None:
    settings = parse_args()
    if not settings.source.exists():
        raise FileNotFoundError(
            f"{settings.source} not found. Run 6_packing_density_tradeoff.py first."
        )
    frame = pd.read_csv(settings.source)
    scaled = relative_to_reference(full_density(frame), settings.reference_k)
    if scaled.empty:
        raise RuntimeError(f"no circuit carries k = {settings.reference_k}")
    table = curves(scaled)
    print_summary(table, settings)
    plot(table, settings)


if __name__ == "__main__":
    main()
