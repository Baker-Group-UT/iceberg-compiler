"""
4_single_patch_detection_vs_noise.py -- when a patch stops catching what goes wrong.

**Claim.** A `[[k+2, k, 2]]` patch puts `k` logical qubits behind a single pair of
detectors. It catches any one fault, but the share of faults it cannot distinguish grows
with `k`, so its ability to detect errors degrades as the code gets larger -- and for every
`k` there is a physical error rate past which the patch is no longer worth the qubits it
costs. This is the figure that makes choosing `k` a real decision *before* AQORN is
mentioned, so it belongs in the problem statement rather than the results.

x = noise scale (`ScaledNoiseModel`'s alpha: `p1 = 10^-3 a`, `p2 = 5 x 10^-3 a`,
`T1, T2 ∝ 1/a`, so alpha = 1 is the Table 2 machine and the axis is "how much worse than
that"). Two stacked panels: (a) logical selection rate, (b) TVD, one line per `k`, with the
unencoded circuit as the reference the code has to beat.

**One patch, and no idle or movement noise.** Nothing about placement or packing is in this
picture -- there is only one patch, so there is nowhere to move between and no second patch
to idle against. `distance_noise` and `idle_noise` are both off. What is left is the code
and the faults it does or does not catch, which is the only thing this figure is about.

**Program width is `k`, because that is what one patch holds.** A single `[[k+2, k, 2]]`
patch carries exactly `k` logical qubits, so the k = 2 line runs 2-qubit programs and the
k = 10 line runs 10-qubit programs. That is not a confound to be removed, it is what the
code size *means*; a figure that held width fixed would not be measuring one patch.

**Total gate count is held fixed across `k`, and that is the control that matters.** The
obvious alternative -- fixed gates per qubit -- makes the program grow with `k`, so a larger
code would face proportionally more faults and the figure would conflate "this code detects
less" with "this circuit has more to detect". Holding the gate budget fixed gives every
code size the same number of fault opportunities, so the difference between the lines is
the code's detection power and not the circuit's size. It costs an asymmetry in depth
(60 gates on 2 qubits is deep, on 10 qubits is shallow), which is harmless precisely
because idle noise is off.

**The unencoded reference is measured, not assumed.** Each program is also run with no
encoding at the same alpha, and its TVD against the ideal distribution is the line the
patch has to stay below. Where a `k` line crosses it is the honest statement of when that
code size stops earning its qubits. The reference is a *band* rather than a line because it
depends on program width, which varies with `k` by construction: the band spans the widths
swept, and `print_summary` reports the per-`k` value and the crossing point for each.

Selection rate has no unencoded counterpart -- an unencoded circuit discards nothing, so its
selection rate is 1 by definition -- which is exactly why panel (a) is the cost side and
panel (b) is the benefit side.

Bands on the `k` lines are +-1 standard deviation over the random circuits at that point.

Run with:

    PYTHONPATH=. python experiments/figures/4_single_patch_detection_vs_noise/4_single_patch_detection_vs_noise.py
    PYTHONPATH=. python .../4_single_patch_detection_vs_noise.py --from-csv <path>
    PYTHONPATH=. python .../4_single_patch_detection_vs_noise.py --circuits 2 --alphas 0.1 1 10
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from analysis.metrics import total_variation_distance
from experiments.shared.figures import COLUMN_WIDTH, data_dir, save_figure, use_style
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
from experiments.shared.simulation import (
    SimSettings,
    add_sim_arguments,
    build_config,
    ensure_measurements,
    evaluate,
    naive_packing,
    settings_from_args,
)
from experiments.shared.figures import stamp_provenance
from simulator.config_presets import ScaledSimulator
from simulator.simulator import Simulator

NAME = "4_single_patch_detection_vs_noise"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: Log-spaced from "better than the Table 2 machine" to "far worse". Measured range: at
#: alpha = 0.05 every code sits near its sampling floor and at alpha = 20 the widest
#: programs have lost the distribution entirely, so the interesting behaviour -- the
#: crossings -- happens inside this span rather than at its edges.
DEFAULT_ALPHAS: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)

#: Matches every other k figure in the suite. Odd k is not expressible -- `n_code = k + 2`
#: must be even, and the pipeline rejects odd code sizes outright.
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: Program gates, held fixed across k. See the module docstring on why this rather than
#: gates per qubit.
DEFAULT_TOTAL_GATES = 60

#: Fixed raw shots. Not calibrated per circuit: the x axis *is* the noise scale here, so
#: solving alpha per point would delete the axis.
DEFAULT_REPETITIONS = 60_000


@dataclass
class Settings:
    alphas: Sequence[float] = DEFAULT_ALPHAS
    k_values: Sequence[int] = DEFAULT_K_VALUES
    total_gates: int = DEFAULT_TOTAL_GATES
    locality: float = 0.85
    immutable_ratio: float = 0.0
    circuits: int = 12
    circuit_seed: int = 20260730
    repetitions: int = DEFAULT_REPETITIONS
    sim: SimSettings = field(default_factory=SimSettings)
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False
    jobs: Optional[int] = None


def _circuit_for(settings: Settings, k: int, index: int):
    """
    A program of `total_gates` gates on `k` qubits -- one patch's worth of logical qubits.

    `structured_program_circuit` takes gates *per qubit*, so the fixed total is converted
    here. That conversion is the whole point of the control; see the module docstring.
    """
    seed = settings.circuit_seed + index
    per_qubit = max(1, round(settings.total_gates / k))
    blocks = plant_blocks(k, min(4, k), seed=seed)
    circuit = structured_program_circuit(
        k, per_qubit, seed=seed, blocks=blocks,
        locality=settings.locality,
        immutable_ratio=settings.immutable_ratio,
        clifford_only=True,
    )
    return seed, ensure_measurements(circuit)


def _run_point(job) -> List[Dict]:
    """
    One (k, alpha, circuit): the encoded patch and the unencoded reference, same noise.

    Module-level and self-contained because `parallel_map` spawns rather than forks.
    """
    settings, k, alpha, index = job
    seed, circuit = _circuit_for(settings, k, index)

    # One patch: naive_packing(k, k) puts all k qubits in patch 0.
    placement = naive_packing(k, k)
    config = build_config(
        settings.sim, n_code=k + 2, alpha=alpha, repetitions=settings.repetitions
    )
    measured = evaluate(circuit, placement, config)
    if measured is None:
        return []

    # The same program, same alpha, no encoding. `Simulator.run` returns both the noisy
    # and the ideal distribution, so the reference TVD comes from one call.
    reference = ScaledSimulator(alpha=alpha, repetitions=settings.repetitions)
    if settings.sim.stim:
        reference = reference.with_stim()
    result = Simulator(reference).run(circuit)
    unencoded = total_variation_distance(result.noisy_dist, result.ideal_dist)

    return [{
        "seed": seed,
        "circuit_index": index,
        "k": k,
        "n_code": k + 2,
        "num_qubits": k,
        "patches": len({patch for patch, _ in placement}),
        "program_gates": circuit.size(),
        "alpha": alpha,
        "repetitions": settings.repetitions,
        "lsr": measured["lsr"],
        "tvd": measured["tvd"],
        "unencoded_tvd": unencoded,
        "accepted_shots": settings.repetitions * measured["lsr"],
        "one_qubit_gates": measured["one_qubit_gates"],
        "two_qubit_gates": measured["two_qubit_gates"],
        "circuit_depth": measured["circuit_depth"],
    }]


def compute(settings: Settings) -> pd.DataFrame:
    # Idle and movement noise are meaningless with one patch, and leaving them on would
    # charge this figure for mechanisms it is explicitly excluding.
    settings = replace(
        settings,
        sim=replace(settings.sim, idle_noise=False, distance_noise=False),
    )
    points = [
        (settings, k, alpha, index)
        for k in settings.k_values
        for alpha in settings.alphas
        for index in range(settings.circuits)
    ]
    print(
        f"[{NAME}] {len(settings.k_values)} code sizes x {len(settings.alphas)} noise "
        f"scales x {settings.circuits} circuits, {settings.total_gates} gates each, "
        f"{settings.repetitions:,} shots/point"
    )
    produced = parallel_map(_run_point, points, jobs=settings.jobs, label="k x alpha")
    rows = [row for batch in produced if batch for row in batch]
    return pd.DataFrame(rows)


def curves(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Mean LSR, TVD and unencoded TVD per (k, alpha), with the spread over circuits."""
    rows: List[Dict] = []
    for k in settings.k_values:
        for alpha in sorted(frame["alpha"].unique()):
            block = frame[(frame["k"] == k) & (frame["alpha"] == alpha)]
            if block.empty:
                continue
            rows.append({
                "k": k,
                "alpha": alpha,
                "lsr": float(block["lsr"].mean()),
                "lsr_sd": float(block["lsr"].std(ddof=1)) if len(block) > 1 else 0.0,
                "tvd": float(block["tvd"].mean()),
                "tvd_sd": float(block["tvd"].std(ddof=1)) if len(block) > 1 else 0.0,
                "unencoded_tvd": float(block["unencoded_tvd"].mean()),
                "accepted_shots": float(block["accepted_shots"].mean()),
                "circuits": int(len(block)),
            })
    return pd.DataFrame(rows)


def crossing(block: pd.DataFrame) -> Optional[float]:
    """
    The alpha at which this code's TVD first exceeds the unencoded circuit's.

    Interpolated in log-alpha between the bracketing samples rather than snapped to a
    swept point, so the number does not depend on how finely the axis happens to be
    sampled. Returns None when the code never loses over the swept range.
    """
    block = block.sort_values("alpha")
    better = block["tvd"].to_numpy() <= block["unencoded_tvd"].to_numpy()
    alphas = block["alpha"].to_numpy()
    for index in range(1, len(block)):
        if better[index - 1] and not better[index]:
            gap_low = block["unencoded_tvd"].iloc[index - 1] - block["tvd"].iloc[index - 1]
            gap_high = block["unencoded_tvd"].iloc[index] - block["tvd"].iloc[index]
            span = gap_low - gap_high
            weight = gap_low / span if span else 0.0
            low, high = np.log(alphas[index - 1]), np.log(alphas[index])
            return float(np.exp(low + weight * (high - low)))
    return None


def print_summary(table: pd.DataFrame, settings: Settings) -> None:
    alphas = sorted(table["alpha"].unique())

    for column, title in (("lsr", "selection rate"), ("tvd", "TVD")):
        print(f"\n[{NAME}] {title} by noise scale and k")
        header = "  alpha".ljust(10) + "".join(f"{'k=' + str(k):>9}" for k in settings.k_values)
        if column == "tvd":
            header += f"{'unenc.':>10}"
        print(header)
        for alpha in alphas:
            cells = []
            for k in settings.k_values:
                row = table[(table["k"] == k) & (table["alpha"] == alpha)]
                cells.append(f"{'-':>9}" if row.empty else f"{row.iloc[0][column]:>9.4f}")
            line = f"  {alpha:<8g}" + "".join(cells)
            if column == "tvd":
                reference = table[table["alpha"] == alpha]["unencoded_tvd"].mean()
                line += f"{reference:>10.4f}"
            print(line)

    print(f"\n[{NAME}] where each code stops beating the unencoded circuit")
    for k in settings.k_values:
        block = table[table["k"] == k]
        if block.empty:
            continue
        point = crossing(block)
        if point is None:
            wins = (block["tvd"] <= block["unencoded_tvd"]).all()
            verdict = (
                "never loses over the swept range" if wins
                else "never beats the unencoded circuit at any swept alpha"
            )
            print(f"  k={k:<3} {verdict}")
        else:
            print(f"  k={k:<3} crosses at alpha = {point:.2f}")

    print(f"\n[{NAME}] selection rate cost at alpha = 1 (the Table 2 machine)")
    at_one = table[np.isclose(table["alpha"], 1.0)]
    for _, row in at_one.iterrows():
        print(
            f"  k={int(row['k']):<3} LSR {row['lsr']:.4f}  TVD {row['tvd']:.4f}  "
            f"(unencoded {row['unencoded_tvd']:.4f})"
        )


def _plain_decimal(value: float, _position=None) -> str:
    """`0.001`, `0.05`, `1`, `20` -- never `10^{-3}`, which the style guide rules out."""
    if value <= 0:
        return ""
    if value >= 1:
        return f"{value:g}"
    text = f"{value:.6f}".rstrip("0")
    return text.rstrip(".") if text.endswith(".") else text


def plot(table: pd.DataFrame, settings: Settings) -> List[Path]:
    """
    One panel, two y axes: selection rate on the left, TVD on the right.

    Was two stacked panels sharing x. Both metrics answer the same question -- what a code
    size buys as noise rises -- and the reading that matters is where a rising selection
    rate stops paying for itself in TVD, which is a comparison *within* an alpha. Stacked
    panels put that comparison a vertical eye-movement apart; twinned axes put it on one
    line.

    The cost is that a twin axis has to encode two variables at once. Colour carries the
    code size and line style carries the metric -- solid with markers for selection rate,
    dashed without for TVD -- rather than colour carrying the metric, because the k
    comparison is the one the figure exists to make and it must survive greyscale. The
    style's `prop_cycle` couples colour, marker and dash together, so the colours are taken
    from it by hand here and the other two set per metric.
    """
    use_style()
    # Height chosen so the *axes* stay about as tall as one of the two panels this
    # replaced, rather than as tall as both. Merging the panels is meant to put the two
    # metrics on one line, not to buy the figure more page.
    fig, left = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.62))
    right = left.twinx()

    # Not the style's `prop_cycle`, which runs blue, dark slate, tan, mid grey, light grey
    # -- a sequence with no ordering a reader can use, so k = 6 landing between k = 4 and
    # k = 8 was invisible. Paired instead: the smallest code keeps the house blue and stands
    # alone, then two tans and two greys, light before dark inside each pair. Code size then
    # reads as "which family, then which of the two" rather than as five arbitrary hues.
    #
    # `#0097fd`, `#cdc2c0` and `#636363` are the style's own; the other two are darkened and
    # lightened from them rather than newly chosen, so the figure stays in the paper's
    # palette. Luminance falls monotonically inside each pair, which is what keeps the
    # pairing legible in greyscale.
    palette = ["#0097fd", "#cdc2c0", "#8f7f7a", "#9e9e9e", "#4a4a4a"]
    markers = ["o", "s", "^", "D", "v"]

    for index, k in enumerate(settings.k_values):
        block = table[table["k"] == k].sort_values("alpha")
        if block.empty:
            continue
        x = block["alpha"].to_numpy()
        colour = palette[index % len(palette)]
        for axis, column, style in (
            (left, "lsr", dict(linestyle="-", marker=markers[index % len(markers)])),
            (right, "tvd", dict(linestyle="--", marker="")),
        ):
            y = block[column].to_numpy()
            spread = block[f"{column}_sd"].to_numpy()
            axis.plot(
                x, y, color=colour, label=f"$k={k}$" if axis is left else None,
                linewidth=1.0, markersize=3, **style,
            )
            # The TVD axis is log, which excludes zero, so its lower edge is clipped to a
            # small positive value rather than allowed to go negative and silently drop
            # the whole band.
            lower = np.clip(y - spread, 1e-4, None) if axis is right else y - spread
            axis.fill_between(
                x, lower, y + spread, color=colour, alpha=0.12, linewidth=0,
            )

    # The unencoded reference is not drawn. It is a *band* rather than a line -- it depends
    # on program width, which varies with k by construction -- so on a panel that already
    # carries ten series plus their spread bands it read as one more grey smear rather than
    # as a threshold. `print_summary` still reports the per-k unencoded value and the alpha
    # at which each code size crosses it, which is the number the claim actually rests on,
    # and `unencoded_tvd` stays in the CSV.

    left.set_xscale("log")
    left.set_ylabel("logical selection rate  (solid)")
    left.set_ylim(0.0, 1.0)
    right.set_ylabel("total variation distance  (dashed)")
    # Log y, the style guide's stated exception to the zero baseline. TVD spans three
    # orders of magnitude here (0.003 to 0.97), and on a linear axis the entire
    # low-noise half -- where the code sizes differ by a factor of ten and the figure's
    # claim lives -- collapses onto zero. Log excludes zero rather than contradicting
    # the rule, and equal factors read as equal distances, which is the right geometry
    # for a quantity that compounds.
    right.set_yscale("log")
    # Plain decimals on both log axes. Matplotlib's default `LogFormatterSciNotation`
    # writes `10^{-1}`, which the style guide rules out -- and it does it as mathtext,
    # which under `mathtext.fontset: cm` + `svg.fonttype: none` would emit Computer Modern
    # font positions rather than Unicode and corrupt the exponent. Two problems, one fix.
    for axis in (left.xaxis, right.yaxis):
        axis.set_major_formatter(FuncFormatter(_plain_decimal))
        axis.set_minor_formatter(plt.NullFormatter())
    # The twin draws its own frame on top of the host's; without this the right spine and
    # the host's hidden top spine both show and the panel gains a box.
    right.grid(False)
    for spine in ("top",):
        left.spines[spine].set_visible(False)
        right.spines[spine].set_visible(False)
    right.spines["right"].set_visible(True)
    # Literal U+03B1, not `$\alpha$`. Mathtext under `mathtext.fontset: cm` plus
    # `svg.fonttype: none` emits Computer Modern *font positions* rather than Unicode, and
    # alpha sits outside Latin-1, so it extracted as an unrelated letter and rendered as one
    # in any SVG viewer without the CM fonts. Times New Roman has U+03B1, so the literal is
    # correct everywhere. See 9_code_size_tradeoff for the case where no literal exists.
    left.set_xlabel("noise scale α   (α=1 is the Table 2 machine)")

    # One legend, above the axes. There is no free corner inside the panel -- selection
    # rate fills the upper left going down and TVD fills the lower left going up -- so any
    # in-panel placement sits on data.
    #
    # The code sizes are all it carries: a separate solid/dashed key would be redundant
    # with the axis labels, which already name which style belongs to which metric, and
    # those labels are unavoidable where a legend entry is not.
    left.legend(
        frameon=False, fontsize=7, ncol=len(settings.k_values),
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.0, handlelength=1.6, handletextpad=0.4, borderpad=0.0,
    )

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(defaults.alphas))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values))
    parser.add_argument("--total-gates", type=int, default=defaults.total_gates,
                        help="Program gates, held fixed across k so every code faces the "
                             "same number of fault opportunities.")
    parser.add_argument("--locality", type=float, default=defaults.locality)
    parser.add_argument("--circuits", type=int, default=defaults.circuits)
    parser.add_argument("--circuit-seed", type=int, default=defaults.circuit_seed)
    parser.add_argument("--repetitions", type=int, default=defaults.repetitions,
                        help="Fixed raw shots. Not calibrated per circuit: the x axis is "
                             "the noise scale, so solving alpha per point would delete it.")
    add_jobs_argument(parser)
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--standard", action="store_true",
                        help="the standard translation settings (`SimSettings.with_standard`: "
                             "layer-sorted, transversal in-patch CZ / XCX at k = 2)")
    args = parser.parse_args()
    sim = settings_from_args(args, defaults.sim)
    if args.standard:
        sim = sim.with_standard()
    return Settings(
        alphas=args.alphas,
        k_values=args.k_values,
        total_gates=args.total_gates,
        locality=args.locality,
        circuits=args.circuits,
        circuit_seed=args.circuit_seed,
        repetitions=args.repetitions,
        sim=sim,
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
    else:
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
            kept = data_dir(SCRIPT) / f"{NAME}.csv"
            frame.to_csv(kept, index=False)
            print(f"[{NAME}] persisted to {kept}")

    table = curves(frame, settings)
    if table.empty:
        raise RuntimeError("no points survived")
    print_summary(table, settings)
    plot(table, settings)


if __name__ == "__main__":
    main()
