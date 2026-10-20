"""
depth_vs_gates_tradeoff.py -- what packing spends to buy two-qubit gates.

**Claim.** Packing does not improve depth and gate count together; it trades one for the
other, and which way it trades is set by the noise model it is given. On the ising family
at k=2 the default objective removes 240 two-qubit gates and pays 5.4x the depth for them
(`ising_n98`: 2q 1655 -> 1415, depth 43 -> 232). That is the right call when movement
between patches is what breaks the run and the wrong one when idling is, so the figure
asks whether the planner actually responds to being told which regime it is in.

**Layout.** A butterfly: one shared vertical axis, the depth the placement costs growing
leftwards and the two-qubit gates it saves growing rightwards. Both against the naive
packing, so the figure reads as "what packing spends" against "what it buys" at each noise
balance.

Both wings are reductions against naive, so both grow outwards when packing wins, and a
loss crosses the centre line inwards in red. Nothing is ever clipped or capped: bar length
is the value, and a wing hidden at zero would delete the finding the figure exists for.
Each wing scales to its own data -- they are percentages of different resources, so a
shared axis is not the meaningful comparison it appears to be, and a single wide spread
would otherwise pay for the whole figure.

**The vertical axis is the idle/movement noise ratio**, swept with the geometric mean of the
two *contributions* held constant (`SimSettings.with_noise_ratio`). Moving one scalar and
pinning the other would confound the balance between the two mechanisms with how much total
noise there is, and a crossover here has to mean "which mechanism dominates" or it means
nothing. Low is movement-dominated, where trading depth for gates should pay; high is
idle-dominated, where it should not.

The helper used to equalise the two *scalars* at ratio 1, which is not the balance point:
a unit of `idle_time_scalar` buys about 30x the error a unit of `distance_scalar` does, so
the old ratio 1 sat well inside the idle-dominated regime and the old sweep's
movement-dominated end was nothing of the kind. `MOVEMENT_BALANCE` now carries the measured
exchange rate. This figure's numbers were regenerated on the corrected axis; anything quoted
from before it is not comparable.

**Circuit metrics only, no simulation.** The quantity in question is what the *planner*
chooses, and that is fully determined by the translated circuit -- one plan and two
translations per cell, against a simulation per cell for LSR or TVD. The corresponding
outcome figure is `lsr_tvd_vs_k`; this one is the mechanism, and keeping it cheap is what
lets the ratio axis have enough points to show a crossover if there is one.

Improvements are paired within a circuit: the packed and naive arms are translated under
the *same* commutation setting, the one the plan chose, because a placement priced
against the commuted view and translated without the pass describes no circuit.

Run with:

    python -m experiments.figures.depth_vs_gates_tradeoff.depth_vs_gates_tradeoff
    python -m ... --circuits ising_n26 ising_n42 ising_n66     # the family that shows it
    python -m ... --n-code 6 --refresh
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import qiskit
from tqdm import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.figures import (
    COLUMN_WIDTH,
    data_dir,
    save_figure,
    use_style,
)
from experiments.shared.simulation import (
    SimSettings,
    measured_regime,
    noise_profile_for,
)
from iceberg.utils import logical_capacity
from packing.plan import naive_packing, plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

SCRIPT = Path(__file__).resolve()

#: Idle-to-movement *contribution* ratios, log-spaced in powers of four and symmetric about
#: 1.0 so the movement-dominated and idle-dominated halves get equal resolution -- the
#: question is where the trade flips, and an asymmetric sweep would bias where it looks like
#: it flipped.
#:
#: Widened from 1/16..16 when `SimSettings.with_noise_ratio` was corrected. That helper used
#: to equalise the two noise *scalars* at ratio 1, and the scalars buy error at rates that
#: differ by about 30x (measured: 0.41 per unit idle against 0.017 per unit distance), so the
#: old sweep never left the idle-dominated half and its "movement-dominated" end was nothing
#: of the kind. `MOVEMENT_BALANCE` now fixes the exchange rate; these ratios span 300:1
#: movement to 96:1 idling, which is what the axis label has always claimed.
#:
#: Numbers quoted from runs before that change are on the old axis and were re-measured after
#: it. The corrected sweep makes the trade cleaner rather than weaker: depth reduction now
#: rises monotonically +3.86% -> +7.52% while 2q reduction falls +4.99% -> +3.61%, which is
#: the response to the noise regime this figure exists to demonstrate. The four
#: movement-dominated columns agree to two decimals, so the planner saturates once movement
#: is priced high enough -- worth knowing, and invisible on the old narrow axis.
RATIOS: Tuple[float, ...] = (
    1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0, 256.0,
)

#: Left wing, right wing. Depth is the resource packing spends and two-qubit gates are
#: what it buys, so depth goes on the left where a reader starts.
#: `(column, axis label, side, palette index, sign)`.
#:
#: The sign is what keeps both wings growing outwards. Depth is stored as a *reduction*,
#: like every other metric in this suite, but on the circuits that motivate the figure it
#: is reliably negative -- packing buys two-qubit gates by spending depth -- so the wing
#: plots `-depth_reduction` and is labelled as the cost it is. Without the flip the bars
#: point inwards across the centre line and the butterfly reads inside out.
#:
#: Slate for the price, blue for the gain, so colour tracks good-versus-bad rather than
#: which panel a bar is in.
#: Both wings are reductions -- gains -- growing outwards, and a loss crosses the centre
#: line inwards in red.
#:
#: An earlier version plotted the left wing negated, as "depth spent", because on the ising
#: family depth reduction is reliably negative and the bars otherwise point inwards. That
#: was calibrated on four circuits and is wrong for the suite: across 75 QASMBench proxies
#: depth reduction is *positive at every ratio* (+3.9% movement-dominated rising to +7.5%
#: idle-dominated), so negating it would send the whole wing inwards instead. Reductions it
#: is, and ising reads as the inward excursion it actually is.
WINGS: Sequence[Tuple[str, str, str, int, float]] = (
    ("depth_reduction", "Depth reduction (%)", "left", 1, 1.0),      # #3b4c56 dark slate
    ("two_qubit_reduction", "2q gate reduction (%)", "right", 0, 1.0),  # #0097fd blue
)

#: Losses. The deep end of `figures.HEATMAP_CMAP`'s negative arm, so "worse than naive"
#: reads the same here as in the improvement heatmaps.
NEGATIVE = "#7d1a30"

#: Cross-patch two-qubit gates removed, as a percent of naive's. Recorded because it is the
#: term `PackingCost.price` multiplies by `movement_cost`, so it is the resource the
#: movement-dominated end of the sweep is supposed to be about -- and it is not plotted,
#: because measured this way the answer is not the one that story predicts. It comes out
#: *negative* on this family wherever packing does anything at all (-75% at ratio 1,
#: -111% at 16): the placement accepts more traffic to buy gate savings.
#:
#: Treat it as a diagnostic rather than as the objective's own tally. `cost.py` counts the
#: same events on the *translated view*, after `cancel_inverse_gates` and possibly
#: `commute_hadamards`, where this counts them on the raw program; the two need not agree
#: and the discrepancy has not been chased down. That is why it stays in the CSV and the
#: printed table and out of the figure.
CROSSINGS = "crossing_reduction"


def _metrics(
    program: qiskit.QuantumCircuit,
    placement: List[Tuple[int, int]],
    n_code: int,
    *,
    commute: bool,
) -> Optional[Tuple[int, int, int]]:
    """1q, 2q, depth of the translated circuit. `None` when the placement is rejected."""
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_merging = True
    config.translator_config.enable_h_commutation = commute
    try:
        circuit = translate_permutation(program, placement, config=config).circuit
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    return one_q, two_q, circuit.depth()


def _reduction(reference: float, challenger: float) -> float:
    return 0.0 if reference <= 0 else 100.0 * (reference - challenger) / reference


def _crossings(program: qiskit.QuantumCircuit, placement: List[Tuple[int, int]]) -> int:
    """
    Two-qubit gates in the *program* whose operands land in different patches.

    Counted on the program rather than the translation because that is what the objective
    charges movement for, and because a cross-patch gate becomes several physical gates --
    counting those would mix the traffic in with the gate-count wing that already plots it.
    """
    return sum(
        1
        for instruction in program.data
        if len(instruction.qubits) == 2
        and placement[program.find_bit(instruction.qubits[0]).index][0]
        != placement[program.find_bit(instruction.qubits[1]).index][0]
    )


def collect(
    circuits: Sequence[Tuple[str, qiskit.QuantumCircuit]],
    n_code: int,
    ratios: Sequence[float] = RATIOS,
) -> pd.DataFrame:
    """One row per (circuit, ratio): what packing gained on depth and on 2q gates."""
    k = logical_capacity(n_code)
    rows: List[Dict[str, float | str | bool]] = []

    for ratio in tqdm(list(ratios), desc="noise ratios"):
        settings = SimSettings().with_noise_ratio(ratio)
        profile = noise_profile_for(settings)
        for name, program in circuits:
            plan = plan_packing(program, n_code, noise=profile, verify=True)
            fallback = naive_packing(program.num_qubits, k)
            naive = _metrics(program, fallback, n_code, commute=plan.h_commutation)
            packed = _metrics(
                program, plan.placement, n_code, commute=plan.h_commutation,
            )
            if naive is None or packed is None:
                continue
            # The row's true position on the regime axis: what the naive baseline's
            # schedule actually charges at these scalars (alpha-invariant).
            try:
                regime = measured_regime(
                    program, fallback, n_code=n_code,
                    idle_time_scalar=settings.idle_time_scalar,
                    distance_scalar=settings.distance_scalar,
                    h_commutation=plan.h_commutation, merging=True,
                )
            except Exception:  # noqa: BLE001
                continue
            rows.append(
                {
                    "circuit": name,
                    "qubits": program.num_qubits,
                    "n_code": n_code,
                    "ratio": ratio,
                    "measured_ratio": regime["measured_ratio"],
                    "movement_share": regime["movement_share"],
                    "idle_scalar": settings.idle_time_scalar,
                    "distance_scalar": settings.distance_scalar,
                    "h_commutation": plan.h_commutation,
                    "fell_back": plan.fell_back,
                    "patches": plan.patches,
                    "depth_reduction": _reduction(naive[2], packed[2]),
                    "two_qubit_reduction": _reduction(naive[1], packed[1]),
                    "one_qubit_reduction": _reduction(naive[0], packed[0]),
                    CROSSINGS: _reduction(
                        _crossings(program, fallback),
                        _crossings(program, plan.placement),
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame, *, output_name: str) -> List[Path]:
    use_style()
    import matplotlib.pyplot as plt

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ratios = sorted(frame["ratio"].unique())
    positions = np.arange(len(ratios), dtype=float)

    fig, axes = plt.subplots(
        ncols=2, figsize=(COLUMN_WIDTH, 0.34 * len(ratios) + 1.25), sharey=True
    )
    outward: List[float] = []
    inward: List[float] = []
    zeros: List[List[bool]] = []
    # Mean traffic removed per ratio, for the annotation on flat rows. Absent from CSVs
    # written before this column existed, in which case flat rows just say "no change".
    crossing = (
        {r: float(v) for r, v in frame.groupby("ratio")[CROSSINGS].mean().items()}
        if CROSSINGS in frame.columns
        else {}
    )

    for axis, (column, label, side, colour_index, sign) in zip(axes, WINGS):
        grouped = frame.groupby("ratio")[column]
        means = sign * np.array([grouped.get_group(r).mean() for r in ratios])
        # Standard error, not the +-1 SD the house style asks for elsewhere. The bar is a
        # mean over circuits and the question is whether it differs from zero; the
        # circuit-to-circuit SD answers a different question and is enormous here -- ~45
        # points against means of 2-7 -- so plotting it sets the axis 10x too wide and
        # every bar becomes a sliver. The spread itself is not being hidden: it is in the
        # CSV per circuit, and `summarise` prints the means it belongs to.
        spreads = np.nan_to_num(
            np.array([
                grouped.get_group(r).std(ddof=1) / max(1.0, np.sqrt(grouped.get_group(r).count()))
                for r in ratios
            ])
        )

        # Bars are never clipped and never rescaled. An earlier version capped the inward
        # half at a fraction of the outward one, borrowing `heatmap_norm`'s
        # `negative_headroom` -- which is fine for a colour scale, where saturating means
        # "at least this much", and wrong for a bar, where length *is* the value: five
        # different depth costs between 10% and 31% all came out the same length.
        #
        # Gains grow outwards in the wing's colour; a loss crosses the centre line inwards
        # in red. Nothing is hidden at zero -- this figure exists to show that packing can
        # pay for gates with depth, so a wing clipped at zero would delete the finding.
        # That is the opposite of the ablation, where a negative segment means a pass the
        # compiler would simply have declined to run.
        axis.barh(
            positions,
            means,
            height=0.68,
            color=[palette[colour_index] if v >= 0 else NEGATIVE for v in means],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        axis.errorbar(
            means, positions, xerr=spreads, fmt="none",
            ecolor="#bdbdbd", elinewidth=0.7, capsize=1.6, zorder=1,
        )
        axis.set_xlabel(label)
        axis.grid(True, axis="x", linestyle="--", linewidth=0.4)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        outward.append(float(np.nanmax(means + spreads, initial=0.0)))
        inward.append(float(-np.nanmin(means - spreads, initial=0.0)))
        zeros.append([abs(value) < 0.005 for value in means])

    # A row is flat only when *both* wings are flat -- a wing that is zero while the other
    # moved is a genuine "this resource was untouched" and needs no explanation.
    flat_rows = [all(wing[row] for wing in zeros) for row in range(len(ratios))]

    # Each wing scaled to its own data. Sharing one scale is tempting -- it makes the
    # trade legible as an asymmetry -- but the wings are percentages of *different
    # resources*, so a shared axis is not the meaningful comparison it looks like, and a
    # single wide spread pays for the whole figure: on the ising family the depth wing
    # reaches +-49 at ratio 2, which on a shared scale squashes a 10% gate saving into a
    # sliver. Cost-versus-gain is carried by the sign convention and the labels instead.
    for index, axis in enumerate(axes):
        out = max(outward[index] * 1.10, 1.0)
        back = max(inward[index] * 1.10, 0.10 * out)
        # Mirrored, so the depth wing grows leftwards.
        axis.set_xlim(*((out, -back) if WINGS[index][2] == "left" else (-back, out)))
        axis.axvline(0, color="#3b4c56", linewidth=0.9, zorder=4)
        # Re-sign the mirrored labels: direction and colour carry the meaning, and a
        # reader should not have to work out that "-20" on the left wing is a 20% cost.
        axis.xaxis.set_major_formatter(lambda value, _: f"{abs(value):g}")
        # A row flat on both wings is a real result, not missing data, and on this family
        # it is the whole movement-dominated half. Mark it as what it is: the planner
        # shipped a placement metrically identical to naive. On `ising_n26` that is
        # provably all there is -- 200 shuffled restarts descended cannot beat naive at
        # ratio 1/16 or 1/4, because a chain paired in program order already captures half
        # its edges inside patches. Placed just off the centre line, which would hide it.
        if index == 0:
            for position, flat in zip(positions, flat_rows):
                if flat:
                    axis.annotate(
                        "= naive", xy=(0.0, position), xytext=(-4, 0),
                        textcoords="offset points", va="center", ha="right",
                        color="#636363", fontsize=plt.rcParams["xtick.labelsize"],
                    )

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([_measured_row_label(frame, r) for r in ratios])
    for axis in axes:
        axis.tick_params(axis="y", length=0)

    # Which end is which, in the axis label rather than as annotations in the panels. The
    # ratio is the whole independent variable and a reader should not have to reconstruct
    # its direction from the caption.
    fig.supylabel("Measured idle / movement noise ratio\n(low: movement-dominated)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    return save_figure(fig, output_name, SCRIPT)


def _ratio_label(ratio: float) -> str:
    """`1/8` rather than `0.125`, so the symmetry about 1 is visible in the labels."""
    if ratio >= 1:
        return f"{ratio:g}"
    return f"1/{round(1 / ratio):g}"


def _measured_row_label(frame: pd.DataFrame, ratio: float) -> str:
    """
    Geometric mean of the measured regime in one nominal row.

    The nominal ratio orders the rows (it is the sweep); the label is the ratio the
    naive baselines actually experienced there. Falls back to the nominal label on a
    CSV that predates the measured column.
    """
    if "measured_ratio" not in frame.columns:
        return _ratio_label(ratio)
    values = frame[frame["ratio"] == ratio]["measured_ratio"].to_numpy()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return _ratio_label(ratio)
    return f"{float(np.exp(np.log(values).mean())):.2g}"


def summarise(frame: pd.DataFrame) -> None:
    print(f"\n[depth_vs_gates_tradeoff] {frame['circuit'].nunique()} circuits, "
          f"{len(frame)} cells")
    print(f"  {'ratio':>7} {'depth %':>9} {'2q %':>9} {'1q %':>9} {'traffic %':>10} "
          f"{'fellback':>9} {'commute on':>11}")
    for ratio in sorted(frame["ratio"].unique()):
        cell = frame[frame["ratio"] == ratio]
        traffic = cell[CROSSINGS].mean() if CROSSINGS in cell.columns else float("nan")
        print(
            f"  {_ratio_label(ratio):>7} {cell['depth_reduction'].mean():>9.2f} "
            f"{cell['two_qubit_reduction'].mean():>9.2f} "
            f"{cell['one_qubit_reduction'].mean():>9.2f} "
            f"{traffic:>10.2f} "
            f"{cell['fell_back'].sum():>4}/{len(cell):<4} "
            f"{cell['h_commutation'].sum():>6}/{len(cell):<4}"
        )
    low, high = frame["ratio"].min(), frame["ratio"].max()
    for column, label in (("depth_reduction", "depth"), ("two_qubit_reduction", "2q")):
        movement = frame.loc[frame["ratio"] == low, column].mean()
        idle = frame.loc[frame["ratio"] == high, column].mean()
        print(f"  {label:>5}: movement-dominated {movement:+.2f}% -> "
              f"idle-dominated {idle:+.2f}%   (shift {idle - movement:+.2f} pp)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--n-code", type=int, default=4, help="Code size; k = n_code - 2.")
    parser.add_argument("--qasm-root", type=Path, default=Path("benchmarks/qasmbench"))
    parser.add_argument("--max-gates", type=int, default=2000)
    parser.add_argument("--max-qubits", type=int, default=100)
    parser.add_argument("--circuits", nargs="*", default=None,
                        help="Restrict to these QASMBench names.")
    parser.add_argument("--ratios", type=float, nargs="*", default=None,
                        help="Idle/distance ratios to sweep. Defaults to RATIOS.")
    parser.add_argument("--refresh", action="store_true",
                        help="Recompute rather than reading the cached CSV.")
    parser.add_argument("--persist", action="store_true",
                        help="Also write the CSV to the paper's results/data.")
    args = parser.parse_args()

    name = f"depth_vs_gates_tradeoff_n{args.n_code}"
    cache = SCRIPT.parent / f"{name}.csv"

    if cache.exists() and not args.refresh:
        frame = pd.read_csv(cache)
        print(f"[depth_vs_gates_tradeoff] reusing {cache} (--refresh to recompute)")
    else:
        circuits = load_proxy_circuits(
            args.qasm_root, args.max_gates, args.max_qubits, 1.0
        )
        if args.circuits:
            wanted = set(args.circuits)
            circuits = [(n, c) for n, c in circuits if n in wanted]
        if not circuits:
            raise SystemExit("no circuits matched")
        frame = collect(circuits, args.n_code, args.ratios or RATIOS)
        if frame.empty:
            raise SystemExit("every placement was rejected; nothing to plot")
        frame.to_csv(cache, index=False)
        if args.persist:
            frame.to_csv(data_dir(SCRIPT) / f"{name}.csv", index=False)

    summarise(frame)
    for path in plot(frame, output_name=name):
        print(f"[depth_vs_gates_tradeoff] wrote {path}")


if __name__ == "__main__":
    main()
