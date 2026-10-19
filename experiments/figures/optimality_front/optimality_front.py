"""
optimality_front.py -- how close AQORN's placement is to the best placement, on its own cost model
and on what the paper measures, for every program small enough to enumerate (4-6 qubits, [[4,2,2]],
tightest packing).

(a) One 6-qubit program: every placement of the complete tightest space (partition x patch order
    x lane map, both H-commutation settings) as LSR against logical error rate, the Pareto front of
    that space, and the placements AQORN, naive packing and the interaction-count partition pick.
    The example is the 6-qubit program whose AQORN logical-error gap is the median of its size,
    so it is representative rather than chosen.
(b) Every program: the gap between each packer's placement and the best placement on each metric,
    in percent of the best -- AQORN's own cost (the search gap), the measured LSR, logical error
    rate (LER) and TVD, and the two-qubit gate count and depth. Dot: median over programs;
    bar: interquartile range. Not +-1 SD (docs/figure-style.md): the gaps are right-skewed -- a
    few programs sit far out -- so mean +- SD runs below zero for nearly every bar and the mean
    describes no typical program. Say so in the caption. `--spread sd` draws mean +- 1 SD
    clipped at 0.

DATA. Until the measurement exists the figure is drawn from a toy model (`synthesize`, every
figure stamped SYNTHETIC). The experiment that replaces it writes one row per placement:

    program, num_qubits, placement, commute,
    cost                AQORN's own objective for this placement (standard planner, lower better)
    lsr                 logical selection rate, measured (higher better)
    ler                 logical error rate = silent logical error per kept shot, occupied lanes
                        only, measured (lower better)
    tvd                 TVD of the kept distribution, measured (lower better; 0 for programs
                        whose ideal output is uniform, which TVD cannot see)
    two_qubit_gates     physical two-qubit gates after translation (lower better)
    depth               physical depth after translation, every program layer-sorted
    pick                comma-separated arms that chose this placement ("aqorn", "naive",
                        "interaction"), empty otherwise
    source              "synthetic" or "measured"

"Best" on a measured metric must be decided on shots independent of the ones the gap is read
from (winner's curse; see reports/optimality.md section 2). The measurement
(`experiments/optimality/exhaustive_front.py`) writes those as `lsr_sel`, `ler_sel`, `tvd_sel`:
the best placement is chosen on them and every gap is read on `lsr`, `ler`, `tvd`, so a gap
can come out slightly negative. A placement counts as beating a pick on both LSR and LER only
if it does so on both sets of shots.

    PYTHONPATH=. venv/bin/python -m experiments.figures.optimality_front.optimality_front \\
        --from-csv results/optimality_front/space.csv

    PYTHONPATH=. venv/bin/python -m experiments.figures.optimality_front.optimality_front
    PYTHONPATH=. venv/bin/python -m experiments.figures.optimality_front.optimality_front \\
        --from-csv experiments/figures/optimality_front/optimality_front_space.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.shared.figures import COLUMN_WIDTH, data_dir, save_figure, use_style

NAME = "optimality_front"
HERE = Path(__file__).resolve().parent
SPACE_CSV = HERE / f"{NAME}_space.csv"
GAPS_CSV = HERE / f"{NAME}_gaps.csv"
SEED = 20261009

#: Arms in palette order: AQORN keeps series 1 (blue / circle) as in every other figure.
ARMS = [("aqorn", "AQORN"), ("naive", "Naive"), ("interaction", "Interaction partition")]

#: (column, row label, higher is better). Order is the row order of panel (b), top to bottom:
#: the model AQORN optimises, then what is measured, then resources.
METRICS = [
    ("cost", "AH*", False),          # AQORN's own objective
    ("lsr", "LSR", True),
    ("ler", "LER", False),
    ("tvd", "TVD", False),
    ("two_qubit_gates", "#2q", False),
    ("depth", "Depth", False),
]
#: Panel (b) draws a separator under these rows (model | measured | resources).
SEPARATE_AFTER = ("cost", "tvd")

#: Program counts per width, as in the exhaustive suite (experiments/optimality/exhaustive_silent.py).
PROGRAMS = {4: 29, 5: 24, 6: 22}


# --- synthetic data -------------------------------------------------------------------------


def pairings(qubits):
    """Every partition of `qubits` into pairs, plus one solo qubit when the count is odd."""
    qubits = list(qubits)
    if not qubits:
        yield []
        return
    if len(qubits) % 2:
        for solo in qubits:
            rest = [q for q in qubits if q != solo]
            for pairs in pairings(rest):
                yield pairs + [(solo,)]
        return
    first, rest = qubits[0], qubits[1:]
    for partner in rest:
        remaining = [q for q in rest if q != partner]
        for pairs in pairings(remaining):
            yield [(first, partner)] + pairs


def tightest_space(n: int):
    """
    The complete tightest [[4,2,2]] space: partition x patch order x lane map.

    24 / 720 / 720 placements at 4 / 5 / 6 qubits, the counts of the real enumeration
    (`experiments/revision/exp7_optimality/space.py::complete_placements`). Yields
    (partition index, lane-aligned fraction, order rank, placement as [patch, lane] per qubit).
    """
    partitions = list(pairings(range(n)))
    for index, partition in enumerate(partitions):
        for order_rank, order in enumerate(itertools.permutations(range(len(partition)))):
            for flips in itertools.product((0, 1), repeat=len(partition)):
                placement = [None] * n
                for patch, members, flip in zip(order, partition, flips):
                    lanes = (0, 1) if not flip else (1, 0)
                    for qubit, lane in zip(members, lanes):
                        placement[qubit] = [patch, lane]
                aligned = 1.0 - sum(flips) / len(partition)
                yield index, aligned, order_rank, placement


def synthesize(seed: int = SEED) -> pd.DataFrame:
    """
    A toy space with the structure the measurements have shown, not their values.

    Each partition j gets a merge share m_j (how alike its pairs act: merging removes gadgets,
    which helps every metric at once) and a cross-patch share t_j (interactions placed across
    patches: cross-patch gadgets leak no silent error but cost more gates). Lane alignment scales
    m; patch order perturbs depth; H commutation (c = 1) removes a share h of the gadgets. With
    per-program constants D0, S0, G0, L0, h, rho and small log-normal noise:

        D    = D0 (1 - 0.35 m) (1 + 0.12 t^2) (1 - h c)          expected detected error
        LSR  = exp(-D)                                           + binomial noise, 200k shots
        LER  = S0 (1 - 0.40 m) (1 - 0.6 sqrt t) (1 - h c)        + binomial noise on kept shots
        TVD  = rho LER                                           rho = 0 on uniform outputs
        2q   = G0 (1 - 0.30 m) (1 + 0.40 t) (1 - 0.5 h c)
        dep  = L0 (1 - 0.12 m) (1 + 0.08 t) (1 - 0.3 h c) (1 + 0.04 order + noise)
        cost = D e^{N(0, 0.05)} + 10 LER e^{N(0, 0.08)}          AQORN's model: right in trend,
                                                                 wrong per placement

    AQORN picks argmin cost x e^{N(0, 0.025)} (a search that lands near its optimum, not always
    on it); naive takes program-order pairing in program order; the interaction partition the
    partition with the most interactions inside patches (least t), with random lanes.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n, count in PROGRAMS.items():
        space = list(tightest_space(n))
        partition = np.array([s[0] for s in space])
        aligned = np.array([s[1] for s in space])
        order = np.array([s[2] for s in space], dtype=float)
        placements = [json.dumps(s[3]) for s in space]
        parts = partition.max() + 1
        for index in range(count):
            d0 = rng.uniform(0.25, 0.45) + 0.03 * n
            s0 = rng.uniform(0.020, 0.040) * n / 5
            g0, l0 = rng.uniform(20, 40) * n, rng.uniform(10, 18) * n
            h = rng.uniform(0.05, 0.20)
            rho = 0.0 if rng.random() < 0.1 else rng.uniform(0.75, 1.0)
            m_part = rng.beta(2.0, 2.5, parts)
            t_part = rng.uniform(0.0, 1.0, parts)

            for commute in (0, 1):
                m = m_part[partition] * (0.85 + 0.15 * aligned)
                t = t_part[partition]
                size = len(space)
                noise = lambda sd: np.exp(rng.normal(0.0, sd, size))  # noqa: E731
                detected = d0 * (1 - 0.35 * m) * (1 + 0.12 * t**2) * (1 - h * commute) * noise(0.015)
                silent = s0 * (1 - 0.40 * m) * (1 - 0.6 * np.sqrt(t)) * (1 - h * commute) * noise(0.03)
                lsr = np.exp(-detected)
                kept = 200_000 * lsr
                lsr = np.clip(lsr + rng.normal(0, np.sqrt(lsr * (1 - lsr) / 200_000)), 0, 1)
                ler = np.clip(silent + rng.normal(0, np.sqrt(silent / kept)), 1e-6, 1)
                tvd = rho * ler * noise(0.02)
                two_q = np.round(g0 * (1 - 0.30 * m) * (1 + 0.40 * t) * (1 - 0.5 * h * commute) * noise(0.02))
                depth = np.round(l0 * (1 - 0.12 * m) * (1 + 0.08 * t) * (1 - 0.3 * h * commute)
                                 * (1 + 0.04 * order) * noise(0.03))
                cost = detected * noise(0.05) + 10.0 * ler * noise(0.08)
                rows.append(pd.DataFrame({
                    "program": f"synthetic_{n}q_{index:02d}", "num_qubits": n,
                    "placement": placements, "commute": bool(commute), "cost": cost, "lsr": lsr,
                    "ler": ler, "tvd": tvd, "two_qubit_gates": two_q, "depth": depth,
                    "_partition": partition, "_t": t_part[partition], "_order": order,
                    "_aligned": aligned,
                }))

    frame = pd.concat(rows, ignore_index=True)
    frame["pick"] = ""
    for program, group in frame.groupby("program", sort=False):
        on = group[group["commute"]]
        searched = group["cost"] * np.exp(rng.normal(0.0, 0.025, len(group)))
        picks = {
            "aqorn": searched.idxmin(),
            "naive": on[(on["_partition"] == 0) & (on["_order"] == 0) & (on["_aligned"] == 1.0)].index[0],
            "interaction": rng.choice(on[on["_t"] == on["_t"].min()].index),
        }
        for arm, row in picks.items():
            frame.loc[row, "pick"] = ",".join(filter(None, [frame.loc[row, "pick"], arm]))
    frame["source"] = "synthetic"
    return frame.drop(columns=[c for c in frame.columns if c.startswith("_")])


# --- analysis -------------------------------------------------------------------------------


def picked(group: pd.DataFrame, arm: str) -> pd.Series:
    rows = group[group["pick"].fillna("").str.split(",").apply(lambda arms: arm in arms)]
    if len(rows) != 1:
        raise ValueError(f"{group['program'].iloc[0]}: {len(rows)} rows picked by {arm}")
    return rows.iloc[0]


def best_value(group: pd.DataFrame, column: str, higher: bool) -> float:
    """
    The best placement's value on `column`. Where independent selection shots exist
    (`<column>_sel`), the best placement is chosen on them and its value read on `column`.
    """
    chooser = f"{column}_sel" if f"{column}_sel" in group.columns else column
    index = group[chooser].idxmax() if higher else group[chooser].idxmin()
    return float(group.loc[index, column])


def beats_on_both(group: pd.DataFrame, row: pd.Series) -> pd.Series:
    """Placements better than `row` on LSR and on LER -- on both sets of shots when there are two."""
    better = (group["lsr"] > row["lsr"]) & (group["ler"] < row["ler"])
    if "lsr_sel" in group.columns and "ler_sel" in group.columns:
        better &= (group["lsr_sel"] > row["lsr_sel"]) & (group["ler_sel"] < row["ler_sel"])
    return better


def gaps(space: pd.DataFrame) -> pd.DataFrame:
    """Per program and arm: percent gap to the best placement on each metric, and dominance."""
    out = []
    for program, group in space.groupby("program", sort=False):
        for arm, _ in ARMS:
            row = picked(group, arm)
            record = {"program": program, "num_qubits": int(row["num_qubits"]), "arm": arm}
            for column, _, higher in METRICS:
                best = best_value(group, column, higher)
                if best <= 0:  # TVD on a uniform output: not measurable
                    record[column] = np.nan
                    continue
                gap = (best - row[column]) if higher else (row[column] - best)
                record[column] = 100.0 * gap / best
            better = beats_on_both(group, row)
            record["dominated_by_pct"] = 100.0 * better.mean()
            record["on_front"] = not better.any()
            out.append(record)
    return pd.DataFrame(out)


def front(points: pd.DataFrame) -> pd.DataFrame:
    """Placements no other placement beats on both LSR (higher) and LER (lower), by LER."""
    ordered = points.sort_values(["ler", "lsr"], ascending=[True, False])
    keep, best = [], -np.inf
    for index, lsr in ordered["lsr"].items():
        if lsr > best:
            keep.append(index)
            best = lsr
    return ordered.loc[keep]


def example_program(space: pd.DataFrame, table: pd.DataFrame) -> str:
    """The 6-qubit program whose AQORN LER gap is the median of its size."""
    six = table[(table["arm"] == "aqorn") & (table["num_qubits"] == table["num_qubits"].max())]
    ranked = six.sort_values("ler").reset_index(drop=True)
    return ranked.loc[(len(ranked) - 1) // 2, "program"]


def summarize(table: pd.DataFrame) -> None:
    print(f"[{NAME}] mean gap to the best placement (%), {table['program'].nunique()} programs")
    columns_ = [c for c, _, _ in METRICS] + ["dominated_by_pct"]
    print(table.groupby("arm")[columns_].mean().loc[[a for a, _ in ARMS]].round(2).to_string())
    print(f"[{NAME}] median gap to the best placement (%)")
    columns = [c for c, _, _ in METRICS] + ["dominated_by_pct"]
    print(table.groupby("arm")[columns].median().loc[[a for a, _ in ARMS]].round(2).to_string())
    print(f"[{NAME}] AQORN cost gap by width, mean +- SD (P(opt)):")
    aqorn = table[table["arm"] == "aqorn"]
    for n, group in aqorn.groupby("num_qubits"):
        print(f"    {n} qubits: {group['cost'].mean():.2f} +- {group['cost'].std():.2f}% "
              f"({(group['cost'] < 1e-9).mean():.0%} optimal)")
    print(f"[{NAME}] on the LSR/LER front: " + ", ".join(
        f"{label} {table[table['arm'] == arm]['on_front'].mean():.0%}" for arm, label in ARMS))
    blind = table[(table["arm"] == "aqorn") & table["tvd"].isna()]["program"].nunique()
    print(f"[{NAME}] TVD row excludes {blind} uniform-output programs")


# --- drawing --------------------------------------------------------------------------------


def series_style(i: int) -> dict:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    return {key: values[i % len(values)] for key, values in cycle.items()}


def panel_label(ax, letter: str, text: str) -> None:
    label = ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, fontweight="bold",
                    ha="left", va="bottom")
    ax.annotate(f" {text}", xy=(1.0, 0.0), xycoords=label, va="bottom", ha="left")


def watermark(ax) -> None:
    ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, rotation=30, ha="center",
            va="center", fontsize=16, color="#d0d0d0", alpha=0.6, zorder=0)


def panel_space(ax, group: pd.DataFrame) -> None:
    ax.scatter(group["ler"], group["lsr"], s=1.5, color="#cfcfcf", linewidths=0, zorder=1,
               rasterized=False)
    edge = front(group)
    # marker/linestyle explicit: the style cycles them, and a bare plot inherits a series marker.
    ax.step(edge["ler"], edge["lsr"], where="post", color="#636363", linewidth=0.8,
            marker="none", linestyle="-", zorder=2)

    # Single-metric peaks: open rings, so they read as references and not as packers.
    lsr_chooser = "lsr_sel" if "lsr_sel" in group.columns else "lsr"
    ler_chooser = "ler_sel" if "ler_sel" in group.columns else "ler"
    peaks = {"best LSR": group.loc[group[lsr_chooser].idxmax()],
             "best LER": group.loc[group[ler_chooser].idxmin()]}
    for text, row in peaks.items():
        ax.scatter(row["ler"], row["lsr"], s=28, facecolors="none", edgecolors="#636363",
                   linewidths=0.7, zorder=3)
    top, left = peaks["best LSR"], peaks["best LER"]
    # Labels on the front's empty upper-left side, so neither reads as naming a packer's marker.
    ax.annotate("best LSR", (top["ler"], top["lsr"]), xytext=(-6, 0), textcoords="offset points",
                fontsize=8, ha="right", va="center")
    # Below and to the right: to its left the label runs into the y axis.
    ax.annotate("best LER", (left["ler"], left["lsr"]), xytext=(3, -6),
                textcoords="offset points", fontsize=8, ha="left", va="top")

    for i, (arm, _) in enumerate(ARMS):
        row = picked(group, arm)
        style = series_style(i)
        # Thin black edge: markers sit on the grey cloud, and the style's edgeless markers
        # (lines.markeredgewidth 0) vanish into it in greyscale.
        ax.plot(row["ler"], row["lsr"], linestyle="none", marker=style["marker"],
                color=style["color"], markersize=5, markeredgecolor="black",
                markeredgewidth=0.4, zorder=4)

    # Truncated x as well (10-11): from zero the placements filled the right third of the
    # panel. Pad by the same share of the span as y, so the cloud fills the axes.
    xspan = group["ler"].max() - group["ler"].min()
    ax.set_xlim(group["ler"].min() - 0.08 * xspan, group["ler"].max() + 0.08 * xspan)
    # Truncated y: the whole space spans ~15% of LSR, and from zero it collapses to a band.
    span = group["lsr"].max() - group["lsr"].min()
    ax.set_ylim(group["lsr"].min() - 0.05 * span, group["lsr"].max() + 0.12 * span)
    # Three ticks at most: four decimal labels crowd a half-column axis.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.set_xlabel("Logical error rate")
    ax.set_ylabel("Logical selection rate")


def panel_gaps(ax, table: pd.DataFrame, spread: str = "iqr") -> None:
    offsets = np.linspace(-0.24, 0.24, len(ARMS))
    reach = 0.0
    for i, (arm, _) in enumerate(ARMS):
        style = series_style(i)
        sub = table[table["arm"] == arm]
        for row, (column, _, _) in enumerate(METRICS):
            values = sub[column].dropna()
            if spread == "sd":   # mean +- 1 SD, clipped at 0 (house style)
                centre, sd = float(values.mean()), float(values.std())
                low, high = max(0.0, centre - sd), centre + sd
            else:                # median and interquartile range
                low, centre, high = (float(v) for v in np.percentile(values, [25, 50, 75]))
            reach = max(reach, high)
            y = row + offsets[i]
            ax.plot([low, high], [y, y], color=style["color"],
                    linewidth=0.8, marker="none", linestyle="-", zorder=2)
            ax.plot(centre, y, linestyle="none", marker=style["marker"], color=style["color"],
                    markersize=4, markeredgecolor="black", markeredgewidth=0.3, zorder=3)
    for row, (column, _, _) in enumerate(METRICS):
        if column in SEPARATE_AFTER:
            ax.axhline(row + 0.5, color="#bdbdbd", linestyle=":", linewidth=1.0, zorder=1)

    # Symlog: AQORN's cost gap is a fraction of a percent while the conventional packers'
    # logical-error gaps run past 100%; a linear axis hides the first, a log one drops zero.
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    # A sliver below zero so a zero-gap marker is drawn whole rather than cut by the spine.
    ax.set_xlim(-0.35, max(400.0, 1.2 * reach))
    ax.set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(METRICS)), [label for _, label, _ in METRICS])
    ax.set_ylim(len(METRICS) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    # A dot plot is read across to the x ticks, so it needs the vertical grid and not the
    # style's horizontal one.
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Gap to best (%)")


def draw(space: pd.DataFrame, table: pd.DataFrame, output_dir=None, versioned=False,
         spread: str = "iqr"):
    use_style()
    synthetic = (space["source"] == "synthetic").any()
    example = example_program(space, table)
    fig, (left, right) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.65),
                                      gridspec_kw={"width_ratios": [1.0, 1.0]})  # equal plotting areas
    panel_space(left, space[space["program"] == example])
    panel_gaps(right, table, spread)
    panel_label(left, "a", "one program")
    panel_label(right, "b", f"{table['program'].nunique()} programs")
    if synthetic:
        watermark(left)
        watermark(right)

    handles = [Line2D([], [], linestyle="none", marker=series_style(i)["marker"],
                      color=series_style(i)["color"], markeredgecolor="black",
                      markeredgewidth=0.4, markersize=5, label=label)
               for i, (_, label) in enumerate(ARMS)]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(ARMS), frameon=False,
               handletextpad=0.2, columnspacing=0.9)
    name = f"{NAME}_synthetic" if synthetic else NAME
    paths = save_figure(fig, name, Path(__file__), override=output_dir, versioned=versioned)
    plt.close(fig)
    print(f"[{NAME}] panel (a) example: {example}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="redraw from a cached space CSV (synthetic cache or measured data)")
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the synthetic space")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--versioned", action="store_true", help="write _vN instead of replacing")
    parser.add_argument("--spread", choices=("iqr", "sd"), default="iqr",
                        help="panel (b): median + IQR (default) or mean +- 1 SD clipped at 0")
    parser.add_argument("--persist", action="store_true",
                        help="also copy the CSVs to results/data/ (only worth it for measured data)")
    parser.add_argument("--cost-label", default=None,
                        help="row label for the objective (default AH*); e.g. 'objective' when the "
                             "cost column is the standard objective (final_front.py)")
    args = parser.parse_args()
    if args.cost_label:
        METRICS[0] = ("cost", args.cost_label, False)

    if args.from_csv:
        space = pd.read_csv(args.from_csv, keep_default_na=False, na_values=[""])
        space["pick"] = space["pick"].fillna("")
        if (space["source"] != "synthetic").any() and "cost_model_version" in space:
            from experiments.shared.figures import check_provenance
            check_provenance(space, args.from_csv)
    else:
        space = synthesize(args.seed)
        space.to_csv(SPACE_CSV, index=False)
        print(f"[{NAME}] synthetic space (seed {args.seed}): {len(space)} placements, "
              f"{space['program'].nunique()} programs -> {SPACE_CSV}")

    table = gaps(space)
    gaps_csv = GAPS_CSV if not args.from_csv else Path(args.from_csv).with_name(
        Path(args.from_csv).stem + "_gaps.csv")
    table.to_csv(gaps_csv, index=False)
    summarize(table)
    if args.persist:
        for path in (SPACE_CSV if not args.from_csv else args.from_csv, gaps_csv):
            shutil.copy(path, data_dir(Path(__file__)) / Path(path).name)

    synthetic = (space["source"] == "synthetic").any()
    for path in draw(space, table, args.output_dir, args.versioned or not synthetic, args.spread):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
