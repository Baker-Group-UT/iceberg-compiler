"""
concat_figures.py -- the simplified concatenation figures, from `concat_standard.py`'s CSV.

    concat_gain      (option A, one column, two panels)
        (a) AQORN against its own baseline at one level and at two: % reduction in two-qubit
            gates, depth and logical error rate (mean over circuits, bootstrap 95% CI), at each
            circuit's calibrated noise (the two-level baseline keeps half its shots).
        (b) logical error rate against the physical two-qubit error rate p2 = 3e-3 alpha,
            pooled over circuits: one level / two levels x baseline / AQORN. The band marks
            the calibrated noise of panel (a) (interquartile range over circuits).
    concat_tradeoff  (option C, one panel)
        shots kept (LSR) against logical error rate at the calibrated noise, per circuit
        (small points) and per arm (mean LSR, geometric-mean logical error; large markers).
        Arrows: baseline -> AQORN at each level, one level -> two levels for the baseline.

"Logical error rate" is the code's silent error per kept shot (`postselect_silent`): among
shots every check accepts, the fraction with a wrong decoded output. Post-selection decoding
throughout. `--summary` prints the numbers the text cites (outer / inner split, index order,
the erasure decoder, the factors in panel b).

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_figures.concat_figures \
        --from-csv results/concatenation/concat_standard.csv --output-dir ../../../../results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
P2_PER_ALPHA = 3e-3          # NEUTRAL_ATOM two-qubit error at alpha = 1
BLUE, BLUE_LIGHT, GRAY = "#0097fd", "#a6dafe", "#636363"
LER = "postselect_silent"
LSR = "postselect_lsr"
PAIRS = {"1 level": ("single_base", "single_std"), "2 levels": ("concat_BB", "concat_AA")}
METRICS = [("two_qubit_gates", "2q gates"), ("depth", "Depth"), (LER, "Logical\nerror rate")]
LINES = [  # arm, label, colour, filled, linestyle, marker
    ("single_base", "[[4,2,2]], baseline", GRAY, False, "--", "o"),
    ("single_std", "[[4,2,2]], AQORN", BLUE, False, "--", "o"),
    ("concat_BB", "[[16,4,4]], baseline", GRAY, True, "-", "s"),
    ("concat_AA", "[[16,4,4]], AQORN", BLUE, True, "-", "s"),
]
MIN_EVENTS = 20              # a pooled point needs this many logical errors to be drawn


def calibrated(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["alpha_label"] == "calibrated"]


def wide(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(index="circuit_name", columns="arm", values=column)


def reduction(frame: pd.DataFrame, column: str, base: str, arm: str) -> pd.Series:
    """Per-circuit % reduction of `column`, arm against base (circuits with base 0 dropped)."""
    table = wide(frame, column)
    ok = table[base] > 0
    return ((table[base] - table[arm]) / table[base] * 100)[ok]


def bootstrap_ci(values: np.ndarray, *, draws: int = 10_000, seed: int = 7) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def geo_ratio(frame: pd.DataFrame, column: str, base: str, arm: str) -> float:
    table = wide(frame, column)
    ratio = (table[arm] / table[base]).replace([0, np.inf], np.nan).dropna()
    return float(np.exp(np.log(ratio).mean()))


def pooled(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Logical error per kept shot pooled over circuits (errors / kept shots), per fixed alpha."""
    fixed = frame[(frame["alpha_label"] != "calibrated") & (frame["arm"] == arm)]
    kept = fixed[LSR]                        # fraction kept; every circuit has the same shots
    errors = fixed[LER] * kept
    out = pd.DataFrame({"kept": kept.groupby(fixed["alpha"]).sum(),
                        "errors": errors.groupby(fixed["alpha"]).sum()})
    shots = frame.attrs.get("frame_shots", 200_000)
    out["events"] = out["errors"] * shots
    out["rate"] = out["errors"] / out["kept"]
    return out


# ----------------------------------------------------------------------------------------
# option A
# ----------------------------------------------------------------------------------------


def panel_gain(ax, frame: pd.DataFrame) -> None:
    cal = calibrated(frame)
    x = np.arange(len(METRICS))
    width = 0.36
    styles = {"1 level": dict(color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.6),
              "2 levels": dict(color=BLUE, edgecolor=BLUE, linewidth=0.6)}
    for i, (level, (base, arm)) in enumerate(PAIRS.items()):
        means, lows, highs, counts = [], [], [], []
        for column, _ in METRICS:
            values = reduction(cal, column, base, arm).to_numpy()
            low, high = bootstrap_ci(values)
            means.append(values.mean())
            lows.append(values.mean() - low)
            highs.append(high - values.mean())
            counts.append((int((values > 0).sum()), len(values)))
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=[lows, highs], capsize=2, label=level,
               error_kw={"elinewidth": 0.6, "capthick": 0.6}, **styles[level])
        for xi, mean, high, (better, total) in zip(x + offset, means, highs, counts):
            ax.text(xi, mean + high + 1.5, f"{better}/{total}", ha="center", va="bottom",
                    fontsize=6.5, color="#444444")
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_ylabel("Reduction vs. baseline (%)")
    ax.set_ylim(0, max(80, ax.get_ylim()[1]))
    ax.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=1.0)
    ax.set_title("(a) AQORN's gain at one and two levels", loc="left")


def panel_scaling(ax, frame: pd.DataFrame, standalone: bool = False) -> None:
    for arm, label, colour, filled, linestyle, marker in LINES:
        curve = pooled(frame, arm)
        curve = curve[curve["events"] >= MIN_EVENTS]
        ax.plot(curve.index * P2_PER_ALPHA, curve["rate"], color=colour, linestyle=linestyle,
                marker=marker, markersize=3.5, markerfacecolor=colour if filled else "white",
                markeredgecolor=colour, markeredgewidth=0.8, label=label)
    alphas = calibrated(frame).drop_duplicates("circuit_name")["alpha"] * P2_PER_ALPHA
    ax.axvspan(alphas.quantile(0.25), alphas.quantile(0.75), color="#e9e9e9", zorder=0,
               linewidth=0)
    ax.text(alphas.median(), 0.97, "calibrated" if standalone else "(a)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8 if standalone else 7, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="both")
    ax.set_xlabel(r"Physical two-qubit error rate $p_2$")
    ax.set_ylabel("Logical error rate")
    ax.legend(loc="upper left", fontsize=8 if standalone else 7.5, handlelength=2.2)
    if not standalone:
        ax.set_title("(b) Logical error rate vs. noise", loc="left")


def draw_scaling(frame: pd.DataFrame, output_dir=None):
    """Panel (b) alone (10-11: the paper keeps only this plot)."""
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(3.487, 2.6))
    panel_scaling(ax, frame, standalone=True)
    paths = save_figure(fig, "concat_scaling", Path(__file__), override=output_dir,
                        versioned=True)
    plt.close(fig)
    return paths


def draw_gain(frame: pd.DataFrame, output_dir=None):
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(3.487, 4.6))
    panel_gain(axes[0], frame)
    panel_scaling(axes[1], frame)
    paths = save_figure(fig, "concat_gain", Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------
# option C
# ----------------------------------------------------------------------------------------


def _log_label(value: float, _position) -> str:
    """$10^{e}$ at decades, $3{\\times}10^{e}$ between them."""
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = value / 10 ** exponent
    if abs(mantissa - 1) < 1e-6:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}{{\times}}10^{{{exponent}}}$"


def centre(cal: pd.DataFrame, arm: str, shots: int) -> Tuple[float, float]:
    sub = cal[cal["arm"] == arm]
    floor = 0.5 / (sub[LSR] * shots)         # half an event where none was seen
    ler = np.maximum(sub[LER], floor)
    return float(sub[LSR].mean()), float(np.exp(np.log(ler).mean()))


def draw_tradeoff(frame: pd.DataFrame, output_dir=None):
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    cal = calibrated(frame)
    shots = frame.attrs.get("frame_shots", 200_000)
    fig, ax = plt.subplots(figsize=(3.487, 2.7))
    centres: Dict[str, Tuple[float, float]] = {}
    for arm, label, colour, filled, _, marker in LINES:
        sub = cal[cal["arm"] == arm]
        ler = np.maximum(sub[LER], 0.5 / (sub[LSR] * shots))
        ax.scatter(sub[LSR], ler, s=7, marker=marker, alpha=0.35, linewidths=0.6,
                   facecolors=colour if filled else "none", edgecolors=colour, zorder=2)
        centres[arm] = centre(cal, arm, shots)
        ax.scatter(*centres[arm], s=46, marker=marker, linewidths=1.2, zorder=4,
                   facecolors=colour if filled else "white", edgecolors=colour, label=label)

    def arrow(a: str, b: str, colour: str, text: str, dx: float, dy: float,
              ha: str = "center", at_end: bool = False) -> None:
        (x0, y0), (x1, y1) = centres[a], centres[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0, shrinkA=5,
                                    shrinkB=5, mutation_scale=8))
        xm, ym = (x1, y1) if at_end else ((x0 + x1) / 2, np.sqrt(y0 * y1))
        ax.annotate(text, (xm, ym), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    color=colour, ha=ha, va="center", linespacing=1.0, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none",
                              alpha=0.85))

    ratio = lambda a, b: centres[a][1] / centres[b][1]
    arrow("single_base", "single_std", BLUE,
          f"AQORN\n{ratio('single_base', 'single_std'):.1f}\u00d7 lower", 7, 0, "left")
    arrow("concat_BB", "concat_AA", BLUE,
          f"AQORN\n{ratio('concat_BB', 'concat_AA'):.1f}\u00d7 lower", 10, 0, "left", True)
    arrow("single_base", "concat_BB", GRAY,
          f"concatenation\n{ratio('single_base', 'concat_BB'):.1f}\u00d7 lower", 8, -13)
    ax.set_xlim(right=1.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", axis="both")
    ax.set_xlabel("Shots kept (LSR)")
    ax.set_ylabel("Logical error rate")
    ax.legend(loc="upper left", bbox_to_anchor=(0.14, 1.0), fontsize=7.5, handletextpad=0.3,
              borderaxespad=0.3, ncol=2, columnspacing=0.8)
    paths = save_figure(fig, "concat_tradeoff", Path(__file__), override=output_dir,
                        versioned=True)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------
# correction (concat_correct.py's CSV)
# ----------------------------------------------------------------------------------------


def _pooled_mode(frame: pd.DataFrame, arm: str, rounds: int, mode: str) -> pd.DataFrame:
    fixed = frame[(frame["alpha_label"] != "calibrated") & (frame["arm"] == arm)
                  & (frame["rounds"] == rounds)]
    kept = fixed[f"{mode}_lsr"]
    errors = fixed[f"{mode}_silent"] * kept
    out = pd.DataFrame({"lsr": kept.groupby(fixed["alpha"]).mean(),
                        "kept": kept.groupby(fixed["alpha"]).sum(),
                        "errors": errors.groupby(fixed["alpha"]).sum()})
    out["events"] = out["errors"] * frame.attrs.get("frame_shots", 200_000)
    out["rate"] = out["errors"] / out["kept"]
    return out


def draw_correction(frame: pd.DataFrame, output_dir=None, *, rounds: int = 4,
                    mode: str = "lookup99"):
    """
    Two levels, baseline and AQORN: detection (checks at the end, post-selection; Figure A's
    configuration) against correction (`rounds` mid-circuit inner rounds, lookup decoder).
    (a) shots discarded, (b) logical error rate, both against p2.
    """
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(3.487, 4.4), sharex=True)
    curves = [("concat_BB", 0, "postselect", GRAY, "-", True, "baseline, detection"),
              ("concat_AA", 0, "postselect", BLUE, "-", True, "AQORN, detection"),
              ("concat_BB", rounds, mode, GRAY, "--", False, "baseline, correction"),
              ("concat_AA", rounds, mode, BLUE, "--", False, "AQORN, correction")]
    for arm, r, m, colour, linestyle, filled, label in curves:
        curve = _pooled_mode(frame, arm, r, m)
        style = dict(color=colour, linestyle=linestyle, marker="s", markersize=3.5,
                     markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                     markeredgewidth=0.8, label=label)
        axes[0].plot(curve.index * P2_PER_ALPHA, 1 - curve["lsr"], **style)
        good = curve[curve["events"] >= MIN_EVENTS]
        axes[1].plot(good.index * P2_PER_ALPHA, good["rate"], **style)
    alphas = calibrated(frame).drop_duplicates("circuit_name")["alpha"] * P2_PER_ALPHA
    for ax in axes:
        ax.axvspan(alphas.quantile(0.25), alphas.quantile(0.75), color="#e9e9e9", zorder=0,
                   linewidth=0)
        ax.set_xscale("log")
        ax.grid(True, which="major", axis="both")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Shots discarded")
    axes[0].legend(loc="upper left", fontsize=7.5, handlelength=2.2)
    axes[0].set_title("(a) Shots discarded", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Logical error rate")
    axes[1].set_xlabel(r"Physical two-qubit error rate $p_2$")
    axes[1].set_title("(b) Logical error rate", loc="left")
    paths = save_figure(fig, "concat_correction", Path(__file__), override=output_dir,
                        versioned=True)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------
# numbers for the text
# ----------------------------------------------------------------------------------------


def summary(frame: pd.DataFrame) -> str:
    cal = calibrated(frame)
    lines = [f"circuits {cal['circuit_name'].nunique()}, calibrated alpha median "
             f"{cal.drop_duplicates('circuit_name')['alpha'].median():.3f} "
             f"(p2 {cal.drop_duplicates('circuit_name')['alpha'].median() * P2_PER_ALPHA:.1e}), "
             f"logical optimisation {sorted(set(frame.get('logical_optimization', ['?'])))}, "
             f"outer plans equal Fig 6's: {int(cal.drop_duplicates('circuit_name')['outer_plan_matches_fig6'].sum())}"]
    lines.append("noiseless: min LSR %.4f, max TVD %.4f" % (frame["noiseless_lsr"].min(),
                                                            frame["noiseless_tvd"].max()))
    lines.append("\n% change vs reference, mean [median] (better/total), calibrated noise")
    comparisons = [("1 level: AQORN vs baseline", "single_base", "single_std"),
                   ("2 levels: AQORN both vs baseline", "concat_BB", "concat_AA"),
                   ("2 levels: outer only", "concat_BB", "concat_AB"),
                   ("2 levels: inner only", "concat_BB", "concat_BA"),
                   ("2 levels: index order (not spread)", "concat_BB", "concat_AA_index"),
                   ("index vs spread (both AQORN)", "concat_AA", "concat_AA_index")]
    for label, base, arm in comparisons:
        cells = []
        for column, name in (("one_qubit_gates", "1q"), ("two_qubit_gates", "2q"),
                             ("depth", "depth"), (LSR, "LSR"), (LER, "LER")):
            values = reduction(cal, column, base, arm)
            if column == LSR:
                values = -values
            cells.append(f"{name} {values.mean():+.1f} [{values.median():+.1f}] "
                         f"({int((values > 0).sum())}/{len(values)})")
        cells.append(f"LER geo ratio {geo_ratio(cal, LER, base, arm):.2f}x")
        lines.append(f"{label:38s} " + "; ".join(cells))
    lines.append("\nmeans at calibrated noise (LSR, LER, TVD, physical qubits)")
    for arm in sorted(cal["arm"].unique()):
        sub = cal[cal["arm"] == arm]
        lines.append(f"  {arm:16s} LSR {sub[LSR].mean():.3f}  LER {sub[LER].mean():.2e}  "
                     f"TVD {sub['postselect_tvd'].mean():.4f}  qubits {sub['physical_qubits'].mean():.1f}")
    lines.append("\npooled logical error rate by alpha (events in brackets)")
    for arm, label, *_ in LINES + [("concat_AB", "2 levels, outer only", None, None, None, None),
                                   ("concat_BA", "2 levels, inner only", None, None, None, None),
                                   ("concat_AA_index", "2 levels, index order", None, None, None,
                                    None)]:
        curve = pooled(frame, arm)
        slope = np.polyfit(np.log(curve.index), np.log(curve["rate"].clip(lower=1e-12)), 1)[0]
        cells = "  ".join(f"{a:g}: {r:.2e} ({e:.0f})" for a, r, e in
                          zip(curve.index, curve["rate"], curve["events"]))
        lines.append(f"  {label:26s} {cells}  slope {slope:.2f}")
    lines.append("\nper-circuit geometric-mean factors by alpha")
    for label_alpha in [l for l in frame["alpha_label"].unique()]:
        sub = frame[frame["alpha_label"] == label_alpha]
        f = lambda a, b: geo_ratio(sub, LER, a, b)
        lines.append(f"  {label_alpha:10s} 1L base/1L AQORN {1 / f('single_base', 'single_std'):.2f}x  "
                     f"1L base/2L base {1 / f('single_base', 'concat_BB'):.2f}x  "
                     f"2L base/2L AQORN {1 / f('concat_BB', 'concat_AA'):.2f}x  "
                     f"1L base/2L AQORN {1 / f('single_base', 'concat_AA'):.2f}x")
    two = cal[cal["arm"].isin(["concat_BB", "concat_AA", "concat_AA_index"])]
    lines.append("\ndecoders at calibrated noise (mean LSR / LER / TVD)")
    for arm, sub in two.groupby("arm"):
        lines.append(f"  {arm:16s} post-select {sub[LSR].mean():.3f} / {sub[LER].mean():.2e} / "
                     f"{sub['postselect_tvd'].mean():.4f}   erasure {sub['erasure_lsr'].mean():.3f} / "
                     f"{sub['erasure_silent'].mean():.2e} / {sub['erasure_tvd'].mean():.4f}")
    return "\n".join(lines)


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", default="results/concatenation/concat_standard.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--correction-csv", default=None,
                        help="concat_correct.py's CSV: draw concat_correction instead")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--mode", default="lookup99")
    parser.add_argument("--scaling-only", action="store_true",
                        help="draw only panel (b) of concat_gain, as concat_scaling")
    args = parser.parse_args()
    if args.correction_csv:
        frame = pd.read_csv(args.correction_csv)
        output = Path(args.output_dir) if args.output_dir else None
        for path in draw_correction(frame, output, rounds=args.rounds, mode=args.mode):
            print(f"[concat_figures] saved {path}")
        return
    frame = pd.read_csv(args.from_csv)
    sidecar = Path(args.from_csv + ".json")
    if sidecar.exists():
        frame.attrs["frame_shots"] = json.loads(sidecar.read_text())["settings"]["frame_shots"]
    if args.summary:
        print(summary(frame))
    output = Path(args.output_dir) if args.output_dir else None
    if args.scaling_only:
        for path in draw_scaling(frame, output):
            print(f"[concat_figures] saved {path}")
        return
    for path in draw_gain(frame, output) + draw_tradeoff(frame, output):
        print(f"[concat_figures] saved {path}")


if __name__ == "__main__":
    main()
