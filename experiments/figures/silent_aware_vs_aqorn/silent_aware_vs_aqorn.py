"""
silent_aware_vs_aqorn.py -- per-circuit gain of the silent-error-aware cost model over AQORN's.

Companion to `ablation_silent_aware` (Fig 6 regenerated with the silent-aware packing stage)
and drawn from the same cached CSV, so no new simulation: for each of Fig 6's 71 circuits,
the full pipeline planned with the silent-aware cost model against the full pipeline planned
with AQORN's original cost model, which is the baseline here. AQORN's arm is Fig 6's own
sample; the silent-aware arm is Fig 6's pipeline with only the planner swapped.

Five panels in Fig 6's order. Every panel is signed so that up is better:

    #1q gates, #2q gates, depth   % reduction against AQORN
    LSR                           % improvement against AQORN (more shots kept)
    TVD                           % reduction against AQORN (more accurate)

Bars are single raw samples per circuit, at Fig 6's per-circuit calibration and shot count,
exactly as Fig 6's are; the equal-shot comparison with bootstrap intervals is Experiment8b in
`REVISION_RESULTS.md`. Circuits whose plan the new cost model left unchanged are exact ties
and are marked at zero. Stars mark the circuits Fig 6 withholds from TVD (baseline TVD > 0.95,
output support beyond any affordable shot count). A panel whose lowest bar is a lone extreme
outlier is cut just below the next-worst bar, with the clipped bar's value printed beside it
(`_outlier_floor`); at weight 10 that is ising_n98's depth, 5x AQORN's.

Deviation from house style, inherited on purpose from Fig 6: the same 22 in canvas and size
ladder (`AXIS_PT` in the Fig 6 script), so the two figures read side by side.

Run:
    PYTHONPATH=. venv/bin/python -m experiments.figures.silent_aware_vs_aqorn.silent_aware_vs_aqorn
    ... --from-csv experiments/figures/ablation_silent_aware/ablation_silent_aware_w3_versus_aqorn.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "silent_aware_vs_aqorn"
SOURCE = HERE.parent / "ablation_silent_aware"

#: (column stem in the versus CSV, panel label, True if lower is better)
PANELS = (
    ("one_qubit_gates", "#1q gates", True),
    ("two_qubit_gates", "#2q gates", True),
    ("circuit_depth", "depth", True),
    ("lsr", "LSR", False),
    ("tvd", "TVD", True),
)
GROUPS = ("small", "medium", "large")


def improvements(versus: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit: each panel's % gain of the silent-aware arm over AQORN's."""
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    out = versus[["size_group", "circuit_idx", "circuit_name", "num_qubits", "baseline_tvd",
                  "plan_changed", "commute", "commute_aqorn"]].copy()
    for stem, _, lower_is_better in PANELS:
        aqorn, silent = versus[f"aqorn_{stem}"].astype(float), versus[f"silent_{stem}"].astype(float)
        if lower_is_better:
            gain = 100.0 * (aqorn - silent) / aqorn.where(aqorn > 0)
        else:
            gain = 100.0 * (silent - aqorn) / aqorn.where(aqorn > 0)
        out[stem] = gain.fillna(0.0)
    out["tvd_resolved"] = out["baseline_tvd"].astype(float) <= fig6.TVD_SATURATION
    return out


def _outlier_floor(lows: List[float], mean: float) -> Optional[float]:
    """
    Where to cut a panel's axis when its lowest bar is an outlier, or None to keep it whole.

    An outlier here is a bar below -50% that is more than twice as far below zero as the
    next one: a single circuit that would otherwise set the scale for all 71. The cut sits
    35% below the next-worst bar (and below the average bar), so everything else keeps its
    true height.
    """
    if len(lows) < 2 or lows[0] >= -50.0 or lows[0] >= 2.0 * min(lows[1], -1e-9):
        return None
    return 1.35 * min(lows[1], mean, -10.0)


def plot(gains: pd.DataFrame, weight: float, output_dir: Optional[Path],
         series: Optional[str] = None, output_name: Optional[str] = None) -> List[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,  # noqa: F401 -- Fig 6's rcParams
    )
    from experiments.shared.figures import save_figure

    fig, axes = plt.subplots(nrows=len(PANELS), ncols=1, figsize=(22, 20), sharex=True)
    bar_width, gap, bar_step = 0.825, 4.0, 1.1  # Fig 6's geometry
    ordered = {g: sorted(gains.loc[gains["size_group"] == g, "circuit_name"]) for g in GROUPS}
    by_name = gains.set_index("circuit_name")

    for ax, (stem, label, _) in zip(axes, PANELS):
        positions: List[float] = []
        labels: List[str] = []
        points: List[tuple] = []  # (x, value) of every drawn bar, for the outlier clip
        x = 0.0
        low = high = 0.0
        for group in GROUPS:
            names = ordered[group]
            if not names:
                continue
            xs = np.arange(len(names)) * bar_step + x
            values = by_name.loc[names, stem].astype(float).to_numpy()
            drawn = np.ones(len(names), dtype=bool)
            if stem == "tvd":
                drawn = by_name.loc[names, "tvd_resolved"].to_numpy(dtype=bool)
            ax.bar(xs[drawn], values[drawn], width=bar_width, color="#0097fd",
                   edgecolor="white", alpha=0.85,
                   label=series or f"silent-aware (weight {weight:g}) vs. AQORN")
            same = ~by_name.loc[names, "plan_changed"].to_numpy(dtype=bool)
            if (same & drawn).any():  # no legend entry for a marker that never appears
                ax.plot(xs[same & drawn], np.zeros(int((same & drawn).sum())), linestyle="none",
                        marker="o", markersize=5, color="#bdbdbd",
                        label="plan unchanged (identical to AQORN)")
            if stem == "tvd":
                for xi in xs[~drawn]:
                    ax.plot([xi], [0.0], marker="*", markersize=14, color="#3b4c56",
                            linestyle="none", clip_on=False)
            if drawn.any():
                low = min(low, float(values[drawn].min()))
                high = max(high, float(values[drawn].max()))
                points.extend(zip(xs[drawn].tolist(), values[drawn].tolist()))
            positions.extend(xs.tolist())
            labels.extend(names)
            x = xs[-1] + gap

        scope = gains.loc[gains["tvd_resolved"]] if stem == "tvd" else gains
        mean = float(scope[stem].astype(float).mean())
        avg_x, avg_width = x + 1.0, 5 * bar_width
        ax.bar(avg_x, mean, width=avg_width, color="#0097fd", edgecolor="white", alpha=0.95)
        ax.annotate(f"{mean:+.1f}%", (avg_x, mean), textcoords="offset points",
                    xytext=(0, 6 if mean >= 0 else -22), ha="center", fontsize=16)
        ax.axvline(x - gap / 2.0, color="#bdbdbd", linestyle=":", linewidth=1.0)
        ax.set_xlim(right=avg_x + avg_width / 2.0 + (avg_x - avg_width / 2.0 - (x - gap / 2.0)))
        positions.append(avg_x)
        labels.append("average")

        low, high = min(low, mean), max(high, mean)
        # One bar far below the rest (ising_n98's depth at weight 10 is 5x AQORN's, -400%)
        # would flatten every other bar in its panel. Cut the axis below the next-worst bar
        # and print the clipped bar's true value beside it; the caption has to say so.
        floor = _outlier_floor(sorted(v for _, v in points), mean)
        if floor is not None:
            low = floor
            for xi, value in points:
                if value < floor:
                    ax.annotate(f"{value:.0f}%", (xi, floor), textcoords="offset points",
                                xytext=(7, 4), ha="left", va="bottom", fontsize=16)
        span = max(high - low, 1.0)
        ax.set_ylim(low if floor is not None else (low - 0.08 * span if low < 0 else 0.0),
                    high + 0.12 * span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylabel(label, fontsize=fig6.AXIS_PT)
        ax.set_xticks([])
        ax.set_xticks(positions, minor=True)
        ax.set_xticklabels(labels, rotation=45, ha="right", minor=True)
        ticks = ax.get_xticklabels(minor=True)
        if ticks:
            ticks[-1].set_rotation(0)
            ticks[-1].set_ha("center")
        ax.tick_params(axis="x", which="major", bottom=False, labelbottom=False)
        ax.tick_params(axis="x", which="minor", pad=5)

    axes[-1].set_xlabel("Benchmark circuit (grouped by size)", fontsize=fig6.AXIS_PT)
    handles, names = axes[0].get_legend_handles_labels()
    unique = dict(zip(names, handles))
    # The comparison first, the tie marker second.
    order = sorted(unique, key=lambda name: name.startswith("plan unchanged"))
    axes[0].legend([unique[n] for n in order], order, loc="best", frameon=False)
    fig.supylabel("Improvement over AQORN's cost model (%)", fontsize=fig6.AXIS_PT)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    paths = save_figure(fig, output_name or f"{NAME}_w{weight:g}", Path(__file__), override=output_dir,
                        versioned=True)
    plt.close(fig)
    return paths


def summarize(gains: pd.DataFrame) -> Dict[str, float]:
    print(f"[{NAME}] {int(gains['plan_changed'].sum())}/{len(gains)} circuits changed; "
          f"TVD resolved on {int(gains['tvd_resolved'].sum())}")
    out = {}
    for stem, label, _ in PANELS:
        scope = gains.loc[gains["tvd_resolved"]] if stem == "tvd" else gains
        changed = scope.loc[scope["plan_changed"]]
        out[stem] = float(scope[stem].mean())
        print(f"  {label:10s} mean {scope[stem].mean():+6.2f}% over {len(scope)} "
              f"(changed only: {changed[stem].mean():+6.2f}% over {len(changed)}; "
              f"better on {int((changed[stem] > 0).sum())}, worse on {int((changed[stem] < 0).sum())})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--silent-weight", type=float, default=3.0)
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="The `_versus_aqorn.csv` ablation_silent_aware cached "
                             "(default: that script's cache for --silent-weight).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--series-label", default=None)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()
    source = args.from_csv or SOURCE / f"ablation_silent_aware_w{args.silent_weight:g}_versus_aqorn.csv"
    if not source.exists():
        raise SystemExit(f"{source} not found: run ablation_silent_aware first")
    gains = improvements(pd.read_csv(source))
    gains.to_csv(HERE / f"{NAME}_w{args.silent_weight:g}_plot_data.csv", index=False)
    summarize(gains)
    for path in plot(gains, args.silent_weight, args.output_dir, args.series_label, args.stem):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
