"""
noise_ratio_packers.py -- packing gains across machines: how AQORN and the partitioner baselines
fare against naive as the balance between idle and distance (movement) noise changes.

Reads `experiments.optimality.noise_ratio_packers`' CSV (Fig 6's 71 circuits x 7 idle/distance
ratios, 1/64 .. 64; ratio 64 is Fig 6's machine). At every ratio the noise level is calibrated so
naive keeps the share of shots it keeps on Fig 6's machine (`cal`), so each ratio is compared at
the same post-selection pressure. Three stacked panels over the ratio (log axis): (a) logical
error per kept shot, (b) TVD (circuits whose naive TVD is above zero and below saturation),
(c) shots kept, each as the geometric mean of arm / naive over circuits with a 95% bootstrap
interval (resampling circuits).

Arms: AQORN (the compiler), interaction partition + AQORN's descent (a cost-model-blind seed
searched on AQORN's objective), interaction partition alone (the seed, cost-model blind).

    PYTHONPATH=. venv/bin/python -m experiments.figures.noise_ratio.noise_ratio_packers \
        --from-csv results/noise_ratio_packers/noise_ratio_packers_final.csv [--output-dir DIR]
Writes noise_ratio_packers_vN.{svg,pdf} (versioned).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {"aqorn": dict(label="AQORN", color="#0096ff", marker="o", dx=1.0),
        "interaction_partition_descent": dict(label="IP + AQORN descent", color="#2b3a47",
                                              marker="s", dx=1.12),
        "interaction_partition": dict(label="IP (seed only)", color="#9a9a9a", marker="v",
                                      dx=0.89)}
#: Naive's TVD at or above this means the proxy's ideal output is near uniform: no signal.
SATURATED_TVD = 0.95


def ratios(frame: pd.DataFrame, alpha: str, boots: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ratio, part in frame.groupby("ratio"):
        for metric in ("ler", "tvd", "lsr"):
            wide = part.pivot_table(index="circuit", columns="arm", values=f"{metric}_{alpha}")
            if metric == "tvd":
                naive = wide["naive"]
                wide = wide[(naive > 0) & (naive < SATURATED_TVD)]
            for arm in ARMS:
                r = np.log(wide[arm] / wide["naive"]).replace([np.inf, -np.inf], np.nan).dropna()
                draws = [r.values[rng.integers(0, len(r), len(r))].mean() for _ in range(boots)]
                lo, hi = np.exp(np.percentile(draws, [2.5, 97.5]))
                rows.append(dict(ratio=ratio, metric=metric, arm=arm, gm=float(np.exp(r.mean())),
                                 lo=float(lo), hi=float(hi), n=len(r)))
    return pd.DataFrame(rows)


def plot(tab: pd.DataFrame, out_dir=None, note=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 4.4), sharex=True)
    labels = {"ler": "logical error / naive", "tvd": "TVD / naive", "lsr": "shots kept / naive"}
    for ax, metric in zip(axes, ("ler", "tvd", "lsr")):
        part = tab[tab.metric == metric]
        for arm, st in ARMS.items():
            s = part[part.arm == arm].sort_values("ratio")
            ax.errorbar(s.ratio * st["dx"], s.gm, yerr=[s.gm - s.lo, s.hi - s.gm],
                        color=st["color"], marker=st["marker"], markersize=3.5, capsize=1.5,
                        lw=1.1, label=st["label"])
        ax.axhline(1, color="#cccccc", lw=0.6, zorder=0)
        ax.axvline(64, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
        ax.set_ylabel(labels[metric])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}x"))
    ax = axes[-1]
    ax.set_xscale("log", base=2)
    ticks = sorted(tab.ratio.unique())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"1/{1 / v:g}" if v < 1 else (f"{v:g}\n(Fig 6)" if v == 64 else f"{v:g}")))
    ax.set_xlabel("idle / distance noise ratio ($\\leftarrow$ movement dominates)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=6.5, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.55, 1.0), handlelength=1.6, columnspacing=1.0)
    if note:
        fig.suptitle(note, fontsize=6, color="#b00000", y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = save_figure(fig, "noise_ratio_packers", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", required=True)
    parser.add_argument("--alpha", default="cal", choices=("cal", "fig6", "a1"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--note", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.from_csv)
    tab = ratios(frame, args.alpha)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))
    out = Path(args.from_csv).with_name(Path(args.from_csv).stem + f"_figure_{args.alpha}.csv")
    tab.to_csv(out, index=False)
    for path in plot(tab, args.output_dir, args.note):
        print(f"[noise_ratio_packers figure] saved {path}")


if __name__ == "__main__":
    main()
