"""Run from the exact-patches worktree:
    MPLBACKEND=Agg PYTHONPATH=. ../../../venv/bin/python <this file> <output dir>
"""
"""DRAFT (scratchpad only): the reference figures with log-ratio segments, so cell means are
geometric means and the placement bar is the same quantity in both figures."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from experiments.figures.density_ablation_sweep import density_references as dr

D = Path("experiments/figures/density_ablation_sweep")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
table = dr.arm_values(pd.read_csv(D / "density_ablation_sweep_split_std.csv"),
                      pd.read_csv(D / "density_ablation_sweep_split_std_recount.csv"),
                      pd.read_csv(D / "density_ablation_sweep_split_std_floors.csv"))


def log_reference_segments(table, which):
    """Per (circuit, n_codes, quantity): ln improvement factors; patch + placement = net exactly."""
    wide, info = dr._wide(table)
    n = info["num_qubits"]
    target = np.ceil(n / 2) if which == "tight" else n
    is_reference = wide.index.get_level_values("n_codes").to_numpy() == target.to_numpy()
    circuits = wide.index.get_level_values("circuit_name")

    def at_reference(column):
        per_circuit = wide.loc[is_reference, column].droplevel("n_codes")
        return pd.Series(circuits.map(per_circuit).to_numpy(dtype=float), index=wide.index)

    frames, dropped = [], {}
    for panel, (_, quantities, lower) in dr.PANELS.items():
        sign = -1.0 if lower else 1.0
        for quantity in quantities:
            ref, naive, packed = (at_reference((quantity, "h_merge")), wide[(quantity, "h_merge")],
                                  wide[(quantity, "packed")])
            with np.errstate(divide="ignore", invalid="ignore"):
                patch, place = sign * np.log(naive / ref), sign * np.log(packed / naive)
            frame = pd.DataFrame({"patch_count": patch, "placement": place, "net": patch + place})
            measurable = pd.Series(True, index=wide.index)
            if quantity == "tvd":
                measurable = (at_reference(("tvd_raw", "h_merge"))
                              >= dr.SIGNAL_OVER_FLOOR * at_reference(("floor", "h_merge"))).fillna(False)
            positive = (ref > 0) & (naive > 0) & (packed > 0)
            dropped[quantity] = int((measurable & ~positive).sum()), int(measurable.sum())
            frame = frame.where(measurable & positive)
            frame["panel"], frame["quantity"] = panel, quantity
            frames.append(frame.join(info).reset_index())
    return pd.concat(frames, ignore_index=True), dropped


FACTORS = [0.2, 0.25, 1 / 3, 0.5, 2 / 3, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 3, 4, 5, 8]


def _label(f):
    return f"{f:.2f}".rstrip("0").rstrip(".") + "\u00d7"


def style_log_axis(ax, label, low, high):
    ax.set_ylabel(label)
    span = max(high - low, np.log(1.2))
    ax.set_ylim(min(0.0, low) - 0.06 * span, max(0.0, high) + 0.10 * span)
    lo, hi = ax.get_ylim()
    spacing, ticks = (hi - lo) / 5.5, [0.0]
    for f in sorted([f for f in FACTORS if f > 1], key=lambda f: f):
        if np.log(f) <= hi and np.log(f) - ticks[-1] >= spacing:
            ticks.append(np.log(f))
    down = 0.0
    for f in sorted([f for f in FACTORS if f < 1], reverse=True):
        if np.log(f) >= lo and down - np.log(f) >= spacing:
            ticks.insert(0, np.log(f)); down = np.log(f)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_label(np.exp(t)) if abs(t) > 1e-12 else "1×" for t in ticks])
    ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot(which):
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    parts = [("patch_count", "patch count (naive at P vs reference)", "#9bd3f5"),
             ("placement", "placement (AQORN vs naive at P)", "#0097fd")]
    raw, dropped = log_reference_segments(table, which)
    segments = dr._with_bins(raw)
    centers = sorted(segments["plot_bin"].dropna().unique())
    fig, axes = dr._figure()
    for ax, (panel, (label, _, _)) in zip(axes, dr.PANELS.items()):
        block = segments[segments["panel"] == panel].dropna(subset=["net"])
        low, high = 0.0, np.log(1.2)
        for x, group, cell in dr._cells(block, centers, 4):
            means = cell.groupby("circuit_name")[["patch_count", "placement", "net"]].mean().mean()
            up = down = 0.0
            for column, _, color in parts:
                value = float(means[column])
                ax.bar(x, value, bottom=up if value >= 0 else down, width=dr.WIDTH, color=color,
                       edgecolor="white", alpha=0.9)
                up, down = (up + value, down) if value >= 0 else (up, down + value)
            net = float(means["net"])
            ax.plot([x - dr.WIDTH / 2, x + dr.WIDTH / 2], [net, net], color="black", linewidth=1.2,
                    linestyle="-", marker="", solid_capstyle="butt", zorder=5)
            ax.text(x, up, dr.GROUP_INITIAL[group], ha="center", va="bottom", fontsize=5,
                    color="#444444")
            low, high = min(low, down, net), max(high, up, net)
        style_log_axis(ax, label, low, high)
    handles = [mpatches.Patch(color=c, alpha=0.9, label=l) for _, l, c in parts]
    handles.append(mlines.Line2D([], [], color="black", linewidth=1.2, linestyle="-", marker="",
                                 label="net"))
    reference = "naive tight packing" if which == "tight" else "naive sparse packing"
    paths = dr._finish(fig, axes, centers, handles,
                       f"improvement factor over the {reference} (geometric mean)",
                       f"density_vs_naive_{which}_geomean", OUT)
    return raw, dropped, paths


pd.set_option("display.width", 250)
for which in ("tight", "sparse"):
    raw, dropped, paths = plot(which)
    print(f"== vs {which}: wrote {[p.name for p in paths]}")
    print("   dropped for a non-positive value (of measurable points):",
          {q: f"{d}/{m}" for q, (d, m) in dropped.items() if d})
    f = dr._with_bins(raw)
    per = f.groupby(["panel", "plot_bin", "circuit_name"])[["patch_count", "placement", "net"]].mean().reset_index()
    g = per.groupby(["panel", "plot_bin"])[["patch_count", "placement", "net"]].mean()
    g = np.exp(g)
    g.index = pd.MultiIndex.from_tuples([(p, round(1 / b, 2)) for p, b in g.index], names=["panel", "density"])
    print(g.round(2).unstack("density").to_string())
