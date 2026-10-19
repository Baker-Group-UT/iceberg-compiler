"""
density_ablation_matched.py
---------------------------

Re-plot of the standard density ablation with the packing segment split in two.

The packed arm always uses ceil(n/2) patches, whatever density the trivial arms are given
(see `density_ablation_sweep.py`, `--standard`). So the submitted "+ Packing" segment is
placement quality *plus* a patch-count change. This script separates them, from the CSV
alone -- no new simulation:

    baseline -> + H commutation -> + Merging -> + Patch count -> + Placement

  * **+ Patch count**: the h_merge arm at the packed arm's patch count minus the h_merge arm
    at this density. The matched arm is not re-simulated: it is the h_merge arm already
    simulated in the same circuit's row whose `n_codes == packed_patches` (the densest
    point, ceil(n/2) patches; for density-1 rows the packed arm *is* trivial at n patches,
    so the row itself). Same alpha, same repetitions per circuit, as in the sweep.
  * **+ Placement**: packed minus that matched h_merge. Packing's own contribution at equal
    patch count.

Components are stage deltas against the same baseline (as in the CSV) and are stacked with
`stacked_percent`, so the top of every bar equals the full AQORN improvement, unchanged
from the submitted-style figure. Only how the packing share is split changes.

Other changes (see `DENSITY_ABLATION_MATCHED_SPEC.md`):
  * TVD panel only for circuits of <= `--tvd-max-qubits` qubits (default 18; the floor
    swamps the signal beyond that) and not for rows whose baseline TVD is saturated.
  * bars carry their circuit count; bins with fewer than `--min-circuits` are not drawn.
  * a summary CSV with mean / median / bootstrap 95% CI per (bin, component, metric).
  * absolute baseline / matched / packed LSR medians are printed per bin.

    python -m experiments.figures.density_ablation_sweep.density_ablation_matched \
        --csv experiments/figures/density_ablation_sweep/density_ablation_sweep_standard.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from experiments.shared.figures import FULL_WIDTH, save_figure, stacked_percent, use_style

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "density_ablation_sweep_standard.csv"
SIZE_GROUPS = ["small", "medium", "large"]
GROUP_INITIAL = {"small": "S", "medium": "M", "large": "L"}

STAGES = [
    ("h_commutation", "H commutation", "#3b4c56"),
    ("merging", "+ Merging", "#cdc2c0"),
    ("patch_count", "+ Patch count", "#9bd3f5"),
    ("placement", "+ Placement", "#0097fd"),
]

# (label, metric key). Gate panel averages 1q and 2q, as the submitted figure does.
PANELS = [
    ("#gates (1q, 2q avg)", ("one_qubit_gates", "two_qubit_gates")),
    ("depth", ("depth",)),
    ("LSR", ("lsr",)),
    ("TVD (n ≤ {tvd_max})", ("tvd",)),
    ("silent error / kept shot", ("silent_per_kept",)),
]
LOWER_IS_BETTER = {"one_qubit_gates", "two_qubit_gates", "depth", "tvd", "silent_per_kept"}
ARMS = ("baseline", "h_only", "h_merge", "packed")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _cum(metric: str, base: pd.Series, arm: pd.Series) -> pd.Series:
    """Cumulative percent improvement / reduction of `arm` over `base`."""
    base = base.astype(float)
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    out = 100.0 * sign * (arm.astype(float) - base) / base
    return out.where(base > 0)


def _column(arm: str, metric: str) -> str:
    return f"{arm}_{metric}"


def build_stages(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """
    One row per (circuit, n_codes) with stage deltas per metric:
    `<metric>__<stage>` for the four stages, and the matched arm's absolute values.
    """
    rows = df[df["component"] == "packing"].copy()  # every arm's columns are on every row
    rows["n_codes"] = rows["n_codes"].astype(int)
    rows["packed_patches"] = rows["packed_patches"].astype(int)

    # The matched h_merge arm: same circuit, row with n_codes == packed_patches.
    keyed = rows.set_index(["circuit_name", "n_codes"])
    match_keys = list(zip(rows["circuit_name"], rows["packed_patches"]))
    present = [k in keyed.index for k in match_keys]
    rows["matched_found"] = present
    matched = keyed.reindex(match_keys)
    for metric in metrics:
        rows[f"matched_{metric}"] = matched[_column("h_merge", metric)].to_numpy()

    for metric in metrics:
        base = rows[_column("baseline", metric)]
        c_h = _cum(metric, base, rows[_column("h_only", metric)])
        c_m = _cum(metric, base, rows[_column("h_merge", metric)])
        c_x = _cum(metric, base, rows[f"matched_{metric}"])
        c_p = _cum(metric, base, rows[_column("packed", metric)])
        rows[f"{metric}__h_commutation"] = c_h
        rows[f"{metric}__merging"] = c_m - c_h
        rows[f"{metric}__patch_count"] = c_x - c_m
        rows[f"{metric}__placement"] = c_p - c_x
    rows["density"] = rows["num_qubits"] / rows["n_codes"]
    return rows


def assign_bins(rows: pd.DataFrame, n_bins: int = 6) -> pd.DataFrame:
    rows = rows.copy()
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows["plot_bin"] = pd.cut(
        rows["fraction_codes_used"], bins=edges, labels=centers, include_lowest=True
    ).astype(float)
    return rows


def applicable(rows: pd.DataFrame, metric: str, tvd_max: int) -> pd.DataFrame:
    """Rows a metric may be drawn from."""
    ok = rows["matched_found"]
    if metric == "tvd":
        ok = ok & (rows["num_qubits"] <= tvd_max) & (rows["baseline_tvd"] < 0.95)
    return rows[ok]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def boundaries(block: pd.DataFrame, metrics: Sequence[str]) -> Optional[np.ndarray]:
    """Mean over circuits (and over `metrics`) of each circuit's stacked boundaries."""
    per_metric = []
    for metric in metrics:
        cols = [f"{metric}__{stage}" for stage, _, _ in STAGES]
        values = block.groupby("circuit_name")[cols].mean().fillna(0.0)
        if values.empty:
            return None
        per_metric.append(np.asarray([stacked_percent(r) for r in values.to_numpy()]).mean(axis=0))
    return np.mean(per_metric, axis=0)


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(rows: pd.DataFrame, tvd_max: int) -> pd.DataFrame:
    out: List[Dict] = []
    metrics = sorted({m for _, ms in PANELS for m in ms} | {"silent_per_kept"})
    for metric in metrics:
        sub = applicable(rows, metric, tvd_max)
        for (group, bin_), block in sub.groupby(["size_group", "plot_bin"]):
            circuits = block.groupby("circuit_name")
            for stage, _, _ in STAGES:
                per_circuit = circuits[f"{metric}__{stage}"].mean().to_numpy(dtype=float)
                per_circuit = per_circuit[~np.isnan(per_circuit)]
                lo, hi = bootstrap_ci(per_circuit)
                out.append({
                    "metric": metric, "size_group": group, "density": 1.0 / bin_,
                    "stage": stage, "n_circuits": len(per_circuit),
                    "mean": per_circuit.mean() if len(per_circuit) else np.nan,
                    "median": np.median(per_circuit) if len(per_circuit) else np.nan,
                    "ci_lo": lo, "ci_hi": hi,
                })
    return pd.DataFrame(out)


def absolute_lsr(rows: pd.DataFrame) -> pd.DataFrame:
    sub = rows[rows["matched_found"]]
    grouped = sub.groupby("plot_bin")
    return pd.DataFrame({
        "density": 1.0 / grouped["fraction_codes_used"].mean(),
        "n_rows": grouped.size(),
        "baseline_lsr_med": grouped["baseline_lsr"].median(),
        "h_merge_lsr_med": grouped["h_merge_lsr"].median(),
        "matched_lsr_med": grouped["matched_lsr"].median(),
        "packed_lsr_med": grouped["packed_lsr"].median(),
        "patch_ratio_mean": (sub["n_codes"] / sub["packed_patches"]).groupby(sub["plot_bin"]).mean(),
    })


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot(rows: pd.DataFrame, tvd_max: int, min_circuits: int, stem: str,
         out_dir: Optional[Path]) -> List[Path]:
    centers = sorted(rows["plot_bin"].dropna().unique())
    bin_step, bar_width = 2.6, 0.82
    offsets = np.array([-0.84, 0.0, 0.84])
    use_style()
    fig, axes = plt.subplots(3, 2, figsize=(FULL_WIDTH, 1.55 * 3 + 1.25), sharex=True)
    axes = axes.ravel()
    axes[-1].set_visible(False)
    plotted: List[Tuple] = []

    for ax, (label, metrics) in zip(axes[:len(PANELS)], PANELS):
        label = label.format(tvd_max=tvd_max)
        sub = applicable(rows, metrics[0], tvd_max)
        top = 0.0
        for b_idx, center in enumerate(centers):
            for g_idx, group in enumerate(SIZE_GROUPS):
                block = sub[(sub["plot_bin"] == center) & (sub["size_group"] == group)]
                if block["circuit_name"].nunique() < min_circuits:
                    continue
                bounds = boundaries(block, metrics)
                if bounds is None:
                    continue
                x = b_idx * bin_step + offsets[g_idx]
                for i, (_, _, color) in enumerate(STAGES):
                    ax.bar(x, bounds[i + 1] - bounds[i], bottom=bounds[i], width=bar_width,
                           color=color, edgecolor="white", alpha=0.85)
                top = max(top, float(bounds[-1]))
                plotted.append((label, group, center, block["circuit_name"].nunique()))
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0.0, top=max(top, 1.0) * 1.06)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        small = plt.rcParams["font.size"] * 0.62
        low, high = ax.get_ylim()
        for b_idx, center in enumerate(centers):
            for g_idx, group in enumerate(SIZE_GROUPS):
                block = sub[(sub["plot_bin"] == center) & (sub["size_group"] == group)]
                if block["circuit_name"].nunique() < min_circuits:
                    continue
                bounds = boundaries(block, metrics)
                if bounds is None:
                    continue
                ax.text(b_idx * bin_step + offsets[g_idx], float(bounds[-1]) + 0.012 * (high - low),
                        GROUP_INITIAL[group], ha="center", va="bottom", fontsize=small,
                        color="#444444", zorder=5)

    ticks = [i * bin_step + offsets[1] for i in range(len(centers))]
    for ax in axes[3:5]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{1.0 / c:.2f}" for c in centers])
    fig.supxlabel("trivial-arm packing density")
    axes[-2].set_xlim(-1.32, (len(centers) - 1) * bin_step + 1.32)
    fig.legend(
        handles=[mpatches.Patch(color=c, alpha=0.85, label=l) for _, l, c in STAGES],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(STAGES), frameon=False,
        fontsize=plt.rcParams["font.size"] * 0.9, columnspacing=1.2, handlelength=1.4,
        handletextpad=0.4, borderpad=0.0,
    )
    fig.supylabel("percent improvement / reduction")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.90))
    paths = save_figure(fig, stem, Path(__file__), out_dir, versioned=True)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--tvd-max-qubits", type=int, default=18)
    ap.add_argument("--min-circuits", type=int, default=4,
                    help="draw a bar only if its (bin, size group) has this many circuits")
    ap.add_argument("--output-dir", type=Path, default=HERE / "matched_draft")
    ap.add_argument("--stem", default="density_ablation_matched")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metrics = sorted({m for _, ms in PANELS for m in ms} | {"lsr", "silent_per_kept"})
    rows = assign_bins(build_stages(df, metrics))
    missing = int((~rows["matched_found"]).sum())
    print(f"[matched] {len(rows)} (circuit, density) rows; {missing} without a matched arm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, args.tvd_max_qubits)
    summary_path = args.output_dir / f"{args.stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[matched] wrote {summary_path}")

    print("\nAbsolute LSR (medians) and patch ratio, per density bin:")
    print(absolute_lsr(rows).round(3).to_string())

    for path in plot(rows, args.tvd_max_qubits, args.min_circuits, args.stem, args.output_dir):
        print(f"[matched] wrote {path}")


if __name__ == "__main__":
    main()
