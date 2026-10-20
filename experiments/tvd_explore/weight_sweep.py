"""
weight_sweep.py -- which silent-error weight, with the transversal in-patch CZ/XCX switch (wT).

Reads the Fig 6 runs `ablation_variant_w<W>T.csv` (each: the weight-W silent-aware planner with
TRANSVERSAL_INTRA_TWO_QUBIT on its merging stages, Fig 6's circuits, alpha, shot counts and seeds;
see `ablation_variant.py`) and reports, per weight:

    caption convention   per-circuit cumulative reduction over the three stages, unclipped,
                         averaged over circuits (TVD over Fig 6's resolved circuits only), as
                         `values.tex` quotes Fig 6 -- for TVD, LSR, 1q, 2q and depth
    paired vs w10T       per-circuit difference against the current standard, with a bootstrap
                         95% interval over circuits and win / loss counts
    split-half           the weight a rule picks on a random half of the circuits, scored on the
                         other half: how often each weight is picked, and the held-out gain over
                         w10T -- the check that a chosen weight is not fitted to noise

Rules: `max_tvd` (largest TVD reduction) and `knee` (largest LSR improvement among weights whose
TVD reduction is within `tolerance` points of the best).

    PYTHONPATH=. venv/bin/python -m experiments.tvd_explore.weight_sweep
    ... --output-dir results/tvd_explore/figures
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path("results/tvd_explore")
NAME = "weight_sweep"
COMPONENTS = ("h_commutation", "merging", "packing")
METRICS = {
    "tvd": "pct_reduction_logical_total_variation_distance",
    "lsr": "pct_improvement_logical_selection_rate",
    "two_q": "pct_reduction_two_qubit_gates",
    "one_q": "pct_reduction_one_qubit_gates",
    "depth": "pct_reduction_circuit_depth",
}
STANDARD = 10.0


def load(weights) -> Dict[float, pd.DataFrame]:
    out = {}
    for w in weights:
        path = HERE / f"ablation_variant_w{w:g}T.csv"
        if path.exists():
            out[float(w)] = pd.read_csv(path)
    return out


def per_circuit(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit: cumulative value of every metric; TVD NaN where Fig 6 withholds it."""
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    resolved = fig6._tvd_is_resolved(rows)
    out = {}
    for key, column in METRICS.items():
        frame = rows if key != "tvd" else rows.loc[resolved]
        wide = frame.pivot_table(index="circuit_name", columns="component", values=column)
        out[key] = wide[list(COMPONENTS)].sum(axis=1)
    table = pd.DataFrame(out)
    table["size_group"] = rows.drop_duplicates("circuit_name").set_index("circuit_name")["size_group"]
    return table


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 4000):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def curve(tables: Dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for w, t in sorted(tables.items()):
        row = {"weight": w, "circuits": len(t), "tvd_circuits": int(t["tvd"].notna().sum())}
        for key in METRICS:
            row[key] = float(t[key].mean())
        for group in ("small", "medium", "large"):
            sub = t[t["size_group"] == group]
            row[f"tvd_{group}"] = float(sub["tvd"].mean())
            row[f"lsr_{group}"] = float(sub["lsr"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired(tables: Dict[float, pd.DataFrame], rng) -> pd.DataFrame:
    ref = tables[STANDARD]
    rows = []
    for w, t in sorted(tables.items()):
        row = {"weight": w}
        for key in ("tvd", "lsr", "two_q", "depth"):
            diff = (t[key] - ref[key]).to_numpy()
            low, high = bootstrap_ci(diff, rng)
            row[f"{key}_diff"] = float(np.nanmean(diff))
            row[f"{key}_ci"] = f"[{low:+.2f}, {high:+.2f}]"
            row[f"{key}_better"] = int((diff > 1e-9).sum())
            row[f"{key}_worse"] = int((diff < -1e-9).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def choose(means: pd.DataFrame, rule: str, tolerance: float) -> float:
    if rule == "max_tvd":
        return float(means["tvd"].idxmax())
    best = means["tvd"].max()
    eligible = means[means["tvd"] >= best - tolerance]
    return float(eligible["lsr"].idxmax())


def split_half(tables: Dict[float, pd.DataFrame], rng, *, rule: str, tolerance: float,
               splits: int = 2000) -> pd.DataFrame:
    names = sorted(tables[STANDARD].index)
    groups = tables[STANDARD]["size_group"].reindex(names).to_numpy()
    picks: List[float] = []
    held_out_tvd, held_out_lsr = [], []
    for _ in range(splits):
        mask = np.zeros(len(names), dtype=bool)
        for g in np.unique(groups):                     # stratified by size group
            idx = np.flatnonzero(groups == g)
            mask[rng.choice(idx, size=len(idx) // 2, replace=False)] = True
        train = [n for n, m in zip(names, mask) if m]
        test = [n for n, m in zip(names, mask) if not m]
        means = pd.DataFrame({w: t.loc[train, ["tvd", "lsr"]].mean() for w, t in tables.items()}).T
        pick = choose(means, rule, tolerance)
        picks.append(pick)
        held_out_tvd.append(tables[pick].loc[test, "tvd"].mean() - tables[STANDARD].loc[test, "tvd"].mean())
        held_out_lsr.append(tables[pick].loc[test, "lsr"].mean() - tables[STANDARD].loc[test, "lsr"].mean())
    counts = pd.Series(picks).value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame({"picked_pct": counts}).assign(
        held_out_tvd_gain_vs_w10T=float(np.mean(held_out_tvd)),
        held_out_lsr_gain_vs_w10T=float(np.mean(held_out_lsr)))


def plot(means: pd.DataFrame, diffs: pd.DataFrame, output_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure

    style = Path(__file__).resolve().parents[1] / "shared" / "aqorn.mplstyle"
    plt.style.use(style)
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 2.6))
    for ax, (key, label) in zip(axes, (("tvd", "TVD reduction (%)"), ("lsr", "LSR improvement (%)"))):
        ax.plot(means["weight"], means[key], marker="o", markersize=4, label="all circuits")
        for group, marker in (("small", "s"), ("medium", "^"), ("large", "D")):
            ax.plot(means["weight"], means[f"{key}_{group}"], marker=marker, markersize=3,
                    linewidth=0.8, label=group)
        ax.axvline(STANDARD, color="#bdbdbd", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks(list(means["weight"]), [f"{w:g}" for w in means["weight"]])
        ax.set_xticks([], minor=True)
        ax.set_xlabel("Silent-error weight (transversal CZ/XCX on)")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--weights", nargs="+", type=float,
                        default=[3, 5, 7, 10, 12, 15, 20, 30])
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="knee rule: TVD points below the best still eligible")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    raw = load(args.weights)
    if STANDARD not in raw:
        raise SystemExit("ablation_variant_w10T.csv is required (the reference)")
    tables = {w: per_circuit(rows) for w, rows in raw.items()}
    means = curve(tables)
    pd.set_option("display.width", 220)
    print(f"[{NAME}] weights with Fig 6 runs: {sorted(tables)}")
    print("\nCaption convention (per-circuit cumulative, unclipped; TVD on resolved circuits):")
    print(means.round(2).to_string(index=False))
    diffs = paired(tables, rng)
    print("\nPaired against w10T (mean difference in points, bootstrap 95% CI, circuits better / worse):")
    print(diffs.to_string(index=False))
    for rule in ("max_tvd", "knee"):
        result = split_half(tables, rng, rule=rule, tolerance=args.tolerance)
        print(f"\nSplit-half ({rule}, tolerance {args.tolerance}): % of 2000 splits picking each weight, "
              f"and the held-out gain of the picked weight over w10T (points)")
        print(result.round(2).to_string())
    if not args.no_plot:
        for path in plot(means, diffs, Path(args.output_dir) if args.output_dir else None):
            print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
