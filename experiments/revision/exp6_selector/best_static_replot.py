"""
best_static_replot.py -- Experiment6a: Fig 8 with the comparator its text claims.

Fig 8 plots "worst static k, naive", while the caption and text claim
the selector beats even the *best* static code chosen with hindsight. A worst-static
curve cannot support that, so this replots Fig 8's own data (no new simulation) with the
best static code size, chosen with hindsight three ways -- each a stronger comparator
than the last, and each biased *towards* the static arm because it is a max over noisy
estimates:

    global       one k for the whole figure (every circuit, every noise ratio)
    per ratio    the best k for each noise-ratio column, shared by all circuits
    per point    the best k for each (circuit, ratio) -- an oracle over static codes

Given Experiment6b's CSV (`--aqorn-csv`), the same three comparators are built from AQORN's
planner at every static k, which isolates code-size *selection* from placement: the
selector against the best static k *with the same packing optimisations*.

Every series is LSR relative to naive k = 2 within the same (circuit, ratio), summarised
as the geometric mean over circuits with a +-1 standard-error band in log space -- Fig 8's
own convention (`_log_band`), so the numbers are comparable with the submitted figure.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.revision.common import FIG8_CSV, RESULTS, log, write_sidecar

NAME = "exp6a"
K_VALUES = (2, 4, 6, 8, 10)


def load(fig8_csv: Path, aqorn_csv: Optional[Path]) -> pd.DataFrame:
    from experiments.figures.resource_selection_headroom.resource_selection_headroom import (
        complete_circuits,
    )

    frame = complete_circuits(pd.read_csv(fig8_csv), K_VALUES)
    if aqorn_csv is not None and Path(aqorn_csv).exists():
        extra = pd.read_csv(aqorn_csv)
        extra = extra[extra["circuit"].isin(set(frame["circuit"]))]
        frame = pd.concat([frame, extra], ignore_index=True, sort=False)
    return frame


def per_row(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (circuit, ratio), every series' LSR divided by naive k=2's."""
    pivot = frame.pivot_table(index=["circuit", "ratio"], columns="series", values="lsr",
                              aggfunc="mean")
    oracle = (frame[frame["series"].isin(["oracle_cand", "selector"])]
              .groupby(["circuit", "ratio"])["lsr"].max())
    ref = pivot["naive_k2"].replace(0.0, np.nan)
    rel = pd.DataFrame(index=pivot.index)
    for column in pivot.columns:
        if column == "oracle_cand":
            continue
        rel[column] = pivot[column] / ref
    rel["oracle"] = oracle.reindex(pivot.index) / ref
    return rel


def static_comparators(rel: pd.DataFrame, prefix: str) -> Dict[str, pd.Series]:
    """worst / best-global / best-per-ratio / best-per-point over `<prefix>_k{k}` columns."""
    columns = [f"{prefix}_k{k}" for k in K_VALUES if f"{prefix}_k{k}" in rel.columns]
    if not columns:
        return {}
    # Only rows carrying the whole static ladder: a max or min over a partial ladder would
    # move with whichever arms happened to survive (a miscompile skip, a failed column).
    complete = rel[columns].notna().all(axis=1)
    ladder = rel.loc[complete, columns]
    if ladder.empty:
        return {}
    logs = np.log(ladder.clip(lower=1e-12))
    out: Dict[str, pd.Series] = {}
    out[f"{prefix}_k_worst"] = ladder.min(axis=1).reindex(rel.index)
    best_global = logs.mean().idxmax()
    out[f"{prefix}_k_best_global"] = ladder[best_global].reindex(rel.index).rename(best_global)
    per_ratio = []
    chosen = {}
    for ratio, block in logs.groupby(level="ratio"):
        means = block.mean()
        if not means.notna().any():
            continue
        column = means.idxmax()
        chosen[ratio] = column
        per_ratio.append(ladder.loc[block.index, column])
    out[f"{prefix}_k_best_per_ratio"] = pd.concat(per_ratio).reindex(rel.index)
    out[f"{prefix}_k_best_per_point"] = ladder.max(axis=1).reindex(rel.index)
    out[f"_{prefix}_rows"] = pd.Series({"complete_rows": int(complete.sum()),
                                        "rows": int(len(rel))})
    out[f"_{prefix}_choices"] = pd.Series({"global": best_global, **chosen})
    return out


def ladder(rel: pd.DataFrame, series: Dict[str, pd.Series]) -> pd.DataFrame:
    from experiments.figures.resource_selection_headroom.resource_selection_headroom import (
        _log_band,
    )

    rows = []
    for name, values in series.items():
        if name.startswith("_"):
            continue
        for ratio, block in values.groupby(level="ratio"):
            mean, low, high = _log_band(block.to_numpy())
            rows.append({"series": name, "ratio": ratio, "geomean": mean, "band_lo": low,
                         "band_hi": high, "circuits": int(block.notna().sum())})
    return pd.DataFrame(rows)


def paired(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """a against b, row by row: geometric-mean ratio, its log-SE, share of rows a >= b."""
    both = pd.concat([a, b], axis=1).dropna()
    both = both[(both.iloc[:, 0] > 0) & (both.iloc[:, 1] > 0)]
    if both.empty:
        return {"geomean_ratio": math.nan, "log_se": math.nan, "share_a_ge_b": math.nan,
                "rows": 0}
    logs = np.log(both.iloc[:, 0] / both.iloc[:, 1])
    return {
        "geomean_ratio": float(np.exp(logs.mean())),
        "log_se": float(logs.std(ddof=1) / math.sqrt(len(logs))) if len(logs) > 1 else 0.0,
        "share_a_ge_b": float((logs >= -1e-12).mean()),
        "rows": int(len(logs)),
    }


def plot(table: pd.DataFrame, frame: pd.DataFrame, out: Path, keys: Sequence[str],
         labels: Dict[str, str], name: str) -> List[Path]:
    import matplotlib.pyplot as plt

    from experiments.figures.resource_selection_headroom.resource_selection_headroom import (
        _log_ticks,
        _measured_positions,
        _plain,
        _plain_decimal,
    )
    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    plt.rcdefaults()
    use_style()
    ratios = sorted(table["ratio"].unique())
    x = _measured_positions(frame, ratios)
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 3.1))
    axis.axhline(1.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)
    for key in keys:
        block = table[table["series"] == key].set_index("ratio").reindex(ratios)
        if block["geomean"].isna().all():
            continue
        line, = axis.plot(x, block["geomean"], label=labels.get(key, key), markersize=3.2,
                          linewidth=1.2)
        axis.fill_between(x, block["band_lo"], block["band_hi"], color=line.get_color(),
                          alpha=0.15, linewidth=0)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    low, high = axis.get_ylim()
    axis.set_yticks(_log_ticks(low, high))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(_plain))
    axis.yaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_ylim(low, high)
    axis.set_ylabel("shots kept, vs. naive [[4,2,2]]")
    axis.set_xlabel("measured idling / movement noise ratio")
    tick_labels = [_plain_decimal(v) for v in x]
    tick_labels[0] += "\nmovement"
    tick_labels[-1] += "\nidling"
    axis.set_xticks(x)
    axis.set_xticklabels(tick_labels)
    axis.xaxis.set_minor_locator(plt.NullLocator())
    axis.legend(fontsize=7, loc="best", frameon=True, framealpha=1.0, facecolor="white",
                edgecolor="none")
    paths = save_figure(fig, name, Path(__file__), override=out, versioned=True)
    plt.close(fig)
    return paths


LABELS = {
    "oracle": "oracle over the selector's candidates",
    "selector": "AQORN selector",
    "aqorn_fixed": "AQORN, static $k=2$",
    "naive_k_best_per_point": "best static $k$ per point, naive",
    "naive_k_best_per_ratio": "best static $k$ per ratio, naive",
    "naive_k_best_global": "best static $k$ (one $k$), naive",
    "naive_k_worst": "worst static $k$, naive",
    "aqorn_k_best_per_point": "best static $k$ per point, AQORN",
    "aqorn_k_best_per_ratio": "best static $k$ per ratio, AQORN",
    "aqorn_k_best_global": "best static $k$ (one $k$), AQORN",
    "selector_rerun": "AQORN selector (re-run)",
    "compile_circuit": "AQORN full compiler (compile_circuit)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--fig8-csv", type=Path, default=FIG8_CSV)
    parser.add_argument("--aqorn-csv", type=Path, default=None,
                        help="Experiment6b output; adds best-static-AQORN comparators.")
    parser.add_argument("--extra-csv", type=Path, nargs="*", default=[],
                        help="More series in Fig 8's schema, e.g. Experiment6d's "
                             "compiled_selector.csv (series compile_circuit).")
    parser.add_argument("--out", type=Path, default=RESULTS / "exp6")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = load(args.fig8_csv, args.aqorn_csv)
    for extra in args.extra_csv:
        frame = pd.concat([frame, pd.read_csv(extra)], ignore_index=True, sort=False)
    rel = per_row(frame)
    series: Dict[str, pd.Series] = {
        "oracle": rel["oracle"], "selector": rel["selector"],
        "aqorn_fixed": rel["aqorn_fixed"],
    }
    series.update(static_comparators(rel, "naive"))
    aqorn_columns = [f"aqorn_k{k}" for k in K_VALUES if f"aqorn_k{k}" in rel.columns]
    if aqorn_columns:
        series.update(static_comparators(rel, "aqorn"))
        if "selector_rerun" in rel.columns:
            series["selector_rerun"] = rel["selector_rerun"]
    compiler = "compile_circuit" in rel.columns
    if compiler:
        series["compile_circuit"] = rel["compile_circuit"]
    table = ladder(rel, series)
    suffix = f"_{args.tag}" if args.tag else (
        "_with_compiler" if compiler else ("_with_aqorn" if aqorn_columns else ""))
    table_path = args.out / f"fig8_best_static{suffix}.csv"
    if table_path.exists():
        raise SystemExit(f"{table_path} exists; refusing to overwrite (use --tag)")
    table.to_csv(table_path, index=False)

    comparisons = {}
    for static in ("naive_k_worst", "naive_k_best_global", "naive_k_best_per_ratio",
                   "naive_k_best_per_point", "aqorn_k_best_global", "aqorn_k_best_per_ratio",
                   "aqorn_k_best_per_point"):
        if static in series:
            comparisons[f"selector_vs_{static}"] = paired(series["selector"], series[static])
            if "selector_rerun" in series:
                comparisons[f"selector_rerun_vs_{static}"] = paired(series["selector_rerun"],
                                                                     series[static])
    comparisons["oracle_vs_selector"] = paired(series["oracle"], series["selector"])
    if compiler:
        for static in ("aqorn_k_best_global", "aqorn_k_best_per_ratio", "aqorn_k_best_per_point",
                       "naive_k_best_global", "naive_k_worst"):
            if static in series:
                comparisons[f"compile_circuit_vs_{static}"] = paired(
                    series["compile_circuit"], series[static])
        comparisons["compile_circuit_vs_selector"] = paired(series["compile_circuit"],
                                                            series["selector"])
    choices = {key: value.to_dict() for key, value in series.items() if key.startswith("_")}
    pd.DataFrame(comparisons).T.to_csv(args.out / f"fig8_selector_comparisons{suffix}.csv")
    write_sidecar(table_path, experiment=NAME, started=started, status="complete",
                  settings={"fig8_csv": str(args.fig8_csv), "aqorn_csv": str(args.aqorn_csv)},
                  extra={"comparisons": comparisons, "choices": choices,
                         "circuits": int(frame["circuit"].nunique())})

    # At most five series: the house style's cycle repeats after five, and two lines sharing
    # a colour, marker and dash pattern cannot be told apart. The full compiler, when present,
    # takes slot 0 -- the AQORN position in every figure of the suite.
    keys = ["oracle", "selector", "aqorn_fixed", "naive_k_best_per_point", "naive_k_worst"]
    if aqorn_columns:
        keys = ["selector", "aqorn_k_best_global", "aqorn_k_best_per_point", "oracle",
                "naive_k_worst"]
    if compiler:
        keys = ["compile_circuit", "selector", "aqorn_k_best_global", "oracle", "naive_k_worst"]
    for path in plot(table, frame, args.out, keys, LABELS, f"fig8_best_static{suffix}"):
        log(NAME, f"wrote {path}")
    log(NAME, f"static choices: {choices}")
    for key, value in comparisons.items():
        log(NAME, f"{key}: geomean ratio {value['geomean_ratio']:.3f} "
                  f"(log-SE {value['log_se']:.3f}), selector >= static on "
                  f"{value['share_a_ge_b'] * 100:.0f}% of {value['rows']} rows")


if __name__ == "__main__":
    main()
