"""
selection_weight.py -- Experiment6f: the selector's ranking weight, swept on Fig 8's candidates.

`AqornStandard.selection_weight` prices a silent error when `compile_circuit` ranks its
candidates (every code size's planned and naive placement) on the standard objective,
detected + w x silent; the planners' own search weights are separate and unchanged. Fig 8
under the standard (`resource_selection_headroom --standard`) priced and simulated every
candidate the selector considered -- 10 per (circuit, noise ratio) column, each with its model
components and its measured selection rate (LSR) and silent error per kept shot (frames) -- so
re-ranking them at any weight needs no new simulation.

For each weight: the candidate each column would pick, and its measured LSR and silent error
per kept shot relative to naive [[4,2,2]] (geometric mean over 65 columns), the code sizes
picked, and how often the pick is the measured best. Against it: the measured trade-off frontier
of the same candidates (per column, argmin of -ln LSR + lambda x ln silent-per-kept, lambda
swept), the two measured oracles, and the standard planner at k = 2 alone.

    PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.selection_weight
Writes results/revision/exp6/selection_weight[_vN].csv and the figure
results/figures/selection_weight[_vN].{svg,pdf}.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("experiments/figures/resource_selection_headroom/resource_selection_headroom_standard.csv")
OUT = Path("results/revision/exp6")
WEIGHTS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1e4, math.inf)
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)


def load(path: Path):
    frame = pd.read_csv(path)
    candidates = frame[frame["series"] == "oracle_cand"].copy()
    reference = frame[frame["series"] == "naive_k2"].set_index(["circuit", "ratio"])
    fixed = frame[frame["series"] == "aqorn_fixed"].set_index(["circuit", "ratio"])
    candidates["spk"] = candidates["silent_per_kept"].clip(lower=1e-6)
    return candidates, reference, fixed


def pick_model(block: pd.DataFrame, weight: float) -> pd.Series:
    """The candidate `compile_circuit` keeps at this ranking weight (ties: fewer detected)."""
    if math.isinf(weight):
        key = list(zip(block["objective_silent"], block["objective_detected"]))
    else:
        key = list(zip(block["objective_detected"] + weight * block["objective_silent"],
                       block["objective_detected"]))
    return block.iloc[min(range(len(block)), key=lambda i: key[i])]


def pick_measured(block: pd.DataFrame, lam: float) -> pd.Series:
    """The candidate a measured exchange rate prefers: -ln LSR + lambda ln silent-per-kept."""
    lsr = np.log(block["lsr"].clip(lower=1e-6).to_numpy())
    spk = np.log(block["spk"].to_numpy())
    score = spk if math.isinf(lam) else -lsr + lam * spk
    return block.iloc[int(np.argmin(score))]


def sweep(candidates, reference, picker, values, label):
    rows = []
    for value in values:
        picks = []
        for key, block in candidates.groupby(["circuit", "ratio"]):
            chosen = picker(block, value)
            ref = reference.loc[key]
            best_lsr = block["lsr"].max()
            best_spk = block["spk"].min()
            picks.append({
                "log_lsr": math.log(max(chosen["lsr"], 1e-6) / ref["lsr"]),
                "log_spk": math.log(chosen["spk"] / max(ref["silent_per_kept"], 1e-6)),
                "k": int(chosen["n_code"]) - 2,
                "best_on_lsr": chosen["lsr"] >= best_lsr - 1e-12,
                "best_on_spk": chosen["spk"] <= best_spk + 1e-12,
            })
        picks = pd.DataFrame(picks)
        rows.append({"curve": label, "value": value,
                     "shots_x": math.exp(picks["log_lsr"].mean()),
                     "silent_per_kept_x": math.exp(picks["log_spk"].mean()),
                     "k_picked": json.dumps(picks["k"].value_counts().sort_index().to_dict()),
                     "measured_best_on_shots": int(picks["best_on_lsr"].sum()),
                     "measured_best_on_accuracy": int(picks["best_on_spk"].sum()),
                     "columns": len(picks)})
    return pd.DataFrame(rows)


def references(candidates, reference, fixed) -> pd.DataFrame:
    keys = list(candidates.groupby(["circuit", "ratio"]).groups)
    ref = reference.loc[keys]
    fix = fixed.loc[keys]
    best_lsr = candidates.groupby(["circuit", "ratio"])["lsr"].max().loc[keys]
    best_spk = candidates.groupby(["circuit", "ratio"])["spk"].min().loc[keys]

    def gm(x):
        return float(np.exp(np.log(x).mean()))

    return pd.DataFrame([
        {"curve": "naive [[4,2,2]]", "shots_x": 1.0, "silent_per_kept_x": 1.0},
        {"curve": "standard planner at k=2", "shots_x": gm(fix["lsr"] / ref["lsr"]),
         "silent_per_kept_x": gm(fix["silent_per_kept"].clip(lower=1e-6)
                                 / ref["silent_per_kept"].clip(lower=1e-6))},
        {"curve": "shots oracle (LSR only)", "shots_x": gm(best_lsr / ref["lsr"]),
         "silent_per_kept_x": float("nan")},
        {"curve": "accuracy oracle (silent per kept only)", "shots_x": float("nan"),
         "silent_per_kept_x": gm(best_spk / ref["silent_per_kept"].clip(lower=1e-6))},
    ])


def plot(model: pd.DataFrame, frontier: pd.DataFrame, refs: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure, use_style

    use_style()
    fig, ax = plt.subplots(figsize=(3.487, 2.8))
    ax.plot(frontier["silent_per_kept_x"], frontier["shots_x"], marker="s", markersize=3,
            linestyle="--", label="best trade, measured")
    ax.plot(model["silent_per_kept_x"], model["shots_x"], marker="o", markersize=4,
            label="selector, ranking weight w")
    for _, row in model.iterrows():
        if row["value"] in (0.0, 10.0, 100.0, 1000.0) or math.isinf(row["value"]):
            text = "w=inf" if math.isinf(row["value"]) else f"w={row['value']:g}"
            ax.annotate(text, (row["silent_per_kept_x"], row["shots_x"]), fontsize=7,
                        xytext=(4, 2), textcoords="offset points")
    for _, row in refs.dropna().iterrows():
        ax.plot(row["silent_per_kept_x"], row["shots_x"], marker="D", markersize=5,
                linestyle="none", label=row["curve"])
    # Both axes are ratios to naive [[4,2,2]], so log scales (equal steps = equal factors).
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("silent error per kept shot / naive")
    ax.set_ylabel("shots kept / naive")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    paths = save_figure(fig, "selection_weight", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    candidates, reference, fixed = load(Path(args.source))
    model = sweep(candidates, reference, pick_model, WEIGHTS, "selector (model ranking)")
    frontier = sweep(candidates, reference, pick_measured, LAMBDAS, "measured frontier")
    refs = references(candidates, reference, fixed)
    OUT.mkdir(parents=True, exist_ok=True)
    path, version = OUT / "selection_weight.csv", 2
    while path.exists():
        path, version = OUT / f"selection_weight_v{version}.csv", version + 1
    pd.concat([model, frontier, refs]).to_csv(path, index=False)
    pd.set_option("display.width", 220)
    print(f"[selection_weight] {candidates.groupby(['circuit', 'ratio']).ngroups} columns, "
          f"{len(candidates)} candidates; wrote {path}")
    for frame in (model, frontier, refs):
        print(frame.round(3).to_string(index=False))
    if not args.no_plot:
        for out in plot(model, frontier, refs, args.output_dir):
            print(f"[selection_weight] saved {out}")


if __name__ == "__main__":
    main()
