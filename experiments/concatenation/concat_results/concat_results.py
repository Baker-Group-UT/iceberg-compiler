"""
concat_results.py -- AQORN at the outer and inner levels of the concatenated code, and what the
concatenation buys over one level.

(a) physical two-qubit gates and depth, % reduction against the concatenated baseline (concat_BB:
    naive at both levels), per AQORN placement: outer only, inner passes only, inner AQORN, both.
(b) at each circuit's calibrated noise (concat_BB keeps half its shots): LSR improvement and
    silent-error reduction against concat_BB, post-selection decoding.
(c) silent error per kept shot against the physical noise scale alpha (geometric mean over
    circuits): one level vs two, spread vs index order.
(d) the two decoders on the AQORN-at-both-levels circuit: shots kept and silent error.
Error bars: +-1 SD over circuits.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_results.concat_results \
        --from-csv results/concatenation/concat_eval.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
NAME = "concat_results"
REFERENCE = "concat_BB"
LEVEL_ARMS = [("concat_AB", "Outer AQORN"), ("concat_Bp", "Inner passes"),
              ("concat_BA", "Inner AQORN"), ("concat_AA", "Both levels"),
              ("concat_AA_T", "Both, w10T")]
SCALING_ARMS = [("single_base", "1 level, baseline"), ("single_w10", "1 level, AQORN"),
                ("concat_BB", "2 levels, baseline"), ("concat_AA", "2 levels, AQORN"),
                ("concat_AA_index", "2 levels, AQORN, index order")]


def _style(i: int):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    return {key: values[i % len(values)] for key, values in cycle.items()}


def _paired(frame: pd.DataFrame, column: str, *, lower_is_better: bool, label: str = "calibrated"):
    """Per-circuit % change of `column` against concat_BB, for every arm, at one alpha label."""
    sub = frame[frame["alpha_label"] == label]
    wide = sub.pivot_table(index="circuit_name", columns="arm", values=column)
    reference = wide[REFERENCE]
    if lower_is_better:
        return (reference.values[:, None] - wide.values) / reference.values[:, None] * 100, wide.columns
    return (wide.values - reference.values[:, None]) / reference.values[:, None] * 100, wide.columns


def _bars(ax, groups, arms, frame, title, ylabel):
    width = 0.8 / len(arms)
    x = np.arange(len(groups))
    for i, (arm, label) in enumerate(arms):
        means, sds = [], []
        for column, lower in groups.values():
            values, columns = _paired(frame, column, lower_is_better=lower)
            series = pd.DataFrame(values, columns=columns)[arm]
            means.append(series.mean())
            sds.append(series.std())
        ax.bar(x + (i - (len(arms) - 1) / 2) * width, means, width, yerr=sds, capsize=2,
               label=label, error_kw={"elinewidth": 0.6}, color=_style(i)["color"])
    ax.set_xticks(x, list(groups))
    ax.axhline(0, color="#636363", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def pooled_silent(frame: pd.DataFrame, arm: str, mode: str = "postselect") -> pd.Series:
    """Silent error per kept shot pooled over circuits (events / kept shots), by fixed alpha."""
    fixed = frame[(frame["alpha_label"] != "calibrated") & (frame["arm"] == arm)]
    kept = fixed[f"{mode}_lsr"]
    events = fixed[f"{mode}_silent"] * kept
    return events.groupby(fixed["alpha"]).sum() / kept.groupby(fixed["alpha"]).sum()


def panel_scaling(ax, frame: pd.DataFrame) -> None:
    for i, (arm, label) in enumerate(SCALING_ARMS):
        if arm not in set(frame["arm"]):
            continue
        silent = pooled_silent(frame, arm)
        silent = silent[silent > 0]
        style = _style(i)
        ax.plot(silent.index, silent.values, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], markersize=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Noise scale $\alpha$")
    ax.set_ylabel("Silent error per kept shot")
    ax.set_title("(c) Silent error vs noise (pooled)", loc="left")
    ax.legend(fontsize=8, frameon=False)


def panel_decoders(ax, frame: pd.DataFrame) -> None:
    sub = frame[(frame["alpha_label"] == "calibrated") & (frame["arm"] == "concat_AA")]
    values = {"Post-select": (sub["postselect_lsr"], sub["postselect_silent"]),
              "Erasure": (sub["erasure_lsr"], sub["erasure_silent"])}
    x = np.arange(2)
    for i, (label, (lsr, silent)) in enumerate(values.items()):
        style = _style(i)
        ax.bar(i - 0.2, lsr.mean(), 0.38, yerr=lsr.std(), capsize=2, color=style["color"],
               error_kw={"elinewidth": 0.6})
    ax.set_xticks(x - 0.2, list(values))
    ax.set_ylabel("Shots kept")
    ax.set_ylim(0, 1)
    twin = ax.twinx()
    for i, (label, (lsr, silent)) in enumerate(values.items()):
        twin.plot([i + 0.2], [silent.mean()], marker="s", color="#3b4c56", linestyle="none")
        twin.errorbar([i + 0.2], [silent.mean()], yerr=[silent.std()], color="#3b4c56",
                      capsize=2, elinewidth=0.6)
    twin.set_yscale("log")
    twin.set_ylabel("Silent per kept (squares)")
    ax.set_title("(d) Decoders, both levels AQORN", loc="left")


def draw(frame: pd.DataFrame, output_dir=None):
    from experiments.shared.figures import save_figure

    plt.style.use(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(7.14, 5.2))
    _bars(axes[0, 0], {"2q gates": ("two_qubit_gates", True), "Depth": ("depth", True)},
          LEVEL_ARMS, frame, "(a) Gates vs concatenated baseline", "Reduction (%)")
    axes[0, 0].legend(fontsize=8, frameon=False, ncol=2)
    _bars(axes[0, 1], {"LSR": ("postselect_lsr", False), "Silent error": ("postselect_silent", True)},
          LEVEL_ARMS, frame, "(b) Fidelity at calibrated noise", "Improvement (%)")
    panel_scaling(axes[1, 0], frame)
    panel_decoders(axes[1, 1], frame)
    paths = save_figure(fig, NAME, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, lower in (("two_qubit_gates", True), ("one_qubit_gates", True), ("depth", True),
                          ("postselect_lsr", False), ("postselect_silent", True),
                          ("postselect_tvd", True), ("erasure_lsr", False),
                          ("erasure_silent", True)):
        values, columns = _paired(frame, column, lower_is_better=lower)
        table = pd.DataFrame(values, columns=columns)
        for arm in columns:
            rows.append({"metric": column, "arm": arm, "mean_pct": table[arm].mean(),
                         "sd_pct": table[arm].std(), "median_pct": table[arm].median(),
                         "circuits": int(table[arm].notna().sum())})
    out = pd.DataFrame(rows)
    print(out.pivot_table(index="arm", columns="metric", values="mean_pct").round(1).to_string())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", default="results/concatenation/concat_eval.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.from_csv)
    summarize(frame)
    for path in draw(frame, Path(args.output_dir) if args.output_dir else None):
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
