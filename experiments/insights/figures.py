"""
figures.py -- the figures for reports/why_aqorn_works.md and reports/lsr_vs_tvd.md.

Reads the CSVs `census.py`, `structure.py` and `lsr_tvd.py` wrote (results/insights/) and
draws, in the house style, versioned (never overwriting):

    insights_mechanisms     where each stage's 1q / 2q savings come from, by gadget mechanism
    insights_error_paths    selection-rate change vs the schedule's detected-error change, and
                            TVD change vs silent error per kept shot, every Fig 6 arm
    insights_lsr_tvd        how LSR and the error moved together, by kind of change (quadrant
                            shares), and the noise axis (alpha swept, frames)

    PYTHONPATH=. venv/bin/python -m experiments.insights.figures
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("results/insights")
STAGES = ("H commutation", "merging", "packing")
#: attribution group -> mechanism shown
MECHANISM = {"Hadamards (one lane)": "single-lane Hadamards removed",
             "Hadamards fused (HH)": "HH macros added",
             "two-qubit in patch": "two-qubit gadgets",
             "two-qubit across patches": "two-qubit gadgets",
             "two-qubit fused (CXT/CZT)": "two-qubit gadgets",
             "rotations": "other", "phase gates": "other", "Paulis": "other", "other": "other"}
MECHANISM_ORDER = ["single-lane Hadamards removed", "HH macros added", "two-qubit gadgets", "other"]


def mechanisms(attr: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure, use_style

    use_style()
    base = attr.drop_duplicates("circuit")
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 2.4), sharey=False)
    for ax, (key, total, label) in zip(axes, (("d_one", base["baseline_one"].sum(), "1q"),
                                              ("d_two", base["baseline_two"].sum(), "2q"))):
        frame = attr.assign(mechanism=attr["group"].map(MECHANISM))
        pivot = (-100 * frame.groupby(["stage", "mechanism"])[key].sum().unstack(fill_value=0.0)
                 / total).reindex(list(STAGES))
        pivot = pivot.loc[:, pivot.abs().max() > 0.05]
        x = np.arange(len(pivot))
        pos = np.zeros(len(pivot))
        neg = np.zeros(len(pivot))
        # One colour per mechanism in both panels: the style's cycle, indexed by mechanism.
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for mechanism in [m for m in MECHANISM_ORDER if m in pivot.columns]:
            values = pivot[mechanism].to_numpy()
            bottom = np.where(values >= 0, pos, neg)
            ax.bar(x, values, bottom=bottom, width=0.6, label=mechanism,
                   color=palette[MECHANISM_ORDER.index(mechanism) % len(palette)])
            pos += np.where(values >= 0, values, 0)
            neg += np.where(values < 0, values, 0)
        net = pivot.sum(axis=1).to_numpy()
        ax.plot(x, net, linestyle="none", marker="D", color="black", markersize=4,
                label="net saved")
        ax.axhline(0, color="#636363", linewidth=0.8)
        ax.set_xticks(x, list(pivot.index))
        ax.set_ylabel(f"{label} gates saved (% of corpus baseline)")
    handles, labels = axes[1].get_legend_handles_labels()
    more_h, more_l = axes[0].get_legend_handles_labels()
    merged = dict(zip(more_l + labels, more_h + handles))
    fig.legend(merged.values(), merged.keys(), loc="outside upper center", ncol=len(merged),
               frameon=False, fontsize=7)
    paths = save_figure(fig, "insights_mechanisms", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def error_paths(census: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure, use_style

    use_style()
    arms = census.copy()
    base = arms[arms["arm"] == "baseline"].set_index("circuit")
    rest = arms[arms["arm"] != "baseline"].copy()
    rest["d_log_lsr"] = np.log(rest["lsr"]) - rest["circuit"].map(np.log(base["lsr"]))
    rest["d_pred"] = -rest["alpha"] * (rest["E_detected"] - rest["circuit"].map(base["E_detected"]))
    resolved = rest[rest["baseline_tvd"] <= 0.95].copy()
    resolved["d_tvd"] = resolved["tvd"] - resolved["circuit"].map(base["tvd"])
    resolved["d_spk"] = (resolved["silent_per_kept"]
                         - resolved["circuit"].map(base["silent_per_kept"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 2.6))
    labels = {"h_only": "+ H commutation", "h_merge": "+ merging", "packed": "+ packing"}
    for arm, label in labels.items():
        sub = rest[rest["arm"] == arm]
        axes[0].plot(sub["d_pred"], sub["d_log_lsr"], linestyle="none", marker="o",
                     markersize=3, alpha=0.8, label=label)
        sub = resolved[resolved["arm"] == arm]
        axes[1].plot(sub["d_spk"], sub["d_tvd"], linestyle="none", marker="o", markersize=3,
                     alpha=0.8, label=label)
    lim = np.nanmax(np.abs(rest[["d_pred", "d_log_lsr"]].to_numpy()))
    axes[0].plot([-lim, lim], [-lim, lim], color="#bdbdbd", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel(r"$-\alpha\,\Delta E_{\rm detected}$ (schedule, vs baseline)")
    axes[0].set_ylabel(r"$\Delta \ln$ LSR (simulated, vs baseline)")
    axes[1].set_xlabel("change in silent error per kept shot (frames)")
    axes[1].set_ylabel("change in TVD (simulated)")
    for ax in axes:
        ax.axhline(0, color="#636363", linewidth=0.6)
        ax.axvline(0, color="#636363", linewidth=0.6)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    paths = save_figure(fig, "insights_error_paths", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


#: (source, comparison) -> label, in plotting order: the error-removing passes first.
PANELS = [
    (("fig6_standard", "+ H commutation"), "H commutation (Fig 6)"),
    (("fig6_standard", "+ merging"), "merging (Fig 6)"),
    (("fig6_v4", "+ packing"), "packing, v4 (Fig 6)"),
    (("fig6_standard", "+ packing"), "packing, standard (Fig 6)"),
    (("planner_k2", "silent weight up (k=2)"), "silent weight up, k=2"),
    (("planner_k4", "silent weight up (k=4)"), "silent weight up, k=4"),
    (("phase_diagram", "larger code (AQORN at each k)"), "larger code size"),
    (("density", "spread over more patches (baseline)"), "more, emptier patches"),
]
QUADRANTS = ("both better", "shots for accuracy", "accuracy for shots", "both worse")


def lsr_tvd(pairs: pd.DataFrame, sweep: pd.DataFrame, out_dir=None):
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure, use_style

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    rows = []
    for (source, comparison), label in PANELS:
        block = pairs[(pairs["source"] == source) & (pairs["comparison"] == comparison)]
        moved = block[block["quadrant"] != "flat"]
        shares = moved["quadrant"].value_counts(normalize=True)
        rows.append({"label": f"{label} (n={len(moved)})",
                     **{q: 100 * shares.get(q, 0.0) for q in QUADRANTS}})
    table = pd.DataFrame(rows).set_index("label").iloc[::-1]
    left = np.zeros(len(table))
    for quadrant in QUADRANTS:
        ax.barh(np.arange(len(table)), table[quadrant], left=left, label=quadrant, height=0.65)
        left += table[quadrant].to_numpy()
    ax.set_yticks(np.arange(len(table)), list(table.index), fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("pairs that moved (%)")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, 1.22))

    ax = axes[1]
    for circuit, block in sweep.groupby("circuit"):
        if circuit not in ("adder_n10", "bv_n19", "qft_n18"):
            continue
        for arm, style in (("baseline", ":"), ("packed", "-")):
            sub = block[(block["arm"] == arm) & (block["lsr"] > 1e-3)].sort_values("alpha")
            ax.plot(sub["lsr"], sub["silent_per_kept"], linestyle=style, marker="o",
                    markersize=3, label=f"{circuit}, {'AQORN' if arm == 'packed' else 'naive'}")
    ax.set_xscale("log")
    ax.set_xlabel("LSR (noise scale swept)")
    ax.set_ylabel("silent error per kept shot")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    paths = save_figure(fig, "insights_lsr_tvd", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--attribution", default=str(IN / "structure_attribution.csv"))
    parser.add_argument("--census", default=str(IN / "census.csv"))
    parser.add_argument("--pairs", default=str(IN / "lsr_tvd_pairs.csv"))
    parser.add_argument("--sweep", default=str(IN / "lsr_tvd_alpha_sweep.csv"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    for path in (mechanisms(pd.read_csv(args.attribution), args.output_dir)
                 + error_paths(pd.read_csv(args.census), args.output_dir)
                 + lsr_tvd(pd.read_csv(args.pairs), pd.read_csv(args.sweep), args.output_dir)):
        print(f"[insights.figures] saved {path}")


if __name__ == "__main__":
    main()
