"""
options.py -- what it costs to make a solo patch's gauge check available, and what a
sparser packing (every qubit solo) gains from [[4,1,2]] decoding. Reads the CSVs that
`circuit_level.py` writes; no simulation here.

Options for a solo patch, all decoded on the same footing (relative to the shipped arm
decoded as [[4,2,2]]):

    shipped, [[4,1,2]]   decoding only; the check exists only if the spare's frame survived
    fix-up,  [[4,1,2]]   SOLO_GAUGE_PARITY_FIX: one targeted H on the spare before readout
                         when the patch received an odd number of HH
    targeted, [[4,1,2]]  SOLO_H_TARGETED: every solo H as a targeted single-lane H

A variant whose translated circuit equals the shipped one (nothing to fix) is not
re-simulated: `circuit_level` marks it `identical_to_base` and this takes the shipped row.

Density: Fig 6's baseline translation of every circuit up to 18 qubits under the tight
packing and under one qubit per patch (all patches solo), each decoded both ways.

Run (from iceberg-compiler/):
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.options \\
        --main-csv results/solo_gauge/circuit_level.csv \\
        --variants-csv results/solo_gauge/circuit_level_variants.csv \\
        --density-csv results/solo_gauge/circuit_level_density.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TVD_MAX_QUBITS = 18
METRICS = ("lsr_422", "lsr_412", "tvd_422", "tvd_412", "silent_per_kept_422",
           "silent_per_kept_412", "one_qubit_gates", "two_qubit_gates", "depth",
           "gauge_caught", "gauge_harmless", "d_tvd_boot_sd")


def resolve_variants(main: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """Fill identical-to-base variant rows from their base arm's row."""
    out = variants.copy()
    base_rows = main.set_index(["name", "arm"])
    for i, row in out.iterrows():
        if not bool(row.get("identical_to_base", False)):
            continue
        base = row["arm"].split("_")[0]
        source = base_rows.loc[(row["name"], base)]
        for column in METRICS:
            if column in source:
                out.at[i, column] = source[column]
    return out


def option_table(main: pd.DataFrame, variants: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    Per circuit: each option's metrics relative to `base` decoded as [[4,2,2]]. Only the
    circuits the variant table covers, so every option is averaged over the same set.
    """
    ref = main[main["arm"] == base].set_index("name")
    ref = ref[ref.index.isin(set(variants["name"]))]
    rows = []
    for name, r in ref.iterrows():
        options = {"shipped": (r, "412")}
        for tag in ("fix", "tH"):
            v = variants[(variants["arm"] == f"{base}_{tag}") & (variants["name"] == name)]
            if len(v):
                options[tag] = (v.iloc[0], "412")
        for option, (row, rule) in options.items():
            rows.append({
                "name": name, "num_qubits": int(r["num_qubits"]), "base": base,
                "option": option,
                "check": int(row["checks"]) > 0,
                "changed_circuit": option != "shipped" and not bool(row.get("identical_to_base", False)),
                "d_lsr_pct": 100 * (row[f"lsr_{rule}"] - r["lsr_422"]) / r["lsr_422"],
                "d_tvd_pct": 100 * (row[f"tvd_{rule}"] - r["tvd_422"]) / r["tvd_422"],
                "d_silent_pct": 100 * (row[f"silent_per_kept_{rule}"] - r["silent_per_kept_422"])
                / r["silent_per_kept_422"],
                "d_2q_pct": 100 * (row["two_qubit_gates"] - r["two_qubit_gates"]) / r["two_qubit_gates"],
                "d_1q_pct": 100 * (row["one_qubit_gates"] - r["one_qubit_gates"]) / r["one_qubit_gates"],
            })
    return pd.DataFrame(rows)


def summarize_options(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (base, option), block in table.groupby(["base", "option"], sort=False):
        narrow = block[block["num_qubits"] <= TVD_MAX_QUBITS]
        rows.append({
            "base": base, "option": option, "circuits": len(block),
            "check_available": int(block["check"].sum()),
            "circuit_changed": int(block["changed_circuit"].sum()),
            "mean_d_lsr_pct": block["d_lsr_pct"].mean(),
            "mean_d_tvd_pct_n<=18": narrow["d_tvd_pct"].mean(),
            "median_d_tvd_pct_n<=18": narrow["d_tvd_pct"].median(),
            "tvd_better_n<=18": int((narrow["d_tvd_pct"] < 0).sum()),
            "tvd_worse_n<=18": int((narrow["d_tvd_pct"] > 0).sum()),
            "mean_d_silent_pct": block["d_silent_pct"].mean(),
            "mean_d_2q_pct": block["d_2q_pct"].mean(),
            "mean_d_1q_pct": block["d_1q_pct"].mean(),
        })
    return pd.DataFrame(rows)


def density_table(density: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, block in density.groupby("name"):
        arms = block.set_index("arm")
        if "baseline" not in arms.index or "sparse" not in arms.index:
            continue
        ref = arms.loc["baseline"]
        out = {"name": name, "num_qubits": int(ref["num_qubits"]),
               "tight_checks": int(ref["checks"]), "sparse_checks": int(arms.loc["sparse", "checks"]),
               "sparse_patches": int(arms.loc["sparse", "patches"])}
        for arm, label in (("baseline", "tight"), ("sparse", "sparse"), ("sparse_fix", "sparse_fix")):
            if arm not in arms.index:
                continue
            row = arms.loc[arm]
            if bool(row.get("identical_to_base", False)):
                row = arms.loc["sparse"]
            for rule in ("422", "412"):
                out[f"{label}_{rule}_lsr"] = row[f"lsr_{rule}"]
                out[f"{label}_{rule}_tvd"] = row[f"tvd_{rule}"]
                out[f"{label}_{rule}_silent"] = row[f"silent_per_kept_{rule}"]
            out[f"{label}_2q"] = row["two_qubit_gates"]
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["num_qubits", "name"])


OPTION_STYLE = {"shipped": ("#0097fd", "o", "decoding only"),
                "fix": ("#3b4c56", "s", "+ spare fix-up H"),
                "tH": ("#cdc2c0", "^", "targeted solo H")}


def plot_options(table: pd.DataFrame, output_dir: Optional[Path], bases: List[str],
                 stem: str = "solo_gauge_options") -> List[Path]:
    """
    Per circuit, each way of using the solo patch's gauge, relative to the shipped arm
    decoded as [[4,2,2]]: LSR change against silent-error-per-kept-shot change (frames,
    every width; TVD is not used because a variant is a different circuit with its own
    sample, so its TVD difference is unpaired and floor-dominated on small circuits).
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    fig, axes = plt.subplots(1, len(bases), figsize=(COLUMN_WIDTH, 2.5), sharey=True,
                             squeeze=False)
    names = {"baseline": "Baseline", "w10T": "AQORN w10T", "v4": "AQORN (v4)",
             "w10": "AQORN w10"}
    for ax, base in zip(axes[0], bases):
        block = table[table["base"] == base]
        for option, (color, marker, text) in OPTION_STYLE.items():
            sub = block[block["option"] == option]
            # A variant that leaves the circuit unchanged sits exactly on "decoding only".
            sub = sub[sub["check"]] if option == "shipped" else sub[sub["changed_circuit"]]
            ax.plot(sub["d_lsr_pct"], sub["d_silent_pct"], linestyle="none", marker=marker,
                    markersize=3.6, color=color, label=text,
                    markeredgecolor="#3b4c56" if color == "#cdc2c0" else color,
                    markeredgewidth=0.4, zorder=4 if option == "shipped" else 3)
        ax.axhline(0, color="#636363", linewidth=1.0, linestyle="--")
        ax.axvline(0, color="#636363", linewidth=1.0, linestyle="--")
        ax.set_title(names.get(base, base), fontsize=9, pad=2)
        ax.set_xlabel("LSR change (%)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel("silent / kept change (%)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=7,
               handletextpad=0.2, columnspacing=0.6, handlelength=1.0)
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def plot_density(table: pd.DataFrame, output_dir: Optional[Path],
                 stem: str = "solo_gauge_density") -> List[Path]:
    """
    Per circuit (n <= 18): tight packing vs one qubit per patch (every patch solo), each
    decoded as [[4,2,2]] (solid) and as [[4,1,2]] on solo patches (hatched). Colour is the
    packing, fill the decoding.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    combos = [("tight_422", "#3b4c56", False), ("tight_412", "#3b4c56", True),
              ("sparse_422", "#0097fd", False), ("sparse_412", "#0097fd", True)]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 4.2), sharex=True)
    x = np.arange(len(table))
    width = 0.8 / len(combos)
    for ax, metric, label in ((axes[0], "tvd", "TVD"), (axes[1], "lsr", "LSR")):
        for i, (key, color, hatched) in enumerate(combos):
            ax.bar(x - 0.4 + width * (i + 0.5), table[f"{key}_{metric}"], width=width * 0.9,
                   facecolor="white" if hatched else color, edgecolor=color,
                   hatch="//////" if hatched else None, linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(table["name"], rotation=60, ha="right", fontsize=8)
    handles = [Patch(facecolor="#3b4c56", edgecolor="#3b4c56", label="tight packing"),
               Patch(facecolor="#0097fd", edgecolor="#0097fd", label="1 qubit per patch"),
               Patch(facecolor="#636363", edgecolor="#636363", label="[[4,2,2]] decoding (solid)"),
               Patch(facecolor="white", edgecolor="#636363", hatch="//////",
                     label="[[4,1,2]] on solo patches (hatched)")]
    axes[0].legend(handles=handles, ncol=4, fontsize=8, loc="lower left",
                   bbox_to_anchor=(0, 1.0), handlelength=1.4, columnspacing=1.0)
    for ax, tag in zip(axes, "ab"):
        ax.text(-0.07, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold",
                fontsize=10, va="bottom")
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


AVAILABILITY_LABEL = {"baseline": "Baseline", "v4": "AQORN (v4)", "w10": "AQORN w10",
                      "w10T": "AQORN w10T", "baseline_fix": "Baseline + spare fix-up H",
                      "baseline_tH": "Baseline + targeted solo H",
                      "w10T_fix": "w10T + spare fix-up H", "w10T_tH": "w10T + targeted solo H",
                      "sparse": "1 qubit per patch (n \u2264 18)",
                      "sparse_fix": "1 qubit per patch + fix-up H"}


def plot_availability(frames: List[pd.DataFrame], output_dir: Optional[Path],
                      stem: str = "solo_gauge_availability") -> List[Path]:
    """
    Solo patches whose spare survives translation as a valid [[4,1,2]] check, per arm.
    `frames` are read in order and the first frame that has an arm supplies it (so the
    35-circuit main table wins over a narrower variant table for the shipped arms).
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    arms, rows = [], []
    for arm in AVAILABILITY_LABEL:
        for frame in frames:
            block = frame[frame["arm"] == arm]
            if len(block):
                arms.append(arm)
                rows.append((int(block["cand_valid"].sum()), int(block["cand_x_frame"].sum()),
                             int(block["cand_entangled"].sum())))
                break
    valid, xframe, ent = (np.array(v) for v in zip(*rows))
    total = valid + xframe + ent
    y = np.arange(len(arms))[::-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.22 * len(arms) + 0.9))
    ax.barh(y, 100 * valid / total, color="#0097fd", height=0.7, label="check valid")
    ax.barh(y, 100 * xframe / total, left=100 * valid / total, color="#cdc2c0", height=0.7,
            label="lost: spare in X frame (odd HH count)")
    if ent.any():
        ax.barh(y, 100 * ent / total, left=100 * (valid + xframe) / total, color="#3b4c56",
                height=0.7, label="lost: spare entangled")
    for yi, v, t in zip(y, valid, total):
        ax.text(101.5, yi, f"{v}/{t}", ha="left", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([AVAILABILITY_LABEL[a] for a in arms], fontsize=8)
    ax.set_xlabel("solo patches (%)")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", linestyle="--", linewidth=0.4, color="#b0b0b0")
    ax.grid(axis="y", visible=False)
    fig.legend(loc="outside upper center", ncol=2, fontsize=7, handlelength=1.0,
               columnspacing=0.8)
    fig.get_layout_engine().set(rect=(0, 0, 0.9, 1))
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--main-csv", required=True)
    parser.add_argument("--variants-csv", default=None)
    parser.add_argument("--density-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bases", nargs="+", default=["baseline", "w10T"])
    parser.add_argument("--alpha-csv", default=None,
                        help="alpha_one CSV: supplies the variant arms' availability")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    output_dir = Path(args.output_dir) if args.output_dir else None
    main_frame = pd.read_csv(args.main_csv)
    frames = [main_frame] + ([pd.read_csv(args.alpha_csv)] if args.alpha_csv else []) + (
        [pd.read_csv(args.variants_csv)] if args.variants_csv else []) + (
        [pd.read_csv(args.density_csv)] if args.density_csv else [])
    for path in plot_availability(frames, output_dir):
        print(f"[solo_gauge] saved {path}")
    if args.variants_csv:
        variants = resolve_variants(main_frame, pd.read_csv(args.variants_csv))
        table = option_table(main_frame, variants, args.bases[0])
        for base in args.bases[1:]:
            table = pd.concat([table, option_table(main_frame, variants, base)])
        print(summarize_options(table).round(3).to_string(index=False))
        for path in plot_options(table, output_dir, args.bases):
            print(f"[solo_gauge] saved {path}")
    if args.density_csv:
        density = pd.read_csv(args.density_csv)
        dt = density_table(density)
        rel = pd.DataFrame({
            "name": dt["name"], "n": dt["num_qubits"],
            "tight412_tvd%": 100 * (dt["tight_412_tvd"] / dt["tight_422_tvd"] - 1),
            "sparse422_tvd%": 100 * (dt["sparse_422_tvd"] / dt["tight_422_tvd"] - 1),
            "sparse412_tvd%": 100 * (dt["sparse_412_tvd"] / dt["tight_422_tvd"] - 1),
            "sparse412_vs_422_tvd%": 100 * (dt["sparse_412_tvd"] / dt["sparse_422_tvd"] - 1),
            "sparse422_lsr%": 100 * (dt["sparse_422_lsr"] / dt["tight_422_lsr"] - 1),
            "sparse412_lsr%": 100 * (dt["sparse_412_lsr"] / dt["tight_422_lsr"] - 1),
            "sparse_checks": dt["sparse_checks"], "patches": dt["sparse_patches"],
        })
        if "sparse_fix_412_tvd" in dt:
            rel["fix412_tvd%"] = 100 * (dt["sparse_fix_412_tvd"] / dt["tight_422_tvd"] - 1)
            rel["fix412_vs_sparse422_tvd%"] = 100 * (dt["sparse_fix_412_tvd"] / dt["sparse_422_tvd"] - 1)
            rel["fix412_lsr%"] = 100 * (dt["sparse_fix_412_lsr"] / dt["tight_422_lsr"] - 1)
            rel["fix412_vs_sparse422_lsr%"] = 100 * (dt["sparse_fix_412_lsr"] / dt["sparse_422_lsr"] - 1)
            rel["sparse412_vs_422_lsr%"] = 100 * (dt["sparse_412_lsr"] / dt["sparse_422_lsr"] - 1)
            rel["sparse412_vs_422_silent%"] = 100 * (dt["sparse_412_silent"] / dt["sparse_422_silent"] - 1)
            rel["sparse422_vs_tight_silent%"] = 100 * (dt["sparse_422_silent"] / dt["tight_422_silent"] - 1)
        print(rel.round(1).to_string(index=False))
        print(rel.drop(columns=["name"]).median().round(1).to_string())
        for path in plot_density(dt, output_dir):
            print(f"[solo_gauge] saved {path}")


if __name__ == "__main__":
    main()
