"""
structure.py -- why AQORN wins where it wins, from the gadget census (`census.py`).

Reads results/insights/census.csv (one row per Fig 6 circuit and arm, under the standard) and
reports:

  1. attribution   each stage's change in 1q / 2q gates, split by gadget class -- how much of
                   the reduction is Hadamards removed, Hadamards fused into HH, two-qubit gates
                   fused into CXT / CZT, in-patch CZs built transversally, ...
  2. predictors    Spearman correlation of each logical-program feature with each outcome
                   (cumulative reductions over the three stages)
  3. errors        how well the schedule's detected-error budget explains the selection-rate
                   change, and the silent error per kept shot the TVD change; and the silent
                   error budget by gadget class per arm
  4. families      outcomes and dominant mechanism by circuit family, and the top circuits

    PYTHONPATH=. venv/bin/python -m experiments.insights.structure [--census <csv>]
Writes results/insights/structure_attribution[_vN].csv and figures
results/figures/insights_attribution[_vN].{svg,pdf}, insights_errors[_vN].{svg,pdf}.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

OUT = Path("results/insights")
ARMS = ("baseline", "h_only", "h_merge", "packed")
STAGES = (("baseline", "h_only", "H commutation"), ("h_only", "h_merge", "merging"),
          ("h_merge", "packed", "packing"))
#: census class -> mechanism group in the attribution
GROUP = {"H": "Hadamards (one lane)", "HH": "Hadamards fused (HH)",
         "RZ": "rotations", "RX": "rotations", "S": "phase gates", "SX": "phase gates",
         "PAULI": "Paulis", "PAULI_FUSED": "Paulis",
         "CX_in": "two-qubit in patch", "CZ_in": "two-qubit in patch",
         "XCX_in": "two-qubit in patch", "CX_x": "two-qubit across patches",
         "CZ_x": "two-qubit across patches", "XCX_x": "two-qubit across patches",
         "CXT": "two-qubit fused (CXT/CZT)", "CZT": "two-qubit fused (CXT/CZT)",
         "XCXT": "two-qubit fused (CXT/CZT)", "SWAP": "other", "OTHER": "other"}
OUTCOMES = {"1q": ("recorded_one", True), "2q": ("recorded_two", True),
            "depth": ("recorded_depth", True), "LSR": ("lsr", False), "TVD": ("tvd", True)}


def family(name: str) -> str:
    stem = re.sub(r"_n\d+$", "", name)
    return {"ghz_state": "ghz", "cat_state": "cat", "multiplier": "multiply",
            "bigadder": "adder"}.get(stem, stem)


def _cost(arm_rows: pd.DataFrame, transversal: bool) -> Dict[str, Dict[str, float]]:
    """Per census class: (1q, 2q) contributed by this arm, from the same tables as `census`."""
    from experiments.insights.census import PRICE
    from iceberg.costs import CROSS_PATCH_COST, work_table
    from iceberg.gadget_options import transversal_intra

    with transversal_intra(transversal):
        work = work_table(4)
    out = {}
    for name, (key, _) in PRICE.items():
        cost = work[key]
        if name.endswith("_x") and key in CROSS_PATCH_COST:
            cost = CROSS_PATCH_COST[key]
        count = float(arm_rows[f"n_{name}"])
        out[name] = {"one": count * cost[0], "two": count * cost[1]}
    return out


def attribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Per circuit and stage: change in priced 1q / 2q by mechanism group (negative = saved)."""
    rows = []
    for name, block in frame.groupby("circuit"):
        arms = block.set_index("arm")
        costs = {arm: _cost(arms.loc[arm], transversal=arm in ("h_merge", "packed"))
                 for arm in ARMS}
        for a, b, stage in STAGES:
            delta: Dict[str, Dict[str, float]] = {}
            for cls in costs[a]:
                group = GROUP[cls]
                d = delta.setdefault(group, {"one": 0.0, "two": 0.0})
                d["one"] += costs[b][cls]["one"] - costs[a][cls]["one"]
                d["two"] += costs[b][cls]["two"] - costs[a][cls]["two"]
            for group, d in delta.items():
                rows.append({"circuit": name, "family": family(name), "stage": stage,
                             "group": group, "d_one": d["one"], "d_two": d["two"],
                             "baseline_one": float(arms.loc["baseline", "recorded_one"]),
                             "baseline_two": float(arms.loc["baseline", "recorded_two"])})
    return pd.DataFrame(rows)


def outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Per circuit: cumulative reduction (percent) of every outcome, baseline -> packed."""
    rows = []
    for name, block in frame.groupby("circuit"):
        arms = block.set_index("arm")
        base, packed = arms.loc["baseline"], arms.loc["packed"]
        row = {"circuit": name, "family": family(name), "size_group": base["size_group"],
               "tvd_resolved": float(base["baseline_tvd"]) <= 0.95}
        for key, (column, reduction) in OUTCOMES.items():
            b, p = float(base[column]), float(packed[column])
            row[key] = 100 * ((b - p) if reduction else (p - b)) / b if b > 0 else np.nan
        for a, b_arm, stage in STAGES:
            for key in ("2q", "LSR", "TVD"):
                column, reduction = OUTCOMES[key]
                x, y = float(arms.loc[a, column]), float(arms.loc[b_arm, column])
                ref = float(base[column])
                row[f"{key}:{stage}"] = 100 * ((x - y) if reduction else (y - x)) / ref if ref else np.nan
        row["silent_per_kept_reduction"] = 100 * (
            1 - float(packed["silent_per_kept"]) / max(1e-9, float(base["silent_per_kept"])))
        row["d_E_detected"] = float(packed["E_detected"]) - float(base["E_detected"])
        row["alpha"] = float(base["alpha"])
        for feature in FEATURES:
            row[feature] = float(base[feature])
        rows.append(row)
    return pd.DataFrame(rows)


FEATURES = ("n", "gates", "frac_h", "frac_rotation", "frac_s", "frac_two_qubit", "frac_cz_of_2q",
            "h_removable_by_commutation", "h_after_commutation_per_gate",
            "frac_h_in_parallel_layers", "frac_2q_in_parallel_layers", "reuse_per_pair",
            "max_degree", "mean_degree", "frac_2q_nearest_neighbour", "frac_2q_in_naive_patch",
            "parallelism")


def predictors(table: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for feature in FEATURES:
        row = {"feature": feature}
        for key in ("1q", "2q", "depth", "LSR", "TVD"):
            data = table if key != "TVD" else table[table["tvd_resolved"]]
            row[key] = spearmanr(data[feature], data[key]).correlation
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def error_mechanism(frame: pd.DataFrame) -> Dict[str, float]:
    """Do the schedule's detected errors explain LSR, and silent per kept explain TVD?"""
    from scipy.stats import pearsonr, spearmanr

    arms = frame.copy()
    arms["log_lsr"] = np.log(arms["lsr"].clip(lower=1e-9))
    arms["pred"] = -arms["alpha"] * arms["E_detected"]
    # Within circuit: arm minus baseline, which removes each circuit's own offsets.
    base = arms[arms["arm"] == "baseline"].set_index("circuit")
    rest = arms[arms["arm"] != "baseline"].copy()
    rest["d_log_lsr"] = rest["log_lsr"] - rest["circuit"].map(base["log_lsr"])
    rest["d_pred"] = rest["pred"] - rest["circuit"].map(base["pred"])
    resolved = rest[rest["baseline_tvd"] <= 0.95].copy()
    resolved["d_tvd"] = resolved["tvd"] - resolved["circuit"].map(base["tvd"])
    resolved["d_spk"] = resolved["silent_per_kept"] - resolved["circuit"].map(base["silent_per_kept"])
    resolved["d_gadget_silent"] = (resolved["alpha"] * resolved["gadget_silent"]
                                   - resolved["circuit"].map(base["alpha"] * base["gadget_silent"]))
    level = arms[arms["baseline_tvd"] <= 0.95]
    slope = np.polyfit(rest["d_pred"], rest["d_log_lsr"], 1)[0]
    return {
        "dlogLSR_vs_dE_pearson": pearsonr(rest["d_pred"], rest["d_log_lsr"])[0],
        "dlogLSR_vs_dE_slope": slope,
        "TVD_vs_spk_spearman_levels": spearmanr(level["silent_per_kept"], level["tvd"]).correlation,
        "dTVD_vs_dspk_pearson": pearsonr(resolved["d_spk"], resolved["d_tvd"])[0],
        "dspk_vs_dgadget_silent_pearson": pearsonr(resolved["d_gadget_silent"], resolved["d_spk"])[0],
        "arms": float(len(rest)), "resolved_arms": float(len(resolved)),
    }


def silent_budget(frame: pd.DataFrame) -> pd.DataFrame:
    """Corpus share of first-order silent error by gadget class, per arm (alpha = 1 rates)."""
    from experiments.insights.census import PRICE
    from packing.silent import DATA

    shares = []
    for arm in ARMS:
        transversal = arm in ("h_merge", "packed")
        table = pd.read_csv(DATA / ("gadget_rates_transversal.csv" if transversal
                                    else "gadget_rates.csv"))
        rates = table[table["n_code"] == 4].set_index("gadget")["p_logical"]
        rows = frame[frame["arm"] == arm]
        total = {}
        for cls, (_, gadget) in PRICE.items():
            if gadget in rates.index:
                total[GROUP[cls]] = total.get(GROUP[cls], 0.0) + float(
                    (rows[f"n_{cls}"] * rates[gadget]).sum())
        shares.append(pd.Series(total, name=arm))
    return pd.DataFrame(shares).T


def plot(attr: pd.DataFrame, table: pd.DataFrame, out_dir=None) -> List[Path]:
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure, use_style

    use_style()
    paths: List[Path] = []
    # Attribution of the 2q change, corpus total per stage and mechanism (percent of baseline).
    pivot = attr.groupby(["stage", "group"])["d_two"].sum().unstack(fill_value=0.0)
    pivot = -100 * pivot / attr.drop_duplicates("circuit")["baseline_two"].sum()
    pivot = pivot.loc[[s for _, _, s in STAGES]]
    order = pivot.abs().sum().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(3.487, 2.4))
    left = np.zeros(len(pivot))
    for group in order:
        values = pivot[group].to_numpy()
        ax.barh(range(len(pivot)), values, left=np.where(values >= 0, left, 0), label=group)
        left = left + np.where(values >= 0, values, 0)
    ax.set_yticks(range(len(pivot)), list(pivot.index))
    ax.axvline(0, color="#636363", linewidth=0.8)
    ax.set_xlabel("2q gates saved, % of the corpus baseline")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    paths += save_figure(fig, "insights_attribution", Path(__file__),
                         override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def _free(path: Path) -> Path:
    version, candidate = 2, path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        version += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--census", default=str(OUT / "census.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.census)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    gap_one = ((frame["priced_one"] - frame["recorded_one"]).abs() / frame["recorded_one"])
    gap_two = ((frame["priced_two"] - frame["recorded_two"]).abs() / frame["recorded_two"])
    print(f"[structure] {frame['circuit'].nunique()} circuits; census priced vs recorded: "
          f"1q median {gap_one.median():.2%} (95th pct {gap_one.quantile(.95):.1%}), "
          f"2q median {gap_two.median():.2%} (95th pct {gap_two.quantile(.95):.1%})")

    attr = attribution(frame)
    path = _free(OUT / "structure_attribution.csv")
    attr.to_csv(path, index=False)
    corpus_two = attr.drop_duplicates("circuit")["baseline_two"].sum()
    corpus_one = attr.drop_duplicates("circuit")["baseline_one"].sum()
    for key, total, label in (("d_two", corpus_two, "2q"), ("d_one", corpus_one, "1q")):
        pivot = attr.groupby(["group", "stage"])[key].sum().unstack(fill_value=0.0)
        pivot = (-100 * pivot / total).round(2)
        pivot["total"] = pivot.sum(axis=1)
        print(f"\n[structure] corpus {label} saved, % of the corpus baseline "
              f"(negative = added), by mechanism and stage:")
        print(pivot.sort_values("total", ascending=False).to_string())

    table = outcomes(frame)
    table.to_csv(_free(OUT / "structure_outcomes.csv"), index=False)
    print("\n[structure] Spearman correlation of program features with cumulative reductions "
          "(TVD on resolved circuits):")
    print(predictors(table).round(2).to_string())

    print("\n[structure] error mechanism:")
    for key, value in error_mechanism(frame).items():
        print(f"  {key}: {value:.3f}")
    print("\n[structure] first-order silent error by gadget group, corpus sum per arm (alpha 1):")
    budget = silent_budget(frame)
    print(budget.round(3).to_string())
    print((100 * budget / budget.sum()).round(1).to_string())

    fam = table.groupby("family").agg(
        circuits=("circuit", "count"), n_med=("n", "median"),
        frac_h=("frac_h", "mean"), frac_rot=("frac_rotation", "mean"),
        frac_cz=("frac_cz_of_2q", "mean"), h_removable=("h_removable_by_commutation", "mean"),
        q1=("1q", "mean"), q2=("2q", "mean"), depth=("depth", "mean"), LSR=("LSR", "mean"),
        TVD=("TVD", "mean"), spk=("silent_per_kept_reduction", "mean"))
    print("\n[structure] by family (means; reductions in %):")
    print(fam.sort_values("q2", ascending=False).round(2).to_string())
    for key in ("2q", "LSR", "TVD"):
        data = table if key != "TVD" else table[table["tvd_resolved"]]
        top = data.sort_values(key, ascending=False).head(8)
        print(f"\n[structure] top circuits by {key}:")
        print(top[["circuit", "n", key, "2q", "LSR", "TVD", "frac_h", "frac_rotation",
                   "frac_cz_of_2q", "h_removable_by_commutation",
                   f"{key}:H commutation" if key != "2q" else "2q:H commutation",
                   f"{key}:merging", f"{key}:packing"]].round(2).to_string(index=False))
    if not args.no_plot:
        for out in plot(attr, table, args.output_dir):
            print(f"[structure] saved {out}")


if __name__ == "__main__":
    main()
