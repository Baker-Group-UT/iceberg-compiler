"""
summary.py -- the report's tables from `concat_eval.csv`, as markdown.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.summary results/concatenation/concat_eval.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.concatenation.concat_results.concat_results import pooled_silent

REFERENCE = "concat_BB"
LABELS = {
    "single_base": "1 level, baseline", "single_w10": "1 level, AQORN w10",
    "single_w10T": "1 level, AQORN w10T", "concat_BB": "2 levels, baseline / baseline",
    "concat_AB": "outer AQORN / inner baseline", "concat_Bp": "outer baseline / inner passes",
    "concat_BA": "outer baseline / inner AQORN", "concat_Ap": "outer AQORN / inner passes",
    "concat_AA": "**outer AQORN / inner AQORN**", "concat_AA_T": "outer w10T / inner AQORN",
    "concat_AA_index": "outer AQORN / inner index (not spread)",
}
ORDER = list(LABELS)


def paired(frame, column, lower_is_better, label="calibrated"):
    sub = frame[frame["alpha_label"] == label]
    wide = sub.pivot_table(index="circuit_name", columns="arm", values=column)
    ref = wide[REFERENCE]
    change = (ref.values[:, None] - wide.values) if lower_is_better else (wide.values - ref.values[:, None])
    return pd.DataFrame(change / ref.values[:, None] * 100, index=wide.index, columns=wide.columns)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main(path: str) -> None:
    frame = pd.read_csv(path)
    circuits = frame["circuit_name"].nunique()
    cal = frame[frame["alpha_label"] == "calibrated"]
    print(f"circuits: {circuits}; noiseless checks: min LSR {frame['noiseless_lsr'].min():.3f}, "
          f"max TVD {frame['noiseless_tvd'].max():.3f}; calibrated alpha median "
          f"{cal['alpha_star'].median():.3g} (range {cal['alpha_star'].min():.3g}-{cal['alpha_star'].max():.3g})\n")

    reductions = {m: paired(frame, m, True) for m in ("two_qubit_gates", "one_qubit_gates", "depth")}
    fidelity = {"postselect_lsr": paired(frame, "postselect_lsr", False),
                "postselect_silent": paired(frame, "postselect_silent", True),
                "postselect_tvd": paired(frame, "postselect_tvd", True)}
    absolute = cal.groupby("arm")[["physical_qubits", "two_qubit_gates", "depth", "postselect_lsr",
                                   "postselect_silent", "postselect_tvd", "erasure_lsr",
                                   "erasure_silent", "erasure_tvd"]].mean()
    rows = []
    for arm in ORDER:
        if arm not in absolute.index:
            continue
        r = absolute.loc[arm]
        def fmt(df):
            return f"{df[arm].mean():+.1f} ({df[arm].median():+.1f})"
        rows.append([LABELS[arm], f"{r.physical_qubits:.0f}", fmt(reductions["two_qubit_gates"]),
                     fmt(reductions["depth"]), f"{r.postselect_lsr:.3f}", fmt(fidelity["postselect_lsr"]),
                     f"{r.postselect_silent:.2e}", fmt(fidelity["postselect_silent"]),
                     f"{r.postselect_tvd:.4f}"])
    print("At each circuit's calibrated alpha (concat_BB keeps 50% of shots), post-selection. "
          "% change vs 2 levels baseline: mean (median) over circuits; + = better.\n")
    print(table(rows, ["compilation", "phys. qubits", "2q gates", "depth", "LSR", "LSR change",
                       "silent/kept", "silent change", "TVD"]))

    print("\nWins per circuit vs 2-level baseline (calibrated alpha):")
    for arm in ("concat_AB", "concat_Bp", "concat_BA", "concat_AA", "concat_AA_T"):
        lsr = fidelity["postselect_lsr"][arm]
        sil = fidelity["postselect_silent"][arm]
        print(f"  {arm}: LSR better on {int((lsr > 0).sum())}/{lsr.notna().sum()}, "
              f"silent better on {int((sil > 0).sum())}/{sil.notna().sum()}")

    print("\nDecoders (calibrated alpha), mean over circuits:\n")
    rows = []
    for arm in ("concat_BB", "concat_AA", "concat_AA_index"):
        r = absolute.loc[arm]
        rows.append([LABELS[arm], f"{r.postselect_lsr:.3f}", f"{r.erasure_lsr:.3f}",
                     f"{r.postselect_silent:.2e}", f"{r.erasure_silent:.2e}",
                     f"{r.postselect_tvd:.4f}", f"{r.erasure_tvd:.4f}"])
    print(table(rows, ["compilation", "LSR post-select", "LSR erasure", "silent post-select",
                       "silent erasure", "TVD post-select", "TVD erasure"]))

    print("\nSilent error per kept shot, pooled over circuits, by fixed alpha (post-selection):\n")
    alphas = sorted(frame.loc[frame["alpha_label"] != "calibrated", "alpha"].unique())
    rows = []
    for arm in ORDER:
        if arm not in set(frame["arm"]):
            continue
        s = pooled_silent(frame, arm)
        good = s[s > 0]
        slope = np.polyfit(np.log(good.index), np.log(good.values), 1)[0] if len(good) > 1 else np.nan
        rows.append([LABELS[arm]] + [f"{s.get(a, np.nan):.2e}" for a in alphas] + [f"{slope:.2f}"])
    print(table(rows, ["compilation"] + [f"alpha={a:g}" for a in alphas] + ["log-log slope"]))

    single = pooled_silent(frame, "single_base")
    for arm in ("single_w10", "concat_BB", "concat_AA"):
        ratio = (single / pooled_silent(frame, arm)).round(1).to_dict()
        print(f"\n1-level baseline silent / {arm} silent by alpha: {ratio}")

    if "inner_plan" in frame:
        plans = cal[cal["arm"].isin(["concat_BA", "concat_AA", "concat_AA_T"])]
        print("\nInner AQORN placement choices:", plans.groupby("arm")["inner_plan"].value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/concatenation/concat_eval.csv")
