"""
report.py -- the tables of reports/equal_qubit_code_size.md, from equal_qubit_code_size.csv.

    PYTHONPATH=. venv/bin/python -m experiments.revision.equal_qubit.report --csv <csv>

Prints markdown: the winning k per ratio and budget (LSR, logical error per kept shot on
occupied lanes, TVD on the unsaturated circuits), margins over the runner-up with a one-SE test
(paired log ratio across circuits), AQORN / naive per k, AQORN per k relative to naive k = 2 at
Q2, the budget and ancilla sensitivity, and the sanity counts (miscompiled, budget violations,
fallbacks, clip flags).
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from experiments.revision.equal_qubit.equal_qubit_code_size import (
    ANCILLAS, K_VALUES, saturated,
)


def _fmt_ratio(r: float) -> str:
    return f"1/{1 / r:g}" if r < 1 else f"{r:g}"


def winners(frame: pd.DataFrame, budget: str, metric: str, higher: bool,
            exclude=()) -> pd.DataFrame:
    a = frame[(frame.arm == "aqorn") & (frame.budget == budget) & ~frame.circuit.isin(exclude)]
    rows = []
    for ratio, g in a.groupby("ratio"):
        w = g.pivot_table(index="circuit", columns="k", values=metric)
        w = w.where(w > 0)
        gm = np.exp(np.log(w).mean())
        order = gm.sort_values(ascending=not higher)
        best, second = order.index[0], order.index[1]
        d = np.log(w[best] / w[second]).dropna()
        se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else math.nan
        rows.append(dict(ratio=ratio, best=int(best), runner_up=int(second),
                         margin=float(np.exp(d.mean())), se=float(se),
                         clears=bool(abs(d.mean()) > se),
                         wins=int((d > 0).sum() if higher else (d < 0).sum()), n=len(d),
                         order=" > ".join(str(int(k)) for k in order.index)))
    return pd.DataFrame(rows)


def per_k(frame: pd.DataFrame, budget: str, metric: str, exclude=()) -> pd.DataFrame:
    """Geometric mean over circuits of AQORN k / naive k = 2 at Q2, per ratio and k."""
    ref = frame[(frame.arm == "naive") & (frame.k == 2) & (frame.budget == "Q2")]
    ref = ref.set_index(["circuit", "ratio"])[metric]
    a = frame[(frame.arm == "aqorn") & (frame.budget == budget) & ~frame.circuit.isin(exclude)]
    rel = a.set_index(["circuit", "ratio"])
    rel = rel.assign(rel=rel[metric] / ref.reindex(rel.index))
    out = rel.reset_index().groupby(["ratio", "k"]).rel.apply(
        lambda x: float(np.exp(np.log(x[x > 0]).mean()))).unstack("k")
    return out


def aqorn_over_naive(frame: pd.DataFrame, budget: str, metric: str, exclude=()) -> pd.DataFrame:
    part = frame[(frame.budget == budget) & ~frame.circuit.isin(exclude)]
    w = part.pivot_table(index=["circuit", "ratio", "k"], columns="arm", values=metric)
    w = w[(w.aqorn > 0) & (w.naive > 0)]
    r = np.log(w.aqorn / w.naive).reset_index()
    r.columns = ["circuit", "ratio", "k", "lr"]
    return r.groupby(["ratio", "k"]).lr.apply(lambda x: float(np.exp(x.mean()))).unstack("k")


def table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = list(df.columns)
    head = "| ratio | " + " | ".join(f"k = {c}" if isinstance(c, (int, np.integer)) else str(c)
                                     for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [head, sep]
    for idx, row in df.iterrows():
        lines.append(f"| {_fmt_ratio(idx)} | " + " | ".join(
            fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    f = pd.read_csv(args.csv)
    drop = saturated(f)
    used = f.drop_duplicates(["circuit", "ratio", "budget", "k", "arm"])
    over = (used.data_qubits + used.ancillas_counted * ANCILLAS * used.patches > used.budget_qubits)
    print(f"rows {len(f)}; circuits {f.circuit.nunique()}; ratios {sorted(f.ratio.unique())}; "
          f"budgets {sorted(f.budget.unique())}")
    print(f"miscompiled {int(f.miscompiled.sum())}; budget violations {int(over.sum())}; "
          f"rows without a measurement {int(f.lsr.isna().sum())}; TVD-saturated {drop}")
    a = f[f.arm == "aqorn"]
    print("AQORN final-check fallbacks by budget and k:")
    print(a.groupby(["budget", "k"]).plan_fell_back.mean().unstack("k").round(2).to_string())
    print("at the tightest packing (no spare patches), share of rows by budget and k:")
    print(f.groupby(["budget", "k"]).at_tightest.mean().unstack("k").round(2).to_string())
    print("mean occupancy (logical qubits / slots) by budget and k:")
    print(f.groupby(["budget", "k"]).occupancy.mean().unstack("k").round(2).to_string())
    for budget in ("Q2", "1.25Q2", "1.5Q2", "Q2+anc"):
        for metric, higher, label, ex in (("lsr", True, "LSR", ()),
                                          ("logical_error", False, "logical error / kept shot", ()),
                                          ("tvd", False, "TVD (unsaturated circuits)", drop)):
            w = winners(f, budget, metric, higher, ex)
            print(f"\n### {budget}: winning k on {label}\n")
            print("| ratio | best k | runner-up | margin | clears 1 SE | best wins on | order |")
            print("|---|---|---|---|---|---|---|")
            for r in w.itertuples():
                print(f"| {_fmt_ratio(r.ratio)} | {r.best} | {r.runner_up} | x{r.margin:.3f} | "
                      f"{'yes' if r.clears else 'no'} | {r.wins}/{r.n} | {r.order} |")
    for budget in ("Q2", "Q2+anc"):
        for metric, label, ex in (("lsr", "LSR", ()), ("logical_error", "logical error", ()),
                                  ("tvd", "TVD", drop)):
            print(f"\n### {budget}: AQORN k / naive k = 2 at Q2, {label} (geometric mean)\n")
            print(table(per_k(f, budget, metric, ex)))
    for metric, label, ex in (("lsr", "LSR", ()), ("logical_error", "logical error", ()),
                              ("tvd", "TVD", drop)):
        print(f"\n### Q2: AQORN / naive at the same k and P, {label} (geometric mean)\n")
        print(table(aqorn_over_naive(f, "Q2", metric, ex)))


if __name__ == "__main__":
    main()
