"""
final_front.py -- the optimality front under the final compiler, without
re-simulating: AQORN's gap to the exhaustive optimum of the objective it now decides on, and
its measured gaps, over the 75 small programs of `exhaustive_front.py`.

`exhaustive_front.py` measured every tightest [[4,2,2]] placement of 75 programs (4-6 qubits,
both H-commutation settings; Pauli frames on Fig 6's machine at alpha = 1, with independent
selection shots) and saved the optimised programs. The translator has not changed since, so
those measurements stand. What changed is the planner: its commutation decision and final check
now compare candidates on `standard_objective`, and the random restarts are gone. This script:

    plans    every program with the final `plan_standard`, and finds its pick in the space
    prices   every placement of the space (both commutation settings) on `standard_objective`
             at the planner's weight (`packing.standard._standard_price`)
    reports  per program: the search gap -- AQORN's standard objective against the space's
             minimum, over both commutation settings, as AQORN chooses between them -- and the
             measured gaps to the best placement on LSR, logical error and TVD (best chosen on
             the selection shots, read on the others, as `optimality_front.gaps` does), plus
             the share of the space beating the pick on both LSR and logical error. The earlier
             pick (the space's `aqorn`, planned with restarts and the old commutation score) is
             reported beside it.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.final_front \
        --space <dir with space.csv and programs/> --jobs 75
Writes results/optimality_front_final/final_front_space.csv and final_front_gaps.csv.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("results/optimality_front_final")
METRICS = (("lsr", True), ("ler", False), ("tvd", False))


def _key(placement) -> str:
    return json.dumps([[int(a), int(b)] for a, b in placement])


def _price_program(job):
    import warnings

    warnings.filterwarnings("ignore")
    from qiskit import qpy

    from packing.standard import STANDARD, _standard_price, plan_standard, standard_noise_profile

    program_id, path, rows = job
    with open(path, "rb") as handle:
        program = qpy.load(handle)[0]
    noise = standard_noise_profile()
    plan = plan_standard(program)
    pick = (_key(plan.placement), bool(plan.h_commutation))
    prices = [_standard_price(program, json.loads(placement), 4, bool(commute), noise, STANDARD)
              for placement, commute in rows]
    return dict(program=program_id, pick=pick, prices=prices, fell_back=bool(plan.fell_back),
                considered=plan.considered)


def best_value(group: pd.DataFrame, column: str, higher: bool) -> float:
    chooser = f"{column}_sel" if f"{column}_sel" in group.columns else column
    index = group[chooser].idxmax() if higher else group[chooser].idxmin()
    return float(group.loc[index, column])


def gap_row(group: pd.DataFrame, row: pd.Series, arm: str) -> dict:
    record = {"program": row["program"], "num_qubits": int(row["num_qubits"]), "arm": arm}
    live = group[np.isfinite(group["std_cost"])]
    best_cost = live["std_cost"].min()
    record["std_cost"] = 100.0 * (row["std_cost"] - best_cost) / best_cost
    record["std_optimal"] = bool(row["std_cost"] <= best_cost + 1e-12)
    for column, higher in METRICS:
        best = best_value(group, column, higher)
        if best <= 0:
            record[column] = np.nan
            continue
        gap = (best - row[column]) if higher else (row[column] - best)
        record[column] = 100.0 * gap / best
    better = (group["lsr"] > row["lsr"]) & (group["ler"] < row["ler"]) \
        & (group["lsr_sel"] > row["lsr_sel"]) & (group["ler_sel"] < row["ler_sel"])
    record["dominated_by_pct"] = 100.0 * better.mean()
    return record


def main() -> None:
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--space", required=True, help="exhaustive_front output directory")
    parser.add_argument("--jobs", type=int, default=75)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    root = Path(args.space)
    space = pd.read_csv(root / "space.csv")
    space = space[~space["rejected"].fillna(False).astype(bool)].reset_index(drop=True)
    jobs = []
    for program, group in space.groupby("program", sort=False):
        jobs.append((program, str(root / "programs" / f"{program.replace(':', '_')}.qpy"),
                     list(zip(group["placement"], group["commute"]))))
    results = parallel_map(_price_program, jobs, jobs=args.jobs, label="final_front")
    space["std_cost"] = np.nan
    space["pick_final"] = False
    plans = []
    for job, result in zip(jobs, results):
        if result is None:
            print(f"[final_front] {job[0]} failed")
            continue
        index = space.index[space["program"] == job[0]]
        space.loc[index, "std_cost"] = result["prices"]
        keys = space.loc[index, "placement"].map(lambda p: _key(json.loads(p)))
        hit = index[(keys == result["pick"][0]).values
                    & (space.loc[index, "commute"].astype(bool) == result["pick"][1]).values]
        space.loc[hit, "pick_final"] = True
        plans.append(dict(program=job[0], found=len(hit) == 1, fell_back=result["fell_back"],
                          commute=result["pick"][1], placement=result["pick"][0]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    space.to_csv(out / "final_front_space.csv", index=False)
    plans = pd.DataFrame(plans, columns=["program", "found", "fell_back", "commute", "placement"])
    rows = []
    for program, group in space.groupby("program", sort=False):
        final = group[group["pick_final"]]
        if len(final) == 1:
            rows.append(gap_row(group, final.iloc[0], "aqorn_final"))
        old = group[group["pick"].fillna("").str.split(",").apply(lambda arms: "aqorn" in arms)]
        if len(old) == 1:
            rows.append(gap_row(group, old.iloc[0], "aqorn_before"))
        naive = group[group["pick"].fillna("").str.split(",").apply(lambda arms: "naive" in arms)]
        if len(naive) == 1:
            rows.append(gap_row(group, naive.iloc[0], "naive"))
    table = pd.DataFrame(rows)
    table.to_csv(out / "final_front_gaps.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"[final_front] {len(plans)} programs planned; pick found in the space for "
          f"{int(plans.found.sum())}; fell back {int(plans.fell_back.sum())}")
    print("median gap to the best placement, %:")
    print(table.groupby("arm")[["std_cost", "lsr", "ler", "tvd", "dominated_by_pct"]]
          .median().round(2).to_string())
    print("mean gap, %:")
    print(table.groupby("arm")[["std_cost", "lsr", "ler", "tvd", "dominated_by_pct"]]
          .mean().round(2).to_string())
    print("exactly optimal on the standard objective, share by width:")
    print(table.groupby(["arm", "num_qubits"]).std_optimal.mean().round(3).unstack().to_string())
    print("by width, median (std_cost / lsr / ler / tvd):")
    print(table.groupby(["arm", "num_qubits"])[["std_cost", "lsr", "ler", "tvd"]].median()
          .round(2).to_string())


if __name__ == "__main__":
    main()
