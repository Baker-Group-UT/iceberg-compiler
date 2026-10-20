"""
verify_diagnosis.py -- would `plan_standard`'s final check decide better on the standard objective?

For every (circuit, ratio) of `noise_ratio_packers.csv`, price AQORN's search result (aqorn_raw)
and naive the way the resource selector does (`standard_objective`: detected errors on the
standard translation's schedule, on a Clifford stand-in so rotations do not drop it to gate
counting, + the silent term) under that ratio's planner profile, and record which it keeps. The
sweep already measured both placements, so the decisions are scored against frames with no new
simulation: `summarize` compares "check on the standard objective" with the shipped check and
with no check.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.verify_diagnosis --jobs 190
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import warnings
from typing import Dict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SWEEP = "results/noise_ratio_packers/noise_ratio_packers.csv"


def _price(job) -> Dict:
    from experiments.shared.simulation import SimSettings, ensure_measurements, noise_profile_for
    from packing.standard import standard_objective, standard_translator_config
    from translator.core import Translator

    name, original, ratio, commute, placements = job
    profile = noise_profile_for(SimSettings().with_noise_ratio(ratio))
    program = ensure_measurements(original)
    out = {"circuit": name, "ratio": ratio}
    for arm, placement in placements.items():
        placement = [tuple(p) for p in placement]
        translator = Translator(config=standard_translator_config(4, commute=commute))
        circuit = translator.translate(program, placement).circuit
        value = standard_objective(program, placement, 4, circuit, commute=commute, noise=profile)
        out.update({f"{arm}_total": value.total, f"{arm}_detected": value.detected,
                    f"{arm}_silent": value.silent, f"{arm}_currency": value.currency})
    return out


def summarize(sweep: pd.DataFrame, priced: pd.DataFrame, alpha: str = "cal") -> None:
    from scipy.stats import wilcoxon

    lsr, ler, tvd = f"lsr_{alpha}", f"ler_{alpha}", f"tvd_{alpha}"
    wide = {m: sweep.pivot_table(index=["circuit", "ratio"], columns="arm", values=m)
            for m in (lsr, ler, tvd)}
    fell_old = sweep[sweep.arm == "aqorn"].set_index(["circuit", "ratio"]).aqorn_fell_back
    priced = priced.set_index(["circuit", "ratio"])
    fall_new = priced["naive_total"] < priced["aqorn_raw_total"]
    for metric in (lsr, ler, tvd):
        w = wide[metric]
        w["aqorn_new"] = np.where(fall_new.reindex(w.index), w["naive"], w["aqorn_raw"])
    for ratio in sorted(priced.index.get_level_values("ratio").unique()):
        sel = priced.index.get_level_values("ratio") == ratio
        idx = priced.index[sel]
        print(f"ratio {ratio:g}: shipped check falls back on {int(fell_old.reindex(idx).sum())}"
              f"/{len(idx)}, standard-objective check on {int(fall_new[sel].sum())}/{len(idx)}")
        for label, metric in (("LER", ler), ("LSR", lsr), ("TVD", tvd)):
            w = wide[metric].loc[idx]
            if metric == tvd:
                w = w[w["naive"] > 0]
            parts = []
            for other in ("aqorn", "aqorn_raw"):
                r = np.log(w["aqorn_new"] / w[other]).dropna()
                nz = r[r.abs() > 1e-12]
                p = wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
                parts.append(f"vs {'shipped' if other == 'aqorn' else 'no check'} "
                             f"x{np.exp(r.mean()):.3f} (better {int((r < 0).sum() if metric != lsr else (r > 0).sum())}, "
                             f"worse {int((r > 0).sum() if metric != lsr else (r < 0).sum())}, p={p:.2g})")
            print(f"    {label}: " + "; ".join(parts))


def main() -> None:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--sweep", default=SWEEP)
    parser.add_argument("--out", default="results/noise_ratio_packers/verify_diagnosis.csv")
    args = parser.parse_args()
    sweep = pd.read_csv(args.sweep)
    originals = {c.name: c.original for c in load_fig6_circuits()}
    jobs = []
    for (name, ratio), part in sweep.groupby(["circuit", "ratio"]):
        arms = part.set_index("arm")
        jobs.append((name, originals[name], float(ratio), bool(arms.loc["aqorn", "commute"]),
                     {arm: json.loads(arms.loc[arm, "placement"]) for arm in ("aqorn_raw", "naive")}))
    jobs.sort(key=lambda j: -j[1].num_qubits)
    priced = pd.DataFrame(parallel_map(_price, jobs, jobs=args.jobs, label="verify"))
    priced.to_csv(args.out, index=False)
    summarize(sweep, priced)


if __name__ == "__main__":
    main()
