"""
check_space.py -- does every placement the exhaustive optimality sweep measured compute its
program? Pauli frames (how the sweep measured LSR / LER / TVD) only count the flips noise causes
in whatever circuit came out, so a placement the translator miscompiled would be measured as if
it were correct. This samples each sweep program's placements -- two per cluster (grouping x
H-commutation setting: the first and last in enumeration order) plus AQORN's pick -- and runs the
noiseless check of `placement_measure.noiseless_ok` on each.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.check_space --jobs 12
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

NAME = "check_space"


def _check(job) -> List[Dict]:
    from qiskit import qpy

    from experiments.optimality.placement_measure import logical_support, noiseless_ok
    from experiments.tvd_explore.fastframes import at_alpha_one
    from packing.standard import standard_translator_config
    from translator.core import Translator

    program_path, items = job
    with open(program_path, "rb") as stream:
        program = qpy.load(stream)[0]
    support = logical_support(program)
    rows = []
    for pid, commute, placement, why in items:
        result = Translator(config=standard_translator_config(4, commute=commute)).translate(
            program, [tuple(p) for p in json.loads(placement)])
        positions = [tuple(p) for p in result.context.state_positions]
        ok = noiseless_ok(result.circuit, at_alpha_one(result.circuit), positions, support)
        rows.append({"program": pid, "commute": commute, "placement": placement, "why": why,
                     "correct": bool(ok)})
    return rows


def main() -> None:
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--space", default="results/optimality_front/space.csv")
    parser.add_argument("--out", default="results/optimality_front/check_space.csv")
    args = parser.parse_args()
    space = pd.read_csv(args.space, keep_default_na=False, na_values=[""],
                        usecols=["program", "commute", "placement", "placement_index", "pick"])
    space["pick"] = space["pick"].fillna("")

    def grouping(text):
        groups: Dict[int, list] = {}
        for q, (p, _) in enumerate(json.loads(text)):
            groups.setdefault(p, []).append(q)
        return tuple(sorted(tuple(sorted(v)) for v in groups.values()))

    space["grouping"] = space["placement"].map(grouping)
    jobs = []
    for pid, g in space.groupby("program"):
        items = []
        for (commute, _), cluster in g.groupby(["commute", "grouping"]):
            ordered = cluster.sort_values("placement_index")
            for why, row in (("first", ordered.iloc[0]), ("last", ordered.iloc[-1])):
                items.append((pid, bool(commute), row["placement"], why))
        for _, row in g[g["pick"].str.contains("aqorn")].iterrows():
            items.append((pid, bool(row["commute"]), row["placement"], "aqorn"))
        path = f"results/optimality_front/programs/{pid.replace(':', '_')}.qpy"
        jobs.append((path, items))
    results = parallel_map(_check, jobs, jobs=args.jobs, label=NAME)
    frame = pd.DataFrame([r for result in results if result for r in result])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    bad = frame[~frame["correct"]]
    print(f"[{NAME}] {len(frame)} placements of {frame['program'].nunique()} programs checked; "
          f"miscompiled {len(bad)}" + (f": {bad.groupby('program').size().to_dict()}" if len(bad) else "")
          + f"; failed jobs {sum(r is None for r in results)}", flush=True)


if __name__ == "__main__":
    main()
