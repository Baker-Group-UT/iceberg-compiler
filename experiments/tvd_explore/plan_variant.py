"""
plan_variant.py -- plan Fig 6's circuits with the silent-aware planner under translation switches.

    PYTHONPATH=. python -m experiments.tvd_explore.plan_variant <label> --weight 10 \
        [--switch TRANSVERSAL_INTRA_TWO_QUBIT] [--rates results/tvd_explore/gadget_rates_transversal.csv]

The switches are set in every worker before anything is priced, so the planner's tables
(`iceberg.costs.work_table`), its verify translations and the gadget rates all describe
the same translator. Writes results/tvd_explore/plans_<label>.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")


def point(job):
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from experiments.shared.simulation import ensure_measurements
    from translator import explore_switches

    circuit, weight, switches, rates_csv = job
    explore_switches.configure(**{name: True for name in switches})
    rates = UndetectedRates.from_table(pd.read_csv(rates_csv), 4, 3e-3)
    program = ensure_measurements(circuit.original)
    began = time.time()
    plan = plan_silent_aware(program, 4, rates=rates, silent_weight=weight, noise=fig6_noise_profile())
    return [{"name": circuit.name, "placement": json.dumps([[int(a), int(b)] for a, b in plan.placement]),
             "commute": bool(plan.h_commutation), "fell_back": bool(plan.fell_back),
             "currency": plan.currency, "plan_seconds": time.time() - began}]


def main():
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--weight", type=float, default=10.0)
    parser.add_argument("--switch", action="append", default=[])
    parser.add_argument("--rates", default="results/revision/exp8/gadget_rates.csv")
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    circuits = load_fig6_circuits(names=args.names)
    jobs = sorted([(c, args.weight, tuple(args.switch), args.rates) for c in circuits],
                  key=lambda j: -j[0].num_qubits)
    results = parallel_map(point, jobs, jobs=args.jobs, label=f"plan-{args.label}")
    rows = [r for res in results if res for r in res]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"plans_{args.label}.csv"
    pd.DataFrame(rows).assign(source=args.label).to_csv(path, index=False)
    print(f"wrote {path}: {len(rows)} plans, {sum(r is None for r in results)} failed")


if __name__ == "__main__":
    main()
