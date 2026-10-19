"""
check_noiseless.py -- does a variant compute the right distribution on every Fig 6 circuit?

Frame simulation sees only noise, so it cannot see a miscompile (the pipeline has produced
wrong distributions silently before -- `experiments.shared.simulation.noiseless_tvd`). For
each circuit: the variant's arm and the naive arm under the same commutation setting and
switches, both simulated noiselessly through the pipeline; `miscompiled(naive, variant)`
flags a variant whose noiseless TVD exceeds naive's.

    PYTHONPATH=. python -m experiments.tvd_explore.check_noiseless plans_w10T.csv \
        --switch TRANSVERSAL_INTRA_TWO_QUBIT
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")


def point(job):
    from experiments.shared.simulation import ensure_measurements, miscompiled
    from experiments.tvd_explore.screen import _noiseless_tvd
    from translator import explore_switches

    circuit, placement, commute, switches, shots = job
    explore_switches.configure(**{name: True for name in switches})
    proxy = ensure_measurements(circuit.proxy)
    naive = [(i // 2, i % 2) for i in range(proxy.num_qubits)]
    reference = _noiseless_tvd(proxy, naive, commute=commute, merging=True, overrides={}, seed=5)
    clean = _noiseless_tvd(proxy, placement, commute=commute, merging=True, overrides={}, seed=5)
    return [{"name": circuit.name, "num_qubits": proxy.num_qubits, "noiseless_ref": reference,
             "noiseless_tvd": clean, "miscompiled": bool(miscompiled(reference, clean))}]


def main():
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("plans")
    parser.add_argument("--switch", action="append", default=[])
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    plans = pd.read_csv(OUT / args.plans).set_index("name")
    circuits = load_fig6_circuits(names=list(plans.index))
    jobs = sorted([(c, [tuple(p) for p in json.loads(plans.loc[c.name, "placement"])],
                    bool(plans.loc[c.name, "commute"]), tuple(args.switch), args.shots)
                   for c in circuits], key=lambda j: -j[0].num_qubits)
    results = parallel_map(point, jobs, jobs=args.jobs, label="noiseless")
    frame = pd.DataFrame([r for res in results if res for r in res])
    path = OUT / f"noiseless_{Path(args.plans).stem}.csv"
    frame.to_csv(path, index=False)
    print(f"wrote {path}: {len(frame)} circuits checked, {sum(r is None for r in results)} failed, "
          f"{int(frame['miscompiled'].sum())} miscompiled: "
          f"{sorted(frame.loc[frame['miscompiled'], 'name'])}")


if __name__ == "__main__":
    main()
