"""
budget.py -- where weight 10's remaining silent error comes from.

Per Fig 6 circuit and plan (AQORN = weight 0, silent-aware = weight 10, from Experiment8a):

  predicted first-order silent error at alpha = 1, by source, from the gadget rates:
      inpatch_2q   in-patch CX/CZ/XCX gadgets
      one_qubit    single-lane H, S, SX gadgets (before HH credit)
      hh_credit    silent error removed by fired whole-patch HH
  measured (stim.FlipSimulator, frames.py) at alpha = 1 and at Fig 6's alpha:
      p_detected, p_silent, p_silent_given_kept

The gap between predicted and measured silent at alpha = 1 is what first order misses
(pairs of faults whose syndromes cancel, idle and movement inside gadgets).
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import logging
import sys
import time
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")


def components(proxy, placement, commute, rates, noise):
    from experiments.revision.exp8_undetected.cost_model import (
        build_costs, groups_of, silent_constant)
    from packing.tables import GATE_KEYS

    profiles, priced = build_costs(proxy, 4, commute=commute, noise=noise, rates=rates,
                                   silent_weight=1.0)
    _, plain = build_costs(proxy, 4, commute=commute, noise=noise, rates=rates,
                           silent_weight=0.0)
    groups = groups_of(placement)
    owner = {q: i for i, g in enumerate(groups) for q in g}
    inpatch = 0.0
    for (a, b), events in plain.gates.items():
        if owner.get(a) == owner.get(b):
            inpatch += sum(rates.intra.get(GATE_KEYS[k], 0.0) for _, k in sorted(set(events)))
    placement_part = (priced.of(groups) - plain.of(groups)) / priced.silent_scale
    return {"inpatch_2q": inpatch, "one_qubit": silent_constant(profiles, rates),
            "hh_credit": inpatch - placement_part}


def point(job):
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.exp8_undetected.frames import detection_split, translate
    from experiments.revision.exp8_undetected.gadgets import load_rates
    from experiments.shared.simulation import ensure_measurements

    circuit, plans, alpha6 = job
    rates = load_rates(4)
    noise = fig6_noise_profile()
    proxy = ensure_measurements(circuit.proxy)
    rows = []
    for label, (placement, commute) in plans.items():
        placement = [tuple(p) for p in placement]
        row = {"name": circuit.name, "size_group": circuit.size_group,
               "num_qubits": circuit.num_qubits, "plan": label, "commute": commute,
               **components(proxy, placement, commute, rates, noise)}
        encoded = translate(proxy, placement, 4, commute=commute)
        began = time.time()
        for tag, alpha in (("a1", 1.0), ("fig6", alpha6)):
            split = detection_split(encoded, alpha=alpha, shots=20_000, seed=7)
            row.update({f"{k}_{tag}": v for k, v in split.items() if k != "frame_shots"})
        row["frame_seconds"] = time.time() - began
        rows.append(row)
    return rows


def main():
    from experiments.revision.common import fig6_reference, load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    plans_csv = pd.read_csv("results/revision/exp8/planner_static.csv")
    ref = fig6_reference().set_index("name")
    names = sys.argv[1:] or None
    circuits = load_fig6_circuits(names=names)
    jobs = []
    for c in circuits:
        block = plans_csv[plans_csv["name"] == c.name].set_index("silent_weight")
        plans = {"w0": (json.loads(block.loc[0.0, "placement"]), bool(block.loc[0.0, "commute"])),
                 "w10": (json.loads(block.loc[10.0, "placement"]), bool(block.loc[10.0, "commute"]))}
        jobs.append((c, plans, float(ref.loc[c.name, "alpha_final"])))
    jobs.sort(key=lambda j: -j[0].num_qubits)
    results = parallel_map(point, jobs, jobs=12, label="budget")
    rows = [r for res in results if res for r in res]
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "budget.csv", index=False)
    print(f"wrote {OUT / 'budget.csv'} ({frame['name'].nunique()} circuits, "
          f"{sum(r is None for r in results)} failed)")


if __name__ == "__main__":
    main()
