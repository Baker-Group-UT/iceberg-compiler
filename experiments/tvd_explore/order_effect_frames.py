"""
order_effect_frames.py -- how much of Fig 6's LSR / TVD gain is emission order?

`order_effect` showed the depth panel moves with emission order: the merging rewriter
re-sorts into ASAP layers, which only the merging arms get. Idle noise charges depth, so
the simulated metrics may move too. Frame simulation (silent per kept shot is the TVD
stand-in, within 1-2 points of Fig 6, `fastframes`) of Fig 6's baseline and AQORN arms,
each emitted as shipped and with every arm layer-sorted (`LAYER_ORDER_ALL_ARMS`), at the
circuit's Fig 6 alpha and at alpha = 1.

    PYTHONPATH=. python -m experiments.tvd_explore.order_effect_frames
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import math
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")


def point(job):
    from experiments.shared.simulation import ensure_measurements
    from experiments.tvd_explore.fastframes import splits
    from experiments.tvd_explore.screen import _translate
    from translator import explore_switches

    circuit, placement, commute, alpha6, shots, switches = job
    proxy = ensure_measurements(circuit.proxy)
    n = proxy.num_qubits
    tight = [(i // 2, i % 2) for i in range(n)]
    row = {"name": circuit.name, "size_group": circuit.size_group, "num_qubits": n, "alpha6": alpha6}
    arms = (("base", tight, False, False), ("h_merge", tight, commute, True), ("packed", placement, commute, True))
    for order, layer in (("shipped", False), ("layer", True)):
        for arm, perm, com, merge in arms:
            explore_switches.configure(LAYER_ORDER_ALL_ARMS=layer, **{s: merge for s in switches})
            try:
                encoded = _translate(proxy, perm, commute=com, merging=merge, overrides={})
                measured = splits(encoded, (alpha6, 1.0), shots=shots, seed=29)
            finally:
                explore_switches.configure(LAYER_ORDER_ALL_ARMS=False, **{s: False for s in switches})
            row[f"{arm}_{order}_depth"] = encoded.depth()
            for tag, a in (("fig6", alpha6), ("a1", 1.0)):
                row[f"{arm}_{order}_kept_{tag}"] = 1 - measured[a]["p_detected"]
                row[f"{arm}_{order}_silent_{tag}"] = measured[a]["p_silent_given_kept"]
    return row


def main():
    from experiments.revision.common import fig6_reference, load_fig6_circuits
    from experiments.shared.parallel import parallel_map
    from experiments.tvd_explore.order_effect import load_plans

    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", default="aqorn")
    parser.add_argument("--switch", action="append", default=[])
    parser.add_argument("--shots", type=int, default=40_000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    ref = fig6_reference().set_index("name")
    plans = load_plans(args.plans).set_index("name")
    circuits = load_fig6_circuits(names=list(plans.index))
    jobs = sorted([(c, [tuple(p) for p in json.loads(plans.loc[c.name, "placement"])],
                    bool(plans.loc[c.name, "commute"]), float(ref.loc[c.name, "alpha_final"]),
                    args.shots, tuple(args.switch)) for c in circuits], key=lambda j: -j[0].num_qubits)
    frame = pd.DataFrame([r for r in parallel_map(point, jobs, jobs=args.jobs, label="order-frames") if r])
    path = OUT / f"order_effect_frames_{args.tag or args.plans}.csv"
    frame.to_csv(path, index=False)
    print(f"wrote {path}: {len(frame)} circuits")
    for tag in ("fig6", "a1"):
        print(f"  alpha {tag}: mean over circuits, each order against its own baseline")
        for order in ("shipped", "layer"):
            for arm in ("h_merge", "packed"):
                lsr = (frame[f"{arm}_{order}_kept_{tag}"] / frame[f"base_{order}_kept_{tag}"] - 1) * 100
                tvd = (1 - frame[f"{arm}_{order}_silent_{tag}"] / frame[f"base_{order}_silent_{tag}"]) * 100
                print(f"    {order:8s} {arm:8s} LSR {lsr.mean():+6.1f}%   silent/kept reduction {tvd.mean():5.1f}%")
        lsr_b = (frame[f"base_layer_kept_{tag}"] / frame[f"base_shipped_kept_{tag}"] - 1) * 100
        tvd_b = (1 - frame[f"base_layer_silent_{tag}"] / frame[f"base_shipped_silent_{tag}"]) * 100
        print(f"    baseline layer-sorted vs as shipped: LSR {lsr_b.mean():+6.1f}%   silent/kept {tvd_b.mean():5.1f}%")


if __name__ == "__main__":
    main()
