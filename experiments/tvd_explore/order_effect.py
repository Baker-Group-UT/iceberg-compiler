"""
order_effect.py -- what the merging rewriter's re-sort costs Fig 6, in gates.

Whenever a macro fires, `translator.merging.rewrite.build_rewritten_circuit` emits the
whole circuit in ASAP-layer order. `MERGE_KEEP_PROGRAM_ORDER` re-linearizes it as close
to program order as dependencies allow, which is the same unitary but changes which
physical gates end up adjacent, so what the peephole cancellation can remove. This counts
Fig 6's static panels (gates and depth of the original circuits, Fig 6's own
`compute_translation_metrics`) for the merging-enabled stages, re-sort as shipped against
program order, at a plan table's placements and with optional extra switches.

    PYTHONPATH=. python -m experiments.tvd_explore.order_effect --plans aqorn
    PYTHONPATH=. python -m experiments.tvd_explore.order_effect --plans csv:results/tvd_explore/plans_w10T.csv \
        --switch TRANSVERSAL_INTRA_TWO_QUBIT --tag w10T
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


#: Emission-order normalizations: the shipped mix (baseline and +H in program order,
#: merging stages in the rewriter's layer order), program order everywhere, layer order
#: everywhere.
ORDERS = {
    "shipped": {},
    "program": {"MERGE_KEEP_PROGRAM_ORDER": True},
    "layer": {"LAYER_ORDER_ALL_ARMS": True},
}


def point(job):
    import experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated as fig6
    from pipeline.config import PipelineConfig
    from translator import explore_switches

    circuit, placement, commute, switches = job
    resource = fig6.ensure_measurements(circuit.original)
    n = resource.num_qubits
    tight = fig6.tight_trivial_packing(n, math.ceil(n / 2))
    names = set(switches) | {k for spec in ORDERS.values() for k in spec}

    def counts(perm, commute_, merge, order):
        explore_switches.configure(**{s: merge for s in switches}, **ORDERS[order])
        cfg = PipelineConfig()
        cfg.translator_config.enable_h_commutation = commute_
        cfg.translator_config.enable_merging = merge
        try:
            return fig6.compute_translation_metrics(resource, perm, cfg)
        finally:
            explore_switches.configure(**{s: False for s in names})

    row = {"name": circuit.name, "size_group": circuit.size_group, "num_qubits": n}
    for order in ORDERS:
        for arm, perm, com, merge in (("base", tight, False, False), ("h_only", tight, commute, False),
                                      ("h_merge", tight, commute, True), ("packed", placement, commute, True)):
            values = counts(perm, com, merge, order)
            row.update({f"{arm}_{order}_{k}": v for k, v in zip(("1q", "2q", "depth"), values)})
    return row


def load_plans(spec: str) -> pd.DataFrame:
    if spec == "aqorn":
        frame = pd.read_csv("results/revision/exp8/planner_static.csv")
        return frame[frame["silent_weight"] == 0.0][["name", "placement", "commute"]]
    if spec.startswith("w"):
        frame = pd.read_csv("results/revision/exp8/planner_static.csv")
        return frame[frame["silent_weight"] == float(spec[1:])][["name", "placement", "commute"]]
    if spec.startswith("csv:"):
        return pd.read_csv(spec[4:])[["name", "placement", "commute"]]
    raise SystemExit(f"unknown plan source {spec!r}")


def main():
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", default="aqorn")
    parser.add_argument("--switch", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    plans = load_plans(args.plans).set_index("name")
    circuits = load_fig6_circuits(names=list(plans.index))
    jobs = sorted([(c, [tuple(p) for p in json.loads(plans.loc[c.name, "placement"])],
                    bool(plans.loc[c.name, "commute"]), tuple(args.switch)) for c in circuits],
                  key=lambda j: -j[0].num_qubits)
    rows = [r for r in parallel_map(point, jobs, jobs=args.jobs, label="order") if r]
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"order_effect_{args.tag or args.plans.replace(':', '_').replace('/', '_')}.csv"
    frame.to_csv(path, index=False)
    print(f"wrote {path}: {len(frame)} circuits")
    for metric in ("1q", "2q", "depth"):
        print(f"  {metric}: mean reduction vs that order's own baseline (Fig 6 caption convention)")
        for order in ORDERS:
            base = frame[f"base_{order}_{metric}"]
            parts = []
            for arm in ("h_only", "h_merge", "packed"):
                parts.append(f"{arm} {(100 * (base - frame[f'{arm}_{order}_{metric}']) / base).mean():5.1f}%")
            print(f"    {order:8s} baseline mean {base.mean():9.1f}   " + "   ".join(parts))

if __name__ == "__main__":
    main()
