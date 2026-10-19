"""
code_size_undo.py -- the paper's code-size sweep with macro + undo merging off and on.

The sweep (`experiments/figures/merging_improvement/merging_improvement_code_size.py`,
the text's "Sweeping k from 2 to 10" paragraph) reports that merging decays with k
because a whole-patch macro needs a gate on every occupied lane. `HH_MAJORITY_UNDO` and
`TRANSVERSAL_MAJORITY_UNDO` relax that to a majority plus undo gates, so this reruns the
sweep's own `run_experiment` per (circuit, k) in a worker with an arm's switches set --
the planner's verify step and every translation see the same merging -- and adds, per
point, how many macros fired and how many of them needed undo gates, at the naive and at
the packed placement.

    PYTHONPATH=. python -m experiments.tvd_explore.code_size_undo --arm shipped
    PYTHONPATH=. python -m experiments.tvd_explore.code_size_undo --arm undo

Writes results/tvd_explore/code_size_<arm>.csv (the sweep's schema plus the counts).
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import collections
import logging
import time
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")

ARMS = {
    "shipped": {},
    "undo": {"HH_MAJORITY_UNDO": True, "TRANSVERSAL_MAJORITY_UNDO": True},
    "hh_undo": {"HH_MAJORITY_UNDO": True},
    # Every arm layer-sorted, so a merge that fires is not also credited with the re-sort
    # the rewriter applies when it does (`explore_switches.LAYER_ORDER_ALL_ARMS`).
    "shipped_layer": {"LAYER_ORDER_ALL_ARMS": True},
    "undo_layer": {"HH_MAJORITY_UNDO": True, "TRANSVERSAL_MAJORITY_UNDO": True,
                   "LAYER_ORDER_ALL_ARMS": True},
}

MACROS = ("hh", "xx", "zz", "cxt", "czt", "xcxt")


def merge_counts(program, placement, n_code):
    """Macros the merging pass emits at `placement`, and how many carry undo gates."""
    from translator.merging import preprocessing
    from translator.passes import cancel_inverse_gates, commute_hadamards

    fired = collections.Counter()
    original = preprocessing.build_rewritten_circuit

    def counting(circuit, dag, records, layer_index, order_index):
        for record in records:
            name = record.macro_gate.name
            fired[name] += 1
            if record.undo_ops:
                fired[f"{name}_undo"] += 1
                fired["undo_gates"] += len(record.undo_ops)
        return original(circuit, dag, records, layer_index, order_index)

    preprocessing.build_rewritten_circuit = counting
    try:
        processed = cancel_inverse_gates(commute_hadamards(cancel_inverse_gates(program)))
        preprocessing.preprocess_program_for_merging_with_stats(
            processed, list(placement), allow_transversal_swap_merges=True, n_code=n_code
        )
    finally:
        preprocessing.build_rewritten_circuit = original
    return fired


def point(job):
    import experiments.figures.merging_improvement.merging_improvement_code_size as cs
    from translator import explore_switches

    group, name, circuit, k, switches = job
    explore_switches.configure(**switches)
    captured = {}
    planner = cs.packed_permutation

    def capturing(program_circuit, n_code, settings, **kwargs):
        placement, note = planner(program_circuit, n_code, settings, **kwargs)
        captured["packed"] = placement
        return placement, note

    cs.packed_permutation = capturing
    began = time.time()
    try:
        settings = cs.CodeSizeSettings(k_values=(k,))
        frame = cs.run_experiment([(name, circuit)], group, settings)
    finally:
        cs.packed_permutation = planner
    if frame.empty:
        return None
    n_code = k + 2
    trivial = cs.tight_trivial_packing(circuit.num_qubits, k)
    packed = captured.get("packed") or trivial
    for arm, placement in (("naive", trivial), ("packed", packed)):
        counts = merge_counts(circuit, placement, n_code)
        for macro in MACROS:
            frame[f"{arm}_{macro}"] = counts.get(macro, 0)
            frame[f"{arm}_{macro}_undo"] = counts.get(f"{macro}_undo", 0)
        frame[f"{arm}_undo_gates"] = counts.get("undo_gates", 0)
    frame["packed_placement"] = str([tuple(p) for p in packed])
    frame["point_seconds"] = round(time.time() - began, 2)
    return frame


def main():
    import experiments.figures.merging_improvement.merging_improvement_code_size as cs
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--groups", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--k", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    settings = cs.CodeSizeSettings()
    jobs = []
    for group in args.groups:
        for name, circuit in cs.load_group_circuits(group, settings):
            if args.names and name not in args.names:
                continue
            for k in args.k:
                jobs.append((group, name, circuit, k, ARMS[args.arm]))
    # Widest first, and small k first within a width: the slow plans start early.
    jobs.sort(key=lambda j: (-j[2].num_qubits, j[3]))
    print(f"[code_size_undo] arm={args.arm}: {len(jobs)} points")
    frames = parallel_map(point, jobs, jobs=args.jobs, label=f"code-size:{args.arm}")
    failed = [f"{j[1]}@k{j[3]}" for j, f in zip(jobs, frames) if f is None]
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"code_size_{args.tag or args.arm}.csv"
    df.to_csv(path, index=False)
    print(f"[code_size_undo] wrote {path}: {len(df)} rows; {len(failed)} points failed {failed[:10]}")
    cs.print_summary(df, ["1q", "2q", "depth"], settings)


if __name__ == "__main__":
    main()
