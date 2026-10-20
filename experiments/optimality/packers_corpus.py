"""
packers_corpus.py -- Experiment1's packer comparison, redone under the
current standard with the logical optimisation pass, on Fig 6's 71 circuits.

For every circuit, every arm of `standard_packers.ARMS` except simulated annealing (an optimum
estimate, run on the larger example in `sampled_front.py`) at AQORN's H-commutation decision,
measured by `placement_measure.measure`: static counts of the original's translation, AQORN's own
objective, frames on the Clifford proxy at alpha = 1 and at the circuit's Fig 6 alpha (LSR,
occupied-lane logical error rate, exact TVD), and the noiseless correctness check.

Output `results/optimality_front/packers_corpus.csv` (one row per circuit and arm) + sidecar.
Circuits run widest first; a circuit's row set is written as soon as it finishes
(`results/optimality_front/packers_corpus_parts/`), so a rerun skips finished circuits.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.packers_corpus --jobs 12
    ... --names adder_n10 bell_n4          # a subset
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

NAME = "packers_corpus"
PLANS = "experiments/figures/ablation_standard/ablation_standard_logical_v2.csv"


def _run_circuit(job) -> List[Dict]:
    from experiments.optimality.placement_measure import logical_support, measure
    from experiments.optimality.standard_packers import ARMS, run_arms

    name, group, original, proxy, alpha_fig6, shots, random_seconds, part, chosen, annealing_s = job
    part = Path(part)
    if part.exists():
        return pd.read_csv(part).to_dict("records")
    began = time.time()
    arms, notes, commute = run_arms(original, arms=chosen, random_seconds=random_seconds,
                                    annealing_seconds=annealing_s)
    plan_seconds = time.time() - began
    support = logical_support(proxy)
    from experiments.optimality.standard_packers import make_standard_context, standard_gadgets
    from experiments.revision.packers import model_value

    context = make_standard_context(original, commute)
    rows = []
    for arm, placement in arms.items():
        row = {"circuit": name, "size_group": group, "num_qubits": original.num_qubits,
               "arm": arm, "commute": commute, "support_bits": len(support[1]),
               "alpha_fig6": alpha_fig6, "plan_seconds": round(plan_seconds, 1),
               "arm_seconds": notes.get(f"{arm}_seconds"), "arm_error": notes.get(f"{arm}_error"),
               "placement": json.dumps([list(p) for p in placement]) if placement else None}
        if placement:
            with standard_gadgets():   # AQORN's search objective, on this placement's groups
                row["model_value"] = model_value(context, placement)
            try:
                row.update(measure(original, proxy, placement, commute, support,
                                   alphas={"a1": 1.0, "fig6": alpha_fig6}, shots=shots,
                                   seed_key=f"{name}:{arm}"))
            except Exception as error:  # noqa: BLE001
                row["error"] = f"{type(error).__name__}: {error}"[:200]
        rows.append(row)
    for key in ("aqorn_fell_back", "random_budget_drawn", "random_budget_capped",
                "hypergraph_24_fallback", "annealing_restarts", "annealing_sweeps"):
        for row in rows:
            row[key] = notes.get(key)
    frame = pd.DataFrame(rows)
    frame["circuit_seconds"] = round(time.time() - began, 1)
    part.parent.mkdir(parents=True, exist_ok=True)
    temporary = part.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.rename(part)
    return frame.to_dict("records")


def main() -> None:
    from experiments.revision.common import load_fig6_circuits, write_sidecar
    from experiments.shared.parallel import parallel_map
    from translator.logical_optimization import state

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--random-seconds", type=float, default=60.0)
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--arms", nargs="*", default=None,
                        help="arms to run (default: every arm but annealing)")
    parser.add_argument("--annealing-seconds", type=float, default=120.0)
    parser.add_argument("--tag", default="", help="suffix for the parts folder and the CSV")
    parser.add_argument("--out", default="results/optimality_front")
    args = parser.parse_args()
    began = time.time()
    out = Path(args.out)
    print(f"[{NAME}] logical optimisation: {state()}", flush=True)
    alphas = pd.read_csv(PLANS).drop_duplicates("circuit_name").set_index("circuit_name")
    circuits = [c for c in load_fig6_circuits() if not args.names or c.name in args.names]
    from experiments.optimality.standard_packers import ARMS

    chosen = args.arms or [a for a in ARMS if a != "annealing"]
    stem = f"{NAME}_{args.tag}" if args.tag else NAME
    jobs = [(c.name, c.size_group, c.original, c.proxy, float(alphas.loc[c.name, "alpha_final"]),
             args.shots, args.random_seconds, str(out / f"{stem}_parts" / f"{c.name}.csv"),
             chosen, args.annealing_seconds)
            for c in circuits]
    jobs.sort(key=lambda j: -j[2].num_qubits)
    results = parallel_map(_run_circuit, jobs, jobs=args.jobs, label=NAME)
    failed = [j[0] for j, r in zip(jobs, results) if r is None]
    frame = pd.DataFrame([row for result in results if result for row in result])
    path = out / (f"{stem}.csv" if not args.names else f"{stem}_subset.csv")
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment=NAME, settings=vars(args), started=began, status="complete",
                  extra={"failed": failed})
    wrong = frame[frame.get("correct") == False] if "correct" in frame else frame.iloc[:0]  # noqa: E712
    print(f"[{NAME}] {frame['circuit'].nunique()} circuits, {len(frame)} rows -> {path}; failed "
          f"{failed}; miscompiled placements {len(wrong)} {sorted(set(zip(wrong.circuit, wrong.arm)))[:10]}",
          flush=True)


if __name__ == "__main__":
    main()
