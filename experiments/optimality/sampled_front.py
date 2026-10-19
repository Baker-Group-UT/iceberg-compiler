"""
sampled_front.py -- the optimality-front question on a program too large to enumerate, together
with the conventional packers: where AQORN's placement lands among thousands of random placements
and next to every other packer, on kept shots against logical error.

Program: `multiplier_n15` from Fig 6's corpus (15 qubits, 8 patches at [[4,2,2]]; packing beats
naive there by 2q -11%, LSR +32%, TVD -21% in Fig 6 under the standard). Planned on the original,
simulated on its Clifford proxy, as Fig 6 does; single-outcome output, so TVD equals logical error.
Everything is the current standard with the logical optimisation pass; the compiler is frozen.

Placements, all at AQORN's H-commutation decision:
    random          `--random` uniform placements: random pairing (one solo qubit), random lanes,
                    random patch order -- the space AQORN searches, sampled
    aqorn_grouping  `--grouping` placements with AQORN's grouping and random lanes and patch
                    order: what the choice of *which atoms move* is worth at 8 patches
    arm:<name>      every packer in `standard_packers.ARMS` (AQORN, its ablations, simulated
                    annealing on AQORN's objective, and the conventional baselines)

Each is measured by `placement_measure.measure`: static counts, AQORN's objective, frames at
alpha = 1 and at the circuit's Fig 6 alpha (with independent selection shots), and the noiseless
correctness check. Output `results/optimality_front/sampled_front_<program>.csv` (+ sidecar).

    PYTHONPATH=. venv/bin/python -m experiments.optimality.sampled_front --jobs 12
    ... --smoke
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
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

NAME = "sampled_front"
PLANS = "experiments/figures/ablation_standard/ablation_standard_logical_v2.csv"


def random_placement(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    qubits = list(range(n))
    rng.shuffle(qubits)
    groups = [qubits[i:i + 2] for i in range(0, n, 2)]
    order = list(range(len(groups)))
    rng.shuffle(order)
    placement = [None] * n
    for patch, group in zip(order, groups):
        lanes = [0, 1]
        rng.shuffle(lanes)
        for qubit, lane in zip(group, lanes):
            placement[qubit] = (patch, lane)
    return placement


def regrouped(placement, rng: random.Random) -> List[Tuple[int, int]]:
    """The same grouping with random patch order and random lanes."""
    groups: Dict[int, List[int]] = {}
    for qubit, (patch, _) in enumerate(placement):
        groups.setdefault(patch, []).append(qubit)
    order = list(range(len(groups)))
    rng.shuffle(order)
    out = [None] * len(placement)
    for patch, members in zip(order, groups.values()):
        lanes = [0, 1]
        rng.shuffle(lanes)
        for qubit, lane in zip(members, lanes):
            out[qubit] = (patch, lane)
    return out


def _measure_chunk(job) -> List[Dict]:
    from experiments.optimality.placement_measure import measure

    original, proxy, commute, support, alphas, shots, items = job
    rows = []
    for kind, label, placement in items:
        row = {"kind": kind, "label": label, "placement": json.dumps([list(p) for p in placement]),
               "commute": commute}
        try:
            row.update(measure(original, proxy, placement, commute, support, alphas=alphas,
                               shots=shots, seed_key=f"{kind}:{label}"))
        except Exception as error:  # noqa: BLE001 - a rejected placement is recorded
            row["error"] = f"{type(error).__name__}: {error}"[:200]
        rows.append(row)
    return rows


def main() -> None:
    from experiments.optimality.placement_measure import logical_support
    from experiments.optimality.standard_packers import ARMS, run_arms
    from experiments.revision.common import load_fig6_circuits, write_sidecar
    from experiments.shared.parallel import parallel_map
    from translator.logical_optimization import state

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--program", default="multiplier_n15")
    parser.add_argument("--random", type=int, default=2000)
    parser.add_argument("--grouping", type=int, default=300)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--annealing-seconds", type=float, default=300.0)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="results/optimality_front")
    args = parser.parse_args()
    if args.smoke:
        args.random, args.grouping, args.shots, args.annealing_seconds = 6, 3, 20_000, 10.0
    began = time.time()
    print(f"[{NAME}] logical optimisation: {state()}; program {args.program}", flush=True)

    circuit = next(c for c in load_fig6_circuits() if c.name == args.program)
    original, proxy = circuit.original, circuit.proxy
    alpha_fig6 = float(pd.read_csv(PLANS).drop_duplicates("circuit_name")
                       .set_index("circuit_name").loc[args.program, "alpha_final"])
    alphas = {"a1": 1.0, "fig6": alpha_fig6}
    support = logical_support(proxy)
    print(f"[{NAME}] {original.num_qubits} qubits, Fig 6 alpha {alpha_fig6:.3f}, ideal support "
          f"2^{len(support[1])}", flush=True)

    arms, notes, commute = run_arms(original, arms=ARMS,
                                    annealing_seconds=args.annealing_seconds)
    print(f"[{NAME}] arms planned (commute={commute}): "
          + ", ".join(f"{k}{'' if v else ' FAILED'}" for k, v in arms.items()), flush=True)

    rng = random.Random(args.seed)
    items: List[Tuple[str, str, list]] = [("arm", name, p) for name, p in arms.items() if p]
    items += [("random", str(i), random_placement(original.num_qubits, rng))
              for i in range(args.random)]
    items += [("aqorn_grouping", str(i), regrouped(arms["aqorn"], rng))
              for i in range(args.grouping)]
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    jobs = [(original, proxy, commute, support, alphas, args.shots, chunk) for chunk in chunks]
    results = parallel_map(_measure_chunk, jobs, jobs=args.jobs, label=NAME)
    rows = [row for result in results if result for row in result]
    frame = pd.DataFrame(rows)
    frame["program"] = args.program
    path = Path(args.out) / f"{NAME}_{args.program}{'_smoke' if args.smoke else ''}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment=NAME, settings=vars(args), started=began, status="complete",
                  extra={"notes": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                   for k, v in notes.items()},
                         "alphas": alphas, "commute": commute,
                         "failed_chunks": sum(r is None for r in results)})
    bad = frame[frame.get("correct", pd.Series(True, index=frame.index)) == False]  # noqa: E712
    print(f"[{NAME}] {len(frame)} placements -> {path}; miscompiled {len(bad)}; errors "
          f"{int(frame['error'].notna().sum()) if 'error' in frame else 0}", flush=True)


if __name__ == "__main__":
    main()
