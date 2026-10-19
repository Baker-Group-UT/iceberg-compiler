"""
movement_cap_sweep.py -- re-measure MOVEMENT_SHARE_CEILING on the grouped-transport pricing.

The committed value (0.25) and the sweep table in `packing/cost.py` were fitted against
the per-physical-gate movement charge, where one cross-patch logical CNOT paid four round
trips, `mean_patch_hops` was unit-pitch, and `noise_profile_for` priced a crossing at
13 x distance_scalar per hop with distance_scalar applied twice. All four of those moved:
movement is one grouped trip per crossing, hops are real cells, the bridge derives its
coefficient from `NoiseKnobs`, and the scalar enters once. This sweep reprices the cap on
that axis.

Same protocol as the original: QASMBench small proxies at k=2, packing planned under each
(cap, profile) cell, packed and naive translations compared. Reported per cell: mean depth
reduction over naive, mean 2q reduction, and how many circuits came back metrically
unchanged (the failure mode the cap exists to prevent -- movement swamping the objective
and the search returning naive).

Run with:

    PYTHONPATH=. python experiments/tests/movement_cap_sweep.py
    PYTHONPATH=. python experiments/tests/movement_cap_sweep.py --limit 12
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.simulation import SimSettings, noise_profile_for
from packing import cost as packing_cost
from packing.plan import plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]
N_CODE = 4

CAPS = (float("inf"), 0.5, 0.25, 0.15, 0.08)
#: Profiles to sweep: the machine as simulated (scalars 1/1), and the two ratio
#: extremes the old table used, for continuity with its axis.
PROFILES = (
    ("defaults", SimSettings()),
    ("ratio 1/16", SimSettings().with_noise_ratio(1.0 / 16.0)),
    ("ratio 16", SimSettings().with_noise_ratio(16.0)),
)


def tight(num_qubits: int, n_codes: int) -> Perm:
    cap = max(1, math.ceil(num_qubits / n_codes))
    return [(min(i // cap, n_codes - 1), i % cap) for i in range(num_qubits)]


def cfg(commute: bool) -> PipelineConfig:
    c = PipelineConfig(verbose=False, save=False)
    c.translator_config.n_code = N_CODE
    c.translator_config.enable_merging = True
    c.translator_config.enable_h_commutation = commute
    return c


def metrics(program, perm, commute: bool) -> Tuple[int, int, int]:
    circuit = translate_permutation(program, perm, config=cfg(commute)).circuit
    one, two = one_two_qubit_gate_totals(circuit)
    return one, two, circuit.depth()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12, help="circuits to sweep")
    args = parser.parse_args()

    circuits = load_proxy_circuits(
        Path("benchmarks/qasmbench") / "small", gate_limit=2000, qubit_limit=100,
        timeout_seconds=1.0,
    )[: args.limit]
    print(f"{len(circuits)} circuits at n_code={N_CODE} (k=2)\n")

    naive_metrics: Dict[str, Tuple[int, int, int, Perm, bool]] = {}
    for name, program in circuits:
        nq = program.num_qubits
        base = tight(nq, math.ceil(nq / 2))
        probe = plan_packing(program, N_CODE, h_commutation="auto", verify=True)
        commute = probe.h_commutation
        naive_metrics[name] = (*metrics(program, base, commute), base, commute)

    header = f"{'profile':>12} {'cap':>6} {'depth%':>8} {'2q%':>7} {'unchanged':>10}"
    print(header)
    for label, settings in PROFILES:
        profile = noise_profile_for(settings)
        for cap in CAPS:
            packing_cost.MOVEMENT_SHARE_CEILING = cap
            depth_gains, two_gains, unchanged = [], [], 0
            for name, program in circuits:
                n1, n2, nd, base, commute = naive_metrics[name]
                plan = plan_packing(
                    program, N_CODE, noise=profile,
                    h_commutation=commute, verify=True,
                )
                p1, p2, pd = metrics(program, plan.placement, commute)
                depth_gains.append(0.0 if nd <= 0 else 100.0 * (nd - pd) / nd)
                two_gains.append(0.0 if n2 <= 0 else 100.0 * (n2 - p2) / n2)
                if (p1, p2, pd) == (n1, n2, nd):
                    unchanged += 1
            cap_label = "none" if math.isinf(cap) else f"{cap:g}"
            print(
                f"{label:>12} {cap_label:>6} "
                f"{sum(depth_gains) / len(depth_gains):>8.2f} "
                f"{sum(two_gains) / len(two_gains):>7.2f} "
                f"{unchanged:>7}/{len(circuits)}"
            )
        print()


if __name__ == "__main__":
    main()
