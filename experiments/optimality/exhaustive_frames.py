"""
exhaustive_frames.py -- the small-program optimality check on the measured objective.

`exhaustive_silent.py` scores every placement on the planner's objective; what the paper claims is
accuracy (TVD) and kept shots (LSR). This measures every placement of the complete tightest space,
on Fig 6's machine with every program layer-sorted (fair schedule), by Pauli frames
(`stim.FlipSimulator`, the noisy circuit the simulator would run, `tvd_explore.fastframes`):

    lsr      shots kept by post-selection
    silent   P(logical error | kept), the TVD stand-in (TVD ~ 0.95 x silent per kept, memory note)

Against the winner's curse (picking the best of many noisy estimates overstates it -- seen on the
TVD exploration), two stages as in Experiment7b: every placement at `screen_shots`; then the 10 best
by silent, the 10 best by LSR and every arm again at `final_shots` with fresh seeds, and "best" is
decided on the fresh measurement only. H commutation is fixed to the weight-10 planner's choice for
the whole space (the planner's decision; 7b did the same with v4's). Noise at alpha = 1 (fixed, not
Fig 6's per-circuit calibration, which is extreme on small circuits).

Arms: naive, v4, w10, interaction-count partition (shipped translator) and w10T (its switch on --
compared against the best placement under the shipped translator, so a negative regret is possible).

    PYTHONPATH=. venv/bin/python -m experiments.optimality.exhaustive_frames --jobs 3
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import math
import time
import warnings
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.optimality.exhaustive_silent import RATES, RATES_T, Settings as _Base, jobs_for, switched

NAME = "exhaustive_frames"


@dataclass
class Settings:
    configs: Sequence[Tuple[int, int]] = ((4, 2), (5, 2), (6, 2))
    random_circuits: int = 10
    circuit_seed: int = 20260730
    gates_per_qubit: int = 10
    locality: float = 0.85
    qasmbench: bool = True
    silent_weight: float = 10.0
    smoke: bool = False
    alpha: float = 1.0
    screen_shots: int = 20_000
    final_shots: int = 200_000
    finalists: int = 10
    jobs: int = 3
    out: str = "results/optimality"


def _encoded(program, placement, commute: bool):
    from experiments.revision.common import translator_config
    from pipeline.tasks.translate_permutation import translate_permutation

    config = translator_config(4, commute=commute, merging=True)
    config.translator_config.asap_order = True
    return translate_permutation(program, [tuple(p) for p in placement], config=config).circuit


def _frames(circuit, alpha: float, shots: int, seed: int) -> Dict[str, float]:
    from experiments.revision.exp8_undetected.frames import split_from_stim
    from experiments.tvd_explore.fastframes import at_alpha_one, rescale

    noisy = rescale(at_alpha_one(circuit), alpha)
    split = split_from_stim(circuit, noisy, shots=shots, seed=seed)
    return {"lsr": 1.0 - split["p_detected"], "silent": split["p_silent_given_kept"]}


def _run_point(job) -> List[Dict]:
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from experiments.revision.packers import interaction_partition, make_context
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing, plan_packing

    settings_dict, q, k, circuit_id, source, name, seed, circuit = job
    settings = Settings(**settings_dict)
    program = ensure_measurements(circuit)
    noise = fig6_noise_profile()
    base_seed = zlib.crc32(f"{q}:{circuit_id}".encode()) % 1_000_000
    with switched(False):
        rates = UndetectedRates.from_table(pd.read_csv(RATES), 4, 3e-3)
        w10 = plan_silent_aware(program, 4, rates=rates, silent_weight=settings.silent_weight,
                                noise=noise)
        commute = bool(w10.h_commutation)
        v4 = plan_packing(program, 4, h_commutation=commute, verify=True, noise=noise)
        arms = {"w10": [tuple(p) for p in w10.placement], "v4": [tuple(p) for p in v4.placement],
                "naive": naive_packing(q, k)}
        try:
            arms["interaction_partition"] = [tuple(p) for p in interaction_partition(
                make_context(program, 4, commute, noise))]
        except Exception:  # noqa: BLE001
            pass
        screen = []
        for index, (placement, _) in enumerate(exhaustive_space(q, k, "tightest")):
            placement = [tuple(p) for p in placement]
            values = _frames(_encoded(program, placement, commute), settings.alpha,
                             settings.screen_shots, base_seed + index)
            screen.append({"index": index, "placement": placement, **values})
    with switched(True):
        rates_t = UndetectedRates.from_table(pd.read_csv(RATES_T), 4, 3e-3)
        w10T = plan_silent_aware(program, 4, rates=rates_t, silent_weight=settings.silent_weight,
                                 noise=noise, h_commutation=commute)
        w10T_placement = [tuple(p) for p in w10T.placement]
        w10T_final = _frames(_encoded(program, w10T_placement, commute), settings.alpha,
                             settings.final_shots, base_seed + 999_983)

    frame = pd.DataFrame(screen)
    chosen = set(frame.nsmallest(settings.finalists, "silent")["index"]) | set(
        frame.nlargest(settings.finalists, "lsr")["index"])
    key_of = {json.dumps(r["placement"]): r["index"] for r in screen}
    for placement in arms.values():
        chosen.add(key_of[json.dumps([tuple(p) for p in placement])])
    final = {}
    with switched(False):
        for index in sorted(chosen):
            placement = screen[index]["placement"]
            final[index] = _frames(_encoded(program, placement, commute), settings.alpha,
                                   settings.final_shots, base_seed + 500_000 + index)
    best_silent = min(v["silent"] for v in final.values())
    best_lsr = max(v["lsr"] for v in final.values())
    shared = {"num_qubits": q, "config": f"{q}q,k={k}", "circuit_id": circuit_id, "source": source,
              "name": name, "commute": commute, "space": len(screen), "finalists": len(final),
              "best_silent": best_silent, "best_lsr": best_lsr}
    rows = []
    for arm, placement in list(arms.items()) + [("w10T", w10T_placement)]:
        if arm == "w10T":
            values = w10T_final
        else:
            values = final[key_of[json.dumps([tuple(p) for p in placement])]]
        rows.append(dict(shared, arm=arm, placement=json.dumps(placement), **values,
                         silent_regret_abs=values["silent"] - best_silent,
                         silent_regret_pct=(values["silent"] / best_silent - 1) * 100 if best_silent > 0 else math.nan,
                         lsr_regret_pct=(1 - values["lsr"] / best_lsr) * 100,
                         silent_rank_screen=int((frame["silent"] < frame.loc[key_of.get(json.dumps(placement), 0), "silent"]).sum()) + 1
                         if arm != "w10T" else math.nan))
    return rows


def run(settings: Settings) -> Path:
    from experiments.revision.common import write_sidecar
    from experiments.shared.parallel import parallel_map

    base = _Base(configs=settings.configs, random_circuits=settings.random_circuits,
                 circuit_seed=settings.circuit_seed, gates_per_qubit=settings.gates_per_qubit,
                 locality=settings.locality, qasmbench=False, smoke=settings.smoke)
    jobs = [(asdict(settings),) + tuple(j[1:]) for j in jobs_for(base)]
    if settings.qasmbench:
        # Stim cannot run rotations: the QASMBench circuits enter as their Clifford proxies,
        # as Fig 6 and Experiment7b simulate them
        from experiments.revision.common import load_small_qasmbench

        proxies = load_small_qasmbench(proxy=True)
        for q, k in settings.configs:
            for name in sorted(n for n, c in proxies.items() if c.num_qubits == q):
                jobs.append((asdict(settings), q, k, f"qb:{name}", "qasmbench_proxy", name, None,
                             proxies[name]))
    jobs.sort(key=lambda j: -j[1])
    print(f"[{NAME}] {len(jobs)} circuits")
    began = time.time()
    results = parallel_map(_run_point, jobs, jobs=settings.jobs, label=NAME)
    rows = pd.DataFrame([row for r in results if r for row in r])
    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}{'_smoke' if settings.smoke else ''}.csv"
    rows.to_csv(path, index=False)
    failed = [j[3] for j, r in zip(jobs, results) if r is None]
    write_sidecar(path, experiment=NAME, settings=asdict(settings), started=began,
                  status="complete" if not failed else f"failed {failed}")
    print(f"[{NAME}] wrote {path}; failed {failed}")
    table = rows.groupby(["config", "arm"])[["silent_regret_pct", "silent_regret_abs", "lsr_regret_pct"]].agg(["mean", "median"])
    print(table.round(3).to_string())
    return path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--random", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = Settings(random_circuits=args.random, jobs=args.jobs, smoke=args.smoke)
    if args.smoke:
        settings.configs = ((4, 2),)
        settings.screen_shots, settings.final_shots = 5_000, 20_000
    return settings


if __name__ == "__main__":
    run(parse_args())
