"""
exhaustive_simulate.py -- Experiment7b: every placement, simulated -- the real objective.

7a scores placements by translation, which is the packer's objective. What the paper
claims is fidelity, so this scores the same small spaces by simulated LSR and TVD (the
same objective, taken literally).

* Configurations (4, k=2), (5, 2), (6, 2); circuits: the 4-6 qubit QASMBench proxies that
  build, and 20 random structured Clifford programs per configuration (the first 20 of
  7a's, so the two experiments share circuits).
* The space: the *complete* tightest space -- every partition x patch order x lane map
  (24 / 720 / 720 placements), because the translator is not invariant under the
  relabellings the generator's canonical space removes (see `space.py`). `--space full`
  adds canonical representatives of the sparser patch counts.
* Fig 6's machine: Stim, distance + idle noise at scalar 1. Per-circuit alpha from Fig 6's
  two-pilot rule on the naive baseline (both passes off), target LSR 0.06; optionally also
  fixed alpha = 1 (`--alphas calibrated 1`).
* Arms: AQORN planned as Fig 6 plans (with the noise profile; these are proxies, so its
  `verify` really does price expected error), `aqorn_raw`, naive, and every Experiment1
  baseline -- all under the shipped compiler's commutation decision with merging on.
* Two stages against the winner's curse: every placement at 20k raw shots; then the top
  10 by LSR, plus every arm, re-simulated at 200k raw shots with fresh seeds. "Best" is
  decided at 200k. TVD is measured against the *exact* ideal distribution (`Statevector`;
  these programs have at most 6 qubits), and the noiseless finite-sample floor at each
  point's accepted count is recorded beside it.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import (
    RESULTS,
    NoiseSetup,
    ResultTable,
    add_common_arguments,
    exact_ideal_distribution,
    fig6_noise_profile,
    load_small_qasmbench,
    log,
    out_dir,
    placement_key,
    sim_config,
    simulate_arm,
    stream_map,
    tvd_statistics,
)

NAME = "exp7b"
SEED_BASE = 20261007


@dataclass
class Settings:
    configs: Sequence[Tuple[int, int]] = ((4, 2), (5, 2), (6, 2))
    random_circuits: int = 20
    circuit_seed: int = 20260730
    space: str = "tightest"
    alphas: Sequence[str] = ("calibrated",)
    sweep_shots: int = 20_000
    final_shots: int = 200_000
    top: int = 10
    pilot_shots: int = 4_000
    target_selection_rate: float = 0.06
    table: str = "exhaustive_simulate"
    smoke: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp7")


def _tvd(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 1.0
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def jobs_for(settings: Settings) -> List[Tuple]:
    from experiments.revision.exp7_optimality.exhaustive_translate import Settings as S7a
    from experiments.revision.exp7_optimality.exhaustive_translate import _random_circuit

    proxies = load_small_qasmbench(proxy=True)
    jobs: List[Tuple] = []
    for q, k in settings.configs:
        names = sorted(n for n, c in proxies.items() if c.num_qubits == q)
        count = settings.random_circuits
        if settings.smoke:
            names, count = names[:1], 1
        for name in names:
            jobs.append((q, k, f"qb:{name}", "qasmbench", name, proxies[name]))
        for index in range(count):
            seed = settings.circuit_seed + index
            jobs.append((q, k, f"rand:{seed}", "random", f"random_{index}",
                         _random_circuit(q, seed, S7a())))
    return jobs


def _calibrate(program, naive, settings: Settings, seed: int) -> float:
    """Fig 6's two-pilot rule on the naive baseline, both passes off."""
    from experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated import (  # noqa: E501
        BenchmarkSettings,
        calibrated_alpha_from_gate_count,
        corrected_alpha,
    )
    from experiments.revision.common import translate_metrics

    base = translate_metrics(program, naive, 4, commute=False, merging=False)
    fig6 = BenchmarkSettings(target_selection_rate=settings.target_selection_rate)
    alpha = calibrated_alpha_from_gate_count(
        int(base["one_qubit_gates"] + base["two_qubit_gates"]), fig6)
    for step in (1, 2):
        lsr = simulate_arm(program, naive, sim_config(
            NoiseSetup(alpha=alpha), n_code=4, commute=False, merging=False,
            repetitions=settings.pilot_shots, seed=seed + step))["lsr"]
        if step == 1:
            alpha = corrected_alpha(alpha, lsr, settings.target_selection_rate)
    return float(alpha)


def _run_point(job) -> Dict[str, List[Dict]]:
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.revision.packers import run_all_packers
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing

    settings_dict, q, k, circuit_id, source, name, circuit, point_index = job
    settings = Settings(**settings_dict)
    n_code = k + 2
    program = ensure_measurements(circuit)
    exact = exact_ideal_distribution(circuit)
    arms, notes, _ = run_all_packers(program, n_code, noise=fig6_noise_profile())
    commute = bool(notes["commute"])
    arms = {a: [tuple(p) for p in pl] for a, pl in arms.items() if pl is not None}
    naive = [tuple(p) for p in naive_packing(q, k)]
    space = [pl for pl, _complete in exhaustive_space(q, k, settings.space)]
    index_of = {placement_key(pl): i for i, pl in enumerate(space)}
    sweep_shots = 2000 if settings.smoke else settings.sweep_shots
    final_shots = 4000 if settings.smoke else settings.final_shots
    seed0 = SEED_BASE + 10_000_000 * point_index
    rng = np.random.default_rng(seed0)

    summary: List[Dict] = []
    sweep_rows: List[Dict] = []
    for a_index, alpha_label in enumerate(settings.alphas):
        alpha = (_calibrate(program, naive, settings, seed0 + 9_000_000 + 10 * a_index)
                 if alpha_label == "calibrated" else float(alpha_label))
        noise = NoiseSetup(alpha=alpha)

        def run(placement, shots, seed):
            result = simulate_arm(program, placement, sim_config(
                noise, n_code=n_code, commute=commute, merging=True, repetitions=shots,
                seed=seed))
            accepted = int(round(result["lsr"] * shots))
            floor = tvd_statistics(result["noisy"], exact, accepted=accepted, ideal_shots=None,
                                   rng=rng, bootstrap=2, floor_draws=20)["tvd_floor"]
            return {"lsr": result["lsr"], "tvd": result["tvd"],
                    "tvd_exact": _tvd(result["noisy"], exact), "accepted": accepted,
                    "floor": floor, "depth": result["depth"],
                    "two_qubit_gates": result["two_qubit_gates"]}

        began = time.time()
        sweep = []
        for i, placement in enumerate(space):
            r = run(placement, sweep_shots, seed0 + 1_000 * a_index + i)
            r.update(index=i, patches=len({p for p, _ in placement}))
            sweep.append(r)
        sweep_seconds = time.time() - began
        frame = pd.DataFrame(sweep)
        top = list(frame.sort_values("lsr", ascending=False)["index"].head(settings.top))
        arm_indices = {arm: index_of.get(placement_key(pl)) for arm, pl in arms.items()}
        finalists = sorted(set(top) | {i for i in arm_indices.values() if i is not None})
        final: Dict[int, Dict] = {}
        for i in finalists:
            final[i] = run(space[i], final_shots, seed0 + 500_000 + 1_000 * a_index + i)
        best_lsr_index = max(final, key=lambda i: final[i]["lsr"])
        best_tvd_index = min(final, key=lambda i: final[i]["tvd_exact"])
        best_lsr = final[best_lsr_index]["lsr"]
        best_tvd = final[best_tvd_index]["tvd_exact"]

        for arm, i in arm_indices.items():
            row = {
                "num_qubits": q, "k": k, "config": f"{q}q,k={k}", "circuit_id": circuit_id,
                "source": source, "name": name, "alpha_label": alpha_label, "alpha": alpha,
                "arm": arm, "commute": commute, "space": settings.space,
                "space_size": len(space), "in_space": i is not None,
                "sweep_shots": sweep_shots, "final_shots": final_shots,
                "sweep_seconds": sweep_seconds,
                "best_index": best_lsr_index, "best_lsr_200k": best_lsr,
                "best_tvd_exact_200k": best_tvd,
                "best_is_in_sweep_top": best_lsr_index in top,
                "sweep_best_lsr": float(frame["lsr"].max()),
                "sweep_median_lsr": float(frame["lsr"].median()),
                "sweep_worst_lsr": float(frame["lsr"].min()),
                "placement": json.dumps([list(p) for p in arms[arm]]),
            }
            if i is not None:
                s, f = sweep[i], final[i]
                row.update({
                    "index": i,
                    "lsr_20k": s["lsr"], "tvd_exact_20k": s["tvd_exact"],
                    "rank_lsr_20k": int((frame["lsr"] > s["lsr"] + 1e-12).sum()) + 1,
                    "pct_better_lsr_20k": 100.0 * float((frame["lsr"] > s["lsr"] + 1e-12).mean()),
                    "rank_tvd_20k": int((frame["tvd_exact"] < s["tvd_exact"] - 1e-12).sum()) + 1,
                    "lsr_200k": f["lsr"], "tvd_200k": f["tvd"], "tvd_exact_200k": f["tvd_exact"],
                    "accepted_200k": f["accepted"], "tvd_floor_200k": f["floor"],
                    "lsr_regret_pct": (1 - f["lsr"] / best_lsr) * 100 if best_lsr > 0 else math.nan,
                    "tvd_regret_abs": f["tvd_exact"] - best_tvd,
                    "is_best_lsr": i == best_lsr_index, "is_best_tvd": i == best_tvd_index,
                    "lsr_se_200k": math.sqrt(max(f["lsr"] * (1 - f["lsr"]), 1e-12) / final_shots),
                })
            summary.append(row)
        for s in sweep:
            sweep_rows.append({"num_qubits": q, "k": k, "circuit_id": circuit_id,
                               "alpha_label": alpha_label, **s})
    return {"rows": summary, "sweep": sweep_rows}


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / f"{settings.table}.csv",
                        key_columns=["num_qubits", "k", "circuit_id", "alpha_label", "arm"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    sweep = ResultTable(out / f"{settings.table}_sweep.csv",
                        key_columns=["num_qubits", "k", "circuit_id", "alpha_label", "index"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    work = jobs_for(settings)
    last = settings.alphas[-1]
    jobs = [(asdict(settings), *w, i) for i, w in enumerate(work)
            if not table.is_done((w[0], w[1], w[2], last, "aqorn"))]
    jobs.sort(key=lambda j: -j[1])
    log(NAME, f"{len(work)} (config, circuit) points, {len(jobs)} to run; space="
              f"{settings.space}, alphas {list(settings.alphas)}")

    def on_result(_i: int, result: Dict[str, List[Dict]]) -> None:
        sweep.append(result["sweep"])
        table.append(result["rows"])

    counts = stream_map(_run_point, jobs, jobs=settings.jobs, label=NAME, on_result=on_result)
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    sweep.write_sidecar(status)
    frame = table.frame()
    if not frame.empty:
        print(frame.groupby(["config", "alpha_label", "arm"])[
            ["lsr_regret_pct", "tvd_regret_abs", "pct_better_lsr_20k"]].mean()
            .round(3).to_string())
    return table.path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp7")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--random", type=int, default=20)
    parser.add_argument("--space", choices=["tightest", "full"], default="tightest")
    parser.add_argument("--alphas", nargs="+", default=["calibrated"])
    parser.add_argument("--name", default="exhaustive_simulate")
    args = parser.parse_args()
    configs = ((4, 2), (5, 2), (6, 2))
    if args.configs:
        configs = tuple(tuple(int(x) for x in c.split(":")) for c in args.configs)
    if args.smoke and not args.configs:
        configs = ((4, 2), (5, 2))
    return Settings(configs=configs, random_circuits=args.random, space=args.space,
                    alphas=tuple(args.alphas), table=args.name, smoke=args.smoke, jobs=args.jobs,
                    out=str(out_dir(args, "exp7"))), args.resume


if __name__ == "__main__":
    run(*parse_args())
