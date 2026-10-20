"""
weight_sweep_k4.py -- which silent-error weight above k = 2: the planner at [[6,4,2]].

The standard's weight (10) was tuned at k = 2 with the transversal in-patch CZ / XCX
(`weight_sweep.py`). That gadget only exists at k = 2, and on Fig 8's corpus the weight-10
planner at k = 4 kept about a third fewer shots than the weight-0 planner for about 9% less
silent error per kept shot. This sweeps the weight at n_code = 6 to choose it there.

Per circuit (Fig 6's QASMBench circuits with 9..56 qubits, so at least three patches at
k = 4) and machine (`--regimes`):

  * plan the original circuit with `plan_silent_aware` at n_code 6 for every weight (weight 0
    is AQORN's planner exactly), under that machine's noise profile -- Fig 6's convention;
  * translate the Clifford proxy at each distinct placement, every arm layer-sorted
    (`asap_order`), merging on, the plan's H-commutation decision;
  * convert to noisy Stim once at alpha = 1 and rescale exactly
    (`experiments.concatenation.noise.rescale_text`);
  * calibrate alpha on the naive k = 4 baseline (tight packing, passes off, layer-sorted) to a
    selection rate of 0.06 -- Fig 6's rule, solved on frames;
  * frame-simulate every arm at that alpha: selection rate (LSR) and silent logical error per
    kept shot, the TVD stand-in that works at any width (`exp8_undetected/frames.py`).

Regimes: `fig6` is Fig 6's machine (distance and idle scalars 1); `fig8_r1` is Fig 8's
ratio-1 column (`SimSettings().with_noise_ratio(1.0)`: distance 8, idle 0.125).

    PYTHONPATH=. venv/bin/python -m experiments.tvd_explore.weight_sweep_k4 --jobs 12
    PYTHONPATH=. venv/bin/python -m experiments.tvd_explore.weight_sweep_k4 --from-csv <csv>

Writes results/tvd_explore/weight_sweep_k4[_vN].csv (never overwrites) and prints, per
regime and weight, the paired change against weight 0 with bootstrap 95% intervals.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("results/tvd_explore")
NAME = "weight_sweep_k4"
N_CODE = 6
WEIGHTS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
REGIMES = {"fig6": None, "fig8_r1": 1.0}     # None: Fig 6's scalars; else a Fig 8 noise ratio
TARGET_LSR = 0.06
SHOTS = 100_000
CAL_SHOTS = 20_000


def _scalars(regime: str):
    from experiments.shared.simulation import SimSettings

    ratio = REGIMES[regime]
    if ratio is None:
        return 1.0, 1.0
    sim = SimSettings().with_noise_ratio(ratio)
    return sim.distance_scalar, sim.idle_time_scalar


def _profile(regime: str):
    from experiments.shared.simulation import SimSettings, noise_profile_for
    from packing.standard import standard_noise_profile

    ratio = REGIMES[regime]
    if ratio is None:
        return standard_noise_profile()
    return noise_profile_for(SimSettings().with_noise_ratio(ratio))


def _encode(program, placement, *, commute: bool, merging: bool):
    from experiments.shared.simulation import ensure_measurements
    from translator.config import TranslatorConfig
    from translator.core import Translator

    config = TranslatorConfig(n_code=N_CODE, enable_h_commutation=commute,
                              enable_merging=merging, asap_order=True)
    return Translator(config=config).translate(
        ensure_measurements(program), [tuple(p) for p in placement]).circuit


def _frames(encoded, distance: float, idle: float):
    """(alpha -> {lsr, silent_per_kept}) for one encoded circuit, converted once."""
    from experiments.concatenation.noise import rescale_text
    from experiments.revision.exp8_undetected.frames import split_from_stim, stim_params
    from experiments.revision.exp8_undetected.gadgets import noisy_stim

    text = str(noisy_stim(encoded, stim_params(1.0), idle_time_scalar=idle,
                          distance_scalar=distance))

    def at(alpha: float, shots: int, seed: int) -> Dict[str, float]:
        split = split_from_stim(encoded, rescale_text(text, alpha), shots=shots, seed=seed)
        return {"lsr": 1.0 - split["p_detected"], "p_silent": split["p_silent"],
                "silent_per_kept": split["p_silent_given_kept"]}

    return at


def _run(job) -> List[Dict]:
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated import (  # noqa: E501
        ensure_measurements, tight_trivial_packing,
    )
    from packing.silent import load_silent_rates, plan_silent_aware, silent_errors

    circuit, regime, weights, seed = job
    distance, idle = _scalars(regime)
    profile = _profile(regime)
    resource = ensure_measurements(circuit.original)
    n = resource.num_qubits
    rates = load_silent_rates(N_CODE)

    began = time.time()
    plans = {}
    for weight in weights:
        plan = plan_silent_aware(resource, N_CODE, rates=rates, silent_weight=weight, noise=profile)
        plans[weight] = ([list(p) for p in plan.placement], bool(plan.h_commutation),
                         bool(getattr(plan, "fell_back", False)))
    plan_seconds = time.time() - began

    # Baseline: tight trivial packing at k = 4, passes off, layer-sorted. Calibrate on it.
    tight = tight_trivial_packing(n, math.ceil(n / (N_CODE - 2)))
    base = _frames(_encode(circuit.proxy, tight, commute=False, merging=False), distance, idle)
    alpha = 1.0
    for _ in range(6):
        lsr = min(max(base(alpha, CAL_SHOTS, seed)["lsr"], 1e-4), 1 - 1e-6)
        if abs(lsr - TARGET_LSR) < 0.004:
            break
        alpha *= math.log(TARGET_LSR) / math.log(lsr)
    baseline = base(alpha, SHOTS, seed + 1)

    rows: List[Dict] = []
    measured: Dict[str, Dict] = {}
    for weight, (placement, commute, fell_back) in plans.items():
        key = json.dumps([placement, commute])
        if key not in measured:
            encoded = _encode(circuit.proxy, placement, commute=commute, merging=True)
            one, two = one_two_qubit_gate_totals(encoded)
            split = _frames(encoded, distance, idle)(alpha, SHOTS, seed + 2)
            silent_model = silent_errors(resource, [tuple(p) for p in placement], N_CODE,
                                         commute=commute, noise=profile, rates=rates)
            measured[key] = {**split, "one_qubit_gates": one, "two_qubit_gates": two,
                             "depth": encoded.depth(), "S_model": silent_model}
        rows.append({"circuit": circuit.name, "num_qubits": n, "size_group": circuit.size_group,
                     "regime": regime, "weight": weight, "alpha": alpha, "commute": commute,
                     "fell_back": fell_back, "placement": json.dumps(placement),
                     "baseline_lsr": baseline["lsr"],
                     "baseline_silent_per_kept": baseline["silent_per_kept"],
                     **measured[key], "plan_seconds": plan_seconds})
    return rows


def compute(args) -> pd.DataFrame:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    circuits = [c for c in load_fig6_circuits() if 9 <= c.original.num_qubits <= args.max_qubits]
    jobs = [(c, regime, tuple(args.weights), 7_000 + 13 * i)
            for i, c in enumerate(circuits) for regime in args.regimes]
    jobs.sort(key=lambda j: -j[0].original.num_qubits)
    print(f"[{NAME}] {len(circuits)} circuits x {len(args.regimes)} regimes, weights {args.weights}")
    results = parallel_map(_run, jobs, jobs=args.jobs, label=NAME)
    failed = [(j[0].name, j[1]) for j, r in zip(jobs, results) if r is None]
    if failed:
        print(f"[{NAME}] FAILED: {failed}")
    return pd.DataFrame([row for r in results if r for row in r])


def _boot(values: np.ndarray, rng, draws: int = 4000):
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame, seed: int = 20261007) -> pd.DataFrame:
    """Per regime and weight: log-ratio against weight 0, per circuit, then the geomean."""
    rng = np.random.default_rng(seed)
    out = []
    for regime, block in frame.groupby("regime"):
        ref = block[block["weight"] == 0.0].set_index("circuit")
        for weight, sub in block.groupby("weight"):
            sub = sub.set_index("circuit").loc[ref.index]
            row = {"regime": regime, "weight": weight, "circuits": len(sub),
                   "changed": int((sub["placement"] != ref["placement"]).sum()
                                  + ((sub["placement"] == ref["placement"])
                                     & (sub["commute"] != ref["commute"])).sum())}
            for key in ("lsr", "silent_per_kept"):
                logs = np.log(sub[key].clip(lower=1e-6).to_numpy()
                              / ref[key].clip(lower=1e-6).to_numpy())
                low, high = _boot(logs, rng)
                row[f"{key}_x"] = float(np.exp(logs.mean()))
                row[f"{key}_ci"] = f"[{np.exp(low):.3f}, {np.exp(high):.3f}]"
            for key in ("lsr", "silent_per_kept"):           # against the naive k = 4 baseline
                base = "baseline_lsr" if key == "lsr" else "baseline_silent_per_kept"
                logs = np.log(sub[key].clip(lower=1e-6) / sub[base].clip(lower=1e-6))
                row[f"{key}_vs_naive_x"] = float(np.exp(logs.mean()))
            out.append(row)
    return pd.DataFrame(out)


def _free(path: Path) -> Path:
    if not path.exists():
        return path
    version = 2
    while path.with_name(f"{path.stem}_v{version}{path.suffix}").exists():
        version += 1
    return path.with_name(f"{path.stem}_v{version}{path.suffix}")


def main() -> None:
    from experiments.revision.common import write_sidecar

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--weights", nargs="+", type=float, default=list(WEIGHTS))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--max-qubits", type=int, default=56)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--from-csv", default=None)
    args = parser.parse_args()
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        began = time.time()
        frame = compute(args)
        OUT.mkdir(parents=True, exist_ok=True)
        path = _free(OUT / f"{NAME}.csv")
        frame.to_csv(path, index=False)
        write_sidecar(path, experiment=NAME, settings=vars(args), started=began, status="complete")
        print(f"[{NAME}] wrote {path}")
    pd.set_option("display.width", 220)
    print(summarize(frame).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
