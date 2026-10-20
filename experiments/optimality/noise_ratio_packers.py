"""
noise_ratio_packers.py -- does interaction partition + AQORN's descent keep up with AQORN when the
balance between idle and distance (movement) noise changes?

`packers_corpus.py` compared the packers under Fig 6's machine only. This sweeps the phase
diagram's idle/distance ratios (`SimSettings.with_noise_ratio`, 1/64 .. 64; Fig 6's machine,
idle and distance scalars both 1, is ratio 64) on Fig 6's 71 circuits. At each (circuit, ratio):

    planning   the planner's noise profile for that machine (`noise_profile_for`): AQORN is
               `plan_standard(noise=profile)`, which also decides H commutation; every other arm
               runs at that decision, on AQORN's objective built with the same profile
    arms       naive (program order), interaction_partition (cost-model-blind seed only),
               interaction_partition_descent (that seed + AQORN's descent), aqorn_raw (AQORN's
               search with its final naive-fallback check off), aqorn
    measuring  `placement_measure.measure`'s recipe at the ratio's idle / distance scalars:
               static counts of the original's translation, Pauli frames on the Clifford proxy
               (LSR, occupied-lane logical error, exact TVD; `_sel` = independent shots), and the
               noiseless check, at three noise levels: alpha = 1, the circuit's Fig 6 alpha, and
               `cal` -- the alpha at which naive keeps the share of shots it keeps on Fig 6's
               machine (packers_corpus), solved per (circuit, ratio), because at the
               movement-dominated end the Fig 6 alpha saturates (LSR ~0.001, LER ~1). `cal` is
               the comparison to read. Arms that land on the same placement share one
               measurement (`shared_with`).

The compiler is untouched (frozen 2026-10-10). Output `results/noise_ratio_packers/
noise_ratio_packers.csv` (one row per circuit, ratio and arm) + sidecar; each (circuit, ratio) is
written as it finishes (`..._parts/`), so a rerun skips finished work.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.noise_ratio_packers --jobs 180
    ... --names adder_n10 bell_n4 --ratios 0.015625 64      # a subset
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
import zlib
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

NAME = "noise_ratio_packers"
PLANS = "experiments/figures/ablation_standard/ablation_standard_logical_v2.csv"
CORPUS = "results/optimality_front/packers_corpus.csv"
RATIOS = (1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0)
ARMS = ("naive", "interaction_partition", "interaction_partition_descent", "aqorn_raw", "aqorn")
SEARCH_ARMS = ("naive", "interaction_partition", "interaction_partition_descent")
#: `--variants`: AQORN under other `AqornStandard` switches, each planned with `plan_standard`
#: and measured at its own H-commutation decision. Every field the night of 2026-10-10/11 tested
#: is spelled out, so a variant keeps its meaning when the defaults move. In this mode the
#: `aqorn` arm (and the commutation every search arm runs at) is `BASE_VARIANT`, the b664c94
#: standard, so those rows repeat `noise_ratio_packers_std_verify.csv`'s seeds exactly.
SWITCHES = dict(standard_commutation=False, restarts=None, grid_alignment=False)
VARIANTS = {
    "aqorn_b664": dict(SWITCHES),
    "aqorn_comm": dict(SWITCHES, standard_commutation=True),
    "aqorn_comm_norestart": dict(SWITCHES, standard_commutation=True, restarts=0),
    "aqorn_comm_grid": dict(SWITCHES, standard_commutation=True, grid_alignment=True),
}
BASE_VARIANT = "aqorn_b664"


def context_for(program, commute: bool, noise):
    """`standard_packers.make_standard_context` with the machine's planner profile passed in."""
    from experiments.optimality.standard_packers import N_CODE, standard_gadgets
    from experiments.revision.packers import PackContext
    from iceberg.utils import logical_capacity
    from packing.silent import build_costs, load_silent_rates
    from packing.standard import STANDARD
    from packing.tables import translated_view

    rates = load_silent_rates(N_CODE,
                              transversal_intra_two_qubit=STANDARD.transversal_intra_two_qubit)
    with standard_gadgets():
        profiles, cost = build_costs(program, N_CODE, commute=commute, noise=noise, rates=rates,
                                     silent_weight=STANDARD.silent_weight_for(N_CODE),
                                     solo_hh=STANDARD.solo_hh_aware)
    k = logical_capacity(N_CODE)
    tightest = max(1, -(-program.num_qubits // k))
    return PackContext(program=program, n_code=N_CODE, commute=commute, noise=noise,
                       requested_noise=noise, k=k,
                       view=translated_view(program, enable_h_commutation=commute),
                       profiles=profiles, cost=cost, tightest=tightest, slots=tightest)


def noisy_at(encoded, idle: float, distance: float):
    """`fastframes.at_alpha_one` at this machine's idle / distance scalars."""
    from experiments.revision.exp8_undetected.frames import stim_params
    from experiments.revision.exp8_undetected.gadgets import stim_overrides
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    with sb._active_params(stim_overrides(stim_params(1.0), idle_time_scalar=idle,
                                          distance_scalar=distance)):
        layout = positions_for_circuit(encoded, geometry=sb._geometry())
        return sb.qiskit_to_stim(encoded, layout=layout, include_noise=True)


def prepare(original, proxy, placement, commute: bool, *, idle: float, distance: float) -> Dict:
    """One placement translated (static counts of the original) and its proxy converted once,
    at alpha = 1 and this machine's scalars -- the slow step every alpha then reuses."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from experiments.shared.simulation import ensure_measurements
    from packing.standard import standard_translator_config
    from translator.core import Translator

    placement = [tuple(int(v) for v in p) for p in placement]
    translator = Translator(config=standard_translator_config(4, commute=commute))
    static = translator.translate(ensure_measurements(original), placement).circuit
    one, two = one_two_qubit_gate_totals(static)
    result = translator.translate(ensure_measurements(proxy), placement)
    noisy = noisy_at(result.circuit, idle, distance)
    return {"static": {"one_qubit_gates": int(one), "two_qubit_gates": int(two),
                       "depth": int(static.depth())},
            "encoded": result.circuit, "noisy": noisy, "text": str(noisy),
            "positions": [tuple(p) for p in result.context.state_positions],
            "placement": placement, "commute": commute}


def _scaled(prepared: Dict, alpha: float):
    from experiments.concatenation.noise import rescale_text

    return prepared["noisy"] if alpha == 1.0 else rescale_text(prepared["text"], alpha)


def calibrate(prepared: Dict, support, target_lsr: float, *, shots: int = 20_000,
              steps: int = 22) -> float:
    """The alpha at which this placement keeps `target_lsr` of its shots (bisection on log
    alpha; LSR falls monotonically as alpha grows)."""
    import math

    from experiments.optimality.exhaustive_front import frame_metrics

    def lsr(alpha: float) -> float:
        return frame_metrics(prepared["encoded"], _scaled(prepared, alpha), prepared["positions"],
                             support[1], shots=shots, seed=12345)["lsr"]

    low, high = math.log(1e-4), math.log(1e2)
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if lsr(math.exp(middle)) > target_lsr:
            low = middle
        else:
            high = middle
    return math.exp(0.5 * (low + high))


def measure_prepared(prepared: Dict, support, *, alphas: Dict[str, float], shots: int,
                     seed_key: str) -> Dict:
    """`placement_measure.measure`'s frame metrics and noiseless check on a prepared placement."""
    from experiments.optimality.exhaustive_front import frame_metrics
    from experiments.optimality.placement_measure import noiseless_ok

    row: Dict[str, object] = dict(prepared["static"])
    seed = zlib.crc32(f"{seed_key}:{prepared['placement']}:{prepared['commute']}".encode()) \
        % 1_000_000_007
    for label, alpha in alphas.items():
        scaled = _scaled(prepared, alpha)
        for suffix, offset in (("", 0), ("_sel", 1)):
            report = frame_metrics(prepared["encoded"], scaled, prepared["positions"], support[1],
                                   shots=shots, seed=seed + offset)
            for metric in ("lsr", "ler", "tvd"):
                row[f"{metric}{suffix}_{label}"] = report[metric]
    row["correct"] = bool(noiseless_ok(prepared["encoded"], prepared["noisy"],
                                       prepared["positions"], support))
    return row


def _run(job) -> List[Dict]:
    from experiments.optimality.placement_measure import logical_support
    from experiments.optimality.standard_packers import standard_gadgets
    from experiments.revision import packers
    from experiments.shared.simulation import SimSettings, noise_profile_for
    from packing.standard import plan_standard

    name, group, original, proxy, alpha_fig6, target_lsr, ratio, shots, part, variants = job
    part = Path(part)
    if part.exists():
        return pd.read_csv(part).to_dict("records")
    began = time.time()
    sim = SimSettings().with_noise_ratio(ratio)
    idle, distance = float(sim.idle_time_scalar), float(sim.distance_scalar)
    profile = noise_profile_for(sim)

    from packing.standard import STANDARD

    arms = ARMS + tuple(variants)
    base = STANDARD.with_(**VARIANTS[BASE_VARIANT]) if variants else STANDARD
    placements, commutes, seconds, notes = {}, {}, {}, {}
    with standard_gadgets():
        t = time.time()
        plan = plan_standard(original, noise=profile, standard=base)
        commute = bool(plan.h_commutation)
        placements["aqorn"] = [tuple(int(v) for v in p) for p in plan.placement]
        notes["aqorn"] = dict(fell_back=bool(plan.fell_back), aligned=bool(getattr(plan, "aligned", False)))
        seconds["aqorn"] = time.time() - t
        # AQORN's search without its final check (verify: fall back to naive when naive prices
        # lower on the translated schedule), to tell the search from the check
        t = time.time()
        raw = plan_standard(original, noise=profile, h_commutation=commute, verify=False,
                            standard=base)
        placements["aqorn_raw"] = [tuple(int(v) for v in p) for p in raw.placement]
        seconds["aqorn_raw"] = time.time() - t
        contexts = {commute: context_for(original, commute, profile)}
        for arm in SEARCH_ARMS:
            t = time.time()
            placements[arm] = [tuple(int(v) for v in p)
                               for p in getattr(packers, arm)(contexts[commute])]
            seconds[arm] = time.time() - t
        commutes = {arm: commute for arm in ARMS}
        for arm in variants:
            t = time.time()
            other = plan_standard(original, noise=profile, standard=STANDARD.with_(**VARIANTS[arm]))
            placements[arm] = [tuple(int(v) for v in p) for p in other.placement]
            commutes[arm] = bool(other.h_commutation)
            notes[arm] = dict(fell_back=bool(other.fell_back),
                              aligned=bool(getattr(other, "aligned", False)))
            seconds[arm] = time.time() - t
        for value in set(commutes.values()) - set(contexts):
            contexts[value] = context_for(original, value, profile)
        values = {arm: packers.model_value(contexts[commutes[arm]], p)
                  for arm, p in placements.items()}

    support = logical_support(proxy)
    keys = {arm: json.dumps([list(p) for p in placements[arm]]) for arm in arms}
    shared = {arm: next((a for a in arms[:i]
                         if (keys[a], commutes[a]) == (keys[arm], commutes[arm])), None)
              for i, arm in enumerate(arms)}
    prepared: Dict[str, Dict] = {}
    errors: Dict[str, str] = {}
    for arm in arms:
        if shared[arm] is None:
            try:
                prepared[arm] = prepare(original, proxy, placements[arm], commutes[arm],
                                        idle=idle, distance=distance)
            except Exception as error:  # noqa: BLE001 - a failed arm is recorded, not fatal
                errors[arm] = f"{type(error).__name__}: {error}"[:200]
    # One noise level per (circuit, ratio): naive keeps the share of shots it keeps on Fig 6's
    # machine (its packers_corpus LSR at the Fig 6 alpha), so every ratio is compared at the
    # same post-selection pressure instead of saturating where movement noise dominates.
    alpha_cal = calibrate(prepared["naive"], support, target_lsr) if "naive" in prepared \
        else float("nan")
    alphas = {"a1": 1.0, "fig6": alpha_fig6, "cal": alpha_cal}
    measured: Dict[str, Dict] = {}
    rows = []
    for arm in arms:
        source = shared[arm] or arm
        row = {"circuit": name, "size_group": group, "num_qubits": original.num_qubits,
               "ratio": ratio, "idle_scalar": idle, "distance_scalar": distance, "arm": arm,
               "commute": commutes[arm], "support_bits": len(support[1]),
               "alpha_fig6": alpha_fig6, "target_lsr": target_lsr, "alpha_cal": alpha_cal,
               "arm_seconds": round(seconds[arm], 2), "placement": keys[arm],
               "model_value": values[arm], "shared_with": shared[arm],
               "aqorn_fell_back": notes["aqorn"]["fell_back"],
               "fell_back": notes.get(arm, {}).get("fell_back"),
               "aligned": notes.get(arm, {}).get("aligned")}
        try:
            if source in errors:
                raise RuntimeError(errors[source])
            if source not in measured:
                measured[source] = measure_prepared(prepared[source], support, alphas=alphas,
                                                    shots=shots, seed_key=f"{name}:r{ratio}")
            row.update(measured[source])
        except Exception as error:  # noqa: BLE001 - a failed arm is recorded, not fatal
            row["error"] = f"{type(error).__name__}: {error}"[:200]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["job_seconds"] = round(time.time() - began, 1)
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
    parser.add_argument("--ratios", type=float, nargs="*", default=list(RATIOS))
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--tag", default="", help="suffix for the parts folder and the CSV")
    parser.add_argument("--variants", nargs="*", default=[], choices=sorted(VARIANTS),
                        help="extra AQORN arms under other AqornStandard switches; the aqorn arm "
                             "becomes the b664c94 standard (VARIANTS docstring)")
    parser.add_argument("--out", default="results/noise_ratio_packers")
    parser.add_argument("--corpus", default=CORPUS,
                        help="packers_corpus.csv: naive's LSR on Fig 6's machine, the calibration "
                             "target at every ratio")
    args = parser.parse_args()
    began = time.time()
    out = Path(args.out)
    print(f"[{NAME}] logical optimisation: {state()}", flush=True)
    alphas = pd.read_csv(PLANS).drop_duplicates("circuit_name").set_index("circuit_name")
    corpus = pd.read_csv(args.corpus)
    targets = corpus[corpus["arm"] == "naive"].set_index("circuit")["lsr_fig6"]
    circuits = [c for c in load_fig6_circuits() if not args.names or c.name in args.names]
    stem = f"{NAME}_{args.tag}" if args.tag else NAME
    jobs = [(c.name, c.size_group, c.original, c.proxy,
             float(alphas.loc[c.name, "alpha_final"]), float(targets.loc[c.name]), float(ratio),
             args.shots, str(out / f"{stem}_parts" / f"{c.name}__r{ratio:g}.csv"),
             tuple(args.variants))
            for c in circuits for ratio in args.ratios]
    jobs.sort(key=lambda j: (-j[2].num_qubits, j[6]))
    print(f"[{NAME}] {len(circuits)} circuits x {len(args.ratios)} ratios = {len(jobs)} jobs, "
          f"{args.jobs} workers", flush=True)
    results = parallel_map(_run, jobs, jobs=args.jobs, label=NAME)
    failed = [(j[0], j[5]) for j, r in zip(jobs, results) if r is None]
    frame = pd.DataFrame([row for result in results if result for row in result])
    path = out / (f"{stem}.csv" if not args.names else f"{stem}_subset.csv")
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment=NAME, settings=vars(args), started=began, status="complete",
                  extra={"failed": failed})
    wrong = frame[frame.get("correct") == False] if "correct" in frame else frame.iloc[:0]  # noqa: E712
    errors = frame[frame["error"].notna()] if "error" in frame else frame.iloc[:0]
    print(f"[{NAME}] {frame['circuit'].nunique()} circuits, {len(frame)} rows -> {path}; failed "
          f"jobs {failed}; arm errors {len(errors)}; miscompiled placements {len(wrong)} "
          f"{sorted(set(zip(wrong.circuit, wrong.ratio, wrong.arm)))[:10]}", flush=True)


if __name__ == "__main__":
    main()
