"""
exhaustive_silent.py -- how far the weight-10 planner is from the best placement of a small program.

Experiment7a (`experiments/revision/exp7_optimality`, REVISION_RESULTS.md) measured the submitted
planner (v4) against every placement of 4-6 qubit programs. The paper now plans with the
silent-aware weight-10 objective (`exp8_undetected.cost_model.plan_silent_aware`), optionally with
transversal in-patch CZ/XCX (w10T), and compares stages on equal schedules
(`TranslatorConfig.asap_order`). This re-measures optimality for those planners. Changes from 7a:

* arms: naive (Fig 6's baseline placement), v4 (`plan_packing`, Fig 6's call), w10, w10T, and the
  interaction-count partition (the conventional packer 7b found closest to the simulated optimum);
* the space: the complete tightest space (every partition x patch order x lane map, `space.py`)
  under BOTH H-commutation settings, because the planner chooses commutation as well;
* objectives, per placement:
    score      the weight-10 planner's own objective, exactly what `plan_silent_aware` minimises:
               measured cost + 10 x model silent error, in the currency the measurement answered in
               (expected detectable errors for Clifford programs, weighted gates + 12 x depth for
               programs carrying rotations -- Stim cannot schedule rz/rx)
    measured   the same without the silent term: v4's own objective (`plan.measured_cost`)
    fair_*     outcomes on equal schedules (asap_order on): weighted gates + 12 x depth, depth,
               two-qubit gates
    silent     the model's expected silent logical error (probability at alpha = 1)
  The w10T arm is scored over the space translated with its switch on -- a different translator,
  so a different space and optimum (`translator = "transversal"` rows).

Every arm is checked against the space: an arm beating the exhaustive optimum is a bug, and the
w10 arm's recomputed score must equal the value its own planner recorded (`score_matches_planner`).

    PYTHONPATH=. venv/bin/python -m experiments.optimality.exhaustive_silent --jobs 3
    ... --smoke
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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import (
    RESULTS,
    ResultTable,
    load_small_qasmbench,
    log,
    stream_map,
)

NAME = "exhaustive_silent"
RATES = "results/revision/exp8/gadget_rates.csv"
RATES_T = "results/tvd_explore/gadget_rates_transversal.csv"
SWITCH = "TRANSVERSAL_INTRA_TWO_QUBIT"
OBJECTIVES = ("score", "measured", "fair_work_depth", "fair_depth", "fair_two_qubit_gates",
              "silent")
#: arm -> translator its placements are scored under
ARM_TRANSLATOR = {"naive": "shipped", "v4": "shipped", "w10": "shipped",
                  "interaction_partition": "shipped", "w10T": "transversal"}


@dataclass
class Settings:
    configs: Sequence[Tuple[int, int]] = ((4, 2), (5, 2), (6, 2))
    random_circuits: int = 20
    circuit_seed: int = 20260730  # 7a's (the archived script's) base seed
    gates_per_qubit: int = 10
    locality: float = 0.85
    qasmbench: bool = True
    silent_weight: float = 10.0
    smoke: bool = False
    jobs: int = 3
    out: str = str(RESULTS.parent / "optimality")


@contextmanager
def switched(on: bool):
    """The w10T translator for the duration; the commutation cache is keyed without it."""
    from packing import passes
    from translator import explore_switches

    explore_switches.configure(**{SWITCH: on})
    passes.clear_cache()
    try:
        yield
    finally:
        explore_switches.configure(**{SWITCH: False})
        passes.clear_cache()


def _random_circuit(q: int, seed: int, settings: Settings):
    from experiments.shared.random_circuits import plant_blocks, structured_program_circuit

    blocks = plant_blocks(q, min(2, q), seed=seed)
    return structured_program_circuit(
        q, settings.gates_per_qubit, seed=seed, blocks=blocks,
        locality=settings.locality, immutable_ratio=0.0, clifford_only=True,
    )


def jobs_for(settings: Settings) -> List[Tuple]:
    small = load_small_qasmbench(proxy=False) if settings.qasmbench else {}
    jobs: List[Tuple] = []
    for q, k in settings.configs:
        names = sorted(n for n, c in small.items() if c.num_qubits == q)
        if settings.smoke:
            names = names[:1]
        for name in names:
            jobs.append((asdict(settings), q, k, f"qb:{name}", "qasmbench", name, None, small[name]))
        for index in range(1 if settings.smoke else settings.random_circuits):
            seed = settings.circuit_seed + index
            jobs.append((asdict(settings), q, k, f"rand:{seed}", "random", f"random_{index}", seed,
                         _random_circuit(q, seed, settings)))
    return jobs


def _placement(plan) -> List[Tuple[int, int]]:
    return [(int(p), int(s)) for p, s in plan.placement]


class Scorer:
    """`plan_silent_aware`'s score for any placement, with the per-commutation costs built once."""

    def __init__(self, program, n_code: int, rates, noise, weight: float):
        from experiments.revision.exp8_undetected.cost_model import build_costs
        from iceberg.costs import DEFAULT_WORK_WEIGHTS

        self.program, self.n_code, self.rates, self.noise, self.weight = (
            program, n_code, rates, noise, weight)
        self.units = DEFAULT_WORK_WEIGHTS[1] / rates.p2
        self.costs = {}
        for commute in (False, True):
            profiles, priced = build_costs(program, n_code, commute=commute, noise=noise,
                                           rates=rates, silent_weight=1.0)
            _, plain = build_costs(program, n_code, commute=commute, noise=noise, rates=rates,
                                   silent_weight=0.0)
            self.costs[commute] = (profiles, priced, plain)

    def silent(self, placement, commute: bool) -> float:
        from experiments.revision.exp8_undetected.cost_model import groups_of, silent_constant

        profiles, priced, plain = self.costs[commute]
        groups = groups_of(placement)
        return silent_constant(profiles, self.rates) + (
            priced.of(groups) - plain.of(groups)) / priced.silent_scale

    def __call__(self, placement, commute: bool) -> Dict[str, float]:
        from experiments.revision.exp8_undetected.cost_model import measured_with_currency
        from packing.cost import DEFAULT_DEPTH_WEIGHT

        measured, currency = measured_with_currency(
            self.program, placement, self.n_code, depth_weight=DEFAULT_DEPTH_WEIGHT,
            h_commutation=commute, noise=self.noise)
        if measured is None:
            return {"score": math.inf, "measured": math.inf, "silent": math.nan,
                    "currency": currency}
        silent = self.silent(placement, commute)
        scale = 1.0 if currency == "expected_error" else self.units
        return {"score": measured + self.weight * silent * scale, "measured": measured,
                "silent": silent, "currency": currency}


def fair_metrics(program, placement, n_code: int, commute: bool) -> Dict[str, float]:
    """Translated with every program layer-sorted first (`asap_order`), merging on."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from experiments.revision.common import translator_config
    from iceberg.costs import DEFAULT_WORK_WEIGHTS
    from packing.cost import DEFAULT_DEPTH_WEIGHT
    from pipeline.tasks.translate_permutation import translate_permutation

    config = translator_config(n_code, commute=commute, merging=True)
    config.translator_config.asap_order = True
    try:
        circuit = translate_permutation(program, [tuple(p) for p in placement], config=config).circuit
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return {"fair_work_depth": math.inf, "fair_depth": math.inf,
                "fair_two_qubit_gates": math.inf, "fair_one_qubit_gates": math.inf}
    one, two = one_two_qubit_gate_totals(circuit)
    w1, w2 = DEFAULT_WORK_WEIGHTS
    depth = circuit.depth()
    return {"fair_work_depth": float(w1 * one + w2 * two + DEFAULT_DEPTH_WEIGHT * depth),
            "fair_depth": float(depth), "fair_two_qubit_gates": float(two),
            "fair_one_qubit_gates": float(one)}


def _run_point(job) -> Dict[str, List[Dict]]:
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from experiments.revision.packers import interaction_partition, make_context
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing, plan_packing

    settings_dict, q, k, circuit_id, source, name, seed, circuit = job
    settings = Settings(**settings_dict)
    n_code = k + 2
    program = ensure_measurements(circuit)
    noise = fig6_noise_profile()
    rates = {"shipped": UndetectedRates.from_table(pd.read_csv(RATES), n_code, 3e-3),
             "transversal": UndetectedRates.from_table(pd.read_csv(RATES_T), n_code, 3e-3)}

    began = time.time()
    arms: Dict[str, Tuple[List[Tuple[int, int]], bool]] = {}
    planner_values: Dict[str, Dict[str, float]] = {}
    with switched(False):
        v4 = plan_packing(program, n_code, h_commutation="auto", verify=True, noise=noise)
        arms["v4"] = (_placement(v4), bool(v4.h_commutation))
        w10 = plan_silent_aware(program, n_code, rates=rates["shipped"],
                                silent_weight=settings.silent_weight, noise=noise)
        arms["w10"] = (_placement(w10), bool(w10.h_commutation))
        planner_values["w10"] = dict(w10.considered, fell_back=bool(w10.fell_back))
        commute = arms["w10"][1]
        arms["naive"] = (naive_packing(q, k), commute)
        try:
            arms["interaction_partition"] = (
                [tuple(p) for p in interaction_partition(make_context(program, n_code, commute, noise))],
                commute)
        except Exception:  # noqa: BLE001
            pass
    with switched(True):
        w10T = plan_silent_aware(program, n_code, rates=rates["transversal"],
                                 silent_weight=settings.silent_weight, noise=noise)
        arms["w10T"] = (_placement(w10T), bool(w10T.h_commutation))
        planner_values["w10T"] = dict(w10T.considered, fell_back=bool(w10T.fell_back))
    plan_seconds = time.time() - began

    space_rows: List[Dict] = []
    began = time.time()
    for translator in ("shipped", "transversal"):
        with switched(translator == "transversal"):
            scorer = Scorer(program, n_code, rates[translator], noise, settings.silent_weight)
            for commute_ in (False, True):
                for index, (placement, _) in enumerate(exhaustive_space(q, k, "tightest")):
                    placement = [tuple(p) for p in placement]
                    values = scorer(placement, commute_)
                    values.update(fair_metrics(program, placement, n_code, commute_))
                    space_rows.append({"translator": translator, "commute": commute_,
                                       "placement_index": index,
                                       "placement": json.dumps(placement), **values})
    enumerate_seconds = time.time() - began
    space = pd.DataFrame(space_rows)

    shared = {"num_qubits": q, "k": k, "n_code": n_code, "config": f"{q}q,k={k}",
              "circuit_id": circuit_id, "source": source, "name": name, "seed": seed,
              "space_per_setting": int(len(space) // 4), "plan_seconds": plan_seconds,
              "enumerate_seconds": enumerate_seconds}
    rows: List[Dict] = []
    for arm, (placement, commute_) in arms.items():
        translator = ARM_TRANSLATOR[arm]
        frame = space[space["translator"] == translator]
        key = json.dumps([tuple(p) for p in placement])
        own = frame[(frame["placement"] == key) & (frame["commute"] == commute_)]
        row = dict(shared, arm=arm, translator=translator, commute=commute_,
                   placement=key, in_space=bool(len(own)))
        if not len(own):
            rows.append(row)
            continue
        own = own.iloc[0]
        row["currency"] = own["currency"]
        for objective in OBJECTIVES:
            value = float(own[objective])
            best = float(frame[objective].min())
            row[objective] = value
            row[f"{objective}_opt"] = best
            row[f"{objective}_regret_pct"] = (value / best - 1) * 100 if best > 0 else math.nan
            row[f"{objective}_optimal"] = bool(abs(value - best) <= 1e-9 * max(1.0, abs(best)))
            better = int((frame[objective] < value - 1e-9 * max(1.0, abs(value))).sum())
            row[f"{objective}_pct_better"] = 100.0 * better / len(frame)
            row[f"{objective}_beats_space"] = bool(value < best - 1e-9 * max(1.0, abs(best)))
        # the commutation setting alone: the best placement under the arm's own setting
        same = frame[frame["commute"] == commute_]
        row["score_regret_same_commute_pct"] = (float(own["score"]) / float(same["score"].min()) - 1) * 100
        if arm in planner_values:
            fell_back = bool(planner_values[arm].get("fell_back"))
            recorded = planner_values[arm].get("naive" if fell_back else "packed")
            row["planner_fell_back"] = fell_back
            row["planner_recorded"] = recorded
            row["score_matches_planner"] = (
                None if recorded is None or not math.isfinite(recorded)
                else bool(abs(recorded - float(own["score"])) <= 1e-6 * max(1.0, abs(recorded))))
        rows.append(row)
    dump = space.assign(**{c: shared[c] for c in ("num_qubits", "k", "circuit_id")})
    return {"rows": rows, "dump": dump.to_dict("records")}


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = NAME + ("_smoke" if settings.smoke else "")
    table = ResultTable(out / f"{stem}.csv", key_columns=["num_qubits", "k", "circuit_id", "arm"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    dump = ResultTable(out / f"{stem}_space.csv",
                       key_columns=["num_qubits", "k", "circuit_id", "translator", "commute",
                                    "placement_index"],
                       experiment=NAME, settings=asdict(settings), resume=resume)
    jobs = jobs_for(settings)
    todo = [j for j in jobs if not table.is_done((j[1], j[2], j[3], "w10"))]
    todo.sort(key=lambda j: -j[1])
    log(NAME, f"{len(jobs)} (config, circuit) points, {len(todo)} to run")

    def on_result(_index: int, result) -> None:
        dump.append(result["dump"])
        table.append(result["rows"])

    counts = stream_map(_run_point, todo, jobs=settings.jobs, label=NAME, on_result=on_result)
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"points": len(jobs), "failed": counts["error"]})
    dump.write_sidecar(status)
    log(NAME, f"done: {counts}; wrote {table.path}")
    summarize(table.frame())
    return table.path


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame[frame["in_space"].astype(bool)]
    columns = ["score_regret_pct", "measured_regret_pct", "fair_work_depth_regret_pct",
               "fair_depth_regret_pct", "fair_two_qubit_gates_regret_pct", "silent_regret_pct",
               "score_optimal"]
    table = frame.groupby(["config", "arm"])[columns].mean()
    table["circuits"] = frame.groupby(["config", "arm"]).size()
    print(f"\n[{NAME}] mean regret above the exhaustive optimum (%), complete tightest space x both "
          f"commutation settings; P(optimal) on the weight-10 score")
    print(table.round(2).to_string())
    beats = frame[[c for c in frame.columns if c.endswith("_beats_space")]].astype(bool).any(axis=1)
    print(f"[{NAME}] arms beating the space (must be 0): {int(beats.sum())}")
    if "score_matches_planner" in frame:
        checked = frame["score_matches_planner"].dropna()
        print(f"[{NAME}] recomputed score equals the planner's own: "
              f"{int(checked.astype(bool).sum())}/{len(checked)}")
    return table


def parse_args() -> Tuple[Settings, bool]:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=None, help="q:k, e.g. 4:2 6:2")
    parser.add_argument("--random", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    settings = Settings(random_circuits=args.random, jobs=args.jobs, smoke=args.smoke)
    if args.configs:
        settings.configs = tuple(tuple(int(x) for x in c.split(":")) for c in args.configs)
    if args.smoke and not args.configs:
        settings.configs = ((4, 2),)
    if args.out:
        settings.out = args.out
    return settings, not args.no_resume


if __name__ == "__main__":
    run(*parse_args())
