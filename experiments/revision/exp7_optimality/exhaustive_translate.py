"""
exhaustive_translate.py -- Experiment7a (and the 7d stretch): every placement, translated.

Optimality gaps against exhaustive search on small instances, under the same objective.
Rebuilt from `archive/experiments/figures/8_optimality_vs_exhaustive`
(left untouched), with five changes:

* circuits: the 4-6 qubit QASMBench originals as well as random structured programs, and
  50 random programs per configuration instead of 20. The first 20 use the archived seeds
  (`circuit_seed + index`), so they are the archived run's circuits under the current
  cost model.
* arms: naive, `aqorn` (verify on), `aqorn_raw` (verify off) and every Experiment1
  baseline, each translated under the shipped compiler's commutation decision.
* two spaces: (i) the tightest patch count, which is what `plan_packing(spread=0)`
  searches, enumerated *completely* -- every partition x patch order x lane map, because
  the translator is not invariant under the relabellings the archived space quotiented
  out (see `space.py`; the archived optimum was beaten by 2% on a smoke circuit); and
  (ii) the full space, adding canonical representatives of every sparser patch count.
* a decomposition of AQORN's regret on the complete tightest space: the *partition gap*
  (best placement of AQORN's partition, above the optimum) and the *order/lane gap*
  (AQORN's placement above the best placement of its own partition).
* objectives: `work` (weighted gates, the archived objective), `work_depth` (weighted
  gates + 12 per layer -- what `plan.measured_cost` charges a gate-counting plan, i.e. the
  objective `verify` actually enforces), two-qubit gates and depth.
* the model: every placement's `PackingCost` is recorded too, which splits AQORN's regret
  into a *search gap* (AQORN's model value above the model's own optimum over the tightest
  space) and a *model gap* (the model's optimum, translated, above the translated optimum).

Programs carry measurements (`ensure_measurements`), as everywhere in Fig 6; the archived
run translated them without, so its absolute numbers are not comparable digit for digit.

The enumeration is streamed (`space.placements`), which is what makes (5, k=4) at 20,161
placements and (6, k=4) at 259,537 runnable at all -- the generator refused both.

    PYTHONPATH=. venv/bin/python -m experiments.revision.exp7_optimality.exhaustive_translate
    ... --configs 5:4 6:4 --random 20 --name exhaustive_translate_k4      # 7d
    ... --smoke
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import logging
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import (
    RESULTS,
    ResultTable,
    add_common_arguments,
    load_small_qasmbench,
    log,
    out_dir,
    partition_of,
    stream_map,
    translate_metrics,
)

NAME = "exp7a"
OBJECTIVES: Tuple[str, ...] = ("work", "work_depth", "two_qubit_gates", "depth")
#: Placement dumps are kept for spaces at most this large; (6, k=4) would be 5M rows.
DUMP_LIMIT = 25_000


@dataclass
class Settings:
    configs: Sequence[Tuple[int, int]] = ((4, 2), (5, 2), (6, 2))
    random_circuits: int = 50
    circuit_seed: int = 20260730  # the archived script's base seed
    gates_per_qubit: int = 10
    locality: float = 0.85
    qasmbench: bool = True
    space: str = "full"  # "full" or "tightest"
    table: str = "exhaustive_translate"
    smoke: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp7")


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
            names = names[:2]
        for name in names:
            jobs.append((asdict(settings), q, k, f"qb:{name}", "qasmbench", name, None,
                         small[name]))
        count = 2 if settings.smoke else settings.random_circuits
        for index in range(count):
            seed = settings.circuit_seed + index
            jobs.append((asdict(settings), q, k, f"rand:{seed}", "random", f"random_{index}",
                         seed, _random_circuit(q, seed, settings)))
    return jobs


def _run_point(job) -> Dict[str, List[Dict]]:
    """One (configuration, circuit): every arm, then the whole space, then the model."""
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.revision.packers import make_context, model_value, run_all_packers
    from experiments.shared.simulation import ensure_measurements
    from permutations.generator import canonical_relative_index_key
    from packing.lanes import assign_lanes
    from scipy.stats import spearmanr

    settings_dict, q, k, circuit_id, source, name, seed, circuit = job
    settings = Settings(**settings_dict)
    n_code = k + 2
    tight = -(-q // k)
    program = ensure_measurements(circuit)

    started = time.time()
    arms, notes, timings = run_all_packers(program, n_code, noise=None)
    commute = bool(notes["commute"])
    ctx = make_context(program, n_code, commute, None)
    plan_seconds = time.time() - started

    arm_metrics: Dict[str, Dict] = {}
    for arm, placement in arms.items():
        if placement is None:
            continue
        metrics = translate_metrics(program, placement, n_code, commute=commute)
        if metrics is None:
            continue
        canonical = [tuple(p) for p in canonical_relative_index_key(list(placement))]
        if canonical != list(placement):
            again = translate_metrics(program, canonical, n_code, commute=commute)
            metrics["canonical_invariant"] = bool(
                again is not None and all(abs(again[o] - metrics[o]) < 1e-9 for o in OBJECTIVES)
            )
        else:
            metrics["canonical_invariant"] = True
        metrics["model_value"] = model_value(ctx, placement)
        arm_metrics[arm] = metrics

    enumerated = time.time()
    model_cache: Dict[Tuple, float] = {}
    space_rows: List[Dict] = []
    rejected = 0
    for index, (placement, complete) in enumerate(exhaustive_space(q, k, settings.space)):
        metrics = translate_metrics(program, placement, n_code, commute=commute)
        if metrics is None:
            rejected += 1
            continue
        part = partition_of(placement)
        if part not in model_cache:
            model_cache[part] = model_value(ctx, placement)
        space_rows.append({
            "placement_index": index,
            "complete": bool(complete),
            "patches": int(metrics["patches"]),
            **{o: metrics[o] for o in OBJECTIVES},
            "one_qubit_gates": metrics["one_qubit_gates"],
            "model_value": model_cache[part],
            "partition": str(part),
        })
    enumerate_seconds = time.time() - enumerated
    space = pd.DataFrame(space_rows)
    tight_space = space[space["patches"] == tight]
    partition_best = tight_space.groupby("partition")[list(OBJECTIVES)].min()

    optimum = {
        "full": {o: float(space[o].min()) for o in OBJECTIVES} if settings.space == "full"
        else {o: math.nan for o in OBJECTIVES},
        "tight": {o: float(tight_space[o].min()) for o in OBJECTIVES},
    }

    # The model, on the space the search actually explores.
    model_opt = float(tight_space["model_value"].min())
    argmin = tight_space[np.isclose(tight_space["model_value"], model_opt, rtol=0, atol=1e-6)]
    translated_argmin = []
    for part_text in sorted(set(argmin["partition"])):
        groups = [frozenset(g) for g in eval(part_text)]  # noqa: S307 - our own repr
        lanes = assign_lanes(groups, ctx.cost)
        m = translate_metrics(program, [tuple(p) for p in lanes], n_code, commute=commute)
        if m is not None:
            translated_argmin.append(m["work_depth"])
    rho = spearmanr(tight_space["model_value"], tight_space["work_depth"]).statistic \
        if len(tight_space) > 2 else math.nan
    best_of_argmin = [float(partition_best.loc[t, "work_depth"])
                      for t in sorted(set(argmin["partition"])) if t in partition_best.index]

    shared = {
        "num_qubits": q, "k": k, "n_code": n_code, "config": f"{q}q,k={k}",
        "circuit_id": circuit_id, "source": source, "name": name, "seed": seed,
        "commute": commute, "space": settings.space,
        "space_full": int(len(space)) if settings.space == "full" else math.nan,
        "space_tight": int(len(tight_space)), "space_rejected": rejected,
        "tightest_patches": tight,
        "model_opt_tight": model_opt,
        "model_argmin_partitions": int(len(translated_argmin)),
        "model_argmin_partition_best_work_depth": min(best_of_argmin) if best_of_argmin else math.nan,
        "model_argmin_work_depth_min": float(min(translated_argmin)) if translated_argmin else math.nan,
        "model_argmin_work_depth_max": float(max(translated_argmin)) if translated_argmin else math.nan,
        "spearman_model_vs_work_depth": float(rho) if rho == rho else math.nan,
        "plan_seconds": plan_seconds,
        "enumerate_seconds": enumerate_seconds,
        "aqorn_fell_back": notes.get("aqorn_fell_back"),
        "replication_matches_aqorn_raw": notes.get("replication_matches_aqorn_raw"),
        "aqorn_total_evaluations": notes.get("aqorn_total_evaluations"),
        "random_budget_drawn": notes.get("random_budget_drawn"),
        "random_budget_capped": notes.get("random_budget_capped"),
    }
    best_tight = optimum["tight"]["work_depth"]
    if translated_argmin and best_tight > 0:
        shared["model_gap_pct"] = (min(translated_argmin) / best_tight - 1) * 100
    else:
        shared["model_gap_pct"] = math.nan

    rows: List[Dict] = []
    for arm, metrics in arm_metrics.items():
        row = dict(shared)
        row.update({
            "arm": arm,
            "patches": int(metrics["patches"]),
            "one_qubit_gates": metrics["one_qubit_gates"],
            "model_value": metrics["model_value"],
            "canonical_invariant": metrics["canonical_invariant"],
            "search_gap_pct": (metrics["model_value"] / model_opt - 1) * 100
            if model_opt > 0 else math.nan,
            "seconds": timings.get(arm.split("+")[0], math.nan),
        })
        part = str(partition_of(arms[arm]))
        for o in ("work_depth", "depth", "work"):
            best_tight = optimum["tight"][o]
            if part in partition_best.index and best_tight > 0:
                own = float(partition_best.loc[part, o])
                row[f"{o}_partition_gap_pct"] = (own / best_tight - 1) * 100
                row[f"{o}_order_lane_gap_pct"] = (metrics[o] / own - 1) * 100 if own > 0 else math.nan
            else:
                row[f"{o}_partition_gap_pct"] = math.nan
                row[f"{o}_order_lane_gap_pct"] = math.nan
        for o in OBJECTIVES:
            value = metrics[o]
            row[o] = value
            for which, frame in (("full", space), ("tight", tight_space)):
                best = optimum[which][o]
                if which == "full" and settings.space != "full":
                    continue
                row[f"{o}_opt_{which}"] = best
                row[f"{o}_regret_{which}"] = (value / best - 1) * 100 if best > 0 else math.nan
                row[f"{o}_optimal_{which}"] = bool(abs(value - best) < 1e-9)
                better = int((frame[o] < value - 1e-9).sum())
                row[f"{o}_rank_{which}"] = better + 1
                row[f"{o}_pct_better_{which}"] = 100.0 * better / max(1, len(frame))
                row[f"{o}_beats_space_{which}"] = bool(value < best - 1e-9)
        rows.append(row)

    dump: List[Dict] = []
    if len(space) <= DUMP_LIMIT:
        for record in space_rows:
            dump.append({"num_qubits": q, "k": k, "circuit_id": circuit_id, **{
                key: record[key] for key in
                ("placement_index", "complete", "patches", "work", "work_depth",
                 "two_qubit_gates", "depth", "one_qubit_gates", "model_value")
            }})
    return {"rows": rows, "dump": dump}


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    table = ResultTable(out / f"{settings.table}.csv",
                        key_columns=["num_qubits", "k", "circuit_id", "arm"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    dump = ResultTable(out / f"{settings.table}_space.csv",
                       key_columns=["num_qubits", "k", "circuit_id", "placement_index"],
                       experiment=NAME, settings=asdict(settings), resume=resume)
    jobs = jobs_for(settings)
    todo = [j for j in jobs if not table.is_done((j[1], j[2], j[3], "aqorn"))]
    log(NAME, f"{len(jobs)} (config, circuit) points, {len(todo)} to run, "
              f"configs {list(settings.configs)}, space={settings.space}")

    def on_result(_index: int, result: Dict[str, List[Dict]]) -> None:
        dump.append(result["dump"])
        table.append(result["rows"])

    counts = stream_map(_run_point, todo, jobs=settings.jobs, label=NAME, on_result=on_result)
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"points": len(jobs), "failed": counts["error"]})
    dump.write_sidecar(status)
    log(NAME, f"done: {counts}; wrote {table.path}")
    summarize(table.frame())
    return table.path


def summarize(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for which in ("tight", "full"):
        column = f"work_depth_regret_{which}"
        if column not in frame:
            continue
        print(f"\n[{NAME}] mean regret above the exhaustive optimum, {which} space (%)")
        table = frame.pivot_table(index="arm", columns="config",
                                  values=[f"{o}_regret_{which}" for o in ("work_depth", "depth")],
                                  aggfunc="mean")
        print(table.round(2).to_string())
    aq = frame[frame["arm"] == "aqorn"]
    print(f"\n[{NAME}] AQORN optimal (tight, work_depth): "
          f"{aq['work_depth_optimal_tight'].mean() * 100:.0f}% of {len(aq)} circuits; "
          f"search gap mean {frame[frame['arm'] == 'aqorn_raw']['search_gap_pct'].mean():.2f}%, "
          f"model gap mean {aq['model_gap_pct'].mean():.2f}%")


def parse_args() -> Tuple[Settings, bool]:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp7")
    parser.add_argument("--configs", nargs="+", default=None, help="q:k, e.g. 4:2 5:4")
    parser.add_argument("--random", type=int, default=50)
    parser.add_argument("--no-qasmbench", action="store_true")
    parser.add_argument("--space", choices=["full", "tightest"], default="full")
    parser.add_argument("--name", default="exhaustive_translate")
    args = parser.parse_args()
    configs = ((4, 2), (5, 2), (6, 2))
    if args.configs:
        configs = tuple(tuple(int(x) for x in c.split(":")) for c in args.configs)
    if args.smoke and not args.configs:
        configs = ((4, 2), (5, 2))
    settings = Settings(
        configs=configs, random_circuits=args.random, qasmbench=not args.no_qasmbench,
        space=args.space, table=args.name, smoke=args.smoke, jobs=args.jobs,
        out=str(out_dir(args, "exp7")),
    )
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
