"""
packer_static.py -- Experiment1a/1b: every packer, the same passes, gate counts and depth.

Questions: how much packing gains over a conventional locality-aware partitioner under
the same commutation and merging passes; over an interaction-based partitioner and a greedy
search at a comparable budget; the per-family breakdown (done in `summarize.py` from this
CSV); and -- with `--preopt o3` -- how much of the depth reduction survives logical-level
optimization of the input.

Per (circuit, code size): the shipped compiler decides H commutation; every packer in
`experiments.revision.packers` is run under that decision; every placement is translated
with merging on. The three Fig 6 ablation rows (baseline: naive, both passes off; +H;
+H+merge) are translated too, so the cumulative stack can be rebuilt on any input.

Gate counts and depth come from the **original** circuit, as Fig 6's do. The movement and
idle breakdown (`E_*`, expected detectable errors at the NEUTRAL_ATOM rates, from
`simulator.attribution.measured_noise_split`) comes from the **proxy** under the same
placement -- the attribution walks the Stim schedule, which rejects rz/rx, and the proxy
is the circuit Fig 6 simulates. That breakdown is an Experiment5 preview (reductions in movement
and idle time).

Placements are written into the CSV (`placement` column, JSON) so Experiment1c can simulate
exactly these arms without re-running the planners, which take minutes on 140 qubits.

Recorded per circuit, because it changes what AQORN's `verify` guaranteed: whether the
original carries rz/rx. If it does, `measured_noise_split` raises inside
`plan.measured_cost`, which then silently prices the plan on gates + depth instead of
expected error -- so for those circuits the "never worse than naive" check, and the
commutation choice, were made on gate counts (`verify_criterion` column).
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

import pandas as pd

from experiments.revision.common import (
    RESULTS,
    ResultTable,
    add_common_arguments,
    count_rotations,
    fig6_noise_profile,
    load_fig6_circuits,
    log,
    out_dir,
    stream_map,
    translate_metrics,
)

NAME = "exp1a"


@dataclass
class Settings:
    preopt: str = "o1"
    n_codes: Sequence[int] = (4, 6, 8)
    noise_profile: str = "fig6"  # what the planner and every safety net are priced with
    random_seed: int = 0
    smoke: bool = False
    names: Sequence[str] = ()
    jobs: int = 12
    out: str = str(RESULTS / "exp1")


SMOKE_NAMES = ("adder_n10", "qft_n18", "knn_n25", "ising_n42")


def _run_point(job) -> List[Dict]:
    from experiments.revision.packers import run_all_packers
    from experiments.shared.simulation import ensure_measurements
    from iceberg.utils import logical_capacity
    from packing.plan import naive_packing

    settings_dict, circuit, n_code = job
    settings = Settings(**settings_dict)
    profile = fig6_noise_profile()
    program = ensure_measurements(circuit.original)
    proxy = ensure_measurements(circuit.proxy)
    k = logical_capacity(n_code)
    rotations = count_rotations(circuit.original)

    began = time.time()
    placements, notes, timings = run_all_packers(
        program, n_code, noise=profile, random_seed=settings.random_seed
    )
    packers_seconds = time.time() - began
    commute = bool(notes["commute"])
    naive = [tuple(p) for p in naive_packing(program.num_qubits, k)]

    arms: List[Tuple[str, List, bool, bool]] = [
        ("fig6_baseline", naive, False, False),
        ("fig6_h_only", naive, commute, False),
        ("fig6_h_merge", naive, commute, True),
    ]
    arms += [(arm, placement, commute, True) for arm, placement in placements.items()]

    shared = {
        "name": circuit.name, "size_group": circuit.size_group,
        "circuit_idx": circuit.circuit_idx, "family": circuit.family,
        "num_qubits": program.num_qubits, "n_code": n_code, "k": k,
        "preopt": settings.preopt, "commute": commute,
        "rotations_in_original": rotations,
        "verify_criterion": "gates_depth_fallback" if rotations else "expected_error",
        "logical_gates": circuit.original.size(), "logical_depth": circuit.original.depth(),
        "packers_seconds": packers_seconds,
        **{f"note_{key}": value for key, value in notes.items()
           if isinstance(value, (bool, int, float, str))},
    }
    rows: List[Dict] = []
    # Arms often share a placement (most safety-net twins fall back to naive), and on
    # 140 qubits one attribution costs seconds. Same placement + same passes = same
    # translation, so measure each distinct one once.
    cache: Dict[tuple, tuple] = {}
    for arm, placement, h, merge in arms:
        row = dict(shared, arm=arm, h_commutation=h, merging=merge)
        if placement is None:
            row["status"] = "not_applicable_or_failed"
            rows.append(row)
            continue
        key = (tuple(tuple(p) for p in placement), h, merge)
        if key not in cache:
            cache[key] = (
                translate_metrics(program, placement, n_code, commute=h, merging=merge),
                translate_metrics(proxy, placement, n_code, commute=h, merging=merge,
                                  noise_split=True),
            )
        metrics, split = cache[key]
        metrics = dict(metrics) if metrics is not None else None
        if metrics is None:
            row["status"] = "rejected"
            rows.append(row)
            continue
        row.update(metrics)
        row["status"] = "ok"
        if split is not None:
            row.update({key: value for key, value in split.items() if key.startswith("E_")})
            row["proxy_depth"] = split["depth"]
            row["proxy_two_qubit_gates"] = split["two_qubit_gates"]
        row["seconds"] = timings.get(arm.split("+")[0], math.nan)
        row["placement"] = json.dumps([list(p) for p in placement])
        rows.append(row)
    return rows


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / f"packer_static_{settings.preopt}.csv",
                        key_columns=["name", "n_code", "arm"], experiment=NAME,
                        settings=asdict(settings), resume=resume)
    level = 3 if settings.preopt == "o3" else 1
    names = settings.names or (SMOKE_NAMES if settings.smoke else None)
    circuits = load_fig6_circuits(names=names, optimization_level=level)
    jobs = [(asdict(settings), c, n) for n in settings.n_codes for c in circuits
            if not table.is_done((c.name, n, "aqorn"))]
    # Longest first: the planners are superlinear in width.
    jobs.sort(key=lambda j: -j[1].num_qubits * (8 / j[2]))
    log(NAME, f"{len(circuits)} circuits x n_code {list(settings.n_codes)}, "
              f"preopt={settings.preopt}: {len(jobs)} points to run")
    counts = stream_map(_run_point, jobs, jobs=settings.jobs, label=f"{NAME}-{settings.preopt}",
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    report(table.frame())
    return table.path


def report(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    ok = frame[frame["status"] == "ok"]
    base = ok[ok["arm"] == "fig6_baseline"].set_index(["name", "n_code"])
    for metric in ("depth", "two_qubit_gates", "work"):
        merged = ok.join(base[[metric]], on=["name", "n_code"], rsuffix="_base")
        merged["reduction"] = 100 * (merged[f"{metric}_base"] - merged[metric]) / merged[f"{metric}_base"]
        table = merged.pivot_table(index="arm", columns="n_code", values="reduction", aggfunc="mean")
        print(f"\n[{NAME}] mean % reduction in {metric} vs Fig 6 baseline (naive, passes off)")
        print(table.round(2).to_string())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp1")
    parser.add_argument("--preopt", choices=["o1", "o3"], default="o1")
    parser.add_argument("--n-codes", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--names", nargs="+", default=[])
    args = parser.parse_args()
    settings = Settings(preopt=args.preopt, n_codes=tuple(args.n_codes),
                        smoke=args.smoke, names=tuple(args.names), jobs=args.jobs,
                        out=str(out_dir(args, "exp1")))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
