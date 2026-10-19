"""
proxy_static.py -- Experiment3a: does the Clifford proxy change what the compiler builds?

The structural half of the proxy question. Fig 6 *counts* gates on the original
circuits but *simulates* their Clifford proxies (every rz/rx replaced by s), under the
placement planned on the original. If the proxy compiles to a different structure, the
simulated fidelity describes a different circuit from the counted one. No simulation here.

Per circuit, at n_code = 4 and exactly Fig 6's pipeline:

* AQORN is planned on the original (Fig 6's call), giving a placement and a commutation
  decision. The baseline (naive, both passes off), naive with both passes, and AQORN are
  translated twice under identical settings -- once from the original, once from the
  proxy -- and compared: 1q and 2q gates, depth, the merges that fired, and AQORN's
  reduction against the baseline in each.
* AQORN is also planned *on the proxy*, to see whether a pipeline that only ever saw the
  proxy would have chosen the same placement and commutation. Part of any difference is
  mechanical and recorded: `plan.measured_cost` cannot price a schedule containing rz/rx
  (the Stim scanner rejects them), so on originals with rotations `verify` and the
  commutation choice fall back to gates + depth, while on the proxy they use expected
  error.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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
    non_clifford_rotations,
    out_dir,
    stream_map,
    translate_metrics,
)

NAME = "exp3a"
METRICS = ("one_qubit_gates", "two_qubit_gates", "depth", "merge_savings_1q",
           "merge_savings_2q", "merge_savings_weighted", "work")


@dataclass
class Settings:
    static_csv: str = str(RESULTS / "exp1" / "packer_static_o1.csv")
    smoke: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp3")


def _preset(path: str) -> Dict[str, tuple]:
    p = Path(path)
    if not p.exists():
        return {}
    frame = pd.read_csv(p)
    frame = frame[(frame["n_code"] == 4) & (frame["arm"] == "aqorn") & (frame["status"] == "ok")]
    return {r.name: (bool(r.commute), [tuple(x) for x in json.loads(r.placement)])
            for r in frame.itertuples(index=False)}


def _run_circuit(job) -> List[Dict]:
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing, plan_packing

    settings_dict, circuit, preset = job
    original = ensure_measurements(circuit.original)
    proxy = ensure_measurements(circuit.proxy)
    profile = fig6_noise_profile()
    if preset is None:
        plan = plan_packing(original, 4, h_commutation="auto", verify=True, noise=profile)
        commute, aqorn = bool(plan.h_commutation), [tuple(p) for p in plan.placement]
    else:
        commute, aqorn = preset
    proxy_plan = plan_packing(proxy, 4, h_commutation="auto", verify=True, noise=profile)
    proxy_aqorn = [tuple(p) for p in proxy_plan.placement]
    naive = [tuple(p) for p in naive_packing(circuit.num_qubits, 2)]

    arms = [("baseline", naive, False, False), ("h_merge", naive, commute, True),
            ("aqorn", aqorn, commute, True),
            ("aqorn_planned_on_proxy", proxy_aqorn, bool(proxy_plan.h_commutation), True)]
    rows: List[Dict] = []
    shared = {
        "name": circuit.name, "size_group": circuit.size_group, "family": circuit.family,
        "num_qubits": circuit.num_qubits,
        "rotations": count_rotations(circuit.original),
        "non_clifford_rotations": non_clifford_rotations(circuit.original),
        "commute_planned_on_original": commute,
        "commute_planned_on_proxy": bool(proxy_plan.h_commutation),
        "same_placement_when_planned_on_proxy": proxy_aqorn == aqorn,
        "proxy_plan_fell_back": bool(proxy_plan.fell_back),
        "verify_criterion_original": "gates_depth_fallback" if count_rotations(circuit.original)
        else "expected_error",
    }
    for arm, placement, h, merge in arms:
        a = translate_metrics(original, placement, 4, commute=h, merging=merge)
        b = translate_metrics(proxy, placement, 4, commute=h, merging=merge)
        row = dict(shared, arm=arm, h_commutation=h, merging=merge,
                   placement=json.dumps([list(p) for p in placement]))
        for metric in METRICS:
            row[f"original_{metric}"] = a[metric] if a else float("nan")
            row[f"proxy_{metric}"] = b[metric] if b else float("nan")
        row["identical_structure"] = bool(a and b and all(
            abs(a[m] - b[m]) < 1e-9 for m in ("one_qubit_gates", "two_qubit_gates", "depth")))
        rows.append(row)
    return rows


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / "proxy_static.csv", key_columns=["name", "arm"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    names = ("adder_n10", "qft_n18", "bv_n14", "vqe_n4") if settings.smoke else None
    circuits = load_fig6_circuits(names=names)
    presets = _preset(settings.static_csv)
    jobs = [(asdict(settings), c, presets.get(c.name)) for c in circuits
            if not table.is_done((c.name, "aqorn_planned_on_proxy"))]
    jobs.sort(key=lambda j: -j[1].num_qubits)
    log(NAME, f"{len(jobs)} circuits to run ({sum(1 for j in jobs if j[2])} with 1a placements)")
    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    report(table.frame())
    return table.path


def report(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for arm, block in frame.groupby("arm"):
        print(f"[{NAME}] {arm}: identical structure on {int(block['identical_structure'].sum())}"
              f"/{len(block)} circuits")
    one = frame.drop_duplicates("name")
    print(f"[{NAME}] planned on proxy: same placement on "
          f"{int(one['same_placement_when_planned_on_proxy'].sum())}/{len(one)}, same "
          f"commutation on {int((one['commute_planned_on_original'] == one['commute_planned_on_proxy']).sum())}"
          f"/{len(one)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp3")
    parser.add_argument("--static-csv", default=str(RESULTS / "exp1" / "packer_static_o1.csv"))
    args = parser.parse_args()
    return Settings(static_csv=args.static_csv, smoke=args.smoke, jobs=args.jobs,
                    out=str(out_dir(args, "exp3"))), args.resume


if __name__ == "__main__":
    run(*parse_args())
