"""
planner_static.py -- Experiment8a: plan Fig 6's circuits under the silent-aware cost model.

For each of Fig 6's 71 circuits at n = 4 (Fig 6's code size, and the one Experiment1c
simulated) and each silent weight, run `cost_model.plan_silent_aware` with Fig 6's noise
profile on the original circuit, as the shipped planner is run. Weight 0 *is* the shipped
planner; its placement and commutation decision must equal Experiment1a's `aqorn` arm on
every circuit (`matches_shipped`), which is the check that nothing but the cost model
changed.

Recorded per (circuit, weight): the placement and commutation decision, whether `verify`
fell back to naive and in which currency it measured, the model's predicted silent errors
(on the original the planner saw, and on the proxy Fig 6 simulates), the share of the
proxy's two-qubit gates placed inside a patch, gate counts and depth of the original (as
Fig 6 counts them) and the expected-error split of the proxy. Experiment8b simulates.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import time
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
    fig6_noise_profile,
    load_fig6_circuits,
    log,
    out_dir,
    stream_map,
    translate_metrics,
)

NAME = "exp8a"
SMOKE_NAMES = ("adder_n10", "bell_n4", "knn_n25", "ising_n42")


@dataclass
class Settings:
    weights: Sequence[float] = (0.0, 1.0, 3.0, 10.0, 30.0)
    n_code: int = 4
    shipped_csv: str = str(RESULTS / "exp1" / "packer_static_o1.csv")
    smoke: bool = False
    names: Sequence[str] = ()
    jobs: int = 12
    out: str = str(RESULTS / "exp8")


def in_patch_share(circuit, placement) -> float:
    """Fraction of the circuit's two-qubit gates whose operands share a patch."""
    patch = [int(p) for p, _ in placement]
    inside = total = 0
    for instruction in circuit.data:
        if len(instruction.qubits) != 2 or instruction.operation.name in ("measure", "barrier"):
            continue
        a, b = (circuit.find_bit(q).index for q in instruction.qubits)
        total += 1
        inside += patch[a] == patch[b]
    return inside / total if total else float("nan")


def _run_point(job) -> List[Dict]:
    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware, silent_errors
    from experiments.revision.exp8_undetected.gadgets import load_rates
    from experiments.shared.simulation import ensure_measurements

    settings_dict, circuit, weight = job
    settings = Settings(**settings_dict)
    rates = load_rates(settings.n_code, out=settings.out)
    noise = fig6_noise_profile()
    program = ensure_measurements(circuit.original)
    proxy = ensure_measurements(circuit.proxy)

    began = time.time()
    plan = plan_silent_aware(program, settings.n_code, rates=rates, silent_weight=weight,
                             noise=noise)
    seconds = time.time() - began
    placement = [tuple(int(v) for v in p) for p in plan.placement]
    commute = bool(plan.h_commutation)
    row = {
        "name": circuit.name, "size_group": circuit.size_group,
        "circuit_idx": circuit.circuit_idx, "family": circuit.family,
        "num_qubits": program.num_qubits, "n_code": settings.n_code,
        "silent_weight": float(weight), "commute": commute,
        "fell_back": bool(plan.fell_back), "verify_currency": plan.currency,
        "plan_seconds": seconds,
        "silent_pred_original": silent_errors(program, placement, settings.n_code,
                                              commute=commute, noise=noise, rates=rates),
        "silent_pred_proxy": silent_errors(proxy, placement, settings.n_code,
                                           commute=commute, noise=noise, rates=rates),
        "in_patch_share_proxy": in_patch_share(proxy, placement),
        "placement": json.dumps([list(p) for p in placement]),
    }
    original = translate_metrics(program, placement, settings.n_code, commute=commute)
    if original is not None:
        row.update({key: original[key] for key in ("one_qubit_gates", "two_qubit_gates",
                                                    "depth", "work", "work_depth")})
    split = translate_metrics(proxy, placement, settings.n_code, commute=commute,
                              noise_split=True)
    if split is not None:
        row.update({key: value for key, value in split.items() if key.startswith("E_")})
        row["proxy_two_qubit_gates"] = split["two_qubit_gates"]
        row["proxy_depth"] = split["depth"]
    return [row]


def shipped_placements(path: Path, n_code: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame["n_code"] == n_code) & (frame["arm"] == "aqorn")
                  & (frame["status"] == "ok")]
    return frame.set_index("name")[["placement", "commute"]]


def run(settings: Settings, resume: bool = True) -> Path:
    from experiments.revision.exp8_undetected.gadgets import load_rates

    out = Path(settings.out)
    load_rates(settings.n_code, out=out)  # measure the gadget table once, before any worker
    table = ResultTable(out / "planner_static.csv", key_columns=["name", "silent_weight"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    names = settings.names or (SMOKE_NAMES if settings.smoke else None)
    circuits = load_fig6_circuits(names=names)
    jobs = [(asdict(settings), c, float(w)) for c in circuits for w in settings.weights
            if not table.is_done((c.name, float(w)))]
    jobs.sort(key=lambda j: -j[1].num_qubits)
    log(NAME, f"{len(circuits)} circuits x weights {list(settings.weights)}: "
              f"{len(jobs)} plans to run")
    counts = stream_map(_run_point, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))

    frame = table.frame()
    shipped = shipped_placements(Path(settings.shipped_csv), settings.n_code)
    zero = frame[frame["silent_weight"] == 0.0].set_index("name")
    both = zero.join(shipped, rsuffix="_shipped", how="inner")
    mismatched = sorted(both.index[(both["placement"] != both["placement_shipped"])
                                   | (both["commute"] != both["commute_shipped"])])
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={
        "failed": counts["error"], "weight0_checked": int(len(both)),
        "weight0_mismatched_shipped": mismatched,
    })
    log(NAME, f"weight 0 vs shipped AQORN: {len(both) - len(mismatched)}/{len(both)} identical"
              + (f"; MISMATCHED: {mismatched}" if mismatched else ""))
    report(frame)
    return table.path


def report(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    base = frame[frame["silent_weight"] == 0.0].set_index("name")
    rows = []
    for weight, block in frame.groupby("silent_weight"):
        block = block.set_index("name")
        same = (block["placement"] == base["placement"].reindex(block.index)) & (
            block["commute"] == base["commute"].reindex(block.index))
        rows.append({
            "weight": weight, "circuits": len(block), "changed": int((~same).sum()),
            "commute_changed": int((block["commute"] != base["commute"].reindex(block.index)).sum()),
            "fell_back": int(block["fell_back"].sum()),
            "silent_pred_proxy": block["silent_pred_proxy"].mean(),
            "in_patch_share": block["in_patch_share_proxy"].mean(),
            "E_total_proxy": block.get("E_total", pd.Series(dtype=float)).mean(),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp8")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--names", nargs="+", default=[])
    args = parser.parse_args()
    settings = Settings(weights=tuple(args.weights), smoke=args.smoke, names=tuple(args.names),
                        jobs=args.jobs, out=str(out_dir(args, "exp8")))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
