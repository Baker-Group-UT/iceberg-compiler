"""
planted.py -- circuits built so that one packing is provably optimal: can AQORN find it?

Construction (k = 2, [[4,2,2]]). Hidden pairs (a_j, b_j), j = 1..m, n = 2m program qubits. Each
layer applies *pair operations* to disjoint pairs:

    HH(j)        H on a_j and on b_j
    CXT(i -> j)  CX(a_i -> a_j) and CX(b_i -> b_j)

then the labels are scrambled by a random permutation, so nothing about the program order hints at
the pairs. Under the planted packing -- pair j in patch j, a_j on lane 0, b_j on lane 1 -- every H
is half of a whole-patch HH and every CX half of a lane-aligned transversal CXT.

Why that packing is optimal (gate counts, `iceberg.costs.work_table(4)`, weights (1, 13), one
carried atom = 5 units):
    H    alone (7, 4) = 59 units;  as half of HH (4, 0) -> 2  (+ the anchor swap's two carriages,
         movement rather than gates: 5 units each, so 7 per H counting them)
    CX   alone (0, 4) = 52 (in-patch and cross-patch alike); as half of CXT (0, 4) -> 26
Merging can only replace gates by macros, so under ANY packing each H costs >= 2 and each CX >= 26:
sum over gates of those minima is a lower bound on the logical-gate work of every placement, and the
planted packing attains it whenever every pair operation fires as a macro -- checked per circuit by
translating it (`planted_attains_bound`). It is also silent-free (the macros leak no silent error at
first order), the model's lower bound of 0. CZ is left out on purpose: w10T's transversal in-patch
CZ (S on every data qubit, 6 units) undercuts CZT's 26, so with CZ in the mix the bound would not hold
for that translator. H commutation is off for every arm (`h_commutation=False`) so the comparison is
about packing alone; patch *order* is not covered by the bound (it moves transport, not gates).

Decoys (`--decoy`): with probability rho per layer, one extra gate that breaks the structure (a lone
H, or a CX between members of different pairs on misaligned lanes). With decoys the planted packing
is only a reference point -- an arm may beat it.

Arms: naive (tight, program order -- random after scrambling), v4 (`plan_packing`), w10, w10T, the
interaction-count partition, and the planted packing. Measured: planted pairs recovered, translated
work (weighted gates; fair schedule) and depth, model silent error, the weight-10 score. For n <= 6 the
complete tightest space is enumerated too, so "planted = optimum" is checked rather than assumed.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.planted --jobs 3
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import math
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.optimality.exhaustive_silent import RATES, RATES_T, Scorer, fair_metrics, switched

NAME = "planted"
#: Minimum weighted gate work per logical gate (weights (1, 13); carriages are movement, not gates)
H_SHARE, CX_SHARE = 2.0, 26.0


@dataclass
class Settings:
    sizes: Sequence[int] = (4, 6, 8, 12, 16, 24, 32)
    decoys: Sequence[float] = (0.0, 0.1, 0.3)
    seeds: int = 6
    layers_per_qubit: float = 2.0
    exhaustive_max: int = 6
    silent_weight: float = 10.0
    jobs: int = 3
    out: str = "results/optimality"
    tag: str = ""


def planted_circuit(n: int, layers: int, decoy: float, seed: int):
    """(program, planted placement, hidden pairs in program labels, op counts)."""
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    m = n // 2
    hidden = QuantumCircuit(n)
    last: Dict[int, Tuple] = {}
    counts = {"HH": 0, "CXT": 0, "decoy": 0}
    for _ in range(layers):
        free = list(range(m))
        rng.shuffle(free)
        while free:
            j = free.pop()
            if free and rng.random() < 0.6:
                i = free.pop()
                control, target = (i, j) if rng.random() < 0.5 else (j, i)
                op = ("CXT", control, target)
                if last.get(control) == op or last.get(target) == op:
                    control, target = target, control      # the repeat would cancel: reverse it
                    op = ("CXT", control, target)
                for lane in (0, 1):
                    hidden.cx(2 * control + lane, 2 * target + lane)
                last[control] = last[target] = op
                counts["CXT"] += 1
            elif rng.random() < 0.7:
                if last.get(j) == ("HH", j):
                    continue
                hidden.h(2 * j)
                hidden.h(2 * j + 1)
                last[j] = ("HH", j)
                counts["HH"] += 1
        if decoy and rng.random() < decoy:
            if rng.random() < 0.5:
                q = rng.randrange(n)
                hidden.h(q)
                last.pop(q // 2, None)
            else:
                i, j = rng.sample(range(m), 2)
                hidden.cx(2 * i, 2 * j + 1)
                last.pop(i, None)
                last.pop(j, None)
            counts["decoy"] += 1
    order = list(range(n))
    rng.shuffle(order)                      # hidden qubit h -> program qubit order[h]
    program = QuantumCircuit(n)
    for instruction in hidden.data:
        qubits = [order[hidden.find_bit(q).index] for q in instruction.qubits]
        program.append(instruction.operation, qubits)
    placement: List[Optional[Tuple[int, int]]] = [None] * n
    for h in range(n):
        placement[order[h]] = (h // 2, h % 2)
    pairs = [frozenset((order[2 * j], order[2 * j + 1])) for j in range(m)]
    return program, [tuple(p) for p in placement], pairs, counts


def pairs_recovered(placement, pairs) -> float:
    groups: Dict[int, set] = {}
    for q, (patch, _) in enumerate(placement):
        groups.setdefault(int(patch), set()).add(q)
    found = {frozenset(g) for g in groups.values()}
    return sum(p in found for p in pairs) / len(pairs)


def logical_bound(program, commute: bool = False) -> float:
    """Sum over logical gates of their minimum share (H 7, CX 26), after the logical passes."""
    from translator.passes import cancel_inverse_gates, commute_hadamards

    circuit = cancel_inverse_gates(program)
    if commute:
        circuit = cancel_inverse_gates(commute_hadamards(circuit))
    total = 0.0
    for instruction in circuit.data:
        name = instruction.operation.name
        total += H_SHARE if name == "h" else CX_SHARE if name == "cx" else 0.0
    return total


def _run_point(job) -> List[Dict]:
    from experiments.revision.common import fig6_noise_profile, translator_config
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from experiments.revision.packers import interaction_partition, make_context
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing, plan_packing
    from pipeline.tasks.translate_permutation import translate_permutation

    settings_dict, n, decoy, seed = job
    settings = Settings(**settings_dict)
    layers = max(2, int(round(settings.layers_per_qubit * n)))
    circuit, planted, pairs, counts = planted_circuit(n, layers, decoy, seed)
    program = ensure_measurements(circuit)
    noise = fig6_noise_profile()
    rates = {"shipped": UndetectedRates.from_table(pd.read_csv(RATES), 4, 3e-3),
             "transversal": UndetectedRates.from_table(pd.read_csv(RATES_T), 4, 3e-3)}

    arms: Dict[str, Tuple[List[Tuple[int, int]], str]] = {"planted": (planted, "both"),
                                                           "naive": (naive_packing(n, 2), "both")}
    seconds: Dict[str, float] = {}
    with switched(False):
        began = time.time()
        v4 = plan_packing(program, 4, h_commutation=False, verify=True, noise=noise)
        seconds["v4"] = time.time() - began
        arms["v4"] = ([tuple(p) for p in v4.placement], "shipped")
        began = time.time()
        w10 = plan_silent_aware(program, 4, rates=rates["shipped"],
                                silent_weight=settings.silent_weight, noise=noise,
                                h_commutation=False)
        seconds["w10"] = time.time() - began
        arms["w10"] = ([tuple(p) for p in w10.placement], "shipped")
        try:
            arms["interaction_partition"] = (
                [tuple(p) for p in interaction_partition(make_context(program, 4, False, noise))],
                "shipped")
        except Exception:  # noqa: BLE001
            pass
    with switched(True):
        began = time.time()
        w10T = plan_silent_aware(program, 4, rates=rates["transversal"],
                                 silent_weight=settings.silent_weight, noise=noise,
                                 h_commutation=False)
        seconds["w10T"] = time.time() - began
        arms["w10T"] = ([tuple(p) for p in w10T.placement], "transversal")

    from qiskit import QuantumCircuit

    empty = fair_metrics(ensure_measurements(QuantumCircuit(n)), naive_packing(n, 2), 4, False)
    overhead = empty["fair_one_qubit_gates"] + 13.0 * empty["fair_two_qubit_gates"]

    def measure(placement, translator: str) -> Dict[str, float]:
        with switched(translator == "transversal"):
            scorer = Scorer(program, 4, rates[translator], noise, settings.silent_weight)
            values = scorer(placement, False)
            values.update(fair_metrics(program, placement, 4, False))
        # weighted gates of the logical gates alone: encoding and syndrome extraction removed
        values["logical_work"] = (values["fair_one_qubit_gates"]
                                  + 13.0 * values["fair_two_qubit_gates"] - overhead)
        return values

    bound = logical_bound(program)
    shared = {"n": n, "decoy": decoy, "seed": seed, "layers": layers, **{f"ops_{k}": v for k, v in counts.items()},
              "logical_bound": bound, "logical_gates": len(circuit.data)}
    planted_values = {t: measure(planted, t) for t in ("shipped", "transversal")}
    rows: List[Dict] = []
    for arm, (placement, translator) in arms.items():
        for t in (("shipped", "transversal") if translator == "both" else (translator,)):
            values = planted_values[t] if arm == "planted" else measure(placement, t)
            row = dict(shared, arm=arm if translator != "both" else f"{arm}", translator=t,
                       placement=json.dumps([list(p) for p in placement]),
                       pairs_recovered=pairs_recovered(placement, pairs),
                       plan_seconds=seconds.get(arm, 0.0), **values)
            for key in ("fair_work_depth", "fair_depth", "fair_two_qubit_gates", "silent", "score"):
                reference = planted_values[t][key]
                row[f"{key}_vs_planted_pct"] = ((values[key] / reference - 1) * 100
                                               if reference and math.isfinite(reference) else math.nan)
            rows.append(row)

    if n <= settings.exhaustive_max:
        for t in ("shipped", "transversal"):
            with switched(t == "transversal"):
                scorer = Scorer(program, 4, rates[t], noise, settings.silent_weight)
                best = {"fair_work_depth": math.inf, "silent": math.inf, "score": math.inf,
                        "fair_two_qubit_gates": math.inf}
                planted_partition_best = math.inf
                for placement, _ in exhaustive_space(n, 2, "tightest"):
                    placement = [tuple(p) for p in placement]
                    values = scorer(placement, False)
                    values.update(fair_metrics(program, placement, 4, False))
                    for key in best:
                        best[key] = min(best[key], values[key])
                    if pairs_recovered(placement, pairs) == 1.0:
                        planted_partition_best = min(planted_partition_best, values["score"])
            for row in rows:
                if row["translator"] == t:
                    for key, value in best.items():
                        row[f"{key}_opt"] = value
                        row[f"{key}_regret_pct"] = (row[key] / value - 1) * 100 if value else math.nan
                    row["planted_partition_best_score"] = planted_partition_best
    for row in rows:
        planted_row = planted_values[row["translator"]]
        row["planted_attains_bound"] = bool(abs(planted_row["logical_work"] - bound) < 1e-6)
        row["logical_work_vs_bound_pct"] = (row["logical_work"] / bound - 1) * 100 if bound else math.nan
    return rows


def run(settings: Settings) -> Path:
    from experiments.revision.common import write_sidecar
    from experiments.shared.parallel import parallel_map

    jobs = [(asdict(settings), n, d, 7000 + 97 * n + s) for n in settings.sizes
            for d in settings.decoys for s in range(settings.seeds)]
    jobs.sort(key=lambda j: -j[1])
    print(f"[{NAME}] {len(jobs)} circuits")
    began = time.time()
    results = parallel_map(_run_point, jobs, jobs=settings.jobs, label=NAME)
    failed = [j[1:] for j, r in zip(jobs, results) if r is None]
    rows = pd.DataFrame([row for r in results if r for row in r])
    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}{'_' + settings.tag if settings.tag else ''}.csv"
    rows.to_csv(path, index=False)
    write_sidecar(path, experiment=NAME, settings=asdict(settings), started=began,
                  status="complete" if not failed else f"failed {failed}")
    print(f"[{NAME}] wrote {path}; failed {failed}")
    summarize(rows)
    return path


def summarize(rows: pd.DataFrame) -> None:
    shipped = rows[rows["translator"] == "shipped"]
    table = shipped.groupby(["n", "decoy", "arm"])[["pairs_recovered", "logical_work_vs_bound_pct",
                                                     "fair_work_depth_vs_planted_pct",
                                                     "silent_vs_planted_pct", "score_vs_planted_pct"]].mean()
    print(table.round(2).to_string())
    planted = shipped[shipped["arm"] == "planted"]
    print(f"[{NAME}] planted packing attains the gate-wise bound on "
          f"{int(planted['planted_attains_bound'].sum())}/{len(planted)} circuits")


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--decoy", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    settings = Settings(seeds=args.seeds, jobs=args.jobs, tag=args.tag)
    if args.sizes:
        settings.sizes = tuple(args.sizes)
    if args.decoy:
        settings.decoys = tuple(args.decoy)
    return settings


if __name__ == "__main__":
    run(parse_args())
