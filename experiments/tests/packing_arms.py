"""
Why is the packing segment smaller than in the published figure?

Four arms, same circuits, same metrics, circuit counts only (no simulation):

    A  auto commute + packing.plan_packing(verify=True)     <- the new figure
    B  forced commute ON + packing.plan_packing(verify=True)
    C  forced commute ON + packing.plan_packing(verify=False)
    D  forced commute ON + MergeMaximizingPacker            <- the published figure

A->B isolates the commutation decision, B->C isolates verify's fallback,
C->D isolates the packer itself.
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd
import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from archive.packer.merge_packer import MergeMaximizingPacker
from experiments.shared.benchmark_circuits import load_proxy_circuits
from packing.plan import plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation
from translator.passes import cancel_inverse_gates, commute_hadamards

Perm = List[Tuple[int, int]]
N_CODE = 4


def tight(num_qubits: int, n_codes: int) -> Perm:
    cap = max(1, math.ceil(num_qubits / n_codes))
    return [(min(i // cap, n_codes - 1), i % cap) for i in range(num_qubits)]


def cfg(*, merging: bool, commute: bool) -> PipelineConfig:
    c = PipelineConfig(verbose=False, save=False)
    c.translator_config.n_code = N_CODE
    c.translator_config.enable_merging = merging
    c.translator_config.enable_h_commutation = commute
    return c


def metrics(program, perm, config) -> Tuple[int, int, int]:
    circuit = translate_permutation(program, perm, config=config).circuit
    one, two = one_two_qubit_gate_totals(circuit)
    return one, two, circuit.depth()


def pct(reference: float, challenger: float) -> float:
    return 0.0 if reference <= 0 else 100.0 * (reference - challenger) / reference


def preprocess(program, commute: bool):
    out = cancel_inverse_gates(program)
    if commute:
        out = cancel_inverse_gates(commute_hadamards(out))
    return out


def main() -> None:
    only = set(sys.argv[1:])
    circuits = load_proxy_circuits(
        Path("benchmarks/qasmbench"), gate_limit=2000, qubit_limit=100,
        timeout_seconds=1.0,
    )
    if only:
        circuits = [(n, c) for n, c in circuits if n in only]
    print(f"[diag] {len(circuits)} circuits at n_code={N_CODE} (k=2)\n")

    old_packer = MergeMaximizingPacker()
    rows: List[Dict] = []

    for name, program in circuits:
        nq = program.num_qubits
        codes = math.ceil(nq / 2)
        base_perm = tight(nq, codes)

        plan_auto = plan_packing(program, N_CODE, h_commutation="auto", verify=True)
        commute_auto = plan_auto.h_commutation
        plan_on = plan_packing(program, N_CODE, h_commutation=True, verify=True)
        plan_on_nv = plan_packing(program, N_CODE, h_commutation=True, verify=False)
        try:
            old_perm = old_packer.pack(
                preprocess(program, True), num_codes=codes
            ).permutation
        except Exception as error:  # noqa: BLE001
            print(f"  !! {name}: old packer failed: {type(error).__name__}: {error}")
            old_perm = None

        row: Dict = {
            "circuit": name, "qubits": nq,
            "commute_auto": commute_auto,
            "fell_back_auto": plan_auto.fell_back,
            "fell_back_on": plan_on.fell_back,
        }

        # Per arm: the cumulative reduction of the packed row minus the h+merge row,
        # both translated under that arm's commutation setting. That difference is
        # exactly the blue "+ Packing" segment in the figure.
        arms = {
            "A": (commute_auto, plan_auto.placement),
            "B": (True, plan_on.placement),
            "C": (True, plan_on_nv.placement),
        }
        if old_perm is not None:
            arms["D"] = (True, old_perm)

        for arm, (commute, packed_perm) in arms.items():
            try:
                b1, b2, bd = metrics(program, base_perm, cfg(merging=False, commute=False))
                m1, m2, md = metrics(program, base_perm, cfg(merging=True, commute=commute))
                p1, p2, pd_ = metrics(program, packed_perm, cfg(merging=True, commute=commute))
            except Exception as error:  # noqa: BLE001
                print(f"  !! {name} arm {arm}: {type(error).__name__}: {error}")
                continue
            row[f"{arm}_1q"] = pct(b1, p1) - pct(b1, m1)
            row[f"{arm}_2q"] = pct(b2, p2) - pct(b2, m2)
            row[f"{arm}_depth"] = pct(bd, pd_) - pct(bd, md)
            row[f"{arm}_cum_depth"] = pct(bd, pd_)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv("packing_arms.csv", index=False)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    cols = ["circuit", "qubits", "commute_auto", "fell_back_auto", "fell_back_on"]
    for arm in ("A", "B", "C", "D"):
        cols += [c for c in (f"{arm}_1q", f"{arm}_2q", f"{arm}_depth") if c in frame]
    print(frame[cols].round(1).to_string(index=False))

    print("\n=== mean '+ Packing' segment, clipped at zero like the figure ===")
    for arm in ("A", "B", "C", "D"):
        parts = []
        for metric in ("1q", "2q", "depth"):
            column = f"{arm}_{metric}"
            if column in frame:
                parts.append(f"{metric}={frame[column].clip(lower=0.0).mean():6.2f}%")
        if parts:
            print(f"  {arm}: " + "  ".join(parts))

    print("\n=== counts ===")
    print(f"  commute auto == OFF : {(~frame.commute_auto).sum()} / {len(frame)}")
    print(f"  verify fell back (auto): {frame.fell_back_auto.sum()} / {len(frame)}")
    print(f"  verify fell back (forced on): {frame.fell_back_on.sum()} / {len(frame)}")


if __name__ == "__main__":
    main()
