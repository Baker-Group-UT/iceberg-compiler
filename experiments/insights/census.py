"""
census.py -- what each Fig 6 arm is made of: the gadget census behind every reduction.

Every logical gate the translator meets becomes one gadget, and each gadget's price is fixed
by the code (`iceberg.costs.work_table`, `CROSS_PATCH_COST`) and so is its share of detected and
silent error (`packing/data/gadget_rates*.csv`: exact single-fault enumeration per gadget at
alpha = 1). So the 1q / 2q totals of an arm, and the first-order error it carries, are a sum
over a census of gadget classes:

    H        a Hadamard on one lane            (7, 4)   silent 0.82 p2
    HH       Hadamards on both lanes, fused    (4, 0)   silent 0
    RZ, RX   a rotation on one lane            (1, 2)   silent 0.42 p2
    S, SX    phase gates on one lane           (2, 1) / (4, 1)  silent 0.2 p2
    X, Z     Paulis (and their fused XX / ZZ)  (2, 0)   silent 0
    CX_in    CX inside a patch                 (0, 4)   silent 0.80 p2
    CX_x     CX across patches                 (0, 4)   silent 0
    CZ_in    CZ inside a patch                 (1, 3)   silent 0.60 p2  -- (6, 0), 0 if transversal
    CZ_x     CZ across patches                 (0, 4)   silent 0
    CXT/CZT  two CXs / CZs between two patches, fused  (0, 4)  silent 0
    + ENCODE (1, 3) and DECODE (8, 8) per patch

This script rebuilds each arm's gadget census with the translator's own passes (cancel inverse
gates, commute Hadamards, the merging rewrite -- exactly as `translator.core` runs them) on the
four arms of Fig 6 under the standard (`experiments/figures/ablation_standard`: baseline, + H
commutation, + merging, + packing, each with its plan and commutation decision), prices it,
checks the priced 1q / 2q against the recorded translations, and measures each arm's
detected-error budget off the schedule and its silent error per kept shot by Pauli frames at the
circuit's Fig 6 alpha. Also records structural features of each logical program.

    PYTHONPATH=. venv/bin/python -m experiments.insights.census --jobs 12
Writes results/insights/census[_vN].csv: one row per (circuit, arm).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import collections
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/insights")
SOURCE = Path("experiments/figures/ablation_standard/ablation_standard.csv")
ARMS = ("baseline", "h_only", "h_merge", "packed")
CLASSES = ("H", "HH", "RZ", "RX", "S", "SX", "PAULI", "PAULI_FUSED", "CX_in", "CX_x",
           "CZ_in", "CZ_x", "XCX_in", "XCX_x", "CXT", "CZT", "XCXT", "SWAP", "OTHER")
ONE_QUBIT = {"h": "H", "rz": "RZ", "p": "RZ", "t": "RZ", "tdg": "RZ", "rx": "RX",
             "s": "S", "sdg": "S", "sx": "SX", "sxdg": "SX", "x": "PAULI", "y": "PAULI",
             "z": "PAULI"}
MACRO = {"hh": "HH", "xx": "PAULI_FUSED", "zz": "PAULI_FUSED", "cxt": "CXT", "czt": "CZT",
         "xcxt": "XCXT"}
#: census class -> (work-table key, rate-table gadget)
PRICE = {"H": ("H", "H_lane"), "HH": ("HH", "HH"), "RZ": ("RZ", "RZ"), "RX": ("RX", "RX"),
         "S": ("S", "S"), "SX": ("SX", "SX"), "PAULI": ("X", "X"), "PAULI_FUSED": ("XX", "X"),
         "CX_in": ("CX", "CX_intra"), "CX_x": ("CX", "CX_cross"), "CZ_in": ("CZ", "CZ_intra"),
         "CZ_x": ("CZ", "CZ_cross"), "XCX_in": ("XCX", "XCX_intra"),
         "XCX_x": ("XCX", "XCX_cross"), "CXT": ("CXT", "CXT"), "CZT": ("CZT", "CZT"),
         "XCXT": ("XCXT", "XCXT"), "SWAP": ("SWAP", "SWAP_intra")}


def processed(program, permutation, *, commute: bool, merge: bool, transversal: bool):
    """The circuit the gadget stage meets: `translator.core`'s passes, in its order."""
    from iceberg.gadget_options import transversal_intra
    from translator.config import TranslatorConfig
    from translator.merging import preprocess_program_for_merging_with_stats
    from translator.passes import cancel_inverse_gates, commute_hadamards

    swaps = TranslatorConfig().allow_transversal_swap_merges
    with transversal_intra(transversal):
        circuit = cancel_inverse_gates(program)
        if commute:
            circuit = cancel_inverse_gates(commute_hadamards(circuit))
        if merge:
            circuit, *_ = preprocess_program_for_merging_with_stats(
                circuit, [tuple(p) for p in permutation],
                allow_transversal_swap_merges=swaps, n_code=4)
    return circuit


def census(circuit, permutation) -> collections.Counter:
    """Gadget classes of `circuit` under `permutation` (logical qubit -> (patch, lane))."""
    out: collections.Counter = collections.Counter()
    for instruction in circuit.data:
        name = instruction.operation.name.lower()
        if name in ("barrier", "measure", "noise_begin", "noise_end"):
            continue
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if name in MACRO:
            out[MACRO[name]] += 1
        elif len(qubits) == 1:
            out[ONE_QUBIT.get(name, "OTHER")] += 1
        elif len(qubits) == 2 and name in ("cx", "cz", "xcx", "swap"):
            same = permutation[qubits[0]][0] == permutation[qubits[1]][0]
            if name == "swap":
                out["SWAP"] += 1
            else:
                out[f"{name.upper()}_{'in' if same else 'x'}"] += 1
        else:
            out["OTHER"] += 1
    return out


def price(counts, patches: int, *, transversal: bool) -> Dict[str, float]:
    """Priced (1q, 2q) and first-order detected / silent error of a census, at alpha = 1."""
    from iceberg.costs import CROSS_PATCH_COST, work_table
    from iceberg.gadget_options import transversal_intra
    from packing.silent import DATA

    table = pd.read_csv(DATA / ("gadget_rates_transversal.csv" if transversal
                                else "gadget_rates.csv"))
    rates = table[table["n_code"] == 4].set_index("gadget")
    with transversal_intra(transversal):
        work = work_table(4)
    one = two = detected = silent = 0.0
    for name, count in counts.items():
        if name not in PRICE:
            continue
        key, gadget = PRICE[name]
        cost = work[key]
        if name.endswith("_x") and key in CROSS_PATCH_COST:
            cost = CROSS_PATCH_COST[key]
        one += count * cost[0]
        two += count * cost[1]
        if gadget in rates.index:
            detected += count * float(rates.loc[gadget, "p_detected"])
            silent += count * float(rates.loc[gadget, "p_logical"])
    encode, decode = work["ENCODE"], work["DECODE"]
    one += patches * (encode[0] + decode[0])
    two += patches * (encode[1] + decode[1])
    return {"priced_one": one, "priced_two": two, "gadget_detected": detected,
            "gadget_silent": silent}


def features(program) -> Dict[str, float]:
    """Structure of the logical program, before any pass."""
    from qiskit.converters import circuit_to_dag

    from translator.passes import cancel_inverse_gates, commute_hadamards

    gates = [i for i in program.data if i.operation.name not in ("barrier", "measure")]
    names = collections.Counter(i.operation.name for i in gates)
    two = [tuple(sorted(program.find_bit(q).index for q in i.qubits))
           for i in gates if len(i.qubits) == 2]
    pairs = collections.Counter(two)
    degree = collections.Counter()
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    n = program.num_qubits
    total = max(1, len(gates))
    cleaned = cancel_inverse_gates(program)
    commuted = cancel_inverse_gates(commute_hadamards(cleaned))
    h_before = sum(1 for i in cleaned.data if i.operation.name == "h")
    h_after = sum(1 for i in commuted.data if i.operation.name == "h")
    depth = max(1, circuit_to_dag(program).depth())
    # How often a gate's neighbours in its layer are the same kind on another qubit:
    # parallel same-type layers are what the macros fuse.
    layers: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    dag = circuit_to_dag(program)
    for index, layer in enumerate(dag.layers()):
        for node in layer["graph"].op_nodes():
            if node.op.name not in ("barrier", "measure"):
                layers[index][node.op.name] += 1
    parallel_h = sum(c["h"] for c in layers.values() if c["h"] >= 2)
    parallel_cx = sum(c["cx"] + c["cz"] for c in layers.values() if c["cx"] + c["cz"] >= 2)
    return {
        "n": n, "gates": len(gates), "two_qubit": len(two),
        "logical_depth": depth,
        "frac_h": names["h"] / total, "frac_rotation": (names["rz"] + names["rx"]) / total,
        "frac_s": names["s"] / total, "frac_two_qubit": len(two) / total,
        "frac_cz_of_2q": names["cz"] / max(1, len(two)),
        "h_removable_by_commutation": (h_before - h_after) / max(1, h_before),
        "h_after_commutation_per_gate": h_after / total,
        "frac_h_in_parallel_layers": parallel_h / max(1, names["h"]),
        "frac_2q_in_parallel_layers": parallel_cx / max(1, len(two)),
        "distinct_pairs": len(pairs), "reuse_per_pair": len(two) / max(1, len(pairs)),
        "max_degree": max(degree.values()) if degree else 0,
        "mean_degree": (2 * len(pairs) / n) if n else 0.0,
        "frac_2q_nearest_neighbour": sum(c for (a, b), c in pairs.items() if b - a == 1)
        / max(1, len(two)),
        "frac_2q_in_naive_patch": sum(c for (a, b), c in pairs.items() if a // 2 == b // 2)
        / max(1, len(two)),
        "parallelism": len(gates) / (depth * max(1, n)),
    }


def _frames(encoded, alpha: float, seed: int) -> Dict[str, float]:
    from experiments.concatenation.noise import rescale_text
    from experiments.revision.exp8_undetected.frames import split_from_stim, stim_params
    from experiments.revision.exp8_undetected.gadgets import noisy_stim

    noisy = noisy_stim(encoded, stim_params(1.0), idle_time_scalar=1.0, distance_scalar=1.0)
    split = split_from_stim(encoded, rescale_text(str(noisy), alpha), shots=100_000, seed=seed)
    return {"frame_lsr": 1.0 - split["p_detected"],
            "silent_per_kept": split["p_silent_given_kept"], "p_silent": split["p_silent"]}


def _job(job) -> List[Dict]:
    import math as _math

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from packing.standard import STANDARD, standard_translator_config
    from simulator.attribution import measured_noise_split
    from simulator.noise_models import StimNoiseParameters
    from translator.core import Translator

    circuit, record, frames = job
    resource = fig6.ensure_measurements(circuit.original)
    proxy = fig6.ensure_measurements(circuit.proxy)
    n = resource.num_qubits
    tight = fig6.tight_trivial_packing(n, _math.ceil(n / 2))
    placement = [tuple(p) for p in json.loads(record["placement"])]
    commute = bool(record["commute"])
    arms = {"baseline": (tight, False, False), "h_only": (tight, commute, False),
            "h_merge": (tight, commute, True), "packed": (placement, commute, True)}
    base = {"circuit": circuit.name, "size_group": circuit.size_group, **features(resource),
            "alpha": float(record["alpha_final"]), "commute": commute}
    rows, measured = [], {}
    for arm, (perm, arm_commute, merge) in arms.items():
        transversal = merge and STANDARD.transversal_intra_two_qubit
        counts = census(processed(resource, perm, commute=arm_commute, merge=merge,
                                  transversal=transversal), perm)
        patches = len({p for p, _ in perm})
        row = {**base, "arm": arm, **{f"n_{c}": counts.get(c, 0) for c in CLASSES},
               **price(counts, patches, transversal=transversal),
               "recorded_one": float(record[f"{arm}_one_qubit_gates"]),
               "recorded_two": float(record[f"{arm}_two_qubit_gates"]),
               "recorded_depth": float(record[f"{arm}_depth"]),
               "lsr": float(record[f"{arm}_lsr"]), "tvd": float(record[f"{arm}_tvd"]),
               "baseline_tvd": float(record["baseline_tvd"])}
        key = json.dumps([[list(p) for p in perm], arm_commute, merge])
        if key not in measured:
            config = standard_translator_config(4, commute=arm_commute, merging=merge)
            if not merge:
                config.transversal_intra_two_qubit = False
            encoded = Translator(config=config).translate(proxy, list(perm)).circuit
            entry = {"E_detected": measured_noise_split(
                encoded, StimNoiseParameters.neutral_atom())["total"]}
            if frames:
                entry.update(_frames(encoded, base["alpha"], 31 + circuit.circuit_idx))
            measured[key] = entry
        rows.append({**row, **measured[key]})
    return rows


def main() -> None:
    from experiments.revision.common import load_fig6_circuits, write_sidecar
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--names", nargs="+", default=None)
    args = parser.parse_args()
    began = time.time()
    source = pd.read_csv(SOURCE).drop_duplicates("circuit_name").set_index("circuit_name")
    circuits = [c for c in load_fig6_circuits() if not args.names or c.name in args.names]
    jobs = [(c, source.loc[c.name].to_dict(), not args.no_frames) for c in circuits]
    jobs.sort(key=lambda j: -j[0].original.num_qubits)
    results = parallel_map(_job, jobs, jobs=args.jobs, label="census")
    failed = [j[0].name for j, r in zip(jobs, results) if r is None]
    if failed:
        print(f"[census] FAILED: {failed}")
    frame = pd.DataFrame([row for r in results if r for row in r])
    OUT.mkdir(parents=True, exist_ok=True)
    path, version = OUT / "census.csv", 2
    while path.exists():
        path, version = OUT / f"census_v{version}.csv", version + 1
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment="census", settings=vars(args), started=began, status="complete")
    gap_one = (frame["priced_one"] - frame["recorded_one"]).abs() / frame["recorded_one"]
    gap_two = (frame["priced_two"] - frame["recorded_two"]).abs() / frame["recorded_two"]
    print(f"[census] wrote {path}: {len(frame)} rows; priced vs recorded: 1q within "
          f"{gap_one.median():.2%} (median, max {gap_one.max():.1%}), 2q within "
          f"{gap_two.median():.2%} (max {gap_two.max():.1%})")


if __name__ == "__main__":
    main()
