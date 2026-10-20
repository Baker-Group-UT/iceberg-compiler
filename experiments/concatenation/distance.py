"""
distance.py -- which faults get past both levels: exact, from Stim's detector error model.

Test programs have deterministic outputs (a random 4-qubit Clifford followed by its inverse,
re-synthesised so no pass can cancel the two halves), which makes every output bit a valid Stim
observable. Every check of the post-selection decoder -- both syndrome bits and the data parity of
each inner patch, and of each outer patch expressed through the inner-decoded bits -- becomes a
DETECTOR. Then, from the detector error model:

    first-order silent   DEM error mechanisms that flip an observable and no detector: single
                         faults that post-selection keeps and that corrupt the output. Their total
                         probability is the silent error to first order in the physical rates.
    distance             `search_for_undetectable_logical_errors`: the fewest fault mechanisms that
                         together flip an observable while firing no detector (circuit distance,
                         counting DEM mechanisms).

Arms: single level (passes off / on), and the two-level circuit with the inner placement stride
(spread) or index (not spread), inner passes off / on, outer level naive with passes off.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.distance
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

NAME = "concat_distance"


def deterministic_program(num_qubits: int, seed: int, depth: int = 12):
    """A random Clifford U, then U^-1 synthesised afresh: output |0...0>, nothing cancels."""
    import random

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford
    from qiskit.synthesis import synth_clifford_full

    rng = random.Random(seed)
    forward = QuantumCircuit(num_qubits)
    for _ in range(depth):
        kind = rng.random()
        if kind < 0.4:
            a, b = rng.sample(range(num_qubits), 2)
            forward.cx(a, b)
        else:
            getattr(forward, rng.choice(["h", "s", "sx"]))(rng.randrange(num_qubits))
    inverse = synth_clifford_full(Clifford(forward).adjoint())
    program = forward.compose(inverse)
    from qiskit import transpile

    return transpile(program, basis_gates=["h", "s", "sdg", "sx", "x", "z", "cx", "cz"],
                     optimization_level=0)


def _program_gates(program):
    """Map sdg -> s s s (the translator's gate set has no sdg)."""
    from qiskit import QuantumCircuit

    out = QuantumCircuit(program.num_qubits)
    for instruction in program.data:
        name = instruction.operation.name
        qubits = [program.find_bit(q).index for q in instruction.qubits]
        if name == "sdg":
            for _ in range(3):
                out.s(qubits[0])
        else:
            out.append(instruction.operation, qubits)
    return out


def annotated(noisy, check_sets: Sequence[Sequence[int]], output_sets: Sequence[Sequence[int]]):
    import stim

    total = noisy.num_measurements
    out = noisy.copy()
    for records in check_sets:
        out.append("DETECTOR", [stim.target_rec(r - total) for r in sorted(records)])
    for index, records in enumerate(output_sets):
        out.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - total) for r in sorted(records)],
                   index)
    return out


def _xor(*sets) -> Tuple[int, ...]:
    acc: Dict[int, int] = {}
    for records in sets:
        for r in records:
            acc[r] = acc.get(r, 0) ^ 1
    return tuple(sorted(r for r, v in acc.items() if v))


def two_level_sets(maps):
    checks = [records for _, records in maps.inner_checks]
    for patch, data in maps.outer_data.items():
        for q in maps.outer_parity[patch]:
            checks.append(maps.flat_value[q])
        checks.append(_xor(*[maps.flat_value[q] for q in data]))
    outputs = []
    for patch, (a, b) in maps.outputs:
        data = maps.outer_data[patch]
        outputs.append(_xor(maps.flat_value[data[a]], maps.flat_value[data[b]]))
    return checks, outputs


def single_sets(maps):
    return list(maps.checks), [tuple(sorted(pair)) for pair in maps.outputs]


def analyse(noisy_annotated, search: bool = True, budget: int = 4) -> Dict[str, float]:
    dem = noisy_annotated.detector_error_model(decompose_errors=False,
                                               approximate_disjoint_errors=True,
                                               ignore_decomposition_failures=True)
    first_count, first_probability, total_probability = 0, 0.0, 0.0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        targets = instruction.targets_copy()
        detectors = [t for t in targets if t.is_relative_detector_id()]
        observables = [t for t in targets if t.is_logical_observable_id()]
        total_probability += p
        if observables and not detectors:
            first_count += 1
            first_probability += p
    out = {"dem_errors": float(dem.num_errors), "first_order_silent_mechanisms": float(first_count),
           "first_order_silent_probability": first_probability,
           "dem_total_probability": total_probability}
    if search:
        began = time.time()
        errors = noisy_annotated.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=budget,
            dont_explore_edges_with_degree_above=budget,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )
        out["distance"] = float(len(errors))
        out["search_seconds"] = time.time() - began
    return out


def run_point(seed: int, arm: str, alpha: float = 0.1) -> Dict[str, object]:
    from experiments.concatenation import construct as C
    from experiments.concatenation import decode as D
    from experiments.tvd_explore.fastframes import at_alpha_one, rescale

    program = _program_gates(deterministic_program(4, seed))
    placement = [(i // 2, i % 2) for i in range(4)]
    if arm.startswith("single"):
        passes = arm == "single_passes"
        result = C.outer_translate(program, placement, commute=passes, merging=passes)
        maps = D.single_maps_for(result)
        checks, outputs = single_sets(maps)
        circuit = result.circuit
    else:
        layout = "index" if "index" in arm else "stride"
        passes = arm.endswith("passes")
        two = C.compile_two_level(program, placement, outer_commute=False, outer_merging=False,
                                  inner_layout=layout, inner_commute=passes,
                                  inner_merging=passes, require_spread=layout != "index")
        maps = D.maps_for(two)
        checks, outputs = two_level_sets(maps)
        circuit = two.circuit
    noisy = rescale(at_alpha_one(circuit), alpha)
    result = analyse(annotated(noisy, checks, outputs))
    return {"seed": seed, "arm": arm, "alpha": alpha, "logical_gates": len(program.data),
            "physical_qubits": circuit.num_qubits,
            "two_qubit_gates": sum(1 for i in circuit.data if i.operation.name in ("cx", "cz")),
            **result}


ARMS = ("single", "single_passes", "concat_index", "concat_index_passes", "concat_stride",
        "concat_stride_passes")


def _job(args):
    seed, arm, alpha = args
    return run_point(seed, arm, alpha)


def main() -> None:
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", default="results/concatenation")
    args = parser.parse_args()
    jobs = [(seed, arm, args.alpha) for seed in range(args.seeds) for arm in ARMS]
    rows = [r for r in parallel_map(_job, jobs, jobs=args.jobs, label=NAME) if r]
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}.csv"
    frame.to_csv(path, index=False)
    print(frame.groupby("arm")[["two_qubit_gates", "first_order_silent_mechanisms",
                                "first_order_silent_probability", "distance"]].mean().round(5))
    print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
