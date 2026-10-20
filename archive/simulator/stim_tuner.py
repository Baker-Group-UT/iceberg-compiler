"""
stim_tuner.py
-------------
Utility script for empirically tuning the Stim backend noise parameters so that
its behavior more closely matches a reference Qiskit Aer noise model.

The tuner samples random circuits, simulates them with both Aer (reference) and
Stim (candidate) backends, and searches for scalar multipliers that are applied
to the Stim noise parameters (T1, T2, gate times, depolarizing probabilities,
and readout error). The objective is to maximize the average Bhattacharyya
fidelity between the two simulated distributions.

Usage example:

    python stim_tuner.py --circuits 6 --depth 8 --samples 30 --shots 4000

Optionally pass ``--output best_scaling.json`` to dump the best scaling factors
for later reuse (e.g., wiring them into SimulatorConfig.with_stim_scaling()).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import qiskit
from qiskit.circuit.random import random_clifford_circuit

from simulator.config import SimulatorConfig
from simulator.noise_models import (
    DefaultNoiseModel,
    StimParameterScaling,
)
from simulator.simulator import Simulator
from simulator.utils import bhattacharyya


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample from log-uniform distribution over [low, high]."""
    if low <= 0 or high <= 0:
        raise ValueError("log-uniform bounds must be positive.")
    log_low = math.log10(low)
    log_high = math.log10(high)
    return 10 ** rng.uniform(log_low, log_high)


def random_scaling(
    rng: random.Random,
    *,
    t_min: float = 0.1,
    t_max: float = 10.0,
    p_min: float = 0.1,
    p_max: float = 10.0,
) -> StimParameterScaling:
    """
    Create a random scaling candidate.
    Gate-time scalars share the same range (t_min, t_max) while error-related
    scalars (depolarizing + readout) use (p_min, p_max).
    """
    return StimParameterScaling(
        T1=_log_uniform(rng, t_min, t_max),
        T2=_log_uniform(rng, t_min, t_max),
        one_qubit_gate_time=_log_uniform(rng, t_min, t_max),
        two_qubit_gate_time=_log_uniform(rng, t_min, t_max),
        one_qubit_error=_log_uniform(rng, p_min, p_max),
        two_qubit_error=_log_uniform(rng, p_min, p_max),
        meas_time=_log_uniform(rng, t_min, t_max),
        readout_error=_log_uniform(rng, p_min, p_max),
    )


def generate_circuits(
    count: int,
    *,
    min_qubits: int,
    max_qubits: int,
    depth: int,
    seed: int,
    basis: Sequence[str],
) -> list[qiskit.QuantumCircuit]:
    circuits: list[qiskit.QuantumCircuit] = []
    rng = random.Random(seed)
    for idx in range(count):
        q = rng.randint(min_qubits, max_qubits)
        circ = random_clifford_circuit(q, depth, seed=seed + idx, gates=basis)
        circ.measure_all()
        # transpiled = qiskit.transpile(
        #     circ,
        #     basis_gates=list(basis),
        #     optimization_level=0,
        # )
        circuits.append(circ)
    return circuits


def average_fidelity(
    reference: Simulator,
    candidate: Simulator,
    circuits: Iterable[qiskit.QuantumCircuit],
) -> float:
    fidelities: list[float] = []
    for circ in circuits:
        ref_res = reference.noisy(circ)
        cand_res = candidate.noisy(circ)
        fidelities.append(bhattacharyya(ref_res.noisy_dist, cand_res.noisy_dist))
    return float(np.mean(fidelities)) if fidelities else float("nan")


def evaluate_scaling(
    scaling: StimParameterScaling,
    circuits: Sequence[qiskit.QuantumCircuit],
    *,
    noise_model,
    repetitions: int,
    seed: int,
) -> float:
    reference = Simulator(
        SimulatorConfig(
            repetitions=repetitions,
            noise_model=noise_model,
            seed=seed,
            stim=False,
        )
    )
    candidate = Simulator(
        SimulatorConfig(
            repetitions=repetitions,
            noise_model=noise_model,
            seed=seed,
            stim=True,
            stim_scaling=scaling,
        )
    )
    return average_fidelity(reference, candidate, circuits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stim noise tuning helper.")
    parser.add_argument("--circuits", type=int, default=4, help="number of random circuits")
    parser.add_argument("--depth", type=int, default=6, help="random Clifford depth")
    parser.add_argument("--min-qubits", type=int, default=3, help="minimum qubits")
    parser.add_argument("--max-qubits", type=int, default=5, help="maximum qubits")
    parser.add_argument("--shots", type=int, default=2000, help="shots per simulation")
    parser.add_argument("--samples", type=int, default=20, help="number of random scaling candidates")
    parser.add_argument("--seed", type=int, default=7, help="base seed for reproducibility")
    parser.add_argument("--output", type=Path, help="optional JSON file to write best scaling factors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    noise_model = DefaultNoiseModel()
    circuits = generate_circuits(
        args.circuits,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        depth=args.depth,
        seed=args.seed,
        basis=("x", "z", "h", "s", "cx", "cz"),
    )

    identity_scaling = StimParameterScaling()
    best_scaling = identity_scaling
    best_score = evaluate_scaling(
        identity_scaling,
        circuits,
        noise_model=noise_model,
        repetitions=args.shots,
        seed=args.seed,
    )
    print(f"Baseline (identity scaling) avg fidelity: {best_score:.6f}")

    for idx in range(args.samples):
        scaling = random_scaling(rng)
        score = evaluate_scaling(
            scaling,
            circuits,
            noise_model=noise_model,
            repetitions=args.shots,
            seed=args.seed + idx + 1,
        )
        print(
            f"[{idx + 1:02d}/{args.samples}] fidelity={score:.6f} "
            f"scaling={asdict(scaling)}"
        )
        if score > best_score:
            best_score = score
            best_scaling = scaling

    print("\nBest scaling found:")
    for k, v in asdict(best_scaling).items():
        print(f"  {k}: {v:.6f}")
    print(f"Average Bhattacharyya fidelity vs Aer: {best_score:.6f}")

    if args.output:
        data = {
            "scaling": asdict(best_scaling),
            "average_fidelity": best_score,
            "metadata": {
                "circuits": args.circuits,
                "depth": args.depth,
                "shots": args.shots,
                "samples": args.samples,
                "seed": args.seed,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2))
        print(f"Wrote best scaling to {args.output}")


if __name__ == "__main__":
    main()
