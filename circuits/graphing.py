# graphing.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import cirq
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqale_sim.fidelity import bhattacharyya
from permutations_old.permutation_evaluator import (
    evaluate_permutations,
    simulate_with_ps,
)
from permutations_old.pipeline import enumerate_permutations, build_physical_circuit
from experiments_superstaq.simulation.sqale_leakage_model import (
    HilbertNoiseParams,
    SqaleLeakageModel,
    DEFAULT_PARAMS,
)

from sqale_sim.threshold_analysis import alpha_scale_noise_params
from cirq import NoiseModel

USE_CIRQ_NOISE = False

# Define the logical circuit
def logical_circuit() -> cirq.Circuit:
    qA, qB, qC = cirq.LineQubit.range(3)
    return cirq.Circuit(
        cirq.H(qA),
        cirq.CNOT(qA, qB),
        cirq.CNOT(qA, qC),
        cirq.H(qA),
    )

def build_cirq_noise_model(alpha: float) -> cirq.NoiseModel:
    """Return Cirq depolarizing noise model where all gate noise scales with a single alpha."""

    p = min(alpha, 1.0)  # Cirq depolarize() requires p ∈ [0, 1]

    class CustomNoiseModel(cirq.NoiseModel):
        def noisy_operation(self, operation):
            gate = operation.gate
            if isinstance(gate, cirq.Rz) or isinstance(gate, cirq.CZPowGate):
                return [operation, cirq.depolarize(p).on_each(*operation.qubits)]
            elif isinstance(gate, cirq.MeasurementGate):
                return [operation]  # No post-measurement noise
            return operation

    return CustomNoiseModel()

def simulate_physical_accuracy(
    circuit: cirq.Circuit,
    noise_model: cirq.NOISE_MODEL_LIKE,
    *,
    repetitions: int = 10_000,
    oversample: int = 1,
    ch_form: bool = False,
) -> float:
    """Simulate ideal vs noisy distribution of physical circuit and return Bhattacharyya fidelity."""
    import pandas as pd
    from cirq import DensityMatrixSimulator
    from sqale_sim.fidelity import bhattacharyya
    from permutations_old.permutation_evaluator import _ensure_measured

    stripped = _ensure_measured(circuit)

    if USE_CIRQ_NOISE:
        sim = DensityMatrixSimulator()

        # Ideal run (no noise)
        ideal_result = sim.run(stripped, repetitions=repetitions)
        ideal_keys = sorted(ideal_result.measurements.keys())
        ideal_bitstrings = ideal_result.data[ideal_keys].to_numpy().astype(int)
        ideal_strings = ["".join(str(b) for b in row) for row in ideal_bitstrings]
        ideal_dist = dict(pd.Series(ideal_strings).value_counts(normalize=True))

        # Noisy run
        noisy_circuit = stripped.with_noise(noise_model)
        noisy_result = sim.run(noisy_circuit, repetitions=repetitions)
        noisy_bitstrings = noisy_result.data[ideal_keys].to_numpy().astype(int)
        noisy_strings = ["".join(str(b) for b in row) for row in noisy_bitstrings]
        noisy_dist = dict(pd.Series(noisy_strings).value_counts(normalize=True))
    else:
        print("yo")
        ideal_dist, _, noisy_dist, *_ = simulate_with_ps(
            stripped,
            noise_model=noise_model,
            repetitions=repetitions,
            oversample=oversample,
            ch_form=ch_form,
            post_select=True,
            logical_post_select=False
        )  
        print(ideal_dist)
        print(noisy_dist)

    return float(bhattacharyya(ideal_dist, noisy_dist))

def generate_accuracy_vs_noise(
    logical: cirq.Circuit,
    perms: list[list[tuple[int, int]]],
    scalar_values: list[float],
    *,
    use_cirq_noise: bool = True,
    repetitions: int = 100,
    oversample: int = 100,
    ch_form: bool = False,
    n_code: int = 4,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    base_params = DEFAULT_PARAMS

    iceberg_acc = [[] for _ in perms]        # List[perm][scalar]
    postsel_rates = [[] for _ in perms]      # List[perm][scalar]
    physical_acc = []                        # List[scalar]

    for alpha in scalar_values:
        print(f"\n[α = {alpha:.2g}] Simulating...")

        if use_cirq_noise:
            noise_model = build_cirq_noise_model(alpha)
        else:
            scaled_params = alpha_scale_noise_params(base_params, alpha=alpha)
            noise_model = SqaleLeakageModel(**scaled_params.model_dump())

        # Run all permutations
        df = evaluate_permutations(
            permutations=perms,
            logical_circ=logical,
            build_fn=build_physical_circuit,
            n_code=n_code,
            faulty=True,
            noise_model=noise_model,
            repetitions=repetitions,
            oversample=oversample,
            ch_form=ch_form,
            post_select=True,
            compare_against="physical",
        )

        for i, row in enumerate(df.itertuples()):
            iceberg_acc[i].append(row.fidelity)
            postsel_rates[i].append(row.selection_rate_overall)

        # Run 1 physical sim per scalar
        physical_fid = simulate_physical_accuracy(
            logical,
            noise_model=noise_model,
            repetitions=repetitions,
            oversample=oversample,
            ch_form=ch_form,
        )
        physical_acc.append(physical_fid)

    return iceberg_acc, postsel_rates, physical_acc

def plot_accuracy_vs_noise(
    scalars: list[float],
    iceberg_acc: list[list[float]],
    postsel_rates: list[list[float]],
    physical_acc: list[float],
):
    plt.figure(figsize=(12, 6))

    # Postselection Rate
    for i, acc in enumerate(postsel_rates):
        plt.plot(scalars, acc, linestyle="--", marker='o', alpha=0.6, label=f"Postsel (perm {i})")

    # Iceberg Accuracy
    for i, acc in enumerate(iceberg_acc):
        plt.plot(scalars, acc, linestyle="-", marker='x', alpha=0.9, label=f"Iceberg (perm {i})")

    # Physical accuracy (1 line)
    plt.plot(scalars, physical_acc, color="black", linewidth=2, label="Physical Accuracy")

    plt.xscale("log")
    plt.ylim(0, 1.05)
    plt.xlabel("Noise Rate (rz = cz scalar)")
    plt.ylabel("Fidelity / Rate")
    plt.title("Accuracy and Postselection vs Noise (per permutation)")
    plt.legend(ncol=2, fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    logical = logical_circuit()
    code_sizes = [4,4,4]
    perms = enumerate_permutations(logical_circuit=logical, code_sizes=code_sizes)
    random.shuffle(perms)
    perms = perms[:10]
    for perm in perms:
        print(perm)
    scalars = np.logspace(-2, 1, 10)

    if USE_CIRQ_NOISE:
        print("Normalizing scalars for Cirq noise model")
        max_rz = max(scalars)
        scalars = np.clip(scalars / max_rz, 0.0, 1.0)

    print(f"Running {len(perms)} permutations over {len(scalars)} noise levels...")

    iceberg_acc, postsel_rates, physical_acc = generate_accuracy_vs_noise(
        logical=logical,
        perms=perms,
        scalar_values=scalars,
        use_cirq_noise=USE_CIRQ_NOISE,
    )

    plot_accuracy_vs_noise(scalars, iceberg_acc, postsel_rates, physical_acc)

if __name__ == "__main__":
    main()