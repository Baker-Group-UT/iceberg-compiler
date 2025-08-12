# %%
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from math import pi
import matplotlib.pyplot as plt
from circuits.IcebergCode import CircuitClass, IcebergCode, IcebergGrid
from qiskit.qasm2 import dumps
from collections import defaultdict
from circuits.random_circuit_gen import generate_noise_model
from qiskit_aer import AerSimulator
from qiskit import generate_preset_pass_manager, QuantumCircuit, transpile
import numpy as np
import cirq_superstaq as css
from os import environ

if __name__ == "__main__":
    service = css.Service(api_key=environ.get("SUPERSTAQ_API"))
    # %%

    """
    This cell implements the circuit for the generalized Shor's algorithm.
    We have N = 15 and a = 2.
    """

    qubits = cirq.LineQubit.range(7)

    generalized_shors_circuit = cirq.Circuit()

    generalized_shors_circuit.append(cirq.X(qubits[6]))

    generalized_shors_circuit.append(cirq.H(qubits[1]))
    generalized_shors_circuit.append(cirq.CNOT(qubits[1], qubits[4]))
    generalized_shors_circuit.append(cirq.CNOT(qubits[1], qubits[6]))
    generalized_shors_circuit.append(cirq.H(qubits[1]))

    generalized_shors_circuit.append(cirq.H(qubits[2]))
    generalized_shors_circuit.append(cirq.CSWAP(qubits[2], qubits[3], qubits[4]))
    generalized_shors_circuit.append(cirq.CSWAP(qubits[2], qubits[4], qubits[5]))
    generalized_shors_circuit.append(cirq.CSWAP(qubits[2], qubits[5], qubits[6]))
    generalized_shors_circuit.append(
        cirq.Rz(rads=pi / 4).controlled(1)(qubits[1], qubits[2])
    )
    generalized_shors_circuit.append(cirq.H(qubits[2]))

    generalized_shors_circuit.append(cirq.measure(*qubits[1:3], key="result"))

    simulator = cirq.Simulator()
    samples = simulator.run(generalized_shors_circuit, repetitions=10000)
    cirq.plot_state_histogram(samples, plt.subplot())
    plt.show()
    # %%

    """
    Compiled version of the circuit defined in the previous cell.
    """

    generalized_shors_circuit_compiled = cirq.optimize_for_target_gateset(
        generalized_shors_circuit,
        gateset=cirq.CZTargetGateset(
            preserve_moment_structure=False, reorder_operations=False
        ),
        max_num_passes=None,
    )

    print(len(generalized_shors_circuit))
    print(len(generalized_shors_circuit_compiled))
    circuit_2_qiskit = QuantumCircuit.from_qasm_str(generalized_shors_circuit.to_qasm())
    circuit_2_qiskit = transpile(
        circuit_2_qiskit,
        optimization_level=0,
        basis_gates=["x", "z", "h", "cz", "rz", "rzz"],
    )
    circuit_2_qiskit.draw()
    generalized_shors_circuit_compiled = circuit_from_qasm(dumps(circuit_2_qiskit))
    print(generalized_shors_circuit_compiled)
    # %%

    """
    Implements the generalized Shor's algorithm in the Iceberg code.
    """

    generalized_shors_iceberg = IcebergCode(8, cirq.Circuit())


    def get_qubit_index(qubits, index=0):
        return int(str(qubits[index]).split("_")[1])


    for moment in generalized_shors_circuit_compiled:
        for gate in moment:
            if gate.gate == cirq.X:
                generalized_shors_iceberg.X(get_qubit_index(gate.qubits))
            elif gate.gate == cirq.Z:
                generalized_shors_iceberg.Z(get_qubit_index(gate.qubits))
            elif gate.gate == cirq.H:
                generalized_shors_iceberg.H(get_qubit_index(gate.qubits))
            elif gate.gate == cirq.CZ:
                generalized_shors_iceberg.CZ(
                    get_qubit_index(gate.qubits), get_qubit_index(gate.qubits, 1)
                )
            elif isinstance(gate.gate, cirq.Rz):
                generalized_shors_iceberg.RZ(
                    get_qubit_index(gate.qubits), float(gate.gate._rads)
                )

    len(generalized_shors_iceberg.circuit)
    # %%

    """
    Transpiles the Iceberg version using Qiskit, Cirq and Superstaq.
    """

    generalized_shors_iceberg_qiskit = QuantumCircuit.from_qasm_str(
        generalized_shors_iceberg.circuit.to_qasm()
    )
    generalized_shors_iceberg_qiskit = transpile(
        generalized_shors_iceberg_qiskit,
        optimization_level=3,
        basis_gates=["h", "rz", "cz"],
    )
    print(generalized_shors_iceberg_qiskit.draw(output="text"))

    counts = defaultdict(int)
    for inst in generalized_shors_iceberg_qiskit.data:
        counts[inst.operation.name] += 1
    print(counts)

    iceberg_circuit_cztarget = QuantumCircuit.from_qasm_str(
        cirq.optimize_for_target_gateset(
            generalized_shors_iceberg.circuit,
            gateset=cirq.CZTargetGateset(
                preserve_moment_structure=False, reorder_operations=False
            ),
            max_num_passes=None,
        ).to_qasm()
    )
    print(iceberg_circuit_cztarget.draw(output="text"))

    counts = defaultdict(int)
    for inst in iceberg_circuit_cztarget.data:
        counts[inst.operation.name] += 1
    print(counts)

    iceberg_superstaq = service.cq_compile(generalized_shors_iceberg.circuit)

    iceberg_superstaq_qasm = iceberg_superstaq.circuit.to_qasm()
    counts = defaultdict(int)
    counts["GR"] = iceberg_superstaq_qasm.count("GR")
    counts["CZ"] = iceberg_superstaq_qasm.count("cz")
    counts["Rz"] = iceberg_superstaq_qasm.count("rz")
    print(counts)
    # %%

    """
    Implements v1 of Hadamard test circuit (both iceberg and physical).
    The expected output is a uniform distribution on the 4 data qubits.
    """

    hadamard_test_iceberg = IcebergGrid(4, 4, cirq.Circuit(), "0+0+")
    hadamard_test_iceberg.circuit.append(
        cirq.H(hadamard_test_iceberg.icebergs[0].data_qubits[0])
    )
    hadamard_test_iceberg.circuit.append(
        cirq.H(hadamard_test_iceberg.icebergs[0].data_qubits[2])
    )
    hadamard_test_iceberg.circuit.append(
        cirq.CNOT(
            hadamard_test_iceberg.icebergs[0].data_qubits[0],
            hadamard_test_iceberg.icebergs[0].data_qubits[1],
        )
    )
    hadamard_test_iceberg.circuit.append(
        cirq.CNOT(
            hadamard_test_iceberg.icebergs[0].data_qubits[2],
            hadamard_test_iceberg.icebergs[0].data_qubits[3],
        )
    )

    hadamard_test_iceberg.circuit.append(
        cirq.H(hadamard_test_iceberg.icebergs[1].data_qubits[0])
    )
    hadamard_test_iceberg.circuit.append(
        cirq.H(hadamard_test_iceberg.icebergs[1].data_qubits[2])
    )
    hadamard_test_iceberg.circuit.append(
        cirq.CNOT(
            hadamard_test_iceberg.icebergs[1].data_qubits[0],
            hadamard_test_iceberg.icebergs[1].data_qubits[1],
        )
    )
    hadamard_test_iceberg.circuit.append(
        cirq.CNOT(
            hadamard_test_iceberg.icebergs[1].data_qubits[2],
            hadamard_test_iceberg.icebergs[1].data_qubits[3],
        )
    )

    (
        hadamard_test_iceberg.icebergs[0].data_qubits[1],
        hadamard_test_iceberg.icebergs[0].data_qubits[2],
    ) = (
        hadamard_test_iceberg.icebergs[0].data_qubits[2],
        hadamard_test_iceberg.icebergs[0].data_qubits[1],
    )

    hadamard_test_iceberg.CNOT_k_Transversal(0, 1)

    hadamard_test_iceberg.CNOT(0, 1)

    hadamard_test_iceberg.H(range(2))

    hadamard_test_iceberg.decode()
    hadamard_test_iceberg.measure()

    hadamard_test_physical = QuantumCircuit(4)
    hadamard_test_physical.h(0)
    hadamard_test_physical.h(3)
    hadamard_test_physical.cx(0, 2)
    hadamard_test_physical.cx(1, 3)
    hadamard_test_physical.cx(0, 1)
    hadamard_test_physical.h(0)
    hadamard_test_physical.h(1)
    hadamard_test_physical.measure_all()
    # %%

    """
    Helper functions to:
        1. compute the distance between the simulated distribution and the expected distribution
        2. generate the counts (distributions) for the iceberg and physical circuits
        3. run the iceberg and physical circuits for different noise rates
    """


    def compute_distance(
        counts: dict[str, int], expected_distribution: dict[str, float] | None = None
    ):
        num_bits = len(list(counts.keys())[0])
        if expected_distribution is None:
            expected_distribution = dict()
            for k in counts.keys():
                expected_distribution[k] = 1 / (2**num_bits)
        distance = 0
        shots = sum(counts.values())
        for k, v in counts.items():
            distance += abs(v / shots - expected_distribution[k])
        return distance / 2


    def generate_counts(
        iceberg_circuit: CircuitClass,
        physical_circuit: QuantumCircuit,
        alpha: float = 1.0,
        expected_distribution: dict[str, float] | None = None,
    ):
        NUM_SHOTS = 4096
        sim_noise = AerSimulator(noise_model=generate_noise_model(alpha))
        pass_manager = generate_preset_pass_manager(
            optimization_level=0,
            backend=sim_noise,
        )
        hadamard_test_iceberg_qiskit = QuantumCircuit.from_qasm_str(
            iceberg_circuit.circuit.to_qasm()
        )
        job = sim_noise.run(pass_manager.run(hadamard_test_iceberg_qiskit), shots=NUM_SHOTS)
        iceberg_counts = job.result().get_counts()
        hadamard_test_iceberg_qiskit.draw(output="mpl")

        postselected_counts = defaultdict(int)
        for k, v in iceberg_counts.items():
            k = k.split(" ")
            k_valid = ""
            valid = True
            successful = True
            for s in k:
                if len(s) == 1:
                    if s != "0":
                        valid = False
                        break
                    else:
                        continue
                if s.count("0") % 2 == 1:
                    successful = False
                s = int(s, 2)
                if s == 0 or s == 15:
                    k_valid += "00"
                elif s == 3 or s == 12:
                    k_valid += "01"
                elif s == 5 or s == 10:
                    k_valid += "10"
                elif s == 6 or s == 9:
                    k_valid += "11"
            if not valid:
                continue
            if successful:
                postselected_counts[k_valid[::-1]] += v

        postselection_rate = sum(postselected_counts.values()) / sum(
            iceberg_counts.values()
        )

        job_physical = sim_noise.run(pass_manager.run(physical_circuit), shots=NUM_SHOTS)
        physical_counts = job_physical.result().get_counts()

        return (
            postselection_rate,
            compute_distance(postselected_counts, expected_distribution),
            compute_distance(physical_counts, expected_distribution),
            postselected_counts,
            iceberg_counts,
            physical_counts,
        )


    def run_iceberg_and_physical(
        iceberg_circuit: CircuitClass,
        physical_circuit: QuantumCircuit,
        expected_distribution: dict[str, float] | None = None,
    ):
        alphas = np.logspace(-2, 1, 15)
        num_shots = 10
        postselection_rates, iceberg_distances, physical_distances = [], [], []
        for alpha in alphas:
            postselection_rate, iceberg_distance, physical_distance = 0, 0, 0
            for _ in range(num_shots):
                psr, iced, phyd, _, _, _ = generate_counts(
                    iceberg_circuit, physical_circuit, alpha, expected_distribution
                )
                postselection_rate += psr
                iceberg_distance += iced
                physical_distance += phyd
            postselection_rates.append(postselection_rate / num_shots)
            iceberg_distances.append(1 - iceberg_distance / num_shots)
            physical_distances.append(1 - physical_distance / num_shots)

        return alphas, postselection_rates, iceberg_distances, physical_distances


    # %%

    """
    Generate the plots for v1 Hadamard test circuit.
    """

    alphas, postselection_rates, iceberg_distances, physical_distances = (
        run_iceberg_and_physical(hadamard_test_iceberg, hadamard_test_physical)
    )
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        alphas,
        postselection_rates,
        marker="o",
        label="Postselection Rate",
    )
    plt.semilogx(
        alphas,
        iceberg_distances,
        marker="o",
        label="Iceberg Accuracy",
    )
    plt.semilogx(
        alphas,
        physical_distances,
        marker="o",
        label="Physical Accuracy",
    )
    plt.xlabel("Noise Rate (alpha)")
    plt.ylabel("Percentage")
    plt.title("Accuracy vs Noise Rate")
    plt.legend()
    plt.show()
    # %%

    """
    Plot the output distribution for v1 and see how far away it is from the uniform distribution.
    """

    _, _, _, postselected_counts, _, _ = generate_counts(
        hadamard_test_iceberg, hadamard_test_physical
    )
    postselected_counts_keys = sorted(postselected_counts.keys())
    postselected_counts_values = [postselected_counts[k] for k in postselected_counts_keys]

    plt.figure(figsize=(10, 6))
    plt.bar(
        postselected_counts_keys,
        postselected_counts_values,
        tick_label=postselected_counts_keys,
    )
    plt.axhline(
        sum(postselected_counts.values()) / (2 ** len(list(postselected_counts.keys())[0])),
        color="red",
        linestyle="--",
        label="Postselection Rate",
    )
    plt.xlabel("Postselected Counts")
    plt.ylabel("Counts")
    plt.title("Postselected Counts vs Counts")
    plt.legend()
    plt.show()
    # %%

    """
    Implements v2 of Hadamard test circuit (both iceberg and physical).
    """

    hadamard_test_v2_iceberg = IcebergGrid(4, 4, cirq.Circuit(), "000+")

    hadamard_test_v2_iceberg.circuit.append(
        cirq.H(hadamard_test_v2_iceberg.icebergs[1].data_qubits[0])
    )
    hadamard_test_v2_iceberg.circuit.append(
        cirq.H(hadamard_test_v2_iceberg.icebergs[1].data_qubits[2])
    )
    hadamard_test_v2_iceberg.circuit.append(
        cirq.CNOT(
            hadamard_test_v2_iceberg.icebergs[1].data_qubits[0],
            hadamard_test_v2_iceberg.icebergs[1].data_qubits[1],
        )
    )
    hadamard_test_v2_iceberg.circuit.append(
        cirq.CNOT(
            hadamard_test_v2_iceberg.icebergs[1].data_qubits[2],
            hadamard_test_v2_iceberg.icebergs[1].data_qubits[3],
        )
    )

    hadamard_test_v2_iceberg.H(range(2))

    hadamard_test_v2_iceberg.CZ(0, 1)

    hadamard_test_v2_iceberg.CNOT_k_Transversal(0, 1)

    hadamard_test_v2_iceberg.H(range(2))

    hadamard_test_v2_iceberg.decode()
    hadamard_test_v2_iceberg.measure()


    hadamard_test_v2_physical = QuantumCircuit(4)
    hadamard_test_v2_physical.h(0)
    hadamard_test_v2_physical.h(1)
    hadamard_test_v2_physical.h(3)
    hadamard_test_v2_physical.cz(0, 1)
    hadamard_test_v2_physical.cx(0, 2)
    hadamard_test_v2_physical.cx(1, 3)
    hadamard_test_v2_physical.h(0)
    hadamard_test_v2_physical.h(1)
    hadamard_test_v2_physical.measure_all()
    # %%

    """
    Plot the output distribution for v2 and see how far away it is from the expected distribution.
    The expected distribution is uniform over x??x, where the middle two bits are the same.
    """

    hadamard_test_v2_expected_distribution = defaultdict(float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                hadamard_test_v2_expected_distribution[
                    str(i) + str(j) + str(j) + str(k)
                ] = 1 / 8

    alphas, postselection_rates, iceberg_distances, physical_distances = (
        run_iceberg_and_physical(
            hadamard_test_v2_iceberg,
            hadamard_test_v2_physical,
            hadamard_test_v2_expected_distribution,
        )
    )
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        alphas,
        postselection_rates,
        marker="o",
        label="Postselection Rate",
    )
    plt.semilogx(
        alphas,
        iceberg_distances,
        marker="o",
        label="Iceberg Accuracy",
    )
    plt.semilogx(
        alphas,
        physical_distances,
        marker="o",
        label="Physical Accuracy",
    )
    plt.xlabel("Noise Rate (alpha)")
    plt.ylabel("Percentage")
    plt.title("Accuracy vs Noise Rate")
    plt.legend()
    plt.show()
    # %%
