import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager, qasm3
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error, NoiseModel
import argparse
from IcebergCode import IcebergCode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def gen_random_clifford_circ(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    circuit_inv = QuantumCircuit(num_qubits)

    single_qubit_gates = ["h", "rz", "z", "x"]
    two_qubit_gates = ["cx", "cz"]

    # Store all applied gates to create the inverse circuit
    gate_history = []

    for d in range(depth):
        for q in range(num_qubits):
            if np.random.random() < 0.7:
                gate = np.random.choice(single_qubit_gates)
                if gate == "h":
                    circuit.h(q)
                    gate_history.append(("h", q))
                elif gate == "s":
                    circuit.s(q)
                    gate_history.append(("s", q))
                elif gate == "rz":
                    angle = np.random.uniform(0, 2 * np.pi)
                    circuit.rz(angle, q)
                    gate_history.append(("rz", q, angle))
                elif gate == "z":
                    circuit.z(q)
                    gate_history.append(("z", q))
                elif gate == "x":
                    circuit.x(q)
                    gate_history.append(("x", q))

        for q in range(num_qubits - 1):
            if np.random.random() < 0.3:
                control = q
                target = q + 1
                gate = np.random.choice(two_qubit_gates)
                if gate == "cx":
                    circuit.cx(control, target)
                    gate_history.append(("cx", control, target))
                elif gate == "cz":
                    circuit.cz(control, target)
                    gate_history.append(("cz", control, target))

    # Inverse circuit by applying inverse gates in reverse order
    for gate_info in reversed(gate_history):
        if gate_info[0] == "h":
            circuit_inv.h(gate_info[1])
        elif gate_info[0] == "s":
            circuit_inv.sdg(gate_info[1])
        elif gate_info[0] == "rz":
            angle = gate_info[2]
            circuit_inv.rz(-angle, gate_info[1])
        elif gate_info[0] == "z":
            circuit_inv.z(gate_info[1])
        elif gate_info[0] == "x":
            circuit_inv.x(gate_info[1])
        elif gate_info[0] == "cx":
            circuit_inv.cx(gate_info[1], gate_info[2])
        elif gate_info[0] == "cz":
            circuit_inv.cz(gate_info[1], gate_info[2])

    return circuit, circuit_inv


def get_qubit_index(qubits, index=0):
    return int(str(qubits[index]).split("_")[1])


def generate_iceberg_circuit(num_qubits, circuit):
    circuit_iceberg = circuit_from_qasm(qasm3.dumps(circuit))
    iceberg = IcebergCode(num_qubits + 2, cirq.Circuit())
    for moment in circuit_iceberg:
        for gate in moment:
            if gate.gate == cirq.X:
                iceberg.X(get_qubit_index(gate.qubits))
            # elif gate.gate == cirq.Z:
            #     iceberg.Z(get_qubit_index(gate.qubits))
            # elif gate.gate == cirq.S:
            #     iceberg.S(get_qubit_index(gate.qubits))
            elif gate.gate == cirq.H:
                iceberg.H(get_qubit_index(gate.qubits))
            elif gate.gate == cirq.CNOT:
                iceberg.CNOT(
                    get_qubit_index(gate.qubits), get_qubit_index(gate.qubits, 1)
                )
            # elif gate.gate == cirq.CZ:
            #     iceberg.CZ(
            #         get_qubit_index(gate.qubits), get_qubit_index(gate.qubits, 1)
            #     )
            # elif isinstance(gate.gate, cirq.Rz):
            #     iceberg.RZ(get_qubit_index(gate.qubits), float(gate.gate._rads))
    iceberg.decode()
    iceberg.measure()

    return QuantumCircuit.from_qasm_str(iceberg.circuit.to_qasm())


def generate_noise_model(alpha, p_reset=0.03, p_meas=0.03, p_gate=0.001):
    p_reset = alpha * p_reset
    p_meas = alpha * p_meas
    p_gate = alpha * p_gate
    error_reset = pauli_error(
        [("I", 1 - p_reset), ("X", p_reset / 3), ("Z", p_reset / 3), ("Y", p_reset / 3)]
    )
    error_meas = pauli_error(
        [("I", 1 - p_meas), ("X", p_meas / 3), ("Z", p_meas / 3), ("Y", p_meas / 3)]
    )
    error_gate1 = pauli_error(
        [("I", 1 - p_gate), ("X", p_gate / 3), ("Z", p_gate / 3), ("Y", p_gate / 3)]
    )
    error_gate2 = pauli_error(
        [
            ("I", 1 - 5 * p_gate),
            ("X", 5 * p_gate / 3),
            ("Z", 5 * p_gate / 3),
            ("Y", 5 * p_gate / 3),
        ]
    )
    error_gate2 = error_gate2.tensor(error_gate2)

    noise_pauli_channel = NoiseModel()
    noise_pauli_channel.add_all_qubit_quantum_error(error_reset, "reset")
    noise_pauli_channel.add_all_qubit_quantum_error(error_meas, "measure")
    noise_pauli_channel.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_pauli_channel.add_all_qubit_quantum_error(error_gate2, ["cx", "cz"])

    return noise_pauli_channel


def main(qubits, depth, output, simulate, alpha):
    circuit_orig, circuit_inv = gen_random_clifford_circ(qubits, depth)
    # print(circuit_orig)
    # print(circuit_inv)

    ## Check if the optimized circuit is equivalent to the original circuit
    circuit = QuantumCircuit(qubits)
    circuit.compose(circuit_orig, inplace=True)
    circuit.compose(circuit_inv, inplace=True)
    assert Operator(circuit).equiv(
        Operator.from_label("I" * len(circuit.qubits))
    )  # True

    # Generate Iceberg circuit
    iceberg_circuit = generate_iceberg_circuit(qubits, circuit)

    # Draw circuit
    if output is not None:
        output_filename = "circuits/" + (
            output + ".png" if output else "circuit_physical.png"
        )
        circuit.draw(output="mpl", filename=output_filename)

        # Draw Iceberg circuit
        iceberg_output_filename = "circuits/" + (
            output + "_iceberg.png" if output else "circuit_iceberg.png"
        )
        iceberg_circuit.draw(output="mpl", filename=iceberg_output_filename)

        # Export as qasm file
        qasm_filename = "circuits/" + (output + ".qasm" if output else "circuit.qasm")
        qasm_string = qasm3.dumps(circuit)
        with open(qasm_filename, "w") as file:
            file.write(qasm_string)

    # Simulate the circuit if needed
    if simulate:
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()

        # Simulate the circuit
        sim_noise = AerSimulator(noise_model=generate_noise_model(alpha))
        pass_manager = generate_preset_pass_manager(
            optimization_level=0,
            backend=sim_noise,
        )
        job_physical = sim_noise.run(pass_manager.run(meas_circuit))
        counts = job_physical.result().get_counts()
        if output is not None:
            plot_histogram(counts, filename="circuits/histogram_physical.png")
        success_physical = counts["0" * qubits] / sum(counts.values())

        job_iceberg = sim_noise.run(pass_manager.run(iceberg_circuit))
        counts = job_iceberg.result().get_counts()
        if output is not None:
            plot_histogram(
                counts, filename="circuits/histogram_iceberg.png", figsize=(30, 10)
            )
        success_iceberg = (
            counts.get("0" * (qubits + 2) + " 0 0", 0)
            + counts.get("1" * (qubits + 2) + " 0 0", 0)
        ) / sum(
            value if key.split(" ")[1:].count("0") == 2 else 0
            for (key, value) in counts.items()
        )
        return success_physical, success_iceberg
    return None


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Generate a random quantum circuit.")
    # parser.add_argument("--qubits", "-q", type=int, default=6)
    # parser.add_argument("--depth", "-d", type=int, default=30)
    # parser.add_argument("--alpha", "-a", type=float, default=1)
    # parser.add_argument("--output", "-o", type=str)
    # parser.add_argument("--simulate", "-s", action="store_true")

    # print("Generating a random Clifford circuit...")
    # args = parser.parse_args()
    # print(main(args.qubits, args.depth, args.output, args.simulate, args.alpha))

    alphas = np.logspace(-2, 1, 15)
    depths = np.linspace(0, 25, 10)
    qubits = 6
    num_shots = 10
    # generate plots for each alpha varying depth
    for alpha in alphas:
        success_physical = []
        success_iceberg = []
        for depth in depths:
            success_physical.append([])
            success_iceberg.append([])
            for i in range(num_shots):
                ret = main(qubits, int(depth), None, True, alpha)
                if ret is not None:
                    success_physical[-1].append(ret[0])
                    success_iceberg[-1].append(ret[1])

        plt.figure(figsize=(10, 5))

        labels = []

        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        add_label(
            plt.violinplot(
                success_physical,
                positions=depths,
                showmeans=True,
                showmedians=True,
            ),
            "Physical Circuit",
        )
        add_label(
            plt.violinplot(
                success_iceberg,
                positions=depths,
                showmeans=True,
                showmedians=True,
            ),
            "Iceberg Circuit",
        )
        plt.xticks(
            ticks=depths,
            labels=[str(int(depth)) for depth in depths],
        )
        plt.xlabel("Depth")
        plt.ylabel("Success Probability")
        plt.title(f"Success Probability vs Depth (alpha={alpha})")
        plt.legend(
            *zip(*labels),
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.savefig(
            f"circuits/success_prob_depth_alpha_{alpha:.2f}.png",
            bbox_inches="tight",
        )
