import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import argparse
from qiskit.providers.fake_provider import GenericBackendV2


def gen_random_clifford_circ(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    circuit_inv = QuantumCircuit(num_qubits)

    single_qubit_gates = ["h", "s", "z", "x"]
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
        elif gate_info[0] == "z":
            circuit_inv.z(gate_info[1])
        elif gate_info[0] == "x":
            circuit_inv.x(gate_info[1])
        elif gate_info[0] == "cx":
            circuit_inv.cx(gate_info[1], gate_info[2])
        elif gate_info[0] == "cz":
            circuit_inv.cz(gate_info[1], gate_info[2])

    return circuit, circuit_inv


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random Clifford quantum circuit."
    )
    parser.add_argument("--qubits", "-q", type=int, default=5)
    parser.add_argument("--depth", "-d", type=int, default=20)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--simulate", "-s", action="store_true")

    print("Generating a random Clifford circuit...")
    args = parser.parse_args()

    circuit_orig, circuit_inv = gen_random_clifford_circ(args.qubits, args.depth)
    # print(circuit_orig)
    # print(circuit_inv)

    ## Check if the optimized circuit is equivalent to the original circuit
    circuit = QuantumCircuit(args.qubits)
    circuit.compose(circuit_orig, inplace=True)
    circuit.compose(circuit_inv, inplace=True)
    assert Operator(circuit).equiv(
        Operator.from_label("I" * len(circuit.qubits))
    )  # True

    # Draw circuit
    output_filename = "random_circuit_simulation/" + (
        args.output + ".png" if args.output else "circuit.png"
    )
    circuit.draw(output="mpl", filename=output_filename)

    # Export as qasm file
    qasm_filename = "random_circuit_simulation/" + (
        args.output + ".qasm" if args.output else "circuit.qasm"
    )
    qasm_string = qasm3.dumps(circuit)
    with open(qasm_filename, "w") as file:
        file.write(qasm_string)

    # Simulate the circuit if needed
    if args.simulate:
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()

        # Simulate the circuit
        AerSimulator = Aer.get_backend("qasm_simulator")
        circ = transpile(meas_circuit, AerSimulator)
        job = AerSimulator.run(circ)
        result = job.result()
        counts = result.get_counts(meas_circuit)

        # trying fake simulators
        GenericBackendV2_backend = GenericBackendV2(args.qubits)
        # FakeWashingtonV2_simulator = AerSimulator.from_backend(FakeWashingtonV2_backend)
        # circ = transpile(meas_circuit, FakeWashingtonV2_simulator, shots=1024)
        # job = AerSimulator.run(circ)
        # result = job.result()
        # counts = result.get_counts(meas_circuit)

        transpiled_circuit = transpile(meas_circuit, GenericBackendV2_backend)
        transpiled_circuit.draw("mpl")
        job = GenericBackendV2_backend.run(transpiled_circuit)
        counts = job.result().get_counts()
        plot_histogram(counts, filename="random_circuit_simulation/histogram.png")

        print("\nSimulation Results:")
        print(counts)


if __name__ == "__main__":
    main()

