import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from math import pi
import matplotlib.pyplot as plt
from circuits.IcebergCode import IcebergCode, IcebergGrid
import qiskit
from qiskit.qasm2 import dumps
# %%

qubits = cirq.LineQubit.range(7)
qubits
# %%

circuit_2 = cirq.Circuit()

circuit_2.append(cirq.X(qubits[6]))

# circuit.append(cirq.H(qubits[0]))
# circuit.append(cirq.ControlledGate(cirq.IdentityGate(4))(qubits[0], *qubits[3:7]))
# circuit.append(cirq.H(qubits[0]))

circuit_2.append(cirq.H(qubits[1]))
circuit_2.append(cirq.CNOT(qubits[1], qubits[4]))
circuit_2.append(cirq.CNOT(qubits[1], qubits[6]))
# circuit.append(cirq.Rz(rads=pi / 2).controlled(1)(qubits[0], qubits[1]))
circuit_2.append(cirq.H(qubits[1]))

circuit_2.append(cirq.H(qubits[2]))
circuit_2.append(cirq.CSWAP(qubits[2], qubits[3], qubits[4]))
circuit_2.append(cirq.CSWAP(qubits[2], qubits[4], qubits[5]))
circuit_2.append(cirq.CSWAP(qubits[2], qubits[5], qubits[6]))
# circuit.append(cirq.Rz(rads=pi / 2).controlled(1)(qubits[0], qubits[2]))
circuit_2.append(cirq.Rz(rads=pi / 4).controlled(1)(qubits[1], qubits[2]))
circuit_2.append(cirq.H(qubits[2]))

circuit_2.append(cirq.measure(*qubits[1:3], key="result"))

circuit_2
# %%

simulator = cirq.Simulator()
samples = simulator.run(circuit_2, repetitions=10000)
cirq.plot_state_histogram(samples, plt.subplot())
plt.show()
# %%

circuit_2_compiled = cirq.optimize_for_target_gateset(
    circuit_2,
    gateset=cirq.CZTargetGateset(
        preserve_moment_structure=False, reorder_operations=False
    ),
    max_num_passes=None,
)
# %%

print(len(circuit_2))
print(len(circuit_2_compiled))
circuit_2_qiskit = qiskit.QuantumCircuit.from_qasm_str(circuit_2.to_qasm())
circuit_2_qiskit = qiskit.transpile(
    circuit_2_qiskit,
    optimization_level=0,
    basis_gates=["x", "z", "h", "cz", "rz", "rzz"],
)
circuit_2_qiskit.draw()
circuit_2_compiled = circuit_from_qasm(dumps(circuit_2_qiskit))
circuit_2_compiled
# %%

iceberg = IcebergCode(8, cirq.Circuit())


def get_qubit_index(qubits, index=0):
    return int(str(qubits[index]).split("_")[1])


for moment in circuit_2_compiled:
    for gate in moment:
        if gate.gate == cirq.X:
            iceberg.X(get_qubit_index(gate.qubits))
        elif gate.gate == cirq.Z:
            iceberg.Z(get_qubit_index(gate.qubits))
        elif gate.gate == cirq.H:
            iceberg.H(get_qubit_index(gate.qubits))
        elif gate.gate == cirq.CZ:
            iceberg.CZ(get_qubit_index(gate.qubits), get_qubit_index(gate.qubits, 1))
        elif isinstance(gate.gate, cirq.Rz):
            iceberg.RZ(get_qubit_index(gate.qubits), float(gate.gate._rads))

len(iceberg.circuit)
# %%

iceberg_circuit_qiskit = qiskit.QuantumCircuit.from_qasm_str(iceberg.circuit.to_qasm())
iceberg_circuit_qiskit = qiskit.transpile(
    iceberg_circuit_qiskit,
    optimization_level=3,
    basis_gates=["h", "rz", "cz"],
)
iceberg_circuit_qiskit.draw(output="text")
# %%

# stats
from collections import defaultdict

counts = defaultdict(int)
for inst in iceberg_circuit_qiskit.data:
    counts[inst.operation.name] += 1
print(counts)
# %%

iceberg_circuit_cztarget = qiskit.QuantumCircuit.from_qasm_str(
    cirq.optimize_for_target_gateset(
        iceberg.circuit,
        gateset=cirq.CZTargetGateset(
            preserve_moment_structure=False, reorder_operations=False
        ),
        max_num_passes=None,
    ).to_qasm()
)
iceberg_circuit_cztarget.draw(output="text")
# %%

counts = defaultdict(int)
for inst in iceberg_circuit_cztarget.data:
    counts[inst.operation.name] += 1
print(counts)
# %%

import cirq_superstaq as css
from os import environ

service = css.Service(api_key=environ.get("SUPERSTAQ_API"))
# %%

iceberg_superstaq = service.cq_compile(iceberg.circuit)
# %%

iceberg_superstaq_qasm = iceberg_superstaq.circuit.to_qasm()
counts = defaultdict(int)
counts["GR"] = iceberg_superstaq_qasm.count("GR")
counts["CZ"] = iceberg_superstaq_qasm.count("cz")
counts["Rz"] = iceberg_superstaq_qasm.count("rz")
counts
# %%

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

hadamard_test_iceberg.circuit
# %%

hadamard_test_iceberg_compiled = service.cq_compile(hadamard_test_iceberg.circuit)
hadamard_test_iceberg_compiled
# %%

from circuits.random_circuit_gen import generate_noise_model
from qiskit_aer import AerSimulator
from qiskit import generate_preset_pass_manager, QuantumCircuit
from collections import defaultdict
# %%

physical_hadamard_test = QuantumCircuit(4)
physical_hadamard_test.h(0)
physical_hadamard_test.h(3)
physical_hadamard_test.cx(0, 2)
physical_hadamard_test.cx(1, 3)
physical_hadamard_test.cx(0, 1)
physical_hadamard_test.h(0)
physical_hadamard_test.h(1)
physical_hadamard_test.measure_all()
# %%


def compute_distance(counts: dict[str, int]):
    distance = 0
    shots = sum(counts.values())
    num_bits = len(list(counts.keys())[0])
    for v in counts.values():
        distance += abs(v - shots / (2**num_bits))
    return distance / shots / 2


def temp(alpha):
    NUM_SHOTS = 4096
    sim_noise = AerSimulator(noise_model=generate_noise_model(alpha))
    pass_manager = generate_preset_pass_manager(
        optimization_level=0,
        backend=sim_noise,
    )
    hadamard_test_iceberg_qiskit = QuantumCircuit.from_qasm_str(
        hadamard_test_iceberg.circuit.to_qasm()
    )
    job = sim_noise.run(pass_manager.run(hadamard_test_iceberg_qiskit), shots=NUM_SHOTS)
    iceberg_counts = job.result().get_counts()
    hadamard_test_iceberg_qiskit.draw(output="mpl")

    postselected_counts = dict()
    valid_counts = defaultdict(int)
    for k, v in iceberg_counts.items():
        k = k.split(" ")
        k_new = ""
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
            k_new += s
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
        postselected_counts[k_new] = v
        if successful:
            valid_counts[k_valid] += v

    postselection_rate = sum(valid_counts.values()) / sum(iceberg_counts.values())

    job_physical = sim_noise.run(
        pass_manager.run(physical_hadamard_test), shots=NUM_SHOTS
    )
    physical_counts = job_physical.result().get_counts()

    return (
        postselection_rate,
        compute_distance(valid_counts),
        compute_distance(physical_counts),
        valid_counts,
        iceberg_counts,
        physical_counts,
    )


# %%

import numpy as np
import matplotlib.pyplot as plt
# %%

alphas = np.logspace(-2, 1, 15)
num_shots = 10
postselection_rates, iceberg_distances, physical_distances = [], [], []
for alpha in alphas:
    postselection_rate, iceberg_distance, physical_distance = 0, 0, 0
    for _ in range(num_shots):
        psr, iced, phyd, _, _, _ = temp(alpha)
        postselection_rate += psr
        iceberg_distance += iced
        physical_distance += phyd
    postselection_rates.append(postselection_rate / num_shots)
    iceberg_distances.append(1 - iceberg_distance / num_shots)
    physical_distances.append(1 - physical_distance / num_shots)
# %%

# plot the results
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

_, _, _, valid_counts, _, _ = temp(1)
valid_counts_keys = sorted(valid_counts.keys())
valid_counts_values = [valid_counts[k] for k in valid_counts_keys]

plt.figure(figsize=(10, 6))
plt.bar(
    valid_counts_keys,
    valid_counts_values,
    tick_label=valid_counts_keys,
)
plt.axhline(
    sum(valid_counts.values()) / (2 ** len(list(valid_counts.keys())[0])),
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
