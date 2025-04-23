import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from math import pi
import matplotlib.pyplot as plt
from circuits.IcebergCode import IcebergCode
import qiskit
import qiskit.circuit
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
from os.environ import get

service = css.Service(api_key=get("SUPERSTAQ_API"))
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
