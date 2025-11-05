# tcalibrate.py
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.library import SetDensityMatrix  # Option A: explicit instruction

def slice_circuit(qc: QuantumCircuit, start: int, stop: int) -> QuantumCircuit:
    """
    Return a subcircuit containing qc.data[start:stop].
    Compatible with Qiskit 1.x (CircuitInstruction).
    """
    sub = QuantumCircuit(*qc.qregs, *qc.cregs)
    for ci in qc.data[start:stop]:
        sub.append(ci.operation, ci.qubits, ci.clbits)
    return sub

# --- Example circuit (replace with yours) ---
qc = QuantumCircuit(2, 2)
qc.h(0)                  # 0
qc.cx(0, 1)              # 1
qc.rx(0.25, 0)           # 2  <-- we'll make [1:4) noisy
qc.h(1)                  # 3
qc.cz(0, 1)              # 4  <-- back to ideal
qc.measure([0, 1], [0, 1])  # 5

# Indices for the NOISY segment [start, stop)
noisy_start, noisy_stop = 1, 4  # ops with indices 1,2,3 are noisy

# Split into three parts
qc_pre  = slice_circuit(qc, 0, noisy_start)
qc_mid  = slice_circuit(qc, noisy_start, noisy_stop)
qc_post = slice_circuit(qc, noisy_stop, len(qc.data))

# --- Noise model for the middle segment only ---
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h', 'rx'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cx', 'cz'])

# 1) PRE (ideal) — build the state directly (no run, no save)
if len(qc_pre) == 0:
    num_qubits = qc.num_qubits
    psi_pre = Statevector.from_label('0' * num_qubits)
else:
    psi_pre = Statevector.from_instruction(qc_pre)

# 2) MIDDLE (noisy) — Aer density_matrix, SAVE with a label, and read from result.data(0)
sim_mid = AerSimulator(method="density_matrix", noise_model=noise_model)
qc_mid_t = transpile(qc_mid, sim_mid, optimization_level=0)
qc_mid_t.save_density_matrix(label='rho_mid')  # <- explicit save with label

# Pass the ideal prefix state as the initial state
res_mid = sim_mid.run(qc_mid_t, initial_statevector=psi_pre).result()
rho_mid = res_mid.data(0)['rho_mid']  # <- retrieve by label

# 3) POST (ideal) — start from rho_mid using explicit SetDensityMatrix, then compose qc_post
sim_post = AerSimulator(method="density_matrix")  # no noise

qc_post_init = QuantumCircuit(qc.num_qubits, qc.num_clbits)
dm = DensityMatrix(rho_mid)  # ensure type is DensityMatrix
qc_post_init.append(SetDensityMatrix(dm), qc_post_init.qubits)
qc_post_init.compose(qc_post, inplace=True)

qc_post_t = transpile(qc_post_init, sim_post, optimization_level=0)
res_post = sim_post.run(qc_post_t).result()

# Final classical counts (requires measurement in qc_post or earlier)
counts = res_post.get_counts(qc_post_t)
print(counts)
