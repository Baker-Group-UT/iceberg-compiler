from qiskit import QuantumCircuit
import random

def random_circuit(n: int, depth: int = 50, seed: int | None = None) -> QuantumCircuit:
    """
    Generate a random Qiskit QuantumCircuit with n qubits and given depth.
    Mix of CNOT, H, and X gates applied at random locations.
    Before measurement, the operation order is randomly shuffled.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(n, name=f"random_{n}q_depth{depth}")

    ops = []

    # Ensure every qubit is used at least once
    for q in range(n):
        gate = rng.choice(["h", "x"])
        if gate == "h":
            ops.append(("h", [q]))
        else:
            ops.append(("x", [q]))

    # Add random layers
    for _ in range(depth):
        gate_type = rng.choice(["cnot", "h", "x"])
        if gate_type == "cnot":
            q1, q2 = rng.sample(range(n), 2)
            ops.append(("cx", [q1, q2]))
        elif gate_type == "h":
            q = rng.choice(range(n))
            ops.append(("h", [q]))
        else:
            q = rng.choice(range(n))
            ops.append(("x", [q]))

    # Shuffle all operations before measurement
    rng.shuffle(ops)

    # Apply all operations
    for gate, qubits in ops:
        getattr(qc, gate)(*qubits)

    # Add measurements at the end
    qc.measure_all()

    return qc
