import cirq
import random

def random_circuit(n: int, depth: int = 50, seed: int | None = None) -> cirq.Circuit:
    """
    Generate a random Cirq circuit with n qubits and given depth.
    Mix of CNOT, H, and X gates applied at random locations.
    Before measurement, the operation order is randomly shuffled.
    """
    rng = random.Random(seed)
    qubits = [cirq.LineQubit(i) for i in range(n)]
    circuit = cirq.Circuit()

    # Collect operations
    ops = []

    # Ensure every qubit is used at least once
    for q in qubits:
        ops.append(rng.choice([cirq.H(q), cirq.X(q)]))

    # Add random layers
    for _ in range(depth):
        gate_type = rng.choice(["cnot", "h", "x"])
        if gate_type == "cnot":
            q1, q2 = rng.sample(qubits, 2)
            ops.append(cirq.CNOT(q1, q2))
        elif gate_type == "h":
            ops.append(cirq.H(rng.choice(qubits)))
        else:
            ops.append(cirq.X(rng.choice(qubits)))

    # Shuffle all operations before measurement
    rng.shuffle(ops)
    circuit.append(ops)

    # Add measurement at the end
    circuit.append(cirq.measure(*qubits))

    return circuit
