import cirq, random

def random_circuit(n: int, depth: int = 50, seed: int | None = None) -> cirq.Circuit:
    """
    Generate a random Cirq circuit with n qubits and given depth.
    Mix of CNOT, H, and X gates applied at random locations.
    """
    rng = random.Random(seed)
    qubits = [cirq.LineQubit(i) for i in range(n)]
    circuit = cirq.Circuit()

    # Ensure every qubit is used at least once
    for q in qubits:
        circuit.append(rng.choice([cirq.H(q), cirq.X(q)]))

    # Add random layers
    for _ in range(depth):
        gate_type = rng.choice(["cnot", "h", "x"])
        if gate_type == "cnot":
            q1, q2 = rng.sample(qubits, 2)
            circuit.append(cirq.CNOT(q1, q2))
        elif gate_type == "h":
            circuit.append(cirq.H(rng.choice(qubits)))
        else:
            circuit.append(cirq.X(rng.choice(qubits)))

    # add measurements
    for i in range(n):
        circuit.append(cirq.measure(qubits[i]))

    return circuit
