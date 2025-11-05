"""
experiments.shared.random_qasm_dataset_qiskit.py – Generate and save random Qiskit circuits as QASM2.
"""

import os
import random
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.circuit.random import random_circuit, random_clifford_circuit

# ----------------------------------------------------------------------
# Helper: save a single random circuit to dataset as OpenQASM2
# ----------------------------------------------------------------------
def save_random_qasm_circuit(
    dataset_dir: str = "dataset",
    n_qubits: int = 10,
    depth: int | None = None,
    seed: int | None = None,
) -> str:
    """
    Creates a random Qiskit circuit, converts it to OpenQASM2,
    and saves it to the dataset folder.

    Returns:
        The path to the saved .qasm file.
    """
    rng = random.Random(seed)
    if depth is None:
        depth = rng.randint(10, 20)

    # Generate the circuit
    circuit: QuantumCircuit = random_clifford_circuit(num_qubits=n_qubits, num_gates=depth, seed=seed)
    # circuit: QuantumCircuit = random_clifford_circuit(num_qubits=n_qubits, num_gates=depth, seed=seed)

    # Convert to OpenQASM2
    qasm_str = qasm3_dumps(circuit)

    # Ensure dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Filename format: <num_qubits>_<gate_depth>_<seed>.qasm
    filename = f"{n_qubits}_{depth}_{seed}.qasm"
    filepath = os.path.join(dataset_dir, filename)

    # Write to file
    with open(filepath, "w") as f:
        f.write(qasm_str)

    return filepath


# ----------------------------------------------------------------------
# Dataset filler: generate multiple random circuits safely
# ----------------------------------------------------------------------
def fill_random_qasm_dataset(
    num_samples: int = 10,
    n_qubits: int = 10,
    depth_min: int = 10,
    depth_max: int = 20,
    dataset_dir: str = "dataset",
    master_seed: int | None = None,
) -> list[str]:
    """
    Generate `num_samples` random circuits and write them to dataset_dir as
    '<num_qubits>_<gate_depth>_<seed>.qasm'. Ensures no filename collisions.

    Returns:
        List of file paths created.
    """
    os.makedirs(dataset_dir, exist_ok=True)

    # Track existing (names without extension)
    existing_names = {
        os.path.splitext(f)[0]
        for f in os.listdir(dataset_dir)
        if f.endswith(".qasm")
    }

    rng = random.Random(master_seed)
    created_paths: list[str] = []

    while len(created_paths) < num_samples:
        depth = rng.randint(depth_min, depth_max)
        seed = rng.getrandbits(32)
        base_name = f"{n_qubits}_{depth}_{seed}"

        # Check for collisions
        if base_name in existing_names:
            continue

        # Use helper to generate and save
        filepath = save_random_qasm_circuit(
            dataset_dir=dataset_dir, n_qubits=n_qubits, depth=depth, seed=seed
        )

        existing_names.add(base_name)
        created_paths.append(filepath)

    return created_paths


# Example usage:
if __name__ == "__main__":
    paths = fill_random_qasm_dataset(10_000, dataset_dir="clifford_circuits")
    # print("\n".join(paths))
