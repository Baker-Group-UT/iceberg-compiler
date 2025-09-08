# repositroy/circuit_repository.py

import os
import cirq
from repository.utils import safe_filename


class CircuitRepository:
    """
    Handles saving physical circuits to disk.
    """

    def __init__(self, base_dir: str = "results/physical_circuits"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def make_subfolder(self, circuit: cirq.Circuit) -> str:
        name = getattr(circuit, "name", "unnamed").strip().replace(" ", "_")
        subfolder = os.path.join(self.base_dir, safe_filename(name))
        os.makedirs(subfolder, exist_ok=True)
        return subfolder

    def save_circuit(self, circuit: cirq.Circuit, filename: str, subfolder: str):
        try:
            qasm = cirq.qasm(circuit)  # type: ignore[attr-defined]
            if isinstance(qasm, bytes):
                qasm = qasm.decode("utf-8")
            path = os.path.join(subfolder, filename)
            with open(path, "w") as f:
                f.write(qasm)
        except Exception:
            json_path = os.path.join(subfolder, filename.replace(".qasm", ".json"))
            with open(json_path, "w") as f:
                f.write(cirq.to_json(circuit))