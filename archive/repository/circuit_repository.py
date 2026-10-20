# repository/circuit_repository.py (Qiskit)

import os
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.qpy import dump as qpy_dump
from archive.repository.utils import safe_filename


class CircuitRepository:
    """
    Handles saving physical circuits to disk (Qiskit).
    Saves QASM 3 when possible; falls back to QPY.
    """

    def __init__(self, base_dir: str = "results/physical_circuits"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def make_subfolder(self, circuit: QuantumCircuit) -> str:
        name = getattr(circuit, "name", "unnamed")
        if not isinstance(name, str):
            name = str(name)
        name = name.strip().replace(" ", "_")
        subfolder = os.path.join(self.base_dir, safe_filename(name))
        os.makedirs(subfolder, exist_ok=True)
        return subfolder

    def save_circuit(self, circuit: QuantumCircuit, filename: str, subfolder: str):
        """
        If `filename` ends with .qasm or .qasm3, writes QASM 3 text.
        Otherwise writes QASM 3 to <filename>.qasm3.
        On failure, falls back to binary QPY at <filename>.qpy.
        """
        # Normalize extension & target path
        root, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        # Prefer QASM 3 text
        try:
            qasm_text = qasm3_dumps(circuit)
            if ext_lower not in {".qasm", ".qasm3"}:
                filename = root + ".qasm3"
            path = os.path.join(subfolder, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(qasm_text)
            return
        except Exception:
            # Fall back to QPY (binary, version-stable within Qiskit)
            qpy_path = os.path.join(subfolder, root + ".qpy")
            with open(qpy_path, "wb") as fb:
                qpy_dump(circuit, fb)
