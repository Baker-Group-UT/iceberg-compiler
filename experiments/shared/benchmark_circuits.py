"""
Shared helpers for loading and proxying benchmark circuits.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import qiskit
from tqdm import tqdm
import signal


@contextmanager
def _timeout(seconds: float | None):
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise TimeoutError("Timeout must be positive.")

    def _handle_timeout(_signum, _frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _read_circuit_name(readme_path: Path) -> str | None:
    if not readme_path.exists():
        return None
    try:
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("# application"):
                _, value = line.split(":", 1)
                return value.strip()
    except Exception:
        return None
    return None


def create_proxy_from_qasm(path: Path, timeout_seconds: float | None = 1.0) -> qiskit.QuantumCircuit | None:
    """
    Load an OpenQASM file and create a Clifford proxy circuit.
    Steps: transpile to a basis that includes RZ, drop measurements,
    and approximate non-Clifford RZ as S.
    Returns the proxy circuit or None on failure.
    """
    try:
        with _timeout(timeout_seconds):
            qc = qiskit.QuantumCircuit.from_qasm_file(str(path))
            qc = qiskit.transpile(
                qc,
                basis_gates=["h", "x", "z", "cx", "cz", "s", "rz"],
                optimization_level=1,
            )
    except Exception:
        return None

    proxy = qiskit.QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        op = inst.operation if hasattr(inst, "operation") else inst[0]
        qargs = inst.qubits if hasattr(inst, "qubits") else inst[1]
        name = op.name.lower()

        if name == "measure":
            continue
        mapped_qargs = [proxy.qubits[qc.find_bit(q).index] for q in qargs]
        if name in {"rz"}:
            proxy.s(mapped_qargs[0])
            continue
        if name in {"h", "x", "z", "s", "cx", "cz", "barrier"}:
            proxy.append(op, mapped_qargs)
            continue
        return None

    return proxy


def find_qasm_files(root: Path, gate_limit: int | None = None, qubit_limit: int | None = None) -> List[Path]:
    """
    Return all .qasm files under root, excluding any with "_transpiled" in the name.
    If gate_limit is set, only include files whose folder README reports
    "Gate Count" <= gate_limit.
    """
    def read_gate_count(readme_path: Path) -> int | None:
        try:
            for line in readme_path.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("- gate count"):
                    _, value = line.split(":", 1)
                    return int(value.strip())
        except Exception:
            return None
        return None
    
    def read_qubit_count(readme_path: Path) -> int | None:
        try:
            for line in readme_path.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("- qubit count"):
                    _, value = line.split(":", 1)
                    return int(value.strip())
        except Exception:
            return None
        return None

    results = [
        path for path in Path(root).rglob("*.qasm")
        if "_transpiled" not in path.name
    ]
    if gate_limit is None:
        gate_filtered = results
    else:
        gate_filtered = []
        for path in results:
            readme_path = path.parent / "README.md"
            gate_count = read_gate_count(readme_path) if readme_path.exists() else None
            if gate_count is None or gate_count <= gate_limit:
                gate_filtered.append(path)

    if qubit_limit is None:
        return gate_filtered

    filtered: List[Path] = []
    for path in gate_filtered:
        readme_path = path.parent / "README.md"
        qubit_count = read_qubit_count(readme_path) if readme_path.exists() else None
        if qubit_count is None or qubit_count <= qubit_limit:
            filtered.append(path)
    return filtered


def load_proxy_circuits(
    root: Path,
    gate_limit: int | None = None,
    qubit_limit: int | None = None,
    timeout_seconds: float | None = 1.0,
) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    circuits: List[Tuple[str, qiskit.QuantumCircuit]] = []
    for path in tqdm(
        find_qasm_files(root, gate_limit=gate_limit, qubit_limit=qubit_limit),
        desc="proxy circuits",
        leave=False,
    ):
        proxy = create_proxy_from_qasm(path, timeout_seconds=timeout_seconds)
        if proxy is not None:
            name = _read_circuit_name(path.parent / "README.md") or path.stem
            circuits.append((name, proxy))
    return circuits
