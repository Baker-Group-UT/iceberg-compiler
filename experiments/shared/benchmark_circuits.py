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


#: Gates a loaded program may contain once transpiled; anything else and the file is refused.
#: `sx` only ever comes from the logical optimisation pass (the transpile basis has none).
_PROGRAM_GATES = {"h", "x", "z", "s", "sx", "cx", "cz", "rz", "rx", "barrier"}


def program_from_transpiled(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit | None:
    """
    What the experiments call the original: the transpiled circuit with every measurement
    dropped (mid-circuit ones included) and rotations kept. None if any other gate remains.
    """
    program = qiskit.QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        name = inst.operation.name.lower()
        if name == "measure":
            continue
        if name not in _PROGRAM_GATES:
            return None
        program.append(inst.operation, [program.qubits[qc.find_bit(q).index] for q in inst.qubits])
    return program


def rotations_as_s(program: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """The Clifford proxy rule: every rz / rx becomes s, everything else is kept, in order."""
    proxy = qiskit.QuantumCircuit(program.num_qubits)
    for inst in program.data:
        qargs = [proxy.qubits[program.find_bit(q).index] for q in inst.qubits]
        if inst.operation.name in {"rz", "rx"}:
            proxy.s(qargs[0])
        else:
            proxy.append(inst.operation, qargs)
    return proxy


def finish_loaded(qc: qiskit.QuantumCircuit, replace_rz_rx_with_s: bool) -> qiskit.QuantumCircuit | None:
    """
    A transpiled QASM circuit -> the program (or its proxy) every experiment receives.

    The program (`program_from_transpiled`) gets one verified round of logical optimisation
    (`translator.logical_optimization`; nothing when AQORN_LOGICAL_OPT=0) *before* its proxy is
    made, so the proxy has the optimised program's gates with its rotations as S. Whichever is
    returned is registered as already optimised, so the translator and the planner use it as it
    is -- the proxy is never simplified into a structure the program does not have.
    """
    from translator.logical_optimization import logically_optimized, register_optimized

    program = program_from_transpiled(qc)
    if program is None:
        return None
    program = logically_optimized(program)
    out = rotations_as_s(program) if replace_rz_rx_with_s else program
    register_optimized(out)
    return out


def create_proxy_from_qasm(
    path: Path,
    timeout_seconds: float | None = 1.0,
    *,
    replace_rz_rx_with_s: bool = True,
) -> qiskit.QuantumCircuit | None:
    """
    Load an OpenQASM file and create a Clifford proxy circuit.
    Steps: transpile to a basis that includes RZ, drop measurements, one verified round of
    logical optimisation (`finish_loaded`; skipped when AQORN_LOGICAL_OPT=0), and approximate
    non-Clifford RZ as S. With `replace_rz_rx_with_s=False` the (optimised) program itself.
    Returns the circuit or None on failure.
    """
    try:
        with _timeout(timeout_seconds):
            qc = qiskit.QuantumCircuit.from_qasm_file(str(path))
            basis = ["h", "x", "z", "cx", "cz", "s", "rz", "rx"]
            qc = qiskit.transpile(qc, basis_gates=basis, optimization_level=1)
    except Exception:
        return None
    return finish_loaded(qc, replace_rz_rx_with_s)


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

    # Sorted: `rglob` follows the filesystem's directory order (hash order on Linux ext4),
    # and callers keep the last file per circuit name. small/basis_trotter_n4/ holds two
    # programs under that name; sorted, the real one (basis_trotter_n4.qasm) comes last, as
    # it did on the Mac that made Fig 6. Unsorted, Linux loaded the 92-gate basis_test_n4.
    results = sorted(
        path for path in Path(root).rglob("*.qasm")
        if "_transpiled" not in path.name
    )
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
    *,
    replace_rz_rx_with_s: bool = True,
) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    circuits: List[Tuple[str, qiskit.QuantumCircuit]] = []
    for path in tqdm(
        find_qasm_files(root, gate_limit=gate_limit, qubit_limit=qubit_limit),
        desc="proxy circuits",
        leave=False,
    ):
        proxy = create_proxy_from_qasm(
            path,
            timeout_seconds=timeout_seconds,
            replace_rz_rx_with_s=replace_rz_rx_with_s,
        )
        if proxy is not None:
            name = _read_circuit_name(path.parent / "README.md") or path.stem
            circuits.append((name, proxy))
    return circuits
