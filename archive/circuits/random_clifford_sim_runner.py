"""Batch simulator for OpenQASM random Clifford circuits.

This script scans a directory of OpenQASM files (defaulting to
``circuit_set/random_clifford_circuits`` if it exists, otherwise
``circuit_set/clifford_circuits``), converts each circuit to a Qiskit
``QuantumCircuit``, and simulates it with and without injected noise.

For every processed circuit, the script writes results under
``circuit_sims/<circuit_id>/``:

* ``original_openqasm.qasm`` – the raw OpenQASM text that was ingested.
* ``original_dag.json`` – serialized qubit-timeline DAG of the source circuit.
* ``noiseless_dag.json`` and ``noiseless_counts.json`` – DAG snapshot plus shot
  distribution for the measured circuit.
* ``variant_<n>_<gate><qubit>_dag.json`` / ``..._counts.json`` – DAGs and
  distributions for noisy variants with random single-qubit Pauli noise.

Example:
    python -m circuits.random_clifford_sim_runner --limit 50 --variants 3
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.qasm3 import loads
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_aer import Aer
from networkx.readwrite import json_graph

from archive.dag.circuit_dag import build_qubit_graph

try:
    import qiskit_qasm3_import  # noqa: F401  # Ensures OpenQASM 3 loader is registered.
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise MissingOptionalLibraryError(
        libname="qiskit-qasm3-import",
        name="OpenQASM 3 loader",
        pip_install="pip install qiskit-qasm3-import",
        msg=(
            "The qiskit-qasm3-import package is required to load OpenQASM 3 circuits. "
            "Install it in your environment before running this script."
        ),
    ) from exc

DEFAULT_CANDIDATE_DIRS: Sequence[Path] = (
    Path("circuit_set/random_clifford_circuits"),
    Path("circuit_set/clifford_circuits"),
)

AER_SIMULATOR = Aer.get_backend("aer_simulator")


@dataclass(frozen=True)
class NoiseSpec:
    """Describes the single-qubit noise gate inserted into a circuit."""

    gate: str
    qubit: int


def _resolve_circuit_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Circuit directory {explicit} does not exist.")
        return explicit

    for candidate in DEFAULT_CANDIDATE_DIRS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not determine a circuit directory. "
        "Please specify one with --circuits-dir."
    )


def _collect_qasm_files(circuit_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(circuit_dir.glob("*.qasm"))
    if limit is None or limit <= 0 or limit >= len(files):
        return files
    return files[:limit]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _write_dag_json(path: Path, circuit: QuantumCircuit) -> None:
    graph = build_qubit_graph(circuit, decompose=False)
    _write_json(path, json_graph.node_link_data(graph, edges="links"))


def _prepare_for_simulation(circuit: QuantumCircuit, max_depth: int = 4) -> QuantumCircuit:
    """Inline custom gate definitions to keep transpile stable."""
    prepared = circuit.copy()
    for _ in range(max_depth):
        if all(inst.operation.definition is None for inst in prepared.data):
            break
        prepared = prepared.decompose()
    return prepared


def _apply_noise(circuit: QuantumCircuit, noise: NoiseSpec | None) -> QuantumCircuit:
    noisy = circuit.copy()
    if noise is not None:
        gate_method = getattr(noisy, noise.gate, None)
        if gate_method is None:
            raise ValueError(f"Circuit does not support gate '{noise.gate}'.")
        gate_method(noise.qubit)
    noisy.measure_all()
    return noisy


def _simulate(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    compiled = transpile(circuit, backend=AER_SIMULATOR)
    job = AER_SIMULATOR.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return {str(bitstring): int(count) for bitstring, count in counts.items()}


def _build_noise_specs(
    rng: random.Random, num_qubits: int, num_variants: int, gate_choices: Sequence[str]
) -> Iterable[NoiseSpec]:
    for idx in range(num_variants):
        gate = rng.choice(gate_choices)
        target = rng.randrange(num_qubits)
        yield NoiseSpec(gate=gate, qubit=target)


def process_circuits(
    circuit_dir: Path,
    output_dir: Path,
    limit: int,
    shots: int,
    num_variants: int,
    noise_gates: Sequence[str],
    seed: int,
) -> None:
    rng = random.Random(seed)
    qasm_files = _collect_qasm_files(circuit_dir, limit)
    if not qasm_files:
        raise RuntimeError(f"No .qasm files found in {circuit_dir}.")

    print(
        f"Processing {len(qasm_files)} circuits from {circuit_dir} "
        f"with {shots} shots (variants per circuit: {num_variants})."
    )

    for idx, qasm_path in enumerate(qasm_files, start=1):
        circuit_id = qasm_path.stem
        dest_dir = output_dir / circuit_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(qasm_files)}] {circuit_id}")

        qasm_source = qasm_path.read_text()
        logical_circuit = loads(qasm_source)
        sim_base_circuit = _prepare_for_simulation(logical_circuit)

        _write_text(dest_dir / "original_openqasm.qasm", qasm_source)
        _write_dag_json(dest_dir / "original_dag.json", logical_circuit)
        _write_json(
            dest_dir / "metadata.json",
            {
                "source_file": str(qasm_path),
                "shots": shots,
                "variants": num_variants,
                "noise_gates": list(noise_gates),
                "seed": seed,
                "num_qubits": logical_circuit.num_qubits,
                "dag_encoding": "networkx.node_link_data",
            },
        )

        # noiseless = _apply_noise(sim_base_circuit, noise=None)
        # noiseless_counts = _simulate(noiseless, shots)
        # _write_dag_json(dest_dir / "noiseless_dag.json", noiseless)
        # _write_json(dest_dir / "noiseless_counts.json", noiseless_counts)

        # for variant_index, noise in enumerate(
        #     _build_noise_specs(rng, logical_circuit.num_qubits, num_variants, noise_gates),
        #     start=1,
        # ):
        #     noisy_circuit = _apply_noise(sim_base_circuit, noise)
        #     variant_counts = _simulate(noisy_circuit, shots)

        #     tag = f"variant_{variant_index}_{noise.gate}{noise.qubit}"
        #     _write_dag_json(dest_dir / f"{tag}_dag.json", noisy_circuit)
        #     _write_json(dest_dir / f"{tag}_counts.json", variant_counts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate (noiseless + noisy variants) for OpenQASM circuits."
    )
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        default=Path("circuit_set/clifford_circuits"),
        help="Directory containing source .qasm files (auto-detected by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("circuit_sims"),
        help="Where to store simulation artifacts (default: circuit_sims).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of circuits to simulate (<=0 means all). Default: 100.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=5000,
        help="Number of repetitions/shots per simulation (default: 5000).",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=100,
        help="Number of noisy variants per circuit (default: 5).",
    )
    parser.add_argument(
        "--noise-gates",
        nargs="+",
        default=["x", "z"],
        help="Candidate single-qubit gates for noise injection (default: x z).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="PRNG seed used for selecting noisy gates/qubits.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    circuit_dir = _resolve_circuit_dir(args.circuits_dir)
    if args.shots <= 0:
        raise ValueError("Shots must be a positive integer.")
    if args.variants < 0:
        raise ValueError("Variants must be non-negative.")
    noise_gates = [gate.lower() for gate in args.noise_gates]
    if not noise_gates:
        raise ValueError("Provide at least one gate in --noise-gates.")
    process_circuits(
        circuit_dir=circuit_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        shots=args.shots,
        num_variants=args.variants,
        noise_gates=noise_gates,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
