"""
stim_backend.py
----------------
Stim-based backend used by simulator.Simulator when `with_stim(True)` is
selected. The backend converts a Qiskit QuantumCircuit into a Stim circuit,
optionally injects idle + gate noise (matching the original
`stim_conversion_updated.ipynb` prototype), samples it, and returns normalized
probabilities keyed by Qiskit's classical register ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import networkx as nx
import numpy as np
import qiskit
import stim

from simulator.noise_models import StimNoiseParameters, stim_params_from_noise_model

_DEFAULT_STIM_PARAMS = StimNoiseParameters()

# Stim text snippets for basic gates (taken from stim_conversion_updated.ipynb)
_ONE_Q_GATES = {
    "h": lambda q: f"H {q}",
    "x": lambda q: f"X {q}",
    "y": lambda q: f"Y {q}",
    "z": lambda q: f"Z {q}",
    "s": lambda q: f"S {q}",
    "sdg": lambda q: f"S_DAG {q}",
    "reset": lambda q: f"R {q}",
    "measure": lambda q: f"M {q}",
}
_TWO_Q_GATES = {
    "cx": lambda c, t: f"CX {c} {t}",
    "cz": lambda c, t: f"CZ {c} {t}",
    "swap": lambda q0, q1: f"SWAP {q0} {q1}",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def analytic_pta_probs_from_T1_T2(
    t_gate: float,
    T1: float,
    T2: float,
) -> dict[str, float]:
    """
    Analytic PTA probabilities for amplitude damping + dephasing over time `t_gate`.

    Matches the expressions used in the original notebook port.
    Returns dict {\"I\",\"X\",\"Y\",\"Z\"} (sums to 1.0 within floating error).
    """
    if T1 <= 0 or T2 <= 0 or t_gate < 0:
        raise ValueError("T1, T2 must be positive and t_gate non-negative")

    A = math.exp(-t_gate / (2.0 * T1))
    B = math.exp(-t_gate / T2)
    exp_minus_t_over_T1 = A * A  # e^{-t/T1}

    qx = (1.0 - exp_minus_t_over_T1) / 4.0
    qy = qx
    qI = ((1.0 - A) ** 2) / 4.0 + B
    qZ = ((1.0 + A) ** 2) / 4.0 - B

    q = np.array([qI, qx, qy, qZ], dtype=float)
    q = np.maximum(q, 0.0)
    Z = q.sum()
    if Z == 0.0:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q = q / Z
    return {"I": float(q[0]), "X": float(q[1]), "Y": float(q[2]), "Z": float(q[3])}


def _format_key_by_cregs(mem: List[int], circuit: qiskit.QuantumCircuit) -> str:
    """
    Return a space-delimited key per classical register, in circuit.cregs order.
    Each register's substring is reversed to match Qiskit's conventional printed order.
    """
    if not circuit.cregs:
        return "".join(str(int(b)) for b in reversed(mem))

    clbit_to_idx = {cb: i for i, cb in enumerate(circuit.clbits)}
    parts = []
    for creg in circuit.cregs:
        reg_bits = [mem[clbit_to_idx[cb]] for cb in creg]
        parts.append("".join(str(int(b)) for b in reversed(reg_bits)))
    return " ".join(parts)


def _build_qubit_index_map(circuit: qiskit.QuantumCircuit) -> dict[object, int]:
    """Flatten registers into a bijection between Qiskit qubits and integer ids."""
    mapping: dict[object, int] = {}
    for idx, q in enumerate(circuit.qubits):
        mapping[q] = idx
        mapping[idx] = q  # reverse lookup used for ancilla detection
    return mapping


def _default_gate_time_map(
    num_qubits: int,
    one_qubit_gate_time: float,
    two_qubit_gate_time: float,
) -> dict[object, float]:
    """
    Build a simple duration lookup for every qubit and ordered qubit pair.
    """
    gate_time_map: dict[object, float] = {i: one_qubit_gate_time for i in range(num_qubits)}
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue
            gate_time_map[(i, j)] = two_qubit_gate_time
    return gate_time_map


@dataclass(eq=False)
class GateNode:
    """
    Lightweight node wrapper for the DAG conversion.
    """

    qubits: Tuple[int, ...]
    instr: str
    gate_time: float
    index: int
    clbits: Tuple[qiskit.circuit.Clbit, ...]

    def __hash__(self) -> int:
        return hash(self.index)


def _convert_qiskit_to_dag(
    circuit: qiskit.QuantumCircuit,
    qubit_index_map: dict[object, int],
    gate_duration_map: dict[object, float],
    meas_time: float,
    default_one_qubit_time: float,
    default_two_qubit_time: float,
) -> nx.DiGraph:
    """
    Build a DAG where nodes are GateNode objects annotated with timing metadata.
    """
    dag = nx.DiGraph()
    qubit_last_used: dict[int, GateNode] = {}
    node_index = 0

    for inst in circuit.data:
        op = inst.operation
        name = op.name.lower()
        if name == "barrier":
            continue

        qubits = tuple(qubit_index_map[q] for q in inst.qubits)
        if not qubits:
            continue

        if len(qubits) == 1:
            gate_time = meas_time if name == "measure" else gate_duration_map.get(qubits[0], default_one_qubit_time)
        else:
            gate_time = gate_duration_map.get(
                qubits,
                gate_duration_map.get((qubits[1], qubits[0]), default_two_qubit_time),
            )

        node = GateNode(
            qubits=qubits,
            instr=name,
            gate_time=gate_time,
            index=node_index,
            clbits=tuple(inst.clbits),
        )
        node_index += 1
        dag.add_node(node)

        for q in node.qubits:
            if q in qubit_last_used:
                dag.add_edge(qubit_last_used[q], node)
            qubit_last_used[q] = node

    return dag


# ---------------------------------------------------------------------------
# Conversion + sampling
# ---------------------------------------------------------------------------
def _idle_noise_lines(
    duration: float,
    targets: Tuple[int, ...],
    include_noise: bool,
    T1: float,
    T2: float,
) -> List[str]:
    """Generate PTA idle noise for the specified duration."""
    if (
        (not include_noise)
        or duration <= 0
        or not targets
        or T1 <= 0
        or T2 <= 0
    ):
        return []
    probs = analytic_pta_probs_from_T1_T2(duration, T1, T2)
    px = probs["X"]
    py = probs["Y"]
    pz = probs["Z"]
    tgt_str = " ".join(str(t) for t in targets)
    return [f"PAULI_CHANNEL_1({px:.12g},{py:.12g},{pz:.12g}) {tgt_str}"]


def convert_qiskit_to_stim(
    circuit: qiskit.QuantumCircuit,
    *,
    params: StimNoiseParameters = _DEFAULT_STIM_PARAMS,
    include_noise: bool = True,
    measure_all_if_none: bool = True,
) -> Tuple[stim.Circuit, List[Tuple[int, Optional[int]]], int]:
    """
    Convert a Qiskit circuit into Stim text, mirroring stim_conversion_updated.ipynb.

    Returns:
        stim.Circuit,
        measurement map [(qubit_index, classical_index|None) ...] in measurement order,
        num classical bits (after auto-measure logic, if any)
    """
    if circuit.num_qubits == 0:
        raise ValueError("Stim backend requires at least one qubit in the circuit.")

    noise_params = params or _DEFAULT_STIM_PARAMS
    T1 = noise_params.T1
    T2 = noise_params.T2
    one_qubit_gate_time = noise_params.one_qubit_gate_time
    two_qubit_gate_time = noise_params.two_qubit_gate_time
    single_qubit_error = noise_params.one_qubit_error
    two_qubit_error = noise_params.two_qubit_error
    meas_time = noise_params.meas_time

    qubit_index_map = _build_qubit_index_map(circuit)
    gate_time_map = _default_gate_time_map(circuit.num_qubits, one_qubit_gate_time, two_qubit_gate_time)
    dag = _convert_qiskit_to_dag(
        circuit,
        qubit_index_map,
        gate_time_map,
        meas_time,
        one_qubit_gate_time,
        two_qubit_gate_time,
    )

    stim_lines: List[str] = []
    reset_line = "R " + " ".join(str(i) for i in range(circuit.num_qubits))
    stim_lines.append(reset_line)

    meas_map: List[Tuple[int, Optional[int]]] = []
    clbits = list(circuit.clbits)
    clbit_index = {cb: idx for idx, cb in enumerate(clbits)}

    topo_order = list(nx.topological_sort(dag))
    path_lengths: dict[GateNode, float] = {}
    last_finish_time: dict[int, float] = {i: 0.0 for i in range(circuit.num_qubits)}

    for node in topo_order:
        preds = list(dag.predecessors(node))
        start_time = max((path_lengths[p] for p in preds), default=0.0)
        finish_time = start_time + node.gate_time
        path_lengths[node] = finish_time

        for q in node.qubits:
            idle_duration = max(0.0, start_time - last_finish_time.get(q, 0.0))
            stim_lines.extend(_idle_noise_lines(idle_duration, (q,), include_noise, T1, T2))

        if node.instr in _ONE_Q_GATES:
            q = node.qubits[0]
            stim_lines.append(_ONE_Q_GATES[node.instr](q))

            if node.instr == "measure":
                clbits_for_gate = node.clbits if node.clbits else ()
                for cb in clbits_for_gate:
                    meas_map.append((q, clbit_index.get(cb)))
                if not clbits_for_gate:
                    meas_map.append((q, None))
            elif include_noise and node.instr != "reset":
                stim_lines.append(f"DEPOLARIZE1({single_qubit_error:.12g}) {q}")

        elif node.instr in _TWO_Q_GATES:
            ctrl, tgt = node.qubits
            stim_lines.append(_TWO_Q_GATES[node.instr](ctrl, tgt))
            if include_noise:
                stim_lines.append(f"DEPOLARIZE2({two_qubit_error:.12g}) {ctrl} {tgt}")
        else:
            raise NotImplementedError(f"Unsupported gate for Stim backend: {node.instr}")

        for q in node.qubits:
            last_finish_time[q] = finish_time

    had_measures = any(inst.operation.name == "measure" for inst in circuit.data)
    num_clbits = len(clbits)

    if (not had_measures) and measure_all_if_none:
        for qi in range(circuit.num_qubits):
            stim_lines.append(f"M {qi}")
            meas_map.append((qi, qi))
        num_clbits = max(num_clbits, circuit.num_qubits)

    stim_text = "\n".join(stim_lines)
    # pylint: disable=no-member
    stim_circuit = stim.Circuit(stim_program_text=stim_text)
    return stim_circuit, meas_map, num_clbits


def _sample_stim_circuit(
    stim_circ: stim.Circuit,
    shots: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    sampler = stim_circ.compile_sampler(seed=seed)
    return sampler.sample(shots=shots)


def quasi_from_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_stim_backend(
    circuit: qiskit.QuantumCircuit,
    shots: int = 10_000,
    seed: Optional[int] = None,
    *,
    include_noise: bool = True,
    noise_params: Optional[StimNoiseParameters] = None,
    measure_all_if_none: bool = True,
) -> dict[str, float]:
    """
    Entry point for Simulator._stimulate().

    Args:
        circuit: QuantumCircuit to simulate.
        shots: Number of samples to draw.
        seed: Optional RNG seed.
        include_noise: If False, the Stim circuit is generated in the error-free mode.
        measure_all_if_none: Measure every qubit if the circuit lacks explicit measurements.
    """
    stim_circ, meas_map, num_clbits = convert_qiskit_to_stim(
        circuit,
        params=noise_params or _DEFAULT_STIM_PARAMS,
        include_noise=include_noise,
        measure_all_if_none=measure_all_if_none,
    )
    if not meas_map:
        return {}

    samples = _sample_stim_circuit(stim_circ, shots=shots, seed=seed)
    counts: dict[str, int] = {}

    for row in samples:
        mem = [0] * max(num_clbits, 1)
        for k, (_, ci) in enumerate(meas_map):
            if ci is None or ci >= len(mem):
                continue
            mem[ci] = int(row[k])
        key = _format_key_by_cregs(mem, circuit)
        counts[key] = counts.get(key, 0) + 1

    return quasi_from_counts(counts)
