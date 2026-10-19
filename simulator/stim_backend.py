# This function is for stim. It needs to look at the circuit and create a minimal
# paulichannel noise model
# based off of the qubit interactions. I.e.
# We should not create a fully connected graph of the different
# pauli error channels, just the ones that are actually used in the circuit.
# pylint: disable = ['E1101']

from typing import List
import qiskit
import stim
import math

from .noise_models import (manhattan_distance)
from .noise_markers import NOISE_BEGIN, NOISE_END
from .scheduler import compute_idle_times

ONE_Q = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sx": "SQRT_X",
    "measure": "M",
}

TWO_Q = {
    "cx": "CX",
    "cz": "CZ",
    "swap": "SWAP",
}

NOISE_PARAMS = {
    "distance_scalar": 1.0,
    "movement_speed": 1.0,
    "p1": 1e-3,
    "p2": 5e-3,
    "p_meas": 0.02,
    "T1": 100e-6,
    "T2": 80e-6,
    "idle_time_scalar": 1.0,
    "t_1q": 50e-9,
    "t_2q": 300e-9,
}

def _clamp_arr(vals: List[float]) -> List[float]:
    ret: List[float] = []
    for val in vals:
        val = max(0.0, min(1.0, val))
        ret.append(val)
    return ret

def _normalize_probs(vals: List[float]) -> List[float]:
    z = sum(vals)
    if z <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / z for v in vals]

# NOTE: Twirled symmetric depolarizing channel is a symmetric depolarizing channel
def _pauli_twirl_approx(t: float, T1: float, T2: float) -> List[float]:
    p_x = p_y = (1 - math.exp(-t / T1)) / 4
    p_z = (1 - math.exp(-t / T2)) / 2 - (1 - math.exp(-t / T1)) / 4
    p_i = 1 - p_x - p_y - p_z
    p = _normalize_probs(_clamp_arr([p_i, p_x, p_y, p_z]))
    return p

def _noise_from_distance(q0: int, q1: int, layout) -> List[float]:
    pos0, pos1 = layout[q0], layout[q1]
    distance = manhattan_distance(pos0, pos1) * 1e-8 * NOISE_PARAMS["distance_scalar"]
    time_to_move = 2 * distance / NOISE_PARAMS["movement_speed"]
    time_2q = NOISE_PARAMS["t_2q"] + time_to_move

    q = _pauli_twirl_approx(time_2q, NOISE_PARAMS["T1"], NOISE_PARAMS["T2"])
    return q

def add_distance_noise(stim_circuit: stim.Circuit, qubits: list[int], layout) -> None:
    """
    Add a distance based DEPOLARIZE2 instruction that's probability scales off of the 
    distance between two qubits. Should be called only for qubits which interact in the circuit.
    """
    if len(qubits) != 2:
        return
    q0, q1 = qubits

    p2 = NOISE_PARAMS["p2"]
    if p2 > 0:
        stim_circuit.append("DEPOLARIZE2", [q0, q1], [p2])

    if layout is not None:
        p_i, p_x, p_y, p_z = _noise_from_distance(q0, q1, layout)
        if p_x > 0 or p_y > 0 or p_z > 0:
            stim_circuit.append("PAULI_CHANNEL_1", [q0], [p_x, p_y, p_z])
            stim_circuit.append("PAULI_CHANNEL_1", [q1], [p_x, p_y, p_z])

def add_single_qubit_noise(stim_circuit: stim.Circuit, qubits: list[int], gate_name: str) -> None:
    if gate_name == "measure":
        return
    p1 = NOISE_PARAMS["p1"]
    if p1 > 0:
        stim_circuit.append("DEPOLARIZE1", qubits, [p1])

# NOTE: consider adding back in idle time errors
def qiskit_to_stim(circuit: qiskit.QuantumCircuit, layout=None, include_noise: bool = True) -> stim.Circuit:
    out = stim.Circuit()
    last_time = [0.0] * circuit.num_qubits
    has_noise_markers = any(
        inst.operation.name.lower() in {NOISE_BEGIN, NOISE_END} for inst in circuit.data
    )
    noise_enabled = not has_noise_markers

    def _emit_one_q(gate_name: str, qubits: list[int]) -> None:
        gate_time = NOISE_PARAMS["t_1q"]
        if include_noise and noise_enabled and qubits:
            idle_times = compute_idle_times(qubits, last_time, gate_time)
            idle_time_scalar = NOISE_PARAMS["idle_time_scalar"]
            if idle_time_scalar != 0:
                for q, t_idle in idle_times:
                    t_idle *= idle_time_scalar
                    if t_idle > 0:
                        p_i, p_x, p_y, p_z = _pauli_twirl_approx(
                            t_idle, NOISE_PARAMS["T1"], NOISE_PARAMS["T2"]
                        )
                        if p_x > 0 or p_y > 0 or p_z > 0:
                            out.append("PAULI_CHANNEL_1", [q], [p_x, p_y, p_z])
        out.append(ONE_Q[gate_name], qubits)
        if include_noise and noise_enabled:
            add_single_qubit_noise(out, qubits, gate_name)

    def _emit_two_q(gate_name: str, qubits: list[int]) -> None:
        gate_time = NOISE_PARAMS["t_2q"]
        if include_noise and noise_enabled and qubits:
            idle_times = compute_idle_times(qubits, last_time, gate_time)
            idle_time_scalar = NOISE_PARAMS["idle_time_scalar"]
            if idle_time_scalar != 0:
                for q, t_idle in idle_times:
                    t_idle *= idle_time_scalar
                    if t_idle > 0:
                        p_i, p_x, p_y, p_z = _pauli_twirl_approx(
                            t_idle, NOISE_PARAMS["T1"], NOISE_PARAMS["T2"]
                        )
                        if p_x > 0 or p_y > 0 or p_z > 0:
                            out.append("PAULI_CHANNEL_1", [q], [p_x, p_y, p_z])
        out.append(TWO_Q[gate_name], qubits)
        if include_noise and noise_enabled and gate_name != "swap":
            add_distance_noise(out, qubits, layout)

    # basically convert each instruction to stim, inject noise for distance (potench idle later)
    for inst in circuit.data:
        op = inst.operation
        name = op.name.lower()
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == NOISE_BEGIN:
            noise_enabled = True
            last_time = [0.0] * circuit.num_qubits
            continue
        if name == NOISE_END:
            noise_enabled = False
            last_time = [0.0] * circuit.num_qubits
            continue
        if name == "measure":
            if include_noise and noise_enabled:
                p_meas = NOISE_PARAMS["p_meas"]
                if p_meas > 0:
                    out.append("X_ERROR", qubits, [p_meas])
            out.append("M", qubits)
        elif name in ONE_Q: # one qubit interactions
            _emit_one_q(name, qubits)
        elif name in TWO_Q: # two qubit interactions
            _emit_two_q(name, qubits)
        else:
            raise NotImplementedError(f"Unsupported gate for Stim: {name}")
    return out

def _quasi_from_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}

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

def run_stim_backend(
    circuit: qiskit.QuantumCircuit,
    *,
    shots: int = 10_000,
    seed: int | None = None,
    include_noise: bool = True,
    noise_params=None,
    shots_per_chunk: int | None = None,
    progress_callback=None,
    layout=None,
    distance_scalar: float | None = None,
):
    if noise_params is not None:
        NOISE_PARAMS["T1"] = getattr(noise_params, "T1", NOISE_PARAMS["T1"])
        NOISE_PARAMS["T2"] = getattr(noise_params, "T2", NOISE_PARAMS["T2"])
        NOISE_PARAMS["idle_time_scalar"] = getattr(
            noise_params, "idle_time_scalar", NOISE_PARAMS["idle_time_scalar"]
        )
        NOISE_PARAMS["t_1q"] = getattr(noise_params, "one_qubit_gate_time", NOISE_PARAMS["t_1q"])
        NOISE_PARAMS["t_2q"] = getattr(noise_params, "two_qubit_gate_time", NOISE_PARAMS["t_2q"])
        NOISE_PARAMS["p1"] = getattr(noise_params, "one_qubit_error", NOISE_PARAMS["p1"])
        NOISE_PARAMS["p2"] = getattr(noise_params, "two_qubit_error", NOISE_PARAMS["p2"])
        NOISE_PARAMS["p_meas"] = getattr(noise_params, "readout_error", NOISE_PARAMS["p_meas"])
    if distance_scalar is not None:
        NOISE_PARAMS["distance_scalar"] = distance_scalar

    sc = qiskit_to_stim(circuit, layout=layout, include_noise=include_noise)
    sampler = sc.compile_sampler(seed=seed)
    clbits = list(circuit.clbits)
    clbit_index = {cb: idx for idx, cb in enumerate(clbits)}
    meas_map = []
    for inst in circuit.data:
        if inst.operation.name.lower() != "measure":
            continue
        q = circuit.find_bit(inst.qubits[0]).index
        if inst.clbits:
            for cb in inst.clbits:
                meas_map.append((q, clbit_index.get(cb)))
        else:
            meas_map.append((q, None))

    num_clbits = len(clbits)
    if not meas_map or shots <= 0:
        return {}

    counts: dict[str, int] = {}
    chunk_size = max(1, shots_per_chunk or shots)
    remaining = shots
    while remaining > 0:
        current = min(chunk_size, remaining)
        samples = sampler.sample(shots=current)
        for row in samples:
            mem = [0] * max(num_clbits, 1)
            for k, (_, ci) in enumerate(meas_map):
                if ci is None or ci >= len(mem):
                    continue
                mem[ci] = int(row[k])
            key = _format_key_by_cregs(mem, circuit)
            counts[key] = counts.get(key, 0) + 1
        if progress_callback is not None:
            progress_callback(current)
        remaining -= current

    return _quasi_from_counts(counts)
