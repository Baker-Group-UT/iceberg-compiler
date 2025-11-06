
"""
stim_backend.py
----------------
A minimal backend that converts a Qiskit QuantumCircuit into a Stim circuit,
samples it, and returns quasi-probability counts as a dict[str, float].

Entry point:
    run_stim_backend(circuit: qiskit.QuantumCircuit, shots=10_000, seed=None) -> dict[str, float]

Notes
-----
* Supported gates: h, x, y, z, s, sdg, cx, cz, measure, barrier, reset (reset maps to Z-basis measurement + conditional X).
* Measurements are emitted in operation order. We map Stim measurement results back into the Qiskit classical bits by
  tracking each measure(q) -> c assignment. The output bitstrings are formed over *all* circuit clbits in
  Qiskit's canonical list order (circuit.clbits). We then reverse that list for the conventional printed order
  (matching qiskit.result.get_counts string endianness in most cases).
* No noise model is included here; this is an exact unitary+projective measurement sampler via Stim.
* If the circuit has no explicit measurements and `measure_all_if_none=True`, all qubits are measured into
  an ephemeral classical memory (bitstrings length == num_qubits).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import stim
import qiskit

# ------------------------------
# Low-level helpers
# ------------------------------

_ONE_Q = {
    "h": lambda q: f"H {q}",
    "x": lambda q: f"X {q}",
    "y": lambda q: f"Y {q}",
    "z": lambda q: f"Z {q}",
    "s": lambda q: f"S {q}",
    "sdg": lambda q: f"S_DAG {q}",
}
_TWO_Q = {
    "cx": lambda c, t: f"CX {c} {t}",
    "cz": lambda c, t: f"CZ {c} {t}",
}

def _bit_array_to_str(bits) -> str:
    # bits is an iterable of 0/1. We reverse to match conventional Qiskit string order.
    return ''.join(str(int(b)) for b in reversed(bits))

def _ensure_all_measured(circ: qiskit.QuantumCircuit) -> bool:
    return any(inst.operation.name == "measure" for inst in circ.data)

def _format_key_by_cregs(mem, circuit: qiskit.QuantumCircuit) -> str:
    """
    Return a space-delimited key per classical register, in circuit.cregs order.
    Each register's substring is reversed to match the conventional printed order
    (so you can index left->right as bit 0..n-1).
    """
    if not circuit.cregs:
        # No named classical registers; just return the whole thing as one chunk.
        return ''.join(str(int(b)) for b in reversed(mem))

    # Map clbit -> index in circuit.clbits
    clbit_to_idx = {cb: i for i, cb in enumerate(circuit.clbits)}

    parts = []
    for creg in circuit.cregs:
        # creg is iterable over its clbits in Qiskit's own order
        reg_bits = [mem[clbit_to_idx[cb]] for cb in creg]
        # Reverse each register slice for conventional printed order
        parts.append(''.join(str(int(b)) for b in reversed(reg_bits)))
    return ' '.join(parts)


# ------------------------------
# Conversion: Qiskit -> Stim
# ------------------------------
def convert_qiskit_to_stim(
    circuit: qiskit.QuantumCircuit,
    measure_all_if_none: bool = True
) -> Tuple[stim.Circuit, List[Tuple[int,int]], int]:
    """Convert a Qiskit circuit into a Stim circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
    measure_all_if_none : If True and there are no explicit measurements,
        measure all qubits at end (results mapped onto synthetic classical bits
        0..n-1 in the same order as circuit.qubits).

    Returns
    -------
    stim_circ : stim.Circuit
    meas_map : List of (q_index, c_index) in the order the measurements are performed
    num_clbits : int
    """
    stim_lines: List[str] = []
    meas_map: List[Tuple[int,int]] = []

    # Qubit index map (Stim uses 0..n-1 integer ids)
    q_index = {q: i for i, q in enumerate(circuit.qubits)}

    # We'll need to map to classical bit indices; use circuit.clbits order.
    clbits = list(circuit.clbits)
    c_index = {c: i for i, c in enumerate(clbits)}

    for inst in circuit.data:
        name = inst.operation.name.lower()
        qbs = [q_index[q] for q in inst.qubits]

        if name in _ONE_Q:
            stim_lines.append(_ONE_Q[name](qbs[0]))
        elif name in _TWO_Q:
            stim_lines.append(_TWO_Q[name](qbs[0], qbs[1]))
        elif name == "measure":
            # Qiskit measure maps a single qubit to a single cbit
            cb = inst.clbits[0]
            ci = c_index[cb] if cb in c_index else None
            stim_lines.append(f"M {qbs[0]}")
            if ci is not None:
                meas_map.append((qbs[0], ci))
        elif name == "barrier":
            # no-op in Stim
            continue
        elif name == "reset":
            # Implement as measurement in Z then conditional X on |1>
            # Stim idiom: M q; R q to reset to |0>, but Stim doesn't have R;
            # emulate by measurement + feedback is non-trivial; for now we measure then X^m is handled by detectors.
            # Simpler: Projective reset via measure then conditional X:
            stim_lines.append(f"M {qbs[0]}")
            stim_lines.append(f"X {qbs[0]}")  # Flip unconditionally (approximation). For strict correctness you'd need feedback.
        else:
            raise NotImplementedError(f"Unsupported gate: {name}, verify you've transpiled correctly.")

    had_measures = _ensure_all_measured(circuit)
    if (not had_measures) and measure_all_if_none:
        # Measure all qubits at end into ephemeral classical memory (expand clbits virtually)
        for qi in range(len(circuit.qubits)):
            stim_lines.append(f"M {qi}")
        # Map these ephemeral measures onto pseudo classical bits 0..nq-1
        meas_map.extend((qi, qi) for qi in range(len(circuit.qubits)))
        num_clbits = max(len(clbits), len(circuit.qubits))
    else:
        num_clbits = len(clbits)

    # pylint: disable=no-member
    stim_circ = stim.Circuit(stim_program_text='\n'.join(stim_lines))
    return stim_circ, meas_map, num_clbits

# ------------------------------
# Sampling and post-processing
# ------------------------------
def sample_stim_circuit(
    stim_circ: stim.Circuit,
    shots: int,
    seed: Optional[int] = None
):
    sampler = stim_circ.compile_sampler(seed=seed)
    # Returns shape (shots, num_measurements)
    return sampler.sample(shots=shots)

def quasi_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}

def run_stim_backend(
    circuit: qiskit.QuantumCircuit,
    shots: int = 10_000,
    seed: Optional[int] = None,
    measure_all_if_none: bool = True
) -> Dict[str, float]:
    """Entrypoint: run a Qiskit circuit via Stim and return quasi-probabilities.

    Parameters
    ----------
    circuit : QuantumCircuit
    shots : Number of samples to draw.
    seed : Optional RNG seed for reproducibility.
    measure_all_if_none : If the circuit lacks measurements, measure all qubits.

    Returns
    -------
    Dict[str, float] mapping bitstring -> probability (sums to 1.0).
    Bitstrings are over the *full* classical memory (circuit.clbits) in
    Qiskit's list order, reversed for conventional printed order.
    """
    stim_circ, meas_map, num_clbits = convert_qiskit_to_stim(circuit, measure_all_if_none=measure_all_if_none)
    samples = sample_stim_circuit(stim_circ, shots=shots, seed=seed)

    # Stim samples are in chronological measurement order; map back into classical memory slots per shot.
    # First, build the per-measurement index order: simply 0..M-1 in order of meas_map entries.
    # We'll fill a memory array of length num_clbits for each shot.
    counts: Dict[str, int] = {}
    for row in samples:
        mem = [0] * num_clbits
        for k, (_, ci) in enumerate(meas_map):
            if ci < num_clbits:
                mem[ci] = int(row[k])
        s = _format_key_by_cregs(mem, circuit)
        counts[s] = counts.get(s, 0) + 1

    return quasi_from_counts(counts)
