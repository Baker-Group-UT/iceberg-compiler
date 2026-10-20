"""
gate_count_stats.py - Extracts gate usage statistics from Qiskit circuits.
"""

from collections import Counter
from typing import Callable, Dict, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from iceberg.layout import patch_of_qubit


def _code_for_qubit(
    circuit: QuantumCircuit, qubits_per_code: Optional[int]
) -> Callable[[int], Optional[int]]:
    """
    Build the qubit-index -> code-index function for a physical circuit.

    With `qubits_per_code=None` (the default) the assignment is read off the
    circuit's Iceberg registers, which is exact at any code size and in
    fault-tolerant mode. Dividing by a fixed patch width instead — as this module
    used to — assumes n=4 and miscounts once flag ancillas are appended after the
    patches. Ancillas map to None and are skipped by the callers.

    Passing an explicit `qubits_per_code` restores the old contiguous-block
    behaviour for circuits that are not Iceberg-encoded.
    """
    if qubits_per_code is not None:
        if qubits_per_code <= 0:
            raise ValueError("qubits_per_code must be positive.")
        return lambda qindex: qindex // qubits_per_code

    patch_map = patch_of_qubit(circuit)
    return patch_map.get


def extract_gate_count(circuit: QuantumCircuit) -> Dict[str, int]:
    """
    Count each gate type in a Qiskit circuit.

    Args:
        circuit: A Qiskit QuantumCircuit (usually the translated physical one).

    Returns:
        Dictionary mapping gate names to counts.
    """
    counter = Counter()

    # Each circuit instruction is a tuple: (Instruction, qargs, cargs)
    for inst, qargs, cargs in circuit.data:
        if inst is None:
            name = "UNKNOWN"
        else:
            # Qiskit Instruction objects have a .name attribute
            name = inst.name.upper()
        counter[name] += 1

    return dict(counter)


def total_gates(gate_count: Dict[str, int]) -> int:
    """
    Compute the total number of gates from the gate count dictionary.

    Args:
        gate_count: Dictionary returned by extract_gate_count().

    Returns:
        Sum of all gate occurrences.
    """
    return sum(gate_count.values())


def one_two_qubit_gate_totals(
    circuit: QuantumCircuit,
    *,
    include_measure: bool = False,
    include_barrier: bool = False,
    include_swap: bool = False,
) -> Tuple[int, int]:
    """
    Count 1q and 2q operations in a circuit.

    By default this excludes MEASURE and BARRIER instructions since they are not
    "logical gates" and tend to be constant across permutations. SWAP is excluded by
    default too, and deliberately stays that way under the transport model: the
    simulator prices a swap as an atom exchange (movement, not a pulse), so it is not a
    two-qubit *gate* — but note it is not free either; a #2q metric that ignores swaps
    is not a complete error proxy for circuits that route with them.
    """
    one_q = 0
    two_q = 0

    for inst in circuit.data:
        op = getattr(inst, "operation", None)
        if op is None:
            op = inst[0]

        name = getattr(op, "name", "").lower()
        if not include_measure and name == "measure":
            continue
        if not include_barrier and name == "barrier":
            continue
        if not include_swap and name == "swap":
            continue

        n_q = getattr(op, "num_qubits", None)
        if n_q is None:
            qargs = getattr(inst, "qubits", None)
            if qargs is None:
                qargs = inst[1]
            n_q = len(qargs)

        if n_q == 1:
            one_q += 1
        elif n_q == 2:
            two_q += 1

    return one_q, two_q


def one_two_qubit_swap_totals(
    circuit: QuantumCircuit,
    *,
    include_measure: bool = False,
    include_barrier: bool = False,
) -> Tuple[int, int, int]:
    """
    Count 1q, 2q, and SWAP operations in a circuit.
    SWAPs are not included in the 2q total.
    """
    one_q = 0
    two_q = 0
    swaps = 0

    for inst in circuit.data:
        op = getattr(inst, "operation", None)
        if op is None:
            op = inst[0]

        name = getattr(op, "name", "").lower()
        if not include_measure and name == "measure":
            continue
        if not include_barrier and name == "barrier":
            continue
        if name == "swap":
            swaps += 1
            continue

        n_q = getattr(op, "num_qubits", None)
        if n_q is None:
            qargs = getattr(inst, "qubits", None)
            if qargs is None:
                qargs = inst[1]
            n_q = len(qargs)

        if n_q == 1:
            one_q += 1
        elif n_q == 2:
            two_q += 1

    return one_q, two_q, swaps


def gate_counts_by_code(
    circuit: QuantumCircuit,
    *,
    qubits_per_code: Optional[int] = None,
    include_measure: bool = False,
    include_barrier: bool = False,
    include_swap: bool = False,
) -> Dict[int, Tuple[float, float]]:
    """
    Count 1q and 2q operations per code, grouping qubits by contiguous blocks.

    Two-qubit gates are split evenly across the participating codes. Qubits that
    belong to no code (fault-tolerant flag ancillas) are skipped.
    """
    code_for_qubit = _code_for_qubit(circuit, qubits_per_code)

    one_q_counts: Dict[int, float] = {}
    two_q_counts: Dict[int, float] = {}

    for inst, qargs, _ in circuit.data:
        op = inst
        name = getattr(op, "name", "").lower()
        if not include_measure and name == "measure":
            continue
        if not include_barrier and name == "barrier":
            continue
        if not include_swap and name == "swap":
            continue

        qubit_indices = [circuit.find_bit(q).index for q in qargs]
        if len(qubit_indices) == 1:
            code = code_for_qubit(qubit_indices[0])
            if code is None:
                continue
            one_q_counts[code] = one_q_counts.get(code, 0.0) + 1.0
        elif len(qubit_indices) == 2:
            code_i = code_for_qubit(qubit_indices[0])
            code_j = code_for_qubit(qubit_indices[1])
            if code_i is None or code_j is None:
                # A flagged gadget's ancilla leg: attribute it to the patch end.
                code = code_i if code_i is not None else code_j
                if code is None:
                    continue
                two_q_counts[code] = two_q_counts.get(code, 0.0) + 1.0
            elif code_i == code_j:
                two_q_counts[code_i] = two_q_counts.get(code_i, 0.0) + 1.0
            else:
                two_q_counts[code_i] = two_q_counts.get(code_i, 0.0) + 0.5
                two_q_counts[code_j] = two_q_counts.get(code_j, 0.0) + 0.5

    codes = set(one_q_counts) | set(two_q_counts)
    return {code: (one_q_counts.get(code, 0.0), two_q_counts.get(code, 0.0)) for code in codes}


def depth_by_code(
    circuit: QuantumCircuit,
    *,
    qubits_per_code: Optional[int] = None,
    include_measure: bool = False,
    include_barrier: bool = False,
    include_swap: bool = False,
) -> Dict[int, int]:
    """
    Count per-code depth as the number of circuit layers that touch each code.
    """
    code_for_qubit = _code_for_qubit(circuit, qubits_per_code)

    code_depths: Dict[int, int] = {}
    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        ops = layer["graph"].op_nodes()
        codes_in_layer = set()
        for node in ops:
            name = getattr(node.op, "name", "").lower()
            if not include_measure and name == "measure":
                continue
            if not include_barrier and name == "barrier":
                continue
            if not include_swap and name == "swap":
                continue
            for q in node.qargs:
                code = code_for_qubit(circuit.find_bit(q).index)
                if code is not None:
                    codes_in_layer.add(code)
        for code in codes_in_layer:
            code_depths[code] = code_depths.get(code, 0) + 1

    return code_depths
