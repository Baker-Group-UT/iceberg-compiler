"""
translator.utils.py - A utility file for the translator package that contains 
mapping functions and register measurement appending
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import cirq

from iceberg.utils import logical_capacity


def remap_for_perm(perm: List[Tuple[int, int]], k_log: int) -> Dict[cirq.LineQubit, cirq.LineQubit]:
    """
    Remap logical qubits (LineQubit(i)) to physical indices (patch * k_log + slot)
    based on the provided permutation.
    """
    return {
        cirq.LineQubit(i): cirq.LineQubit(p * k_log + s)
        for i, (p, s) in enumerate(perm)
    }


def mapping_for_perm(
    permutation: List[Tuple[int, int]],
    program_circuit: cirq.Circuit,
    *,
    n_code: int = 4,
) -> dict:
    """
    Compute deterministic metadata to support logical→physical circuit translation.
    Returns:
        - phys_circ
        - patch mappings
        - active patch lists
        - state initialization map
    """
    k_log = logical_capacity(n_code)
    remap = remap_for_perm(permutation, k_log)
    phys_circ = program_circuit.transform_qubits(lambda q: remap[q])

    phys_idxs = [q.x for q in remap.values()]
    n_prog = max(phys_idxs) + 1
    n_codes = (n_prog + k_log - 1) // k_log

    logical_to_patch = {i: i // k_log for i in range(n_prog)}
    idx_in_patch = {i: i % k_log for i in range(n_prog)}

    active_by_patch: Dict[int, List[int]] = {p: [] for p in range(n_codes)}
    for li in phys_idxs:
        active_by_patch[li // k_log].append(li)

    state_map = {i: "0" for i in range(n_prog)}

    return {
        "k_log": k_log,
        "phys_circ": phys_circ,
        "n_prog": n_prog,
        "n_codes": n_codes,
        "logical_to_patch": logical_to_patch,
        "idx_in_patch": idx_in_patch,
        "active_by_patch": active_by_patch,
        "state_map": state_map,
    }


def append_named_terminal_measurements(iceberg_set) -> None:
    """
    Append terminal measurements to the IcebergSet circuit using module count.

    For each module p in range(iceberg_set.num_modules), append:
        - data_{p}         (measures that module's n data qubits)
        - patch_{p}_anc0   (X-parity ancilla)
        - patch_{p}_anc1   (Z-parity ancilla)

    Any *additional* physical qubits allocated after the module blocks
    (e.g., FT flag ancillas) are measured together into a single 'flags' register.
    """
    # 1) Strip existing measurements to avoid duplicate-key errors
    circ_no_meas = cirq.Circuit(
        (op for m in iceberg_set.circuit for op in m.operations
         if not isinstance(op.gate, cirq.MeasurementGate)),
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

    # 2) Per-module layout
    n_modules = getattr(iceberg_set, "num_modules", None)
    if n_modules is None:
        raise AttributeError("iceberg_set must have 'num_modules' (module count).")

    n_data = getattr(iceberg_set, "n", 4)       # physical data qubits per module
    n_anc  = getattr(iceberg_set, "n_anc", 2)   # parity ancillas per module (default 2)
    block  = n_data + n_anc

    meas_ops: List[cirq.Operation] = []

    # 3) Module-block measurements by index math
    for p in range(n_modules):
        base = p * block
        data_block = [cirq.LineQubit(base + i) for i in range(n_data)]
        anc0 = cirq.LineQubit(base + n_data + 0)
        anc1 = cirq.LineQubit(base + n_data + 1)
        meas_ops.append(cirq.measure(*data_block, key=f"data_{p}"))
        meas_ops.append(cirq.measure(anc0, key=f"patch_{p}_anc0"))
        meas_ops.append(cirq.measure(anc1, key=f"patch_{p}_anc1"))

    # 4) Collect all remaining (flag) qubits beyond the allocated module blocks
    #    We assume modules were allocated first; any later allocations (flags)
    #    live at indices >= n_modules * block.
    all_qubits = sorted(
        [q for q in circ_no_meas.all_qubits() if isinstance(q, cirq.LineQubit)],
        key=lambda q: q.x
    )
    reserved_end = n_modules * block
    flag_qubits = [q for q in all_qubits if q.x >= reserved_end]

    if flag_qubits:
        # Measure all flags into ONE register to simplify downstream handling
        meas_ops.append(cirq.measure(*flag_qubits, key="flags"))

    # 5) Rebuild circuit with terminal measurements
    iceberg_set.circuit = circ_no_meas + cirq.Circuit(cirq.Moment(meas_ops))
