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
    Append terminal measurements to Iceberg iceberg_set circuit that conform to the 
    expected labeling for post-selection and decoding.

    Adds 3 keys per patch:
        - data_{p}
        - patch_{p}_anc0
        - patch_{p}_anc1
    """
    circ_no_meas = cirq.Circuit(
        (op for m in iceberg_set.circuit for op in m.operations
         if not isinstance(op.gate, cirq.MeasurementGate)),
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

    n_data = getattr(iceberg_set, "n", 4)
    n_anc = getattr(iceberg_set, "n_anc", 2)
    block = n_data + n_anc

    all_idxs = [q.x for q in circ_no_meas.all_qubits() if hasattr(q, "x")]
    max_idx = max(all_idxs) if all_idxs else -1
    total_phys = max_idx + 1
    n_patches = (total_phys + block - 1) // block

    meas_ops: list[cirq.Operation] = []
    for p in range(n_patches):
        base = p * block
        data_block = [cirq.LineQubit(base + i) for i in range(n_data)]
        anc0 = cirq.LineQubit(base + n_data + 0)
        anc1 = cirq.LineQubit(base + n_data + 1)
        meas_ops.append(cirq.measure(*data_block, key=f"data_{p}"))
        meas_ops.append(cirq.measure(anc0, key=f"patch_{p}_anc0"))
        meas_ops.append(cirq.measure(anc1, key=f"patch_{p}_anc1"))

    iceberg_set.circuit = circ_no_meas + cirq.Circuit(cirq.Moment(meas_ops))
