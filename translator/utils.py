"""
translator.utils.py - Qiskit utilities for mapping and measurement metadata
"""

from __future__ import annotations
from typing import List, Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from iceberg.utils import logical_capacity


def remap_for_perm(perm: List[Tuple[int, int]], k_log: int) -> Dict[int, int]:
    """
    Remap logical qubit *indices* i -> physical flat index (patch * k_log + slot)
    based on the provided permutation.
    """
    return {i: (p * k_log + s) for i, (p, s) in enumerate(perm)}


def _clone_cregs(src: QuantumCircuit, dst: QuantumCircuit) -> None:
    """Add classical registers to dst with same names/sizes as src."""
    for creg in src.cregs:
        dst.add_register(ClassicalRegister(len(creg), name=creg.name))


def _new_qc_like(src: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
    """Create a fresh circuit with n_qubits (single qreg) and same cregs as src."""
    qc = QuantumCircuit(QuantumRegister(n_qubits, "q"))
    _clone_cregs(src, qc)
    return qc


def mapping_for_perm(
    permutation: List[Tuple[int, int]],
    program_circuit: QuantumCircuit,
    *,
    n_code: int = 4,
) -> dict:
    """
    Compute deterministic metadata to support logical→physical circuit translation.
    Returns:
      - phys_circ: QuantumCircuit with remapped qubits
      - patch mappings / active lists / state init map
    """
    k_log = logical_capacity(n_code)
    remap = remap_for_perm(permutation, k_log)  # i -> flat phys index
    
    # Determine program size from mapping
    phys_idxs = list(remap.values())
    n_prog = max(phys_idxs) + 1
    n_codes = (n_prog + k_log - 1) // k_log

    # Build a fresh circuit and re-append instructions with remapped qargs
    phys_circ = _new_qc_like(program_circuit, n_prog)

    # Map each original qubit index -> new circuit qubit
    # Assume program_circuit.qubits are in logical order [0..]
    # If not, adjust indexing policy here.
    for inst, qargs, cargs in program_circuit.data:
        mapped_qargs = []
        for q in qargs:
            # old flat index in the original circuit
            old_idx = program_circuit.find_bit(q).index  # position in flat qubit list
            new_idx = remap[old_idx]
            mapped_qargs.append(phys_circ.qubits[new_idx])
        phys_circ.append(inst, mapped_qargs, cargs)

    logical_to_patch = {i: i // k_log for i in range(k_log * n_codes)}
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
