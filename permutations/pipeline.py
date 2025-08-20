# pipeline.py
"""
High–level pipeline:

    logical_circ          (Cirq – logical gate set)
          │
    enumerate_permutations()
          │   →  list[perm]           e.g. [[(0,0),(0,1),(1,0)], …]
          ▼
    summarise_permutations()          (FAST – counts only)
          │   →  pd.DataFrame         one row per perm
          ▼
    build_physical_circuit()          (SLOW – only when you ask for it)
          →  full IcebergGrid circuit
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from collections import Counter
import cirq, pandas as pd

# You only need cirq types in _gate_counts_minus_encoding
from cirq.ops.swap_gates import SwapPowGate

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from permutations.circuit_translator import replay_logical_circuit
from permutations.permutation_generator import unique_logical_permutations, logical_capacity
from iceberg.IcebergCode import IcebergGrid, BarrierGate

__all__ = [
    "print_in_windows",
    "enumerate_permutations",
    "summarise_permutations",
    "build_physical_circuit",
    "_perm_key",
]

# ------------------------------------------------------------------ #
# pretty printing
# ------------------------------------------------------------------ #
def print_in_windows(circ: cirq.Circuit, window: int = 8):
    for t in range(0, len(circ), window):
        sub = cirq.Circuit(circ[t : t + window])
        print(f"\n# Moments {t}–{t+window-1}")
        print(sub)

# ------------------------------------------------------------------ #
# permutations
# ------------------------------------------------------------------ #
def enumerate_permutations(
    code_sizes: List[int],
    logical_circuit: cirq.Circuit,
    *,
    status_interval: float = 0.05,
) -> List[List[Tuple[int,int]]]:
    q = len(logical_circuit.all_qubits())
    return unique_logical_permutations(code_sizes, q, status_interval=status_interval)

# ------------------------------------------------------------------ #
# counting helpers (minus encoding layer)
# ------------------------------------------------------------------ #
_SKIP = (BarrierGate, cirq.MeasurementGate)

def _gate_counts_minus_encoding(circ: cirq.Circuit, n_codes: int) -> Counter:
    """Return non-encoding gate counts for *circ* (Barrier/Meas skipped)."""
    enc_H  = 1 * n_codes
    enc_CX = 3 * n_codes

    cnt = Counter(
        type(op.gate)
        for op in circ.all_operations()
        if not isinstance(op.gate, _SKIP)
    )

    # subtract encoding overhead
    cnt[cirq.HPowGate]  -= enc_H
    cnt[cirq.CXPowGate] -= enc_CX

    # rename canonical labels; drop zeros/unknowns
    canon = {
        cirq.HPowGate        : "H",
        cirq.CXPowGate       : "CNOT",
        cirq.ops.Rz          : "RZ",
        SwapPowGate          : "SWAP",
        cirq.ops.measure     : "Measure",
    }
    return Counter({canon[k]: v for k, v in cnt.items() if v > 0 and k in canon})

# ------------------------------------------------------------------ #
# remap helpers
# ------------------------------------------------------------------ #
def _remap_for_perm(perm: List[Tuple[int,int]], k_log: int) -> Dict[cirq.LineQubit, cirq.LineQubit]:
    """
    LineQubit(i) -> LineQubit(patch*k_log + slot)
    with (patch, slot) from perm[i].
    """
    return {cirq.LineQubit(i): cirq.LineQubit(p * k_log + s)
            for i, (p, s) in enumerate(perm)}

def _perm_key(perm):
    return tuple(tuple(t) for t in perm)

def _mapping_for_perm(
    permutation: List[Tuple[int,int]],
    logical_circ: cirq.Circuit,
    *,
    n_code: int,
):
    """Compute all deterministic mapping metadata shared by summary/builder."""
    k_log    = logical_capacity(n_code)
    remap    = _remap_for_perm(permutation, k_log)
    phys_circ= logical_circ.transform_qubits(lambda q: remap[q])

    phys_idxs = [q.x for q in remap.values()]
    n_prog    = max(phys_idxs) + 1
    n_codes   = (n_prog + k_log - 1) // k_log

    l2p  = {i: i // k_log for i in range(n_prog)}
    idxp = {i: i %  k_log for i in range(n_prog)}

    active_by_patch = {p: [] for p in range(n_codes)}
    for li in phys_idxs:
        active_by_patch[li // k_log].append(li)

    state_map = {i: "0" for i in range(n_prog)}

    return {
        "k_log": k_log,
        "phys_circ": phys_circ,
        "n_prog": n_prog,
        "n_codes": n_codes,
        "logical_to_patch": l2p,
        "idx_in_patch": idxp,
        "active_by_patch": active_by_patch,
        "state_map": state_map,
    }

# ------------------------------------------------------------------ #
# unified builder (single source of truth)
# ------------------------------------------------------------------ #
def build_physical_circuit(
    permutation: List[Tuple[int,int]],
    logical_circ: cirq.Circuit,
    *,
    n_code: int = 4,
    faulty: bool = False,
    verbose: bool = False,
    include_circuit: bool = True,   # <-- let callers drop the circuit to save RAM
) -> Dict:
    meta = _mapping_for_perm(permutation, logical_circ, n_code=n_code)

    fault_log: List[str] = []

    grid = IcebergGrid(n_program=meta["n_prog"], n=n_code, circuit=cirq.Circuit())
    grid.encode(); grid.barrier()
    replay_logical_circuit(
        grid,
        meta["phys_circ"],
        logical_to_patch=meta["logical_to_patch"],
        idx_in_patch=meta["idx_in_patch"],
        active_by_patch=meta["active_by_patch"],
        state_map=meta["state_map"],
        fault_log=fault_log,
        faulty=faulty,
        verbose=verbose,
    )

    counts = _gate_counts_minus_encoding(grid.circuit, meta["n_codes"])

    out = {
        "permutation"   : permutation,
        "fault_tolerant": (len(fault_log) == 0),
        "gate_count"    : counts,
        "log"           : fault_log,
    }
    if include_circuit:
        out["circuit"] = grid.circuit

    # help GC in large scans
    del grid
    return out

# ------------------------------------------------------------------ #
# FAST scan summary (now delegates to the builder)
# ------------------------------------------------------------------ #
def summarise_permutations(
    perms: List[List[Tuple[int,int]]],
    logical_circ: cirq.Circuit,
    *,
    n_code: int = 4,
    faulty: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []

    for perm in perms:
        variant = build_physical_circuit(
            perm,
            logical_circ,
            n_code=n_code,
            faulty=faulty,
            verbose=verbose,
            include_circuit=False,   # don't keep the full circuit during the sweep
        )
        counts = variant["gate_count"]
        rows.append({
            "permutation"   : perm,
            "perm_key"      : _perm_key(perm),
            "fault_tolerant": variant["fault_tolerant"],
            "total_gates"   : int(sum(counts.values())),
            **counts,
        })

    return (pd.DataFrame(rows)
              .fillna(0)
              .sort_values("total_gates")
              .reset_index(drop=True))