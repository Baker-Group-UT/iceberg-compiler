# ---------------- iceberg_gate_translate.py ---------------------------
"""
Translate a *logical* Cirq circuit (restricted gate–set) onto an IcebergGrid.

Simple one-to-one mappings live in TABLE_A below.
Anything with extra FT-logic (HH, CNOT, CZ, CCZ/CCX …) is delegated to
permutations.apply_gates: apply_hadamard, apply_logical_cnot, apply_logical_cz, apply_ccz_like.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
from typing import Dict, List, Iterable, Tuple, Callable
import cirq
from cirq import ops as _cops
from cirq.ops.pauli_gates import _PauliX, _PauliZ
from iceberg.IcebergCode import IcebergGrid
from cirq.ops.common_gates import HPowGate, CXPowGate, ZPowGate, Rz

from permutations.apply_gates import (
    apply_hadamard,
    apply_logical_cnot,
    apply_logical_cz,     # NEW
    apply_ccz_like,       # NEW
)

# ------------------------------------------------------------------ #
#  HPowGate  (single-qubit H, possibly fused to HH)                  #
# ------------------------------------------------------------------ #
def _handle_H(grid: IcebergGrid, op: cirq.Operation, next_op,
              logical_to_patch, idx_in_patch,
              active_by_patch, state_map, fault_log,
              *, faulty: bool, verbose: bool):
    return apply_hadamard(grid, op, next_op,
                          logical_to_patch, active_by_patch,
                          state_map, fault_log,
                          faulty=faulty, verbose=verbose)

# ------------------------------------------------------------------ #
#  CXPowGate  (logical CNOT or transversal CNOT)                     #
# ------------------------------------------------------------------ #
def _handle_CNOT(grid: IcebergGrid, op: cirq.Operation, _unused_next,
                 logical_to_patch, idx_in_patch,
                 active_by_patch, state_map, fault_log,
                 *, faulty: bool, verbose: bool):
    ctl, tgt = (q.x for q in op.qubits)
    return apply_logical_cnot(grid, ctl, tgt,
                              logical_to_patch, idx_in_patch,
                              active_by_patch, state_map, fault_log,
                              faulty=faulty, verbose=verbose)

# ------------------------------------------------------------------ #
#  CZPowGate (logical CZ or transversal CZ)                          #
# ------------------------------------------------------------------ #
def _handle_CZ(grid: IcebergGrid, op: cirq.Operation, _unused_next,
               logical_to_patch, idx_in_patch,
               active_by_patch, state_map, fault_log,
               *, faulty: bool, verbose: bool):
    q0, q1 = (q.x for q in op.qubits)
    # Only support exponent == 1 for now (pure CZ). Other exponents → raise.
    gate = op.gate
    if hasattr(gate, "exponent") and not math.isclose(float(gate.exponent) % 2.0, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise NotImplementedError("Only CZ (exponent==1) supported in translator.")
    return apply_logical_cz(grid, q0, q1,
                            logical_to_patch, idx_in_patch,
                            active_by_patch, state_map, fault_log,
                            faulty=faulty, verbose=verbose)

# ------------------------------------------------------------------ #
#  CCZ / CCX  (translate via FT-friendly CCZ; CCX = H·CCZ·H)         #
# ------------------------------------------------------------------ #
def _handle_CCZ_like(grid: IcebergGrid, op: cirq.Operation, _unused_next,
                     logical_to_patch, idx_in_patch,
                     active_by_patch, state_map, fault_log,
                     *, faulty: bool, verbose: bool):
    a, b, c = (q.x for q in op.qubits)
    gate = op.gate
    if hasattr(gate, "exponent") and not math.isclose(float(gate.exponent) % 2.0, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise NotImplementedError("Only CCZ (exponent==1) supported.")
    return apply_ccz_like(grid, a, b, c, mode="CCZ",
                          logical_to_patch=logical_to_patch,
                          idx_in_patch=idx_in_patch,
                          active_by_patch=active_by_patch,
                          state_map=state_map, fault_log=fault_log,
                          faulty=faulty, verbose=verbose)

def _handle_CCX_like(grid: IcebergGrid, op: cirq.Operation, _unused_next,
                     logical_to_patch, idx_in_patch,
                     active_by_patch, state_map, fault_log,
                     *, faulty: bool, verbose: bool):
    a, b, c = (q.x for q in op.qubits)
    gate = op.gate
    if hasattr(gate, "exponent") and not math.isclose(float(gate.exponent) % 2.0, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise NotImplementedError("Only CCX (exponent==1) supported.")
    # CCX = H(c) · CCZ(a,b,c) · H(c)
    return apply_ccz_like(grid, a, b, c, mode="CCX",
                          logical_to_patch=logical_to_patch,
                          idx_in_patch=idx_in_patch,
                          active_by_patch=active_by_patch,
                          state_map=state_map, fault_log=fault_log,
                          faulty=faulty, verbose=verbose)

# ---------------------------------------------------------------------
#  SIMPLE gate map  (class → (canonical_name , grid_method_name))
# ---------------------------------------------------------------------
TABLE_A: Dict[type, tuple[str,str]] = {
    _PauliX : ("X" , "X"),           # X(π)
    _PauliZ : ("Z" , "Z"),           # Z(π)
    Rz      : ("RZ", "RZ"),          # will need angle
    ZPowGate: ("RZ", "RZ"),          # Z**t == RZ(tπ) up to global phase
    _cops.SwapPowGate: ("SWAP","swap"),
}

# ------------------------------------------------------------------ #
#  Complex table                                                     #
# ------------------------------------------------------------------ #
SPECIAL_HANDLERS: Dict[type, Callable] = {
    HPowGate           : _handle_H,
    CXPowGate          : _handle_CNOT,
    getattr(cirq, "CZPowGate" , cirq.Gate): _handle_CZ,
    getattr(cirq, "CCZPowGate", cirq.Gate): _handle_CCZ_like,
    getattr(cirq, "CCXPowGate", cirq.Gate): _handle_CCX_like,
}

def _rz_angle_from_gate(gate) -> float:
    if hasattr(gate, "rads"):
        return float(gate.rads)
    if hasattr(gate, "_rads"):
        return float(gate._rads)
    if isinstance(gate, ZPowGate):
        return float(gate.exponent) * math.pi
    raise TypeError(f"Unsupported RZ gate type: {type(gate)}")

# --------------------------------------------------------------------- #
#                           Gate translator                             #
# --------------------------------------------------------------------- #
def translate_and_apply(op: cirq.Operation,
                        *,
                        grid: IcebergGrid,
                        logical_to_patch: Dict[int,int],
                        idx_in_patch: Dict[int,int],
                        active_by_patch: Dict[int, List[int]],
                        state_map: Dict[int,str],
                        fault_log: List[str],
                        faulty: bool = False,
                        verbose: bool = False,
                        lookahead: cirq.Operation | None = None):
    g_class = type(op.gate)

    # ---- special handlers ------------------------------------------
    if g_class in SPECIAL_HANDLERS:
        res = SPECIAL_HANDLERS[g_class](
            grid, op,
            lookahead if g_class is _cops.HPowGate else None,
            logical_to_patch, idx_in_patch,
            active_by_patch, state_map, fault_log,
            faulty=faulty, verbose=verbose
        )
        return res

    # ----- simple table-A mapping ------------------------------------
    if g_class in TABLE_A:
        canon, method = TABLE_A[g_class]
        qubits = [q.x for q in op.qubits]

        if canon == "RZ":
            angle = _rz_angle_from_gate(op.gate)
            qlog = qubits[0]
            if verbose:
                print(f"[translate] RZ on logical {qlog} with angle {angle/math.pi:.2f}π")
            grid.RZ(qlog, angle)
            return

        if canon == "SWAP":
            q0, q1 = qubits
            patch, i0 = divmod(q0, grid.k)
            patch2, i1 = divmod(q1, grid.k)
            if patch != patch2:
                raise NotImplementedError("Cross-patch SWAP not supported")
            grid.swap([[patch, i0, i1]], faulty=faulty)
            return

        if len(qubits) == 1:
            getattr(grid, method)(qubits[0])
            return

    # ----- unsupported gate ------------------------------------------
    raise ValueError(f"Gate {op.gate} is not in the Iceberg gate-set")

# ---------------------------------------------------------------------
#  Replay a full logical circuit on the grid
# ---------------------------------------------------------------------
def replay_logical_circuit(grid: IcebergGrid,
                           logical_circuit: cirq.Circuit,
                           *,
                           logical_to_patch: Dict[int,int],
                           idx_in_patch: Dict[int,int],
                           active_by_patch: Dict[int,List[int]],
                           state_map: Dict[int,str],
                           fault_log: List[str],
                           faulty: bool = False,
                           verbose: bool = False):
    """
    For each moment in `logical_circuit`, apply the translated operations to
    `grid`, inserting a barrier after every moment to preserve depth.
    """
    for mom in logical_circuit:
        ops = list(mom.operations)
        skip_next = False
        for i, op in enumerate(ops):
            if skip_next:
                skip_next = False
                continue

            nxt = ops[i+1] if i+1 < len(ops) else None
            consumed = translate_and_apply(
                op,
                grid=grid,
                logical_to_patch=logical_to_patch,
                idx_in_patch=idx_in_patch,
                active_by_patch=active_by_patch,
                state_map=state_map,
                fault_log=fault_log,
                faulty=faulty,
                verbose=verbose,
                lookahead=nxt
            )

            # Only skip for H→HH merge (Hadamard handler returns non-None when it fuses)
            if isinstance(op.gate, HPowGate) and consumed is not None:
                skip_next = True

        grid.barrier()




# ------------------------------------------------------------
#  Quick-check the translation layer
# ------------------------------------------------------------
if __name__ == "__main__":

    import math, pytest

    def fresh_env(n_program=2, n_code=4):
        grid  = IcebergGrid(n_program=n_program, n=n_code, circuit=cirq.Circuit())
        grid.encode(); grid.barrier()
        n_lp  = n_code - 2
        l2p   = {i: i // n_lp for i in range(n_program)}
        idxp  = {i: i %  n_lp for i in range(n_program)}
        active= {p: [q for q in range(n_program) if l2p[q]==p]
                 for p in set(l2p.values())}
        smap  = {i: "0" for i in range(n_program)}
        flog  = []
        return grid, l2p, idxp, active, smap, flog

    # ---- 1. single-patch H fusion ---------------------------------
    q0, q1 = cirq.LineQubit.range(2)
    circ = cirq.Circuit(cirq.Moment([cirq.H(q0), cirq.H(q1)]))
    grid, l2p, idxp, active, smap, flog = fresh_env()
    replay_logical_circuit(grid, circ,
                           logical_to_patch=l2p,
                           idx_in_patch=idxp,
                           active_by_patch=active,
                           state_map=smap,
                           fault_log=flog)
    assert sum(isinstance(g.gate, cirq.HPowGate)
               for g in grid.circuit.all_operations()) == 5        # 1 enc + 4 HH
    assert not flog                                              # HH is FT

    # ---- 2. cross-patch transversal CNOT --------------------------
    circ2 = cirq.Circuit(
        cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)),
    )
    grid, l2p, idxp, active, smap, flog = fresh_env(n_program=4)
    replay_logical_circuit(grid, circ2,
                           logical_to_patch=l2p,
                           idx_in_patch=idxp,
                           active_by_patch=active,
                           state_map=smap,
                           fault_log=flog)
    # one transversal CNOT produces n_code (=4) physical CX
    assert any(isinstance(g.gate, cirq.CXPowGate) for g in grid.circuit.all_operations())
    assert not flog

    # ---------------------------------------------------------------------
    #  SIMPLE-gate translation – smoke tests
    # ---------------------------------------------------------------------
    def count_gate(circ: cirq.Circuit, gate_cls) -> int:
        """Count how many operations of type *gate_cls* appear in *circ*."""
        return sum(isinstance(op.gate, gate_cls) for op in circ.all_operations())

    def strip_encoding(
        circ      : cirq.Circuit,
        *,
        grid      : IcebergGrid,
    ) -> Tuple[int,int,int,int]:
        """
        Remove the *encoding layer* contribution from raw gate counts.

        Each Iceberg patch (code) contributes **1 H + 3 CX** during encoding.
        The number of patches is available as ``grid.num_icebergs``.
        """
        n_codes   = grid.num_icebergs          # NOT the (physical) ``n_code`` size
        enc_H     = 1 * n_codes
        enc_CX    = 3 * n_codes

        h   = count_gate(circ, cirq.HPowGate)          - enc_H
        cx  = count_gate(circ, cirq.CXPowGate)         - enc_CX
        rz  = count_gate(circ, cirq.ops.common_gates.Rz)
        swp = count_gate(circ, cirq.ops.swap_gates.SwapPowGate)
        return h, cx, rz, swp
    
    # ---------------------------------------------------------------------
    #  SIMPLE-gate translation – smoke tests
    # ---------------------------------------------------------------------
    q0, q1 = cirq.LineQubit.range(2)

    # (A) single-qubit X and Z on different patches -----------------------
    logical = cirq.Circuit(cirq.X(q0), cirq.Z(q1))

    grid, l2p, idxp, active, smap, flog = fresh_env()
    replay_logical_circuit(grid, logical,
                        logical_to_patch=l2p, idx_in_patch=idxp,
                        active_by_patch=active, state_map=smap, fault_log=flog)
    print(grid.circuit)
    h, cx, rz, swp = strip_encoding(grid.circuit, grid=grid)
    assert (h, cx, rz, swp) == (0, 0, 0, 0), "X / Z should not add H / CX / RZ / SWAP"
    assert not flog

    # (B) single RZ --------------------------------------------------------
    angle   = math.pi / 7
    logical = cirq.Circuit(cirq.rz(angle)(q0))

    grid, l2p, idxp, active, smap, flog = fresh_env()
    replay_logical_circuit(grid, logical,
                        logical_to_patch=l2p, idx_in_patch=idxp,
                        active_by_patch=active, state_map=smap, fault_log=flog)

    h, cx, rz, swp = strip_encoding(grid.circuit, grid=grid)
    # logical RZ → 2 physical CX + 1 RZ
    assert (rz, cx) == (1, 2) and h == 0 and swp == 0
    assert not flog

    # (C) intra-patch logical SWAP ----------------------------------------
    logical = cirq.Circuit(cirq.SWAP(q0, q1))

    grid, l2p, idxp, active, smap, flog = fresh_env()
    replay_logical_circuit(grid, logical,
                        logical_to_patch=l2p, idx_in_patch=idxp,
                        active_by_patch=active, state_map=smap, fault_log=flog)

    h, cx, rz, swp = strip_encoding(grid.circuit, grid=grid)
    # fault-tolerant SWAP = 3 logical CNOTs  ⇒ 3 × 4 = 12 physical CX
    assert (cx, swp) == (12, 0) and h == 0 and rz == 0
    assert not flog

    print("simple-gate translator tests passed ✅")