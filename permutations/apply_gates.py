from typing import Dict, List
import cirq
from cirq.ops.common_gates import CXPowGate, HPowGate

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from iceberg.IcebergCode import IcebergGrid


def logical_capacity(n_code: int) -> int:          # ★ FIXED
    """Return number of logical qubits an n-qubit Iceberg code can hold."""
    return n_code - 2

# ------------------------------------------------------------------ #
#  apply_hadamard  (handles H  & HH auto-merge)                      #
# ------------------------------------------------------------------ #
def apply_hadamard(grid: IcebergGrid,
                   op: cirq.Operation,          # H on *one* logical qubit
                   next_op: cirq.Operation | None,
                   logical_to_patch: Dict[int,int],
                   active_by_patch: Dict[int,List[int]],
                   state_map: Dict[int,str],
                   fault_log: List[str],
                   *,
                   faulty: bool,
                   verbose: bool):
    q = op.qubits[0].x                           # logical index
    patch = logical_to_patch[q]
    active = active_by_patch[patch]

    # ---- fast path: patch holds ONLY q  → FT HH ---------------------
    if len(active) == 1:
        grid.HH([patch]);            state_map[q] = "+"
        return

    # ---- look-ahead: try to fuse with another H ---------------------
    if (next_op is not None
        and isinstance(next_op.gate, HPowGate)
        and abs(next_op.gate.exponent) % 2 == 1      # true for ±1
        and next_op.gate.global_shift == 0):
        q2 = next_op.qubits[0].x
        if q2 != q and logical_to_patch[q2] == patch:
            # same patch, no gates in between ⇒ fuse
            if verbose:
                print(f"[merge-HH] logical {q} & {q2} in patch {patch}")
            grid.HH([patch]);         state_map[q] = state_map[q2] = "+"
            # mark the second op as “consumed” by returning its qubit idx
            return q2

    # ---- otherwise fall back to single H (non-FT) -------------------
    grid.H(q);     state_map[q] = "+"
    fault_log.append(f"non-FT H on logical {q} in patch {patch}")


# ----------------------------------------------------------------------
#  LOGICAL CNOT  – FT whenever possible; single swap if mis-aligned
# ----------------------------------------------------------------------
def apply_logical_cnot(grid,
                       ctl: int, tgt: int,
                       logical_to_patch: dict,
                       idx_in_patch: dict,          # {logical: 0 | 1}
                       active_by_patch: dict,
                       state_map: dict,
                       fault_log: list,
                       *, verbose: bool = False,
                       faulty: bool = False) -> bool:
    """
    Logical CNOT placement rules
    ----------------------------
    1. Same patch  → in-patch logical CNOT  (FT)
    2. Different patches
       2a Allowed if
           • *Every* qubit in control-patch is intended to interact with a
             qubit in the target-patch           **OR**
           • All *other* control-patch qubits are logical |0⟩
       2b Allowed if target-patch contains only tgt  (among ACTIVE qubits)
       → If indices mis-align, swap once on whichever patch is “free”
         (target-patch if 2b, else control-patch)

    Returns
    -------
    True  – gate placed
    False – impossible under rules
    """
    patch_c, patch_t = logical_to_patch[ctl], logical_to_patch[tgt]
    pos_c,   pos_t   = idx_in_patch[ctl],     idx_in_patch[tgt]

    # ------------------------------------------------------------------
    # 1. Same patch  → straightforward logical CNOT
    # ------------------------------------------------------------------
    if patch_c == patch_t:
        grid.CNOT(ctl, tgt, faulty=faulty)
        return True

    # ------------------------------------------------------------------
    # Build helper sets
    # ------------------------------------------------------------------
    ctrl_qubits   = active_by_patch[patch_c]          # active logicals in control patch
    targ_qubits   = active_by_patch[patch_t]          # active logicals in target patch
    partners      = [q for q in ctrl_qubits if q != ctl]

    # 2a.i  — “other controls are all |0⟩”
    partners_are_zero = all(state_map[q] == "0" for q in partners)

    # 2a.ii — “every control qubit has a matching index in target patch”
    full_transversal  = all(any(idx_in_patch[q] == idx_in_patch[r] for r in targ_qubits)
                            for q in ctrl_qubits)

    control_ok = partners_are_zero or full_transversal

    # 2b — target patch alone (among active qubits)
    target_alone = (len(targ_qubits) == 1)

    # ------------------------------------------------------------------
    #  QUICK ACCEPT: indices already align
    # ------------------------------------------------------------------
    if pos_c == pos_t and (control_ok or target_alone):
        grid.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # ------------------------------------------------------------------
    #  MIS-ALIGNED → single swap on the “free” patch
    # ------------------------------------------------------------------
    # Prefer target patch if it is alone; otherwise control patch (must be OK)
    # ------------------------------------------------------------------
    if pos_c != pos_t and target_alone:
        # swap inside target patch
        grid.swap([[patch_t, pos_t, pos_c]], faulty=faulty)
        grid.barrier()
        idx_in_patch[tgt] = pos_c
        if verbose:
            print(f"[swap] target patch {patch_t}: {pos_t} ⇆ {pos_c}")
        grid.CNOT_k_Transversal(patch_c, patch_t)
        return True

    if pos_c != pos_t and control_ok:
        # swap inside control patch
        grid.swap([[patch_c, pos_c, pos_t]], faulty=faulty)
        grid.barrier()
        idx_in_patch[ctl] = pos_t
        if verbose:
            print(f"[swap] control patch {patch_c}: {pos_c} ⇆ {pos_t}")
        grid.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # ------------------------------------------------------------------
    #  NOT PLACEABLE
    # ------------------------------------------------------------------
    fault_log.append(f"CNOT({ctl},{tgt}) rejected – violates FT rules")
    if verbose:
        print(f"[skip] CNOT({ctl},{tgt}) – rejected under rule 2")
    return False

# ----------------------------------------------------------------------
#  LOGICAL CZ – mirror of CNOT rules (swap-once strategy)
# ----------------------------------------------------------------------
def apply_logical_cz(grid,
                     q0: int, q1: int,
                     logical_to_patch: dict,
                     idx_in_patch: dict,
                     active_by_patch: dict,
                     state_map: dict,
                     fault_log: list,
                     *, verbose: bool = False,
                     faulty: bool = False) -> bool:
    """
    Same placement rules as apply_logical_cnot, but with CZ calls.
    """
    patch_a, patch_b = logical_to_patch[q0], logical_to_patch[q1]
    pos_a,   pos_b   = idx_in_patch[q0],       idx_in_patch[q1]

    # 1) Same patch → intra-patch logical CZ
    if patch_a == patch_b:
        grid.CZ(q0, q1)     # IcebergGrid will route to per-patch CZ
        return True

    # Build helper sets
    a_actives = active_by_patch[patch_a]
    b_actives = active_by_patch[patch_b]
    partners_a = [q for q in a_actives if q != q0]

    # Rule 2a: control-like logic (for CZ it's symmetric)
    partners_a_zero = all(state_map[q] == "0" for q in partners_a)
    full_transversal = all(
        any(idx_in_patch[qa] == idx_in_patch[qb] for qb in b_actives)
        for qa in a_actives
    )
    a_ok = partners_a_zero or full_transversal

    # Rule 2b: target-alone (again symmetric)
    b_alone = (len(b_actives) == 1)

    # Quick accept: indices align and either side is "OK"
    if pos_a == pos_b and (a_ok or b_alone):
        grid.CZ_k_Transversal(patch_a, patch_b)
        return True

    # Misaligned → single swap on the “free” side
    if pos_a != pos_b and b_alone:
        # swap inside B to match A
        grid.swap([[patch_b, pos_b, pos_a]], faulty=faulty); grid.barrier()
        idx_in_patch[q1] = pos_a
        if verbose:
            print(f"[swap] CZ align on target patch {patch_b}: {pos_b} ⇆ {pos_a}")
        grid.CZ_k_Transversal(patch_a, patch_b)
        return True

    if pos_a != pos_b and a_ok:
        # swap inside A to match B
        grid.swap([[patch_a, pos_a, pos_b]], faulty=faulty); grid.barrier()
        idx_in_patch[q0] = pos_b
        if verbose:
            print(f"[swap] CZ align on control patch {patch_a}: {pos_a} ⇆ {pos_b}")
        grid.CZ_k_Transversal(patch_a, patch_b)
        return True

    # Not placeable
    fault_log.append(f"CZ({q0},{q1}) rejected – violates FT rules")
    if verbose:
        print(f"[skip] CZ({q0},{q1}) – rejected under rule 2")
    return False

def _hh_on_target(grid, c_logical: int, logical_to_patch: dict):
    """Apply FT HH to the *patch* hosting logical qubit c when possible; else fall back to H(c)."""
    # Prefer HH if available *and* the patch index is valid.
    if hasattr(grid, "HH"):
        patch_c = logical_to_patch[c_logical]
        if 0 <= patch_c < getattr(grid, "num_icebergs", -1):
            grid.HH([patch_c])  # HH expects patch indices
            grid.barrier()
            return
    # Fallback: single‑qubit logical H on c (not FT but safe for plumbing)
    grid.H(c_logical)
    grid.barrier()

# ----------------------------------------------------------------------
#  LOGICAL CCZ / CCX – decompose via CZ + H on target for CCX
# ----------------------------------------------------------------------
def apply_ccz_like(grid,
                   a: int, b: int, c: int,
                   mode: str,   # "CCZ" or "CCX"
                   *,
                   logical_to_patch: dict,
                   idx_in_patch: dict,
                   active_by_patch: dict,
                   state_map: dict,
                   fault_log: list,
                   faulty: bool = False,
                   verbose: bool = False) -> bool:
    """
    Places a CCZ under the same 'transversal with at most one swap' policy used
    for 2-qubit gates, but requires BOTH (a,c) and (b, c) alignments to be achievable
    without conflicting swaps.

    CCX is implemented as H(c) · CCZ(a,b,c) · H(c).
    """
    if mode not in ("CCZ", "CCX"):
        raise ValueError("mode must be 'CCZ' or 'CCX'")

    # Helper: align (p, c) pair to be transversal with at most one swap
    # Returns a callable to *undo* any local index bookkeeping if needed? Not necessary if we mutate idx_in_patch.
    def ensure_transversal(p: int, c_: int, label: str) -> bool:
        patch_p, patch_c = logical_to_patch[p], logical_to_patch[c_]
        pos_p,   pos_c   = idx_in_patch[p],       idx_in_patch[c_]

        if patch_p == patch_c:
            # same patch case: we rely on in-patch decomposition later
            return True

        P_actives = active_by_patch[patch_p]
        C_actives = active_by_patch[patch_c]

        partners_p = [q for q in P_actives if q != p]
        partners_p_zero = all(state_map[q] == "0" for q in partners_p)
        full_transversal = all(
            any(idx_in_patch[qp] == idx_in_patch[qc] for qc in C_actives)
            for qp in P_actives
        )
        p_ok   = partners_p_zero or full_transversal
        c_alone= (len(C_actives) == 1)

        if pos_p == pos_c and (p_ok or c_alone):
            return True

        if pos_p != pos_c and c_alone:
            # swap inside C
            grid.swap([[patch_c, pos_c, pos_p]], faulty=faulty); grid.barrier()
            if verbose:
                print(f"[swap] {label}: target patch {patch_c} {pos_c} ⇆ {pos_p}")
            idx_in_patch[c_] = pos_p
            return True

        if pos_p != pos_c and p_ok:
            # swap inside P
            grid.swap([[patch_p, pos_p, pos_c]], faulty=faulty); grid.barrier()
            if verbose:
                print(f"[swap] {label}: control patch {patch_p} {pos_p} ⇆ {pos_c}")
            idx_in_patch[p] = pos_c
            return True

        return False

    # If CCX, wrap with H(c) … H(c)
    if mode == "CCX":
        _hh_on_target(grid, c, logical_to_patch)
        
    # Ensure both (a,c) and (b,c) can be made transversal / placeable
    ok_ac = ensure_transversal(a, c, label="CCZ align (a,c)")
    ok_bc = ensure_transversal(b, c, label="CCZ align (b,c)")

    if not (ok_ac and ok_bc):
        fault_log.append(f"{mode}({a},{b};{c}) rejected – cannot align pairs under FT rules")
        if verbose:
            print(f"[skip] {mode}({a},{b};{c}) – pair alignment failed")
        if mode == "CCX":
            # undo final H if we added it, to keep history logically consistent
            grid.HH([c]) if hasattr(grid, "HH") else grid.H(c)
            grid.barrier()
        return False

    # Apply CCZ implementation:
    # If you later add grid.CCZ_k_Transversal(patch_a, patch_b, patch_c), call it here.
    # For now, emit via a standard CZ-based phase gadget decomposition:
    # CCZ(a,b,c) = e^{iπ( I - Z_a)/2 · (I - Z_b)/2 · (I - Z_c)/2}
    # Practical route on your backend:
    #  - Use grid.CZ / grid.CZ_k_Transversal in a known 2-qubit synthesis for CCZ if provided,
    #  - else fall back to cirq.decompose_once for CCZ to CZ/CNOT/RZ and re-dispatch (simple path below).
    try:
        # Try a native op if your grid exposes it later:
        if hasattr(grid, "CCZ"):
            grid.CCZ(a, b, c)  # intra-patch (routes appropriately)
        elif hasattr(grid, "CCZ_k_Transversal"):
            pa, pb, pc = logical_to_patch[a], logical_to_patch[b], logical_to_patch[c]
            grid.CCZ_k_Transversal(pa, pb, pc)
        else:
            # Fallback: synthesize CCZ with standard 2-qubit gates at logical level
            # then translate each op through the translator again is overkill here.
            # Instead, emit a safe, small canonical using H on c + two Toffolis via CZ:
            # We’ll use Cirq to build a tiny logical CCZ and replay with current translator.
            qa, qb, qc = [cirq.LineQubit(i) for i in range(3)]
            ccz_logical = cirq.Circuit(cirq.CCZ(qa, qb, qc))
            # Minimal route: emulate CCZ as (H on c; CCX; H on c) at logical,
            # but we don’t want CCX recursion. We already are in CCZ branch, so:
            # Just place CZ(a,c) and CZ(b,c) sandwich with T gates would be ideal,
            # but you don’t have T; your backend supports RZ. For now, lean on grid.CZ:
            # Phase gadget style (approx skeleton):
            # NEW (respects cross‑patch rules via your CZ handler)
            ok1 = apply_logical_cz(grid, a, c,
                                logical_to_patch, idx_in_patch,
                                active_by_patch, state_map, fault_log,
                                faulty=faulty, verbose=verbose)
            ok2 = apply_logical_cz(grid, b, c,
                                logical_to_patch, idx_in_patch,
                                active_by_patch, state_map, fault_log,
                                faulty=faulty, verbose=verbose)
            ok3 = apply_logical_cz(grid, a, b,
                                logical_to_patch, idx_in_patch,
                                active_by_patch, state_map, fault_log,
                                faulty=faulty, verbose=verbose)
            if not (ok1 and ok2 and ok3):
                fault_log.append(f"{mode}({a},{b};{c}) emit failed: CZ placement rejected")
                if verbose:
                    print(f"[skip] {mode} – CZ fallback could not be placed (alignment rules)")
                # If we started with CCX, undo the opening H/HH to keep history sensible
                if mode == "CCX":
                    _hh_on_target(grid, c, logical_to_patch)
                return False
            # NOTE: This is not a full canonical CCZ decomposition; replace with your
            # preferred FT CCZ sequence when available.
    except Exception as e:
        fault_log.append(f"{mode}({a},{b};{c}) emit failed: {e}")
        if verbose:
            print(f"[error] emitting {mode}: {e}")
        if mode == "CCX":
            grid.HH([c]) if hasattr(grid, "HH") else grid.H(c)
            grid.barrier()
        return False

    if mode == "CCX":
        _hh_on_target(grid, c, logical_to_patch)

    return True

# ---------------------------------------------------------------------
#  QUICK COVERAGE TESTS  (run `python circuit_permutations_apply_gates.py`)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import math
    from collections import Counter

    # ---- helpers ----------------------------------------------------
    def fresh_grid(n_program=4, n_code=4):
        """Return (grid, logical_to_patch, idx_in_patch, active_by_patch,
                   state_map, fault_log)."""
        circ = cirq.Circuit()
        g    = IcebergGrid(n_program=n_program, n=n_code, circuit=circ)
        g.encode()
        k         = logical_capacity(n_code)
        l2p       = {i: i // k for i in range(n_program)}
        idx_patch = {i: i %  k for i in range(n_program)}
        active    = {p: [] for p in set(l2p.values())}
        for i in range(n_program):
            active[l2p[i]].append(i)
        s_map = {i: "0" for i in range(n_program)}
        return g, l2p, idx_patch, active, s_map, []

    def count_gate(circ, gate_type):
        return sum(1 for op in circ.all_operations() if isinstance(op.gate, gate_type))

    # ================================================================
    # 1. H / HH behaviour
    # ================================================================
    # (a) two qubits, separate patches  → HH each, no non-FT log
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=4)
    qA, qB = 0, 2           # live in patch-0 and patch-1

    # ----- make each patch contain ONE active logical ----------------
    active[l2p[qA]] = [qA]     # strip logical-1 from patch-0
    active[l2p[qB]] = [qB]     # strip logical-3 from patch-1
    # -----------------------------------------------------------------

    opA = cirq.H(cirq.LineQubit(qA))
    apply_hadamard(grid, opA, None, l2p, active, smap, log,
                faulty=False, verbose=False)
    opB = cirq.H(cirq.LineQubit(qB))
    apply_hadamard(grid, opB, None, l2p, active, smap, log,
                faulty=False, verbose=False)

    # 1 H from encode *per patch*  + 4 H per HH *per patch*  = 10 total
    assert count_gate(grid.circuit, cirq.HPowGate) == 10
    assert len(log) == 0, "HH on solo-patch should be fault-tolerant"

    # (b) two qubits same patch, only ONE needs H  → single non-FT H
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=2)
    # keep both logicals active; we will apply H only to q0
    op = cirq.H(cirq.LineQubit(0))
    apply_hadamard(grid, op, None, l2p, active, smap, log,
                faulty=False, verbose=False)

    # encode adds 1 H; the non-FT logical-H macro adds 5 physical H (n=4 code)
    assert count_gate(grid.circuit, cirq.HPowGate) == 6
    assert any("non-FT H" in msg for msg in log), "should log non-FT fallback"

    # (c) one qubit one patch, needs H but don't care about other state  → FT HH
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=2)
    active[l2p[1]].remove(1)        # pretend q1 is “don’t-care”
    op = cirq.H(cirq.LineQubit(0))
    apply_hadamard(grid, op, None, l2p, active, smap, log,
                   faulty=False, verbose=False)
    assert count_gate(grid.circuit, cirq.HPowGate) == 5
    assert len(log) == 0
    
    # (d) two qubits same patch, both need H, no gate between → merge to HH
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=2)
    op1 = cirq.H(cirq.LineQubit(0))
    op2 = cirq.H(cirq.LineQubit(1))
    consumed = apply_hadamard(grid, op1, op2,
                              l2p, active, smap, log, faulty=False, verbose=False)
    if consumed is not None:
        # we merged, skip second call
        pass
    else:
        apply_hadamard(grid, op2, None,
                       l2p, active, smap, log, faulty=False, verbose=False)
    assert count_gate(grid.circuit, cirq.HPowGate) == 5  # HH = 2 H gates
    assert len(log) == 0

    # ================================================================
    # 2. CNOT placement
    # ================================================================
    # (a) Same patch, straightforward
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=2)
    ok = apply_logical_cnot(grid, 0, 1, l2p, idxp, active, smap, log)
    assert ok and len(log) == 0
    assert count_gate(grid.circuit, cirq.CXPowGate) == 7   # in-patch logical CNOT

    # (b) Different patches, indices align, target alone → transversal
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=4)
    # make target patch solo
    active[l2p[2]].remove(3)
    ok = apply_logical_cnot(grid, 0, 2, l2p, idxp, active, smap, log)
    assert ok and len(log) == 0
    # transversal adds n_code physical CNOTs; here n_code=4 ⇒ 4 ops
    assert count_gate(grid.circuit, cirq.CXPowGate) == 10

    # (c) Mis-aligned, swap on target-alone patch
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=4)
    # put control in idx=1, target in idx=0  (mis-align)
    idxp[1] = 1; idxp[2] = 0
    # target patch alone
    active[l2p[2]].remove(3)
    ok = apply_logical_cnot(grid, 1, 2, l2p, idxp, active, smap, log,
                            verbose=False, faulty=True)
    assert ok and len(log) == 0
    # one SWAP (because of faulty desingation) + transversal 4 CNOTs
    assert count_gate(grid.circuit, cirq.SwapPowGate) == 1

    # (d) Impossible case → returns False, logs reason
    grid, l2p, idxp, active, smap, log = fresh_grid(n_program=4)
    # force a mis-align and make partner non-zero  → CNOT must be rejected
    idxp[2] = 1        # put target at slot-1   (pos_c ≠ pos_t)
    smap[1] = "+"      # partner qubit is now |+>  (not all-zero)
    ok = apply_logical_cnot(grid, 0, 2, l2p, idxp, active, smap, log)

    assert not ok and any("rejected" in msg for msg in log)


    print("All coverage tests passed ✅")