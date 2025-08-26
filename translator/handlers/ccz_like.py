"""
translator.gate_handlers.apply_ccz_like

Combined handler for logical CCZ and CCX gates with FT-aware placement.
"""

from __future__ import annotations
from typing import List
import cirq

from translator.context import TranslationContext
from translator.handlers.utils import (
    is_full_transversal,
    verbose_log,
)
from translator.handlers.controlled_z import place_logical_cz

def apply_ccz_like(
    op: cirq.Operation,
    context: TranslationContext,
    mode: str = "CCZ",
) -> bool:
    """
    Applies a logical CCZ or CCX gate using FT-safe transversal placement logic.

    For CCX, the operation is decomposed as H(c) · CCZ(a,b,c) · H(c).

    Args:
        op: cirq.Operation representing a 3-qubit CCZ or CCX.
        context: TranslationContext providing the iceberg_set and placement mappings.
        mode: Either "CCZ" or "CCX".

    Returns:
        True if the operation was placed successfully; False otherwise.
    """
    if mode not in ("CCZ", "CCX"):
        raise ValueError("mode must be 'CCZ' or 'CCX'")

    a, b, c = [q.x for q in op.qubits]
    iceberg_set = context.iceberg_set

    def align_pair(p: int, c_: int, label: str) -> bool:
        """Ensures (p, c_) can be aligned for transversal operation."""
        patch_p, patch_c = context.logical_to_patch[p], context.logical_to_patch[c_]
        pos_p, pos_c = context.idx_in_patch[p], context.idx_in_patch[c_]
        actives_p = context.active_by_patch[patch_p]
        actives_c = context.active_by_patch[patch_c]

        partners_p_zero = all(context.state_map[q] == "0" for q in actives_p if q != p)
        full_transversal = is_full_transversal(actives_p, actives_c, context.idx_in_patch)
        p_ok = partners_p_zero or full_transversal
        c_alone = len(actives_c) == 1

        if pos_p == pos_c and (p_ok or c_alone):
            return True

        if pos_p != pos_c and c_alone:
            iceberg_set.SWAP([[patch_c, pos_c, pos_p]], fault_tolerant=context.fault_tolerant)
            # iceberg_set.BARRIER()
            context.idx_in_patch[c_] = pos_p
            verbose_log(context.verbose, f"[swap] {label}: target patch {patch_c} {pos_c} ⇆ {pos_p}")
            return True

        if pos_p != pos_c and p_ok:
            iceberg_set.SWAP([[patch_p, pos_p, pos_c]], fault_tolerant=context.fault_tolerant)
            # iceberg_set.BARRIER()
            context.idx_in_patch[p] = pos_c
            verbose_log(context.verbose, f"[swap] {label}: control patch {patch_p} {pos_p} ⇆ {pos_c}")
            return True

        return False

    if mode == "CCX":
        iceberg_set.HH([c]) if hasattr(iceberg_set, "HH") else iceberg_set.H(c)
        # iceberg_set.BARRIER()

    if not (align_pair(a, c, "align (a,c)") and align_pair(b, c, "align (b,c)")):
        context.fault_log.append(f"{mode}({a},{b};{c}) rejected – cannot align under FT rules")
        verbose_log(context.verbose, f"[skip] {mode}({a},{b};{c}) alignment failed")
        if mode == "CCX":
            iceberg_set.HH([c]) if hasattr(iceberg_set, "HH") else iceberg_set.H(c)
            # iceberg_set.BARRIER()
        return False

    try:
        if hasattr(iceberg_set, "CCZ"):
            iceberg_set.CCZ(a, b, c)
        elif hasattr(iceberg_set, "CCZ_k_Transversal"):
            pa, pb, pc = context.logical_to_patch[a], context.logical_to_patch[b], context.logical_to_patch[c]
            iceberg_set.CCZ_k_Transversal(pa, pb, pc)
        else:
            ok1 = place_logical_cz(a, c, context)
            ok2 = place_logical_cz(b, c, context)
            ok3 = place_logical_cz(a, b, context)
            if not (ok1 and ok2 and ok3):
                raise RuntimeError("Fallback CZ decomposition failed")
    except Exception as e:
        context.fault_log.append(f"{mode}({a},{b};{c}) emit failed: {e}")
        verbose_log(context.verbose, f"[error] emitting {mode}: {e}")
        if mode == "CCX":
            iceberg_set.HH([c]) if hasattr(iceberg_set, "HH") else iceberg_set.H(c)
            # iceberg_set.BARRIER()
        return False

    if mode == "CCX":
        iceberg_set.HH([c]) if hasattr(iceberg_set, "HH") else iceberg_set.H(c)
        # iceberg_set.BARRIER()

    return True
