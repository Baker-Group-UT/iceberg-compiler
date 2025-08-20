"""
translator.handlers.cz_handler

Handler for logical CZ gate placement on Iceberg circuits.
Follows fault-tolerant (FT) placement rules:

1. If both qubits are in the same patch → standard in-patch CZ.
2. If in different patches:
   a. Allowed if either:
      • All partners in control patch are |0⟩, OR
      • Full transversal compatibility with target patch
   b. Also allowed if the target is the only active qubit in its patch
   → If indices misalign, perform SWAP on the free patch to align.

Falls back with logging if no valid FT placement exists.
"""

from __future__ import annotations
from typing import Dict, List
import cirq

from iceberg.iceberg_set import IcebergSet
from translator.context import TranslationContext
from translator.handlers.utils import (
    are_all_partners_zero,
    is_full_transversal,
    is_alone_in_patch,
    verbose_log,
)

def apply_logical_cz(
    op: cirq.Operation,
    context: TranslationContext,
) -> bool:
    """
    Attempts to place a logical CZ gate under FT rules.

    Args:
        op: The logical CZ operation.
        context: TranslationContext with iceberg_set, patch, and state info.

    Returns:
        True if placement succeeds, False if gate must be skipped.
    """
    q_ctl = op.qubits[0].x
    q_tgt = op.qubits[1].x
    return place_logical_cz(q_ctl, q_tgt, context)

def place_logical_cz(
    q_ctl: int,
    q_tgt: int,
    context: TranslationContext,
) -> bool:
    """
    Core logic for placing a logical CZ gate under FT rules.
    Can be used independently from apply_logical_cz.

    Args:
        q0: Control logical qubit index.
        q1: Target logical qubit index.
        context: TranslationContext containing placement metadata.

    Returns:
        True if the CZ gate was placed, False otherwise.
    """
    patch_a = context.logical_to_patch[q_ctl]
    patch_b = context.logical_to_patch[q_tgt]
    pos_a   = context.idx_in_patch[q_ctl]
    pos_b   = context.idx_in_patch[q_tgt]

    iceberg_set = context.iceberg_set

    # Intra-patch → standard CZ
    if patch_a == patch_b:
        iceberg_set.CZ(q_ctl, q_tgt)
        return True

    a_actives = context.active_by_patch[patch_a]
    b_actives = context.active_by_patch[patch_b]

    # Check FT eligibility
    a_ok = (
        are_all_partners_zero(q_ctl, context.active_by_patch, context.logical_to_patch, context.state_map)
        or is_full_transversal(a_actives, b_actives, context.idx_in_patch)
    )
    b_alone = is_alone_in_patch(q_tgt, context.active_by_patch, context.logical_to_patch)

    # Already aligned → FT transversal
    if pos_a == pos_b and (a_ok or b_alone):
        iceberg_set.CZ_k_Transversal(patch_a, patch_b)
        return True

    # SWAP to align if valid
    if pos_a != pos_b and b_alone:
        iceberg_set.SWAP([[patch_b, pos_b, pos_a]], faulty=context.faulty)
        # iceberg_set.BARRIER()
        context.idx_in_patch[q_tgt] = pos_a
        verbose_log(context.verbose, f"[swap] CZ align on target patch {patch_b}: {pos_b} ⇆ {pos_a}")
        iceberg_set.CZ_k_Transversal(patch_a, patch_b)
        return True

    if pos_a != pos_b and a_ok:
        iceberg_set.SWAP([[patch_a, pos_a, pos_b]], faulty=context.faulty)
        # iceberg_set.BARRIER()
        context.idx_in_patch[q_ctl] = pos_b
        verbose_log(context.verbose, f"[swap] CZ align on control patch {patch_a}: {pos_a} ⇆ {pos_b}")
        iceberg_set.CZ_k_Transversal(patch_a, patch_b)
        return True

    # Rejected
    msg = f"CZ({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False
