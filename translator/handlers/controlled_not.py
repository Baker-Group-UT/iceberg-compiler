"""
translator.handlers.cnot

Logical CNOT gate placement logic for Iceberg codes.
Handles both intra-patch and inter-patch scenarios with
fallbacks to SWAP-based transversal correction when possible.
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

def apply_logical_cnot(
    op: cirq.Operation,
    context: TranslationContext,
) -> bool:
    """
    Applies a logical CNOT gate from a Cirq operation.

    Args:
        op: Cirq Operation for logical CNOT.
        context: TranslationContext with circuit and patch state.

    Returns:
        True if placement succeeds, False otherwise.
    """
    q_ctl = op.qubits[0].x
    q_tgt = op.qubits[1].x
    return place_logical_cnot(q_ctl, q_tgt, context)

def place_logical_cnot(
    q_ctl: int,
    q_tgt: int,
    context: TranslationContext,
) -> bool:
    """
    Places a logical CNOT gate under FT rules given control and target indices.

    Args:
        q_ctl: Logical control qubit index.
        q_tgt: Logical target qubit index.
        context: Translation context containing state and patch mapping.

    Returns:
        True if placement succeeded, False otherwise.
    """
    patch_c, patch_t = context.logical_to_patch[q_ctl], context.logical_to_patch[q_tgt]
    pos_c, pos_t     = context.idx_in_patch[q_ctl], context.idx_in_patch[q_tgt]
    iceberg_set = context.iceberg_set

    # Intra-patch CNOT (always valid)
    if patch_c == patch_t:
        iceberg_set.CNOT(q_ctl, q_tgt, fault_tolerant=context.fault_tolerant)
        return True

    # Extract logicals in each patch
    ctrls = context.active_by_patch[patch_c]
    targs = context.active_by_patch[patch_t]

    # Cross-patch constraints
    control_ok = (
        are_all_partners_zero(q_ctl, context.active_by_patch,
                              context.logical_to_patch, context.state_map)
        or is_full_transversal(ctrls, targs, context.idx_in_patch)
    )
    target_alone = is_alone_in_patch(q_tgt, context.active_by_patch, context.logical_to_patch)

    # Aligned for transversal
    if pos_c == pos_t and (control_ok or target_alone):
        iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # Misaligned → try SWAP on target patch if solo
    if pos_c != pos_t and target_alone:
        iceberg_set.SWAP([[patch_t, pos_t, pos_c]], fault_tolerant=context.fault_tolerant)
        # iceberg_set.BARRIER()
        context.idx_in_patch[q_tgt] = pos_c
        verbose_log(context.verbose, f"[swap] target patch {patch_t}: {pos_t} ⇆ {pos_c}")
        iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # Misaligned → try SWAP on control patch if compatible
    if pos_c != pos_t and control_ok:
        iceberg_set.SWAP([[patch_c, pos_c, pos_t]], fault_tolerant=context.fault_tolerant)
        # iceberg_set.BARRIER()
        context.idx_in_patch[q_ctl] = pos_t
        verbose_log(context.verbose, f"[swap] control patch {patch_c}: {pos_c} ⇆ {pos_t}")
        iceberg_set.CNOT_k_Transversal(patch_c, patch_t)
        return True

    # Placement rejected under FT rules
    msg = f"CNOT({q_ctl},{q_tgt}) rejected – violates FT rules"
    context.fault_log.append(msg)
    verbose_log(context.verbose, f"[skip] {msg}")
    return False
