"""
translator.handlers.h.py – Logical Hadamard (H, HH) handler (Qiskit)

Implements the rules for applying logical Hadamard gates on Iceberg codes.
HH merging is orchestrated by the translator preprocessing layer.
"""

from __future__ import annotations

from translator.context import TranslationContext
from translator.handlers.utils import covers_active_lanes, is_patch_solo


def apply_logical_hadamard(
    op,
    context: TranslationContext,
) -> None:
    """
    Applies a logical Hadamard gate (H) or HH macro on an Iceberg Set.

    If the patch contains only one logical qubit, applies HH fault-tolerantly.
    Otherwise, falls back to non-FT H with fault logging if needed.

    Args:
        op: The logical H operation (Qiskit op normalized by dispatcher; has .gate, .qubits).
        context: Shared translation state.

    Returns:
        None. HH macro insertion is handled by preprocessing passes.
    """
    # Resolve *program* (logical) index, then map through permutation and current swaps
    logical_i = context.qc.find_bit(op.qubits[0]).index
    p, s = context.permutation[logical_i]
    phys_id = p * context.k_log + s
    patch, slot = context.current_pos_of(phys_id)
    global_log_i = patch * context.k_log + slot

    from translator import explore_switches

    if not explore_switches.SOLO_H_TARGETED and is_patch_solo(
        global_log_i, context.active_by_patch, context.logical_to_patch
    ):
        context.iceberg_set.HH([patch])
        context.state_map[global_log_i] = "+"
        if explore_switches.SOLO_GAUGE_PARITY_FIX:
            # Count HH per solo patch so `Translation` can restore the spare's frame.
            counts = context.__dict__.setdefault("solo_hh_count", {})
            counts[patch] = counts.get(patch, 0) + 1
        return None

    if explore_switches.H_AS_S_SX_S:
        # H = e^{-i pi/4} S SX S: three single-lane gadgets, three in-block 2q gates.
        context.iceberg_set.S_at(patch, slot)
        context.iceberg_set.SX_at(patch, slot, fault_tolerant=context.fault_tolerant)
        context.iceberg_set.S_at(patch, slot)
        context.state_map[global_log_i] = "+"
        context.fault_log.append(f"H as S.SX.S on logical {global_log_i} in patch {patch}")
        return None

    # Otherwise: always perform a *single-qubit* logical H at the resolved location
    # context.iceberg_set.H(global_log_i, fault_tolerant=context.fault_tolerant)
    context.iceberg_set.H_at(patch, slot, fault_tolerant=context.fault_tolerant)
    context.state_map[global_log_i] = "+"
    if not context.fault_tolerant:
        context.fault_log.append(f"non-FT H on logical {global_log_i} in patch {patch}")
    return None


def _expand_to_single_hadamards(op, context: TranslationContext, reason: str) -> None:
    """Undo a macro that turned out to be inadmissible, one H per lane."""
    for q in op.qubits:
        faux_op = type("TmpOp", (), {"qubits": [q], "gate": op.gate})
        apply_logical_hadamard(faux_op, context)
    context.fault_log.append(f"HH merge {reason}; fell back to single H gates.")


def apply_logical_hh_merge(
    op,
    context: TranslationContext,
) -> None:
    """
    Apply a pre-merged HH macro covering several lanes of one patch.

    `HH` is transversal H on all n physical qubits plus the anchor swap, so it acts
    on *every* lane. It is therefore only a valid substitution for the program's
    gates when the macro covers each occupied lane of the patch; otherwise the
    uncovered lanes would receive an H the program never asked for. At k=2 a
    two-lane macro is the whole patch and the condition is automatic, which is why
    it went unstated while merging was [[4,2,2]]-only.
    """
    logical_indices = [context.qc.find_bit(q).index for q in op.qubits]
    patch_ids = []
    slots = []
    global_indices = []
    for logical_i in logical_indices:
        p, s = context.permutation[logical_i]
        phys_id = p * context.k_log + s
        patch, slot = context.current_pos_of(phys_id)
        patch_ids.append(patch)
        slots.append(slot)
        global_indices.append(patch * context.k_log + slot)

    if len(set(patch_ids)) != 1:
        _expand_to_single_hadamards(op, context, "applied across multiple patches")
        return None

    patch = patch_ids[0]
    if not covers_active_lanes(context, patch, slots):
        _expand_to_single_hadamards(
            op,
            context,
            f"covers slots {sorted(set(slots))} but patch {patch} is not fully covered",
        )
        return None

    context.iceberg_set.HH([patch])
    for idx in global_indices:
        context.state_map[idx] = "+"
    return None
