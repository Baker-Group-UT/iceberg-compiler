"""
translator.handlers.swap.py - logic for applying logical and routing swaps.

Two kinds of SWAP reach the translator, and they need opposite implementations:

* a SWAP in the *program* exchanges the values of two program qubits. Exchanging
  their labels achieves that and costs nothing: after the relabel,
  `state_positions` says program qubit a lives on b's lane, which is exactly
  where a's new value (b's old value) already sits. Later gates are placed there
  and `reduce_distribution` reads the right bit back out.

* a *routing* SWAP, inserted by the merging pass to realign lanes before a
  transversal macro, has to physically move the data — that is its whole purpose,
  since the macros pair lane i with lane i. The labels follow the data, so its
  net effect on the program is the identity.

Both update the tracker; they differ only in whether a physical SWAP is emitted.
Conflating them is what broke both callers in turn: emitting the gate for a
program SWAP made the gate a no-op (the data moved and the labels chased it), and
omitting it for a routing SWAP left the macro's lanes misaligned.
"""

from __future__ import annotations

from iceberg.iceberg_set import IcebergSet
from translator.context import TranslationContext
from translator.handlers.utils import verbose_log


def apply_logical_swap(op, context: TranslationContext) -> bool:
    """
    Apply a program SWAP between two logical qubits in the same patch.

    Implemented as a relabel, so no physical gate is emitted.

    Args:
        op: The logical SWAP operation (Qiskit op: typically (inst, qargs, cargs) normalized by dispatcher).
        context: Translation state with patch mappings and layout.

    Returns:
        True if the gate was placed; False otherwise.
    """
    return _swap(op, context, move_data=False, label="SWAP")


def apply_routing_swap(op, context: TranslationContext) -> bool:
    """
    Apply a compiler-inserted routing SWAP: physically move the data.

    Emitted by `translator.merging.transversal` around transversal macros whose
    two patches have crossed rather than aligned lanes.

    Returns:
        True if the gate was placed; False otherwise.
    """
    return _swap(op, context, move_data=True, label="routing SWAP")


def _swap(op, context: TranslationContext, *, move_data: bool, label: str) -> bool:
    # Get logical (program) qubit indices from the program circuit
    q0 = context.qc.find_bit(op.qubits[0]).index
    q1 = context.qc.find_bit(op.qubits[1]).index

    phys0 = context.phys_index_for_logical(q0)
    phys1 = context.phys_index_for_logical(q1)
    patch0, slot0 = context.current_pos_of(phys0)
    patch1, slot1 = context.current_pos_of(phys1)

    if patch0 != patch1:
        msg = f"{label}({q0},{q1}) rejected – cross-patch swap not supported"
        context.fault_log.append(msg)
        verbose_log(context.verbose, f"[skip] {msg}")
        return False

    if slot0 == slot1:
        return True  # nothing to do

    if move_data:
        iceberg_set: IcebergSet = context.iceberg_set
        iceberg_set.SWAP([[patch0, slot0, slot1]], fault_tolerant=context.fault_tolerant)

    context.update_states_on_swap(patch0, slot0, slot1)
    return True
