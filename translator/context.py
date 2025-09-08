"""
translator.context.py – TranslationContext class

Encapsulates all derived state and shared data used during translation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, TYPE_CHECKING
import cirq

if TYPE_CHECKING:
    from translator.core import Translation

__all__ = ["TranslationContext"]

from iceberg.iceberg_set import IcebergSet
from .utils import mapping_for_perm

@dataclass
class TranslationContext:
    # --- existing static mapping (DO NOT MUTATE) ---
    iceberg_set: "IcebergSet"
    permutation: List[Tuple[int, int]]
    mapping: dict
    logical_to_patch: Dict[int, int]     # static: i // k_log
    idx_in_patch: Dict[int, int]         # static: i % k_log
    active_by_patch: Dict[int, List[int]]# static: physical indices active per patch
    state_map: Dict[int, str]
    fault_log: List[str]
    fault_tolerant: bool
    verbose: bool

    # --- NEW: dynamic, permutation-shaped tracker (MUTABLE) ---
    k_log: int = 2
    # states we care about, in *permutation order*; each is (patch, slot)
    state_positions: List[Tuple[int, int]] = field(default_factory=list)
    # reverse index: current (patch,slot) -> state index in state_positions
    state_index_by_pos: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # physical index -> state index (for quick lookups by phys qubit ID)
    phys_to_state: Dict[int, int] = field(default_factory=dict)

    # ---------- convenience “current view” helpers ----------
    def current_patch_of(self, phys_i: int) -> int:
        si = self.phys_to_state.get(phys_i)
        if si is not None:
            return self.state_positions[si][0]
        return self.logical_to_patch[phys_i]

    def current_slot_of(self, phys_i: int) -> int:
        si = self.phys_to_state.get(phys_i)
        if si is not None:
            return self.state_positions[si][1]
        return self.idx_in_patch[phys_i]

    def current_pos_of(self, phys_i: int) -> Tuple[int,int]:
        return (self.current_patch_of(phys_i), self.current_slot_of(phys_i))

    def update_states_on_swap(self, patch: int, i: int, j: int) -> None:
        """Reflect a SWAP of slots i<->j inside 'patch' for tracked states.

        Updates only the dynamic trackers:
        - state_positions (list of (patch, slot) per tracked state)
        - state_index_by_pos: (patch, slot) -> state index

        NOTE: phys_to_state remains a stable mapping from the *program* physical
        index to the state index; do not mutate it during swaps.
        """
        if i == j:
            return

        a = (patch, i)
        b = (patch, j)

        # Which tracked states (if any) occupy these positions?
        si = self.state_index_by_pos.get(a)
        sj = self.state_index_by_pos.get(b)

        # Move the state at 'a' to 'b'
        if si is not None:
            self.state_index_by_pos.pop(a, None)
            self.state_positions[si] = b
            self.state_index_by_pos[b] = si

        # Move the state at 'b' to 'a'
        if sj is not None:
            self.state_index_by_pos.pop(b, None)
            self.state_positions[sj] = a
            self.state_index_by_pos[a] = sj

    @classmethod
    def from_config(
        cls,
        permutation: List[Tuple[int, int]],
        program_circuit: cirq.Circuit,
        *,
        fault_tolerant: bool = False,
        verbose: bool = False,
    ) -> TranslationContext:
        """Method for creating a TranslationContext"""
        mapping = mapping_for_perm(permutation, program_circuit)
        iceberg_set = IcebergSet(n_program=mapping["n_prog"], n=4, circuit=cirq.Circuit())
        iceberg_set.encode()

        k_log = mapping["k_log"]

        # Dynamic tracker starts as the permutation itself (state order = permutation order)
        state_positions = list(permutation)  # [(patch, slot), ...]
        state_index_by_pos = {pos: i for i, pos in enumerate(state_positions)}

        # Physical indices corresponding to tracked states: phys = patch * k_log + slot
        tracked_phys = [p * k_log + s for (p, s) in state_positions]
        phys_to_state = {phys: i for i, phys in enumerate(tracked_phys)}

        return cls(
            iceberg_set=iceberg_set,
            permutation=permutation,
            mapping=mapping,
            logical_to_patch=mapping["logical_to_patch"],
            idx_in_patch=mapping["idx_in_patch"],
            active_by_patch=mapping["active_by_patch"],
            state_map=mapping["state_map"],
            fault_log=[],
            fault_tolerant=fault_tolerant,
            verbose=verbose,
            k_log=k_log,
            state_positions=state_positions,
            state_index_by_pos=state_index_by_pos,
            phys_to_state=phys_to_state,
        )

    @classmethod
    def from_translation(cls, translation: "Translation") -> "TranslationContext":
        """Convenience method for internal construction from a Translation object."""
        return cls.from_config(
            translation.permutation,
            translation.program_circuit,
            fault_tolerant=translation.fault_tolerant,
            verbose=translation.verbose,
        )
