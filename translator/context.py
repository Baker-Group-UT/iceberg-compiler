"""
translator.context.py – TranslationContext class (Qiskit)

Encapsulates all derived state and shared data used during translation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, TYPE_CHECKING

from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from translator.core import Translation

__all__ = ["TranslationContext"]

from iceberg.iceberg_set import IcebergSet
from .utils import mapping_for_perm


@dataclass
class TranslationContext:
    # --- static mapping (DO NOT MUTATE) ---
    iceberg_set: "IcebergSet"
    permutation: List[Tuple[int, int]]
    mapping: dict
    logical_to_patch: Dict[int, int]      # static: i // k_log
    idx_in_patch: Dict[int, int]          # static: i % k_log
    active_by_patch: Dict[int, List[int]] # static: physical indices active per patch
    state_map: Dict[int, str]
    fault_log: List[str]
    fault_tolerant: bool
    verbose: bool
    encode_override: bool

    # Keep a handle to the *program* circuit being translated (for find_bit, etc.)
    qc: QuantumCircuit

    # --- dynamic, permutation-shaped tracker (MUTABLE) ---
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

    def current_pos_of(self, phys_i: int) -> Tuple[int, int]:
        return (self.current_patch_of(phys_i), self.current_slot_of(phys_i))

    def phys_index_for_logical(self, logical_i: int) -> int:
        p, s = self.permutation[logical_i]
        return p * self.k_log + s

    def update_states_on_swap(self, patch: int, prog_state_i: int, prog_state_j: int) -> None:
        """
        Reflect a SWAP within 'patch' for k_log == 2 by flipping slot 0 <-> 1
        for all tracked states in that patch. The prog_state_i/j args are accepted
        for future generalization but are not required here.
        """
        patch = int(patch)
        k = int(self.k_log)  # assumed 2 for this logic

        # Debug
        if self.verbose:
            print("state positions BEFORE:", self.state_positions)

        # 1) Update state_positions and state_index_by_pos for any state in this patch
        for si, (p, slot) in enumerate(self.state_positions):
            if p == patch:
                old_pos = (p, slot)
                new_pos = (p, 1 - slot)  # flip 0 <-> 1
                # reverse map: remove old, set new
                self.state_index_by_pos.pop(old_pos, None)
                self.state_positions[si] = new_pos
                self.state_index_by_pos[new_pos] = si

        if self.verbose:
            print("state positions AFTER:", self.state_positions)

    @classmethod
    def from_config(
        cls,
        permutation: List[Tuple[int, int]],
        program_circuit: QuantumCircuit,
        *,
        fault_tolerant: bool = False,
        verbose: bool = False,
        encode_override: bool = False,
    ) -> "TranslationContext":
        """Construct a TranslationContext for a given program and permutation."""
        mapping = mapping_for_perm(permutation, program_circuit)
        # Build a fresh physical IcebergSet circuit
        iceberg_qc = QuantumCircuit()
        iceberg_set = IcebergSet(
            n_program=mapping["n_prog"],
            n=4,
            circuit=iceberg_qc,
            encode_override=encode_override
        )
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
            encode_override=encode_override,
            qc=program_circuit,               # <- store program circuit for handlers
            k_log=k_log,
            state_positions=state_positions,
            state_index_by_pos=state_index_by_pos,
            phys_to_state=phys_to_state,
        )

    @classmethod
    def from_translation(cls, translation: "Translation") -> "TranslationContext":
        """Convenience constructor from a Translation object."""
        return cls.from_config(
            translation.permutation,
            translation.program_circuit,
            fault_tolerant=translation.fault_tolerant,
            verbose=translation.verbose,
            encode_override=translation.encode_override,
        )
