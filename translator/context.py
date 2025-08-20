"""
translator.context.py – TranslationContext class

Encapsulates all derived state and shared data used during translation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import cirq

from iceberg.iceberg_set import IcebergSet
from .utils import mapping_for_perm

@dataclass
class TranslationContext:
    """
    Holds shared data required by gate translation handlers.
    Construct via `from_config()` or `from_translation()`.
    """
    iceberg_set: IcebergSet
    permutation: List[Tuple[int, int]]
    mapping: dict
    logical_to_patch: Dict[int, int]
    idx_in_patch: Dict[int, int]
    active_by_patch: Dict[int, List[int]]
    state_map: Dict[int, str]
    fault_log: List[str]
    faulty: bool
    verbose: bool

    @classmethod
    def from_config(
        cls,
        permutation: List[Tuple[int, int]],
        program_circuit: cirq.Circuit,
        *,
        faulty: bool = False,
        verbose: bool = False,
    ) -> TranslationContext:
        mapping = mapping_for_perm(permutation, program_circuit)
        iceberg_set = IcebergSet(n_program=mapping["n_prog"], n=4, circuit=cirq.Circuit())
        iceberg_set.encode();
        # iceberg_set.BARRIER()

        return cls(
            iceberg_set=iceberg_set,
            permutation=permutation,
            mapping=mapping,
            logical_to_patch=mapping["logical_to_patch"],
            idx_in_patch=mapping["idx_in_patch"],
            active_by_patch=mapping["active_by_patch"],
            state_map=mapping["state_map"],
            fault_log=[],
            faulty=faulty,
            verbose=verbose,
        )

    @classmethod
    def from_translation(cls, translation: "TranslationContext.Translation") -> TranslationContext:
        """Convenience method for internal construction from a Translation object."""
        return cls.from_config(
            translation.permutation,
            translation.program_circuit,
            faulty=translation.faulty,
            verbose=translation.verbose,
        )
