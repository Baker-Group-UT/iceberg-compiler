"""
result_types.py - Postprocessed result container with Iceberg postselection.
"""
# pylint: disable=[C0116]

from dataclasses import dataclass
from typing import Dict, Any

import cirq

from simulator.result_types import SqaleSimulationResult


@dataclass
class PostselectedSimulationResult:
    """
    Final enriched result from simulation + Iceberg postselection.
    """
    permutation: Any                    # typically List[Tuple[int, int]]
    circuit: cirq.Circuit               # translated physical circuit
    raw_result: SqaleSimulationResult   # raw simulator output

    postselected_noisy_dist: Dict[str, float]
    postselected_ideal_dist: Dict[str, float]
    was_postselected: bool = True

    logical_fidelity: float = 0.0
    logical_selection_rate: float = 0.0
    # This is the % of shots kept after physical postselection multipled by
    # the % of shots kept after logical postselection
    selection_rate_overall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "permutation": self.permutation,
            "circuit": cirq.to_json(self.circuit),
            "raw_result": self.raw_result.to_dict() if hasattr(self.raw_result, "to_dict") else self.raw_result,
            "postselected_noisy_dist": self.postselected_noisy_dist,
            "postselected_ideal_dist": self.postselected_ideal_dist,
            "was_postselected": self.was_postselected,
            "logical_fidelity": self.logical_fidelity,
            "logical_selection_rate": self.logical_selection_rate,
            "selection_rate_overall": self.selection_rate_overall,
        }
