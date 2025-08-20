"""
simulator/result_types.py

Defines dataclass containers for simulation outputs.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SqaleSimulationResult:
    """
    Output of a single SQALE simulation.

    Attributes:
        raw_fidelity: Bhattacharyya fidelity between ideal and noisy (post-leakage) distributions.
        ideal_dist: Post-leakage ideal distribution.
        noisy_dist: Post-leakage noisy distribution.
    """
    raw_fidelity: float
    ideal_dist: Dict[str, float]
    noisy_dist: Dict[str, float]
    ideal_raw_dist: Dict[str, float]
    noisy_raw_dist: Dict[str, float]
    percent_shots_kept: float = -1
    percent_shots_dropped: float = -1

    def to_dict(self) -> dict:
        return {
            "raw_fidelity": self.raw_fidelity,
            "ideal_dist": self.ideal_dist,
            "noisy_dist": self.noisy_dist,
            "ideal_raw_dist": self.ideal_raw_dist,
            "noisy_raw_dist": self.noisy_raw_dist,
            "percent_shots_kept": self.percent_shots_kept,
            "percent_shots_dropped": self.percent_shots_dropped,
        }
