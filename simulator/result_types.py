"""
simulator/result_types.py

Defines dataclass containers for simulation outputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SimulationResult:
    """
    Output of a single SQALE simulation.

    Attributes:
        raw_fidelity: Bhattacharyya fidelity between ideal and noisy (post-leakage) distributions.
        ideal_dist: Post-leakage ideal distribution.
        noisy_dist: Post-leakage noisy distribution.
    """

    key_order: List[str]
    key_widths: List[int]
    raw_fidelity: Optional[float] = None
    ideal_dist: Optional[Dict[str, float]] = None
    noisy_dist: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "raw_fidelity": self.raw_fidelity,
            "ideal_dist": self.ideal_dist,
            "noisy_dist": self.noisy_dist,
            "key_order": self.key_order,
            "key_widths": self.key_widths,
        }
