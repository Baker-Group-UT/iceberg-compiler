"""Settings for gate-merging heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MergeSettings:
    """Tunable parameters for single-qubit merging heuristics."""

    budget: float = 10.0
    default_cost: float = 1.0
    delay_costs: Dict[str, float] = field(
        default_factory=lambda: {
            "h": 0.05,
            "x": 0.0,
            "z": 0.0,
            "s": 0.01,
            "sdg": 0.01,
            "t": 0.10,
            "tdg": 0.10,
            "cx": 1.0,
            "cz": 1.0,
            "xcx": 1.0,
            "measure": float("inf"),
        }
    )
    zero_delay_costs: bool = True # NOTE: This will set the above vals to 0.0

    def cost_for(self, name: str) -> float:
        if self.zero_delay_costs:
            return 0.0
        return self.delay_costs.get(name, self.default_cost)


DEFAULT_SETTINGS = MergeSettings()
