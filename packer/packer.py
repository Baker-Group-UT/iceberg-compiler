# packer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Any

import qiskit  # adjust import if needed

from .phase_one_graph import build_graph, GraphHeuristics
from .phase_two_balance import balance_graph

GateName = str
LSRTable = Dict[GateName, float]

DEFAULT_LSR_TABLE = {
    "X":   0.99763,
    "Z":   0.99763,
    "H":   0.95871,
    "HH":  0.98625,
    "CX":  0.96106,
    "CXT": 0.96106,
    "CZ":  0.96670,
    "CZT": 0.96108,
    "RZ":  0.98502,
    "S":   0.98870,
    "MEASURE": 1,
}

@dataclass
class PackerResult:
    """Result of running the heuristic packer."""
    # Program qubit index -> patch index (or other representation you choose later)
    qubit_to_patch: Dict[int, int]
    # Constraints / decisions used by phase two (e.g., ("same"|"diff"|"solo", i, j_or_None))
    decisions: List[Tuple[str, int, Optional[int]]]
    # Solo budget actually used
    solo_budget_used: int
    # Which approach was used: "balanced", "greedy", or "defensive"
    approach: str


class HeuristicPacker:
    """
    High-level entrypoint for the heuristic packing algorithm.

    This class orchestrates:
      - Phase 1: graph construction based on gate LSRs and the program circuit.
      - Phase 2: load-balancing heuristic over the graph to pick a mapping.
    """

    def __init__(
        self,
        gate_lsr_table: LSRTable = None,
        patch_capacity: int = 2,
        max_patches: int = None,
        enable_phase_three: bool = False,
    ) -> None:
        self.gate_lsr_table = gate_lsr_table  # raw LSRs, e.g., {"H": 0.95871, "X": 0.99763, ...}
        self.patch_capacity = patch_capacity  # k
        self.max_patches = max_patches        # m
        self.enable_phase_three = enable_phase_three

        if self.gate_lsr_table is None:
            self.gate_lsr_table = DEFAULT_LSR_TABLE

    def pack(
        self,
        program_circuit: qiskit.QuantumCircuit,
        solo_budget: Optional[int] = None,
        approach: Literal["balanced", "greedy", "defensive"] = "balanced",
        enable_phase_three: Optional[bool] = None,
    ) -> PackerResult:
        """
        Run the heuristic algorithm for a given program circuit.

        Steps (to be implemented later):
          - Build heuristic graph in phase_one_graph.build_graph(...)
          - Run phase_two_balance.balance(...) with the chosen approach and solo_budget
          - Convert phase-two decisions into a final qubit_to_patch mapping
        """
        graph: GraphHeuristics = build_graph(
            program_circuit=program_circuit,
            gate_lsr_table=self.gate_lsr_table,
            patch_capacity=self.patch_capacity,
            max_patches=self.max_patches,
        )

        phase_two_result: Any = balance_graph(
            graph,
            solo_budget=solo_budget,
            approach=approach,
        )

        phase_three_enabled = (
            self.enable_phase_three if enable_phase_three is None else enable_phase_three
        )
        final_stage_result: Any = (
            self._apply_phase_three(phase_two_result)
            if phase_three_enabled
            else phase_two_result
        )

        def _extract(field: str, default: Any = None) -> Any:
            if isinstance(final_stage_result, dict):
                return final_stage_result.get(field, default)
            return getattr(final_stage_result, field, default)

        qubit_to_patch: Optional[Dict[int, int]] = _extract("qubit_to_patch")
        if qubit_to_patch is None:
            raise ValueError("balance_graph must provide a qubit_to_patch mapping.")

        decisions: List[Tuple[str, int, Optional[int]]] = _extract("decisions", [])
        solo_budget_used: int = _extract("solo_budget_used", 0)

        return PackerResult(
            qubit_to_patch=qubit_to_patch,
            decisions=decisions,
            solo_budget_used=solo_budget_used,
            approach=approach,
        )

    def _apply_phase_three(self, phase_two_result: Any) -> Any:
        """
        Placeholder for the optional distance-dependent assignment phase.

        Input/output contracts mirror Phase 2 so downstream code can treat the
        result uniformly whether or not phase three logic is enabled.
        """
        # TODO: implement distance-aware refinements to the mapping.
        return phase_two_result
