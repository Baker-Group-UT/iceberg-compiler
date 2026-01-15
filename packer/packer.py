# packer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Any

import qiskit  # adjust import if needed

from .phase_one_graph import (
    build_graph,
    GraphHeuristics,
    WorkTable,
    DEFAULT_WORK_TABLE,
    DEFAULT_WORK_WEIGHTS,
)
from .phase_two_balance import balance_graph_multi, BalanceResult
from .phase_three_map import compute_grid_aligned_mapping, PatchLayoutMetadata

GateName = str

WARN_ON_BUDGET_FILTER = False

@dataclass
class PackingCandidate:
    qubit_to_patch: Dict[int, int]
    decisions: List[Tuple[str, int, Optional[int]]]
    patches_used: int
    approach: str
    solo_budget: int
    patch_weights: Dict[int, float]
    patch_layout: Optional[PatchLayoutMetadata] = None


@dataclass
class PackerResult:
    """Result of running the heuristic packer."""
    qubit_to_patch: Dict[int, int]
    decisions: List[Tuple[str, int, Optional[int]]]
    patches_used: int
    approach: str
    solo_budget: int
    candidates: Dict[str, PackingCandidate]
    patch_weights: Dict[int, float]
    patch_layout: Optional[PatchLayoutMetadata] = None


class HeuristicPacker:
    """
    High-level entrypoint for the heuristic packing algorithm.

    This class orchestrates:
      - Phase 1: graph construction based on gate work counts and the program circuit.
      - Phase 2: load-balancing heuristic over the graph to pick a mapping.
    """

    def __init__(
        self,
        gate_work_table: WorkTable | None = None,
        work_weights: Tuple[float, float] | None = None,
        patch_capacity: int = 2,
        max_patches: int = None,
        enable_phase_three: bool = True,
        explore_all_solo_budgets: bool = True,
        explore_all_approaches: bool = True,
    ) -> None:
        self.gate_work_table = gate_work_table
        self.work_weights = work_weights
        self.patch_capacity = patch_capacity  # k
        self.max_patches = max_patches        # m
        self.enable_phase_three = enable_phase_three
        self.explore_all_solo_budgets = explore_all_solo_budgets
        self.explore_all_approaches = explore_all_approaches

        if self.gate_work_table is None:
            self.gate_work_table = DEFAULT_WORK_TABLE
        if self.work_weights is None:
            self.work_weights = DEFAULT_WORK_WEIGHTS

    def pack(
        self,
        program_circuit: qiskit.QuantumCircuit,
        max_patches: int = None,
        approach: Literal["balanced", "greedy", "defensive"] = "balanced",
        enable_phase_three: Optional[bool] = None,
        explore_solo_budgets: Optional[bool] = None,
        explore_all_approaches: Optional[bool] = None,
    ) -> PackerResult:
        """
        Run the heuristic algorithm for a given program circuit.

        Steps (to be implemented later):
          - Build heuristic graph in phase_one_graph.build_graph(...)
          - Run phase_two_balance.balance(...) with the chosen approach and solo_budget
          - Convert phase-two decisions into a final qubit_to_patch mapping
        """
        if max_patches is None:
            max_patches = self.max_patches

        graph: GraphHeuristics = build_graph(
            program_circuit=program_circuit,
            gate_work_table=self.gate_work_table,
            work_weights=self.work_weights,
            patch_capacity=self.patch_capacity,
            max_patches=max_patches,
        )

        phase_three_enabled = (
            self.enable_phase_three if enable_phase_three is None else enable_phase_three
        )
        explore_budgets = (
            self.explore_all_solo_budgets
            if explore_solo_budgets is None
            else explore_solo_budgets
        )
        explore_approaches_flag = (
            self.explore_all_approaches
            if explore_all_approaches is None
            else explore_all_approaches
        )

        base_budget = self._choose_solo_budget(graph)
        if explore_budgets and graph.solo_budget_max >= graph.solo_budget_min:
            budgets_to_try = list(range(graph.solo_budget_min, graph.solo_budget_max + 1))
        else:
            budgets_to_try = [base_budget]
        budgets_to_try = self._filter_feasible_budgets(budgets_to_try, graph)
        budgets_to_try = budgets_to_try[::-1] # Try it from the top down

        approaches_to_try: List[str] = [approach]
        if explore_approaches_flag:
            approaches_to_try = ["balanced", "greedy", "defensive"]
        # Remove duplicates while preserving order
        ordered: Dict[str, None] = {}
        for entry in approaches_to_try:
            ordered.setdefault(entry, None)
        approaches_to_try = list(ordered.keys())

        all_candidates: Dict[str, PackingCandidate] = {}
        balance_lookup: Dict[str, Any] = {}

        for approach_name in approaches_to_try:
            results_by_budget = balance_graph_multi(
                graph,
                budgets_to_try,
                approach=approach_name,
            )
            for budget, balance_result in results_by_budget.items():
                key = self._candidate_key(approach_name, budget)
                balance_lookup[key] = balance_result
                all_candidates[key] = self._build_candidate(
                    balance_result,
                    approach_name,
                    budget,
                    graph,
                )

        if not all_candidates:
            raise ValueError("Failed to produce any packing candidates.")

        best_key = min(
            all_candidates,
            key=lambda candidate_key: self._score_candidate(all_candidates[candidate_key]),
        )
        best_balance_result: Any = balance_lookup[best_key]
        best_candidate = all_candidates[best_key]

        if phase_three_enabled:
            best_balance_result = self._apply_phase_three(best_balance_result, graph)
            best_candidate = self._build_candidate(
                best_balance_result,
                best_candidate.approach,
                best_candidate.solo_budget,
                graph,
            )
            all_candidates[best_key] = best_candidate

        return PackerResult(
            qubit_to_patch=best_candidate.qubit_to_patch,
            decisions=best_candidate.decisions,
            patches_used=best_candidate.patches_used,
            approach=best_candidate.approach,
            solo_budget=best_candidate.solo_budget,
            candidates=all_candidates,
            patch_weights=best_candidate.patch_weights,
            patch_layout=best_candidate.patch_layout,
        )

    def _apply_phase_three(self, phase_two_result: Any, graph: GraphHeuristics) -> Any:
        """
        Apply a distance-aware patch relabeling so physical adjacency matches
        logical interaction strength.

        Input/output contracts mirror Phase 2 so downstream code can treat the
        result uniformly whether or not phase three logic is enabled.
        """
        qubit_to_patch: Optional[Dict[int, int]] = self._extract_field(
            phase_two_result,
            "qubit_to_patch",
        )
        if not qubit_to_patch:
            return phase_two_result

        try:
            remapped, layout = compute_grid_aligned_mapping(graph, qubit_to_patch)
            # print(remapped, layout)
        except ValueError:
            return phase_two_result

        if isinstance(phase_two_result, BalanceResult):
            phase_two_result.qubit_to_patch = remapped
            phase_two_result.patch_layout = layout
            return phase_two_result

        if isinstance(phase_two_result, dict):
            updated = dict(phase_two_result)
            updated["qubit_to_patch"] = remapped
            updated["patch_layout"] = layout
            return updated

        setattr(phase_two_result, "qubit_to_patch", remapped)
        setattr(phase_two_result, "patch_layout", layout)
        return phase_two_result

    def _candidate_key(self, approach: str, solo_budget: int) -> str:
        return f"{approach}|solo={solo_budget}"

    def _score_candidate(self, candidate: PackingCandidate) -> Tuple[float, float, float, int]:
        weights = list(candidate.patch_weights.values())
        if not weights:
            return (float("inf"), float("inf"), float("inf"), candidate.patches_used)
        total_weight = sum(weights)
        max_weight = max(weights)
        mean_weight = total_weight / len(weights)  # work per used patch
        spread = sum(abs(w - mean_weight) for w in weights) / len(weights)
        return (
            max_weight,
            spread,
            mean_weight,
            candidate.patches_used,  # placeholder for potential future use
        )

    def _build_candidate(
        self,
        source: Any,
        approach: str,
        solo_budget: int,
        graph: GraphHeuristics,
    ) -> PackingCandidate:
        mapping: Optional[Dict[int, int]] = self._extract_field(source, "qubit_to_patch")
        if mapping is None:
            raise ValueError("balance_graph must provide a qubit_to_patch mapping.")
        decisions: List[Tuple[str, int, Optional[int]]] = list(
            self._extract_field(source, "decisions", [])
        )
        patch_layout: Optional[PatchLayoutMetadata] = self._extract_field(source, "patch_layout")
        mapping_copy = dict(mapping)
        patches_used = len(set(mapping_copy.values()))
        patch_weights: Dict[int, float] = {}
        for qubit, patch in mapping_copy.items():
            node = graph.nodes.get(qubit)
            if node is None:
                continue
            patch_weights.setdefault(patch, 0.0)
            patch_weights[patch] += node.base_cost
        return PackingCandidate(
            qubit_to_patch=mapping_copy,
            decisions=decisions,
            patches_used=patches_used,
            approach=approach,
            solo_budget=solo_budget,
            patch_weights=patch_weights,
            patch_layout=patch_layout,
        )

    def _extract_field(self, source: Any, field: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(field, default)
        return getattr(source, field, default)

    def _choose_solo_budget(self, graph: GraphHeuristics) -> int:
        """Select a solo budget based on graph metadata."""
        return graph.solo_budget_min

    def _filter_feasible_budgets(
        self,
        budgets: List[int],
        graph: GraphHeuristics,
    ) -> List[int]:
        feasible: List[int] = []
        removed: List[int] = []
        num_qubits = len(graph.nodes)
        patch_capacity = graph.patch_capacity
        for budget in budgets:
            remaining = num_qubits - budget
            if remaining < 0:
                removed.append(budget)
                continue
            if remaining % patch_capacity != 0:
                removed.append(budget)
                continue
            feasible.append(budget)

        if not feasible:
            raise ValueError(
                "No feasible solo budgets remain after filtering by patch capacity. "
                "Adjust max_patches or patch_capacity."
            )
        if removed and WARN_ON_BUDGET_FILTER:
            print(
                f"[HeuristicPacker] Skipping infeasible solo budgets ("
                f"would leave non-multiples of patch capacity): {sorted(set(removed))}"
            )
        return feasible

    def packing_to_permutation(
        self,
        qubit_to_patch: Dict[int, int],
        num_qubits: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Convert a qubit_to_patch mapping into the permutation format expected by other code.

        Args:
            qubit_to_patch: Mapping of program qubit index -> patch index.
            num_qubits: Optional number of logical qubits; defaults to len(mapping).
        """
        if num_qubits is None:
            num_qubits = len(qubit_to_patch)

        slots_used: Dict[int, int] = {}
        permutation: List[Tuple[int, int]] = []
        for qubit in range(num_qubits):
            patch = qubit_to_patch.get(qubit)
            if patch is None:
                raise ValueError(f"Missing patch assignment for qubit {qubit}.")
            slot = slots_used.get(patch, 0)
            permutation.append((patch, slot))
            slots_used[patch] = slot + 1
        return permutation
