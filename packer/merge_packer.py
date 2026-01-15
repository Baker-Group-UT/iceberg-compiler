from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import qiskit

from packer.phase_one_graph import DEFAULT_WORK_TABLE, DEFAULT_WORK_WEIGHTS, build_graph
from packer.phase_three_map import PatchLayoutMetadata, compute_grid_aligned_mapping
from translator.merging import PairMergeAnalyzer, PairMergeStats

SoloStrategy = Literal["most_gates", "pair_then_solo", "most_hadamards"]


@dataclass(frozen=True)
class MergeWeights:
    hh: float = 1.0
    cx: float = 1.0
    cz: float = 1.0


@dataclass
class MergePackingResult:
    permutation: List[Tuple[int, int]]
    qubit_to_patch: Dict[int, int]
    patches: List[Tuple[int, ...]]
    num_codes: int
    solo_strategy: SoloStrategy
    total_merge_score: float
    patch_layout: Optional[PatchLayoutMetadata] = None


class MergeMaximizingPacker:
    """
    Greedy packer that pairs qubits to maximize merge opportunities.

    Args:
        patch_capacity: Patch size, currently must be 2 because merge scoring
            is based on qubit pairs.
        solo_strategy: How to select qubits that remain in solo patches when
            num_codes > ceil(num_qubits / patch_capacity).
            - "most_gates": preselect the busiest qubits as solos.
            - "pair_then_solo": pair greedily until the remaining qubits must be solo.
            - "most_hadamards": preselect qubits with the most H gates as solos.
        merge_weights: Optional weighting for HH/CX/CZ merge counts.
    """

    def __init__(
        self,
        *,
        patch_capacity: int = 2,
        solo_strategy: SoloStrategy = "pair_then_solo",
        merge_weights: Optional[MergeWeights] = None,
    ) -> None:
        self.patch_capacity = patch_capacity
        self.solo_strategy = solo_strategy
        self.merge_weights = merge_weights or MergeWeights()

    def pack(
        self,
        program_circuit: qiskit.QuantumCircuit,
        num_codes: int,
        *,
        solo_strategy: Optional[SoloStrategy] = None,
    ) -> MergePackingResult:
        if self.patch_capacity != 2:
            raise ValueError("MergeMaximizingPacker currently supports patch_capacity=2 only.")
        num_qubits = program_circuit.num_qubits
        if num_qubits == 0:
            return MergePackingResult(
                permutation=[],
                qubit_to_patch={},
                patches=[],
                num_codes=0,
                solo_strategy=solo_strategy or self.solo_strategy,
                total_merge_score=0.0,
            )

        min_codes = math.ceil(num_qubits / self.patch_capacity)
        max_codes = num_qubits
        if num_codes < min_codes or num_codes > max_codes:
            raise ValueError(
                f"num_codes must be between {min_codes} and {max_codes} "
                f"for {num_qubits} qubits."
            )

        strategy = solo_strategy or self.solo_strategy
        solo_needed = self.patch_capacity * num_codes - num_qubits
        if solo_needed < 0 or solo_needed > num_qubits:
            raise ValueError("Requested num_codes cannot be satisfied with patch_capacity=2.")

        analyzer = PairMergeAnalyzer(program_circuit)
        remaining = set(range(num_qubits))
        solo_patches: List[Tuple[int, ...]] = []
        pair_patches: List[Tuple[int, ...]] = []

        if strategy == "most_gates":
            gate_counts = _count_gates_by_qubit(program_circuit)
            solos = _select_top_qubits(gate_counts, solo_needed)
            solo_patches = [(q,) for q in solos]
            remaining.difference_update(solos)
        elif strategy == "most_hadamards":
            h_counts = _count_hadamards_by_qubit(program_circuit)
            solos = _select_top_qubits(h_counts, solo_needed)
            solo_patches = [(q,) for q in solos]
            remaining.difference_update(solos)
        elif strategy != "pair_then_solo":
            raise ValueError(f"Unknown solo_strategy: {strategy}")

        if strategy == "pair_then_solo":
            while len(remaining) > solo_needed:
                pair = _pick_best_pair(remaining, analyzer, self.merge_weights)
                pair_patches.append(pair)
                remaining.difference_update(pair)
            solo_patches = [(q,) for q in sorted(remaining)]
        else:
            if len(remaining) % 2 != 0:
                raise ValueError("Solo selection left an odd number of qubits to pair.")
            while remaining:
                pair = _pick_best_pair(remaining, analyzer, self.merge_weights)
                pair_patches.append(pair)
                remaining.difference_update(pair)

        patches = solo_patches + pair_patches
        qubit_to_patch, permutation = _build_permutation(patches, num_qubits)
        patch_layout = None
        qubit_to_patch, permutation, patches, patch_layout = self._apply_phase_three(
            program_circuit,
            num_codes,
            qubit_to_patch,
            permutation,
            patches,
        )
        total_merge_score = _total_merge_score(pair_patches, analyzer, self.merge_weights)

        return MergePackingResult(
            permutation=permutation,
            qubit_to_patch=qubit_to_patch,
            patches=patches,
            num_codes=num_codes,
            solo_strategy=strategy,
            total_merge_score=total_merge_score,
            patch_layout=patch_layout,
        )

    def _apply_phase_three(
        self,
        program_circuit: qiskit.QuantumCircuit,
        num_codes: int,
        qubit_to_patch: Dict[int, int],
        permutation: List[Tuple[int, int]],
        patches: List[Tuple[int, ...]],
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]], List[Tuple[int, ...]], Optional[PatchLayoutMetadata]]:
        graph = build_graph(
            program_circuit=program_circuit,
            gate_work_table=DEFAULT_WORK_TABLE,
            work_weights=DEFAULT_WORK_WEIGHTS,
            patch_capacity=self.patch_capacity,
            max_patches=num_codes,
        )
        try:
            remapped, layout = compute_grid_aligned_mapping(graph, qubit_to_patch)
        except ValueError:
            return qubit_to_patch, permutation, patches, None
        patches = _patches_from_mapping(remapped)
        qubit_to_patch, permutation = _build_permutation(patches, program_circuit.num_qubits)
        return qubit_to_patch, permutation, patches, layout


def _total_merge_score(
    pairs: Sequence[Tuple[int, ...]],
    analyzer: PairMergeAnalyzer,
    weights: MergeWeights,
) -> float:
    total = 0.0
    for pair in pairs:
        if len(pair) != 2:
            continue
        stats = analyzer.analyze_pair(pair[0], pair[1])
        total += _score_pair(stats, weights)
    return total


def _score_pair(stats: PairMergeStats, weights: MergeWeights) -> float:
    return (
        weights.hh * stats.hh_merges
        + weights.cx * stats.cx_merges
        + weights.cz * stats.cz_merges
    )


def _pick_best_pair(
    remaining: Iterable[int],
    analyzer: PairMergeAnalyzer,
    weights: MergeWeights,
) -> Tuple[int, int]:
    candidates = sorted(remaining)
    best_pair: Optional[Tuple[int, int]] = None
    best_score = float("-inf")
    for idx, i in enumerate(candidates):
        for j in candidates[idx + 1 :]:
            stats = analyzer.analyze_pair(i, j)
            score = _score_pair(stats, weights)
            if score > best_score or (score == best_score and (i, j) < best_pair):
                best_score = score
                best_pair = (i, j)
    if best_pair is None:
        raise ValueError("Unable to select a qubit pair from the remaining set.")
    return best_pair


def _select_top_qubits(counts: Dict[int, int], k: int) -> List[int]:
    if k <= 0:
        return []
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [qubit for qubit, _ in ordered[:k]]


def _count_gates_by_qubit(program_circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    counts = {i: 0 for i in range(program_circuit.num_qubits)}
    for instruction, qargs, _ in program_circuit.data:
        name = instruction.name.lower()
        if name in ("barrier", "measure"):
            continue
        for qubit in qargs:
            idx = program_circuit.find_bit(qubit).index
            counts[idx] += 1
    return counts


def _count_hadamards_by_qubit(program_circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    counts = {i: 0 for i in range(program_circuit.num_qubits)}
    for instruction, qargs, _ in program_circuit.data:
        name = instruction.name.lower()
        if name in ("barrier", "measure"):
            continue
        if name != "h":
            continue
        for qubit in qargs:
            idx = program_circuit.find_bit(qubit).index
            counts[idx] += 1
    return counts


def _build_permutation(
    patches: Sequence[Tuple[int, ...]],
    num_qubits: int,
) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
    qubit_to_patch: Dict[int, int] = {}
    qubit_to_slot: Dict[int, int] = {}
    for patch_id, patch_qubits in enumerate(patches):
        for slot, qubit in enumerate(sorted(patch_qubits)):
            qubit_to_patch[qubit] = patch_id
            qubit_to_slot[qubit] = slot

    permutation: List[Tuple[int, int]] = []
    for qubit in range(num_qubits):
        patch = qubit_to_patch.get(qubit)
        slot = qubit_to_slot.get(qubit)
        if patch is None or slot is None:
            raise ValueError(f"Missing patch assignment for qubit {qubit}.")
        permutation.append((patch, slot))
    return qubit_to_patch, permutation


def _patches_from_mapping(qubit_to_patch: Dict[int, int]) -> List[Tuple[int, ...]]:
    patch_members: Dict[int, List[int]] = {}
    for qubit, patch in qubit_to_patch.items():
        patch_members.setdefault(patch, []).append(qubit)
    return [tuple(sorted(patch_members[patch])) for patch in sorted(patch_members)]
