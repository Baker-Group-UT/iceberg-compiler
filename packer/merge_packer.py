from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import qiskit

from packer.phase_one_graph import (
    DEFAULT_WORK_TABLE,
    DEFAULT_WORK_WEIGHTS,
    build_graph,
    gate_cost,
)
from packer.phase_three_map import PatchLayoutMetadata, compute_grid_aligned_mapping
from translator.merging import PairMergeAnalyzer, PairMergeStats
from translator.passes import cancel_inverse_gates

SoloStrategy = Literal["most_gates", "pair_then_solo", "most_hadamards", "all"]


@dataclass(frozen=True)
class MergeWeights:
    w1: float = DEFAULT_WORK_WEIGHTS[0]
    w2: float = DEFAULT_WORK_WEIGHTS[1]
    idle_error_weight: float = .2
    distance_error_weight: float = 0.1


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
        merge_weights: Optional weighting for 1q/2q gate costs.
    """

    def __init__(
        self,
        *,
        patch_capacity: int = 2,
        solo_strategy: SoloStrategy = "most_gates",
        merge_weights: Optional[MergeWeights] = None,
        top_k: int = 1,
        num_paths: int = 1,
        selection_decay: float = 0.9,
        debug: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.patch_capacity = patch_capacity
        self.solo_strategy = solo_strategy
        self.merge_weights = merge_weights or MergeWeights()
        self.top_k = top_k
        self.num_paths = num_paths
        self.selection_decay = selection_decay
        self.debug = debug
        self.rng = rng or random.Random()

    # Historical experimental path (MWBM/DP) kept for quick toggling/reference.
    # def pack(
    #     self,
    #     program_circuit: qiskit.QuantumCircuit,
    #     num_codes: int,
    #     *,
    #     solo_strategy: Optional[SoloStrategy] = None,
    # ) -> MergePackingResult:
    #     program_circuit = cancel_inverse_gates(program_circuit)
    #     if self.patch_capacity != 2:
    #         raise ValueError("MergeMaximizingPacker currently supports patch_capacity=2 only.")
    #     num_qubits = program_circuit.num_qubits
    #     if num_qubits == 0:
    #         return MergePackingResult(
    #             permutation=[],
    #             qubit_to_patch={},
    #             patches=[],
    #             num_codes=0,
    #             solo_strategy=solo_strategy or self.solo_strategy,
    #             total_merge_score=0.0,
    #         )
    #
    #     min_codes = math.ceil(num_qubits / self.patch_capacity)
    #     max_codes = num_qubits
    #     if num_codes < min_codes or num_codes > max_codes:
    #         raise ValueError(
    #             f"num_codes must be between {min_codes} and {max_codes} "
    #             f"for {num_qubits} qubits."
    #         )
    #
    #     analyzer = PairMergeAnalyzer(program_circuit)
    #     merge_scores: List[List[Optional[float]]] = [
    #         [None for _ in range(num_qubits)] for _ in range(num_qubits)
    #     ]
    #     pair_gate_counts = _count_twoq_gates_by_pair(program_circuit)
    #     qubit_gate_counts = _count_twoq_gates_by_qubit(program_circuit)
    #     h_counts = _count_hadamards_by_qubit(program_circuit)
    #
    #     for i in range(num_qubits):
    #         solo_counts = qubit_gate_counts.get(i, {"CX": 0, "CZ": 0, "XCX": 0})
    #         merge_scores[i][i] = _solo_score(
    #             h_counts.get(i, 0),
    #             solo_counts.get("CX", 0),
    #             solo_counts.get("CZ", 0),
    #             solo_counts.get("XCX", 0),
    #             self.merge_weights,
    #         )
    #         for j in range(i + 1, num_qubits):
    #             stats = analyzer.analyze_pair(i, j)
    #             pair_counts = pair_gate_counts.get((i, j), {"CX": 0, "CZ": 0, "XCX": 0})
    #             merge_scores[i][j] = _score_pair(stats, self.merge_weights) + _pair_bias(
    #                 pair_counts.get("CX", 0),
    #                 pair_counts.get("CZ", 0),
    #                 pair_counts.get("XCX", 0),
    #                 self.merge_weights,
    #             )
    #
    #     pairs_needed = num_qubits - num_codes
    #     solos_needed = 2 * num_codes - num_qubits
    #     solo_patches, pair_patches, total_score = _select_optimal_patches(
    #         merge_scores,
    #         pairs_needed=pairs_needed,
    #         solos_needed=solos_needed,
    #     )
    #
    #     patches = solo_patches + pair_patches
    #     qubit_to_patch, permutation = _build_permutation(patches, num_qubits)
    #     patch_layout = None
    #     qubit_to_patch, permutation, patches, patch_layout = self._apply_phase_three(
    #         program_circuit,
    #         num_codes,
    #         qubit_to_patch,
    #         permutation,
    #         patches,
    #     )
    #
    #     return MergePackingResult(
    #         permutation=permutation,
    #         qubit_to_patch=qubit_to_patch,
    #         patches=patches,
    #         num_codes=num_codes,
    #         solo_strategy=solo_strategy or self.solo_strategy,
    #         total_merge_score=total_score,
    #         patch_layout=patch_layout,
    #     )

    def pack(
        self,
        program_circuit: qiskit.QuantumCircuit,
        num_codes: int,
        *,
        solo_strategy: Optional[SoloStrategy] = None,
    ) -> MergePackingResult:
        program_circuit = cancel_inverse_gates(program_circuit)
        strategy = solo_strategy or self.solo_strategy
        if strategy == "all":
            return self._pack_best_strategy(
                program_circuit,
                num_codes,
            )
        return self._pack_single_strategy(
            program_circuit,
            num_codes,
            strategy=strategy,
        )

    def _pack_single_strategy(
        self,
        program_circuit: qiskit.QuantumCircuit,
        num_codes: int,
        *,
        strategy: SoloStrategy,
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
                solo_strategy=strategy,
                total_merge_score=0.0,
            )

        min_codes = math.ceil(num_qubits / self.patch_capacity)
        max_codes = num_qubits
        if num_codes < min_codes or num_codes > max_codes:
            raise ValueError(
                f"num_codes must be between {min_codes} and {max_codes} "
                f"for {num_qubits} qubits."
            )

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

        if strategy != "pair_then_solo":
            if len(remaining) % 2 != 0:
                raise ValueError("Solo selection left an odd number of qubits to pair.")
        stop_at = solo_needed if strategy == "pair_then_solo" else 0
        pair_patches = _pick_pairs_best_path(
            remaining,
            analyzer,
            self.merge_weights,
            stop_at=stop_at,
            top_k=self.top_k,
            num_paths=self.num_paths,
            selection_decay=self.selection_decay,
            debug=self.debug,
            rng=self.rng,
        )
        remaining.difference_update({q for pair in pair_patches for q in pair})
        if strategy == "pair_then_solo":
            solo_patches = [(q,) for q in sorted(remaining)]

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

    def _pack_best_strategy(
        self,
        program_circuit: qiskit.QuantumCircuit,
        num_codes: int,
    ) -> MergePackingResult:
        candidates = [
            "most_gates",
            "pair_then_solo",
            "most_hadamards",
        ]
        best_result: Optional[MergePackingResult] = None
        for candidate in candidates:
            result = self._pack_single_strategy(
                program_circuit,
                num_codes,
                strategy=candidate,  # type: ignore[arg-type]
            )
            if best_result is None or result.total_merge_score > best_result.total_merge_score:
                best_result = result
        if best_result is None:
            raise ValueError("Unable to select a packing strategy.")
        return best_result

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
    w1 = weights.w1
    w2 = weights.w2
    h_cost = gate_cost("H", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    hh_cost = gate_cost("HH", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    x_cost = gate_cost("X", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    xx_cost = gate_cost("XX", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    z_cost = gate_cost("Z", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    zz_cost = gate_cost("ZZ", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    cx_cost = gate_cost("CX", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    cxt_cost = gate_cost("CXT", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    cz_cost = gate_cost("CZ", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    czt_cost = gate_cost("CZT", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    xcx_cost = gate_cost("XCX", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    xcxt_cost = gate_cost("XCXT", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    hh_savings = 2.0 * h_cost - hh_cost
    xx_savings = 2.0 * x_cost - xx_cost
    zz_savings = 2.0 * z_cost - zz_cost
    cx_savings = 2.0 * cx_cost - cxt_cost
    cz_savings = 2.0 * cz_cost - czt_cost
    xcx_savings = 2.0 * xcx_cost - xcxt_cost
    pair_adjust = (weights.distance_error_weight - weights.idle_error_weight) * stats.twoq_between
    hh_distance_penalty = weights.idle_error_weight * stats.hh_layer_distance
    return (
        stats.hh_merges * hh_savings
        + stats.xx_merges * xx_savings
        + stats.zz_merges * zz_savings
        + stats.cx_merges * cx_savings
        + stats.cz_merges * cz_savings
        + stats.xcx_merges * xcx_savings
        + pair_adjust
        - hh_distance_penalty
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


def _pick_pairs_best_path(
    remaining: Iterable[int],
    analyzer: PairMergeAnalyzer,
    weights: MergeWeights,
    *,
    stop_at: int,
    top_k: int,
    num_paths: int,
    selection_decay: float,
    debug: bool,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    if top_k <= 0 or num_paths <= 0:
        raise ValueError("top_k and num_paths must be positive.")
    if not (0.0 < selection_decay < 1.0):
        raise ValueError("selection_decay must be between 0 and 1.")
    best_pairs: List[Tuple[int, int]] = []
    best_score = float("-inf")
    for path_idx in range(num_paths):
        local_remaining = set(remaining)
        pairs: List[Tuple[int, int]] = []
        score = 0.0
        round_idx = 0
        while len(local_remaining) > stop_at:
            candidates = _top_k_pairs(local_remaining, analyzer, weights, top_k, debug=debug)
            pair, pair_score = _pick_weighted_pair(candidates, selection_decay, rng)
            if debug:
                print(
                    "[merge_packer] path",
                    path_idx,
                    "round",
                    round_idx,
                    "selected",
                    pair,
                    "score",
                    f"{pair_score:.4f}",
                )
            pairs.append(pair)
            score += pair_score
            local_remaining.difference_update(pair)
            round_idx += 1
        if score > best_score:
            best_score = score
            best_pairs = pairs
    return best_pairs


def _top_k_pairs(
    remaining: Iterable[int],
    analyzer: PairMergeAnalyzer,
    weights: MergeWeights,
    k: int,
    *,
    debug: bool = False,
) -> List[Tuple[Tuple[int, int], float]]:
    candidates = sorted(remaining)
    scored: List[Tuple[Tuple[int, int], float]] = []
    for idx, i in enumerate(candidates):
        for j in candidates[idx + 1 :]:
            stats = analyzer.analyze_pair(i, j)
            score = _score_pair(stats, weights)
            if debug:
                print(
                    "[merge_packer] pair",
                    (i, j),
                    "score",
                    f"{score:.4f}",
                    "stats",
                    f"hh={stats.hh_merges}",
                    f"xx={stats.xx_merges}",
                    f"zz={stats.zz_merges}",
                    f"cxt={stats.cx_merges}",
                    f"czt={stats.cz_merges}",
                    f"twoq_between={stats.twoq_between}",
                    f"hh_dist={stats.hh_layer_distance}",
                )
            scored.append(((i, j), score))
    if not scored:
        raise ValueError("Unable to select a qubit pair from the remaining set.")
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]


def _pick_weighted_pair(
    candidates: Sequence[Tuple[Tuple[int, int], float]],
    selection_decay: float,
    rng: random.Random,
) -> Tuple[Tuple[int, int], float]:
    weights = [selection_decay * (1.0 - selection_decay) ** idx for idx in range(len(candidates))]
    pair, score = rng.choices(candidates, weights=weights, k=1)[0]
    return pair, score


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


# Historical experimental helpers for the MWBM/DP path.
# def _bias_delta_same_vs_diff(gate_name: str, weights: MergeWeights) -> float:
#     w1 = weights.w1
#     w2 = weights.w2
#     if gate_name == "CX":
#         same = (0, 4)
#         diff = (0, 4)
#     elif gate_name == "CZ":
#         same = (1, 3)
#         diff = (0, 4)
#     elif gate_name == "XCX":
#         same = (5, 3)
#         diff = (4, 4)
#     else:
#         return 0.0
#     same_cost = w1 * same[0] + w2 * same[1]
#     diff_cost = w1 * diff[0] + w2 * diff[1]
#     return diff_cost - same_cost
#
#
# def _pair_bias(cx_count: int, cz_count: int, xcx_count: int, weights: MergeWeights) -> float:
#     bias = 0.0
#     bias += cx_count * _bias_delta_same_vs_diff("CX", weights)
#     bias += cz_count * _bias_delta_same_vs_diff("CZ", weights)
#     bias += xcx_count * _bias_delta_same_vs_diff("XCX", weights)
#     return bias
#
#
# def _solo_hh_delta(h_count: int, weights: MergeWeights) -> float:
#     w1 = weights.w1[]
#     w2 = weights.w2
#     h_cost = gate_cost("H", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
#     hh_cost = gate_cost("HH", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
#     return h_count * (h_cost - hh_cost)
#
#
# def _solo_score(
#     h_count: int,
#     cx_count: int,
#     cz_count: int,
#     xcx_count: int,
#     weights: MergeWeights,
# ) -> float:
#     solo = _solo_hh_delta(h_count, weights)
#     solo -= _pair_bias(cx_count, cz_count, xcx_count, weights)
#     return solo
#
#
# def _count_twoq_gates_by_pair(
#     program_circuit: qiskit.QuantumCircuit,
# ) -> Dict[Tuple[int, int], Dict[str, int]]:
#     counts: Dict[Tuple[int, int], Dict[str, int]] = {}
#     for instruction, qargs, _ in program_circuit.data:
#         name = instruction.name.upper()
#         if name in ("BARRIER", "MEASURE"):
#             continue
#         if name not in ("CX", "CZ", "XCX"):
#             continue
#         if len(qargs) != 2:
#             continue
#         i, j = sorted(program_circuit.find_bit(q).index for q in qargs)
#         entry = counts.setdefault((i, j), {"CX": 0, "CZ": 0, "XCX": 0})
#         entry[name] += 1
#     return counts
#
#
# def _count_twoq_gates_by_qubit(
#     program_circuit: qiskit.QuantumCircuit,
# ) -> Dict[int, Dict[str, int]]:
#     counts: Dict[int, Dict[str, int]] = {
#         i: {"CX": 0, "CZ": 0, "XCX": 0} for i in range(program_circuit.num_qubits)
#     }
#     for instruction, qargs, _ in program_circuit.data:
#         name = instruction.name.upper()
#         if name in ("BARRIER", "MEASURE"):
#             continue
#         if name not in ("CX", "CZ", "XCX"):
#             continue
#         if len(qargs) != 2:
#             continue
#         for q in qargs:
#             idx = program_circuit.find_bit(q).index
#             counts[idx][name] += 1
#     return counts
#
#
# def _select_optimal_patches(
#     scores: Sequence[Sequence[Optional[float]]],
#     *,
#     pairs_needed: int,
#     solos_needed: int,
# ) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], float]:
#     num_qubits = len(scores)
#     if pairs_needed < 0 or solos_needed < 0:
#         raise ValueError("pairs_needed and solos_needed must be non-negative.")
#     if 2 * pairs_needed + solos_needed != num_qubits:
#         raise ValueError("pairs_needed and solos_needed do not match num_qubits.")
#
#     def pair_score(i: int, j: int) -> float:
#         lo, hi = (i, j) if i < j else (j, i)
#         value = scores[lo][hi]
#         if value is None:
#             raise ValueError(f"Missing score for pair ({lo}, {hi}).")
#         return float(value)
#
#     def solo_score(i: int) -> float:
#         value = scores[i][i]
#         if value is None:
#             raise ValueError(f"Missing solo score for qubit {i}.")
#         return float(value)
#
#     from functools import lru_cache
#
#     @lru_cache(maxsize=None)
#     def solve(mask: int, pairs_left: int, solos_left: int) -> Tuple[float, Tuple[Tuple[int, ...], ...]]:
#         if mask == 0:
#             if pairs_left == 0 and solos_left == 0:
#                 return 0.0, ()
#             return float("-inf"), ()
#         if pairs_left < 0 or solos_left < 0:
#             return float("-inf"), ()
#
#         i = (mask & -mask).bit_length() - 1
#         best_score = float("-inf")
#         best_choice: Tuple[Tuple[int, ...], ...] = ()
#
#         if solos_left > 0:
#             score_rest, choice_rest = solve(mask & ~(1 << i), pairs_left, solos_left - 1)
#             score_here = solo_score(i) + score_rest
#             if score_here > best_score:
#                 best_score = score_here
#                 best_choice = choice_rest + ((i,),)
#
#         if pairs_left > 0:
#             remaining = mask & ~(1 << i)
#             j = remaining
#             while j:
#                 bit = j & -j
#                 idx = bit.bit_length() - 1
#                 score_rest, choice_rest = solve(remaining & ~bit, pairs_left - 1, solos_left)
#                 score_here = pair_score(i, idx) + score_rest
#                 if score_here > best_score:
#                     best_score = score_here
#                     best_choice = choice_rest + ((i, idx),)
#                 j &= j - 1
#
#         return best_score, best_choice
#
#     full_mask = (1 << num_qubits) - 1
#     total_score, choices = solve(full_mask, pairs_needed, solos_needed)
#     if total_score == float("-inf"):
#         raise ValueError("Unable to find a feasible pairing with the given constraints.")
#
#     solo_patches: List[Tuple[int, ...]] = []
#     pair_patches: List[Tuple[int, ...]] = []
#     for choice in choices:
#         if len(choice) == 1:
#             solo_patches.append((choice[0],))
#         else:
#             i, j = choice
#             pair_patches.append(tuple(sorted((i, j))))
#     return solo_patches, pair_patches, total_score
