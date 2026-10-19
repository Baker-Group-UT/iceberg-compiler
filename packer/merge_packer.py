from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import qiskit

from packer.work_table import (
    DEFAULT_WORK_TABLE,
    DEFAULT_WORK_WEIGHTS,
    build_graph,
    gate_cost,
)
from packer.grid_layout import PatchLayoutMetadata, compute_grid_aligned_mapping
from translator.merging import PairMergeAnalyzer, PairMergeStats
from translator.passes import cancel_inverse_gates

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
    total_merge_score: float
    patch_layout: Optional[PatchLayoutMetadata] = None


class MergeMaximizingPacker:
    """
    Greedy packer that pairs qubits to maximize merge opportunities.

    Pairs qubits greedily using the net-gain formula w_ij - w_ii - w_jj
    (paper eq. 2), where w_ij is the merge savings from pairing qubits i and j,
    and w_ii / w_jj are the solo baselines for each qubit.  Remaining unpaired
    qubits are assigned to solo patches.

    Args:
        patch_capacity: Patch size, currently must be 2.
        merge_weights: Optional weighting for 1q/2q gate costs and error biases.
    """

    def __init__(
        self,
        *,
        patch_capacity: int = 2,
        merge_weights: Optional[MergeWeights] = None,
        top_k: int = 1,
        num_paths: int = 1,
        selection_decay: float = 0.9,
        debug: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.patch_capacity = patch_capacity
        self.merge_weights = merge_weights or MergeWeights()
        self.top_k = top_k
        self.num_paths = num_paths
        self.selection_decay = selection_decay
        self.debug = debug
        self.rng = rng or random.Random()

    def pack(
        self,
        program_circuit: qiskit.QuantumCircuit,
        num_codes: int,
    ) -> MergePackingResult:
        """Pack qubits into patches using the net-gain greedy algorithm.

        Greedily pairs qubits by maximising w_ij - w_ii - w_jj (paper eq. 2)
        until solo_needed qubits remain, then assigns the rest to solo patches.
        """
        if self.patch_capacity != 2:
            raise ValueError("MergeMaximizingPacker currently supports patch_capacity=2 only.")
        program_circuit = cancel_inverse_gates(program_circuit)
        num_qubits = program_circuit.num_qubits
        if num_qubits == 0:
            return MergePackingResult(
                permutation=[],
                qubit_to_patch={},
                patches=[],
                num_codes=0,
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
        pair_patches = _pick_pairs_best_path(
            remaining,
            analyzer,
            self.merge_weights,
            stop_at=solo_needed,
            top_k=self.top_k,
            num_paths=self.num_paths,
            selection_decay=self.selection_decay,
            debug=self.debug,
            rng=self.rng,
        )
        remaining.difference_update({q for pair in pair_patches for q in pair})
        solo_patches: List[Tuple[int, ...]] = [(q,) for q in sorted(remaining)]

        patches = solo_patches + pair_patches
        qubit_to_patch, permutation = _build_permutation(patches, num_qubits)
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
    # T_ij: type-preference bias — per-gate-type depth difference between intra-patch
    # and transversal implementations, weighted by idle noise.
    #
    # From the physical gate table (non-FT, n=4, excluding SWAPs):
    #   CNOT: intra depth 4, transversal depth 1  → Δ = +3·idle  (intra worse, prefer separate)
    #   CZ:   intra depth 3, transversal depth 1  → Δ = +2·idle  (intra worse, prefer separate)
    #   XCX:  intra depth 5, transversal depth 3  → Δ = +2·idle  (intra worse, prefer separate)
    #         but intra saves 3 1q + 1 2q gate vs transversal
    #         net: -3·w1 - w2 + 2·idle  (usually negative → prefer intra for XCX)
    #
    # Positive bias discourages pairing (intra is worse); negative encourages it.
    idle = weights.idle_error_weight
    cx_bias  = stats.cx_between  * (3 * idle)
    cz_bias  = stats.cz_between  * (2 * idle)
    xcx_bias = stats.xcx_between * (-3 * weights.w1 - weights.w2 + 2 * idle)
    type_preference_bias = -(cx_bias + cz_bias + xcx_bias)
    # Delay penalties: discourage merges whose constituent gates are far apart in
    # the circuit and whose encoded primitive contains two-qubit gates.
    hh_distance_penalty = weights.idle_error_weight * stats.hh_layer_distance
    # CXT/CZT/XCXT all contain 2q gates; large layer distances increase encoded depth.
    cx_distance_penalty = weights.idle_error_weight * stats.cx_layer_distance
    cz_distance_penalty = weights.idle_error_weight * stats.cz_layer_distance
    xcx_distance_penalty = weights.idle_error_weight * stats.xcx_layer_distance
    return (
        stats.hh_merges * hh_savings
        + stats.xx_merges * xx_savings
        + stats.zz_merges * zz_savings
        + stats.cx_merges * cx_savings
        + stats.cz_merges * cz_savings
        + stats.xcx_merges * xcx_savings
        + type_preference_bias
        - hh_distance_penalty
        - cx_distance_penalty
        - cz_distance_penalty
        - xcx_distance_penalty
    )


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


def _score_solo(qubit: int, analyzer: PairMergeAnalyzer, weights: MergeWeights) -> float:
    """Solo baseline score (w_ii): savings earned when qubit is alone in its patch.

    Solo-patch optimizations (verified from translator handlers):
    - H → HH macro (hadamard.py:38): is_patch_solo() routes to iceberg_set.HH().
      Each solo H costs HH_cost instead of H_cost. Savings = H_cost - HH_cost per gate.
    - CX/CZ/XCX → always inter-patch transversal (controlled_not.py:88,
      controlled_z.py:95, controlled_x.py:88): is_alone_in_patch() enables the
      transversal path. Work-table cost is unchanged (CX=CXT=(0,4)) but an
      error-rate bias term is applied.
    - XX/ZZ merges on a solo lane: eliminated by cancel_inverse_gates() in practice.
    """
    stats = analyzer.analyze_solo(qubit)
    w1, w2 = weights.w1, weights.w2
    h_cost = gate_cost("H", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    hh_cost = gate_cost("HH", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    x_cost = gate_cost("X", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    xx_cost = gate_cost("XX", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    z_cost = gate_cost("Z", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    zz_cost = gate_cost("ZZ", DEFAULT_WORK_TABLE, w1=w1, w2=w2)
    # Each solo H is encoded as HH (whole-patch Hadamard) — cheaper than H_at()
    h_to_hh_savings = stats.h_count * (h_cost - hh_cost)
    # Solo XX/ZZ merges within a single qubit lane (typically 0 after cancellation)
    xx_savings = stats.xx_merges * (2.0 * x_cost - xx_cost)
    zz_savings = stats.zz_merges * (2.0 * z_cost - zz_cost)
    # Penalty for keeping a 2q-heavy qubit solo: each 2q gate that stays solo
    # forfeits a potential CXT/CZT/XCXT merge opportunity and creates additional
    # sequential inter-patch traffic (depth overhead, movement cost).
    # Calibrated so that ~8 solo 2q gates ≈ 1 H gate savings (55 / 8 ≈ 7 per gate).
    solo_2q_penalty = (h_cost - hh_cost) / 8.0 * stats.twoq_count
    return h_to_hh_savings + xx_savings + zz_savings - solo_2q_penalty


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
            # Net gain: w_ij - w_ii - w_jj (paper eq. 2).
            # w_ii and w_jj are the savings each qubit would earn solo;
            # only pair when the pairing beats their combined solo baselines.
            score = (
                _score_pair(stats, weights)
                - _score_solo(i, analyzer, weights)
                - _score_solo(j, analyzer, weights)
            )
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
