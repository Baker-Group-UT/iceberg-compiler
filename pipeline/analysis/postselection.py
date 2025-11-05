"""
postselection.py - Applies Iceberg postselection to simulator results.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
import qiskit

from pipeline.analysis.result_types import PostselectedSimulationResult
from pipeline.analysis.metrics import (
    logical_fidelity,
    total_variation_distance
)
from translator import TranslationResult, TranslationContext
from iceberg.iceberg_postselect import (
    iceberg_postselect,
)
from simulator.result_types import SimulationResult


def reduce_distribution(dist: Dict[str, float], context: TranslationContext) -> Dict[str, float]:
    """
    Reduce a full measurement distribution down to the tracked logical states.

    We keep only the bits corresponding to the *current* (dynamic) positions of the
    tracked states in `context.state_positions`, preserving that order.

    Bit index convention:
      - For n=4 Iceberg modules we have k_log = 2 logical bits per module.
      - The full bitstring is ordered by modules, then by slot:
            index = patch * k_log + slot
        e.g., for 3 modules (k_log=2): [p0s0, p0s1, p1s0, p1s1, p2s0, p2s1]

    Args:
        dist: dict mapping full bitstrings (e.g., '011100') -> probability.
        context: TranslationContext with dynamic state tracking.

    Returns:
        dict mapping reduced bitstrings (only tracked-state bits, in state order) -> probability.
    """
    k = context.k_log
    # Compute the physical bit indices (into the full bitstring) we care about,
    # in the same order as the tracked states (A, B, C, ...).
    kept_indices = [p * k + s for (p, s) in context.state_positions]

    reduced = defaultdict(float)

    # Optional sanity: infer expected full length from mapping; skip if inconsistent
    # expected_len = context.mapping["n_codes"] * k
    # (We won't enforce strictly, but could assert if you want strictness.)

    for bitstring, prob in dist.items():
        # Build the reduced key by selecting only the tracked-state positions
        try:
            key = "".join(bitstring[i] for i in kept_indices)
        except IndexError as exc:
            # If something is off (e.g., bitstring shorter than expected), skip or raise
            # Raising helps catch upstream mismatches early:
            raise IndexError(
                f"Bitstring length {len(bitstring)} too short for indices {kept_indices}."
            ) from exc
        reduced[key] += prob

    return dict(reduced)

def apply_iceberg_postselection(
    translated_circuits: List,
    results: List[SimulationResult],
) -> List[PostselectedSimulationResult]:
    """
    Combines raw simulator results with Iceberg postselection logic.

    Returns a list of enriched PostselectedSimulationResult objects for downstream
    dataframe creation or analysis.
    """
    postselected_results = []

    for t, result in zip(translated_circuits, results):
        key_order = result.key_order

        postselected_noisy_dist, sel_rate = iceberg_postselect(result.noisy_dist, key_order)
        postselected_ideal_dist, _ = iceberg_postselect(result.ideal_dist, key_order)

        logical_fid = logical_fidelity(postselected_ideal_dist, postselected_noisy_dist)
        logical_total_variation_distance = total_variation_distance(
            postselected_noisy_dist,
            postselected_ideal_dist
        )

        postselected_results.append(
            PostselectedSimulationResult(
                permutation=t.permutation,
                circuit=t.circuit,
                contains_faults= t.contains_faults,
                raw_result=result,
                postselected_noisy_dist=postselected_noisy_dist,
                postselected_ideal_dist=postselected_ideal_dist,
                logical_fidelity=logical_fid,
                logical_selection_rate=sel_rate,
                logical_total_variation_distance=logical_total_variation_distance
            )
        )

    return postselected_results

# 2 SHOTS PER RUN!
def postselect_against_reference(
    translated_circuits: List[TranslationResult],
    ideal_result: SimulationResult,
    noisy_results: List[SimulationResult],
) -> List[PostselectedSimulationResult]:
    """
    Applies Iceberg postselection using a shared ideal result against many noisy results.
    Useful for comparing multiple noisy runs to a single reference.
    """
    postselected_results = []

    for t, result in zip(translated_circuits, noisy_results):
        postselected_noisy_dist, sel_rate = iceberg_postselect(result.noisy_dist, result.key_order)
        postselected_noisy_dist = reduce_distribution(postselected_noisy_dist, t.context)

        logical_fid = logical_fidelity(ideal_result.ideal_dist, postselected_noisy_dist)
        logical_total_variation_distance = total_variation_distance(
            postselected_noisy_dist, ideal_result.ideal_dist
        )

        postselected_results.append(
            PostselectedSimulationResult(
                permutation=t.permutation,
                circuit=t.circuit,
                contains_faults=t.contains_faults,
                raw_result=result,
                postselected_noisy_dist=postselected_noisy_dist,
                postselected_ideal_dist=ideal_result.ideal_dist,
                logical_fidelity=logical_fid,
                logical_selection_rate=sel_rate,
                logical_total_variation_distance=logical_total_variation_distance,
            )
        )

    return postselected_results

def calculate_metrics_for_bare_circuit(
    circuit: qiskit.QuantumCircuit,
    result: SimulationResult,
) -> PostselectedSimulationResult:
    """
    Calculates the metric for circuit.
    """

    logical_fid = logical_fidelity(result.ideal_dist, result.noisy_dist)
    logical_total_variation_distance = total_variation_distance(
        result.noisy_dist,
        result.ideal_dist
    )

    return PostselectedSimulationResult(
        permutation=None,
        circuit=circuit,
        contains_faults= None,
        raw_result=result,
        postselected_noisy_dist=None,
        postselected_ideal_dist=None,
        logical_fidelity=logical_fid,
        logical_selection_rate=None,
        logical_total_variation_distance=logical_total_variation_distance
    )

# NOTE: REDUCE DISTRIBUTION FOR PARALLEL CIRCUIT EXECUTION, USE CAREFULLY
# def reduce_distribution(dist: Dict[str, float], context: "TranslationContext") -> Dict[str, float]:
#     """
#     Reduce a full measurement distribution down to the tracked logical states.

#     Normal mode (e.g. 2-code, 4-bit case):
#         We keep only the bits corresponding to the tracked logical states
#         in `context.state_positions`.

#     Special 6-bit mode (3-code, 6-bit case):
#         Each full bitstring is split into two 3-bit substrings:
#             - even indices 0,2,4  (ABC)
#             - odd indices  1,3,5  (A'B'C')
#         The probability mass of the 6-bit string is divided equally
#         between these two substrings.
#         This treats each accepted shot as yielding two samples.

#     Args:
#         dist: dict mapping full bitstrings (e.g., '011100') -> probability.
#         context: TranslationContext with dynamic state tracking.

#     Returns:
#         dict mapping reduced bitstrings (tracked states, or pooled halves) -> probability.
#     """
#     k = context.k_log
#     kept_indices = [p * k + s for (p, s) in context.state_positions]

#     reduced = defaultdict(float)

#     for bitstring, prob in dist.items():
#         n = len(bitstring)

#         # --- Special case: 6-bit global-pass distribution ---
#         if n == 6:
#             # Split into two halves: even and odd indices
#             even_bits = "".join(bitstring[i] for i in range(0, 6, 2))
#             odd_bits  = "".join(bitstring[i] for i in range(1, 6, 2))

#             # Divide probability equally between the two
#             reduced[even_bits] += prob / 2.0
#             reduced[odd_bits]  += prob / 2.0

#         # --- Default case: just keep tracked-state indices ---
#         else:
#             try:
#                 key = "".join(bitstring[i] for i in kept_indices)
#             except IndexError:
#                 raise IndexError(
#                     f"Bitstring length {n} too short for indices {kept_indices}."
#                 )
#             reduced[key] += prob
            
#     return dict(reduced)

def postselect_against_reference_raw(
    translated_circuits: List[TranslationResult],
    ideal_result: SimulationResult,
    noisy_results: List[SimulationResult],
) -> List[PostselectedSimulationResult]:
    """
    Applies Iceberg postselection using a shared ideal result against many noisy results.
    Useful for comparing multiple noisy runs to a single reference.
    """
    postselected_results = []

    for t, result in zip(translated_circuits, noisy_results):
        postselected_noisy_dist, sel_rate = iceberg_postselect(result.noisy_dist, result.key_order)
        postselected_noisy_dist = reduce_distribution(postselected_noisy_dist, t.context)

        logical_fid = logical_fidelity(ideal_result.ideal_dist, postselected_noisy_dist)
        logical_total_variation_distance = total_variation_distance(
            postselected_noisy_dist, ideal_result.ideal_dist
        )

        postselected_results.append(
            PostselectedSimulationResult(
                permutation=t.permutation,
                circuit=t.circuit,
                contains_faults=t.contains_faults,
                raw_result=result,
                postselected_noisy_dist=postselected_noisy_dist,
                postselected_ideal_dist=ideal_result.ideal_dist,
                logical_fidelity=logical_fid,
                logical_selection_rate=sel_rate,
                logical_total_variation_distance=logical_total_variation_distance,
            )
        )

    return postselected_results
