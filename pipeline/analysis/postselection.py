"""
postselection.py - Applies Iceberg postselection to simulator results.
"""

from __future__ import annotations
from typing import List

from pipeline.analysis.result_types import PostselectedSimulationResult
from pipeline.analysis.metrics import (
    logical_fidelity,
    logical_selection_rate,
    total_variation_distance
)
from simulator.result_types import SqaleSimulationResult

from iceberg.iceberg_postselect import (
    ordered_measurement_keys,
    iceberg_postselect,
)


def apply_iceberg_postselection(
    translated_circuits: List,
    results: List[SqaleSimulationResult],
) -> List[PostselectedSimulationResult]:
    """
    Combines raw simulator results with Iceberg postselection logic.

    Returns a list of enriched PostselectedSimulationResult objects for downstream
    dataframe creation or analysis.
    """
    postselected_results = []

    for t, result in zip(translated_circuits, results):
        key_order = ordered_measurement_keys(t.circuit)

        postselected_noisy_dist = iceberg_postselect(result.noisy_dist, key_order)
        postselected_ideal_dist = iceberg_postselect(result.ideal_dist, key_order)

        logical_fid = logical_fidelity(postselected_ideal_dist, postselected_noisy_dist)
        logical_sel = logical_selection_rate(result.noisy_dist, key_order)
        logical_total_variation_distance = total_variation_distance(
            postselected_noisy_dist,
            postselected_ideal_dist
        )
        selection_rate_overall = result.percent_shots_kept * logical_sel


        postselected_results.append(
            PostselectedSimulationResult(
                permutation=t.permutation,
                circuit=t.circuit,
                contains_faults= t.contains_faults,
                raw_result=result,
                postselected_noisy_dist=postselected_noisy_dist,
                postselected_ideal_dist=postselected_ideal_dist,
                selection_rate_overall=selection_rate_overall,
                was_postselected=True,
                logical_fidelity=logical_fid,
                logical_selection_rate=logical_sel,
                logical_total_variation_distance=logical_total_variation_distance
            )
        )

    return postselected_results
