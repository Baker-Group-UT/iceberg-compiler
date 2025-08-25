"""
result_row.py - Extracts all dataframe row values from simulation output.
"""

from __future__ import annotations

from typing import Dict, Any
from pipeline.analysis.result_types import PostselectedSimulationResult
from pipeline.config import PipelineConfig
from evaluator.gate_count_stats import extract_gate_count, total_gates


def result_to_row(sim_result: PostselectedSimulationResult, pipeline_config: PipelineConfig) -> Dict[str, Any]:
    """
    Converts a PostselectedSimulationResult into a full-featured row dictionary
    for dataframe construction.

    Args:
        sim_result: A single simulation result after postselection.

    Returns:
        Dictionary containing all metrics needed for result analysis.
    """
    perm_key = tuple(tuple(p) for p in sim_result.permutation)
    codes_used = len({p for p, _ in sim_result.permutation})

    gate_count = extract_gate_count(sim_result.circuit)
    total_gate_count = total_gates(gate_count)

    return {
        # Core identifiers
        "permutation": sim_result.permutation,
        "perm_key": perm_key,
        "codes_used": codes_used,

        # Fidelity + error
        "fidelity": sim_result.logical_fidelity,
        "logical_error_rate": 1.0 - sim_result.logical_fidelity,
        "postselected": sim_result.was_postselected,

        # Distributions (baseline, raw, postselected)
        "ideal_dist": sim_result.raw_result.ideal_raw_dist,
        "noisy_dist": sim_result.raw_result.noisy_raw_dist,
        "postselected_dist": sim_result.postselected_noisy_dist,

        # Logical postselection metrics
        "logical_fidelity": sim_result.logical_fidelity,
        "logical_selection_rate": sim_result.logical_selection_rate,
        "selection_rate_overall": sim_result.selection_rate_overall,
        "logical_total_variation_distance": sim_result.logical_total_variation_distance,

        # Gate metrics
        "gate_count": gate_count,
        "total_gates": total_gate_count,
        "contains_faults": sim_result.contains_faults,
        "fault_tolerant": pipeline_config.fault_tolerant,
    }
