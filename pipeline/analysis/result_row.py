"""
result_row.py - Extracts all dataframe row values from simulation output.
"""

from __future__ import annotations

from typing import Dict, Any
import math
from pipeline.analysis.result_types import PostselectedSimulationResult
from pipeline.config import PipelineConfig
from evaluator.gate_count_stats import (
    extract_gate_count,
    total_gates,
    one_two_qubit_swap_totals,
    gate_counts_by_code,
    depth_by_code,
)
from packer.work_table import DEFAULT_WORK_WEIGHTS


def result_to_row(sim_result: PostselectedSimulationResult, pipeline_config: PipelineConfig) -> Dict[str, Any]:
    """
    Converts a PostselectedSimulationResult into a full-featured row dictionary
    for dataframe construction.

    Args:
        sim_result: A single simulation result after postselection.

    Returns:
        Dictionary containing all metrics needed for result analysis.
    """
    if sim_result.permutation is None:
        perm_key = "no_perm_key"
        codes_used = -1
    else:
        perm_key = tuple(tuple(p) for p in sim_result.permutation)
        codes_used = len({p for p, _ in sim_result.permutation})

    gate_count = extract_gate_count(sim_result.circuit)
    total_gate_count = total_gates(gate_count)
    one_q_gates, two_q_gates, swap_gates = one_two_qubit_swap_totals(sim_result.circuit)
    circuit_depth = sim_result.circuit.depth()

    w1, w2 = DEFAULT_WORK_WEIGHTS
    gate_counts = gate_counts_by_code(sim_result.circuit, qubits_per_code=6)
    work_by_code = {
        code: (w1 * one_q + w2 * two_q) for code, (one_q, two_q) in gate_counts.items()
    }
    total_work = sum(work_by_code.values())
    if total_work <= 0 or len(work_by_code) <= 1:
        code_equality_entropy = 0.0
    else:
        n_codes = len(work_by_code)
        shares = [work / total_work for work in work_by_code.values() if work > 0]
        entropy = -sum(p * math.log(p) for p in shares)
        max_entropy = math.log(n_codes)
        code_equality_entropy = (entropy / max_entropy if max_entropy > 0 else 0.0)

    depth_counts = depth_by_code(sim_result.circuit, qubits_per_code=6)
    total_depth = sum(depth_counts.values())
    if total_depth <= 0 or len(depth_counts) <= 1:
        depth_equality_entropy = 0.0
    else:
        n_codes = len(depth_counts)
        shares = [depth / total_depth for depth in depth_counts.values() if depth > 0]
        entropy = -sum(p * math.log(p) for p in shares)
        max_entropy = math.log(n_codes)
        depth_equality_entropy = (entropy / max_entropy if max_entropy > 0 else 0.0)

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
        "ideal_dist": sim_result.raw_result.ideal_dist,
        "noisy_dist": sim_result.raw_result.noisy_dist,
        "postselected_dist": sim_result.postselected_noisy_dist,

        # Logical postselection metrics
        "logical_fidelity": sim_result.logical_fidelity,
        "logical_selection_rate": sim_result.logical_selection_rate,
        "selection_rate_overall": sim_result.selection_rate_overall,
        "logical_total_variation_distance": sim_result.logical_total_variation_distance,

        # Gate metrics
        "gate_count": gate_count,
        "total_gates": total_gate_count,
        "one_qubit_gates": one_q_gates,
        "two_qubit_gates": two_q_gates,
        "swap_gates": swap_gates,
        "circuit_depth": circuit_depth,
        "code_equality_entropy": code_equality_entropy,
        "depth_equality_entropy": depth_equality_entropy,
        "contains_faults": sim_result.contains_faults,
        "fault_tolerant": pipeline_config.fault_tolerant,
        "merge_savings_1q": sim_result.merge_savings_1q,
        "merge_savings_2q": sim_result.merge_savings_2q,
        "merge_savings_weighted": sim_result.merge_savings_weighted,
    }
