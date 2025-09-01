"""
simulator/utils.py

Helper functions for distribution postprocessing and fidelity computation.
As well as noise scaling utilities for use with the SQALE leakage simulator.
Includes helpers for creating scaled Hilbert noise parameter sets.
"""

from typing import Dict
from math import sqrt

import cirq
from iceberg.gates import BarrierGate
from experiments_superstaq.simulation.sqale_leakage_model import HilbertNoiseParams, DEFAULT_PARAMS


def remove_leakage_states(
    dist: Dict[str, float], track_stats: bool = False
) -> tuple[Dict[str, float], int, int]:
    """
    Removes leakage ('2') states and returns:

    - Renormalized filtered distribution
    - Shots kept (no leakage)
    - Shots dropped (with leakage)
    """
    kept = {k: v for k, v in dist.items() if "2" not in k}
    dropped = {k: v for k, v in dist.items() if "2" in k}

    total_kept = round(sum(kept.values()), 6)
    total_dropped = round(sum(dropped.values()), 6)

    z = total_kept or 1.0
    filtered_dist = {k: v / z for k, v in kept.items()}
    if track_stats:
        return filtered_dist, total_kept, total_dropped
    else:
        return filtered_dist


def bhattacharyya(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute Bhattacharyya fidelity between two distributions.

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        Classical Bhattacharyya fidelity.
    """
    keys = set(p) | set(q)
    return sum(sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)

def scale_noise_params(
    base: HilbertNoiseParams,
    rz_scalar: float,
    cz_scalar: float,
) -> HilbertNoiseParams:
    """
    Scales only the RZ and CZ transition matrices of the base HilbertNoiseParams.
    Overrotation terms and classifier errors are preserved.

    Args:
        base: Original HilbertNoiseParams object.
        rz_scalar: Scalar applied to RZ transition matrix.
        cz_scalar: Scalar applied to CZ transition matrix.

    Returns:
        New HilbertNoiseParams object with scaled RZ and CZ matrices.
    """
    if base.rz_transition_matrix is None or base.cz_transition_matrix is None:
        raise ValueError("Transition matrices must be initialized in base parameters.")

    scaled_rz = [[x * rz_scalar for x in row] for row in base.rz_transition_matrix]
    scaled_cz = [[x * cz_scalar for x in row] for row in base.cz_transition_matrix]

    return HilbertNoiseParams(
        gr_static_overrotation_rads=0.0,
        rz_relative_overrotation=0.0,
        rz_transition_matrix=scaled_rz,
        cz_transition_matrix=scaled_cz,
        classifier_errors=base.classifier_errors,
        initial_state_probs=base.initial_state_probs,
    )

def alpha_scale_noise_params(
    base: HilbertNoiseParams = DEFAULT_PARAMS,
    alpha: float = 1.0,
) -> HilbertNoiseParams:
    """
    Scales all components of HilbertNoiseParams uniformly by a global alpha.

    Includes transition matrices, overrotation terms, and classifier errors.

    Args:
        base: Original HilbertNoiseParams object.
        alpha: Global scalar applied to all tunable parameters.

    Returns:
        New HilbertNoiseParams object with scaled values.
    """
    if base.rz_transition_matrix is None or base.cz_transition_matrix is None:
        raise ValueError("Transition matrices must be initialized in base parameters.")

    scaled = base.model_copy(deep=True)

    scaled.rz_transition_matrix = [
        [x * alpha for x in row] for row in base.rz_transition_matrix
    ]
    scaled.cz_transition_matrix = [
        [x * alpha for x in row] for row in base.cz_transition_matrix
    ]
    scaled.gr_static_overrotation_rads *= alpha
    scaled.rz_relative_overrotation *= alpha
    scaled.classifier_errors = (
        scaled.classifier_errors[0] * alpha,
        scaled.classifier_errors[1] * alpha,
    )

    return scaled

def remove_barriers(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Returns a copy of the circuit with all BarrierGates removed.
    """
    return cirq.Circuit(
        op for op in circuit.all_operations()
        if not isinstance(op.gate, BarrierGate)
    )
