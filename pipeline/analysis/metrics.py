"""
logical_metrics.py - Computes postselected fidelity and selection rates in logical space.
"""

from simulator.utils import bhattacharyya
from iceberg.iceberg_postselect import iceberg_kept_mass


def logical_fidelity(post_ideal: dict[str, float], post_noisy: dict[str, float]) -> float:
    """
    Bhattacharyya fidelity between postselected ideal and postselected noisy logical distributions.
    """
    if not post_ideal or not post_noisy:
        return 0.0
    return bhattacharyya(post_ideal, post_noisy)

def logical_selection_rate(noisy_dist: dict[str, float],  key_order: list[str]) -> float:
    """
    Fraction of leakage-free shots retained after Iceberg postselection.

    Args:
        noisy_dist: Raw simulator output (may include leakage).
        key_order: Measurement key order for Iceberg PS.

    Returns:
        selection_rate_iceberg_given_no_leak:
            = mass accepted by Iceberg / mass with no leakage
    """
    if not noisy_dist:
        return 0.0

    # Step 1: Filter out leakage
    total_percent= sum(noisy_dist.values())
    if total_percent == 0:
        return 0.0

    # Step 2: Iceberg postselection on no-leak states
    kept_mass = iceberg_kept_mass(noisy_dist, key_order)
    return kept_mass

def total_variation_distance(post_noisy: dict[str, float], post_ideal: dict[str, float]) -> float:
    """
    Computes the total variation distance (TVD) between two probability distributions.

    TVD(P, Q) = 0.5 * sum_x |P(x) - Q(x)|

    Args:
        noisy_dist: Empirical (noisy) probability distribution after postselection.
        ideal_dist: Reference (ideal) probability distribution after postselection.

    Returns:
        A float between 0 and 1 representing the total variation distance.
    """
    if not post_noisy or not post_ideal:
        return 1.0  # Max distance if either is empty

    keys = set(post_noisy) | set(post_ideal)
    return 0.5 * sum(abs(post_noisy.get(k, 0.0) - post_ideal.get(k, 0.0)) for k in keys)
