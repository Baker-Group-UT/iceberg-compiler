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
