"""
logical_metrics.py - Computes postselected fidelity and selection rates in logical space.
"""

from simulator.utils import bhattacharyya


def logical_fidelity(post_ideal: dict[str, float], post_noisy: dict[str, float]) -> float:
    """
    Bhattacharyya fidelity between postselected ideal and postselected noisy logical distributions.
    """
    if not post_ideal or not post_noisy:
        return 0.0
    return bhattacharyya(post_ideal, post_noisy)

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
