"""
iceberg_postselect.py - Defines the logic for postselecting iceberg histograms.
"""

from __future__ import annotations
from typing import Dict, List



__all__ = [
    "iceberg_postselect",
]

# ------------------------------------------------------------------
# Iceberg-specific logic
# ------------------------------------------------------------------

# Physical 4-qubit → logical 2-qubit map (even-parity codewords only)
_PHYS2LOGIC = {
    "0000": "00",
    "1111": "00",
    "0011": "10",
    "1100": "10",
    "0101": "01",
    "1010": "01",
    "0110": "11",
    "1001": "11",
}

def split_qiskit_bitstring(
    bitstring: str,
    key_order: list[str],
) -> dict[str, str]:
    """
    Split a space-delimited Qiskit bitstring into a dict keyed by register name.
    Example:
        '00 10 0101 1111' + ['m_parity_0','m_parity_1','m_data_0','m_data_1']
        → {'m_parity_0':'00','m_parity_1':'10','m_data_0':'0101','m_data_1':'1111'}
    """
    parts = bitstring.split()
    if len(parts) != len(key_order):
        raise ValueError(f"Bitstring/register count mismatch: {len(parts)} vs {len(key_order)}")
    return dict(zip(key_order, parts))


# ===================== PUBLIC FUNCTIONS =====================

def iceberg_postselect(dist: Dict[str, float], key_order: List[str]) -> Dict[str, any]:
    """
    Qiskit-style postselection.

    Args:
        dist: probability distribution {bitstring: p}.
        key_order: list of register names in output order.

    Returns:
        {
            "logical_dist": normalized logical distribution,
            "sel_rate": fraction of total probability kept after postselection
        }
    """
    if not dist:
        return {"logical_dist": {}, "sel_rate": 0.0}

    data_keys = [k for k in key_order if "data" in k]
    check_zero_keys = [k for k in key_order if ("parity" in k) or ("flag" in k)]

    logical_dist: Dict[str, float] = {}
    kept_mass = 0.0
    total_mass = sum(dist.values()) or 1.0

    for outcome, p in dist.items():
        parts = outcome.split()
        if len(parts) != len(key_order):
            continue
        regmap = dict(zip(key_order, parts))

        # Reject outcomes with any 1 in parity/flag registers
        if any("1" in regmap[k] for k in check_zero_keys):
            continue

        try:
            logical = "".join(_PHYS2LOGIC[regmap[k]] for k in data_keys)
        except KeyError:
            continue  # invalid codeword pattern

        logical_dist[logical] = logical_dist.get(logical, 0.0) + float(p)
        kept_mass += float(p)

    # normalize surviving probability
    Z = sum(logical_dist.values()) or 1.0
    logical_dist = {k: v / Z for k, v in logical_dist.items()}

    sel_rate = kept_mass / total_mass
    return logical_dist, sel_rate
