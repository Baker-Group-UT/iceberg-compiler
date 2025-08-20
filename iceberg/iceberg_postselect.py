"""
iceberg_postselect.py - Defines the logic for postselecting iceberg histograms.
"""

from __future__ import annotations

from typing import Dict, List

import cirq

__all__ = [
    "ordered_measurement_keys",
    "iceberg_postselect",
    "iceberg_kept_mass",
]

def ordered_measurement_keys(circuit: cirq.Circuit) -> List[str]:
    """
    Return measurement keys in the deterministic order used by
    ``leakage_sim.estimate_distribution``.
    """
    circuit = cirq.synchronize_terminal_measurements(circuit)
    final_ops = sorted(circuit[-1], key=cirq.measurement_key_name)
    return [
        (op.gate.key.name if hasattr(op.gate.key, "name") else op.gate.key)
        for op in final_ops
    ]

# ------------------------------------------------------------------
# Iceberg-specific logic
# ------------------------------------------------------------------
_PHYS2LOGIC = {
    "0000": "00",
    "1111": "00",
    "0011": "01",
    "1100": "01",
    "0101": "10",
    "1010": "10",
    "0110": "11",
    "1001": "11",
}

def iceberg_postselect(
    dist: Dict[str, float], key_order: List[str]
) -> Dict[str, float]:
    """
    Collapse 4-qubit physical codewords → 2-qubit logical outcomes per Iceberg
    patch and discard any shot whose ancilla registers are non-zero.

    Expects registers to be named appropriately. 00_anc, 01_anc, ... for ancilla. Data marked as data_ 
    """
    logical_dist: Dict[str, float] = {}

    anc_cregs = [k for k in key_order if "_anc" in k]
    data_cregs = sorted(
        [k for k in key_order if k.startswith("data_")],
        key=lambda n: int(n.split("_")[-1]),
    )

    # -- helper: build reg-map for one outcome string -----------------------
    def _build_regmap(bits_or_blocks: str | list[str]) -> Dict[str, str]:
        if isinstance(bits_or_blocks, list):  # space-delimited from simulator
            return dict(zip(key_order, bits_or_blocks))

        regmap: Dict[str, str] = {}
        pos = 0
        for k in key_order:
            n = 1 if "_anc" in k else 4
            regmap[k] = bits_or_blocks[pos : pos + n]
            pos += n
        return regmap

    # -- iterate over Monte-Carlo outcomes ----------------------------------
    for outcome, p in dist.items():
        regmap = (
            _build_regmap(outcome.split(" ")) if " " in outcome else _build_regmap(outcome)
        )

        # 1. reject leaked ancillas
        if any(regmap[a] != "0" for a in anc_cregs):
            continue

        # 2. map data codewords → logical bits
        logical = ""
        for dk in data_cregs:  # data_0, data_1, …
            word = regmap[dk]
            if word.count("0") % 2:
                logical = None
                break
            mapped = _PHYS2LOGIC.get(word)
            if mapped is None:
                logical = None
                break
            logical += mapped

        if logical is None:
            continue

        logical = logical[::-1]  # legacy little-endian ordering
        logical_dist[logical] = logical_dist.get(logical, 0.0) + p

    Z = sum(logical_dist.values()) or 1.0
    return {k: v / Z for k, v in logical_dist.items()}

def iceberg_kept_mass(dist: Dict[str, float], key_order: List[str]) -> float:
    """
    Return the total probability mass retained by Iceberg post-selection
    (ancillas==0 and valid 4→2 codeword projection), without renormalizing.

    `dist` MUST be the distribution *after* removing leakage (i.e., built from
    raw by dropping outcomes containing '2'), and its values must still be the
    original (pre-renormalization) probabilities.
    """
    if not dist:
        return 0.0

    anc_cregs = [k for k in key_order if "_anc" in str(k)]
    data_cregs = sorted(
        [k for k in key_order if str(k).startswith("data_")],
        key=lambda n: int(str(n).split("_")[-1]),
    )

    def _build_regmap(bits_or_blocks: str | list[str]) -> Dict[str, str]:
        # Simulator sometimes returns space-delimited blocks per key
        if isinstance(bits_or_blocks, list):
            return dict(zip(key_order, bits_or_blocks))
        regmap: Dict[str, str] = {}
        pos = 0
        for k in key_order:
            n = 1 if "_anc" in str(k) else 4
            regmap[str(k)] = bits_or_blocks[pos : pos + n]
            pos += n
        return regmap

    kept = 0.0
    for outcome, p in dist.items():
        regmap = (
            _build_regmap(outcome.split(" "))
            if " " in outcome
            else _build_regmap(outcome)
        )

        # 1) ancillas must be 0
        if any(regmap[str(a)] != "0" for a in anc_cregs):
            continue

        # 2) each data block must be a valid codeword with even parity
        valid = True
        for dk in data_cregs:
            word = regmap[str(dk)]
            if word.count("0") % 2 == 1:
                valid = False
                break
            if _PHYS2LOGIC.get(word) is None:
                valid = False
                break

        if valid:
            kept += float(p)

    return kept
