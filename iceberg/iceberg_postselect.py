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

from typing import Dict, List

def iceberg_postselect(
    dist: Dict[str, float], key_order: List[str]
) -> Dict[str, float]:
    """
    Collapse 4-qubit physical codewords → 2-qubit logical outcomes per Iceberg
    patch and discard any shot whose ancilla registers are non-zero OR any bit
    in the 'flags' register is 1.

    Registers expected:
      - data_{p}     (length 4 each, for n=4 codes)
      - patch_{p}_anc0, patch_{p}_anc1 (length 1 each)
      - flags        (variable length; all flag qubits measured together)

    dist: maps outcome strings to probabilities. Outcome may be a single
    concatenated bitstring or a space-delimited string per register in key_order.
    """
    logical_dist: Dict[str, float] = {}

    # classify registers
    anc_cregs = [k for k in key_order if "_anc" in k]
    data_cregs = sorted(
        [k for k in key_order if k.startswith("data_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    flags_key = next((k for k in key_order if k == "flags"), None)

    # -- helper: build reg-map for one outcome string -----------------------
    def _build_regmap(bits_or_blocks: str | List[str]) -> Dict[str, str]:
        # Space-delimited: already per-register
        if isinstance(bits_or_blocks, list):
            return dict(zip(key_order, bits_or_blocks))

        total_len = len(bits_or_blocks)

        # Known lengths: ancilla=1, data=4 (for n=4 per-module codes)
        lens: Dict[str, int | None] = {}
        for k in key_order:
            if k in data_cregs:
                lens[k] = 4
            elif k in anc_cregs:
                lens[k] = 1
            elif flags_key is not None and k == flags_key:
                lens[k] = None  # infer after summing knowns
            else:
                # default unknowns to 1-bit (defensive)
                lens[k] = 1

        # Infer flags length as "everything left over"
        if flags_key is not None:
            known = sum(int(v) for v in lens.values() if v is not None)
            rem = total_len - known
            if rem < 0:
                raise ValueError(
                    f"Outcome shorter than expected: total={total_len}, known={known}"
                )
            lens[flags_key] = rem

        # Slice
        regmap: Dict[str, str] = {}
        pos = 0
        for k in key_order:
            n = int(lens[k])  # type: ignore[arg-type]
            regmap[k] = bits_or_blocks[pos : pos + n]
            pos += n
        return regmap

    # -- iterate over Monte-Carlo outcomes ----------------------------------
    for outcome, p in dist.items():
        regmap = (
            _build_regmap(outcome.split(" "))
            if " " in outcome
            else _build_regmap(outcome)
        )

        # 1) reject if any ancilla bit is non-zero
        if any(regmap[a] != "0" for a in anc_cregs):
            continue

        # 2) reject if any flag bit is 1
        if flags_key is not None:
            flags_bits = regmap.get(flags_key, "")
            if any(b == "1" for b in flags_bits):
                continue

        # 3) map each data block (4-bit word) → 1 logical bit
        logical = ""
        for dk in data_cregs:  # data_0, data_1, …
            word = regmap[dk]
            # Quick sanity checks consistent with n=4 [[4,2,2]] code structure
            if len(word) != 4 or (word.count("0") % 2):
                logical = None
                break
            mapped = _PHYS2LOGIC.get(word)  # expects a dict[str, str]
            if mapped is None:
                logical = None
                break
            logical += mapped

        if logical is None:
            continue

        # Legacy little-endian ordering
        logical = logical[::-1]
        logical_dist[logical] = logical_dist.get(logical, 0.0) + p

    # Normalize
    Z = sum(logical_dist.values()) or 1.0
    return {k: v / Z for k, v in logical_dist.items()}

def iceberg_kept_mass(dist: Dict[str, float], key_order: List[str]) -> float:
    """
    Return the total probability mass retained by Iceberg post-selection
    (ancillas==0, flags all zero, and valid 4→2 codeword projection),
    without renormalizing.

    `dist` MUST be the distribution *after* removing leakage (i.e., built from
    raw by dropping outcomes containing '2'), and its values must still be the
    original (pre-renormalization) probabilities.
    """
    if not dist:
        return 0.0

    # Classify registers
    anc_cregs = [k for k in key_order if "_anc" in str(k)]
    data_cregs = sorted(
        [k for k in key_order if str(k).startswith("data_")],
        key=lambda n: int(str(n).split("_")[-1]),
    )
    flags_key = next((k for k in key_order if str(k) == "flags"), None)

    # Helper to build per-register map from an outcome
    def _build_regmap(bits_or_blocks: str | List[str]) -> Dict[str, str]:
        # Already space-delimited into per-register blocks
        if isinstance(bits_or_blocks, list):
            return {str(k): v for k, v in zip(key_order, bits_or_blocks)}

        total_len = len(bits_or_blocks)

        # Known lengths for n=4 codes: data=4, anc=1; flags = remainder
        lens: Dict[str, int | None] = {}
        for k in key_order:
            ks = str(k)
            if ks in (str(dc) for dc in data_cregs):
                lens[ks] = 4
            elif ks in (str(ac) for ac in anc_cregs):
                lens[ks] = 1
            elif flags_key is not None and ks == str(flags_key):
                lens[ks] = None  # infer after summing knowns
            else:
                lens[ks] = 1  # defensive default

        if flags_key is not None:
            known = sum(int(v) for v in lens.values() if v is not None)
            rem = total_len - known
            if rem < 0:
                raise ValueError(
                    f"Outcome shorter than expected: total={total_len}, known={known}"
                )
            lens[str(flags_key)] = rem

        regmap: Dict[str, str] = {}
        pos = 0
        for k in key_order:
            ks = str(k)
            n = int(lens[ks])  # type: ignore[arg-type]
            regmap[ks] = bits_or_blocks[pos : pos + n]
            pos += n
        return regmap

    kept = 0.0
    for outcome, p in dist.items():
        regmap = (
            _build_regmap(outcome.split(" "))
            if " " in outcome
            else _build_regmap(outcome)
        )

        # 1) ancillas must be all zero
        if any(regmap[str(a)] != "0" for a in anc_cregs):
            continue

        # 2) flags (if present) must be all zero
        if flags_key is not None:
            flags_bits = regmap.get(str(flags_key), "")
            if any(b == "1" for b in flags_bits):
                continue

        # 3) each data block must be a valid 4-bit codeword with even parity and known mapping
        valid = True
        for dk in data_cregs:
            word = regmap[str(dk)]
            if len(word) != 4 or (word.count("0") % 2 == 1):
                valid = False
                break
            if _PHYS2LOGIC.get(word) is None:
                valid = False
                break

        if valid:
            kept += float(p)

    return kept