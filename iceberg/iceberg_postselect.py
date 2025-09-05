"""
iceberg_postselect.py - Defines the logic for postselecting iceberg histograms.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import cirq

__all__ = [
    "ordered_measurement_keys",
    "iceberg_postselect",
    "iceberg_kept_mass",
]

# FIXME: probably need to reorder patch and flag qubits too
def ordered_measurement_keys(circuit: cirq.Circuit) -> List[str]:
    """
    Return measurement keys in the deterministic order used by
    ``leakage_sim.estimate_distribution``, with data_* registers ordered to
    match the simulator's concatenation (highest patch → lowest).
    """
    circuit = cirq.synchronize_terminal_measurements(circuit)
    final_ops = sorted(circuit[-1], key=cirq.measurement_key_name)
    keys = [
        (op.gate.key.name if hasattr(op.gate.key, "name") else op.gate.key)
        for op in final_ops
    ]

    # Split into data_* and everything else (ancillas, flags, etc.)
    data_keys = [k for k in keys if str(k).startswith("data_")]
    other_keys = [k for k in keys if not str(k).startswith("data_")]

    # Sort data_* by their numeric suffix DESC so data_{max},...,data_0
    # This matches observed simulator concatenation: Patch_max, ..., Patch_0
    try:
        data_keys_desc = sorted(
            data_keys,
            key=lambda n: int(str(n).split("_")[1]),
            reverse=True,
        )
    except Exception:
        # Fallback: just reverse the discovered order
        data_keys_desc = list(reversed(data_keys))

    return data_keys_desc + other_keys
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

# ===================== BIG HELPERS (shared) =====================

def _classify_registers(key_order: List[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    From key_order, return (anc_cregs, data_cregs_sorted, flags_key).
    """
    anc = [k for k in key_order if "_anc" in str(k)]
    data = sorted([k for k in key_order if str(k).startswith("data_")],
                  key=lambda n: int(str(n).split("_")[-1]))
    flags = next((k for k in key_order if str(k) == "flags"), None)
    return anc, data, flags

def _build_regmap(
    bits_or_blocks: str | List[str],
    key_order: List[str],
    data_cregs: List[str],
    anc_cregs: List[str],
    flags_key: Optional[str],
    *,
    data_word_len: int = 4,
) -> Dict[str, str]:
    """
    Parse an outcome (concatenated or space-delimited) into {key -> bitstring}.
    Infers flags width as 'remaining bits' after known lengths.
    """
    # Already space-delimited per register
    if isinstance(bits_or_blocks, list):
        return {str(k): v for k, v in zip(key_order, bits_or_blocks)}

    total_len = len(bits_or_blocks)

    # Known lengths: ancilla=1, data=data_word_len, flags=infer remainder
    lens: Dict[str, int | None] = {}
    data_set = set(map(str, data_cregs))
    anc_set = set(map(str, anc_cregs))
    flags_name = str(flags_key) if flags_key is not None else None

    for k in key_order:
        ks = str(k)
        if ks in data_set:
            lens[ks] = data_word_len
        elif ks in anc_set:
            lens[ks] = 1
        elif flags_name is not None and ks == flags_name:
            lens[ks] = None  # infer later
        else:
            lens[ks] = 1  # defensive default

    if flags_name is not None:
        known = sum(int(v) for v in lens.values() if v is not None)
        rem = total_len - known
        if rem < 0:
            raise ValueError(f"Outcome shorter than expected: total={total_len}, known={known}")
        lens[flags_name] = rem

    regmap: Dict[str, str] = {}
    pos = 0
    for k in key_order:
        ks = str(k)
        n = int(lens[ks])  # type: ignore[arg-type]
        regmap[ks] = bits_or_blocks[pos : pos + n]
        pos += n
    return regmap

def _valid_data_blocks(
    regmap: Dict[str, str],
    data_cregs: List[str],
    mapping: Dict[str, str],
    *,
    data_word_len: int = 4,
) -> bool:
    """
    Check each data block has the right length, even parity, and maps via `mapping`.
    """
    for dk in data_cregs:
        word = regmap[str(dk)]
        if len(word) != data_word_len or (word.count("0") % 2 == 1):
            return False
        if mapping.get(word) is None:
            return False
    return True

# ===================== PUBLIC FUNCTIONS =====================

def iceberg_postselect(dist: Dict[str, float], key_order: List[str]) -> Dict[str, float]:
    """
    Collapse 4-qubit physical codewords → 2-qubit logical outcomes per Iceberg
    patch and discard any shot whose ancilla registers are non-zero OR any bit
    in the 'flags' register is 1.
    """
    if not dist:
        return {}

    anc_cregs, data_cregs, flags_key = _classify_registers(key_order)
    logical_dist: Dict[str, float] = {}

    for outcome, p in dist.items():
        tokens = outcome.split(" ") if " " in outcome else outcome
        regmap = _build_regmap(tokens, key_order, data_cregs, anc_cregs, flags_key)

        # reject if any ancilla bit is non-zero
        if any(regmap[str(a)] != "0" for a in anc_cregs):
            continue

        # reject if any flag bit is 1
        if flags_key is not None:
            if any(b == "1" for b in regmap.get(str(flags_key), "")):
                continue

        # validate data blocks
        if not _valid_data_blocks(regmap, data_cregs, _PHYS2LOGIC, data_word_len=4):
            continue

        # build logical string (then reverse for legacy little-endian)
        logical = "".join(_PHYS2LOGIC[regmap[dk]] for dk in data_cregs)[::-1]
        logical_dist[logical] = logical_dist.get(logical, 0.0) + float(p)

    Z = sum(logical_dist.values()) or 1.0
    return {k: v / Z for k, v in logical_dist.items()}

def iceberg_kept_mass(dist: Dict[str, float], key_order: List[str]) -> float:
    """
    Return the total probability mass retained by Iceberg post-selection
    (ancillas==0, flags all zero, and valid 4→2 codeword projection),
    without renormalizing.
    """
    if not dist:
        return 0.0

    anc_cregs, data_cregs, flags_key = _classify_registers(key_order)
    kept = 0.0

    for outcome, p in dist.items():
        tokens = outcome.split(" ") if " " in outcome else outcome
        regmap = _build_regmap(tokens, key_order, data_cregs, anc_cregs, flags_key)

        # ancillas must be zero
        if any(regmap[str(a)] != "0" for a in anc_cregs):
            continue

        # flags (if present) must be all zero
        if flags_key is not None:
            if any(b == "1" for b in regmap.get(str(flags_key), "")):
                continue

        # data blocks must be valid codewords
        if not _valid_data_blocks(regmap, data_cregs, _PHYS2LOGIC, data_word_len=4):
            continue

        kept += float(p)

    return kept