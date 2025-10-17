"""
iceberg_postselect.py - Defines the logic for postselecting iceberg histograms.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional


__all__ = [
    "iceberg_postselect",
    "iceberg_kept_mass",
]

# ------------------------------------------------------------------
# Iceberg-specific logic
# ------------------------------------------------------------------

# Physical 4-qubit → logical 2-qubit map (even-parity codewords only)
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

def _classify_registers(key_order: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    From key_order, return (anc_cregs, data_cregs_sorted, flags_cregs).

    Conventions supported:
      - Ancillas: any register whose name contains "_anc" (e.g., "patch0_anc0").
      - Data:     registers named "data_<idx>" (sorted by <idx> ASC for block assembly).
      - Flags:    either a single "flags" register (multi-bit) OR many "flag<id>" 1-bit regs.
                  We collect *all* that start with "flag" and also the literal "flags".
    """
    anc = [k for k in key_order if "_anc" in k]
    data = sorted([k for k in key_order if k.startswith("data_")],
                  key=lambda n: int(n.split("_")[-1]))
    flags: List[str] = []
    # Single multi-bit "flags"
    if "flags" in key_order:
        flags.append("flags")
    # Many single-bit "flag<id>" registers
    flags.extend(sorted([k for k in key_order if k.startswith("flag") and k != "flags"],
                        key=lambda s: int("".join(ch for ch in s if ch.isdigit())) if any(ch.isdigit() for ch in s) else 0))
    return anc, data, flags


def _build_regmap(
    bits_or_blocks: str | List[str],
    key_order: List[str],
    data_cregs: List[str],
    anc_cregs: List[str],
    flags_cregs: List[str],
    *,
    data_word_len: int = 4,
) -> Dict[str, str]:
    """
    Parse an outcome (concatenated or space-delimited) into {key -> bitstring}.
    Known lengths:
      - data registers: data_word_len
      - ancilla registers: 1
      - flag registers:
          * if a single 'flags' entry exists, infer its length as the remainder
          * otherwise, each 'flag<id>' is 1 bit
      - any other register: default to 1 bit (defensive)
    """
    if isinstance(bits_or_blocks, list):
        return {str(k): v for k, v in zip(key_order, bits_or_blocks)}

    total_len = len(bits_or_blocks)
    lens: Dict[str, Optional[int]] = {}

    data_set = set(data_cregs)
    anc_set = set(anc_cregs)
    flags_set = set(flags_cregs)

    has_single_flags = ("flags" in flags_set) and (len(flags_set) == 1)

    for k in key_order:
        if k in data_set:
            lens[k] = data_word_len
        elif k in anc_set:
            lens[k] = 1
        elif k in flags_set:
            lens[k] = None if has_single_flags and k == "flags" else 1
        else:
            lens[k] = 1  # default 1-bit

    # If there's a single 'flags' register, infer its width from remaining bits
    if has_single_flags:
        known = sum(int(v) for v in lens.values() if v is not None)
        rem = total_len - known
        if rem < 0:
            raise ValueError(f"Outcome shorter than expected: total={total_len}, known={known}")
        lens["flags"] = rem

    regmap: Dict[str, str] = {}
    pos = 0
    for k in key_order:
        n = int(lens[k])  # type: ignore[arg-type]
        regmap[k] = bits_or_blocks[pos : pos + n]
        pos += n

    return regmap


def _valid_data_blocks(
    regmap: Dict[str, str],
    data_cregs: List[str],
    mapping: Dict[str, str],
    *,
    data_word_len: int = 4,
) -> bool:
    """Check each data block has the right length, even parity, and maps via `mapping`."""
    for dk in data_cregs:
        word = regmap[dk]
        if len(word) != data_word_len or (word.count("0") % 2 == 1):
            return False
        if mapping.get(word) is None:
            return False
    return True


def _any_flag_set(regmap: Dict[str, str], flags_cregs: List[str]) -> bool:
    """
    Return True if any flag register contains a '1'.
    Supports both a single multi-bit 'flags' and many 1-bit 'flag<id>' regs.
    """
    for fk in flags_cregs:
        if any(b == "1" for b in regmap.get(fk, "")):
            return True
    return False


# ===================== PUBLIC FUNCTIONS =====================

def iceberg_postselect(dist: Dict[str, float], key_order: List[str]) -> Dict[str, float]:
    """
    Collapse 4-qubit physical codewords → 2-qubit logical outcomes per Iceberg
    patch and discard any shot whose ancilla registers are non-zero OR any bit
    in any flags register is 1.

    Inputs:
      dist: mapping from outcome string -> probability mass.
            Outcome may be a single concatenated bitstring OR space-delimited
            per-register blocks in `key_order`.
      key_order: list of register names (from `ordered_measurement_keys`).
    """
    if not dist:
        return {}

    anc_cregs, data_cregs, flags_cregs = _classify_registers(key_order)
    logical_dist: Dict[str, float] = {}

    for outcome, p in dist.items():
        tokens = outcome.split(" ") if " " in outcome else outcome
        regmap = _build_regmap(tokens, key_order, data_cregs, anc_cregs, flags_cregs)

        # reject if any ancilla bit is non-zero
        if any(regmap[a] != "0" for a in anc_cregs):
            continue

        # reject if any flag bit is 1
        if _any_flag_set(regmap, flags_cregs):
            continue

        # validate data blocks
        if not _valid_data_blocks(regmap, data_cregs, _PHYS2LOGIC, data_word_len=4):
            continue

        # build logical string in module order then reverse for legacy little-endian
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

    anc_cregs, data_cregs, flags_cregs = _classify_registers(key_order)
    kept = 0.0

    for outcome, p in dist.items():
        tokens = outcome.split(" ") if " " in outcome else outcome
        regmap = _build_regmap(tokens, key_order, data_cregs, anc_cregs, flags_cregs)

        # ancillas must be zero
        if any(regmap[a] != "0" for a in anc_cregs):
            continue

        # flags (if present) must be all zero
        if _any_flag_set(regmap, flags_cregs):
            continue

        # data blocks must be valid codewords
        if not _valid_data_blocks(regmap, data_cregs, _PHYS2LOGIC, data_word_len=4):
            continue

        kept += float(p)

    return kept
