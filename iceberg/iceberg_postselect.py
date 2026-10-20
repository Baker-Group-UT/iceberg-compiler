"""
iceberg_postselect.py — classical postselection and decoding of Iceberg results.

Postselection keeps a shot only if every syndrome bit is 0: the two parity bits
of each patch (`m_parity_*`) and every flag ancilla of the fault-tolerant gadgets
(`flag*`). Surviving shots have their data registers (`m_data_*`) decoded to
logical bitstrings.

Bit ordering
------------
Worth being explicit, because getting this wrong produces a plausible-looking
distribution rather than an error — and n=4 cannot tell the two readings apart
(for a 4-bit even-parity codeword, reversing the bits leaves the decoded logical
value unchanged, so the k=2 lookup table this module used to carry was silently
reversal-invariant).

* `simulator.Simulator` inverts Aer's little-endian output, so in the strings
  that arrive here position `i` of a register *is* bit `i` — no reversal — and
  register groups appear in `circuit.cregs` order. `IcebergModule.measure` maps
  data qubit `d` to bit `d`, so position `d` is data qubit `d`.
* Downstream (`pipeline.analysis.postselection.reduce_distribution`) indexes the
  decoded logical string as `patch * k + slot`, left to right. So this function
  emits slot 0 first within each patch, and patches in `key_order` order.

Decoding itself is `CodeLayout.decode_logical`: the logical value of slot i is
the parity of the two measured bits in the support of Zbar_i. Decoding against
the X-anchor instead — the natural-looking `bit(lane) XOR bit(data_0)` rule —
silently applies the patch-wide map C_A to the result. At k=2, C_A is the lane
swap (0 1), so a lookup table built by hand from k=2 data absorbs the error into
what looks like an endianness choice; at k > 2 it corrupts the distribution.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .layout import CodeLayout

__all__ = ["iceberg_postselect", "decode_data_register", "split_qiskit_bitstring"]


def split_qiskit_bitstring(bitstring: str, key_order: List[str]) -> Dict[str, str]:
    """
    Split a space-delimited Qiskit bitstring into a dict keyed by register name.

    Example:
        '00 10 0101 1111' + ['m_parity_0','m_parity_1','m_data_0','m_data_1']
        -> {'m_parity_0':'00','m_parity_1':'10','m_data_0':'0101','m_data_1':'1111'}
    """
    parts = bitstring.split()
    if len(parts) != len(key_order):
        raise ValueError(
            f"Bitstring/register count mismatch: {len(parts)} vs {len(key_order)}"
        )
    return dict(zip(key_order, parts))


def decode_data_register(register: str) -> str | None:
    """
    Decode one patch's data register into its logical bitstring, slot 0 first.

    Returns None if the outcome is not a codeword (odd parity), which is the
    Z-stabilizer check applied to the final data measurement.
    """
    layout = CodeLayout(len(register))
    bits = [int(b) for b in register]  # position i is data qubit i
    if not layout.is_codeword(bits):
        return None
    return "".join(str(b) for b in layout.decode_logical(bits))


def _decode_all(registers: Dict[str, str], data_keys: List[str]) -> str | None:
    """Concatenate the decoded patches in `data_keys` order, or None if any fails."""
    decoded: List[str] = []
    for key in data_keys:
        patch = decode_data_register(registers[key])
        if patch is None:
            return None
        decoded.append(patch)
    return "".join(decoded)


# ===================== PUBLIC FUNCTIONS =====================

# TODO: expand postselect to give more information about percent keywords leaked?
def iceberg_postselect(
    dist: Dict[str, float],
    key_order: List[str],
) -> Tuple[Dict[str, float], float]:
    """
    Postselect and decode a Qiskit-style measurement distribution.

    Args:
        dist: probability distribution {bitstring: p}.
        key_order: register names in output order.

    Returns:
        (logical_dist, sel_rate) where `logical_dist` is normalized over the
        surviving shots and `sel_rate` is the fraction of probability kept.
    """
    if not dist:
        return {}, 0.0

    data_keys = [k for k in key_order if "data" in k]
    syndrome_keys = [k for k in key_order if ("parity" in k) or ("flag" in k)]

    logical_dist: Dict[str, float] = {}
    kept_mass = 0.0
    total_mass = sum(dist.values()) or 1.0

    for outcome, p in dist.items():
        parts = outcome.split()
        if len(parts) != len(key_order):
            continue
        registers = dict(zip(key_order, parts))

        if any("1" in registers[k] for k in syndrome_keys):
            continue

        logical = _decode_all(registers, data_keys)
        if logical is None:
            continue  # not a codeword

        logical_dist[logical] = logical_dist.get(logical, 0.0) + float(p)
        kept_mass += float(p)

    normalization = sum(logical_dist.values()) or 1.0
    logical_dist = {k: v / normalization for k, v in logical_dist.items()}
    return logical_dist, kept_mass / total_mass
