"""
decode.py -- two-level decoding of a concatenated circuit's measurement record.

Every quantity is a parity of measured bits, so a decoder is a set of index sets and decoding
is XOR over columns of a (shots x records) bit matrix. The same maps decode real samples (bits)
and Pauli-frame flips (bits relative to the noiseless run): every check is 0 in the noiseless
circuit, so a check *fails* exactly when its parity is 1 in either reading.

Inner level, per inner patch j (registers m_parity_j, m_data_j of C2):
    checks   both syndrome bits, and the parity of the four data bits (the Z-stabilizer read off
             the final data measurement)
    values   the Z measurement of each C1 qubit = Zbar of the inner lane holding it at the end,
             the parity of the two data bits in `CodeLayout.z_support(slot)`
Outer level, per outer patch o (registers m_parity_o, m_data_o of C1, each bit now a C1 qubit's
inner-decoded value):
    checks   the two syndrome bits, and the data codeword parity
    outputs  program qubit i -> parity of the two outer data bits in z_support of its final slot

Two decoders:
    postselect  keep a shot only if every inner and every outer check passes (the paper's rule,
                applied at both levels)
    erasure     inner checks do not discard; a failed inner patch marks the C1 qubits it holds as
                *erased* -- located, unreliable. Per outer patch: no erasure -> post-select on its
                checks; erased syndrome ancillas only -> drop their checks, keep the rest; one
                erased data qubit -> drop the patch's checks and replace the erased bit by the
                parity of the other three (a distance-2 code corrects one located erasure), so
                its outputs are read from qubits the flagged inner patch did not touch; two or
                more erased data qubits -> discard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from iceberg.layout import CodeLayout

INNER = CodeLayout(4)
OUTER = CodeLayout(4)


@dataclass
class Maps:
    records: List[Tuple[str, int]]
    inner_patches: int
    #: (inner patch, record indices) -- each parity must be 0
    inner_checks: List[Tuple[int, Tuple[int, ...]]]
    #: per C1 (flat) qubit: record indices whose parity is its measured value
    flat_value: List[Tuple[int, ...]]
    #: per C1 qubit: inner patch holding it at the end
    flat_home: List[int]
    #: per outer patch: flat qubits of its syndrome bits, and of its data bits in data order
    outer_parity: Dict[int, List[int]]
    outer_data: Dict[int, List[int]]
    #: per program qubit: (outer patch, (data index, data index)) of its Zbar support
    outputs: List[Tuple[int, Tuple[int, int]]]


def records_of(circuit) -> List[Tuple[str, int]]:
    from experiments.revision.exp8_undetected.frames import _records

    return _records(circuit)


def _by_register(records):
    rows: Dict[str, Dict[int, int]] = {}
    for row, (name, index) in enumerate(records):
        rows.setdefault(name, {})[index] = row
    return rows


def maps_for(two) -> Maps:
    """Decoder maps for a `construct.TwoLevel`."""
    records = records_of(two.inner.circuit)
    rows = _by_register(records)
    inner_checks: List[Tuple[int, Tuple[int, ...]]] = []
    patches = sorted(int(name.rsplit("_", 1)[1]) for name in rows if name.startswith("m_data_"))
    for j in patches:
        parity = rows[f"m_parity_{j}"]
        data = rows[f"m_data_{j}"]
        for bit in sorted(parity):
            inner_checks.append((j, (parity[bit],)))
        inner_checks.append((j, tuple(data[d] for d in sorted(data))))
    # mid-circuit rounds (`m_parity_round{r}_{patch}`): each outcome is a check of its patch
    for name in sorted(rows):
        if name.startswith("m_parity_round"):
            j = int(name.rsplit("_", 1)[1])
            for bit in sorted(rows[name]):
                inner_checks.append((j, (rows[name][bit],)))
    flat_value: List[Tuple[int, ...]] = []
    flat_home: List[int] = []
    for patch, slot in two.inner.context.state_positions:
        data = rows[f"m_data_{patch}"]
        anchor, lane = INNER.z_support(slot)
        flat_value.append((data[anchor], data[lane]))
        flat_home.append(int(patch))
    outer_parity: Dict[int, List[int]] = {}
    outer_data: Dict[int, List[int]] = {}
    for (name, bit), flat in two.flat.measured.items():
        patch = int(name.rsplit("_", 1)[1])
        target = outer_parity if name.startswith("m_parity_") else outer_data
        target.setdefault(patch, {})[bit] = flat
    outer_parity = {p: [v[b] for b in sorted(v)] for p, v in outer_parity.items()}
    outer_data = {p: [v[b] for b in sorted(v)] for p, v in outer_data.items()}
    outputs = [(int(patch), OUTER.z_support(slot)) for patch, slot in
               two.outer.context.state_positions]
    return Maps(records=records, inner_patches=len(patches), inner_checks=inner_checks,
                flat_value=flat_value, flat_home=flat_home, outer_parity=outer_parity,
                outer_data=outer_data, outputs=outputs)


def _parity(bits: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    out = np.zeros(bits.shape[0], dtype=bool)
    for c in columns:
        out ^= bits[:, c]
    return out


def decode(bits: np.ndarray, maps: Maps) -> Dict[str, np.ndarray]:
    """
    `bits`: (shots x records) bool, real outcomes or flips. Returns, per decoder, a `keep` mask and
    the decoded program bits (shots x program qubits) -- outcomes, or output flips for frames.
    """
    shots = bits.shape[0]
    inner_fail = np.zeros((shots, maps.inner_patches), dtype=bool)
    patch_column = {}
    for patch, columns in maps.inner_checks:
        column = patch_column.setdefault(patch, len(patch_column))
        inner_fail[:, column] |= _parity(bits, columns)
    any_inner = inner_fail.any(axis=1)
    values = np.stack([_parity(bits, cols) for cols in maps.flat_value], axis=1)
    erased = np.stack([inner_fail[:, patch_column[home]] for home in maps.flat_home], axis=1)

    keep_post = ~any_inner
    keep_erasure = np.ones(shots, dtype=bool)
    corrected = values.copy()
    for patch, data in maps.outer_data.items():
        parity = maps.outer_parity[patch]
        data_fail = np.zeros(shots, dtype=bool)
        for q in data:
            data_fail ^= values[:, q]
        parity_fail = [values[:, q] for q in parity]
        checks_fail = data_fail | np.logical_or.reduce(parity_fail)
        keep_post &= ~checks_fail

        erased_data = erased[:, data]
        erased_parity = erased[:, parity]
        n_data = erased_data.sum(axis=1)
        clean = (n_data == 0) & ~erased_parity.any(axis=1)
        only_parity = (n_data == 0) & erased_parity.any(axis=1)
        one_data = n_data == 1
        # no erasure: every check; erased ancillas only: the checks they do not carry
        surviving_parity = np.zeros(shots, dtype=bool)
        for column, fail in enumerate(parity_fail):
            surviving_parity |= fail & ~erased_parity[:, column]
        ok = (clean & ~checks_fail) | (only_parity & ~(data_fail | surviving_parity)) | one_data
        keep_erasure &= ok
        # one erased data qubit: complete its bit from the other three
        for position, q in enumerate(data):
            others = [d for d in data if d != q]
            completion = np.zeros(shots, dtype=bool)
            for d in others:
                completion ^= values[:, d]
            replace = one_data & erased_data[:, position]
            corrected[replace, q] = completion[replace]

    def outputs(source: np.ndarray) -> np.ndarray:
        columns = []
        for patch, (a, b) in maps.outputs:
            data = maps.outer_data[patch]
            columns.append(source[:, data[a]] ^ source[:, data[b]])
        return np.stack(columns, axis=1)

    return {"postselect": keep_post, "postselect_out": outputs(values),
            "erasure": keep_erasure, "erasure_out": outputs(corrected),
            "inner_flagged": inner_fail.sum(axis=1)}


# ----------------------------------------------------------------------------------------
# single level, for the reference arms
# ----------------------------------------------------------------------------------------


@dataclass
class SingleMaps:
    records: List[Tuple[str, int]]
    checks: List[Tuple[int, ...]]
    outputs: List[Tuple[int, int]]


def single_maps_for(result) -> SingleMaps:
    records = records_of(result.circuit)
    rows = _by_register(records)
    checks: List[Tuple[int, ...]] = []
    for name, by_index in rows.items():
        if name.startswith("m_parity_") or "flag" in name:
            checks.extend((by_index[i],) for i in sorted(by_index))
        elif name.startswith("m_data_"):
            checks.append(tuple(by_index[i] for i in sorted(by_index)))
    outputs = []
    for patch, slot in result.context.state_positions:
        data = rows[f"m_data_{patch}"]
        anchor, lane = INNER.z_support(slot)
        outputs.append((data[anchor], data[lane]))
    return SingleMaps(records=records, checks=checks, outputs=outputs)


def decode_single(bits: np.ndarray, maps: SingleMaps) -> Dict[str, np.ndarray]:
    fail = np.zeros(bits.shape[0], dtype=bool)
    for columns in maps.checks:
        fail |= _parity(bits, columns)
    out = np.stack([bits[:, a] ^ bits[:, b] for a, b in maps.outputs], axis=1)
    return {"postselect": ~fail, "postselect_out": out}
