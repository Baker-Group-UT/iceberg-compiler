"""
lookup.py -- a correcting decoder for the iceberg circuits: exact first-order lookup.

Post-selection discards every shot in which any check fires. This decoder corrects instead
wherever one fault explains what the checks saw:

  1. Every single-fault mechanism of the noisy Stim circuit is listed: each Pauli component of
     every DEPOLARIZE1 / DEPOLARIZE2 / PAULI_CHANNEL_1 / X_ERROR / Y_ERROR / Z_ERROR, with its
     probability (`mechanisms`).
  2. Each is injected alone into a Pauli-frame simulation (`single_fault_flips`: one
     `stim.FlipSimulator` lane per mechanism), giving exactly which measurement records it
     flips -- through every later gate, mid-circuit round and readout.
  3. The records give the mechanism's syndrome (every check: inner and outer, mid-circuit and
     final) and its effect on the decoded program output (`syndromes`). Mechanisms with the
     same syndrome are pooled: the decoder's correction for a syndrome is the output flip with
     the most probability, and its confidence is that flip's share (`Table`).
  4. A shot is kept and corrected when its syndrome is empty (nothing to do) or is in the table
     with confidence >= `threshold`; otherwise (two or more faults, or an ambiguous single
     fault) it is discarded (`apply`).

Exact to first order in the physical error rate: with only the program's Clifford gates, a
Pauli frame is all there is to track, so a correction is an XOR on the decoded output (no
feedback is needed). Mechanisms that flip the output and fire no check are not correctable by
any decoder; they are the circuit-distance-1 floor that post-selection has too. Decisions use
the alpha = 1 probabilities; every channel scales (almost exactly) linearly with alpha, so the
same table serves every alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PAULI_1 = ("X", "Y", "Z")
_PAULI_2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]


@dataclass
class Mechanisms:
    clean: object                  # stim.Circuit without noise
    position: np.ndarray           # inject before clean[position]
    x_qubits: np.ndarray           # (mechanisms, 2) qubit or -1
    z_qubits: np.ndarray           # (mechanisms, 2) qubit or -1
    probability: np.ndarray
    num_qubits: int
    records_before: np.ndarray     # records produced by clean[:i]


def _xz(pauli: str) -> Tuple[bool, bool]:
    return pauli in "XY", pauli in "YZ"


def mechanisms(noisy) -> Mechanisms:
    """Every single-fault mechanism of `noisy` (a flattened stim.Circuit), and the clean circuit."""
    import stim

    clean = stim.Circuit()
    position: List[int] = []
    xs: List[Tuple[int, int]] = []
    zs: List[Tuple[int, int]] = []
    probability: List[float] = []
    records: List[int] = [0]

    def add(where: int, paulis: Sequence[Tuple[str, int]], p: float) -> None:
        if p <= 0:
            return
        x = [q for pauli, q in paulis if _xz(pauli)[0]] + [-1, -1]
        z = [q for pauli, q in paulis if _xz(pauli)[1]] + [-1, -1]
        position.append(where)
        xs.append((x[0], x[1]))
        zs.append((z[0], z[1]))
        probability.append(p)

    for inst in noisy.flattened():
        name = inst.name
        targets = [t.value for t in inst.targets_copy()]
        args = inst.gate_args_copy()
        where = len(clean)
        if name == "DEPOLARIZE1":
            for q in targets:
                for pauli in _PAULI_1:
                    add(where, [(pauli, q)], args[0] / 3)
        elif name == "DEPOLARIZE2":
            for a, b in zip(targets[::2], targets[1::2]):
                for pa, pb in _PAULI_2:
                    add(where, [(pa, a), (pb, b)], args[0] / 15)
        elif name == "PAULI_CHANNEL_1":
            for q in targets:
                for pauli, p in zip(_PAULI_1, args):
                    add(where, [(pauli, q)], p)
        elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            for q in targets:
                add(where, [(name[0], q)], args[0])
        elif name in ("PAULI_CHANNEL_2", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1",
                      "E", "ELSE_CORRELATED_ERROR"):
            raise NotImplementedError(f"noise channel {name} is not enumerated")
        elif name == "MPP":
            raise NotImplementedError("MPP records are not counted")
        else:
            # a TICK after every instruction stops Stim from fusing it with the next one, so
            # positions index instructions exactly (TICK does nothing to a frame)
            clean.append(inst)
            produced = len(targets) if name in ("M", "MR", "MX", "MY", "MZ") else 0
            records.append(records[-1] + produced)
            clean.append("TICK")
            records.append(records[-1])
    return Mechanisms(clean=clean, position=np.array(position, dtype=np.int64),
                      x_qubits=np.array(xs, dtype=np.int64).reshape(-1, 2),
                      z_qubits=np.array(zs, dtype=np.int64).reshape(-1, 2),
                      probability=np.array(probability), num_qubits=noisy.num_qubits,
                      records_before=np.array(records, dtype=np.int64))


def single_fault_flips(mech: Mechanisms, *, batch: int = 8192) -> np.ndarray:
    """(mechanisms x records) bool: the measurement records each mechanism flips on its own."""
    import stim

    total = mech.clean.num_measurements
    out = np.zeros((len(mech.probability), total), dtype=bool)
    order = np.argsort(mech.position, kind="stable")
    for b0 in range(0, len(order), batch):
        lanes = order[b0:b0 + batch]
        width = len(lanes)
        positions = mech.position[lanes]
        start = int(positions[0])
        sim = stim.FlipSimulator(batch_size=width, disable_stabilizer_randomization=True,
                                 num_qubits=mech.num_qubits)
        cursor = start
        boundaries = np.flatnonzero(np.diff(positions)) + 1
        for chunk in np.split(np.arange(width), boundaries):
            where = int(positions[chunk[0]])
            if where > cursor:
                sim.do(mech.clean[cursor:where])
                cursor = where
            for part, qubits in (("X", mech.x_qubits), ("Z", mech.z_qubits)):
                mask = np.zeros((mech.num_qubits, width), dtype=bool)
                for column in range(2):
                    q = qubits[lanes[chunk], column]
                    ok = q >= 0
                    mask[q[ok], chunk[ok]] = True
                if mask.any():
                    sim.broadcast_pauli_errors(pauli=part, mask=mask)
        if cursor < len(mech.clean):
            sim.do(mech.clean[cursor:])
        flips = sim.get_measurement_flips(bit_packed=False).T     # (width, records after start)
        before = int(mech.records_before[start])
        out[lanes, before:] = flips
    return out


def syndromes(bits: np.ndarray, maps, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    (checks, outputs) from record bits or flips: every check parity (shots x checks) and the
    decoded program output (shots x program qubits), both as `decode.py` computes them.
    """
    from experiments.concatenation.decode import _parity

    if kind == "single":
        checks = np.stack([_parity(bits, c) for c in maps.checks], axis=1)
        outputs = np.stack([bits[:, a] ^ bits[:, b] for a, b in maps.outputs], axis=1)
        return checks, outputs
    inner = [_parity(bits, cols) for _, cols in maps.inner_checks]
    values = np.stack([_parity(bits, cols) for cols in maps.flat_value], axis=1)
    outer = []
    for patch, data in maps.outer_data.items():
        parity = np.zeros(bits.shape[0], dtype=bool)
        for q in data:
            parity ^= values[:, q]
        outer.append(parity)
        outer.extend(values[:, q] for q in maps.outer_parity[patch])
    checks = np.stack(inner + outer, axis=1)
    outputs = np.stack([values[:, maps.outer_data[patch][a]] ^ values[:, maps.outer_data[patch][b]]
                        for patch, (a, b) in maps.outputs], axis=1)
    return checks, outputs


def _keys(rows: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(np.packbits(rows, axis=1))
    return packed.view(np.dtype((np.void, packed.shape[1]))).ravel()


@dataclass
class Table:
    """First-order decoding table: syndrome -> (correction, confidence)."""

    correction: Dict[bytes, np.ndarray] = field(default_factory=dict)
    confidence: Dict[bytes, float] = field(default_factory=dict)
    #: first-order probabilities at alpha = 1: detected, correctable at confidence 1, silent
    summary: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, mech: Mechanisms, flips: np.ndarray, maps, kind: str) -> "Table":
        checks, outputs = syndromes(flips, maps, kind)
        detected = checks.any(axis=1)
        p = mech.probability
        table = cls()
        silent = (~detected) & outputs.any(axis=1)
        keys = _keys(checks[detected])
        out_keys = _keys(outputs[detected])
        pd = p[detected]
        outs = outputs[detected]
        pooled: Dict[bytes, Dict[bytes, float]] = {}
        example: Dict[bytes, np.ndarray] = {}
        for key, okey, prob, row in zip(keys, out_keys, pd, outs):
            k, o = bytes(key), bytes(okey)
            bucket = pooled.setdefault(k, {})
            bucket[o] = bucket.get(o, 0.0) + prob
            example.setdefault(o, row)
        unanimous = 0.0
        for k, bucket in pooled.items():
            best = max(bucket, key=bucket.get)
            total = sum(bucket.values())
            table.correction[k] = example[best]
            table.confidence[k] = bucket[best] / total
            if len(bucket) == 1:
                unanimous += total
        table.summary = {"mechanisms": float(len(p)), "detected": float(pd.sum()),
                         "unanimous": float(unanimous), "silent": float(p[silent].sum()),
                         "syndromes": float(len(pooled))}
        return table

    def apply(self, bits: np.ndarray, maps, kind: str, threshold: float
              ) -> Tuple[np.ndarray, np.ndarray]:
        """(keep, corrected outputs) for record bits or flips."""
        checks, outputs = syndromes(bits, maps, kind)
        fired = checks.any(axis=1)
        keep = ~fired
        corrected = outputs.copy()
        rows = np.flatnonzero(fired)
        if len(rows):
            keys = _keys(checks[rows])
            unique, inverse = np.unique(keys, return_inverse=True)
            groups = np.split(rows[np.argsort(inverse, kind="stable")],
                              np.cumsum(np.bincount(inverse))[:-1])
            for key, hit in zip(unique, groups):
                k = bytes(key)
                if k in self.confidence and self.confidence[k] >= threshold:
                    keep[hit] = True
                    corrected[hit] ^= self.correction[k]
        return keep, corrected
