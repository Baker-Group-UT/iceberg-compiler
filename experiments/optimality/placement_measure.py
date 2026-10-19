"""
placement_measure.py -- one placement of a program, measured the way the optimality figures
measure it, at any width, plus the check frames cannot do.

    support = logical_support(proxy)              # ideal outcomes: a + span(basis), any width
    row = measure(original, proxy, placement, commute, support, alphas={"a1": 1.0, ...})

Per placement:
    static     1q / 2q gates and depth of the original program's translation (what Fig 6's
               static panels count) under the standard translator
    cost       AQORN's own objective (`exhaustive_front.StandardScorer` on the original)
    lsr, ler, tvd and *_sel per alpha
               Pauli frames on the Clifford proxy's noisy circuit (Fig 6's machine), occupied-lane
               logical error per kept shot, exact TVD = P(flip outside span(basis) | kept); the
               `_sel` values are independent shots, for choosing a "best" without the winner's
               curse. Alphas other than 1 rescale the one alpha = 1 conversion exactly
               (`concatenation.noise.rescale_text`)
    correct    noiseless check, frames cannot see a miscompile: the noise-free encoded proxy is
               sampled (with randomisation) and every shot must pass post-selection and decode to
               an ideal outcome (in a + span(basis)), and the decoded outcomes must span V
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

_DATA_REGISTER = re.compile(r"data_(\d+)$")
_STIM_GATES = {"h": "H", "s": "S", "sx": "SQRT_X", "x": "X", "z": "Z", "cx": "CX", "cz": "CZ",
               "sdg": "S_DAG", "sxdg": "SQRT_X_DAG", "y": "Y", "swap": "SWAP"}


def logical_support(proxy, shots: int = 2048) -> Tuple[int, List[int]]:
    """(a, basis): the ideal outcomes of a Clifford program are uniform on a + span(basis)."""
    import stim

    from experiments.optimality.exhaustive_front import support_basis as _reduce  # noqa: F401

    circuit = stim.Circuit()
    n = proxy.num_qubits
    for inst in proxy.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        targets = [proxy.find_bit(q).index for q in inst.qubits]
        circuit.append(_STIM_GATES[name], targets)
    circuit.append("M", list(range(n)))
    samples = circuit.compile_sampler(seed=7).sample(shots)
    values = [int(sum(int(b) << q for q, b in enumerate(row))) for row in samples]
    a = values[0]
    basis: Dict[int, int] = {}
    for x in values[1:]:
        v = x ^ a
        for pivot in sorted(basis, reverse=True):
            if (v >> pivot) & 1:
                v ^= basis[pivot]
        if v:
            basis[v.bit_length() - 1] = v
    return a, [basis[p] for p in sorted(basis, reverse=True)]


def bits(value: int, n: int) -> np.ndarray:
    """A Python int (any width) as a length-n bool vector, bit q = program qubit q."""
    return np.array([(value >> q) & 1 for q in range(n)], dtype=bool)


def outside_rows(rows: np.ndarray, basis: Sequence[int]) -> np.ndarray:
    """
    Which rows of a (shots, n) bool matrix lie outside span(basis), at any width. Bit vectors
    are never packed into machine integers: a 64-bit pack silently wraps from qubit 64 on (the
    overflow that corrupted TVD and the noiseless check on the >= 64-qubit circuits).
    """
    n = rows.shape[1]
    out = np.zeros(len(rows), dtype=bool)
    if len(basis) >= n:
        return out                                  # uniform output: every vector is inside
    live = np.flatnonzero(rows.any(axis=1))
    rest = rows[live].copy()
    for vector in basis:                            # descending pivots (support_basis order)
        pivot = int(vector).bit_length() - 1
        hit = rest[:, pivot]
        rest[hit] ^= bits(vector, n)
    out[live] = rest.any(axis=1)
    return out


def gf2_rank(rows: np.ndarray) -> int:
    matrix = rows.copy()
    rank = 0
    for column in range(matrix.shape[1]):
        pivots = np.flatnonzero(matrix[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + pivots[0]
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        others = np.flatnonzero(matrix[:, column])
        others = others[others != rank]
        matrix[others] ^= matrix[rank]
        rank += 1
        if rank == matrix.shape[0]:
            break
    return rank


def noiseless_ok(encoded, noisy, positions, support, shots: int = 256) -> bool:
    """Every noise-free shot passes post-selection and decodes into the ideal outcome set."""
    from experiments.concatenation.noise import rescale_text
    from experiments.revision.exp8_undetected.frames import _records
    from iceberg.layout import CodeLayout

    a, basis = support
    clean = rescale_text(str(noisy), 0.0)
    samples = clean.compile_sampler(seed=11).sample(shots)
    rows: Dict[str, Dict[int, int]] = {}
    for row, (name, index) in enumerate(_records(encoded)):
        rows.setdefault(name, {})[index] = row
    flags = np.zeros(shots, dtype=bool)
    slot: Dict[Tuple[int, int], np.ndarray] = {}
    reference = None
    for name, by_index in rows.items():
        block = samples[:, [by_index[i] for i in sorted(by_index)]].T
        if "parity" in name or "flag" in name:
            reference = block[:, :1] if reference is None else reference
            flags |= (block != block[:, :1]).any(axis=0)   # deterministic across shots
        elif "data" in name:
            parity = block.sum(axis=0) % 2
            flags |= parity != parity[0]
            patch = int(_DATA_REGISTER.search(name).group(1))
            code = CodeLayout(block.shape[0])
            for lane in range(code.k):
                anchor, qubit = code.z_support(lane)
                slot[(patch, lane)] = block[anchor] ^ block[qubit]
    if flags.any():
        return False
    outcomes = np.stack([slot[tuple(position)] for position in positions], axis=1)
    distinct = np.unique(outcomes, axis=0)
    if outside_rows(distinct ^ bits(a, len(positions)), basis).any():
        return False
    return gf2_rank(distinct ^ distinct[0]) == len(basis)


_SCORERS: Dict[str, object] = {}


def _scorer(original):
    """`StandardScorer` built once per program and process (it builds the cost models)."""
    from experiments.optimality.exhaustive_front import StandardScorer
    from experiments.shared.simulation import ensure_measurements
    from translator.logical_optimization import _body_and_tail, fingerprint

    program = ensure_measurements(original)
    key = fingerprint(_body_and_tail(program)[0])
    if key not in _SCORERS:
        _SCORERS.clear()
        _SCORERS[key] = StandardScorer(program)
    return _SCORERS[key]


def measure(original, proxy, placement, commute: bool, support, *, alphas: Dict[str, float],
            shots: int = 200_000, seed_key: str = "") -> Dict[str, object]:
    """Static counts, AQORN's objective, frame metrics per alpha, and the noiseless check."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from experiments.concatenation.noise import rescale_text
    from experiments.optimality.exhaustive_front import frame_metrics
    from experiments.shared.simulation import ensure_measurements
    from experiments.tvd_explore.fastframes import at_alpha_one
    from packing.standard import standard_translator_config
    from translator.core import Translator

    placement = [tuple(int(v) for v in p) for p in placement]
    translator = Translator(config=standard_translator_config(4, commute=commute))
    row: Dict[str, object] = {}
    static = translator.translate(ensure_measurements(original), placement).circuit
    one, two = one_two_qubit_gate_totals(static)
    row.update({"one_qubit_gates": int(one), "two_qubit_gates": int(two),
                "depth": int(static.depth())})
    row.update(_scorer(original)(placement, commute))

    result = translator.translate(ensure_measurements(proxy), placement)
    positions = [tuple(p) for p in result.context.state_positions]
    noisy = at_alpha_one(result.circuit)
    text = str(noisy)
    seed = zlib.crc32(f"{seed_key}:{placement}:{commute}".encode()) % 1_000_000_007
    a, basis = support
    for label, alpha in alphas.items():
        scaled = noisy if alpha == 1.0 else rescale_text(text, alpha)
        report = frame_metrics(result.circuit, scaled, positions, basis, shots=shots, seed=seed)
        select = frame_metrics(result.circuit, scaled, positions, basis, shots=shots,
                               seed=seed + 1)
        for metric in ("lsr", "ler", "tvd"):
            row[f"{metric}_{label}"] = report[metric]
            row[f"{metric}_sel_{label}"] = select[metric]
    row["correct"] = bool(noiseless_ok(result.circuit, noisy, positions, support))
    return row
