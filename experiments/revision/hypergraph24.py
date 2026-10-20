"""
hypergraph24.py -- stage 1 of [24] as a packing baseline: Pauli-based computation, then a
connectivity-minimising partition of the rotation hypergraph.

[24] is Sethi, Khan, Poster, Anand, Baker, "Optimizing Logical Mappings for Quantum
Low-Density Parity Check Codes", arXiv 2603.17167. Its stage 1, which this file reproduces:

1. Write the program as Clifford gates plus non-Clifford Pauli rotations and convert it to
   Pauli-based computation (Litinski): commute every Clifford to the end. Rotation j,
   exp(-i theta_j P_j / 2), then acts about P_j' = C_j^dag P_j C_j, where C_j is the product
   of every Clifford applied before it, and the program is
   U = C_total . R_{P_m'}(theta_m) ... R_{P_1'}(theta_1).
2. Vertices are logical qubits; each Pauli-product rotation is one hyperedge on supp(P_j').
   Identical hyperedges are merged, weight = multiplicity.
3. One static mapping for the whole circuit: KaHyPar splits the hypergraph into
   M = ceil(n / cap) blocks (cap = logical qubits per module), imbalance eps = 0.06,
   minimising connectivity km1 = sum_e w_e (lambda_e - 1), where lambda_e is the number of
   blocks hyperedge e touches. KaHyPar's balance bound is |block| <= (1 + eps) ceil(n / M).

Stage 2 places the modules relative to a magic-state factory. Iceberg patches have no
factory, so it has no analogue here and is skipped.

Faithful: the PBC conversion (a `stim.Tableau` is the Clifford frame; rz/rx/ry/p by a
multiple of pi/2 within `CLIFFORD_TOL` quarter turns -- the convention of
`common.non_clifford_rotations` -- are Cliffords folded into it; t, tdg and every other
angle are rotations), the hyperedges and their merging, M, eps, the balance bound and the
objective. The conjugation direction was checked by brute force (`pbc_unitary_error`): on
300 random 2-3 qubit circuits, C_total . prod_j R_{P_j'} matches the circuit's unitary to
2e-15 with P' = C^dag P C, while C P C^dag fails on 260 of them; and every signed P' of
six benchmark originals (3,636 rotations, up to 45 qubits) equals qiskit's independent
`Pauli.evolve(Clifford(prefix), frame="h")`.

What differs, and why:

* *Partitioner.* KaHyPar is not installed, so `partition_km1` is a flat (single-level)
  multi-start k-way Fiduccia-Mattheyses search on the same objective and bound, not
  KaHyPar's multilevel scheme. Seed 0 starts from program order, the rest from random
  balanced partitions; each is refined by FM passes -- single-vertex moves that respect the
  bound, and pairwise swaps, the only moves left once every block is full (cap = 2) --
  with exact incremental gains from a pin-count matrix and rollback to the best prefix of
  each pass, until a pass finds no improvement. Against exhaustive enumeration it found
  the optimum on all 1,600 random hypergraphs with n <= 8 at cap 2 and 3, and on all 150
  with n = 10 and 12 at cap 2-4 (program order alone, `seeds=1`: 98% and 79%). 150 qubits
  with 3,000-5,000 hyperedges take 1-2 s at cap 2 and 4; dense ones (mean |e| ~ 100, from a
  scrambled frame) 3-6 s; all 71 Fig 6 originals at both caps, 7 s together.
* *Capacity.* A module cannot hold more than `cap` qubits, so the bound is
  L = min(cap, floor((1 + eps) ceil(n / M))). The two agree for cap <= 16; above that
  KaHyPar's bound alone could put cap + 1 qubits in a module. Conversely, for small n the
  bound is tighter than cap, faithfully: qpe_n9 at cap 4 must split 3/3/3 (L = 3), and
  scores km1 15 against program order's 12, which fills 4/4/1.
* *Clifford-only circuits.* No non-Clifford rotation means no hyperedge and nothing for
  [24] to optimise (faithful); `hypergraph24_groups` returns program order -- qubits chunked
  consecutively into blocks of `cap`, the repo's naive packing -- and sets `fallback`.
* *Input.* Build the hypergraph from the original circuit (`BenchCircuit.original`). The
  Clifford proxy that Fig 6 simulates replaces every rz/rx with s, so it is Clifford-only
  and would always fall back. Measurements and barriers are ignored (the Fig 6 loader has
  already dropped measurements; any left in another input are treated as terminal).

    pbc_form(circuit)                  rotations (P', theta) in program order + the frame
    pbc_hyperedges(circuit)            {frozenset(qubits): multiplicity}, |e| >= 2 only
    partition_km1(n, edges, cap)       M blocks, each <= L, minimising km1
    km1(blocks, edges)                 the connectivity metric
    hypergraph24_groups(circuit, cap)  (groups, info): the packer
    pbc_unitary_error(circuit)         the brute-force conjugation check (<= ~10 qubits)
    exhaustive_km1(n, edges, cap)      the true optimum by enumeration (n <= ~10)

    PYTHONPATH=. venv/bin/python -m experiments.revision.hypergraph24 --caps 2 4
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import math
import time
from dataclasses import dataclass
from itertools import chain
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import stim

#: [24]'s KaHyPar imbalance.
EPSILON = 0.06
#: An angle is Clifford when it is within this many quarter turns of a multiple of pi/2.
CLIFFORD_TOL = 1e-9
#: Gains are sums of integer multiplicities; this only absorbs float noise in the sums.
GAIN_TOL = 1e-9

#: Instructions that act on no qubit's state, or that [24]'s stage 1 does not see.
IGNORED = frozenset({"barrier", "measure", "delay", "id", "i"})

#: Pauli rotations exp(-i theta P / 2), up to global phase: gate -> (Pauli letter for each
#: qarg, fixed angle or None to read `params[0]`). p(l) = u1(l) = e^{il/2} rz(l), t = rz(pi/4)
#: and tdg = rz(-pi/4) up to phase; rzx puts Z on its first qarg and X on its second.
ROTATIONS: Dict[str, Tuple[str, Optional[float]]] = {
    "rz": ("Z", None), "rx": ("X", None), "ry": ("Y", None),
    "p": ("Z", None), "u1": ("Z", None),
    "t": ("Z", math.pi / 4), "tdg": ("Z", -math.pi / 4),
    "rzz": ("ZZ", None), "rxx": ("XX", None), "ryy": ("YY", None), "rzx": ("ZX", None),
}

_PAULI_2X2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ----------------------------------------------------------------------------------------
# Pauli-based computation
# ----------------------------------------------------------------------------------------


@dataclass
class PBCProgram:
    """
    A circuit in Litinski's form: `rotations[j] = (P_j', theta_j)` in program order, then
    `frame` (C_total, every Clifford of the circuit). The circuit's unitary is, up to global
    phase, frame . R_{P_m'}(theta_m) ... R_{P_1'}(theta_1), R_P(theta) = exp(-i theta P / 2).
    """

    num_qubits: int
    rotations: List[Tuple[stim.PauliString, float]]
    frame: stim.Tableau


def _quarter_turns(theta: float) -> Optional[int]:
    """theta / (pi/2) mod 4 if theta is a multiple of pi/2 (a Clifford angle), else None."""
    quarters = theta / (math.pi / 2)
    nearest = round(quarters)
    return int(nearest) % 4 if abs(quarters - nearest) <= CLIFFORD_TOL else None


def _letters_matrix(letters: str) -> np.ndarray:
    """The Pauli `letters[i]` on qarg i, little endian (qarg 0 least significant)."""
    out = np.eye(1, dtype=complex)
    for letter in letters:
        out = np.kron(_PAULI_2X2[letter], out)
    return out


_TABLEAU_CACHE: Dict[Tuple, stim.Tableau] = {}


def _rotation_tableau(letters: str, quarters: int) -> stim.Tableau:
    """exp(-i quarters (pi/4) P) for quarters in {1, 2, 3}, built from its exact matrix."""
    key = ("rotation", letters, quarters)
    if key not in _TABLEAU_CACHE:
        root = 1 / math.sqrt(2)
        identity, pauli = {1: (root, -1j * root), 2: (0.0, -1j), 3: (root, 1j * root)}[quarters]
        pauli_matrix = _letters_matrix(letters)
        matrix = identity * np.eye(len(pauli_matrix)) + pauli * pauli_matrix
        _TABLEAU_CACHE[key] = stim.Tableau.from_unitary_matrix(matrix, endian="little")
    return _TABLEAU_CACHE[key]


def _gate_tableau(operation) -> stim.Tableau:
    """
    A Clifford gate's tableau from qiskit's own matrix (little endian, like qiskit), so no
    gate convention is restated here. Cached per (name, params) when the params are numbers.
    """
    name = operation.name.lower()
    try:
        key = ("gate", name, tuple(round(float(p), 12) for p in operation.params))
    except (TypeError, ValueError):  # matrix-valued or symbolic parameters
        key = None
    if key is not None and key in _TABLEAU_CACHE:
        return _TABLEAU_CACHE[key]
    try:
        tableau = stim.Tableau.from_unitary_matrix(operation.to_matrix(), endian="little")
    except Exception as error:  # noqa: BLE001 - no matrix, or not a Clifford
        raise ValueError(
            f"{name!r} is neither a Clifford gate nor a Pauli rotation "
            f"({', '.join(sorted(ROTATIONS))}); transpile to Clifford + rz/rx first"
        ) from error
    if key is not None:
        _TABLEAU_CACHE[key] = tableau
    return tableau


def _classify(operation) -> Tuple[str, object]:
    """('skip', None) | ('rotation', (letters, theta)) | ('clifford', stim.Tableau)."""
    name = operation.name.lower()
    if name in IGNORED:
        return "skip", None
    if name in ROTATIONS:
        letters, fixed = ROTATIONS[name]
        theta = float(operation.params[0]) if fixed is None else fixed
        quarters = _quarter_turns(theta)
        if quarters is None:
            return "rotation", (letters, theta)
        if quarters == 0:
            return "skip", None
        return "clifford", _rotation_tableau(letters, quarters)
    return "clifford", _gate_tableau(operation)


def _pull_back(frame: stim.Tableau, letters: str, qubits: Sequence[int]) -> stim.PauliString:
    """
    C^dag P C for P = letters on qubits, C = the frame (every Clifford so far). The single-
    qubit factors commute, so their images multiply to the image of P, sign included.
    """
    image = None
    for letter, qubit in zip(letters, qubits):
        if letter == "X":
            term = frame.inverse_x_output(qubit)
        elif letter == "Y":
            term = frame.inverse_y_output(qubit)
        else:
            term = frame.inverse_z_output(qubit)
        image = term if image is None else image * term
    return image


def pbc_form(circuit) -> PBCProgram:
    """
    Litinski's transformation. `frame` holds C, the product of every Clifford so far:
    `append` composes the next gate *after* it (C <- G C), and a rotation about P met with
    frame C is moved past C as a rotation about C^dag P C (`inverse_*_output`), because
    R_P(theta) C = C R_{C^dag P C}(theta).
    """
    n = circuit.num_qubits
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    frame = stim.Tableau(n)
    rotations: List[Tuple[stim.PauliString, float]] = []
    for instruction in circuit.data:
        kind, payload = _classify(instruction.operation)
        if kind == "skip":
            continue
        qubits = [index[q] for q in instruction.qubits]
        if kind == "rotation":
            letters, theta = payload
            rotations.append((_pull_back(frame, letters, qubits), theta))
        else:
            frame.append(payload, qubits)
    return PBCProgram(num_qubits=n, rotations=rotations, frame=frame)


def _merge(rotations) -> Tuple[Dict[FrozenSet[int], int], int]:
    """Supports of size >= 2, merged by multiplicity; plus the count of size-1 supports."""
    edges: Dict[FrozenSet[int], int] = {}
    singletons = 0
    for pauli, _ in rotations:
        support = frozenset(pauli.pauli_indices())
        if len(support) >= 2:
            edges[support] = edges.get(support, 0) + 1
        else:
            singletons += 1
    return edges, singletons


def pbc_hyperedges(circuit) -> dict[frozenset[int], int]:
    """
    [24]'s hypergraph: one hyperedge per non-Clifford rotation, on the support of its Pauli
    after every earlier Clifford is commuted past it; identical hyperedges merged with
    weight = multiplicity. Size-1 hyperedges are dropped -- no partition can cut them.
    """
    return _merge(pbc_form(circuit).rotations)[0]


# ----------------------------------------------------------------------------------------
# the conjugation check
# ----------------------------------------------------------------------------------------


def _pauli_string_matrix(pauli: stim.PauliString) -> np.ndarray:
    """Dense matrix of a stim Pauli string, sign included, little endian like qiskit."""
    letters = "".join("IXYZ"[pauli[q]] for q in range(len(pauli)))
    return complex(pauli.sign) * _letters_matrix(letters)


def _filtered(circuit, keep_rotations: bool):
    """The circuit without measurements and barriers (and without rotations if asked)."""
    from qiskit import QuantumCircuit

    out = QuantumCircuit(circuit.num_qubits)
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        kind, _ = _classify(instruction.operation)
        if kind == "skip" or (kind == "rotation" and not keep_rotations):
            continue
        out.append(instruction.operation, [out.qubits[index[q]] for q in instruction.qubits])
    return out


def pbc_unitary_error(circuit, rotations: Optional[List[Tuple[stim.PauliString, float]]] = None
                      ) -> float:
    """
    1 - |Tr(U^dag V)| / 2^n for U the circuit's unitary and V = C_total . prod_j R_{P_j'}.
    Zero exactly when the PBC form is right up to global phase. Every matrix here is built
    independently of stim's: U and C_total by qiskit (C_total = the circuit with its
    non-Clifford rotations deleted), the rotations from the Pauli strings by hand. Pass
    `rotations` to score some other list against the same circuit. Dense: small n only.
    """
    from qiskit.quantum_info import Operator

    n = circuit.num_qubits
    if rotations is None:
        rotations = pbc_form(circuit).rotations
    target = Operator(_filtered(circuit, keep_rotations=True)).data
    product = np.eye(2 ** n, dtype=complex)
    for pauli, theta in rotations:
        matrix = _pauli_string_matrix(pauli)
        product = (math.cos(theta / 2) * np.eye(2 ** n) - 1j * math.sin(theta / 2) * matrix
                   ) @ product
    product = Operator(_filtered(circuit, keep_rotations=False)).data @ product
    return float(1.0 - abs(np.trace(target.conj().T @ product)) / 2 ** n)


# ----------------------------------------------------------------------------------------
# the hypergraph
# ----------------------------------------------------------------------------------------


def block_count(num_qubits: int, cap: int) -> int:
    """M = ceil(n / cap): the fewest modules that hold the program."""
    return max(1, -(-int(num_qubits) // int(cap)))


def size_limit(num_qubits: int, cap: int, eps: float = EPSILON) -> int:
    """L = min(cap, floor((1 + eps) ceil(n / M))): KaHyPar's bound, capped by the module."""
    average = -(-int(num_qubits) // block_count(num_qubits, cap))
    return min(int(cap), math.floor((1.0 + eps) * average + 1e-9))


def program_order(num_qubits: int, cap: int) -> List[List[int]]:
    """Qubits 0..n-1 chunked consecutively into blocks of `cap`: the naive packing."""
    return [list(range(start, min(start + cap, num_qubits)))
            for start in range(0, num_qubits, cap)]


def km1(blocks, hyperedges) -> float:
    """Connectivity: sum_e w_e (lambda_e - 1). Every pin must be in exactly one block."""
    where: Dict[int, int] = {}
    for label, block in enumerate(blocks):
        for qubit in block:
            if int(qubit) in where:
                raise ValueError(f"qubit {qubit} is in two blocks")
            where[int(qubit)] = label
    total = 0.0
    for edge, weight in hyperedges.items():
        if len(edge) < 2:
            continue
        try:
            touched = {where[int(q)] for q in edge}
        except KeyError as error:
            raise ValueError(f"qubit {error.args[0]} of a hyperedge is in no block") from None
        total += float(weight) * (len(touched) - 1)
    return total


@dataclass(frozen=True)
class _Hypergraph:
    """
    Hyperedges as CSR both ways (pins of edge e: `pins[edge_ptr[e]:edge_ptr[e + 1]]`) and
    as a dense 0/1 incidence matrix, 8 E n bytes: 6 MB at 5,000 hyperedges on 150 qubits.
    """

    n: int
    weights: np.ndarray       # (E,)
    edge_ptr: np.ndarray      # (E + 1,)
    pins: np.ndarray          # (P,), sorted within each edge
    pin_edge: np.ndarray      # (P,) the edge each pin belongs to
    vertex_ptr: np.ndarray    # (n + 1,)
    vertex_edges: np.ndarray  # (P,) edges of v: vertex_edges[vertex_ptr[v]:vertex_ptr[v + 1]]
    incidence: sp.csr_matrix  # (E, n)
    dense: np.ndarray         # (E, n), the same matrix
    degree: np.ndarray        # (n,) total weight of the edges at each vertex

    @property
    def num_edges(self) -> int:
        return len(self.weights)


def _hypergraph(num_qubits: int, hyperedges) -> _Hypergraph:
    """Validate, drop what no partition can cut (|e| < 2, w = 0), merge, sort canonically."""
    n = int(num_qubits)
    merged: Dict[Tuple[int, ...], float] = {}
    for edge, weight in hyperedges.items():
        pins = tuple(sorted({int(v) for v in edge}))
        if pins and (pins[0] < 0 or pins[-1] >= n):
            raise ValueError(f"hyperedge {sorted(pins)} has a pin outside 0..{n - 1}")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"hyperedge {sorted(pins)} has weight {weight}")
        if len(pins) >= 2 and weight > 0:
            merged[pins] = merged.get(pins, 0.0) + weight
    edges = sorted(merged)
    lengths = np.array([len(e) for e in edges], dtype=np.int64)
    edge_ptr = np.zeros(len(edges) + 1, dtype=np.int64)
    np.cumsum(lengths, out=edge_ptr[1:])
    pins = np.fromiter(chain.from_iterable(edges), dtype=np.int64, count=int(edge_ptr[-1]))
    pin_edge = np.repeat(np.arange(len(edges), dtype=np.int64), lengths)
    weights = np.array([merged[e] for e in edges], dtype=float)
    vertex_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pins, minlength=n), out=vertex_ptr[1:])
    incidence = sp.csr_matrix((np.ones(len(pins)), pins, edge_ptr), shape=(len(edges), n))
    return _Hypergraph(
        n=n, weights=weights, edge_ptr=edge_ptr, pins=pins, pin_edge=pin_edge,
        vertex_ptr=vertex_ptr, vertex_edges=pin_edge[np.argsort(pins, kind="stable")],
        incidence=incidence, dense=incidence.toarray(),
        degree=np.bincount(pins, weights=weights[pin_edge], minlength=n),
    )


# ----------------------------------------------------------------------------------------
# k-way FM
# ----------------------------------------------------------------------------------------


class _KWayFM:
    """
    One partition and everything its move gains need, updated move by move.

    phi[e, b]   pins of hyperedge e in block b (the pin-count matrix); lambda_e = #{b: phi > 0}
    benefit[u]  sum of w_e over e at u with phi[e, part[u]] == 1 ("u is alone in e's block")
    conn[u, b]  sum of w_e over e at u with phi[e, b] > 0
    corr[u, x]  sum over e holding both u and x of w_e (alone(e, u) + alone(e, x))

    Moving u to block t gains benefit[u] - degree[u] + conn[u, t]: edges u was alone in lose
    a block, edges with no pin in t gain one. Swapping u and x (different blocks) gains the
    two move gains minus corr[u, x]: an edge holding both keeps its pin counts, but each
    move alone would have credited it for every endpoint that was alone in its block.
    """

    def __init__(self, graph: _Hypergraph, part: np.ndarray, blocks: int, limit: int):
        self.graph = graph
        self.blocks = int(blocks)
        self.limit = int(limit)
        self.part = np.asarray(part, dtype=np.int64).copy()
        self.size = np.bincount(self.part, minlength=self.blocks)
        self.phi, self.benefit, self.conn, self.corr = self._from_scratch()

    def _from_scratch(self):
        g, n = self.graph, self.graph.n
        block = self.part[g.pins]
        phi = np.zeros((g.num_edges, self.blocks), dtype=np.int64)
        np.add.at(phi, (g.pin_edge, block), 1)
        alone_weight = g.weights[g.pin_edge] * (phi[g.pin_edge, block] == 1)
        benefit = np.bincount(g.pins, weights=alone_weight, minlength=n)
        conn = np.asarray(g.incidence.T @ ((phi > 0) * g.weights[:, None]), dtype=float)
        alone = sp.csr_matrix((alone_weight, g.pins, g.edge_ptr), shape=(g.num_edges, n))
        half = (alone.T @ g.incidence).toarray()
        corr = half + half.T
        np.fill_diagonal(corr, 0.0)
        return phi, benefit, conn, corr

    def km1(self) -> float:
        return float(((self.phi > 0).sum(axis=1) - 1) @ self.graph.weights)

    def consistent(self) -> bool:
        """The incremental state equals a rebuild from `part` (a test hook)."""
        fresh = self._from_scratch()
        return (np.array_equal(self.size, np.bincount(self.part, minlength=self.blocks))
                and all(np.allclose(a, b) for a, b in zip(
                    (self.phi, self.benefit, self.conn, self.corr), fresh)))

    def move(self, v: int, target: int) -> None:
        """
        Move v to `target` and update phi, benefit, conn and corr. Only the edges at v change
        pin counts, and only v, the source's other members and the target's members can
        change "alone" status, so every update is a product of v's incidence rows
        (deg(v) x n) with at most 2L columns -- no rebuild.
        """
        g = self.graph
        source = int(self.part[v])
        if source == target:
            return
        edges = g.vertex_edges[g.vertex_ptr[v]:g.vertex_ptr[v + 1]]
        if len(edges):
            weight = g.weights[edges]
            old_s = self.phi[edges, source]
            old_t = self.phi[edges, target]
            new_s, new_t = old_s - 1, old_t + 1
            self.phi[edges, source] = new_s
            self.phi[edges, target] = new_t
            rows = g.dense[edges]

            left = new_s == 0  # the source no longer touches these edges
            if left.any():
                self.conn[:, source] -= weight[left] @ rows[left]
            joined = old_t == 0  # the target touches these edges for the first time
            if joined.any():
                self.conn[:, target] += weight[joined] @ rows[joined]

            # Change of alone(e, u), per edge at v: v itself; the one pin left in the source
            # (alone now); the one pin already in the target (alone no longer).
            stay = np.flatnonzero(self.part == source)
            stay = stay[stay != v]
            arrive = np.flatnonzero(self.part == target)
            whom = np.concatenate(([v], stay, arrive))
            change = np.empty((len(edges), len(whom)))
            change[:, 0] = (new_t == 1).astype(float) - (old_s == 1)
            change[:, 1:1 + len(stay)] = rows[:, stay] * (new_s == 1)[:, None]
            change[:, 1 + len(stay):] = -rows[:, arrive] * (old_t == 1)[:, None]
            change *= weight[:, None]
            self.benefit[whom] += change.sum(axis=0)
            delta = change.T @ rows  # delta[i, x]: change of corr[whom[i], x] via whom[i]
            self.corr[whom] += delta
            self.corr[:, whom] += delta.T
            self.corr[whom, whom] = 0.0
        self.part[v] = target
        self.size[source] -= 1
        self.size[target] += 1

    def best_step(self, free: np.ndarray) -> Optional[Tuple[float, List[Tuple[int, int]]]]:
        """
        The best feasible step among unlocked vertices, negative gains allowed: a single
        move into a block below the limit, or a swap across two blocks. Ties go to the
        single move, then to the lowest (vertex, block) / (vertex, vertex) in index order.
        """
        index = np.flatnonzero(free)
        if not len(index):
            return None
        part = self.part[index]
        gains = (self.benefit[index] - self.graph.degree[index])[:, None] + self.conn[index]
        best: Optional[Tuple[float, List[Tuple[int, int]]]] = None

        room = self.size < self.limit
        if room.any():
            single = np.where(room[None, :], gains, -np.inf)
            single[np.arange(len(index)), part] = -np.inf
            flat = int(np.argmax(single))
            value = float(single.flat[flat])
            if value > -np.inf:
                i, block = divmod(flat, self.blocks)
                best = (value, [(int(index[i]), int(block))])

        if len(index) >= 2:
            cross = gains[:, part]  # cross[i, j]: gain of moving index[i] into j's block
            swap = cross + cross.T - self.corr[np.ix_(index, index)]
            swap[part[:, None] == part[None, :]] = -np.inf
            flat = int(np.argmax(swap))  # symmetric, so the first maximum has i < j
            value = float(swap.flat[flat])
            if value > -np.inf and (best is None or value > best[0] + GAIN_TOL):
                i, j = divmod(flat, len(index))
                best = (value, [(int(index[i]), int(part[j])), (int(index[j]), int(part[i]))])
        return best

    def fm_pass(self) -> float:
        """
        One FM pass: take the best step, lock what moved, repeat until nothing is movable;
        then roll back to the best prefix. Returns that prefix's gain (0 if none was better).
        """
        free = np.ones(self.graph.n, dtype=bool)
        undo: List[Tuple[int, int]] = []
        total = best = 0.0
        keep = 0
        while True:
            step = self.best_step(free)
            if step is None:
                break
            gain, moves = step
            for vertex, target in moves:
                undo.append((vertex, int(self.part[vertex])))
                self.move(vertex, target)
                free[vertex] = False
            total += gain
            if total > best + GAIN_TOL:
                best, keep = total, len(undo)
        for vertex, source in reversed(undo[keep:]):
            self.move(vertex, source)
        return best

    def refine(self) -> int:
        """FM passes until one finds no improvement. Returns the number of passes."""
        passes = 1
        while self.fm_pass() > GAIN_TOL:
            passes += 1
        return passes


def _initial_part(n: int, blocks: int, limit: int, seed_index: int, rng_seed: int) -> np.ndarray:
    """
    Seed 0: program order in chunks of L (the naive packing whenever L == cap). Others: a
    uniformly random balanced partition (sizes differ by at most one) from
    `default_rng(rng_seed + seed_index)`. Both always fill exactly M blocks.
    """
    if seed_index == 0:
        return np.arange(n, dtype=np.int64) // limit
    order = np.random.default_rng(rng_seed + seed_index).permutation(n)
    part = np.empty(n, dtype=np.int64)
    part[order] = np.arange(n, dtype=np.int64) % blocks
    return part


def _blocks(part: np.ndarray, blocks: int) -> List[List[int]]:
    """Non-empty blocks, each sorted, ordered by their smallest qubit."""
    out = [np.flatnonzero(part == b).tolist() for b in range(blocks)]
    return sorted((b for b in out if b), key=lambda b: b[0])


def partition_km1(num_qubits: int, hyperedges: dict, cap: int, *, eps: float = 0.06,
                  seeds: int = 10, rng_seed: int = 0) -> list[list[int]]:
    """
    [24]'s partition: M = ceil(n / cap) blocks of at most L qubits (see `size_limit`),
    minimising km1. Multi-start FM: `seeds` starts (0: program order; s >= 1: random
    balanced from `default_rng(rng_seed + s)`), each refined to convergence; the lowest km1
    wins, ties to the lowest seed. Deterministic for fixed inputs (edges are sorted
    canonically first, so dict order does not matter either).
    """
    n = int(num_qubits)
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    if n <= 0:
        return []
    blocks = block_count(n, cap)
    limit = size_limit(n, cap, eps)
    if blocks == 1:
        return [list(range(n))]
    graph = _hypergraph(n, hyperedges)
    if graph.num_edges == 0:
        return _blocks(_initial_part(n, blocks, limit, 0, rng_seed), blocks)
    best_value, best_part = math.inf, None
    for seed_index in range(max(1, int(seeds))):
        fm = _KWayFM(graph, _initial_part(n, blocks, limit, seed_index, rng_seed),
                     blocks, limit)
        fm.refine()
        value = fm.km1()
        if value < best_value - GAIN_TOL:
            best_value, best_part = value, fm.part.copy()
    return _blocks(best_part, blocks)


def exhaustive_km1(num_qubits: int, hyperedges: dict, cap: int, *, eps: float = EPSILON
                   ) -> Tuple[float, List[List[int]]]:
    """
    The true optimum of `partition_km1`'s problem, by enumerating every partition into at
    most M blocks of at most L (restricted growth strings). Bell(n) leaves: n <= ~10.
    """
    n = int(num_qubits)
    blocks, limit = block_count(n, cap), size_limit(n, cap, eps)
    edges = [(tuple(int(q) for q in e), float(w)) for e, w in hyperedges.items()
             if len(e) >= 2]
    labels = [0] * n
    sizes: List[int] = []
    best: List = [math.inf, None]

    def visit(i: int) -> None:
        if n - i > sum(limit - s for s in sizes) + (blocks - len(sizes)) * limit:
            return
        if i == n:
            value = sum(w * (len({labels[q] for q in e}) - 1) for e, w in edges)
            if value < best[0]:
                best[0], best[1] = value, list(labels)
            return
        for b in range(len(sizes)):
            if sizes[b] < limit:
                labels[i] = b
                sizes[b] += 1
                visit(i + 1)
                sizes[b] -= 1
        if len(sizes) < blocks:
            labels[i] = len(sizes)
            sizes.append(1)
            visit(i + 1)
            sizes.pop()

    visit(0)
    return best[0], _blocks(np.array(best[1], dtype=np.int64), blocks)


# ----------------------------------------------------------------------------------------
# the packer
# ----------------------------------------------------------------------------------------


def hypergraph24_groups(circuit, cap: int, *, eps: float = 0.06, seeds: int = 10
                        ) -> tuple[list[list[int]], dict]:
    """
    [24] stage 1 on one circuit: PBC hyperedges, then `partition_km1` into modules of `cap`.
    Pass the original (non-Clifford) circuit. A circuit with no hyperedge gets program
    order and `fallback=True`.

    info: num_hyperedges, total_weight, max/mean_hyperedge_size (over distinct hyperedges;
    0 when there are none), km1_value, km1_program_order (the naive packing, for reference),
    fallback, seconds -- plus the rotation counts, M, L and the time split.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    started = time.perf_counter()
    n = circuit.num_qubits
    program = pbc_form(circuit)
    edges, singletons = _merge(program.rotations)
    converted = time.perf_counter()
    naive = program_order(n, cap)
    groups = partition_km1(n, edges, cap, eps=eps, seeds=seeds) if edges else naive
    finished = time.perf_counter()
    sizes = [len(e) for e in edges]
    info = {
        "num_qubits": n,
        "cap": int(cap),
        "blocks": block_count(n, cap) if n else 0,
        "size_limit": size_limit(n, cap, eps) if n else 0,
        "eps": float(eps),
        "seeds": int(seeds),
        "num_rotations": len(program.rotations),
        "singleton_rotations": singletons,
        "num_hyperedges": len(edges),
        "total_weight": int(sum(edges.values())),
        "max_hyperedge_size": max(sizes, default=0),
        "mean_hyperedge_size": float(np.mean(sizes)) if sizes else 0.0,
        "km1_value": km1(groups, edges),
        "km1_program_order": km1(naive, edges),
        "fallback": not edges,
        "pbc_seconds": converted - started,
        "partition_seconds": finished - converted,
        "seconds": finished - started,
    }
    return groups, info


# ----------------------------------------------------------------------------------------
# CLI: the per-circuit table
# ----------------------------------------------------------------------------------------

DEFAULT_NAMES: Tuple[str, ...] = (
    "adder_n10", "qft_n18", "ising_n26", "knn_n25", "bv_n140", "qugan_n39", "multiplier_n45",
)


def load_originals(names: Sequence[str]) -> Dict[str, object]:
    """
    Originals by name: Fig 6 circuits through `load_fig6_circuits`, which silently drops
    anything else; other QASMBench names (multiplier_n45 is not a Fig 6 circuit) through
    the same level-1 `build_circuit`.
    """
    from experiments.revision.common import (
        QASM_ROOT,
        _qasm_by_name,
        build_circuit,
        fig6_reference,
        load_fig6_circuits,
    )

    known = set(fig6_reference()["name"])
    fig6 = [name for name in names if name in known]
    out = {c.name: c.original for c in load_fig6_circuits(names=fig6)} if fig6 else {}
    for name in names:
        if name in out:
            continue
        for group in ("small", "medium", "large"):
            files = _qasm_by_name(QASM_ROOT / group)
            if name in files:
                out[name] = build_circuit(files[name], proxy=False)
                break
        if out.get(name) is None:
            raise SystemExit(f"no QASMBench circuit named {name}")
    return {name: out[name] for name in names}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--names", nargs="+", default=list(DEFAULT_NAMES))
    parser.add_argument("--caps", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eps", type=float, default=EPSILON)
    args = parser.parse_args(argv)
    print(f"{'circuit':16s} {'n':>4s} {'cap':>3s} {'edges':>6s} {'weight':>6s} "
          f"{'mean|e|':>7s} {'max|e|':>6s} {'km1':>8s} {'km1 naive':>9s} {'s':>7s}")
    for name, circuit in load_originals(args.names).items():
        for cap in args.caps:
            _, info = hypergraph24_groups(circuit, cap, eps=args.eps, seeds=args.seeds)
            print(f"{name:16s} {info['num_qubits']:4d} {cap:3d} {info['num_hyperedges']:6d} "
                  f"{info['total_weight']:6d} {info['mean_hyperedge_size']:7.2f} "
                  f"{info['max_hyperedge_size']:6d} {info['km1_value']:8.0f} "
                  f"{info['km1_program_order']:9.0f} {info['seconds']:7.2f}"
                  + ("  (fallback: no hyperedges)" if info["fallback"] else ""), flush=True)


if __name__ == "__main__":
    main()
