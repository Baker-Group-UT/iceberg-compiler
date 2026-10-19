"""
construct.py -- two-level [[4,2,2]] x [[4,2,2]] compilation by running the translator twice.

    program --(outer translation)--> C1 --(flatten)--> inner program --(inner translation)--> C2

* The OUTER level is the paper's compiler unchanged: each pair of program qubits shares an
  outer [[4,2,2]] patch (4 data + 2 syndrome ancillas), so C1 is an ordinary translated circuit:
  encoding, logical gadgets, end-of-circuit syndrome extraction, then every qubit measured in Z.
* Every measurement in C1 comes after its last gate (checked in `flatten`), so C1 is a unitary
  followed by Z measurements of all its qubits. Its qubits -- outer data AND outer ancillas --
  become the logical qubits of the INNER level: the same translator, run on C1 with an inner
  placement, encodes each pair of them in an inner [[4,2,2]] patch. The Z measurement of an outer
  qubit is read off its inner patch's final data measurement as that lane's Zbar.
* Concatenated distance needs the inner placement to be SPREAD: no inner patch may hold two
  qubits of the same outer patch. Otherwise one inner weight-2 error -- Xbar_1 Xbar_2 = X_1 X_2,
  undetected by the inner code -- lands as a weight-2 error on one outer patch, which the outer
  code may not detect either: distance 2. Spread, an undetected inner error touches at most one
  qubit of each outer patch, which the outer code detects. On the data qubits this is the
  [[16,4,4]] code: four inner patches, two outer patches, each outer patch taking one logical
  qubit from every inner patch.

Inner layouts (`inner_layouts`), for P outer patches = 6P outer qubits in 3P inner patches:

    stride        outer qubit q -> inner patch q mod 3P, slot q div 3P. The simplest spread rule.
    role_aligned  outer patches paired (2c, 2c+1); inner patch (c, role) holds role `role` of
                  both. Each outer patch of a pair runs the same encoding and syndrome extraction,
                  so their identical gates land on the two lanes of one inner patch, side by side.
    index         outer qubits (2b, 2b+1) -> inner patch b: NOT spread (the cautionary layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

Perm = List[Tuple[int, int]]
QUBITS_PER_OUTER_PATCH = 6   # 4 data + 2 syndrome ancillas (n = 4)
ROLE_NAMES = ("d0", "d1", "d2", "d3", "a0", "a1")


@dataclass
class Flattened:
    """C1 as an inner-level program, and how its measurements map back."""

    program: QuantumCircuit
    #: flat qubit -> (outer patch, role index 0..5); roles 0-3 data, 4-5 ancillas
    roles: List[Tuple[int, int]]
    #: (C1 register name, bit index) -> flat qubit measured into it
    measured: Dict[Tuple[str, int], int]
    outer_patches: int


@dataclass
class TwoLevel:
    """Everything a decoder needs about one compiled two-level circuit."""

    outer: object          # TranslationResult of the outer level (C1)
    flat: Flattened
    inner_placement: Perm
    inner: object          # TranslationResult of the inner level (C2)
    labels: Dict[str, object] = field(default_factory=dict)

    @property
    def circuit(self) -> QuantumCircuit:
        return self.inner.circuit


def translator(n_code: int, *, commute: bool, merging: bool, asap: bool = True,
               transversal: bool = False, rounds: int = 0):
    """`transversal`: in-patch CZ / XCX built transversally (the standard's gadget set).
    `rounds`: mid-circuit syndrome rounds on every patch (`TranslatorConfig.syndrome_rounds`)."""
    from translator.config import TranslatorConfig
    from translator.core import Translator

    return Translator(config=TranslatorConfig(
        n_code=n_code, enable_h_commutation=commute, enable_merging=merging,
        fault_tolerant=False, asap_order=asap, transversal_intra_two_qubit=transversal,
        syndrome_rounds=rounds,
    ))


def outer_translate(program: QuantumCircuit, placement: Perm, *, commute: bool, merging: bool,
                    asap: bool = True, transversal: bool = False, rounds: int = 0):
    from experiments.shared.simulation import ensure_measurements

    return translator(4, commute=commute, merging=merging, asap=asap,
                      transversal=transversal, rounds=rounds).translate(
        ensure_measurements(program), [tuple(p) for p in placement])


def flatten(c1: QuantumCircuit) -> Flattened:
    """
    C1's gates on one flat register, measurements dropped and recorded.

    Refuses a C1 in which any gate acts on a qubit after that qubit was measured: then C1 would
    not be "unitary, then measure everything", and deferring the measurements would be wrong.
    """
    index = {q: i for i, q in enumerate(c1.qubits)}
    roles: List[Tuple[int, int]] = []
    for register in c1.qregs:
        kind, patch = register.name.split("_")
        for bit in range(register.size):
            roles.append((int(patch), bit if kind == "data" else 4 + bit))
    if [r.name.split("_")[0] for r in c1.qregs] != ["data", "parity"] * (len(c1.qregs) // 2):
        raise ValueError(f"unexpected C1 registers: {[r.name for r in c1.qregs]}")
    program = QuantumCircuit(c1.num_qubits, c1.num_qubits)
    measured: Dict[Tuple[str, int], int] = {}
    done = set()
    # C1's SWAPs (the whole-patch HH's anchor exchange, lane realignment) are atom exchanges at
    # the outer level. One level up they are SWAPs of two logical qubits, which a relabel
    # implements exactly and for free -- the inner translator does the same for a program SWAP
    # inside one patch, but refuses one across patches, and spread puts the two qubits of an
    # outer patch in different inner patches. Every C1 SWAP is inside one outer patch, so the
    # relabel never changes which outer patch a wire belongs to (checked).
    wire = list(range(c1.num_qubits))
    for instruction in c1.data:
        name = instruction.operation.name
        qubits = [wire[index[q]] for q in instruction.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            register, bit = c1.find_bit(instruction.clbits[0]).registers[0]
            measured[(register.name, bit)] = qubits[0]
            done.add(qubits[0])
            continue
        if done.intersection(qubits):
            raise ValueError(f"{name} on {qubits} after measurement: C1 is not unitary-then-measure")
        if name == "swap":
            a, b = (index[q] for q in instruction.qubits)
            if roles[wire[a]][0] != roles[wire[b]][0]:
                raise ValueError("C1 SWAP across outer patches: the relabel would move a wire")
            wire[a], wire[b] = wire[b], wire[a]
            continue
        program.append(instruction.operation, qubits)
    for q in range(c1.num_qubits):
        program.measure(q, q)
    if len(measured) != c1.num_qubits:
        raise ValueError("C1 does not measure every qubit exactly once")
    return Flattened(program=program, roles=roles, measured=measured,
                     outer_patches=c1.num_qubits // QUBITS_PER_OUTER_PATCH)


def is_spread(placement: Perm, roles: Sequence[Tuple[int, int]]) -> bool:
    """No inner patch holds two qubits of one outer patch."""
    seen = set()
    for (inner, _), (outer, _) in zip(placement, roles):
        if (inner, outer) in seen:
            return False
        seen.add((inner, outer))
    return True


def inner_layouts(flat: Flattened) -> Dict[str, Perm]:
    """The named inner placements of the module docstring."""
    count = len(flat.roles)
    blocks = -(-count // 2)
    layouts: Dict[str, Perm] = {
        "stride": [(q % blocks, q // blocks) for q in range(count)],
        "index": [(q // 2, q % 2) for q in range(count)],
    }
    # role-aligned: outer patches in pairs; an unpaired last patch gets inner patches of its own
    aligned: Dict[int, Tuple[int, int]] = {}
    next_block = 0
    pairs = flat.outer_patches // 2
    by_key = {(outer, role): q for q, (outer, role) in enumerate(flat.roles)}
    for pair in range(pairs):
        for role in range(QUBITS_PER_OUTER_PATCH):
            for slot, outer in enumerate((2 * pair, 2 * pair + 1)):
                aligned[by_key[(outer, role)]] = (next_block, slot)
            next_block += 1
    if flat.outer_patches % 2:
        last = flat.outer_patches - 1
        for role in range(QUBITS_PER_OUTER_PATCH):
            aligned[by_key[(last, role)]] = (next_block, 0)
            next_block += 1
    layouts["role_aligned"] = [aligned[q] for q in range(count)]
    return layouts


def inner_translate(flat: Flattened, placement: Perm, *, commute: bool, merging: bool,
                    asap: bool = True, transversal: bool = False, rounds: int = 0):
    return translator(4, commute=commute, merging=merging, asap=asap,
                      transversal=transversal, rounds=rounds).translate(
        flat.program, [tuple(p) for p in placement])


def compile_two_level(program: QuantumCircuit, outer_placement: Perm, *, outer_commute: bool,
                      outer_merging: bool, inner_placement: Optional[Perm] = None,
                      inner_layout: str = "role_aligned", inner_commute: bool = False,
                      inner_merging: bool = False, asap: bool = True,
                      require_spread: bool = True) -> TwoLevel:
    outer = outer_translate(program, outer_placement, commute=outer_commute,
                            merging=outer_merging, asap=asap)
    flat = flatten(outer.circuit)
    if inner_placement is None:
        inner_placement = inner_layouts(flat)[inner_layout]
    if require_spread and not is_spread(inner_placement, flat.roles):
        raise ValueError("inner placement is not spread")
    inner = inner_translate(flat, inner_placement, commute=inner_commute, merging=inner_merging,
                            asap=asap)
    return TwoLevel(outer=outer, flat=flat, inner_placement=[tuple(p) for p in inner_placement],
                    inner=inner)
