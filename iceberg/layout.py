"""
layout.py — the [[n, n-2, 2]] Iceberg code convention, in one place.

Every module that needs to know *where* a logical qubit lives inside a patch —
the gate constructions in `iceberg_module.py`, the cross-patch constructions in
`iceberg_set.py`, the classical decoder in `iceberg_postselect.py`, and the
cost model in `costs.py` — goes through `CodeLayout`. Keeping the convention in
a single object is what makes the family generalize: the k=2 code used to encode
this convention implicitly in hardcoded indices and lookup tables, and those
copies disagreed in ways that cancelled only at n=4.

The convention
--------------
A patch holds `n` data qubits (n even) and encodes `k = n - 2` logical qubits::

    data index      0            1 .. n-2          n-1
    role            X-anchor     lanes 0 .. k-1    Z-anchor

    stabilizers     S_X = X^{\\otimes n}      S_Z = Z^{\\otimes n}
    logical X       Xbar_i = X[x_anchor] X[lane(i)]
    logical Z       Zbar_i = Z[z_anchor] Z[lane(i)]
    |0...0>_L       (|0^n> + |1^n>) / sqrt(2)

The two anchors are what make the logical operators weight-2 instead of
weight-n. They are *not* interchangeable: X-type logical operations are built
against the X-anchor and Z-type against the Z-anchor. Mixing them up produces a
gate that is a correct logical operation at k=2 and silently wrong beyond it,
because the "wrong anchor" version differs from the intended one by the
patch-wide map `C_A` below.

Useful identities (both verified in tests/test_iceberg_logical_action.py):

    X[x_anchor] X[z_anchor]  ==  prod_i Xbar_i     (mod S_X)
    Z[x_anchor] Z[z_anchor]  ==  prod_i Zbar_i     (mod S_Z)

so exchanging the two anchors applies the patch-wide Clifford

    C_A :  Xbar_i -> prod_{j != i} Xbar_j ,  Zbar_i -> prod_{j != i} Zbar_j

which is the GF(2) linear map A = I + J (every logical bit gets XORed with the
total parity). A is an involution because k is even. `C_A` is exactly the
correction that turns transversal H into a patch-wide logical Hadamard, and
because `C_A` is realized by the *physical swap of the two anchors* it costs one
SWAP at any n — see `IcebergModule.HH`. At k=2, and only at k=2, A is the lane
permutation (0 1), which is why the k=2 code could get away with swapping the
two lane qubits instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

__all__ = [
    "CodeLayout",
    "DATA_REGISTER_PREFIX",
    "PARITY_REGISTER_PREFIX",
    "FLAG_REGISTER_PREFIX",
    "code_size_of",
    "patch_count_of",
    "patch_of_qubit",
    "qubits_per_patch_of",
]

# Register-name prefixes emitted by `IcebergModule` / `physical.FlaggedGates`.
# They are the contract that lets any consumer recover the layout of an encoded
# circuit without being told the code size.
DATA_REGISTER_PREFIX = "data_"
PARITY_REGISTER_PREFIX = "parity_"
FLAG_REGISTER_PREFIX = "flag_"


@dataclass(frozen=True)
class CodeLayout:
    """Qubit roles and logical-operator supports of one [[n, n-2, 2]] patch."""

    n: int

    def __post_init__(self) -> None:
        if self.n < 4:
            raise ValueError(f"n must be at least 4, got {self.n}")
        if self.n % 2:
            raise ValueError(f"n must be even, got {self.n}")

    # --- shape ---

    @property
    def k(self) -> int:
        """Number of logical qubits (lanes) in the patch."""
        return self.n - 2

    def __len__(self) -> int:
        return self.n

    # --- roles ---

    @property
    def x_anchor(self) -> int:
        """Data index shared by every logical X operator."""
        return 0

    @property
    def z_anchor(self) -> int:
        """Data index shared by every logical Z operator."""
        return self.n - 1

    def lane(self, i: int) -> int:
        """Data index carrying logical qubit `i`."""
        self.validate_lane(i)
        return i + 1

    def lanes(self) -> range:
        return range(self.k)

    def lane_indices(self) -> list[int]:
        """Data indices of all lanes, in lane order."""
        return [i + 1 for i in range(self.k)]

    # --- logical operator supports ---

    def x_support(self, i: int) -> tuple[int, int]:
        """Data indices of Xbar_i."""
        return (self.x_anchor, self.lane(i))

    def z_support(self, i: int) -> tuple[int, int]:
        """Data indices of Zbar_i."""
        return (self.z_anchor, self.lane(i))

    # --- validation ---

    def validate_lane(self, i: int) -> None:
        if not isinstance(i, (int,)) or isinstance(i, bool):
            raise TypeError(f"lane index must be an int, got {i!r}")
        if i < 0 or i >= self.k:
            raise ValueError(f"lane index {i} out of range for k={self.k}")

    def validate_lanes(self, lanes: Iterable[int]) -> list[int]:
        """Validate a collection of distinct lanes; returns them as a list."""
        out = list(lanes)
        for i in out:
            self.validate_lane(i)
        if len(set(out)) != len(out):
            raise ValueError(f"duplicate lane indices: {out}")
        return out

    # --- classical decoding of a Z-basis data measurement ---

    def is_codeword(self, bits: Sequence[int]) -> bool:
        """
        True if `bits` (data index order) is a valid Z-basis codeword outcome.

        Codewords have even weight: that is the S_Z = Z^{\\otimes n} check.
        """
        if len(bits) != self.n:
            raise ValueError(f"expected {self.n} bits, got {len(bits)}")
        return sum(bits) % 2 == 0

    def decode_logical(self, bits: Sequence[int]) -> list[int]:
        """
        Logical bit values from a Z-basis data measurement, in lane order.

        Zbar_i = Z[z_anchor] Z[lane(i)], so the logical value of lane `i` is just
        the parity of those two measured bits. Decoding against the *X*-anchor
        instead yields C_A applied to the logical string (see module docstring) —
        the trap the hardcoded k=2 lookup table used to hide.
        """
        if len(bits) != self.n:
            raise ValueError(f"expected {self.n} bits, got {len(bits)}")
        reference = bits[self.z_anchor]
        return [bits[self.lane(i)] ^ reference for i in range(self.k)]


# --------------------------------------------------------------------------
# Recovering the layout of an already-built encoded circuit
# --------------------------------------------------------------------------
#
# Consumers downstream of the translator (the noise model's grid, the per-code
# gate-count statistics) need to know which patch each physical qubit belongs to.
# They used to divide the qubit index by a hardcoded 6 = n + 2, which is wrong at
# any other code size and wrong in fault-tolerant mode at *every* code size,
# because the flag ancillas are appended after the patches. Reading the register
# names instead is exact in both cases.


def _data_registers(circuit: "QuantumCircuit") -> List:
    return [reg for reg in circuit.qregs if reg.name.startswith(DATA_REGISTER_PREFIX)]


def code_size_of(circuit: "QuantumCircuit") -> int:
    """
    The code size n of an encoded circuit, from its data registers.

    Raises ValueError if the circuit has no patches or mixes code sizes (the
    compiler assumes a uniform code set).
    """
    registers = _data_registers(circuit)
    if not registers:
        raise ValueError("circuit has no Iceberg data registers")
    sizes = {len(reg) for reg in registers}
    if len(sizes) != 1:
        raise ValueError(f"circuit mixes code sizes: {sorted(sizes)}")
    return sizes.pop()


def patch_count_of(circuit: "QuantumCircuit") -> int:
    """How many encoded patches the circuit holds."""
    return len(_data_registers(circuit))


def qubits_per_patch_of(circuit: "QuantumCircuit") -> int:
    """Physical qubits owned by one patch: n data qubits plus 2 parity qubits."""
    return code_size_of(circuit) + 2


def patch_of_qubit(circuit: "QuantumCircuit") -> Dict[int, int]:
    """
    Map physical qubit index to patch index.

    Flag ancillas belong to no patch and are omitted, so callers should treat a
    missing key as "not part of a patch" rather than assume dense coverage.
    """
    mapping: Dict[int, int] = {}
    patch_by_register: Dict[str, int] = {}
    for patch, register in enumerate(_data_registers(circuit)):
        patch_by_register[register.name] = patch
        suffix = register.name[len(DATA_REGISTER_PREFIX):]
        patch_by_register[f"{PARITY_REGISTER_PREFIX}{suffix}"] = patch

    for register in circuit.qregs:
        patch = patch_by_register.get(register.name)
        if patch is None:
            continue
        for qubit in register:
            mapping[circuit.find_bit(qubit).index] = patch
    return mapping
