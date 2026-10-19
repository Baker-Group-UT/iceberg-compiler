"""
iceberg_module.py — one encoded patch of the [[n, n-2, 2]] Iceberg code.

`IcebergModule` owns n data qubits plus 2 parity qubits and exposes the logical
gate set as methods. Qubit roles and the logical-operator convention live in
`layout.CodeLayout`; physical gate emission (including the flagged
fault-tolerant gadgets) lives in `physical.FlaggedGates`.

Design of the gate constructions
--------------------------------
Every logical operator is weight-2 and shares an anchor with all the others of
its type: `Xbar_i = X[x_anchor] X[lane_i]` and `Zbar_i = Z[z_anchor] Z[lane_i]`.
That gives the gate set a Z/X duality which the constructions follow literally:

    Z-type ops  anchor on data[z_anchor]  and are built from `cz`
    X-type ops  anchor on data[x_anchor]  and are built from `x_basis_cz`

So S/CZ/RZ/RZZ are Z-type, SX/XCX/RX are X-type, and each X-type construction is
its Z-type partner with (z_anchor, cz) replaced by (x_anchor, x_basis_cz). Using
the wrong anchor yields a gate that happens to be correct at k=2 and is silently
wrong beyond it, because the two anchors differ by the patch-wide map C_A (see
`layout`); that was the origin of the old k=2-only `SX` and `XCX` special cases.

For the whole-patch Hadamard, C_A is realized by physically swapping the two
anchors, so `HH` is transversal H plus one SWAP at any n.
"""

from __future__ import annotations

from logging import warning
from typing import Collection

from qiskit import QuantumCircuit, QuantumRegister

from .base import CircuitClass
from .layout import CodeLayout
from .physical import FlaggedGates, measure_many
from .utils import SharedInt, rand_clifford_angles

SWAP_GATE_ALLOWED = True
BARRIERS = True


class IcebergModule(CircuitClass):
    """
    A single [[n, n-2, 2]] Iceberg error-detection patch.

    Registers created per instance:
      - data:   n qubits — carries the k = n-2 logical qubits plus the two
                anchors, `data[layout.x_anchor]` and `data[layout.z_anchor]`,
                which are the shared support of the logical X and Z operators.
      - parity: 2 qubits — the syndrome ancillas used by `decode`.

    Logical qubits are addressed by lane index 0 .. k-1.
    """

    def __init__(
        self,
        n: int,
        circuit: QuantumCircuit,
        ids: SharedInt,
        initialize: str | None = None,
    ):
        self.layout = CodeLayout(n)
        self._ids = ids
        self.circuit = circuit
        self._encoded = False

        self._data = QuantumRegister(self.n, name=f"data_{int(ids)}")
        circuit.add_register(self._data)
        self._parity = QuantumRegister(2, name=f"parity_{int(ids)}")
        circuit.add_register(self._parity)
        ids += 1

        self.data_qubits = list(self._data)
        self.parity_qubits = list(self._parity)

        if initialize is not None:
            self.encode(initialize)

    # --- shape and qubit roles ---

    @property
    def n(self) -> int:
        """Physical data qubits in the patch."""
        return self.layout.n

    @property
    def k(self) -> int:
        """Logical qubits (lanes) in the patch."""
        return self.layout.k

    @property
    def x_anchor(self) -> object:
        """Data qubit shared by every logical X operator."""
        return self.data_qubits[self.layout.x_anchor]

    @property
    def z_anchor(self) -> object:
        """Data qubit shared by every logical Z operator."""
        return self.data_qubits[self.layout.z_anchor]

    def lane(self, i: int) -> object:
        """Data qubit carrying logical qubit `i`."""
        return self.data_qubits[self.layout.lane(i)]

    def gates(self, fault_tolerant: bool = False) -> FlaggedGates:
        """Physical two-qubit gate emitter for this patch's circuit."""
        return FlaggedGates(
            self.circuit,
            self._ids,
            fault_tolerant=fault_tolerant,
            allow_swap=SWAP_GATE_ALLOWED,
        )

    @staticmethod
    def _as_lanes(i: int | Collection[int]) -> list[int]:
        return [i] if isinstance(i, int) else list(i)

    # --- initialization / encoding / decoding ---

    def encode(self, initialize: str | None = None):
        """
        Initialize the patch into |initialize>_L.

        Accepts a binary string of length k (e.g. "00" or "0+"); defaults to all
        zeros, prepared as the n-qubit GHZ state (|0^n> + |1^n>)/sqrt(2).
        """
        if self._encoded:
            warning(f"{self.__class__.__name__} is already encoded")
            return

        if initialize is None:
            initialize = "0" * self.k
        elif (initialize.count("0") + initialize.count("+") != self.k) or (
            len(initialize) != self.k
        ):
            raise ValueError("Invalid initialization string")

        self._encoded = True

        if "+" in initialize:
            warning(f"{self.__class__.__name__}: + state initialization not implemented")
            return

        self.circuit.h(self.data_qubits[0])
        for i in range(self.n - 1):
            self.circuit.cx(self.data_qubits[i], self.data_qubits[i + 1])

    def decode(self, key_prefix: str = ""):
        """
        Measure both stabilizers into the parity register.

        parity[0] takes the Z^{\\otimes n} syndrome (data parity), parity[1] the
        X^{\\otimes n} syndrome. Any 1 in the register means a detected error.
        """
        for dq in self.data_qubits:
            self.circuit.cx(dq, self.parity_qubits[0])
        if BARRIERS:
            self.BARRIER()

        for dq in self.data_qubits:
            self.circuit.h(dq)
            self.circuit.cx(dq, self.parity_qubits[1])
            self.circuit.h(dq)
        if BARRIERS:
            self.BARRIER()

        measure_many(self.circuit, self.parity_qubits, f"m_parity_{key_prefix}")

    #: CNOT order of `syndrome_round` at n = 4, as (check, data index). "Z" is parity[0], which
    #: collects Z^{\otimes 4} (data controls it); "X" is parity[1], prepared and read in the X
    #: basis, which collects X^{\otimes 4} (it controls the data). An exhaustive search over the
    #: 840 interleavings that measure both checks (72 qualify; 2026-10-10) picked one in which
    #: every single fault inside the round either flips one of its two outcomes or leaves no
    #: logical error. The naive order -- Z check, then X check -- has two hook faults (an ancilla
    #: error spreading to two data qubits: a weight-2 logical operator) that do neither.
    ROUND_SCHEDULE_4 = (("Z", 0), ("X", 0), ("X", 1), ("Z", 1),
                        ("Z", 2), ("X", 2), ("X", 3), ("Z", 3))

    def syndrome_round(self, key: str):
        """
        One mid-circuit measurement of both stabilizers into a new register `key`.

        Uses the patch's two parity qubits without resetting them: in the noiseless circuit
        every check reads 0, which leaves the ancilla in |0>, the state the next round (and the
        final `decode`) expects; under noise an ancilla flip persists into later outcomes,
        which a decoder reading the whole record sees. Fault-tolerant at n = 4 (see
        `ROUND_SCHEDULE_4`); other code sizes are not implemented.
        """
        if self.n != 4:
            raise NotImplementedError("syndrome_round is implemented for n = 4 only")
        z_check, x_check = self.parity_qubits
        self.circuit.h(x_check)
        for check, index in self.ROUND_SCHEDULE_4:
            data = self.data_qubits[index]
            if check == "Z":
                self.circuit.cx(data, z_check)
            else:
                self.circuit.cx(x_check, data)
        self.circuit.h(x_check)
        measure_many(self.circuit, self.parity_qubits, key)

    def measure(self, key: str):
        """Measure the data qubits into a classical register named `key`."""
        measure_many(self.circuit, self.data_qubits, key)

    # --- logical Paulis ---

    def X(self, i: int | Collection[int]):
        """Apply logical X to lane(s) `i`."""
        self._apply_pauli(i, self.circuit.x, self.x_anchor)

    def Z(self, i: int | Collection[int]):
        """Apply logical Z to lane(s) `i`."""
        self._apply_pauli(i, self.circuit.z, self.z_anchor)

    def _apply_pauli(self, i, emit, anchor):
        """
        Xbar_i = X[x_anchor] X[lane_i], so an odd number of lanes also flips the
        anchor; an even number leaves it alone because the anchor factors cancel.
        """
        lanes = self.layout.validate_lanes(self._as_lanes(i))
        if len(lanes) % 2:
            emit(anchor)
        for j in lanes:
            emit(self.lane(j))

    # --- single-lane Cliffords ---

    def S(self, i: int | Collection[int], fault_tolerant: bool = False):
        """
        Apply logical S to lane(s) `i`.

        S = exp(-i pi/4 Zbar_i) up to phase, and Zbar_i = Z[z_anchor] Z[lane_i],
        so this is the standard S-S-CZ realization of a two-qubit ZZ(pi/2)
        rotation on the Z-type support.
        """
        gates = self.gates(fault_tolerant)
        for j in self.layout.validate_lanes(self._as_lanes(i)):
            self.circuit.s(self.z_anchor)
            gates.cz(self.lane(j), self.z_anchor)
            self.circuit.s(self.lane(j))

    def SX(self, i: int | Collection[int], fault_tolerant: bool = False):
        """
        Apply logical SX to lane(s) `i`.

        The X-type dual of `S`: SX = exp(-i pi/4 Xbar_i) with Xbar_i anchored on
        `x_anchor`, realized as SX-SX-x_basis_cz. Same cost as `S` plus the two
        Hadamards inside `x_basis_cz`.
        """
        gates = self.gates(fault_tolerant)
        for j in self.layout.validate_lanes(self._as_lanes(i)):
            self.circuit.sx(self.x_anchor)
            gates.x_basis_cz(self.lane(j), self.x_anchor)
            self.circuit.sx(self.lane(j))

    def H(self, i: int | Collection[int], fault_tolerant: bool = False):
        """
        Apply logical H to lane(s) `i`.

        A Hadamard has to move a lane's support between the two anchors, so
        unlike the other single-lane Cliffords it needs a three-qubit gadget
        (4 two-qubit gates, 7 single-qubit gates, independent of n).

        Hadamards on *every* lane are much cheaper as one patch-wide operation,
        so that case is routed to `HH`.
        """
        lanes = self.layout.validate_lanes(self._as_lanes(i))
        if len(lanes) == self.k:
            self.HH(fault_tolerant=fault_tolerant)
            return

        gates = self.gates(fault_tolerant)
        for j in lanes:
            lane = self.lane(j)
            gates.cz(lane, self.z_anchor)
            self.circuit.h(self.x_anchor)
            self.circuit.h(lane)
            gates.cz(self.x_anchor, lane)
            self.circuit.h(lane)
            gates.cz(self.x_anchor, lane)
            self.circuit.h(self.x_anchor)
            self.circuit.h(lane)
            gates.cz(lane, self.z_anchor)
            self.circuit.x(self.x_anchor)
            self.circuit.z(self.z_anchor)

    def HH(self, fault_tolerant: bool = False):
        """
        Apply logical H to every lane at once: n Hadamards and one SWAP.

        Transversal H is *not* a patch-wide logical Hadamard on its own — it maps
        Xbar_i to the product of Zbar_j over j != i, i.e. it is H^k composed with
        the patch-wide map C_A of `layout`. C_A is realized by exchanging the two
        anchors, so transversal H followed by SWAP(x_anchor, z_anchor) is exactly
        H^k for every even n.

        At k=2, C_A is the lane permutation (0 1), which is why swapping the two
        *lane* qubits also worked there; that shortcut does not generalize.

        The anchor exchange is a relabeling, so it is not flag-protected even in
        fault-tolerant mode; `fault_tolerant` only reaches the CNOT decomposition
        used when `SWAP_GATE_ALLOWED` is off.
        """
        for q in self.data_qubits:
            self.circuit.h(q)
        self.gates(fault_tolerant).swap(self.x_anchor, self.z_anchor)

    # --- two-lane Cliffords ---

    def CNOT(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply logical CNOT from lane `i` to lane `j`.

        Four CNOTs between the control's Z-type support and the target's X-type
        support, which is the general recipe for a logical CNOT between two
        weight-2 logical qubits. Cost is independent of n.
        """
        self._require_distinct_lanes(i, j)
        gates = self.gates(fault_tolerant)
        gates.cx(self.lane(i), self.x_anchor)
        gates.cx(self.z_anchor, self.x_anchor)
        gates.cx(self.z_anchor, self.lane(j))
        gates.cx(self.lane(i), self.lane(j))

    def CZ(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply logical CZ between lanes `i` and `j`.

        CZ = S_i S_j exp(i pi/4 Zbar_i Zbar_j) up to phase. Expanding each factor
        on the Z-type supports, the anchor's two S gates fuse into a single Z and
        the lane S gates cancel against the ZZ rotation, leaving three CZs.
        """
        self._require_distinct_lanes(i, j)
        from translator import explore_switches

        if explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT and self.k == 2:
            for q in self.data_qubits:
                self.circuit.s(q)
            self.circuit.z(self.lane(i))
            self.circuit.z(self.lane(j))
            return
        gates = self.gates(fault_tolerant)
        gates.cz(self.lane(i), self.lane(j))
        self.circuit.z(self.z_anchor)
        gates.cz(self.lane(i), self.z_anchor)
        gates.cz(self.lane(j), self.z_anchor)

    def XCX(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply logical XCX between lanes `i` and `j` — CZ conjugated by H on both.

        The X-type dual of `CZ`, term for term: `z_anchor` becomes `x_anchor`,
        `cz` becomes `x_basis_cz`, and the fused anchor Z becomes an X. The two
        Hadamards of the anchor's `x_basis_cz` pair cancel between the two anchor
        terms, so the cost is 3 two-qubit gates and 5 single-qubit gates at any n.
        """
        self._require_distinct_lanes(i, j)
        from translator import explore_switches

        if explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT and self.k == 2:
            for q in self.data_qubits:
                self.circuit.sx(q)
            self.circuit.x(self.lane(i))
            self.circuit.x(self.lane(j))
            return
        gates = self.gates(fault_tolerant)
        self.circuit.x(self.x_anchor)
        self.circuit.h(self.x_anchor)
        gates.cx(self.x_anchor, self.lane(i))
        gates.cx(self.x_anchor, self.lane(j))
        self.circuit.h(self.x_anchor)
        gates.x_basis_cz(self.lane(i), self.lane(j))

    def SWAP(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply logical SWAP between lanes `i` and `j`.

        Exchanging two lane qubits exchanges both Xbar and Zbar for those lanes
        and touches nothing else, so a bare physical SWAP is a logical SWAP for
        any n and any lane pair. Fault-tolerant mode instead uses three logical
        CNOTs, which keeps every two-qubit gate flag-protected.
        """
        self._require_distinct_lanes(i, j)
        if not fault_tolerant:
            self.gates(fault_tolerant).swap(self.lane(i), self.lane(j))
            return
        self.CNOT(i, j, fault_tolerant=True)
        self.CNOT(j, i, fault_tolerant=True)
        self.CNOT(i, j, fault_tolerant=True)

    def _require_distinct_lanes(self, i: int, j: int) -> None:
        self.layout.validate_lane(i)
        self.layout.validate_lane(j)
        if i == j:
            raise ValueError("lane indices must differ")

    # --- parametric rotations ---

    def RZ(self, i: int, rads: float, fault_tolerant: bool = False):
        """Apply logical RZ(rads) = exp(-i rads Zbar_i / 2) to lane `i`."""
        self.layout.validate_lane(i)
        self._pauli_rotation(self.lane(i), self.z_anchor, fault_tolerant, x_type=False, rads=rads)

    def RX(self, i: int, rads: float, fault_tolerant: bool = False):
        """Apply logical RX(rads) = exp(-i rads Xbar_i / 2) to lane `i`."""
        self.layout.validate_lane(i)
        self._pauli_rotation(self.lane(i), self.x_anchor, fault_tolerant, x_type=True, rads=rads)

    def RZZ(self, i: int, j: int, rads: float, fault_tolerant: bool = False):
        """
        Apply logical RZZ(rads) = exp(-i rads Zbar_i Zbar_j / 2).

        The two z_anchor factors of Zbar_i Zbar_j cancel, so this is a plain
        two-qubit ZZ rotation on the lane qubits.
        """
        self._require_distinct_lanes(i, j)
        self._pauli_rotation(self.lane(i), self.lane(j), fault_tolerant, x_type=False, rads=rads)

    def _pauli_rotation(self, a, b, fault_tolerant: bool, *, x_type: bool, rads: float):
        """
        exp(-i rads P_a P_b / 2) as CNOT - single-qubit rotation - CNOT.

        For ZZ the rotation sits on the CNOT's target; the XX case is the same
        circuit conjugated by Hadamards, which reverses the CNOT and turns the rz
        into an rx — so emitting CNOT(a -> b) works for both and only the qubit
        carrying the rotation changes.
        """
        gates = self.gates(fault_tolerant)
        gates.cx(a, b)
        if x_type:
            self.circuit.rx(rads, a)
        else:
            self.circuit.rz(rads, b)
        gates.cx(a, b)

    # --- misc helpers ---

    def initialize_arbitrary_state(self, indices: Collection[int] | None = None):
        """
        Prepare Haar-random single-qubit states on `indices` lanes.

        Must be called *before* `encode()`: it acts on the bare lane qubits.
        """
        if indices is None:
            indices = self.layout.lanes()
        elif isinstance(indices, int):
            indices = [indices]
        for j in indices:
            theta, phi = rand_clifford_angles()
            q = self.lane(j)
            self.circuit.rz(phi, q)
            self.circuit.h(q)
            self.circuit.rz(theta, q)
            self.circuit.h(q)

    def BARRIER(self):
        """Insert a barrier across every physical qubit of this patch."""
        self.circuit.barrier(*(self.data_qubits + self.parity_qubits))
