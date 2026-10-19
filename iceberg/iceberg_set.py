"""
iceberg_set.py — a register of [[n, n-2, 2]] patches with cross-patch logic.

`IcebergSet` lays out `n_program` logical qubits over `ceil(n_program / k)`
identical `IcebergModule` patches and addresses them by a single global index,
`logical = patch * k + slot`. Intra-patch operations delegate straight to the
patch; the interesting part is the cross-patch constructions.

Cross-patch two-qubit gates
---------------------------
Every logical operator of a patch is weight-2 (see `layout.CodeLayout`), so a
two-qubit logical gate between patches is the product of the corresponding
physical gate over all four pairs drawn from the two supports:

    CNOT   Z-support of control  x  X-support of target   -> 4 x cx
    CZ     Z-support of control  x  Z-support of target   -> 4 x cz
    XCX    X-support of control  x  X-support of target   -> 4 x x_basis_cz

Cost is 4 two-qubit gates regardless of n, and — because only the supports
matter — regardless of which slot each logical qubit occupies. The slot-equality
restriction the k=2 code carried was an artifact of the way it built XCX out of a
transversal-Hadamard sandwich, not a property of the code.

Transversal variants (`*_k_Transversal`) act on all k lanes at once for n
two-qubit gates rather than 4k, at the cost of pairing lane i with lane i. The
crossed anchor pattern in `CZ_k_Transversal` / `XCX_k_Transversal` is what
cancels the patch-wide map C_A that transversal Hadamards introduce.
"""

from __future__ import annotations

from typing import Collection, Dict, Iterable, List, Sequence

from qiskit import QuantumCircuit

from .iceberg_module import IcebergModule
from .layout import CodeLayout
from .physical import FlaggedGates
from .utils import SharedInt

BARRIERS = True


class IcebergSet:
    """A set of identically-sized [[n, n-2, 2]] IcebergModules."""

    def __init__(
        self,
        n_program: int,
        n: int,
        circuit: QuantumCircuit,
        initialize: str | None = None,
        encode_override: bool = False,
    ):
        """
        Create the patches needed to encode `n_program` logical qubits.

        Parameters
        ----------
        n_program : int
            Total number of logical qubits to encode.
        n : int
            Physical data qubits per patch. Must be even and at least 4.
        circuit : QuantumCircuit
            The circuit to which all operations are appended.
        initialize : str | None
            Optional binary/+ string to initialize the logical state.
        encode_override : bool
            Skip emission of the encoding circuits (the caller supplies its own
            state preparation).
        """
        self.layout = CodeLayout(n)
        self.n_program = n_program
        self.num_modules = (n_program + self.k - 1) // self.k
        self._n_total = SharedInt()
        self.modules = [
            IcebergModule(n, circuit, self._n_total) for _ in range(self.num_modules)
        ]
        self.circuit = circuit
        self._encoded = False
        self._encode_override = encode_override

        if initialize is not None:
            self.encode(initialize)

    # ---------------- shape ----------------

    @property
    def n(self) -> int:
        return self.layout.n

    @property
    def k(self) -> int:
        """Logical qubits per patch."""
        return self.layout.k

    def gates(self, fault_tolerant: bool = False) -> FlaggedGates:
        return FlaggedGates(self.circuit, self._n_total, fault_tolerant=fault_tolerant)

    # ---------------- addressing ----------------

    def locate(self, logical: int) -> tuple[int, int]:
        """Split a global logical index into (patch, slot)."""
        if logical < 0 or logical >= self.num_modules * self.k:
            raise ValueError(f"logical index {logical} out of range")
        return divmod(logical, self.k)

    def _group_by_patch(self, indices: int | Collection[int]) -> Dict[int, set]:
        """Group global logical indices into {patch: {slot, ...}}."""
        if isinstance(indices, int):
            indices = [indices]
        grouped: Dict[int, set] = {}
        for logical in indices:
            if logical < 0 or logical >= self.n_program:
                raise ValueError("Index out of range")
            patch, slot = self.locate(logical)
            slots = grouped.setdefault(patch, set())
            if slot in slots:
                raise ValueError("Duplicate index")
            slots.add(slot)
        return grouped

    def _validate_slots(self, patch: int, slot: int | Iterable[int]) -> set:
        """Check a (patch, slots) address and return the slots as a set."""
        if patch < 0 or patch >= self.num_modules:
            raise ValueError("Patch out of range")
        slots = {slot} if isinstance(slot, int) else set(slot)
        for s in slots:
            if s < 0 or s >= self.k:
                raise ValueError("Slot out of range")
        return slots

    def _broadcast(self, op: str, indices: int | Collection[int], **kwargs) -> None:
        """Apply a single-qubit patch method to logical indices, patch by patch."""
        for patch, slots in self._group_by_patch(indices).items():
            getattr(self.modules[patch], op)(slots, **kwargs)

    def _apply_at(self, op: str, patch: int, slot, **kwargs) -> None:
        """Apply a single-qubit patch method to explicit (patch, slots)."""
        slots = self._validate_slots(patch, slot)
        if slots:
            getattr(self.modules[patch], op)(slots, **kwargs)

    # ---------------- encoding / measurement ----------------

    def encode(self, initialize: str | None = None):
        """
        Encode every patch, optionally into the state named by `initialize`.

        `initialize`, if given, must have length `k * num_modules` and is split
        across patches in order.
        """
        if self._encoded:
            return
        self._encoded = True

        for i, module in enumerate(self.modules):
            init_sub = initialize[self.k * i : self.k * (i + 1)] if initialize else None
            if not self._encode_override:
                module.encode(init_sub)
        if BARRIERS:
            self.BARRIER()

    def decode(self):
        """Measure the stabilizers of every patch into its parity register."""
        if BARRIERS:
            self.BARRIER()
        for idx, module in enumerate(self.modules):
            module.decode(key_prefix=f"{idx}")

    def syndrome_round(self, index: int):
        """Mid-circuit check of every patch into `m_parity_round{index}_{patch}`."""
        for idx, module in enumerate(self.modules):
            module.syndrome_round(key=f"m_parity_round{index}_{idx}")

    def measure(self):
        """Measure every patch's data qubits into `m_data_{patch}`."""
        for idx, module in enumerate(self.modules):
            module.measure(key=f"m_data_{idx}")

    def initialize_arbitrary_state(
        self,
        codes: Collection[int] | None = None,
        indices: Collection[int] | None = None,
    ):
        """
        Prepare Haar-random logical states on selected patches/slots.

        Parameters
        ----------
        codes : Collection[int] | None
            Patch indices to initialize. `None` -> all.
        indices : Collection[int] | None
            Slots within each selected patch. `None` -> all.
        """
        if codes is None:
            codes = range(self.num_modules)
        elif isinstance(codes, int):
            codes = [codes]
        for i in codes:
            self.modules[i].initialize_arbitrary_state(indices)

    # ---------------- single-qubit logical ops ----------------

    def X(self, i: int | Collection[int]):
        """Apply logical X to the given logical qubit(s)."""
        self._broadcast("X", i)

    def Z(self, i: int | Collection[int]):
        """Apply logical Z to the given logical qubit(s)."""
        self._broadcast("Z", i)

    def S(self, i: int | Collection[int]):
        """Apply logical S to the given logical qubit(s)."""
        self._broadcast("S", i)

    def SX(self, i: int | Collection[int], fault_tolerant: bool = False):
        """Apply logical SX to the given logical qubit(s)."""
        self._broadcast("SX", i, fault_tolerant=fault_tolerant)

    def H(self, i: int | Collection[int], fault_tolerant: bool = False):
        """Apply logical H to the given logical qubit(s)."""
        self._broadcast("H", i, fault_tolerant=fault_tolerant)

    def X_at(self, patch: int, slot: int | Collection[int]):
        """Apply logical X to one or more slots of a specific patch."""
        self._apply_at("X", patch, slot)

    def Z_at(self, patch: int, slot: int | Collection[int]):
        """Apply logical Z to one or more slots of a specific patch."""
        self._apply_at("Z", patch, slot)

    def S_at(self, patch: int, slot: int | Collection[int]):
        """Apply logical S to one or more slots of a specific patch."""
        self._apply_at("S", patch, slot)

    def SX_at(self, patch: int, slot: int | Collection[int], *, fault_tolerant: bool = False):
        """Apply logical SX to one or more slots of a specific patch."""
        self._apply_at("SX", patch, slot, fault_tolerant=fault_tolerant)

    def H_at(self, patch: int, slot: int | Collection[int], *, fault_tolerant: bool = False):
        """Apply logical H to one or more slots of a specific patch."""
        self._apply_at("H", patch, slot, fault_tolerant=fault_tolerant)

    def HH(self, patches: int | Collection[int], fault_tolerant: bool = False):
        """Apply the whole-patch Hadamard to the selected patches."""
        if isinstance(patches, int):
            patches = [patches]
        for p in patches:
            self.modules[p].HH(fault_tolerant=fault_tolerant)

    def RZ(self, i: int, rads: float, fault_tolerant: bool = False):
        """Apply logical RZ to one logical qubit."""
        patch, slot = self.locate(i)
        self.modules[patch].RZ(slot, rads, fault_tolerant=fault_tolerant)

    def RX(self, i: int, rads: float, fault_tolerant: bool = False):
        """Apply logical RX to one logical qubit."""
        patch, slot = self.locate(i)
        self.modules[patch].RX(slot, rads, fault_tolerant=fault_tolerant)

    def RZZ(self, i: int, j: int, rads: float, fault_tolerant: bool = False):
        """Apply logical RZZ between two logical qubits in the same patch."""
        pi, qi = self.locate(i)
        pj, qj = self.locate(j)
        if pi != pj:
            raise NotImplementedError("Cross-patch RZZ not implemented")
        self.modules[pi].RZZ(qi, qj, rads, fault_tolerant=fault_tolerant)

    def SWAP(self, swaps: List[Sequence[int]], fault_tolerant: bool = False):
        """
        Apply logical SWAPs within patches.

        Each entry of `swaps` is [patch, slot_i, slot_j].
        """
        for patch, i, j in swaps:
            self.modules[patch].SWAP(i, j, fault_tolerant=fault_tolerant)

    # ---------------- two-qubit logical ops ----------------

    def CNOT(self, i: int, j: int, fault_tolerant: bool = False, xcx_override: bool = False):
        """
        Apply a logical CNOT between two logical qubits.

        Intra-patch delegates to the patch. Cross-patch uses four CNOTs between
        the control's Z-support and the target's X-support; the pair of lane
        qubits is left unflagged because a fault there is already detected by the
        patches' own stabilizers.

        `xcx_override` is retained for call-site compatibility and has no effect:
        cross-patch placement no longer requires the two slots to agree.
        """
        del xcx_override
        pi, qi = self.locate(i)
        pj, qj = self.locate(j)
        if pi == pj:
            self.modules[pi].CNOT(qi, qj, fault_tolerant=fault_tolerant)
            return

        control, target = self.modules[pi], self.modules[pj]
        gates = self.gates(fault_tolerant)
        gates.cx(control.lane(qi), target.lane(qj), flagged=False)
        gates.cx(control.lane(qi), target.x_anchor)
        gates.cx(control.z_anchor, target.lane(qj))
        gates.cx(control.z_anchor, target.x_anchor)

    def CZ(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical CZ between two logical qubits.

        Cross-patch: four CZs between the two Z-supports. The lane-lane and
        anchor-anchor pairs are left unflagged, matching the transversal pattern.
        """
        pi, qi = self.locate(i)
        pj, qj = self.locate(j)
        if pi == pj:
            self.modules[pi].CZ(qi, qj, fault_tolerant=fault_tolerant)
            return

        control, target = self.modules[pi], self.modules[pj]
        gates = self.gates(fault_tolerant)
        gates.cz(control.lane(qi), target.lane(qj), flagged=False)
        gates.cz(control.lane(qi), target.z_anchor)
        gates.cz(control.z_anchor, target.lane(qj))
        gates.cz(control.z_anchor, target.z_anchor, flagged=False)

    def XCX(self, i: int, j: int, fault_tolerant: bool = False):
        """
        Apply a logical XCX between two logical qubits.

        Cross-patch: the X-type dual of `CZ`, i.e. `x_basis_cz` over the four
        pairs drawn from the two X-supports. Grouping the pairs by their
        control-side qubit lets the Hadamards of each `x_basis_cz` pair cancel,
        leaving 4 two-qubit gates and 4 Hadamards at any n.

        The Hadamards are hoisted *outside* the run of two-qubit gates. Sandwiching
        each source's pair (H, CX, CX, H per source) is the same unitary — each H
        commutes with the other source's gates — but the inner Hadamards touch the
        control patch mid-run, which split the transport group and made a logical
        XCX cost two round trips where CNOT and CZ cost one. CNOT/CZ emit their four
        gates back to back already and need no hoisting.
        """
        pi, qi = self.locate(i)
        pj, qj = self.locate(j)
        if pi == pj:
            self.modules[pi].XCX(qi, qj, fault_tolerant=fault_tolerant)
            return

        control, target = self.modules[pi], self.modules[pj]
        gates = self.gates(fault_tolerant)
        # (is_lane, qubit) for each end of the two X-supports.
        sources = ((True, control.lane(qi)), (False, control.x_anchor))
        sinks = ((True, target.lane(qj)), (False, target.x_anchor))
        for _, source in sources:
            self.circuit.h(source)
        for source_is_lane, source in sources:
            for sink_is_lane, sink in sinks:
                both_lanes = source_is_lane and sink_is_lane
                gates.cx(source, sink, flagged=False if both_lanes else None)
        for _, source in sources:
            self.circuit.h(source)

    # ---------------- transversal (all-lane) variants ----------------

    def CNOT_k_Transversal(self, i: int, j: int):
        """Lane-wise logical CNOT on all k lanes: n physical CNOTs, not flagged."""
        control, target = self._transversal_pair(i, j)
        for q in range(self.n):
            self.circuit.cx(control[q], target[q])

    def CZ_k_Transversal(self, i: int, j: int):
        """
        Lane-wise logical CZ on all k lanes: n physical CZs, not flagged.

        The anchors are crossed (x_anchor to z_anchor and back) while the lanes
        run straight; that crossing is what makes the operation lane-diagonal.
        """
        control, target = self._transversal_pair(i, j)
        for q, r in self._crossed_pairs():
            self.circuit.cz(control[q], target[r])

    def XCX_k_Transversal(self, i: int, j: int):
        """
        Lane-wise logical XCX on all k lanes: n CNOTs plus 2n Hadamards.

        Transversal Hadamards on the control patch implement H^k composed with
        the patch-wide map C_A; crossing the anchors in the CNOT layer applies
        C_A a second time, so the two cancel and what remains is a lane-wise XCX.
        """
        control, target = self._transversal_pair(i, j)
        for q in range(self.n):
            self.circuit.h(control[q])
        for q, r in self._crossed_pairs():
            self.circuit.cx(control[q], target[r])
        for q in range(self.n):
            self.circuit.h(control[q])

    def _transversal_pair(self, i: int, j: int) -> tuple[list, list]:
        if i == j:
            raise ValueError("patch indices must differ")
        return self.modules[i].data_qubits, self.modules[j].data_qubits

    def _crossed_pairs(self) -> list[tuple[int, int]]:
        """Data-index pairing that exchanges the two anchors and keeps lanes."""
        x, z = self.layout.x_anchor, self.layout.z_anchor
        return [(x, z)] + [(q, q) for q in self.layout.lane_indices()] + [(z, x)]

    # ---------------- utilities ----------------

    def BARRIER(self, patches: int | Collection[int] | None = None):
        """Insert a barrier across the physical qubits of the selected patches."""
        if patches is None:
            modules = self.modules
        else:
            if isinstance(patches, int):
                patches = [patches]
            modules = [self.modules[i] for i in patches]

        qubits: list = []
        for module in modules:
            qubits.extend(module.data_qubits + module.parity_qubits)
        self.circuit.barrier(*qubits)
