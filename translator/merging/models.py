"""Data models for merge detection and application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Tuple

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode


Lane = int


@dataclass(frozen=True)
class PlannedOp:
    gate: Gate
    qargs: Tuple[object, ...]


@dataclass(frozen=True, eq=False)
class MacroReplacement:
    """
    Instruction replacement plan covering HH, XX, ZZ, CXT, CZT and XCXT merges.

    `nodes` holds every original instruction the macro subsumes, in the order the
    rewriter will encounter them. All but the last are dropped; the macro is
    emitted in place of the last, which is why the gates it replaces must be
    safe to delay until that point.

    A macro spans up to k instructions, not two: HH and the transversal macros are
    whole-patch operations, so at k slots per patch it takes k separate program
    gates to make one. `first`/`second` remain available as the window endpoints
    the conflict analysis reasons about -- the earliest gate that moves and the
    point everything lands -- and are exactly `nodes[0]` and `nodes[-1]`.
    """

    nodes: Tuple[DAGOpNode, ...]
    macro_gate: Gate
    macro_qargs: Tuple[object, ...]
    blocked_qubits: Tuple[object, ...]
    pre_ops: Tuple[PlannedOp, ...] = ()
    post_ops: Tuple[PlannedOp, ...] = ()
    #: Branch v2-tvd-explore (`explore_switches.*_MAJORITY_UNDO`): program-level gates
    #: emitted right after the macro to take it back off the lanes no program gate asked
    #: for. Empty for every macro the shipped pass builds. Each one costs a single of the
    #: kind the macro replaces, so savings subtract them.
    undo_ops: Tuple[PlannedOp, ...] = ()

    @property
    def first(self) -> DAGOpNode:
        """The earliest instruction the macro absorbs; dropped by the rewriter."""
        return self.nodes[0]

    @property
    def second(self) -> DAGOpNode:
        """The instruction the macro is emitted in place of."""
        return self.nodes[-1]

    @property
    def dropped(self) -> Tuple[DAGOpNode, ...]:
        """Every instruction removed outright, i.e. all but the emission point."""
        return self.nodes[:-1]

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError(
                f"a macro must subsume at least two instructions, got {len(self.nodes)}"
            )


@dataclass(frozen=True)
class PairMergeStats:
    """
    Summary of potential HH/CX/CZ merges for a pair of qubits.

    Note on `cx_merges`/`cz_merges`/`xcx_merges`: these are always 0. The analyzer
    asks the transversal detector about the two qubits sitting in *separate solo*
    patches, and a transversal macro over one lane per patch fuses nothing. That has
    been true since the pairwise implementation, which arrived at the same answer by
    rejecting the slot map rather than by declining to look; `packer.merge_packer`
    weights these terms and so has never seen them contribute.
    """

    hh_merges: int = 0
    xx_merges: int = 0
    zz_merges: int = 0
    cx_merges: int = 0
    cz_merges: int = 0
    xcx_merges: int = 0
    twoq_between: int = 0
    cx_between: int = 0
    cz_between: int = 0
    xcx_between: int = 0
    hh_layer_distance: int = 0
    cx_layer_distance: int = 0
    cz_layer_distance: int = 0
    xcx_layer_distance: int = 0


@dataclass(frozen=True)
class SoloMergeStats:
    """Summary of savings available when a qubit is placed alone in its patch."""

    h_count: int = 0       # H gates → HH macro (solo H is always whole-patch Hadamard)
    xx_merges: int = 0     # sequential X-X pairs on solo lane (usually 0 after cancellation)
    zz_merges: int = 0     # sequential Z-Z pairs on solo lane (usually 0 after cancellation)
    twoq_count: int = 0    # total CX/CZ/XCX gates on this qubit (for error-rate bias)


@dataclass
class MergeConflictInfo:
    """
    Conflict metadata for a single merge candidate.

    conflict_savings_1q/2q are the summed gate-count savings contributed by
    conflicting merges. conflict_weight applies the 1q/2q weights.
    """

    conflicts: Set[int]
    conflict_savings_1q: int
    conflict_savings_2q: int
    conflict_weight: float
    group: int | None = None


@dataclass
class _WaitingState:
    lane: Lane
    node: DAGOpNode
    remaining: float


@dataclass
class _PendingTransversal:
    node: DAGOpNode
    gate_name: str
    control_patch: int
    target_patch: int
    control_qubit: object
    target_qubit: object
    control_slot: int
    target_slot: int
