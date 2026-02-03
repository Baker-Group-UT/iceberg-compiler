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
    """Instruction replacement plan covering HH, CXT, and CZT merges."""

    first: DAGOpNode
    second: DAGOpNode
    macro_gate: Gate
    macro_qargs: Tuple[object, ...]
    blocked_qubits: Tuple[object, ...]
    pre_ops: Tuple[PlannedOp, ...] = ()
    post_ops: Tuple[PlannedOp, ...] = ()


@dataclass(frozen=True)
class PairMergeStats:
    """Summary of potential HH/CX/CZ merges for a pair of qubits."""

    hh_merges: int = 0
    xx_merges: int = 0
    zz_merges: int = 0
    cx_merges: int = 0
    cz_merges: int = 0
    twoq_between: int = 0
    hh_layer_distance: int = 0


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
