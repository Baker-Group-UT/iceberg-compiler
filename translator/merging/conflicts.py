"""Conflict detection and weighting for merge candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from qiskit.dagcircuit import DAGOpNode

from .models import MacroReplacement, MergeConflictInfo

DEFAULT_MERGE_WORK_TABLE: Dict[str, Tuple[int, int]] = {
    "X": (2, 0),
    "Z": (2, 0),
    "H": (7, 4),
    "HH": (4, 0),
    "XX": (2, 0),
    "ZZ": (2, 0),
    "CX": (0, 4),
    "CZ": (1, 3),
    "CXT": (0, 4),
    "CZT": (0, 4),
}

DEFAULT_MERGE_WORK_WEIGHTS: Tuple[float, float] = (1.0, 13.0)


@dataclass(frozen=True)
class _Segment:
    idx: int
    merge: MacroReplacement
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self) -> "_Segment":
        if (self.x1, self.y1) <= (self.x2, self.y2):
            return self
        return _Segment(
            idx=self.idx,
            merge=self.merge,
            x1=self.x2,
            y1=self.y2,
            x2=self.x1,
            y2=self.y1,
        )

    def y_at(self, x: float) -> float:
        if self.x1 == self.x2:
            return float(min(self.y1, self.y2))
        t = (x - self.x1) / (self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)

    def bbox(self) -> Tuple[int, int, int, int]:
        return (
            min(self.x1, self.x2),
            max(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.y1, self.y2),
        )


def compute_merge_conflicts(
    merges: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    qubit_indices: Dict[object, int],
    *,
    work_table: Optional[Dict[str, Tuple[int, int]]] = None,
    work_weights: Tuple[float, float] = DEFAULT_MERGE_WORK_WEIGHTS,
) -> List[MergeConflictInfo]:
    """
    Compute conflicts between merge candidates using a Bentley–Ottmann sweep.

    Conflicts are detected by segment crossings in (layer, qubit_index) space
    plus shared-endpoint conflicts (same DAG node).
    """
    if work_table is None:
        work_table = DEFAULT_MERGE_WORK_TABLE

    segments = _build_segments(merges, layer_index, qubit_indices)
    conflicts = _bentley_ottmann_conflicts(segments)
    _add_shared_endpoint_conflicts(merges, conflicts)

    savings_by_merge = _merge_savings_counts(merges, work_table)
    infos: List[MergeConflictInfo] = []
    for idx, _ in enumerate(merges):
        conflict_ids = conflicts.get(idx, set())
        s1 = sum(savings_by_merge[conflict_idx][0] for conflict_idx in conflict_ids)
        s2 = sum(savings_by_merge[conflict_idx][1] for conflict_idx in conflict_ids)
        weight = s1 * work_weights[0] + s2 * work_weights[1]
        infos.append(
            MergeConflictInfo(
                conflicts=set(conflict_ids),
                conflict_savings_1q=s1,
                conflict_savings_2q=s2,
                conflict_weight=weight,
                group=None,
            )
        )
    return infos


def _build_segments(
    merges: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    qubit_indices: Dict[object, int],
) -> List[_Segment]:
    segments: List[_Segment] = []
    for idx, merge in enumerate(merges):
        first_q = _single_qubit_for_node(merge.first)
        second_q = _single_qubit_for_node(merge.second)
        if first_q is None or second_q is None:
            continue
        if first_q not in qubit_indices or second_q not in qubit_indices:
            continue
        x1 = layer_index.get(merge.first, 0)
        x2 = layer_index.get(merge.second, 0)
        y1 = qubit_indices[first_q]
        y2 = qubit_indices[second_q]
        segments.append(_Segment(idx=idx, merge=merge, x1=x1, y1=y1, x2=x2, y2=y2).normalized())
    return segments


def _single_qubit_for_node(node: DAGOpNode) -> Optional[object]:
    if not node.qargs:
        return None
    if len(node.qargs) != 1:
        return None
    return node.qargs[0]


def _add_shared_endpoint_conflicts(
    merges: Sequence[MacroReplacement],
    conflicts: Dict[int, Set[int]],
) -> None:
    node_to_merges: Dict[DAGOpNode, List[int]] = {}
    for idx, merge in enumerate(merges):
        node_to_merges.setdefault(merge.first, []).append(idx)
        node_to_merges.setdefault(merge.second, []).append(idx)

    for ids in node_to_merges.values():
        if len(ids) < 2:
            continue
        for i in ids:
            conflicts.setdefault(i, set()).update(j for j in ids if j != i)


def _merge_savings_counts(
    merges: Sequence[MacroReplacement],
    work_table: Dict[str, Tuple[int, int]],
) -> List[Tuple[int, int]]:
    savings: List[Tuple[int, int]] = []
    for merge in merges:
        name = getattr(merge.macro_gate, "name", "").upper()
        orig = _macro_to_original(name)
        if orig is None:
            savings.append((0, 0))
            continue
        base = work_table.get(orig)
        macro = work_table.get(name)
        if base is None or macro is None:
            savings.append((0, 0))
            continue
        s1 = 2 * base[0] - macro[0]
        s2 = 2 * base[1] - macro[1]
        savings.append((s1, s2))
    return savings


def _macro_to_original(name: str) -> Optional[str]:
    mapping = {
        "HH": "H",
        "XX": "X",
        "ZZ": "Z",
        "CXT": "CX",
        "CZT": "CZ",
    }
    return mapping.get(name)


# --- Bentley–Ottmann sweep for segment intersections ---


def _bentley_ottmann_conflicts(segments: Sequence[_Segment]) -> Dict[int, Set[int]]:
    """
    Sweep-line intersection detection (Bentley–Ottmann style).

    For each segment start, check intersections against active segments whose
    x-range overlaps the new segment.
    """
    conflicts: Dict[int, Set[int]] = {s.idx: set() for s in segments}
    if not segments:
        return conflicts

    active: List[_Segment] = []
    segments_sorted = sorted(segments, key=lambda s: (s.x1, s.x2, s.y1, s.y2))

    for seg in segments_sorted:
        active = [a for a in active if a.x2 >= seg.x1]
        for other in active:
            if _segments_intersect(seg, other):
                _record_conflict(conflicts, seg.idx, other.idx)
        active.append(seg)

    return conflicts


def _segments_intersect(a: _Segment, b: _Segment) -> bool:
    ax1, ax2, ay1, ay2 = a.x1, a.x2, a.y1, a.y2
    bx1, bx2, by1, by2 = b.x1, b.x2, b.y1, b.y2

    if max(ax1, ax2) < min(bx1, bx2) or max(bx1, bx2) < min(ax1, ax2):
        return False

    return _ccw(ax1, ay1, ax2, ay2, bx1, by1) != _ccw(ax1, ay1, ax2, ay2, bx2, by2) and \
        _ccw(bx1, by1, bx2, by2, ax1, ay1) != _ccw(bx1, by1, bx2, by2, ax2, ay2)


def _ccw(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> bool:
    return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)


def _record_conflict(conflicts: Dict[int, Set[int]], a: int, b: int) -> None:
    if a == b:
        return
    conflicts.setdefault(a, set()).add(b)
    conflicts.setdefault(b, set()).add(a)
