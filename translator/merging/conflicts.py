"""Conflict detection and weighting for merge candidates."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from iceberg.costs import DEFAULT_WORK_WEIGHTS as _CODE_WORK_WEIGHTS
from iceberg.costs import pauli_macro_cost
from iceberg.costs import work_table as code_work_table

from .models import MacroReplacement, MergeConflictInfo

# Code size the merging pass is calibrated for; see `packer.work_table` for the
# same knob on the packing side. Costs live in `iceberg.costs` so that the
# constructions and the cost model cannot drift apart.
DEFAULT_CODE_SIZE: int = 4

DEFAULT_MERGE_WORK_TABLE: Dict[str, Tuple[int, int]] = code_work_table(DEFAULT_CODE_SIZE)

DEFAULT_MERGE_WORK_WEIGHTS: Tuple[float, float] = _CODE_WORK_WEIGHTS

#: Macro name -> the per-lane gate it stands in for. One table, so the savings
#: estimate in `preprocessing` and the candidate ranking in `selection` cannot
#: disagree about what a macro replaces.
MACRO_TO_ORIGINAL: Dict[str, str] = {
    "HH": "H",
    "XX": "X",
    "ZZ": "Z",
    "CXT": "CX",
    "CZT": "CZ",
    "XCXT": "XCX",
}


def macro_cost(
    name: str, fused: int, work_table: Dict[str, Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """
    (1q, 2q) cost of emitting one macro that fuses `fused` original gates.

    XX/ZZ macros fuse *any* group of m >= 2 coincident Paulis, and the emitted
    cost depends on m -- m + (m mod 2) single-qubit gates -- so the fixed
    work-table entry (the whole-patch m = k value) is wrong for partial groups.
    Everything else (HH, the transversal macros) emits the same circuit whatever
    subset of lanes asked for it, so the table entry is the cost. Shared by
    `preprocessing`, `selection`, and the conflict weighting here, so the three
    estimates cannot disagree about what a macro costs.
    """
    if name in ("XX", "ZZ"):
        return pauli_macro_cost(fused)
    return work_table.get(name)


def merge_work_table(n_code: int = DEFAULT_CODE_SIZE) -> Dict[str, Tuple[int, int]]:
    """
    Gate costs at a given code size.

    The whole-patch and transversal macros scale with n -- HH is (n, 0) and CXT is
    (0, n) -- so scoring a k=10 merge against the n=4 table would misprice exactly
    the merges that only exist at larger k.
    """
    return code_work_table(n_code)


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
    dag: Optional[DAGCircuit] = None,
    order_index: Optional[Dict[DAGOpNode, int]] = None,
) -> List[MergeConflictInfo]:
    """
    Compute conflicts between merge candidates.

    Detects three classes of conflict:
      1. Geometric (segment crossings in layer × qubit space) via Bentley–Ottmann.
      2. Shared-endpoint (two merges referencing the same DAG node).
      3. Cross-gate: merge B's blocking window would delay a 2-qubit gate G
         past merge A's macro, even though G precedes A in the original circuit.
         Requires ``dag`` and ``order_index`` to be supplied.
    """
    if work_table is None:
        work_table = DEFAULT_MERGE_WORK_TABLE

    segments = _build_segments(merges, layer_index, qubit_indices)
    conflicts = _bentley_ottmann_conflicts(segments)
    _add_shared_endpoint_conflicts(merges, conflicts)

    if dag is not None and order_index is not None:
        _add_cross_gate_conflicts(merges, layer_index, order_index, dag, conflicts)
    _add_first_in_window_conflicts(merges, layer_index, order_index, conflicts)

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


def filter_single_qubit_conflicting_with_transversal(
    single_records: Sequence[MacroReplacement],
    transversal_records: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
) -> List[MacroReplacement]:
    """
    Remove any single-qubit records that conflict with a transversal record.

    Uses the same first-in-window logic as _add_first_in_window_conflicts:
    a single-qubit record A conflicts with transversal record B when:
      - A.first falls in B's window [B.first_pos, B.second_pos)
      - A.first touches any qubit in B.macro_qargs
      - A.second fires after B.second

    This must be called AFTER transversal detection, since transversal records
    are not available when select_best_group runs.
    """
    def pos(n: DAGOpNode) -> Tuple[int, int]:
        return (layer_index.get(n, 0), order_index.get(n, 0))

    filtered = []
    for merge_a in single_records:
        a_first_pos = pos(merge_a.first)
        a_second_pos = pos(merge_a.second)
        a_first_qubits = set(merge_a.first.qargs)
        conflict = False
        for merge_b in transversal_records:
            b_first_pos = pos(merge_b.first)
            b_second_pos = pos(merge_b.second)
            if not (b_first_pos <= a_first_pos < b_second_pos):
                continue
            if not (a_first_qubits & set(merge_b.macro_qargs)):
                continue
            if a_second_pos <= b_second_pos:
                continue
            conflict = True
            break
        if not conflict:
            filtered.append(merge_a)
    return filtered


def _add_first_in_window_conflicts(
    merges: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    conflicts: Dict[int, Set[int]],
) -> None:
    """
    Detect conflicts where merge A's first node falls inside merge B's window.

    When A.first is dropped by the rewriter, the operation it represents is
    reconstituted by the macro at A.second. If A.first sits inside B's active
    window [B.first, B.second) AND touches any qubit B's macro operates on AND
    A.second fires after B.second, then A.first has been silently moved from
    before B's macro to after it — a reordering violation.

    This is checked symmetrically for both (A, B) and (B, A).

    Still `O(C^2)` -- the pairs genuinely have to be visited -- but the per-pair work is
    hoisted out. Positions and qarg sets used to be rebuilt inside the inner loop, so a
    set of B's macro qargs was constructed once per *pair* rather than once per merge.
    """
    def pos(n: DAGOpNode) -> Tuple[int, int]:
        return (layer_index.get(n, 0), order_index.get(n, 0))

    first_positions = [pos(merge.first) for merge in merges]
    second_positions = [pos(merge.second) for merge in merges]
    first_qubit_sets = [set(merge.first.qargs) for merge in merges]
    macro_qubit_sets = [set(merge.macro_qargs) for merge in merges]

    for idx_a in range(len(merges)):
        a_first_pos = first_positions[idx_a]
        a_second_pos = second_positions[idx_a]
        a_first_qubits = first_qubit_sets[idx_a]

        for idx_b in range(len(merges)):
            if idx_a == idx_b:
                continue
            if idx_b in conflicts.get(idx_a, set()):
                continue

            b_first_pos = first_positions[idx_b]
            b_second_pos = second_positions[idx_b]

            # A.first must fall strictly inside B's window.
            if not (b_first_pos <= a_first_pos < b_second_pos):
                continue

            # A.first must touch a qubit that B's macro operates on.
            if a_first_qubits.isdisjoint(macro_qubit_sets[idx_b]):
                continue

            # A's macro must fire after B's macro — otherwise there's no reorder.
            if a_second_pos <= b_second_pos:
                continue

            _record_conflict(conflicts, idx_a, idx_b)


def _add_cross_gate_conflicts(
    merges: Sequence[MacroReplacement],
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    dag: DAGCircuit,
    conflicts: Dict[int, Set[int]],
) -> None:
    """
    Detect indirect conflicts bridged by a 2-qubit gate G.

    Merge A and merge B conflict via G when ALL of the following hold:
      - G is in B's blocking window: pos(B.first) <= pos(G) < pos(B.second)
        (B's rewriter block would delay G until after B.second fires)
      - G should precede A's macro in the original circuit: pos(G) < pos(A.second)
      - B fires after A: pos(B.second) > pos(A.second)
        (so the delay pushes G past A's macro — a reorder)
      - G touches one of B's blocked_qubits (so it is actually caught by B's block)
      - G touches one of A's macro_qargs (so it has an ordering relationship with A)

    Indexed rather than enumerated, which is worth a paragraph because the obvious
    reading of that predicate is a triple loop over (A, B, every two-qubit gate) and
    that is what this used to be. `O(C^2 * G2)` is cubic in circuit size, since the
    candidate count grows with the gate count, and it dominated everything: on H-dense
    circuits at a few thousand gates it was >99% of the whole merging pass, and the
    `pos` closure alone was called 9.5M times on a 2000-gate circuit.

    The observation that removes the third factor is that **the qualifying G depend only
    on B**. Gather them once per B into `first_hit`, mapping each qubit to the earliest
    position at which a gate caught by B's block touches it, and the pair test collapses
    to a lookup over A's macro qargs:

        conflict(A, B)  <=>  pos(B.second) > pos(A.second)
                             and min(first_hit[q] for q in A.macro_qargs) < pos(A.second)

    `first_hit` carries the window and B's block; the comparison carries the other two
    conditions. Cost is `O(C * window + C^2 * k)`.

    The edge set is unchanged, and that is the property to protect when editing here --
    `tests/test_merging_conflict_index.py` pins it against the enumerated form on random
    circuits. Two features of the old loop looked load-bearing and were not: the
    already-conflicting `continue` and the `break` at the first witness are both
    short-circuits over an existence question, and which witness G supplied the edge was
    never recorded. Iterating B outer rather than A outer is likewise invisible, since
    `_record_conflict` is symmetric.
    """
    pos_cache: Dict[DAGOpNode, Tuple[int, int]] = {}

    def pos(n: DAGOpNode) -> Tuple[int, int]:
        cached = pos_cache.get(n)
        if cached is None:
            cached = (layer_index.get(n, 0), order_index.get(n, 0))
            pos_cache[n] = cached
        return cached

    # Pre-build the set of all merge nodes so we can skip them as G candidates.
    # A first/second node is never emitted as a plain gate, so it can never be
    # the bridging gate that gets reordered.
    merge_nodes: Set[DAGOpNode] = set()
    for m in merges:
        merge_nodes.add(m.first)
        merge_nodes.add(m.second)

    # 2-qubit non-merge op nodes, sorted by position with a parallel key array, so a
    # blocking window can be bisected out instead of found by scanning every gate.
    bridges = [
        n for n in dag.topological_op_nodes()
        if len(n.qargs) == 2 and n not in merge_nodes
    ]
    if not bridges:
        return
    bridges.sort(key=pos)
    keys = [pos(n) for n in bridges]

    a_second_positions = [pos(m.second) for m in merges]
    a_macro_qubits = [set(m.macro_qargs) for m in merges]

    for idx_b, merge_b in enumerate(merges):
        b_blocked = set(merge_b.blocked_qubits)
        if not b_blocked:
            # No block, so no gate is ever caught and B bridges to nothing.
            continue
        b_first_pos = pos(merge_b.first)
        b_second_pos = pos(merge_b.second)

        low = bisect_left(keys, b_first_pos)
        high = bisect_left(keys, b_second_pos)

        # Earliest position at which a gate inside B's window and caught by B's block
        # touches each qubit. `keys` ascends over the slice, so the first write is the
        # minimum and later ones can be skipped.
        first_hit: Dict[object, Tuple[int, int]] = {}
        for offset in range(low, high):
            qargs = bridges[offset].qargs
            if b_blocked.isdisjoint(qargs):
                continue
            key = keys[offset]
            for qubit in qargs:
                if qubit not in first_hit:
                    first_hit[qubit] = key
        if not first_hit:
            continue

        for idx_a in range(len(merges)):
            if idx_a == idx_b:
                continue
            a_second_pos = a_second_positions[idx_a]
            # B must fire strictly after A for its delay to push G past A.
            if b_second_pos <= a_second_pos:
                continue
            # Skip pairs that are already recorded as conflicting.
            if idx_b in conflicts.get(idx_a, set()):
                continue
            for qubit in a_macro_qubits[idx_a]:
                hit = first_hit.get(qubit)
                if hit is not None and hit < a_second_pos:
                    _record_conflict(conflicts, idx_a, idx_b)
                    break


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
    """
    (1q, 2q) saved by each candidate: what its `len(nodes)` originals would have
    cost, minus the macro. The count used to be hardcoded at 2, which was only
    right at k=2 -- it understated HH at larger patches and ignored the size of
    partial Pauli fusions entirely.
    """
    savings: List[Tuple[int, int]] = []
    for merge in merges:
        name = getattr(merge.macro_gate, "name", "").upper()
        orig = _macro_to_original(name)
        if orig is None:
            savings.append((0, 0))
            continue
        fused = len(merge.nodes)
        base = work_table.get(orig)
        macro = macro_cost(name, fused, work_table)
        if base is None or macro is None:
            savings.append((0, 0))
            continue
        # Undo gates (branch v2-tvd-explore) are singles of the same kind, emitted.
        net = fused - len(getattr(merge, "undo_ops", ()))
        s1 = net * base[0] - macro[0]
        s2 = net * base[1] - macro[1]
        savings.append((s1, s2))
    return savings


def _macro_to_original(name: str) -> Optional[str]:
    mapping = {
        "HH": "H",
        "XX": "X",
        "ZZ": "Z",
        "CXT": "CX",
        "CZT": "CZ",
        "XCXT": "XCX",
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