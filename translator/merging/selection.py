"""Selection and grouping of merge candidates based on conflicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .conflicts import compute_merge_conflicts
from .models import MacroReplacement, MergeConflictInfo


@dataclass(frozen=True)
class MergePriority:
    savings_weight: float
    conflict_weight: float
    order_key: Tuple[int, int, int]


def select_best_group(
    candidates: Sequence[MacroReplacement],
    layer_index: Dict,
    order_index: Dict,
    qubit_indices: Dict[object, int],
) -> List[MacroReplacement]:
    """
    Assign groups to single-qubit merge candidates and return group 0 merges.

    Two-qubit merges are excluded from grouping.
    """
    single_indices = [
        idx for idx, merge in enumerate(candidates) if _is_single_qubit_merge(merge)
    ]
    if not single_indices:
        return []

    conflict_info_all = compute_merge_conflicts(candidates, layer_index, qubit_indices)
    priorities = _build_priorities(candidates, single_indices, layer_index, order_index)
    conflict_info = _filter_conflicts(conflict_info_all, single_indices, priorities)

    group_of = {idx: None for idx in single_indices}
    analysis_order = sorted(single_indices, key=lambda i: priorities[i].order_key)

    pending: List[Tuple[int, int]] = [(idx, 0) for idx in analysis_order]
    seen = set()
    while pending:
        idx, start_group = pending.pop(0)
        key = (idx, start_group, group_of.get(idx))
        if key in seen:
            continue
        seen.add(key)
        group_of[idx] = None
        _place_merge(
            idx,
            start_group,
            group_of,
            conflict_info,
            priorities,
            pending,
        )

    return _pick_best_group(candidates, single_indices, group_of, priorities)


def _place_merge(
    idx: int,
    start_group: int,
    group_of: Dict[int, int | None],
    conflict_info: Dict[int, MergeConflictInfo],
    priorities: Dict[int, MergePriority],
    pending: List[Tuple[int, int]],
) -> None:
    group = start_group
    while True:
        conflicts = conflict_info[idx].conflicts
        conflict_in_group = [
            other for other in conflicts if group_of.get(other) == group
        ]
        if not conflict_in_group:
            group_of[idx] = group
            return

        # Resolve against the first conflicting merge in this group (deterministic).
        other = sorted(conflict_in_group, key=lambda i: priorities[i].order_key)[0]
        if _is_better(priorities[idx], priorities[other]):
            group_of[other] = None
            pending.append((other, group + 1))
            continue
        group += 1


def _pick_best_group(
    candidates: Sequence[MacroReplacement],
    single_indices: Sequence[int],
    group_of: Dict[int, int | None],
    priorities: Dict[int, MergePriority],
) -> List[MacroReplacement]:
    group_scores: Dict[int, float] = {}
    for idx in single_indices:
        group = group_of.get(idx)
        if group is None:
            continue
        group_scores[group] = group_scores.get(group, 0.0) + priorities[idx].savings_weight

    if not group_scores:
        return []

    best_group = max(sorted(group_scores.keys()), key=lambda g: group_scores[g])
    return [candidates[idx] for idx in single_indices if group_of.get(idx) == best_group]


def _is_better(a: MergePriority, b: MergePriority) -> bool:
    if a.savings_weight != b.savings_weight:
        return a.savings_weight > b.savings_weight
    if a.conflict_weight != b.conflict_weight:
        return a.conflict_weight < b.conflict_weight
    return a.order_key < b.order_key


def _filter_conflicts(
    infos: Sequence[MergeConflictInfo],
    single_indices: Iterable[int],
    priorities: Dict[int, MergePriority],
) -> Dict[int, MergeConflictInfo]:
    single_set = set(single_indices)
    result: Dict[int, MergeConflictInfo] = {}
    for idx in single_indices:
        info = infos[idx]
        filtered = set(conflict for conflict in info.conflicts if conflict in single_set)
        conflict_weight = sum(priorities[c].savings_weight for c in filtered)
        result[idx] = MergeConflictInfo(
            conflicts=filtered,
            conflict_savings_1q=info.conflict_savings_1q,
            conflict_savings_2q=info.conflict_savings_2q,
            conflict_weight=conflict_weight,
            group=None,
        )
    return result


def _build_priorities(
    candidates: Sequence[MacroReplacement],
    single_indices: Iterable[int],
    layer_index: Dict,
    order_index: Dict,
) -> Dict[int, MergePriority]:
    priorities: Dict[int, MergePriority] = {}
    for idx in single_indices:
        merge = candidates[idx]
        savings = _merge_savings_weight(merge)
        order_key = (
            min(layer_index.get(merge.first, 0), layer_index.get(merge.second, 0)),
            order_index.get(merge.first, 0),
            order_index.get(merge.second, 0),
        )
        priorities[idx] = MergePriority(
            savings_weight=savings,
            conflict_weight=0.0,
            order_key=order_key,
        )
    return priorities


def _merge_savings_weight(merge: MacroReplacement) -> float:
    from .conflicts import DEFAULT_MERGE_WORK_TABLE, DEFAULT_MERGE_WORK_WEIGHTS

    name = getattr(merge.macro_gate, "name", "").upper()
    mapping = {
        "HH": "H",
        "XX": "X",
        "ZZ": "Z",
    }
    orig = mapping.get(name)
    if orig is None:
        return 0.0
    base = DEFAULT_MERGE_WORK_TABLE.get(orig)
    macro = DEFAULT_MERGE_WORK_TABLE.get(name)
    if base is None or macro is None:
        return 0.0
    s1 = 2 * base[0] - macro[0]
    s2 = 2 * base[1] - macro[1]
    return s1 * DEFAULT_MERGE_WORK_WEIGHTS[0] + s2 * DEFAULT_MERGE_WORK_WEIGHTS[1]


def _is_single_qubit_merge(merge: MacroReplacement) -> bool:
    name = getattr(merge.macro_gate, "name", "").lower()
    return name in {"hh", "xx", "zz"}
