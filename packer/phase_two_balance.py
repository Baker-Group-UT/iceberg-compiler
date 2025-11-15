from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple, Set

from qiskit import QuantumCircuit

from .phase_one_graph import GraphHeuristics, EdgeItem, build_graph


Decision = Tuple[str, int, Optional[int]]


@dataclass
class BalanceResult:
    decisions: List[Decision]
    qubit_to_patch: Dict[int, int]
    solo_budget_used: int


def balance_graph(
    graph: GraphHeuristics,
    solo_budget: Optional[int] = None,
    approach: Literal["balanced", "greedy", "defensive"] = "balanced",
) -> BalanceResult:
    chosen_budget = graph.solo_budget_min if solo_budget is None else solo_budget
    if chosen_budget < graph.solo_budget_min or chosen_budget > graph.solo_budget_max:
        raise ValueError(
            f"Solo budget {chosen_budget} is outside "
            f"[{graph.solo_budget_min}, {graph.solo_budget_max}]"
        )

    load: List[EdgeItem] = list(graph.edges)
    load.sort(key=lambda e: e.weight)
    mean_weight = (
        sum(edge.weight for edge in load) / len(load) if load else 0.0
    )

    unassigned: Set[int] = set(graph.nodes.keys())
    decisions: List[Decision] = []
    ignored: Set[Decision] = set()
    qubit_to_patch: Dict[int, int] = {}
    patch_contents: Dict[int, List[int]] = {}
    patch_locked: Dict[int, bool] = {}
    next_patch_id = 0
    solo_budget_remaining = chosen_budget

    def create_patch(locked: bool = False) -> Optional[int]:
        nonlocal next_patch_id
        if len(patch_contents) >= graph.max_patches:
            return None
        patch_id = next_patch_id
        next_patch_id += 1
        patch_contents[patch_id] = []
        patch_locked[patch_id] = locked
        return patch_id

    def find_patch_with_space(
        min_capacity: int = 1,
        exclude: Optional[Set[int]] = None,
        allow_locked: bool = False,
        allow_new: bool = True,
    ) -> Optional[int]:
        for patch_id, members in patch_contents.items():
            if exclude and patch_id in exclude:
                continue
            if patch_locked.get(patch_id, False) and not allow_locked:
                continue
            if graph.patch_capacity - len(members) >= min_capacity:
                return patch_id
        if allow_new:
            return create_patch()
        return None

    def assign_qubit_to_patch(qubit: int, patch_id: int) -> bool:
        if qubit in qubit_to_patch:
            return True
        members = patch_contents.get(patch_id)
        if members is None:
            return False
        if patch_locked.get(patch_id, False) and members:
            return False
        if len(members) >= graph.patch_capacity:
            return False
        members.append(qubit)
        qubit_to_patch[qubit] = patch_id
        unassigned.discard(qubit)
        return True

    def rollback_qubit(qubit: int) -> None:
        patch_id = qubit_to_patch.pop(qubit, None)
        if patch_id is None:
            return
        members = patch_contents.get(patch_id)
        if members and qubit in members:
            members.remove(qubit)
        unassigned.add(qubit)

    def solo_qubit(qubit: int) -> bool:
        nonlocal solo_budget_remaining
        if qubit not in unassigned or solo_budget_remaining <= 0:
            return False
        patch_id = create_patch(locked=True)
        if patch_id is None:
            return False
        if assign_qubit_to_patch(qubit, patch_id):
            solo_budget_remaining -= 1
            return True
        rollback_qubit(qubit)
        return False

    def group_same(i: int, j: int) -> bool:
        if i not in unassigned and j not in unassigned:
            patch_i = qubit_to_patch.get(i)
            patch_j = qubit_to_patch.get(j)
            return patch_i is not None and patch_i == patch_j
        if i in unassigned and j in unassigned:
            patch_id = find_patch_with_space(min_capacity=2, allow_locked=False)
            if patch_id is None:
                return False
            if assign_qubit_to_patch(i, patch_id) and assign_qubit_to_patch(j, patch_id):
                return True
            rollback_qubit(i)
            rollback_qubit(j)
            return False
        if i in unassigned and j not in unassigned:
            patch_j = qubit_to_patch.get(j)
            if patch_j is None or patch_locked.get(patch_j, False):
                return False
            if assign_qubit_to_patch(i, patch_j):
                return True
            rollback_qubit(i)
            return False
        if j in unassigned and i not in unassigned:
            patch_i = qubit_to_patch.get(i)
            if patch_i is None or patch_locked.get(patch_i, False):
                return False
            if assign_qubit_to_patch(j, patch_i):
                return True
            rollback_qubit(j)
            return False
        return False

    def group_diff(i: int, j: int) -> bool:
        patch_i = qubit_to_patch.get(i)
        patch_j = qubit_to_patch.get(j)

        if i in unassigned and patch_i is None:
            target = find_patch_with_space(exclude={patch_j} if patch_j is not None else None)
            if target is None or not assign_qubit_to_patch(i, target):
                return False
            patch_i = target
        if j in unassigned and patch_j is None:
            target = find_patch_with_space(exclude={patch_i} if patch_i is not None else None)
            if target is None or not assign_qubit_to_patch(j, target):
                rollback_qubit(i)
                return False
            patch_j = target

        if patch_i is None or patch_j is None:
            return False
        if patch_i == patch_j:
            return False
        return True

    left = 0
    right = len(load) - 1

    def get_next_edge() -> Optional[Tuple[EdgeItem, str]]:
        nonlocal left, right
        if left > right:
            return None
        if approach == "greedy":
            edge = load[left]
            left += 1
            return edge, "good"
        if approach == "defensive":
            edge = load[right]
            right -= 1
            return edge, "good"
        # balanced
        left_edge = load[left]
        right_edge = load[right]
        if left == right:
            left += 1
            return left_edge, "good"
        left_delta = abs(left_edge.weight - mean_weight)
        right_delta = abs(right_edge.weight - mean_weight)
        if left_delta >= right_delta:
            left += 1
            return left_edge, "good"
        right -= 1
        return right_edge, "bad"

    def handle_single_qubit() -> bool:
        if len(unassigned) != 1:
            return False
        remaining = next(iter(unassigned))
        if solo_budget_remaining > 0 and solo_qubit(remaining):
            decisions.append(("solo", remaining, None))
            return True
        existing_patch = find_patch_with_space(
            min_capacity=1, allow_new=False, allow_locked=False
        )
        if existing_patch is None:
            return False
        before = list(patch_contents.get(existing_patch, []))
        if not assign_qubit_to_patch(remaining, existing_patch):
            return False
        partner = before[0] if before else remaining
        decisions.append(("same", partner, remaining))
        return True

    # for item in load:
    #     print(item)
    # print()
    while left <= right and unassigned:
        if len(unassigned) == 1 and handle_single_qubit():
            continue
        next_item = get_next_edge()
        # print(next_item)
        if next_item is None:
            break
        edge, side = next_item
        key = (edge.kind, edge.i, edge.j)

        if key in ignored:
            continue

        if edge.kind == "solo":
            if edge.i not in unassigned:
                continue
            if side == "good":
                ignored.add(key)
            else:
                if solo_budget_remaining > 0 and solo_qubit(edge.i):
                    decisions.append(("solo", edge.i, None))
                else:
                    ignored.add(key)
            continue

        if edge.j is None:
            continue
        if edge.i not in unassigned or edge.j not in unassigned:
            continue

        if side == "good":
            if edge.kind == "same":
                if group_same(edge.i, edge.j):
                    decisions.append(("same", edge.i, edge.j))
                else:
                    ignored.add(key)
            elif edge.kind == "diff":
                ignored.add(key)
        else:
            if edge.kind == "diff":
                if group_diff(edge.i, edge.j):
                    decisions.append(("diff", edge.i, edge.j))
                else:
                    ignored.add(key)
            else:
                ignored.add(key)

    if unassigned:
        remaining = sorted(unassigned)
        while remaining:
            batch = remaining[: graph.patch_capacity]
            remaining = remaining[graph.patch_capacity :]
            if len(batch) == 1:
                qubit = batch[0]
                if solo_budget_remaining > 0 and solo_qubit(qubit):
                    decisions.append(("solo", qubit, None))
                    continue
                existing_patch = find_patch_with_space(
                    min_capacity=1, allow_new=False, allow_locked=False
                )
                if existing_patch is None:
                    raise ValueError(
                        "Unable to place final qubit without exceeding solo budget."
                    )
                before = list(patch_contents.get(existing_patch, []))
                if not assign_qubit_to_patch(qubit, existing_patch):
                    raise ValueError(
                        f"Failed to assign qubit {qubit} to patch {existing_patch}."
                    )
                partner = before[0] if before else qubit
                decisions.append(("same", partner, qubit))
                continue

            patch_id = find_patch_with_space(min_capacity=len(batch))
            if patch_id is None:
                raise ValueError("Unable to assign all qubits within patch limits.")
            existing_members = list(patch_contents.get(patch_id, []))
            anchor = existing_members[0] if existing_members else batch[0]
            for idx, qubit in enumerate(batch):
                if not assign_qubit_to_patch(qubit, patch_id):
                    raise ValueError(f"Failed to assign qubit {qubit} to patch {patch_id}.")
                partner = anchor if (existing_members or idx > 0) else qubit
                decisions.append(("same", partner, qubit))

    solo_budget_used = chosen_budget - solo_budget_remaining

    return BalanceResult(
        decisions=decisions,
        qubit_to_patch=qubit_to_patch,
        solo_budget_used=solo_budget_used,
    )


def _generate_random_circuit(
    num_qubits: int = 6,
    num_gates: int = 10,
    gate_set: Tuple[str, ...] = ("x", "z", "h", "cx"),
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_gates):
        gate = random.choice(gate_set)
        if gate == "cx":
            control, target = random.sample(range(num_qubits), 2)
            qc.cx(control, target)
        elif gate == "x":
            qc.x(random.randrange(num_qubits))
        elif gate == "z":
            qc.z(random.randrange(num_qubits))
        else:
            qc.h(random.randrange(num_qubits))
    return qc


def _smoke_test() -> None:
    print("=== Phase-Two Balance Smoke Test ===")
    random.seed(9)
    qc = _generate_random_circuit()
    print("Random circuit:")
    print(qc.draw())
    gate_lsr_table = {
        "X": 0.997,
        "Z": 0.997,
        "H": 0.958,
        "HH": 0.986,
        "CX": 0.961,
        "CXT": 0.961,
        "CZ": 0.967,
        "CZT": 0.961,
    }
    graph = build_graph(
        program_circuit=qc,
        gate_lsr_table=gate_lsr_table,
        patch_capacity=2,
        max_patches=7,
    )
    print(
        f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"solo_budget_range=[{graph.solo_budget_min}, {graph.solo_budget_max}]"
    )

    result = balance_graph(
        graph,
        solo_budget=7,
        approach="defensive",
    )
    print("\nDecisions (press Enter to step through):")
    for idx, decision in enumerate(result.decisions, start=1):
        input(f"({idx}/{len(result.decisions)}) Press Enter...")
        print(f"  -> {decision}")
    print("\nFinal qubit_to_patch mapping:")
    for qubit, patch in sorted(result.qubit_to_patch.items()):
        print(f"  q{qubit} -> patch {patch}")
    print(f"\nSolo budget used: {result.solo_budget_used}")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    _smoke_test()
