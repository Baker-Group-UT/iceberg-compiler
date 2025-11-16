from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, DefaultDict

import qiskit

from translator.merging import PairMergeAnalyzer

GateName = str


@dataclass
class NodeHeuristics:
    qubit_index: int
    base_cost: float
    solo_cost: float


@dataclass
class EdgeItem:
    kind: Literal["solo", "same", "diff"]
    i: int
    j: Optional[int]
    weight: float


@dataclass
class GraphHeuristics:
    nodes: Dict[int, NodeHeuristics]
    edges: List[EdgeItem]
    patch_capacity: int
    max_patches: int
    solo_budget_min: int
    solo_budget_max: int


SOLO_TWO_QUBIT_OPTIMIZATIONS: Dict[str, str] = {
    "CX": "CXT",
    "CZ": "CZT",
}

NEGATE_SOLO_COST = True


def gate_cost(gate_name: str, gate_lsr_table: Dict[str, float]) -> float:
    """Return -log10(lsr) for a gate."""
    try:
        lsr = gate_lsr_table[gate_name]
    except KeyError as exc:
        raise KeyError(f"Missing LSR entry for gate '{gate_name}'.") from exc
    return -math.log10(lsr)


def build_graph(
    program_circuit: qiskit.QuantumCircuit,
    gate_lsr_table: Dict[str, float],
    patch_capacity: int,
    max_patches: int = None,
) -> GraphHeuristics:
    num_qubits = program_circuit.num_qubits
    if max_patches is None:
        max_patches = num_qubits
    if num_qubits > patch_capacity * max_patches:
        raise ValueError(
            "Circuit with {q} qubits cannot fit in {m} patches of capacity {k}.".format(
                q=num_qubits,
                m=max_patches,
                k=patch_capacity,
            )
        )
    solo_budget_max = patch_capacity * max_patches - num_qubits
    solo_budget_min = 0 if (num_qubits % patch_capacity == 0) else 1

    single_qubit_gates: Dict[int, List[str]] = {i: [] for i in range(num_qubits)}
    two_qubit_gates: DefaultDict[Tuple[int, int], List[str]] = defaultdict(list)

    for instruction, qargs, _ in program_circuit.data:
        name = instruction.name.upper()
        qubit_indices = [program_circuit.find_bit(q).index for q in qargs]
        if len(qubit_indices) == 1:
            single_qubit_gates[qubit_indices[0]].append(name)
        elif len(qubit_indices) == 2:
            i, j = sorted(qubit_indices)
            two_qubit_gates[(i, j)].append(name)
        else:
            # Only handle one- and two-qubit gates in this heuristic builder.
            continue

    qubit_two_qubit_gate_names: Dict[int, List[str]] = {i: [] for i in range(num_qubits)}
    for (i, j), gate_names in two_qubit_gates.items():
        qubit_two_qubit_gate_names[i].extend(gate_names)
        qubit_two_qubit_gate_names[j].extend(gate_names)

    merge_analyzer = PairMergeAnalyzer(program_circuit)

    def merge_delta(target: str, originals: Tuple[str, ...]) -> float:
        try:
            target_cost = gate_cost(target, gate_lsr_table)
            original_cost = sum(gate_cost(name, gate_lsr_table) for name in originals)
            return target_cost - original_cost
        except KeyError:
            return 0.0

    hh_merge_delta = merge_delta("HH", ("H", "H"))
    cx_merge_delta = merge_delta("CXT", ("CX", "CX"))
    cz_merge_delta = merge_delta("CZT", ("CZ", "CZ"))

    nodes: Dict[int, NodeHeuristics] = {}
    edges: List[EdgeItem] = []

    for qubit in range(num_qubits):
        base = sum(gate_cost(name, gate_lsr_table) for name in single_qubit_gates[qubit])
        solo = compute_solo_cost(
            base_cost=base,
            single_qubit_gate_names=single_qubit_gates[qubit],
            two_qubit_gate_names=qubit_two_qubit_gate_names[qubit],
            gate_lsr_table=gate_lsr_table,
        )
        nodes[qubit] = NodeHeuristics(
            qubit_index=qubit,
            base_cost=base,
            solo_cost=solo,
        )
        solo_weight = -solo if NEGATE_SOLO_COST else solo
        edges.append(EdgeItem(kind="solo", i=qubit, j=None, weight=solo_weight))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            pair_gate_names = two_qubit_gates.get((i, j), [])
            two_q_cost = sum(gate_cost(name, gate_lsr_table) for name in pair_gate_names)
            merge_stats = merge_analyzer.analyze_pair(i, j)
            delta_same = merge_stats.hh_merges * hh_merge_delta
            delta_diff = (
                merge_stats.cx_merges * cx_merge_delta
                + merge_stats.cz_merges * cz_merge_delta
            )
            base_pair_cost = nodes[i].base_cost + nodes[j].base_cost + two_q_cost
            edges.append(
                EdgeItem(
                    kind="same",
                    i=i,
                    j=j,
                    weight=base_pair_cost + delta_same,
                )
            )
            edges.append(
                EdgeItem(
                    kind="diff",
                    i=i,
                    j=j,
                    weight=base_pair_cost + delta_diff,
                )
            )

    return GraphHeuristics(
        nodes=nodes,
        edges=edges,
        patch_capacity=patch_capacity,
        max_patches=max_patches,
        solo_budget_min=solo_budget_min,
        solo_budget_max=solo_budget_max,
    )


def compute_solo_cost(
    base_cost: float,
    single_qubit_gate_names: List[str],
    two_qubit_gate_names: List[str],
    gate_lsr_table: Dict[str, float],
) -> float:
    solo_total = base_cost
    hh_delta = None
    if "HH" in gate_lsr_table and "H" in gate_lsr_table:
        hh_delta = gate_cost("HH", gate_lsr_table) - gate_cost("H", gate_lsr_table)
    if hh_delta is not None:
        h_count = sum(1 for name in single_qubit_gate_names if name == "H")
        solo_total += h_count * hh_delta

    optimized_two_q = 0.0
    for gate_name in two_qubit_gate_names:
        optimized_name = SOLO_TWO_QUBIT_OPTIMIZATIONS.get(gate_name, gate_name)
        optimized_two_q += gate_cost(optimized_name, gate_lsr_table)
    solo_total += optimized_two_q
    return solo_total
