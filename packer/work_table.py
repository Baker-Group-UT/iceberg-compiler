from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, DefaultDict

import qiskit

from translator.merging import PairMergeAnalyzer

GateName = str
WorkTable = Dict[GateName, Tuple[int, int]]

# Work table derived from Iceberg logical gate implementations (n=4, non-FT).
# Values are (num_1q_gates, num_2q_gates).
DEFAULT_WORK_TABLE: WorkTable = {
    "X": (2, 0),
    "Y": (2, 0),
    "Z": (2, 0),
    "H": (7, 4),
    "HH": (4, 0), # Swap set to 0 weight
    "XX": (2, 0),
    "ZZ": (2, 0),
    "S": (2, 1),
    "SX": (4, 1),
    "RZ": (1, 2),
    "RX": (1, 2),
    "CX": (0, 4),
    "CZ": (1, 3),
    "XCX": (5, 3),
    "CXT": (0, 4),
    "CZT": (0, 4),
    "XCXT": (8, 4),
    "SWAP": (0, 0), # Set to 0 weight
    "MEASURE": (0, 0),
}

DEFAULT_WORK_WEIGHTS: Tuple[float, float] = (1.0, 13.0)


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
    "XCX": "XCXT",
}

NEGATE_SOLO_COST = True


def gate_cost(
    gate_name: str,
    gate_work_table: WorkTable,
    *,
    w1: float,
    w2: float,
) -> float:
    """Return additive work cost for a logical gate."""
    try:
        n1, n2 = gate_work_table[gate_name]
    except KeyError as exc:
        raise KeyError(f"Missing work entry for gate '{gate_name}'.") from exc
    return w1 * n1 + w2 * n2


def build_graph(
    program_circuit: qiskit.QuantumCircuit,
    gate_work_table: WorkTable,
    *,
    work_weights: Tuple[float, float],
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
        if name in ("BARRIER", "MEASURE"):
            continue
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
    w1, w2 = work_weights

    def merge_delta(target: str, originals: Tuple[str, ...]) -> float:
        try:
            target_cost = gate_cost(target, gate_work_table, w1=w1, w2=w2)
            original_cost = sum(gate_cost(name, gate_work_table, w1=w1, w2=w2) for name in originals)
            return target_cost - original_cost
        except KeyError:
            return 0.0

    hh_merge_delta = merge_delta("HH", ("H", "H"))
    cx_merge_delta = merge_delta("CXT", ("CX", "CX"))
    cz_merge_delta = merge_delta("CZT", ("CZ", "CZ"))
    xcx_merge_delta = merge_delta("XCXT", ("XCX", "XCX"))

    nodes: Dict[int, NodeHeuristics] = {}
    edges: List[EdgeItem] = []

    for qubit in range(num_qubits):
        base = sum(
            gate_cost(name, gate_work_table, w1=w1, w2=w2)
            for name in single_qubit_gates[qubit]
        )
        solo = compute_solo_cost(
            base_cost=base,
            single_qubit_gate_names=single_qubit_gates[qubit],
            two_qubit_gate_names=qubit_two_qubit_gate_names[qubit],
            gate_work_table=gate_work_table,
            work_weights=work_weights,
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
            two_q_cost = sum(
                gate_cost(name, gate_work_table, w1=w1, w2=w2) for name in pair_gate_names
            )
            merge_stats = merge_analyzer.analyze_pair(i, j)
            delta_same = merge_stats.hh_merges * hh_merge_delta
            delta_diff = (
                merge_stats.cx_merges * cx_merge_delta
                + merge_stats.cz_merges * cz_merge_delta
                + merge_stats.xcx_merges * xcx_merge_delta
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
    gate_work_table: WorkTable,
    work_weights: Tuple[float, float],
) -> float:
    solo_total = base_cost
    w1, w2 = work_weights
    hh_delta = None
    if "HH" in gate_work_table and "H" in gate_work_table:
        hh_delta = gate_cost("HH", gate_work_table, w1=w1, w2=w2) - gate_cost(
            "H", gate_work_table, w1=w1, w2=w2
        )
    if hh_delta is not None:
        h_count = sum(1 for name in single_qubit_gate_names if name == "H")
        solo_total += h_count * hh_delta

    optimized_two_q = 0.0
    for gate_name in two_qubit_gate_names:
        optimized_name = SOLO_TWO_QUBIT_OPTIMIZATIONS.get(gate_name, gate_name)
        optimized_two_q += gate_cost(optimized_name, gate_work_table, w1=w1, w2=w2)
    solo_total += optimized_two_q
    return solo_total
