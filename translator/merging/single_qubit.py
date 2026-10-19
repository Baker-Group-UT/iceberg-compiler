"""Single-qubit merge detection for H/X/Z."""

from __future__ import annotations

from typing import Dict, List, Tuple

from qiskit.dagcircuit import DAGOpNode

from .gates import HHGate, XXGate, ZZGate
from .layer import collect_events
from .models import MacroReplacement
from .settings import MergeSettings


SKIP_NAMES = {"barrier"}
MACRO_NAMES = {"hh", "xx", "zz", "cxt", "czt", "xcxt"}
CNOT_NAMES = {"cx", "cnot", "xcx"}


def find_single_qubit_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Tuple,
    settings: MergeSettings,
    *,
    gate_name: str,
    macro_gate_factory,
) -> List[MacroReplacement]:
    lane_map = {qubits[0]: 0, qubits[1]: 1}
    events = collect_events(layer_index, order_index, lane_map)
    merges: List[MacroReplacement] = []

    for start_idx, (_, _, lanes, node) in enumerate(events):
        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES or name in MACRO_NAMES:
            continue
        touches_both = len(set(lanes)) > 1
        is_measure = name == "measure"
        multi_qubit = len(node.qargs) > 1
        if touches_both or is_measure or multi_qubit:
            continue
        if name != gate_name:
            continue
        waiting_lane = lanes[0]
        merges.extend(
            _scan_forward_for_merges(
                events,
                start_idx,
                waiting_lane,
                node,
                settings,
                gate_name,
                macro_gate_factory,
            )
        )

    return merges


def _scan_forward_for_merges(
    events: List[Tuple[int, int, List[Lane], DAGOpNode]],
    start_idx: int,
    waiting_lane: Lane,
    waiting_node: DAGOpNode,
    settings: MergeSettings,
    gate_name: str,
    macro_gate_factory,
) -> List[MacroReplacement]:
    merges: List[MacroReplacement] = []
    remaining = settings.budget
    seen_cnot_lanes: set[Lane] = set()

    for idx in range(start_idx + 1, len(events)):
        _, _, lanes, node = events[idx]
        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES:
            continue
        if name in MACRO_NAMES:
            break

        touches_both = len(set(lanes)) > 1
        is_measure = name == "measure"
        multi_qubit = len(node.qargs) > 1
        if touches_both or is_measure:
            break

        if multi_qubit:
            if name not in CNOT_NAMES:
                break
            if lanes:
                seen_cnot_lanes.add(lanes[0])
                if len(seen_cnot_lanes) > 1:
                    break
            cost = settings.cost_for(name)
            if cost == float("inf"):
                break
            remaining -= cost
            if remaining < 0:
                break
            continue

        if waiting_lane in lanes:
            # NOTE: Temporarily allow same-lane gates while generating potentials.
            cost = settings.cost_for(name)
            if cost == float("inf"):
                break
            remaining -= cost
            if remaining < 0:
                break
            continue

        if name == gate_name:
            merges.append(_build_single_qubit_record(waiting_node, node, macro_gate_factory))

        cost = settings.cost_for(name)
        if cost == float("inf"):
            break
        remaining -= cost
        if remaining < 0:
            break

    return merges


def find_hh_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Tuple,
    settings: MergeSettings,
    consumed: set,
) -> List[MacroReplacement]:
    return find_single_qubit_merges_for_patch(
        layer_index,
        order_index,
        qubits,
        settings,
        gate_name="h",
        macro_gate_factory=HHGate,
    )


def find_xx_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Tuple,
    settings: MergeSettings,
    consumed: set,
) -> List[MacroReplacement]:
    return find_single_qubit_merges_for_patch(
        layer_index,
        order_index,
        qubits,
        settings,
        gate_name="x",
        macro_gate_factory=XXGate,
    )


def find_zz_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Tuple,
    settings: MergeSettings,
    consumed: set,
) -> List[MacroReplacement]:
    return find_single_qubit_merges_for_patch(
        layer_index,
        order_index,
        qubits,
        settings,
        gate_name="z",
        macro_gate_factory=ZZGate,
    )


def _build_single_qubit_record(
    first: DAGOpNode,
    second: DAGOpNode,
    macro_gate_factory,
) -> MacroReplacement:
    waiting_qubit = first.qargs[0]
    lead_qubit = second.qargs[0]
    return MacroReplacement(
        first=first,
        second=second,
        macro_gate=macro_gate_factory(),
        macro_qargs=(waiting_qubit, lead_qubit),
        blocked_qubits=(waiting_qubit,),
    )
