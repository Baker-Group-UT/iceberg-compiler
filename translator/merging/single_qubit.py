"""
Single-qubit merge detection for H/X/Z, at any number of slots per patch.

Two shapes of macro live here and they have different admissibility rules:

`hh`
    Whole-patch. `HH` is transversal H on all n physical qubits plus the anchor
    swap, so it acts on *every* lane of the patch whether or not a program gate
    asked for it. It is therefore only a legal substitution when the group covers
    every lane that carries a logical qubit. Lanes left idle by an uneven packing
    may be disturbed freely -- nothing reads them, and `apply_logical_hadamard`
    already fires a whole-patch HH for a patch holding one qubit.

`xx` / `zz`
    Per-lane. `Xbar_i = X[x_anchor] X[lane i]`, so a group of any size is legal and
    an even-sized group cancels the anchor factor outright, which is where the
    saving comes from. No coverage condition.

At k=2 the two rules coincide -- a group of two *is* the whole patch -- which is
why the pairwise version needed no notion of coverage.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from qiskit.dagcircuit import DAGOpNode

from .gates import HHGate, XXGate, ZZGate
from .layer import collect_events
from .models import Lane, MacroReplacement
from .settings import MergeSettings

SKIP_NAMES = {"barrier"}
MACRO_NAMES = {"hh", "xx", "zz", "cxt", "czt", "xcxt"}
CNOT_NAMES = {"cx", "cnot", "xcx"}


def find_single_qubit_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Sequence,
    settings: MergeSettings,
    *,
    gate_name: str,
    macro_gate_factory: Callable[[int], object],
    require_full_coverage: bool,
) -> List[MacroReplacement]:
    """
    Propose macros fusing `gate_name` across the lanes of one patch.

    Every gate of the right kind is tried as a group leader, and the scan forward
    collects one partner per remaining lane until the delay budget runs out. The
    result is a list of *candidates*; `selection.select_best_group` decides which
    of the overlapping ones to keep.
    """
    lane_map = {qubit: lane for lane, qubit in enumerate(qubits)}
    events = collect_events(layer_index, order_index, lane_map)
    lane_count = len(qubits)
    merges: List[MacroReplacement] = []

    for start_idx, (_, _, lanes, node) in enumerate(events):
        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES or name in MACRO_NAMES:
            continue
        if len(set(lanes)) > 1 or name == "measure" or len(node.qargs) > 1:
            continue
        if name != gate_name:
            continue
        record = _scan_forward_for_group(
            events,
            start_idx,
            lanes[0],
            node,
            settings,
            gate_name,
            macro_gate_factory,
            lane_count=lane_count,
            require_full_coverage=require_full_coverage,
            qubits=qubits,
        )
        if record is not None:
            merges.append(record)

    return merges


def _scan_forward_for_group(
    events: List[Tuple[int, int, List[Lane], DAGOpNode]],
    start_idx: int,
    leader_lane: Lane,
    leader_node: DAGOpNode,
    settings: MergeSettings,
    gate_name: str,
    macro_gate_factory: Callable[[int], object],
    *,
    lane_count: int,
    require_full_coverage: bool,
    qubits: Sequence = (),
) -> Optional[MacroReplacement]:
    """
    Walk forward from a group leader, collecting one gate per uncovered lane.

    Stops at the first event that cannot be crossed. Returns the largest group
    found, or None if it never reached a usable size: two lanes for a per-lane
    macro, every lane for a whole-patch one.
    """
    remaining = settings.budget
    found: Dict[Lane, DAGOpNode] = {leader_lane: leader_node}
    seen_cnot_lanes: set[Lane] = set()

    for idx in range(start_idx + 1, len(events)):
        _, _, lanes, node = events[idx]
        name = getattr(node.op, "name", "").lower()
        if name in SKIP_NAMES:
            continue
        if name in MACRO_NAMES:
            break

        touches_several = len(set(lanes)) > 1
        multi_qubit = len(node.qargs) > 1
        if touches_several or name == "measure":
            break

        if multi_qubit:
            # A two-qubit gate reaching outside the patch can be crossed, but only
            # on one lane: two of them on different lanes would have to be
            # reordered against each other.
            if name not in CNOT_NAMES:
                break
            if lanes:
                seen_cnot_lanes.add(lanes[0])
                if len(seen_cnot_lanes) > 1:
                    break
            remaining = _spend(remaining, settings.cost_for(name))
            if remaining is None:
                break
            continue

        lane = lanes[0]
        if name == gate_name and lane not in found:
            found[lane] = node
            if len(found) == lane_count:
                break
            continue

        # Anything else -- including a second gate of the same kind on a lane
        # already in the group -- has to be crossed on the budget.
        remaining = _spend(remaining, settings.cost_for(name))
        if remaining is None:
            break

    required = lane_count if require_full_coverage else 2
    if len(found) < required:
        if (
            require_full_coverage
            and len(qubits) == lane_count
            and _majority_undo_applies(gate_name, len(found), lane_count)
        ):
            return _build_undo_record(found, qubits, macro_gate_factory)
        return None
    return _build_record(found, macro_gate_factory)


def _majority_undo_applies(gate_name: str, wanted: int, lane_count: int) -> bool:
    """
    Branch v2-tvd-explore: whether a partial H group may fire as HH plus undo.

    `wanted` targeted H's (59 weighted units each) against HH (n + 10) plus
    `lane_count - wanted` targeted undo H's: cheaper whenever 2 * wanted > lane_count,
    for every n < 49. HH leaks no silent error and a targeted H does, so the silent count
    falls by the same rule.
    """
    from translator import explore_switches

    return (
        explore_switches.HH_MAJORITY_UNDO
        and gate_name == "h"
        and wanted >= 2
        and 2 * wanted > lane_count
    )


def _build_undo_record(
    found: Dict[Lane, DAGOpNode],
    qubits: Sequence,
    macro_gate_factory: Callable[[int], object],
) -> MacroReplacement:
    """
    A whole-patch macro over every occupied lane, undone on the lanes not in `found`.

    `macro_qargs` is the patch's occupied lanes in slot order, as `_build_record`'s is
    for a full group, so the handler's coverage check passes. The undo H's follow the
    macro directly; on those lanes the pair is the identity, so where the rewriter
    places it among their other gates does not matter, and they need no blocking.
    If the handler rejects the macro anyway it expands it into one H per lane, which
    the undo H's then cancel: still the program's gates.
    """
    from qiskit.circuit.library import HGate

    from .models import PlannedOp

    program_order = tuple(found.values())
    undo = tuple(
        PlannedOp(gate=HGate(), qargs=(qubit,))
        for lane, qubit in enumerate(qubits)
        if lane not in found
    )
    return MacroReplacement(
        nodes=program_order,
        macro_gate=macro_gate_factory(len(qubits)),
        macro_qargs=tuple(qubits),
        blocked_qubits=tuple(node.qargs[0] for node in program_order[:-1]),
        undo_ops=undo,
    )


def _spend(remaining: float, cost: float) -> Optional[float]:
    """Deduct `cost`, or None when the gate is uncrossable or the budget is gone."""
    if cost == float("inf"):
        return None
    remaining -= cost
    return None if remaining < 0 else remaining


def _build_record(
    found: Dict[Lane, DAGOpNode],
    macro_gate_factory: Callable[[int], object],
) -> MacroReplacement:
    """
    Turn a lane -> node group into a replacement plan.

    Two orders are needed and they are not the same one. `macro_qargs` goes in lane
    order, because the handlers read it as a list of slots. `nodes` goes in program
    order, because the rewriter drops all but the last and emits the macro where
    that last one was -- picking the wrong node there would move gates backwards.
    `found` is filled by a forward scan over `(layer, order)`-sorted events, so its
    insertion order is program order.
    """
    program_order = tuple(found.values())
    lane_order = tuple(found[lane] for lane in sorted(found))
    return MacroReplacement(
        nodes=program_order,
        macro_gate=macro_gate_factory(len(lane_order)),
        macro_qargs=tuple(node.qargs[0] for node in lane_order),
        blocked_qubits=tuple(node.qargs[0] for node in program_order[:-1]),
    )


def find_hh_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Sequence,
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
        require_full_coverage=True,
    )


def find_xx_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Sequence,
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
        require_full_coverage=False,
    )


def find_zz_merges_for_patch(
    layer_index: Dict[DAGOpNode, int],
    order_index: Dict[DAGOpNode, int],
    qubits: Sequence,
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
        require_full_coverage=False,
    )
