"""
HH merging pre-processing
=========================
Given a permutation and a program circuit:
    For each patch:
        two pointers, one for each qubit
        increment pointers along layers until they encounter an H
            if p0 == H and p1 == H:
                HH
            if only one pointer has an H, we create a ghost pointer for the H lane
                then increment ghost pointer and other pointer, checking for two qubit gates
                increment until either GP and OP see two-qubit gates (scary vibes, could be path)
                OR we run out lookahead
                if we encounter a H on OP and neither of the exit conditions have been met,
                we delay the original H to an HH

This way, we minimize runtime, we create HH's where reasonable without introducing much delay to the circuit.

Pseudo-code sketch
------------------
def preprocess(program_circuit, permutation, settings):
    dag = circuit_to_dag(program_circuit)
    layers = list(dag.serial_layers())  # cirq-style "moments"
    patch_pairs = build_patch_pairs(permutation)  # patch -> (lane_a, lane_b)
    merges = []
    for patch, (qa, qb) in patch_pairs.items():
        merges += find_hh_merges(qa, qb, layers, settings)
    dag = apply_merges(dag, merges)  # replace delayed H pairs with HH macro gates
    return dag_to_circuit(dag), merges

def find_hh_merges(qa, qb, layers, settings):
    cursors = {qa: LayerCursor(qa), qb: LayerCursor(qb)}
    ghost = None  # stores pending H info (lane, layer_idx, node, remaining budget)
    results = []
    while not cursors_exhausted(cursors, layers):
        for lane in (qa, qb):
            cursors[lane].advance_to_next_gate(layers)

        ga = cursors[qa].current_gate()
        gb = cursors[qb].current_gate()
        if ga is None or gb is None:
            break  # one lane ran out of gates

        if ga.name == "h" and gb.name == "h":
            results.append(make_merge(qa, qb, ga, gb))
            cursors[qa].consume()
            cursors[qb].consume()
            ghost = None
            continue

        # spawn or update ghost pointer when exactly one lane sees H
        if ga.name == "h" and gb.name != "h" and ghost is None:
            ghost = GhostState(qubit=qa, node=ga, budget=settings.hh_budget)
            cursors[qa].pause()
            cursors[qb].consume()
            continue
        if gb.name == "h" and ga.name != "h" and ghost is None:
            ghost = GhostState(qubit=qb, node=gb, budget=settings.hh_budget)
            cursors[qb].pause()
            cursors[qa].consume()
            continue

        if ghost:
            lead = qb if ghost.qubit == qa else qa
            lead_gate = cursors[lead].current_gate()
            if is_two_qubit(lead_gate) or touches_other_patch(lead_gate):
                pending = ghost
                ghost = None
                cursors[pending.qubit].consume_original_h()
                continue

            cost = settings.delay_costs.get(lead_gate.name, settings.default_cost)
            if ghost.budget < cost:
                pending = ghost
                ghost = None
                cursors[pending.qubit].consume_original_h()
                continue

            if lead_gate.name == "h":
                results.append(make_merge(ghost.qubit, lead, ghost.node, lead_gate))
                cursors[lead].consume()
                cursors[ghost.qubit].consume_original_h()
                ghost = None
            else:
                ghost.budget -= cost
                cursors[lead].consume()
                continue

        # neither lane is on an H or ghost was resolved
        if not ghost:
            cursors[qa].consume()
            cursors[qb].consume()

    return results

def apply_merges(dag, merges):
    for merge in merges:
        dag.remove_op_node(merge.left)
        dag.remove_op_node(merge.right)
        dag.apply_operation_back(HHGate(), qargs=[merge.qa, merge.qb], metadata=merge.timing)
    return dag

The actual implementation can build atop these helpers, keeping the delay-cost
table and budget in a config object so HH, CX, and CZ merging share the same tuning knobs.

CNOT
look at ipad
"""

from .analysis import PairMergeAnalyzer
from .gates import CXTGate, CZTGate, XCXTGate, HHGate, XXGate, ZZGate
from .models import MacroReplacement, MergeConflictInfo, PairMergeStats, PlannedOp, SoloMergeStats
from .conflicts import compute_merge_conflicts
from .selection import select_best_group
from .preprocessing import preprocess_program_for_merging, preprocess_program_for_merging_with_stats
from .settings import DEFAULT_SETTINGS, MergeSettings

__all__ = [
    "CXTGate",
    "CZTGate",
    "XCXTGate",
    "DEFAULT_SETTINGS",
    "HHGate",
    "XXGate",
    "ZZGate",
    "MergeConflictInfo",
    "MergeSettings",
    "MacroReplacement",
    "PairMergeAnalyzer",
    "PairMergeStats",
    "PlannedOp",
    "SoloMergeStats",
    "compute_merge_conflicts",
    "select_best_group",
    "preprocess_program_for_merging",
    "preprocess_program_for_merging_with_stats",
]
