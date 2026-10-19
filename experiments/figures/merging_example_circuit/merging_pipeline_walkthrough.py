"""Narrate every step of the merging pass on the worked example, for the figure.

`merging_example_circuit.py --trace` already lists what each stage produced. This
prints the reasoning *inside* each stage -- the slot table, the rejected
candidates, the conflict graph with its edge classes, the transversal slot map
and its price -- because that is the material the merging diagram's panels are
drawn from and none of it is visible in a list of winners.

The seven steps, and the panel each feeds:

    [1] layer        every instruction gets one time layer
    [2] occupancy    patch -> slot -> qubit                          (panel a)
    [3] propose 1q   hh needs full coverage; xx/zz take subsets      (panel b)
    [4] select       conflict graph, greedy pick                     (panel c)
    [5] propose 2q   slot map, routing swaps, gain arithmetic        (d inset)
    [6] drop         step-4 winners a realignment would invalidate
    [7] rewrite      macros emitted, displaced gates delayed         (panel d)

Everything printed is read out of `translator.merging`, never restated. Where a
number has to be broken into parts to be drawable -- the transversal gain, the
step-6 predicate -- the parts are recombined and asserted against the real
function, so this script fails loudly rather than narrating a stale story if the
implementation moves underneath it.

Two things it surfaces that the pass itself throws away:

  * **Rejected candidates.** The finders return `None` for a group that fails its
    admissibility rule, so a coverage failure leaves no trace. Step 3 re-runs the
    same finder with the coverage requirement relaxed and diffs, which recovers
    exactly the groups the rule killed -- the dashed boxes in panel (b).
  * **An optimality check.** Step 4 is greedy, not an exact maximum-weight
    independent set. On a candidate set this small the exact answer is brute
    forceable, so the script computes it and reports whether greedy matched. Do
    not caption panel (c) as an optimal solve without reading that line.

Run:
    venv/bin/python \\
        experiments/figures/merging_example_circuit/merging_pipeline_walkthrough.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
# The sibling module holds the circuit; the repo root holds `translator`. Adding
# both means this runs as a plain script, without a PYTHONPATH incantation.
for _path in (HERE, HERE.parents[2]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qiskit.converters import circuit_to_dag  # noqa: E402
from qiskit.dagcircuit import DAGOpNode  # noqa: E402

from merging_example_circuit import N_CODE, PERMUTATION, toy_circuit  # noqa: E402

WIDTH = 78


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def rule(char: str = "-", title: str = "") -> None:
    if not title:
        print(char * WIDTH)
        return
    head = f"{char * 3} {title} "
    print(head + char * max(0, WIDTH - len(head)))


def wrapped_lines(text: str, indent: str, hanging: Optional[str] = None) -> List[str]:
    """`text` hard-wrapped to the banner width, first line at `indent`."""
    continuation = indent if hanging is None else hanging
    lines: List[str] = []
    current = indent
    for word in text.split():
        if current.strip() and len(current) + 1 + len(word) > WIDTH:
            lines.append(current)
            current = continuation + word
        else:
            current += ("" if not current.strip() else " ") + word
    if current.strip():
        lines.append(current)
    return lines


def wrap(text: str, indent: str = "  ") -> None:
    for line in wrapped_lines(text, indent):
        print(line)


def wrap_entry(label: str, text: str, indent: str = "    ", gap: int = 9) -> None:
    """A label in the left margin with its wrapped body in a fixed column."""
    body_indent = indent + " " * gap
    lines = wrapped_lines(text, body_indent, body_indent)
    print(indent + f"{label:<{gap}}" + lines[0][len(body_indent):])
    for line in lines[1:]:
        print(line)


def qname(qubit, qubit_indices: Dict[object, int]) -> str:
    return f"q{qubit_indices[qubit]}"


def node_label(node: DAGOpNode, qubit_indices: Dict[object, int]) -> str:
    args = ",".join(qname(q, qubit_indices) for q in node.qargs)
    return f"{node.op.name.upper()}({args})"


def node_at(node: DAGOpNode, layer_index, qubit_indices) -> str:
    return f"{node_label(node, qubit_indices)}@L{layer_index[node]}"


def record_key(record, order_index) -> Tuple:
    """Identity of a candidate by what it fuses, so two runs can be diffed."""
    return (
        record.macro_gate.name,
        tuple(sorted(order_index.get(n, 0) for n in record.nodes)),
    )


def lanes_of(record, qubit_indices) -> str:
    return " ".join(qname(q, qubit_indices) for q in record.macro_qargs)


# ---------------------------------------------------------------------------
# [1] layer
# ---------------------------------------------------------------------------

#: Names too wide for a grid column. Display only; nothing keys off these.
ABBREVIATIONS = {"routing_swap": "rswap", "measure": "meas"}


def step_1_layer(dag, circuit, qubit_indices, explain: bool = True) -> Tuple[Dict, Dict]:
    from translator.merging.layer import layer_by_node
    from translator.merging.single_qubit import MACRO_NAMES

    layer_index, order_index = layer_by_node(dag)

    rule("-", "[1] LAYER")
    if explain:
        wrap(
            "Each instruction gets exactly one time layer: one past the latest "
            "layer of any gate it waits on (merging.layer.layer_by_node). This is "
            "the same layering the packing model uses, and it is the coordinate "
            "system the rest of the pass measures delay windows in -- not the "
            "merge test itself."
        )
        print()

    n_layers = max(layer_index.values()) + 1
    grid: Dict[Tuple[int, int], str] = {}
    for node, layer in layer_index.items():
        name = node.op.name.lower()
        label = ABBREVIATIONS.get(name, name).upper()
        for position, qubit in enumerate(node.qargs):
            # Control/target only where the roles differ: a macro spans its whole
            # patch and has no such split, so marking one lane "c" would lie.
            mark = label
            if len(node.qargs) == 2 and name not in MACRO_NAMES:
                mark += "·c" if position == 0 else "·t"
            grid[(layer, qubit_indices[qubit])] = mark

    column = max(6, max(len(mark) for mark in grid.values()) + 2)
    qubits = sorted(qubit_indices.values())
    print("        " + "".join(f"q{i}".ljust(column) for i in qubits))
    print("        " + "-" * (column * len(qubits)))
    for layer in range(n_layers):
        cells = "".join(grid.get((layer, q), ".").ljust(column) for q in qubits)
        print(f"  L{layer:<5} {cells}".rstrip())
    print()
    wrap(
        f"{len(layer_index)} instructions in {n_layers} layers. One gate, one "
        "layer -- but a group may span several, since the delay window lets a "
        "member wait. The qiskit drawer packs columns to avoid overlapping "
        "wires and does not agree with this table: when annotating the figure, "
        "these layers are the truth and the drawn columns are not."
    )
    return layer_index, order_index


# ---------------------------------------------------------------------------
# [2] occupancy  (panel a)
# ---------------------------------------------------------------------------

def step_2_occupancy(permutation, circuit, qubit_indices, k, explain: bool = True):
    from translator.merging.preprocessing import _build_patch_lanes

    patch_lanes = _build_patch_lanes(permutation, circuit.qubits)

    rule("-", "[2] OCCUPANCY  (panel a)")
    if explain:
        wrap(
            "The packer's permutation read as patch -> slot -> qubit "
            "(preprocessing._build_patch_lanes). Computes nothing; it is the "
            "lookup every coverage check consults. Slot = index within the patch, "
            "so lane i of one patch is what a transversal macro pairs with lane i "
            "of another."
        )
        print()
    for patch in sorted(patch_lanes):
        qubits = patch_lanes[patch]
        slots = "  ".join(
            f"slot {slot}={qname(q, qubit_indices)}" for slot, q in enumerate(qubits)
        )
        print(f"  patch {patch} : {slots}   [{len(qubits)}/{k} occupied]")
    if explain:
        print()
        wrap(
            "Full coverage means every *occupied* lane, not every slot. A patch "
            "holding 3 of 4 qubits yields three lanes and a whole-patch macro "
            "over those three is legal -- the idle slot carries no logical qubit, "
            "so disturbing it is free. Patches holding fewer than two qubits are "
            "dropped here: there is nothing to fuse."
        )
    return patch_lanes


# ---------------------------------------------------------------------------
# [3] propose single-qubit groups  (panel b)
# ---------------------------------------------------------------------------

def step_3_propose_single(layer_index, order_index, patch_lanes, qubit_indices,
                          settings, work_table, explain: bool = True):
    from translator.merging.gates import HHGate, XXGate, ZZGate
    from translator.merging.selection import _merge_savings_weight
    from translator.merging.single_qubit import find_single_qubit_merges_for_patch

    kinds = (
        ("h", "hh", HHGate, True),
        ("x", "xx", XXGate, False),
        ("z", "zz", ZZGate, False),
    )

    admitted: List = []
    admitted_patch: Dict[int, int] = {}
    rejected: List[Tuple[int, str, object]] = []

    for patch, qubits in sorted(patch_lanes.items()):
        for gate_name, macro_name, factory, needs_coverage in kinds:
            found = find_single_qubit_merges_for_patch(
                layer_index, order_index, qubits, settings,
                gate_name=gate_name, macro_gate_factory=factory,
                require_full_coverage=needs_coverage,
            )
            for record in found:
                admitted_patch[id(record)] = patch
            admitted.extend(found)

            if not needs_coverage:
                continue
            # Same finder, coverage requirement lifted: whatever it returns that
            # the real run did not is precisely a group the coverage rule killed.
            relaxed = find_single_qubit_merges_for_patch(
                layer_index, order_index, qubits, settings,
                gate_name=gate_name, macro_gate_factory=factory,
                require_full_coverage=False,
            )
            kept = {record_key(r, order_index) for r in found}
            rejected.extend(
                (patch, macro_name, r)
                for r in relaxed
                if record_key(r, order_index) not in kept
            )

    rule("-", "[3] PROPOSE SINGLE-QUBIT GROUPS  (panel b)")
    if explain:
        wrap(
            "Same-kind single-qubit gates on one patch's lanes. Every gate of "
            "that kind is tried as a group leader and the scan walks forward "
            "collecting one partner per uncovered lane, stopping at the first "
            "event it cannot cross. That forward reach is the delay window -- the "
            "members do not have to share a layer, only to be close enough that "
            "one can wait."
        )
        print()
        cost_note = (
            "all crossings free (zero_delay_costs=True), so the window is bounded "
            "by uncrossable events -- a measure, a second cross-patch CX, a gate "
            "touching several lanes -- rather than by the budget"
            if settings.zero_delay_costs
            else f"budget {settings.budget:g}, spent per crossed gate"
        )
        wrap(f"delay window: {cost_note}.")
        print()
        wrap("admissibility:")
        wrap_entry("hh", "whole-patch. Must cover every occupied lane. "
                         "All-or-nothing: transversal H on a subset is not a "
                         "logical operation at all.")
        wrap_entry("xx/zz", "per-lane. Any subset of >= 2 lanes is legal. A group "
                            "of m emits m + (m mod 2) gates, so an even group "
                            "cancels the anchor outright and an odd one pays for "
                            "it -- the parity is priced, not required.")
        print()

    print(f"  ADMITTED ({len(admitted)}):")
    for idx, record in enumerate(admitted):
        patch = admitted_patch[id(record)]
        span = sorted(layer_index[n] for n in record.nodes)
        size = len(record.nodes)
        lane_count = len(patch_lanes[patch])
        parity = "" if record.macro_gate.name == "hh" else (
            "  even" if size % 2 == 0 else "  ODD (anchor not cancelled)"
        )
        print(
            f"    #{idx}  {record.macro_gate.name}  patch {patch}  "
            f"[{lanes_of(record, qubit_indices)}]  {size}/{lane_count}  "
            f"L{span[0]}..L{span[-1]}  "
            f"w={_merge_savings_weight(record, work_table):g}{parity}"
        )
    if not admitted:
        print("    (none)")

    print()
    print(f"  REJECTED BY THE COVERAGE RULE ({len(rejected)}):")
    for patch, macro_name, record in sorted(
        rejected, key=lambda item: order_index.get(item[2].first, 0)
    ):
        span = sorted(layer_index[n] for n in record.nodes)
        lane_count = len(patch_lanes[patch])
        missing = lane_count - len(record.nodes)
        print(
            f"    {macro_name}  patch {patch}  [{lanes_of(record, qubit_indices)}]  "
            f"{len(record.nodes)}/{lane_count}  L{span[0]}..L{span[-1]}  "
            f"-- {missing} lane{'s' if missing != 1 else ''} short"
        )
    if not rejected:
        print("    (none)")
    if explain:
        print()
        wrap(
            "Overlapping rivals are all proposed; nothing is resolved yet. Every "
            "H in the circuit leads its own attempt, which is why one coverage "
            "failure shows up as several rejected rows -- each leader reaches a "
            "different distance before an uncrossable gate stops it. The rejected "
            "rows are the panel-(b) contrast: same gate kind, lanes short, so the "
            "patch pays full targeted price for every gate in them."
        )
    return admitted


# ---------------------------------------------------------------------------
# [4] select a non-conflicting subset  (panel c)
# ---------------------------------------------------------------------------

def classify_conflicts(candidates, layer_index, order_index, qubit_indices, dag):
    """Conflict edges labelled by which detector first found them.

    `compute_merge_conflicts` runs four detectors into one shared dict and each
    skips pairs already recorded, so replaying them in that same order and
    diffing after each attributes every edge to its true introducer.
    """
    from translator.merging.conflicts import (
        _add_cross_gate_conflicts,
        _add_first_in_window_conflicts,
        _add_shared_endpoint_conflicts,
        _bentley_ottmann_conflicts,
        _build_segments,
    )

    def edges(mapping: Dict[int, Set[int]]) -> Set[Tuple[int, int]]:
        return {
            (min(a, b), max(a, b))
            for a, others in mapping.items()
            for b in others
        }

    conflicts = _bentley_ottmann_conflicts(
        _build_segments(candidates, layer_index, qubit_indices)
    )
    labelled: Dict[Tuple[int, int], str] = {e: "geometric" for e in edges(conflicts)}

    for label, detector in (
        ("shared-gate", lambda: _add_shared_endpoint_conflicts(candidates, conflicts)),
        ("cross-gate", lambda: _add_cross_gate_conflicts(
            candidates, layer_index, order_index, dag, conflicts)),
        ("reorder-window", lambda: _add_first_in_window_conflicts(
            candidates, layer_index, order_index, conflicts)),
    ):
        before = edges(conflicts)
        detector()
        for edge in edges(conflicts) - before:
            labelled[edge] = label
    return labelled


def best_independent_set(indices, weights, edges) -> Tuple[float, Tuple[int, ...]]:
    """Exact maximum-weight independent set, for checking the greedy pick."""
    best_weight, best_set = 0.0, ()
    for size in range(len(indices), -1, -1):
        for subset in combinations(indices, size):
            chosen = set(subset)
            if any(a in chosen and b in chosen for a, b in edges):
                continue
            total = sum(weights[i] for i in subset)
            if total > best_weight:
                best_weight, best_set = total, subset
    return best_weight, best_set


def step_4_select(candidates, layer_index, order_index, qubit_indices, dag,
                  work_table, explain: bool = True):
    from translator.merging.selection import (
        _is_single_qubit_merge,
        _merge_savings_weight,
        select_best_group,
    )

    rule("-", "[4] SELECT A NON-CONFLICTING SUBSET  (panel c)")
    if explain:
        wrap(
            "Candidates are nodes; an edge means they cannot both commit. Four "
            "detectors contribute edges (conflicts.compute_merge_conflicts): two "
            "candidates sharing a gate, two whose (layer x qubit) segments cross, "
            "and two cases where one candidate's blocking window would reorder a "
            "gate past the other's macro."
        )
        print()

    single = [i for i, c in enumerate(candidates) if _is_single_qubit_merge(c)]
    weights = {i: _merge_savings_weight(candidates[i], work_table) for i in single}
    labelled = classify_conflicts(candidates, layer_index, order_index, qubit_indices, dag)
    edges = {e: label for e, label in labelled.items()
             if e[0] in weights and e[1] in weights}

    print(f"  nodes ({len(single)}):")
    for idx in single:
        degree = sum(1 for a, b in edges if idx in (a, b))
        print(
            f"    #{idx}  {candidates[idx].macro_gate.name}"
            f"[{lanes_of(candidates[idx], qubit_indices)}]"
            f"  w={weights[idx]:g}  degree={degree}"
        )
    print()
    print(f"  edges ({len(edges)}):")
    for (a, b), label in sorted(edges.items()):
        print(f"    #{a} -- #{b}   ({label})")
    if not edges:
        print("    (none -- every candidate can commit together)")
    print()

    chosen = list(select_best_group(
        candidates, layer_index, order_index, qubit_indices,
        dag=dag, work_table=work_table,
    ))
    chosen_ids = [i for i in single if any(candidates[i] is c for c in chosen)]
    total = sum(weights[i] for i in chosen_ids)
    print(f"  chosen: {', '.join('#' + str(i) for i in chosen_ids) or '(none)'}"
          f"   total w={total:g}")
    for idx in single:
        if idx in chosen_ids:
            continue
        rivals = [b if a == idx else a for a, b in edges if idx in (a, b)]
        beat_by = ", ".join(f"#{r} (w={weights[r]:g})" for r in rivals if r in chosen_ids)
        print(f"  dropped #{idx} (w={weights[idx]:g})"
              + (f" -- conflicts with {beat_by}" if beat_by else ""))
    print()

    if len(single) <= 16:
        exact, exact_set = best_independent_set(single, weights, list(edges))
        verdict = "MATCHED the exact optimum" if abs(exact - total) < 1e-9 else (
            f"MISSED it -- exact pick is "
            f"{', '.join('#' + str(i) for i in exact_set)}"
        )
        wrap(
            f"optimality check: greedy total w={total:g}, exact maximum-weight "
            f"independent set w={exact:g} -- greedy {verdict}."
        )
    else:
        wrap(f"optimality check skipped: {len(single)} candidates is too many to "
             "brute force.")
    if explain:
        wrap(
            "Selection ranks on predicted savings, then conflict count, then "
            "position, and grows groups greedily -- it does not solve the MWIS. "
            "The picture is right for panel (c); the caption should not say "
            "'optimal'."
        )
    return chosen


# ---------------------------------------------------------------------------
# [5] propose transversal two-qubit groups  (panel d inset)
# ---------------------------------------------------------------------------

def transversal_alignment(record, patch_lanes, qubit_indices):
    """(control slot, control qubit, target slot, target qubit) per fused gate."""
    meta = getattr(record.macro_gate, "metadata", {}) or {}
    control_patch = meta.get("control_patch")
    target_patch = meta.get("target_patch")
    slot_of = {
        qubit: slot
        for patch, qubits in patch_lanes.items()
        for slot, qubit in enumerate(qubits)
    }
    patch_of = {q: patch for patch, qubits in patch_lanes.items() for q in qubits}

    rows = []
    for node in record.nodes:
        first, second = node.qargs[0], node.qargs[1]
        if patch_of.get(first) == control_patch:
            control, target = first, second
        else:
            control, target = second, first
        rows.append((slot_of[control], control, slot_of[target], target))
    rows.sort()
    return control_patch, target_patch, rows


def step_5_propose_transversal(
    layer_index, order_index, patch_lanes, qubit_indices, consumed,
    allow_swap_merges, n_code, explain: bool = True,
):
    from iceberg.costs import CROSS_PATCH_COST, carriage_cost, work_table
    from translator.merging.conflicts import DEFAULT_MERGE_WORK_WEIGHTS
    from translator.merging.transversal import (
        _MACRO_COST_KEYS,
        _macro_gain,
        find_transversal_merges,
    )

    records = find_transversal_merges(
        layer_index, order_index, patch_lanes, qubit_indices,
        consumed, allow_swap_merges, n_code=n_code,
    )

    rule("-", "[5] PROPOSE TRANSVERSAL TWO-QUBIT GROUPS  (panel d inset)")
    if explain:
        wrap(
            "k same-kind two-qubit gates whose controls fill one patch and "
            "targets fill the other. Picture the macro as a rigid zipper between "
            "two patch columns: it joins slot i to slot i, all k at once, in one "
            "layer. Draw an arrow from each program gate's control lane to its "
            "target lane -- if every arrow runs horizontally the zipper fits, and "
            "if any cross it does not."
        )
        print()
        wrap(
            "A crossing is fixed by physically permuting the target patch: one "
            "routing swap per transposition, at most k-1, emitted before the "
            "macro and undone after. A routing swap is an atom exchange, so "
            "realignment is not free and its price is subtracted from the group's "
            "gain before it may commit."
        )
        print()
    print(f"  swap merges: {'ON' if allow_swap_merges else 'OFF'}")
    print(f"  committed groups: {len(records)}")

    w1, w2 = DEFAULT_MERGE_WORK_WEIGHTS
    table = work_table(n_code)

    for record in records:
        print()
        gate_name = record.nodes[0].op.name.lower()
        control_patch, target_patch, rows = transversal_alignment(
            record, patch_lanes, qubit_indices
        )
        lane_count = len(record.nodes)
        transpositions = len(record.pre_ops)

        print(f"  {lane_count} x {gate_name.upper()}   "
              f"patch {control_patch} (controls) -> patch {target_patch} (targets)")
        print("  slot map:")
        for control_slot, control, target_slot, target in rows:
            aligned = control_slot == target_slot
            arrow = "---->" if aligned else "--\\/-"
            print(
                f"    c{control_slot} ({qname(control, qubit_indices)}) {arrow} "
                f"t{target_slot} ({qname(target, qubit_indices)})   "
                f"{'aligned' if aligned else 'CROSSED'}"
            )

        print(f"  coverage: controls fill {lane_count}/"
              f"{len(patch_lanes[control_patch])} of patch {control_patch}, "
              f"targets fill {lane_count}/{len(patch_lanes[target_patch])} of "
              f"patch {target_patch}")
        if transpositions == 0:
            print("  alignment: identity -- 0 routing swaps")
        else:
            print(f"  alignment: not the identity -- {transpositions} routing swap"
                  f"{'s' if transpositions != 1 else ''} "
                  f"(bound is k-1 = {len(patch_lanes[target_patch]) - 1})")
            for planned in record.pre_ops:
                args = ",".join(qname(q, qubit_indices) for q in planned.qargs)
                print(f"      {planned.gate.name}({args})  before the macro, "
                      f"undone after")

        macro_key, single_key = _MACRO_COST_KEYS[gate_name]
        single = CROSS_PATCH_COST.get(single_key, table[single_key])
        macro = table[macro_key]
        singles_total = lane_count * (single[0] * w1 + single[1] * w2)
        macro_total = macro[0] * w1 + macro[1] * w2
        realign = 2 * transpositions * carriage_cost("SWAP", (w1, w2))
        gain = singles_total - macro_total - realign
        # The parts must recombine into the number the gate actually uses.
        assert abs(gain - _macro_gain(gate_name, lane_count, n_code, transpositions)) < 1e-9

        swap_price = carriage_cost("SWAP", (w1, w2))
        print(f"  price (transversal._macro_gain), weights 1q={w1:g} 2q={w2:g}:")
        for sign, label, value in (
            (" ", f"{lane_count} x {single_key} cross-patch {single}", singles_total),
            ("-", f"1 x {macro_key} macro at n={n_code} {macro}", -macro_total),
            ("-", f"realign: 2 x {transpositions} swap x {swap_price:g}", -realign),
        ):
            print(f"    {sign} {label:<40}{value:>10.2f}")
        print(f"      {'-' * 50}")
        print(f"      {'gain':<40}{gain:>10.2f}   "
              f"{'> 0  COMMIT' if gain > 0 else '<= 0  REJECT'}")

    if any(record.pre_ops for record in records):
        probe = find_transversal_merges(
            layer_index, order_index, patch_lanes, qubit_indices,
            set(consumed), False, n_code=n_code,
        )
        print()
        wrap(
            f"with allow_transversal_swap_merges=False the misaligned group is "
            f"rejected outright rather than priced: "
            f"{len(probe)} group(s) survive instead of {len(records)}."
        )
    return records


# ---------------------------------------------------------------------------
# [6] drop invalidated picks
# ---------------------------------------------------------------------------

def step_6_drop(selected, transversal, layer_index, order_index, qubit_indices,
                explain: bool = True):
    from translator.merging.conflicts import (
        filter_single_qubit_conflicting_with_transversal,
    )

    def pos(node) -> Tuple[int, int]:
        return (layer_index.get(node, 0), order_index.get(node, 0))

    rule("-", "[6] DROP INVALIDATED PICKS")
    if explain:
        wrap(
            "The aftershock of step 5. Its routing swaps physically moved the "
            "target patch's atoms and its macro delayed whatever it displaced, so "
            "a step-4 winner may now find its gates on the wrong side of the "
            "macro. Each winner is re-tested against every committed transversal "
            "group and dropped when all three conditions hold:"
        )
        print()
        wrap_entry("(a)", "its first gate falls inside the transversal's window "
                          "[first, second)", "      ", 5)
        wrap_entry("(b)", "that gate touches a qubit the transversal macro "
                          "operates on", "      ", 5)
        wrap_entry("(c)", "its own macro would fire after the transversal's -- so "
                          "the gate gets silently moved from before the macro to "
                          "after it", "      ", 5)
        print()

    predicted = []
    for merge_a in selected:
        verdicts = []
        conflict = False
        for merge_b in transversal:
            in_window = pos(merge_b.first) <= pos(merge_a.first) < pos(merge_b.second)
            touches = bool(set(merge_a.first.qargs) & set(merge_b.macro_qargs))
            reorders = pos(merge_a.second) > pos(merge_b.second)
            hit = in_window and touches and reorders
            conflict = conflict or hit
            verdicts.append(
                f"{merge_b.macro_gate.name}: "
                f"(a) {'yes' if in_window else 'no'} "
                f"(b) {'yes' if touches else 'no'} "
                f"(c) {'yes' if reorders else 'no'}"
            )
        print(f"  {merge_a.macro_gate.name}[{lanes_of(merge_a, qubit_indices)}]  "
              f"vs {' | '.join(verdicts) if verdicts else '(no transversal groups)'}"
              f"  -> {'DROPPED' if conflict else 'kept'}")
        if conflict:
            predicted.append(merge_a)

    survivors = filter_single_qubit_conflicting_with_transversal(
        selected, transversal, layer_index, order_index
    )
    dropped = [r for r in selected if r not in survivors]
    # The per-condition table above is a reading of the real predicate; if the
    # two ever disagree, the table is the thing that has gone stale.
    assert [id(r) for r in predicted] == [id(r) for r in dropped], (
        "the displayed conditions no longer match "
        "filter_single_qubit_conflicting_with_transversal"
    )

    print()
    print(f"  dropped: {len(dropped)}   surviving single-qubit groups: {len(survivors)}")
    if explain:
        wrap(
            "Coverage is checked again at emission for the same reason: a swap "
            "may have moved a qubit out of the lane a group was proposed for, so "
            "an hh that covered every occupied lane at proposal time might no "
            "longer."
        )
    return survivors


# ---------------------------------------------------------------------------
# [7] rewrite  (panel d)
# ---------------------------------------------------------------------------

def step_7_rewrite(circuit, dag, records, layer_index, order_index, qubit_indices,
                   work_table, explain: bool = True):
    from translator.merging.conflicts import DEFAULT_MERGE_WORK_WEIGHTS
    from translator.merging.preprocessing import _merge_savings_for_records
    from translator.merging.rewrite import build_rewritten_circuit

    rule("-", "[7] REWRITE  (panel d)")
    if explain:
        wrap(
            "Each surviving group becomes one macro. Of the instructions it "
            "absorbs, all but the last are deleted and the macro is emitted where "
            "the last one stood -- which is why every absorbed gate has to be safe "
            "to delay that far. Gates caught by the block in between are held and "
            "re-emitted after the macro."
        )
        print()
    for record in records:
        absorbed = " ".join(node_at(n, layer_index, qubit_indices) for n in record.dropped)
        print(f"  {record.macro_gate.name}[{lanes_of(record, qubit_indices)}]")
        print(f"      absorbs {absorbed}")
        print(f"      emits at {node_at(record.second, layer_index, qubit_indices)}")
        for planned, when in (
            *((op, "pre") for op in record.pre_ops),
            *((op, "post") for op in record.post_ops),
        ):
            args = ",".join(qname(q, qubit_indices) for q in planned.qargs)
            print(f"      {when}: {planned.gate.name}({args})")

    savings_1q, savings_2q = _merge_savings_for_records(records, work_table)
    weighted = (savings_1q * DEFAULT_MERGE_WORK_WEIGHTS[0]
                + savings_2q * DEFAULT_MERGE_WORK_WEIGHTS[1])
    rewritten = build_rewritten_circuit(circuit, dag, records, layer_index, order_index)

    print()
    print(f"  {len(circuit.data)} -> {len(rewritten.data)} instructions")
    print(f"  pass savings: {savings_1q:g} 1q, {savings_2q:g} 2q physical gates "
          f"(weighted {weighted:g})")
    print()
    print(rewritten.draw(output="text"))
    return rewritten, savings_1q, savings_2q


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def walkthrough(
    circuit=None,
    permutation: Sequence[Tuple[int, int]] = PERMUTATION,
    n_code: int = N_CODE,
    passes: int = 2,
    allow_swap_merges: bool = True,
) -> None:
    from translator.merging.conflicts import merge_work_table
    from translator.merging.settings import DEFAULT_SETTINGS

    circuit = toy_circuit() if circuit is None else circuit
    source = circuit  # kept for the end-of-run cross-check; `circuit` is rewritten
    work_table = merge_work_table(n_code)
    settings = DEFAULT_SETTINGS
    k = n_code - 2
    original_count = len(circuit.data)
    total_1q = total_2q = 0.0

    rule("=")
    print(f"  merging pipeline walkthrough -- {original_count} instructions, "
          f"n_code={n_code} (k={k})")
    rule("=")
    print(circuit.draw(output="text"))

    for pass_number in range(1, max(1, passes) + 1):
        print()
        rule("#", f"PASS {pass_number}")
        print()

        # The prose is identical every pass; print it once so a multi-pass run
        # stays readable and the second pass shows only what changed.
        explain = pass_number == 1
        dag = circuit_to_dag(circuit)
        qubit_indices = {q: circuit.find_bit(q).index for q in circuit.qubits}

        layer_index, order_index = step_1_layer(dag, circuit, qubit_indices, explain)
        print()
        patch_lanes = step_2_occupancy(
            permutation, circuit, qubit_indices, k, explain
        )
        if not patch_lanes:
            print("\n  no patch holds two qubits; nothing can be fused")
            break
        print()
        candidates = step_3_propose_single(
            layer_index, order_index, patch_lanes, qubit_indices, settings,
            work_table, explain,
        )
        print()
        selected = step_4_select(
            candidates, layer_index, order_index, qubit_indices, dag, work_table,
            explain,
        )

        consumed = {r.first for r in selected} | {r.second for r in selected}
        print()
        transversal = step_5_propose_transversal(
            layer_index, order_index, patch_lanes, qubit_indices, consumed,
            allow_swap_merges, n_code, explain,
        )
        print()
        survivors = step_6_drop(
            selected, transversal, layer_index, order_index, qubit_indices, explain
        )

        records = list(survivors) + list(transversal)
        if not records:
            print()
            rule("-", "[7] REWRITE  (panel d)")
            print("  nothing survived; the pass ends and the circuit is unchanged")
            break

        print()
        circuit, savings_1q, savings_2q = step_7_rewrite(
            circuit, dag, records, layer_index, order_index, qubit_indices,
            work_table, explain,
        )
        total_1q += savings_1q
        total_2q += savings_2q

    print()
    rule("=")
    print("  final macro'd circuit")
    rule("=")
    print(circuit.draw(output="text"))
    print()
    print(f"  instructions: {original_count} -> {len(circuit.data)}")
    print(f"  measured savings: {total_1q:g} 1q, {total_2q:g} 2q physical gates")

    # This function drives the stages by hand to narrate them, so it could in
    # principle wire them up differently from the pipeline it claims to explain.
    # Run the real entry point on the same inputs and refuse to agree with
    # itself unless the two land in the same place.
    from translator.merging.preprocessing import (
        preprocess_program_for_merging_with_stats,
    )

    expected_circuit, expected_1q, expected_2q, _ = (
        preprocess_program_for_merging_with_stats(
            source, permutation,
            allow_transversal_swap_merges=allow_swap_merges,
            num_merge_passes=passes, n_code=n_code,
        )
    )
    actual = (len(circuit.data), total_1q, total_2q)
    expected = (len(expected_circuit.data), expected_1q, expected_2q)
    if actual != expected:
        raise AssertionError(
            f"the walkthrough diverged from preprocess_program_for_merging: "
            f"narrated {actual}, pipeline produced {expected}"
        )
    print("  agrees with preprocess_program_for_merging on the same inputs")
    rule("=")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--passes", type=int, default=2,
                        help="merge passes to run (default 2, the pipeline value)")
    parser.add_argument("--n-code", type=int, default=N_CODE,
                        help=f"code size; k = n_code - 2 (default {N_CODE})")
    parser.add_argument("--no-swap-merges", action="store_true",
                        help="reject misaligned transversal groups instead of "
                             "pricing their routing swaps")
    args = parser.parse_args()

    walkthrough(
        n_code=args.n_code,
        passes=args.passes,
        allow_swap_merges=not args.no_swap_merges,
    )


if __name__ == "__main__":
    main()
