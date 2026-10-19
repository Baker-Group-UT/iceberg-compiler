"""Narrate every step of the packing pass, the way the merging walkthrough does.

`plan_packing` returns a placement and a one-line summary. This prints the
reasoning inside each stage -- the view that gets profiled, the signed graph, the
seed partition, every accepted descent move, the restart comparison, the lane
solve, and the two comparisons that can overrule all of it.

The stages, as `packing/plan.py` lists them:

    [1] profile    score the circuit the translator will actually place
    [2] objective  build PackingCost: a function of the PARTITION, not the placement
    [3] seed       multilevel-partition the signed graph, for global structure
    [4] descend    sweep qubits move/spread/swap on the true objective
    [5] restarts   descend from naive and shuffled seeds too, keep the best
    [6] lanes      assign slots exactly, once, on the final partition
    [7] commute?   do [1]-[6] both ways and keep the cheaper placement
    [8] verify     translate against the naive packing and keep the cheaper

Steps 1-6 run once per commutation arm, so they print twice -- the same shape as
the merging walkthrough's two passes. Prose is printed for the first arm only.

Everything printed is read out of `packing`, never restated. The price
decomposition in step 2 is recombined and asserted against `PackingCost.price`,
and the whole run is asserted against `plan_packing` on the same inputs, so this
fails loudly rather than narrating a pipeline that no longer exists.

Subject circuits:

    --circuit toy        (default) the 13-instruction merging example, so the two
                         walkthroughs describe the same circuit. Its descent makes
                         zero moves -- the seed is already optimal at 8 qubits --
                         which is honest but shows step 4 doing nothing.
    --circuit qpe_n9     a QASMBench proxy where the descent actually bites: 3
                         patches, 4 accepted moves. Any benchmark name works.
    --circuit adder_n10  where `verify` rejects the packed placement and returns
                         the naive one, which is step 8 earning its keep.

Run:
    venv/bin/python \\
        experiments/figures/merging_example_circuit/packing_pipeline_walkthrough.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parents[2], HERE.parents[1] / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qiskit import QuantumCircuit  # noqa: E402

from merging_example_circuit import N_CODE, toy_circuit  # noqa: E402
# The formatting helpers live with the merging walkthrough; sharing them keeps the
# two scripts looking like one pair rather than two conventions.
from merging_pipeline_walkthrough import rule, wrap, wrap_entry  # noqa: E402

QASM_ROOT = HERE.parents[2] / "benchmarks" / "qasmbench"


# ---------------------------------------------------------------------------
# subject circuit
# ---------------------------------------------------------------------------

def load_circuit(name: str) -> Tuple[str, QuantumCircuit]:
    """The merging toy, or a QASMBench proxy by benchmark name."""
    if name == "toy":
        return "merging example (toy)", toy_circuit()

    from benchmark_circuits import create_proxy_from_qasm

    hits = sorted(QASM_ROOT.rglob(f"{name}/{name}.qasm"))
    if not hits:
        raise SystemExit(f"no benchmark named {name!r} under {QASM_ROOT}")
    circuit = create_proxy_from_qasm(hits[0])
    if circuit is None:
        raise SystemExit(f"{name} would not reduce to a Clifford proxy")
    return f"{name} (Clifford proxy)", circuit


def gate_histogram(circuit: QuantumCircuit) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for instruction in circuit.data:
        name = instruction.operation.name.lower()
        counts[name] = counts.get(name, 0) + 1
    return counts


def fmt_groups(groups: Sequence, limit: int = 8) -> str:
    shown = [f"{{{','.join('q' + str(q) for q in sorted(g))}}}" for g in groups if g]
    if len(shown) > limit:
        return " ".join(shown[:limit]) + f" ... (+{len(shown) - limit} more)"
    return " ".join(shown) or "(none)"


# ---------------------------------------------------------------------------
# [1] profile
# ---------------------------------------------------------------------------

def step_1_profile(program: QuantumCircuit, commute: bool, explain: bool = True):
    from packing.profiles import extract_profiles
    from packing.tables import translated_view

    view = translated_view(program, enable_h_commutation=commute)
    profiles = extract_profiles(view)

    rule("-", f"[1] PROFILE  (h_commutation={commute})")
    if explain:
        wrap(
            "The objective is keyed by gate kind, so it has to score the circuit "
            "the translator will actually place -- not the program as written. "
            "`translated_view` runs the translator's own program-level passes "
            "first, which rewrite kinds: a CX whose control carries a commuted "
            "Hadamard becomes an XCX, and XCX costs (5,3) intra-patch where CX "
            "costs (0,4). Scoring the raw program instead is worth 13-49% error."
        )
        print()

    before, after = gate_histogram(program), gate_histogram(view)
    for label, circuit, counts in (
        ("program", program, before),
        ("view   ", view, after),
    ):
        rendered = "  ".join(f"{name}x{count}" for name, count in sorted(counts.items()))
        print(f"  {label}: {len(circuit.data):>4} instructions   {rendered}")
    changed = sorted(set(before) | set(after))
    diffs = [
        f"{name} {before.get(name, 0)}->{after.get(name, 0)}"
        for name in changed
        if before.get(name, 0) != after.get(name, 0)
    ]
    print(f"  kinds rewritten: {', '.join(diffs) if diffs else 'none'}")

    print()
    print(f"  per-qubit profiles ({len(profiles)}):")
    for index in sorted(profiles):
        profile = profiles[index]
        print(
            f"    q{index:<3} cast={sorted(profile.cast())}  "
            f"1q h/x/z={profile.count('h')}/{profile.count('x')}/{profile.count('z')}  "
            f"2q={profile.two_qubit_count()}  depth={profile.depth}"
        )
    if explain:
        print()
        wrap(
            "`cast` is who this qubit interacts with -- the only thing that makes "
            "co-location worth anything -- and the layer numbers behind these "
            "counts are what the coincidence windows for HH and Pauli fusion are "
            "measured in."
        )
    return view, profiles


# ---------------------------------------------------------------------------
# [2] objective
# ---------------------------------------------------------------------------

def price_breakdown(cost, groups) -> Dict[str, float]:
    """`PackingCost.price` split into the terms it sums, checked against it."""
    from packing.cost import PARALLEL_SHARE
    from packing.tables import mean_patch_hops

    live = [g for g in groups if g]
    total = cost.unary(frozenset())
    for group in live:
        total = total + cost.unary(group)
    for index, left in enumerate(live):
        for right in live[index + 1:]:
            total = total + cost.binary(left, right)

    patches = len(live)
    busiest = cost.busiest_patch_depth(live)
    hops = cost.movement_hops(live)
    depth = (1.0 - PARALLEL_SHARE) * total.depth + PARALLEL_SHARE * busiest
    # `price` groups the next four into one `work` term; they are split here because
    # the split is the readable part. The assert below is what keeps them honest.
    moved = max(0.0, total.crossings - total.crossing_savings)
    travelled = hops if hops is not None else moved * mean_patch_hops(patches, cost.n_code + 2)
    movement = cost.movement_cost * max(
        0.0, moved + cost.distance_scale * (travelled - moved)
    )
    idle = cost.idle_cost * depth * patches * (cost.n_code + 2)

    parts = {
        "solo": cost.solo,
        "per-patch encode+decode": patches * cost.per_patch,
        "1q gates": total.one_qubit,
        "2q gates": total.two_qubit,
        "macro savings": -total.savings,
        f"depth x {cost.depth_weight:g}": cost.depth_weight * depth,
        "movement": movement,
        "idle": idle,
    }
    priced = cost.price(patches, total, busiest, hops)
    assert abs(sum(parts.values()) - priced) < 1e-6, (
        f"price decomposition drifted: parts {sum(parts.values())} vs price {priced}"
    )
    parts["_total"] = priced
    parts["_depth_estimate"] = depth
    parts["_busiest"] = busiest
    parts["_crossings"] = total.crossings
    parts["_crossing_savings"] = total.crossing_savings
    parts["_hops"] = travelled
    return parts


def step_2_objective(program, profiles, n_code, noise, depth_weight, spread,
                     budget, explain: bool = True):
    from iceberg.costs import DEFAULT_WORK_WEIGHTS
    from iceberg.utils import logical_capacity
    from packing.cost import PARALLEL_SHARE, PackingCost

    cost = PackingCost(
        profiles=profiles,
        n_code=n_code,
        num_qubits=program.num_qubits,
        depth_weight=depth_weight,
        movement_cost=noise.movement_cost,
        idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale,
    )
    k = logical_capacity(n_code)
    tightest = max(1, -(-program.num_qubits // k))
    slots = tightest + max(0, spread)
    ceiling = budget.patch_ceiling(n_code)
    if ceiling is not None:
        slots = max(tightest, min(slots, ceiling))

    rule("-", "[2] OBJECTIVE")
    if explain:
        wrap(
            "The state is a *partition* of the logical qubits into patches, not a "
            "placement. Slot order enters no term -- a transversal macro fires on "
            "a lane bijection, and realignment is priced in expectation over "
            "alignments -- so every slot permutation of a group scores "
            "identically and the search explores a space k! times smaller."
        )
        print()
        # Printed verbatim, not wrapped: the columns are the point.
        for line in (
            "cost(partition) =  solo                                  constant",
            "                +  P x (ENCODE + DECODE)                 per live patch",
            "                +  sum over patches      unary(A)        intra, less HH/Pauli",
            "                +  sum over patch pairs  binary(A, B)    cross, less macros",
            "                +  movement(crossings, P) + idle(depth, P)",
        ):
            print(f"  {line}")
        print()

    w1, w2 = DEFAULT_WORK_WEIGHTS
    print(f"  n_code={n_code}  k={k}  qubits={program.num_qubits}")
    print(f"  weights: 1q={w1:g}  2q={w2:g}  depth={depth_weight:g} per layer  "
          f"parallel_share={PARALLEL_SHARE:g}")
    print(f"  noise: movement={noise.movement_cost:g}  idle={noise.idle_cost:g}  "
          f"distance_scale={noise.distance_scale:g}")
    print(f"  patch slots: {tightest} tightest + {max(0, spread)} spread = {slots}"
          + (f" (capped at {ceiling} by budget)" if ceiling is not None else ""))
    if explain:
        print()
        wrap(
            "Empty slots are free -- only live patches are charged -- so `spread` "
            "bounds how far the search may spread rather than what it costs. At "
            "movement=0 an extra patch costs ENCODE+DECODE against the handful of "
            "units a crossing saves, so zero spread is right for gate counting "
            "and only a movement-pricing profile makes spreading pay."
        )
    return cost, slots


# ---------------------------------------------------------------------------
# [3] seed
# ---------------------------------------------------------------------------

def step_3_seed(cost, profiles, slots, noise, partitioner, explain: bool = True):
    from packing.graph import build_graph
    from packing.partition import partition as feasible_partition
    from packing.seed import PARTITIONERS, graph_seed

    graph = build_graph(profiles, cost.n_code, noise=noise, num_qubits=cost.num_qubits)
    options = PARTITIONERS[partitioner]
    result = feasible_partition(graph, cost.n_code, slots, **options)
    packing = graph_seed(cost, profiles, slots, noise=noise, partitioner=partitioner)

    rule("-", "[3] SEED")
    if explain:
        wrap(
            "The interaction graph is a bad description of cost and a much better "
            "search guide, which is not a contradiction. Cut weight is a pairwise "
            "relaxation defined on every pair at once, so minimising it moves the "
            "whole partition coherently. The true objective is conjunctive -- a "
            "transversal macro needs every lane of two patches, HH every lane of "
            "one -- so its landscape is plateaus and deep local optima that a "
            "move-at-a-time descent cannot cross. The graph seeds, and nothing "
            "downstream consults it again."
        )
        print()

    print(f"  signed interaction graph: {graph.summary()}")
    ranked = sorted(graph.weights.items(), key=lambda item: -item[1])
    for label, entries in (
        ("strongest attracting", ranked[:4]),
        ("strongest repelling", [e for e in ranked[::-1] if e[1] < 0][:4]),
    ):
        rendered = "  ".join(
            f"q{a}-q{b}:{w:+.0f}" for (a, b), w in entries
        ) or "(none)"
        print(f"    {label}: {rendered}")

    print()
    print(f"  partitioner {partitioner!r} = {options}")
    print(f"  multilevel result: {result.summary()}")
    print(f"    groups: {fmt_groups(result.groups)}")
    print()
    print(f"  seed partition (split to k, padded to {slots} slots): "
          f"{fmt_groups(packing.groups)}")
    print(f"  seed cost: {packing.value():.0f}")
    if explain:
        print()
        wrap(
            "Coarsen, solve the small graph, uncoarsen refining at each level. "
            "The FM passes inside that refinement optimise the cut, and the "
            "descent below re-finds their gains on the true objective -- so they "
            "are nearly redundant and kept for the one circuit where they are "
            "not. Anything over capacity is split and any qubit the partitioner "
            "dropped is reinserted, because a partition over k is not a legal "
            "placement."
        )
    return packing


# ---------------------------------------------------------------------------
# [4] descend
# ---------------------------------------------------------------------------

def step_4_descend(packing, max_rounds, explain: bool = True):
    from packing.search import _candidates, descend

    neighbourhood = sum(1 for _ in _candidates(packing))
    result = descend(packing, max_rounds=max_rounds)

    rule("-", "[4] DESCEND")
    if explain:
        wrap(
            "Sweep the qubits, moving each to its best home, until a sweep changes "
            "nothing. Three move shapes, all changes of membership -- there is "
            "deliberately no lane move, because the objective cannot read lanes:"
        )
        print()
        wrap_entry("move", "q joins a patch that has room", "      ", 9)
        wrap_entry("spread", "q leaves for an empty slot, raising the patch count",
                   "      ", 9)
        wrap_entry("swap", "q and r exchange patches -- the only way to move anything "
                           "once every patch is full, which is exactly what the "
                           "tightest packing is", "      ", 9)
        print()
        wrap(
            "Per-qubit best-improvement, not one globally-steepest move per round: "
            "a steepest round costs a full neighbourhood scan and buys a single "
            "move. A move changes one or two patches, so only their unary terms "
            "and the binary terms incident to them move; `_delta` computes exactly "
            "that set, which is what makes thousands of candidates per sweep "
            "affordable."
        )
        print()

    print(f"  neighbourhood at the seed: {neighbourhood} candidate moves")
    print(f"  accepted ({result.moves}):")
    for index, (kind, value) in enumerate(result.accepted, start=1):
        print(f"    {index:>3}. {kind:<7} -> cost {value:.0f}")
    if not result.accepted:
        print("    (none -- the seed is already a local optimum)")
    print()
    print(f"  {result.summary()}")
    print(f"  partition: {fmt_groups(packing.groups)}")
    return result


# ---------------------------------------------------------------------------
# [5] restarts
# ---------------------------------------------------------------------------

def step_5_restarts(cost, packing, result, slots, max_rounds, restarts, num_qubits,
                    explain: bool = True):
    from packing.plan import SMALL_ENOUGH_TO_RESTART
    from packing.search import descend, naive_seed, shuffled_seed

    if restarts is None:
        restarts = 4 if num_qubits <= SMALL_ENOUGH_TO_RESTART else 0
    rng = random.Random(0)
    alternatives = [("naive", naive_seed(cost, slots))]
    alternatives += [
        (f"shuffled {i + 1}", shuffled_seed(cost, slots, rng))
        for i in range(max(0, restarts))
    ]

    rule("-", "[5] RESTARTS")
    if explain:
        wrap(
            "The naive seed is always tried, at any width. The graph seed "
            "optimises cut weight, which tracks the objective well when the "
            "objective is mostly gate counts and badly when a profile prices "
            "movement -- a 1D chain is a partition naive already nearly solves, "
            "and the descent from the graph seed can land above it and not climb "
            "out. Seeding from naive costs one descent and makes 'no worse than "
            "naive' hold in the model rather than only after step 8 re-measures "
            "it."
        )
        print()
        wrap(
            f"The shuffled restarts are size-gated at {SMALL_ENOUGH_TO_RESTART} "
            "qubits: on small circuits the basins are few, so which one the seed "
            "lands in decides the answer; on large ones they changed nothing and "
            "cost 2-3x."
        )
        print()

    incumbent = packing.value()
    print(f"  restarts: {restarts} shuffled "
          f"({'circuit is small enough' if num_qubits <= SMALL_ENOUGH_TO_RESTART else 'size-gated off'}"
          f", {num_qubits} qubits)")
    print(f"  graph seed after descent: {incumbent:.0f}")
    for label, candidate in alternatives:
        seeded = candidate.value()
        descend(candidate, max_rounds=max_rounds)
        landed = candidate.value()
        better = landed < packing.value() - 1e-9
        if better:
            packing = candidate
            result.packing = candidate
        print(f"    {label:<12} seed {seeded:>10.0f} -> descended {landed:>10.0f}   "
              f"{'ADOPTED' if better else 'kept incumbent'}")
    print()
    print(f"  best partition: {fmt_groups(packing.groups)}  cost {packing.value():.0f}")
    return packing


# ---------------------------------------------------------------------------
# [6] lanes
# ---------------------------------------------------------------------------

def step_6_lanes(cost, packing, explain: bool = True):
    from packing.lanes import assign_lanes

    groups = packing.partition()
    placement = assign_lanes(groups, cost)

    rule("-", "[6] LANES")
    if explain:
        wrap(
            "The objective is a function of the partition, so lanes are not a "
            "search variable -- but they still have to be chosen, and the choice "
            "is not free. A transversal macro whose control-slot-to-target-slot "
            "map is not the identity is realigned by permuting the target patch, "
            "and those routing swaps displace gates in time even though they cost "
            "no gates. So lane order is worth getting right cheaply and worth "
            "nothing at all to search over."
        )
        print()
        wrap(
            "One pass, patches heaviest-traffic first, each solved exactly "
            "against everything already placed -- a k-by-k linear assignment, "
            "maximising gates whose endpoints land in matching slots."
        )
        print()

    by_patch: Dict[int, Dict[int, int]] = {}
    for qubit, (patch, slot) in enumerate(placement):
        by_patch.setdefault(patch, {})[slot] = qubit
    for patch in sorted(by_patch):
        slots = "  ".join(
            f"slot {slot}=q{qubit}" for slot, qubit in sorted(by_patch[patch].items())
        )
        print(f"  patch {patch} : {slots}")

    # What the merging pass will later test: does a cross-patch gate land on
    # matching slots, or does it owe a routing swap?
    aligned = crossed = 0
    rows: List[str] = []
    for (left, right), events in sorted(cost.gates.items()):
        lp, ls = placement[left]
        rp, rs = placement[right]
        if lp == rp:
            continue
        if ls == rs:
            aligned += 1
        else:
            crossed += 1
        rows.append(
            f"    q{left}(p{lp}s{ls}) - q{right}(p{rp}s{rs})  x{len(events)}  "
            f"{'aligned' if ls == rs else 'CROSSED'}"
        )
    print()
    print(f"  cross-patch pairs: {aligned} on matching slots, {crossed} crossed")
    for row in rows[:12]:
        print(row)
    if len(rows) > 12:
        print(f"    ... (+{len(rows) - 12} more)")
    if explain:
        print()
        wrap(
            "Those matching slots are exactly the alignment step 5 of the merging "
            "pass tests. Every pair the lane solve puts on matching slots is a "
            "routing swap the transversal macro does not have to pay for later."
        )
    return placement


# ---------------------------------------------------------------------------
# [7] and [8]: the two comparisons
# ---------------------------------------------------------------------------

def measured(program, placement, n_code, commute, depth_weight, requested_noise):
    from packing.plan import measured_cost

    value = measured_cost(
        program, placement, n_code, depth_weight=depth_weight,
        h_commutation=commute, noise=requested_noise,
    )
    return float("inf") if value is None else value


def step_7_arms(program, arms, n_code, depth_weight, requested_noise):
    rule("-", "[7] CHOOSE THE COMMUTATION ARM")
    wrap(
        "Both arms are planned in full and the cheaper *measured* placement wins. "
        "The decision cannot be made before the partition: `commutation_pays` "
        "answers the same question on the naive packing, and the two placements "
        "disagree -- naive preferring the pass off while the packed placement "
        "prefers it on was worth 1.3% on one circuit, and left step 8 falling "
        "back to naive where the packed arm was in fact better."
    )
    print()
    wrap(
        "Model costs are NOT comparable across arms: each is priced against its "
        "own commuted view, so the two describe different circuits. Only the "
        "measured numbers are on one scale, which is why both get translated."
    )
    print()
    scored = []
    for plan in arms:
        value = measured(program, plan["placement"], n_code, plan["commute"],
                         depth_weight, requested_noise)
        scored.append((value, plan))
        print(f"  commute={str(plan['commute']):<5}  model {plan['cost']:>10.0f}"
              f"   measured {value:>12.2f}")
    best_value, best = min(scored, key=lambda pair: pair[0])
    print(f"  kept commute={best['commute']} at {best_value:.2f}")
    return best, best_value, [(f"commute={p['commute']}", v) for v, p in scored]


def step_8_verify(program, best, ours, n_code, k, depth_weight, requested_noise,
                  verify: bool):
    from packing.plan import naive_packing

    rule("-", "[8] VERIFY AGAINST NAIVE")
    wrap(
        "The model is a model -- it over-counts merging by 2-3%, which on one "
        "circuit was enough to prefer a placement 12.9% worse than naive. One "
        "extra translation turns 'usually better' into 'never worse', which is "
        "what makes the packing contribution safe to quote as a gain."
    )
    print()
    if not verify or len(set(p for p, _ in best["placement"])) <= 1:
        print("  skipped (verify off, or a single patch leaves nothing to compare)")
        return best["placement"], False

    fallback = naive_packing(program.num_qubits, k)
    theirs = measured(program, fallback, n_code, best["commute"], depth_weight,
                      requested_noise)
    print(f"  packed {ours:>12.2f}")
    print(f"  naive  {theirs:>12.2f}")
    fell_back = ours == float("inf") or theirs < ours
    print(f"  -> {'NAIVE KEPT (packed rejected)' if fell_back else 'packed kept'}")
    if requested_noise is None:
        print()
        wrap(
            "Currency here is weighted gates plus depth, because no noise profile "
            "was supplied -- the check has to be in whatever the search "
            "optimised, or it drifts from the objective. Pass a profile and both "
            "become expected detectable errors on the real schedule."
        )
    return (fallback if fell_back else best["placement"]), fell_back


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def walkthrough(
    program: QuantumCircuit,
    label: str,
    n_code: int = N_CODE,
    spread: int = 0,
    max_rounds: int = 8,
    restarts: Optional[int] = None,
    partitioner: str = "feasible",
    commute: str = "auto",
    verify: bool = True,
) -> None:
    from iceberg.utils import logical_capacity
    from packing.budget import ResourceBudget
    from packing.cost import DEFAULT_DEPTH_WEIGHT
    from packing.graph import NoiseProfile

    requested_noise = None  # mirrors plan_packing's default: gate counting
    noise = NoiseProfile()
    budget = ResourceBudget()
    depth_weight = DEFAULT_DEPTH_WEIGHT
    k = logical_capacity(n_code)

    rule("=")
    print(f"  packing pipeline walkthrough -- {label}")
    print(f"  {program.num_qubits} qubits, {len(program.data)} instructions, "
          f"n_code={n_code} (k={k})")
    rule("=")

    settings = (True, False) if commute == "auto" else (commute == "on",)
    arms: List[Dict] = []
    for index, arm_commute in enumerate(settings):
        explain = index == 0
        print()
        rule("#", f"ARM commute={arm_commute}")
        print()

        _, profiles = step_1_profile(program, arm_commute, explain)
        print()
        cost, slots = step_2_objective(
            program, profiles, n_code, noise, depth_weight, spread, budget, explain
        )
        print()
        packing = step_3_seed(cost, profiles, slots, noise, partitioner, explain)
        print()
        result = step_4_descend(packing, max_rounds, explain)
        print()
        packing = step_5_restarts(
            cost, packing, result, slots, max_rounds, restarts,
            program.num_qubits, explain,
        )
        print()
        placement = step_6_lanes(cost, packing, explain)

        parts = price_breakdown(cost, packing.partition())
        print()
        print("  final cost breakdown:")
        for name, value in parts.items():
            if name.startswith("_"):
                continue
            print(f"    {name:<28}{value:>12.1f}")
        print(f"    {'-' * 40}")
        print(f"    {'total':<28}{parts['_total']:>12.1f}")
        print(f"    (depth estimate {parts['_depth_estimate']:.1f}, busiest patch "
              f"{parts['_busiest']:.1f}, crossings {parts['_crossings']:.0f} "
              f"less {parts['_crossing_savings']:.0f} folded into macros)")

        arms.append({
            "commute": arm_commute,
            "placement": placement,
            "cost": packing.value(),
            "patches": len(packing.partition()),
        })

    print()
    best, ours, considered = step_7_arms(
        program, arms, n_code, depth_weight, requested_noise
    ) if len(arms) > 1 else (
        arms[0],
        measured(program, arms[0]["placement"], n_code, arms[0]["commute"],
                 depth_weight, requested_noise),
        [],
    )
    print()
    placement, fell_back = step_8_verify(
        program, best, ours, n_code, k, depth_weight, requested_noise, verify
    )

    print()
    rule("=")
    print("  final placement (qubit -> patch, slot)")
    rule("=")
    for qubit, (patch, slot) in enumerate(placement):
        print(f"    q{qubit:<3} -> patch {patch}, slot {slot}")
    patches = len({patch for patch, _ in placement})
    print()
    print(f"  {patches} patches, {patches * (n_code + 2)} physical qubits, "
          f"commute={best['commute']}" + (", naive kept" if fell_back else ""))

    # Same guarantee the merging walkthrough carries: this drives the stages by
    # hand, so it has to land where the real entry point lands.
    from packing.plan import plan_packing

    expected = plan_packing(
        program, n_code, spread=spread, max_rounds=max_rounds, restarts=restarts,
        partitioner=partitioner,
        h_commutation="auto" if commute == "auto" else (commute == "on"),
        verify=verify,
    )
    if list(expected.placement) != list(placement):
        raise AssertionError(
            f"the walkthrough diverged from plan_packing: "
            f"narrated {placement}, pipeline produced {expected.placement}"
        )
    print("  agrees with plan_packing on the same inputs")
    rule("=")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--circuit", default="toy",
                        help="'toy' for the merging example, or a QASMBench name "
                             "such as qpe_n9 or adder_n10")
    parser.add_argument("--n-code", type=int, default=N_CODE,
                        help=f"code size; k = n_code - 2 (default {N_CODE})")
    parser.add_argument("--spread", type=int, default=0,
                        help="empty patch slots above the tightest packing")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=None,
                        help="shuffled restarts; default is size-gated")
    parser.add_argument("--partitioner", default="feasible",
                        choices=["feasible", "swapped", "packed"])
    parser.add_argument("--commute", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the naive comparison in step 8")
    args = parser.parse_args()

    label, program = load_circuit(args.circuit)
    walkthrough(
        program, label,
        n_code=args.n_code, spread=args.spread, max_rounds=args.max_rounds,
        restarts=args.restarts, partitioner=args.partitioner,
        commute=args.commute, verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
