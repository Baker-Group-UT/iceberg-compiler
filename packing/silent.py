"""
silent.py -- AQORN's planner with a price on silent logical errors (the standard planner).

Moved here from `experiments/revision/exp8_undetected/cost_model.py` (Experiment8) when the
weight-10 planner with transversal in-patch CZ / XCX ("w10T") became the standard configuration
(2026-10-07, `packing.standard`); that module now re-exports these names, so its callers and
their results are unchanged. The code below is the experiment's, verbatim; what is new is
`UndetectedRates` living here, the packaged rate tables (`packing/data/`) and the
`plan_silent_aware` entry point, which loads the rates for the code size when none are given
and plans under the gadget options it is told (`iceberg.gadget_options`).

Rate tables (silent logical error per gadget execution at alpha = 1, by exact single-fault
enumeration, `experiments/revision/exp8_undetected/gadgets.py`):
    data/gadget_rates.csv              n_code 4, 6, 8 (Experiment8) and 10, 12 (scaling run)
    data/gadget_rates_transversal.csv  n_code 4 with the transversal in-patch CZ / XCX: equal
                                       to the first except CZ_intra = XCX_intra = 0
The transversal construction only exists at k = 2, so at n_code > 4 both settings use the
first table.

Experiment8's description, unchanged:

cost_model.py -- Experiment8, step 2: AQORN's planner with a price on silent errors.

AQORN's objective (`packing.cost.PackingCost`) counts every error once, whether the code
catches it or not. Experiment1c found the consequence: packers that put more two-qubit
gates inside a patch keep more shots and give worse answers, because an in-patch gadget
turns part of its faults into logical errors the checks cannot see. `gadgets.py` measures
how much: 0.6-0.83 p2 of silent error per in-patch CX/CZ/XCX and 0.82 p2 per single-lane
Hadamard, and exactly zero for cross-patch gadgets, transversal macros and the
whole-patch HH, at every code size measured.

`SilentAwareCost` is `PackingCost` plus `silent_weight` x (expected silent logical errors),
in the objective's own currency (`w2 / p2` weighted units per unit of probability, the
rate `packing.resource.noise.packing_profile` converts every other error at):

    silent(partition) = sum over in-patch pairs      sum over their gates  u_intra(kind)
                      + sum over cross-patch pairs   sum over their gates  u_cross(kind)
                      - sum over fired macros        m * u_cross(kind) - u_macro(kind)
                      - sum over fired HH            |patch| * u_H_lane - u_HH

A silent error is then charged `1 + silent_weight` times what a detected one is: once by
the base objective, `silent_weight` more times here. `silent_weight = 0` is AQORN.

`plan_silent_aware` is `packing.plan.plan_packing` step for step with three changes, all
switched off at `silent_weight = 0`:

    objective   `SilentAwareCost` for the descent
    seed graph  each gate edge loses `scale * (u_intra - u_cross)`: what co-locating the
                pair now costs in silent error
    selection   the commutation arms and `verify` add `silent_weight * silent(placement)`
                to `measured_cost`, in that measurement's currency -- expected errors, or
                weighted units when it falls back to gate counting (rz/rx in the program)

Everything else -- profiles, partitioner, descent, restarts, lanes, the naive fallback --
is the library's own code, so `silent_weight = 0` reproduces the shipped placement. That is
checked on every circuit (`planner_static.py`), not assumed.

A fixed patch count (`patches`, added 2026-10-08)
-------------------------------------------------
By default the patch count is an outcome: the seed's `target` is a ceiling, the objective
charges per live patch, and the naive seed and `verify`'s fallback are the tightest packing,
so a plan comes back at the tightest count whatever `spread` allows (105 of 105 density-sweep
cases above it). `patches=P` makes the count an input instead, for comparisons that hold the
machine fixed -- the density ablation's packed arm against a trivial packing at the same P:

    seed        `partition(..., exact=True)` -- exactly P occupied groups
    descent     `descend(fixed_count=True)` -- swaps, and moves that keep both patches live
    alternates  the naive packing at P (`search.program_order`) and balanced restarts at P
    verify      against the naive packing at P, not the tightest

At P = tightest nothing here can bind, so `patches=tightest` returns the default plan.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from packing.cost import DEFAULT_DEPTH_WEIGHT, PackingCost, Term
from packing.tables import GATE_KEYS, _complete_bijections, _coincident_layers

Placement = List[Tuple[int, int]]

TWO_QUBIT_KINDS = ("CX", "CZ", "XCX")
MACROS = {"CX": "CXT", "CZ": "CZT", "XCX": "XCXT"}
DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class UndetectedRates:
    """Silent-error probability per gadget execution at one code size, alpha = 1."""

    n_code: int
    p2: float
    intra: Dict[str, float]
    cross: Dict[str, float]
    macro: Dict[str, float]
    h_lane: float
    hh: float
    #: Single-qubit gates on one lane, keyed like `packing.tables.GATE_KEYS`'s values. H is
    #: the single-lane gadget (`h_lane`); a fired HH is credited separately.
    one_qubit: Dict[str, float]

    @classmethod
    def from_table(cls, table, n_code: int, p2: float) -> "UndetectedRates":
        rows = table[table["n_code"] == n_code].set_index("gadget")["p_logical"]
        return cls(
            n_code=n_code, p2=p2,
            intra={kind: float(rows[f"{kind}_intra"]) for kind in TWO_QUBIT_KINDS},
            cross={kind: float(rows[f"{kind}_cross"]) for kind in TWO_QUBIT_KINDS},
            macro={kind: float(rows[MACROS[kind]]) for kind in TWO_QUBIT_KINDS},
            h_lane=float(rows["H_lane"]), hh=float(rows["HH"]),
            one_qubit={"H": float(rows["H_lane"]), "S": float(rows["S"]),
                       "SX": float(rows["SX"]), "RZ": float(rows["RZ"]),
                       "RX": float(rows["RX"]), "X": float(rows["X"]),
                       "Z": float(rows["Z"]), "Y": 0.0},
        )


@lru_cache(maxsize=None)
def load_silent_rates(n_code: int, *, transversal_intra_two_qubit: bool = False,
                      p2: Optional[float] = None) -> UndetectedRates:
    """
    The packaged rates for `n_code`, for the gadget set in use.

    `p2` is the two-qubit error the rates are normalised by (the objective's currency); the
    default is the neutral-atom machine's, which is what every planned figure used.
    """
    import pandas as pd

    if p2 is None:
        from simulator.noise_models import StimNoiseParameters

        p2 = StimNoiseParameters.neutral_atom().two_qubit_error
    name = ("gadget_rates_transversal.csv" if transversal_intra_two_qubit and n_code == 4
            else "gadget_rates.csv")
    table = pd.read_csv(DATA / name)
    if n_code not in set(table["n_code"]):
        raise KeyError(f"no packaged silent-error rates for n_code={n_code} ({name})")
    return UndetectedRates.from_table(table, n_code, p2)


class SilentAwareCost(PackingCost):
    """`PackingCost` plus `silent_weight` x expected silent logical errors."""

    def __init__(self, profiles, n_code: int, *, rates: UndetectedRates,
                 silent_weight: float = 0.0, **kwargs) -> None:
        super().__init__(profiles, n_code, **kwargs)
        if rates.n_code != n_code:
            raise ValueError(f"rates are for n_code={rates.n_code}, cost for {n_code}")
        self.rates = rates
        self.silent_weight = float(silent_weight)
        #: Weighted units per unit of silent-error probability, times the weight.
        self.silent_scale = self.silent_weight * self.weights[1] / rates.p2
        if not self.silent_scale:
            return
        for pair, events in self.gates.items():
            # Deduplicated like `intra`, raw like `cross`, so the two stay term for term.
            inside = sum(rates.intra.get(GATE_KEYS[kind], 0.0)
                         for _, kind in sorted(set(events)))
            across = sum(rates.cross.get(GATE_KEYS[kind], 0.0) for _, kind in events)
            self.intra[pair] = self.intra[pair] + Term(two_qubit=self.silent_scale * inside)
            self.cross[pair] = self.cross[pair] + Term(two_qubit=self.silent_scale * across)

    def _within(self, members: Sequence[int]) -> float:
        savings = super()._within(members)
        if not self.silent_scale:
            return savings
        if len(members) == 1:
            # A solo patch's H is a whole-patch HH (`solo_hh`), which leaks nothing.
            if not self.solo_hh:
                return savings
            count = len(self.layers["h"].get(members[0], frozenset()))
            return savings + self.silent_scale * count * (self.rates.h_lane - self.rates.hh)
        if len(members) < 2:
            return savings
        # A fired HH replaces |patch| single-lane Hadamards, each a leaky gadget, with one
        # transversal one that leaks nothing.
        coincident = _coincident_layers(
            [self.layers["h"].get(q, frozenset()) for q in members], self.window
        )
        return savings + self.silent_scale * len(coincident) * (
            len(members) * self.rates.h_lane - self.rates.hh
        )

    def _transversal(self, events, left: int, right: int) -> Tuple[float, float]:
        savings, crossing_savings = super()._transversal(events, left, right)
        if not self.silent_scale or not events or left != right:
            return savings, crossing_savings
        # Re-derive which macros the parent fired, by its own rule, and credit the silent
        # error of the singles each one replaces. Zero for every rate table measured so far
        # (cross-patch gadgets and macros are both exactly silent-free), kept general.
        from iceberg.costs import carriage_cost

        w1, w2 = self.weights
        harmonic = sum(1.0 / i for i in range(1, left + 1)) if left else 0.0
        realign = 2.0 * max(0.0, left - harmonic) * carriage_cost("SWAP", self.weights)
        for kind, crossings in events.items():
            entry = self._macro.get(kind)
            if entry is None:
                continue
            single, macro = entry
            gain = left * (single[0] * w1 + single[1] * w2) - (
                macro[0] * w1 + macro[1] * w2) - realign
            if gain <= 0:
                continue
            key = GATE_KEYS[kind]
            fired = _complete_bijections(crossings, left, self.window)
            savings += self.silent_scale * fired * (
                left * self.rates.cross.get(key, 0.0) - self.rates.macro.get(key, 0.0)
            )
        return savings, crossing_savings


def groups_of(placement: Placement) -> List[frozenset]:
    groups: Dict[int, set] = {}
    for qubit, (patch, _) in enumerate(placement):
        groups.setdefault(int(patch), set()).add(qubit)
    return [frozenset(groups[p]) for p in sorted(groups)]


def build_costs(program, n_code: int, *, commute: bool, noise, rates: UndetectedRates,
                silent_weight: float, depth_weight: float = DEFAULT_DEPTH_WEIGHT,
                solo_hh: bool = False):
    """(profiles, cost) exactly as `packing.plan._plan_under` builds them, plus the term."""
    from packing.profiles import extract_profiles
    from packing.tables import translated_view

    view = translated_view(program, enable_h_commutation=commute)
    profiles = extract_profiles(view)
    cost = SilentAwareCost(
        profiles, n_code, rates=rates, silent_weight=silent_weight,
        num_qubits=program.num_qubits, depth_weight=depth_weight,
        movement_cost=noise.movement_cost, idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale, solo_hh=solo_hh,
    )
    return profiles, cost


def silent_constant(profiles, rates: UndetectedRates) -> float:
    """Silent error every placement of this view pays alike: each single-qubit gadget's own."""
    total = 0.0
    for profile in profiles.values():
        for event in profile.events:
            if event.partner is not None:
                continue
            key = GATE_KEYS.get(event.kind)
            if key is not None:
                total += rates.one_qubit.get(key, 0.0)
    return total


def silent_errors(program, placement: Placement, n_code: int, *, commute: bool, noise,
                  rates: UndetectedRates, solo_hh: bool = False) -> float:
    """
    Expected silent logical errors of a placement, as a probability at alpha = 1.

    The placement-dependent part is the objective at weight 1 minus the objective at
    weight 0, divided back out of its units -- by construction the quantity the search
    priced, not a second model of it. On top goes what every placement of this view pays
    alike (`silent_constant`), which a search can ignore but a comparison *across*
    commutation settings cannot: commuting Hadamards changes how many lane gadgets exist.
    """
    profiles, priced = build_costs(program, n_code, commute=commute, noise=noise,
                                   rates=rates, silent_weight=1.0, solo_hh=solo_hh)
    _, plain = build_costs(program, n_code, commute=commute, noise=noise, rates=rates,
                           silent_weight=0.0, solo_hh=solo_hh)
    groups = groups_of(placement)
    return silent_constant(profiles, rates) + (
        priced.of(groups) - plain.of(groups)) / priced.silent_scale


def silent_aware_graph(cost: SilentAwareCost, profiles, noise):
    """`packing.graph.build_graph`, with each gate edge charged its silent cost."""
    from packing.graph import InteractionGraph, build_graph

    graph = build_graph(profiles, cost.n_code, noise=noise, num_qubits=cost.num_qubits)
    if not cost.silent_scale:
        return graph
    weights = dict(graph.weights)
    for (left, right), events in cost.gates.items():
        inside = sum(cost.rates.intra.get(GATE_KEYS[kind], 0.0)
                     for _, kind in sorted(set(events)))
        across = sum(cost.rates.cross.get(GATE_KEYS[kind], 0.0) for _, kind in events)
        weights[(left, right)] = weights.get((left, right), 0.0) - cost.silent_scale * (
            inside - across)
    return InteractionGraph(weights, graph.num_qubits)


def _seed(cost: SilentAwareCost, profiles, slots: int, *, noise, partitioner: str,
          exact: bool = False):
    """
    `packing.seed.graph_seed` on the silent-aware graph (identical at weight 0).

    `exact` partitions into exactly `slots` occupied patches (`partition(exact=True)`).
    """
    from packing.partition import partition as feasible_partition
    from packing.search import Packing
    from packing.seed import DEFAULT_PARTITIONER, PARTITIONERS

    graph = silent_aware_graph(cost, profiles, noise)
    options = PARTITIONERS.get(partitioner, PARTITIONERS[DEFAULT_PARTITIONER])
    result = feasible_partition(graph, cost.n_code, slots, exact=exact, **options)
    groups = [set(group) for group in result.groups if group]
    k = cost.k
    tidy: List[set] = []
    for group in groups:
        members = sorted(group)
        for start in range(0, len(members), k):
            tidy.append(set(members[start:start + k]))
    placed = {q for group in tidy for q in group}
    for qubit in [q for q in range(cost.num_qubits) if q not in placed]:
        for group in tidy:
            if len(group) < k:
                group.add(qubit)
                break
        else:
            tidy.append({qubit})
    if exact and len(tidy) != slots:
        raise RuntimeError(f"exact seed has {len(tidy)} patches against a target of {slots}")
    while len(tidy) < slots:
        tidy.append(set())
    return Packing(cost, tidy)


def _plan_under(program, n_code: int, *, commute: bool, budget, noise, rates,
                silent_weight: float, depth_weight: float, spread: int, max_rounds: int,
                restarts: Optional[int], partitioner: str, patches: Optional[int] = None,
                solo_hh: bool = False):
    """
    `packing.plan._plan_under` with `SilentAwareCost` and the silent-aware seed.

    `patches` holds every stage at exactly that many live patches (module docstring);
    `_plan_silent_aware` has checked it is reachable and within the budget.
    """
    from iceberg.utils import logical_capacity
    from packing.lanes import assign_lanes
    from packing.plan import SMALL_ENOUGH_TO_RESTART, PackingPlan
    from packing.search import descend, naive_seed, shuffled_seed

    k = logical_capacity(n_code)
    profiles, cost = build_costs(program, n_code, commute=commute, noise=noise, rates=rates,
                                 silent_weight=silent_weight, depth_weight=depth_weight,
                                 solo_hh=solo_hh)
    exact = patches is not None
    if exact:
        slots = patches
    else:
        tightest = max(1, -(-program.num_qubits // k))
        slots = tightest + max(0, spread)
        ceiling = budget.patch_ceiling(n_code)
        if ceiling is not None:
            slots = max(tightest, min(slots, ceiling))

    packing = _seed(cost, profiles, slots, noise=noise, partitioner=partitioner, exact=exact)
    result = descend(packing, max_rounds=max_rounds, fixed_count=exact)
    if restarts is None:
        restarts = 4 if program.num_qubits <= SMALL_ENOUGH_TO_RESTART else 0
    rng = random.Random(0)
    alternatives = [naive_seed(cost, slots, patches=patches)]
    alternatives += [shuffled_seed(cost, slots, rng) for _ in range(max(0, restarts))]
    for candidate in alternatives:
        if exact and candidate.live != patches:
            continue  # a restart that could not be balanced into exactly `patches`
        descend(candidate, max_rounds=max_rounds, fixed_count=exact)
        if candidate.value() < packing.value() - 1e-9:
            packing = candidate
            result.packing = candidate

    groups = packing.partition()
    if exact and len(groups) != patches:
        raise RuntimeError(f"plan has {len(groups)} patches against a target of {patches}")
    return PackingPlan(
        placement=assign_lanes(groups, cost),
        n_code=n_code,
        cost=packing.value(),
        patches=len(groups),
        physical_qubits=len(groups) * (n_code + 2),
        h_commutation=commute,
        descent=result,
    )


def measured_with_currency(program, placement: Placement, n_code: int, *,
                           depth_weight: float, h_commutation: bool,
                           noise) -> Tuple[Optional[float], str]:
    """
    `packing.plan.measured_cost`, line for line, also saying which currency it answered in.

    'expected_error' when the schedule could be priced, 'gates_depth' when it fell back to
    weighted gates + depth (Stim cannot scan rz/rx, so originals carrying rotations always
    do), 'rejected' when translation failed. `planner_static.py` checks the value equals
    the library's on every circuit.
    """
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from iceberg.costs import DEFAULT_WORK_WEIGHTS
    from pipeline.config import PipelineConfig
    from pipeline.tasks.translate_permutation import translate_permutation

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = True
    config.translator_config.fault_tolerant = False
    try:
        translated = translate_permutation(program, placement, config=config)
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None, "rejected"
    one, two = one_two_qubit_gate_totals(translated.circuit)
    w1, w2 = DEFAULT_WORK_WEIGHTS
    cost = w1 * one + w2 * two + depth_weight * translated.circuit.depth()
    if noise is None:
        return cost, "gates_depth"

    from simulator.attribution import measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    try:
        split = measured_noise_split(
            translated.circuit, StimNoiseParameters.neutral_atom(),
            idle_time_scalar=noise.idle_time_scalar,
            distance_scalar=noise.distance_scalar,
        )
    except Exception:  # noqa: BLE001 - same fallback as the library
        return cost, "gates_depth"
    return split["total"], "expected_error"


def _plan_silent_aware(
    program,
    n_code: int,
    *,
    rates: UndetectedRates,
    silent_weight: float,
    noise=None,
    budget=None,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    spread: int = 0,
    max_rounds: int = 8,
    restarts: Optional[int] = None,
    partitioner: Optional[str] = None,
    h_commutation: Union[bool, str] = "auto",
    verify: bool = True,
    patches: Optional[int] = None,
    solo_hh: bool = False,
):
    """`packing.plan.plan_packing`, scoring and searching with the silent-error term."""
    from iceberg.costs import DEFAULT_WORK_WEIGHTS
    from iceberg.utils import logical_capacity
    from packing.budget import ResourceBudget
    from packing.graph import NoiseProfile
    from packing.plan import naive_packing
    from packing.seed import DEFAULT_PARTITIONER

    requested_noise = noise
    noise = noise or NoiseProfile()
    budget = budget or ResourceBudget()
    partitioner = partitioner or DEFAULT_PARTITIONER
    k = logical_capacity(n_code)
    tightest = max(1, -(-program.num_qubits // k))
    if patches is not None:
        if spread:
            raise ValueError("`patches` fixes the patch count, so `spread` cannot be combined "
                             "with it")
        if not tightest <= patches <= max(1, program.num_qubits):
            raise ValueError(
                f"{program.num_qubits} qubits at k = {k} fit between {tightest} and "
                f"{program.num_qubits} patches, not {patches}"
            )
        ceiling = budget.patch_ceiling(n_code)
        if ceiling is not None and patches > ceiling:
            raise ValueError(f"the budget allows at most {ceiling} patches at n = {n_code}, "
                             f"not {patches}")
    shared = dict(budget=budget, noise=noise, rates=rates, silent_weight=silent_weight,
                  depth_weight=depth_weight, spread=spread, max_rounds=max_rounds,
                  restarts=restarts, partitioner=partitioner, patches=patches,
                  solo_hh=solo_hh)
    units = DEFAULT_WORK_WEIGHTS[1] / rates.p2
    audit: Dict[str, object] = {"currency": None}

    def score(placement: Placement, commute: bool) -> float:
        measured, currency = measured_with_currency(
            program, placement, n_code, depth_weight=depth_weight,
            h_commutation=commute, noise=requested_noise)
        audit["currency"] = currency
        if measured is None:
            return float("inf")
        if not silent_weight:
            return measured
        silent = silent_errors(program, placement, n_code, commute=commute, noise=noise,
                               rates=rates, solo_hh=solo_hh)
        return measured + silent_weight * silent * (
            1.0 if currency == "expected_error" else units)

    if h_commutation == "auto":
        arms = [_plan_under(program, n_code, commute=commute, **shared)
                for commute in (True, False)]
        scored = [(score(arm.placement, arm.h_commutation), arm) for arm in arms]
        ours, plan = min(scored, key=lambda pair: pair[0])
        plan.considered = [(f"commute={arm.h_commutation}", value) for value, arm in scored]
    else:
        plan = _plan_under(program, n_code, commute=bool(h_commutation), **shared)
        ours = None

    if verify and plan.patches > 1:
        # At a fixed count the fallback is the naive packing at that count: the tightest
        # one would answer a different question (fewer patches), and win it for that reason.
        fallback = naive_packing(program.num_qubits, k, patches)
        if ours is None:
            ours = score(plan.placement, plan.h_commutation)
        theirs = score(fallback, plan.h_commutation)
        plan.verified = True
        plan.considered = plan.considered + [("packed", ours), ("naive", theirs)]
        if ours == float("inf") or theirs < ours:
            plan.placement = fallback
            plan.patches = tightest if patches is None else patches
            plan.physical_qubits = plan.patches * (n_code + 2)
            plan.fell_back = True
    plan.currency = audit["currency"]
    return plan


def plan_silent_aware(
    program,
    n_code: int,
    *,
    rates: Optional[UndetectedRates] = None,
    silent_weight: float,
    noise=None,
    budget=None,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    spread: int = 0,
    max_rounds: int = 8,
    restarts: Optional[int] = None,
    partitioner: Optional[str] = None,
    h_commutation: Union[bool, str] = "auto",
    verify: bool = True,
    transversal_intra_two_qubit: bool = False,
    patches: Optional[int] = None,
    solo_hh: bool = False,
):
    """
    `packing.plan.plan_packing`, scoring and searching with the silent-error term.

    `rates` defaults to the packaged table for `n_code` and the gadget set. With
    `transversal_intra_two_qubit` the whole plan -- cost tables, commutation decision, every
    verify translation -- runs under the transversal in-patch CZ / XCX
    (`iceberg.gadget_options`), as the measured w10T plans did with the global switch on.

    `patches` plans at exactly that many patches -- between ceil(m / k) and m, within the
    budget, and not together with `spread` -- and `verify` then guards against the naive
    packing at that count (module docstring). None leaves the count to the objective.

    `solo_hh` prices a solo patch's Hadamards as the whole-patch HH the translator emits
    (`PackingCost.solo_hh`), in the objective and in `verify`'s silent-error term. Off is the
    historical model, which charges them as leaky single-lane gadgets.
    """
    from iceberg.gadget_options import transversal_intra

    if rates is None:
        rates = load_silent_rates(n_code, transversal_intra_two_qubit=transversal_intra_two_qubit)
    with transversal_intra(transversal_intra_two_qubit):
        return _plan_silent_aware(
            program, n_code, rates=rates, silent_weight=silent_weight, noise=noise,
            budget=budget, depth_weight=depth_weight, spread=spread, max_rounds=max_rounds,
            restarts=restarts, partitioner=partitioner, h_commutation=h_commutation,
            verify=verify, patches=patches, solo_hh=solo_hh,
        )
