"""
inner_plan.py -- AQORN's placement search at the inner level, restricted to spread placements.

The inner level's program is C1 (the outer translation, flattened). AQORN's planner would pack
C1's qubits freely, and its cost model has no notion of the outer code: left alone it would pair
interacting qubits of one outer patch in one inner patch -- exactly the index-order layout that
drops the concatenated distance to 2. So this runs AQORN's own machinery (the weight-10
silent-aware cost model of `exp8_undetected.cost_model`, the lane assignment of `packing.lanes`,
and `plan_silent_aware`'s score as the verify step) over the spread placements only:

  1. starts: the stride and role-aligned layouts (`construct.inner_layouts`);
  2. steepest descent on the model, `cost.of(groups)`, over swaps of two C1 qubits between inner
     patches that keep the placement spread;
  3. lanes by `assign_lanes`; then, as AQORN's verify does, every candidate (each start as given,
     each descent result, both H-commutation settings) is scored by translating it --
     measured expected error + weight x model silent error -- and the lowest is kept. A start is
     always a candidate, so the result is never worse than the naive layouts on that score.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.concatenation.construct import Flattened, inner_layouts, is_spread

Perm = List[Tuple[int, int]]


class Scorer:
    """`plan_silent_aware`'s score for any placement of `program`, costs built once per setting."""

    def __init__(self, program, n_code: int, rates, noise, weight: float):
        from experiments.revision.exp8_undetected.cost_model import build_costs
        from iceberg.costs import DEFAULT_WORK_WEIGHTS

        self.program, self.n_code, self.rates, self.noise, self.weight = (
            program, n_code, rates, noise, weight)
        self.units = DEFAULT_WORK_WEIGHTS[1] / rates.p2
        self.costs = {}
        for commute in (False, True):
            profiles, priced = build_costs(program, n_code, commute=commute, noise=noise,
                                           rates=rates, silent_weight=weight)
            _, unit = build_costs(program, n_code, commute=commute, noise=noise, rates=rates,
                                  silent_weight=1.0)
            _, plain = build_costs(program, n_code, commute=commute, noise=noise, rates=rates,
                                   silent_weight=0.0)
            self.costs[commute] = (profiles, priced, unit, plain)

    def model(self, groups, commute: bool) -> float:
        return self.costs[commute][1].of(groups)

    def silent(self, placement, commute: bool) -> float:
        from experiments.revision.exp8_undetected.cost_model import groups_of, silent_constant

        profiles, _, unit, plain = self.costs[commute]
        groups = groups_of(placement)
        return silent_constant(profiles, self.rates) + (
            unit.of(groups) - plain.of(groups)) / unit.silent_scale

    def score(self, placement, commute: bool) -> Tuple[float, str]:
        from experiments.revision.exp8_undetected.cost_model import measured_with_currency
        from packing.cost import DEFAULT_DEPTH_WEIGHT

        measured, currency = measured_with_currency(
            self.program, placement, self.n_code, depth_weight=DEFAULT_DEPTH_WEIGHT,
            h_commutation=commute, noise=self.noise)
        if measured is None:
            return math.inf, currency
        scale = 1.0 if currency == "expected_error" else self.units
        return measured + self.weight * self.silent(placement, commute) * scale, currency


def _groups(placement: Perm) -> List[set]:
    blocks: Dict[int, set] = {}
    for q, (patch, _) in enumerate(placement):
        blocks.setdefault(int(patch), set()).add(q)
    return [blocks[b] for b in sorted(blocks)]


def _spread_groups(groups: Sequence[set], roles) -> bool:
    return all(len({roles[q][0] for q in g}) == len(g) for g in groups)


def descend(groups: List[set], scorer: Scorer, commute: bool, roles, *, max_rounds: int = 12):
    """Steepest descent on the model over spread-preserving swaps. Returns (groups, value, moves)."""
    current = [set(g) for g in groups]
    value = scorer.model([frozenset(g) for g in current], commute)
    moves = 0
    for _ in range(max_rounds):
        best = None
        for a in range(len(current)):
            for b in range(a + 1, len(current)):
                for qa in current[a]:
                    for qb in current[b]:
                        ga = (current[a] - {qa}) | {qb}
                        gb = (current[b] - {qb}) | {qa}
                        if not _spread_groups((ga, gb), roles):
                            continue
                        trial = list(current)
                        trial[a], trial[b] = ga, gb
                        v = scorer.model([frozenset(g) for g in trial], commute)
                        if v < value - 1e-12 and (best is None or v < best[0]):
                            best = (v, a, b, ga, gb)
        if best is None:
            break
        value, a, b, ga, gb = best
        current[a], current[b] = ga, gb
        moves += 1
    return current, value, moves


def plan_inner(flat: Flattened, *, rates, noise, weight: float = 10.0,
               max_rounds: int = 30) -> Dict[str, object]:
    """The spread placement and commutation setting AQORN's machinery picks for C1."""
    from packing.lanes import assign_lanes

    scorer = Scorer(flat.program, 4, rates, noise, weight)
    layouts = inner_layouts(flat)
    candidates: List[Tuple[str, Perm, bool]] = []
    for commute in (True, False):
        for start_name in ("stride", "role_aligned"):
            start = layouts[start_name]
            candidates.append((f"{start_name}", start, commute))
            groups, _, moves = descend(_groups(start), scorer, commute, flat.roles,
                                       max_rounds=max_rounds)
            placement = [tuple(p) for p in assign_lanes([frozenset(g) for g in groups],
                                                         scorer.costs[commute][1])]
            if is_spread(placement, flat.roles):
                candidates.append((f"{start_name}+descent({moves})", placement, commute))
    scored = []
    seen = set()
    for label, placement, commute in candidates:
        key = (tuple(map(tuple, placement)), commute)
        if key in seen:
            continue
        seen.add(key)
        value, currency = scorer.score(placement, commute)
        scored.append((value, label, placement, commute, currency))
    scored.sort(key=lambda item: item[0])
    value, label, placement, commute, currency = scored[0]
    return {"placement": [tuple(p) for p in placement], "commute": bool(commute), "score": value,
            "label": label, "currency": currency,
            "considered": [(lab, com, val) for val, lab, _, com, _ in scored]}
