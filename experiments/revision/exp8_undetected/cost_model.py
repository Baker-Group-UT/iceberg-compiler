"""
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
"""

# The planner moved into the library when weight 10 with transversal in-patch CZ / XCX became
# the standard configuration (2026-10-07): `packing.silent`, code unchanged. These names stay
# importable from here so every Experiment8 / figure script runs, and plans, exactly as before.
from __future__ import annotations  # noqa: F404 - kept for parity with the original module

from packing.silent import (  # noqa: F401
    MACROS,
    TWO_QUBIT_KINDS,
    Placement,
    SilentAwareCost,
    UndetectedRates,
    _plan_under,
    _seed,
    build_costs,
    groups_of,
    load_silent_rates,
    measured_with_currency,
    plan_silent_aware,
    silent_aware_graph,
    silent_constant,
    silent_errors,
)
