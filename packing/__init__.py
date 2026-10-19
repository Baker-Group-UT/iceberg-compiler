"""
placement — pack logical qubits into code patches, in four stages.

    from packing import plan_packing
    plan = plan_packing(circuit, n_code=6)
    plan.placement        # [(patch, slot), ...] per logical qubit
    print(plan.summary())

`plan_packing` needs a code size, because choosing one is a different question with a
different objective (`packing.resource`). To have both answered, and a physical circuit
back rather than a placement, start one level up:

    from packing import compile_circuit
    built = compile_circuit(circuit)
    built.circuit         # the physical circuit, ready to simulate

A rewrite of `grouping`, not an addition to it. Same problem, same cost accounting, one
objective downstream of the seed instead of three.

Why it is shorter
-----------------
The old stack had seven stages and three objectives: signed edge weights drove the seed,
cut weight drove the partitioner, and the analytic cost drove a final local search.
Measuring each stage against what it actually contributed found four that could be
removed outright:

    the patch-count scan returned one distinct count on 41 of 63 QASMBench circuits, and
    the tightest packing won on all 63

    the FM refinement inside uncoarsening, where it fires, moves the cut 1.5% and the
    analytic cost 10-20% -- real wins, found through an objective blind to them

    66% of the local-search neighbourhood was provably zero-gain, because it permuted
    lanes and the objective is a function of the partition alone

    the search and the measurement validating it used different depth weights, 0.0 and
    12.0

Why the graph stayed
--------------------
The obvious next step -- delete the interaction graph too, since the objective it
approximates is now cheap -- was tried and does not work. Descent on the true objective
leaves the naive partition untouched on `knn_n31` at k=10 while a partition 13% cheaper
*by that objective* exists. Cut weight is a pairwise relaxation, so minimising it moves
the whole partition coherently; the true objective is conjunctive and its landscape is
plateaus and deep local optima. A worse description of cost makes a better search guide.

So the graph seeds, and nothing after the seed consults it again. `seed.py` carries the
measurements.

Modules
-------
`cost`    the objective, as an incremental function of the partition
`seed`    the graph's one good job: global structure to start from
`search`  `descend`: move / spread / swap sweeps on the real objective
`lanes`   exact slot assignment, against every already-placed patch
`plan`    `plan_packing`, including the translate-and-compare safety net
`silent`  `plan_silent_aware`: the objective plus a price on silent logical errors
`standard` the paper's configuration (w10T): `STANDARD`, `plan_standard`, configs
`compiler` `compile_circuit`: resource selection, then packing, then one physical circuit
          (the standard planner by default; `planner="v4"` is the submitted compiler)
"""

from .cost import DEFAULT_DEPTH_WEIGHT, PackingCost, Term
from .lanes import assign_lanes
from .plan import (
    PackingPlan,
    measured_cost,
    naive_packing,
    pack,
    plan_packing,
)
from .search import DescentResult, Move, Packing, descend, naive_seed
from .seed import graph_seed

# Last, and it has to be: `compiler` reaches down into `packing.resource`, which imports
# `plan` and `tables` back out of this package, so it can only be loaded once the names
# above are bound.
from .compiler import Arm, CompiledCircuit, compile_circuit
from .silent import UndetectedRates, load_silent_rates, plan_silent_aware
from .standard import (
    STANDARD,
    AqornStandard,
    plan_standard,
    standard_noise_profile,
    standard_pipeline_config,
    standard_translator_config,
)

__all__ = [
    "DEFAULT_DEPTH_WEIGHT",
    "PackingCost",
    "Term",
    "assign_lanes",
    "PackingPlan",
    "measured_cost",
    "naive_packing",
    "pack",
    "plan_packing",
    "DescentResult",
    "Move",
    "Packing",
    "descend",
    "naive_seed",
    "graph_seed",
    "Arm",
    "CompiledCircuit",
    "compile_circuit",
    "UndetectedRates",
    "load_silent_rates",
    "plan_silent_aware",
    "STANDARD",
    "AqornStandard",
    "plan_standard",
    "standard_noise_profile",
    "standard_pipeline_config",
    "standard_translator_config",
]
