"""
plan.py — the whole algorithm, end to end.

    profile   the circuit the translator will actually place
    seed      multilevel-partition the signed graph, for global structure
    descend   sweep qubits on the real objective until a sweep changes nothing
    lanes     assign slots exactly, once, on the final partition
    commute?  do all of the above both ways, and keep the cheaper placement
    verify    translate against the naive packing and keep the cheaper

The commutation question moved from the top of that list to the bottom. It used to be
answered first, by `commutation_pays`, on the naive packing -- but the profile, the seed
graph and the objective are all functions of the commuted view, so answering it first
means answering it about a placement that does not exist yet. `plan_packing` documents
what that cost.

Five stages where there were seven, and -- more to the point -- one objective downstream
of the seed where there were three. What went away:

*the patch-count scan* -- five candidates, of which 41 of 63 QASMBench circuits returned
    one distinct count, and the tightest packing won all 63. The count is not a separate
    decision: `spread` moves raise it when it pays and the objective charges
    `ENCODE + DECODE` per live patch, so the descent settles it.

*the separate analytic local search* -- `refine.refine_placement` ran after the
    partitioner with its own move set, two thirds of which could not change the cost.
    `descend` is that stage rebuilt: same objective, membership moves only, and a
    per-qubit sweep rather than one globally-steepest move per round. It is where the
    improvement now comes from: on `knn_n31` at k=6 the seed alone is 5.3% better than
    naive and the descent takes it to 14.6%.

Note what did *not* go away, despite an earlier version of this docstring saying so: the
FM passes and `_consolidate` inside `multilevel.partition` still run, because `seed`
calls that function unmodified. They optimise the cut, and `descend` re-finds their gains
on the true objective -- stubbing both out changes the seed cost on 3 of 9 hard circuits
and the *final* cost on 1 (`adder_n64` at k=6, +1.94%). So they are nearly redundant now,
and kept for that one case rather than because they are pulling their weight.

*lane search* -- `reorder`, and same-patch `swap`. The objective is a function of the
    partition, so both are provably zero-gain; they were 72 of 109 candidate moves on
    `qpe_n9`, scored on every round. Lanes are now decided once, at the end.

*the second depth weight* -- `plan_placement` scored with `depth_weight=0.0` while the
    figure path measured with 12.0, so the search and the measurement validating it
    wanted different things. One number does both here.

What did *not* go away is the interaction graph, and `seed.py` explains why at length:
it is a poor description of cost but a much better search guide than the true objective,
whose landscape is conjunctive and full of deep local optima. Deleting it was tried and
cost three points of mean improvement at k >= 6.

`verify` is kept and, unlike `grouping.plan.plan_placement`, is wired in by default. The
model is a model -- it over-counts merging by 2-3%, which on `vqe_n4` at k=2 was enough
to prefer a placement 12.9% worse than naive. Two translations turn "usually better" into
"never worse", which is what makes the packing contribution safe to quote as a gain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from qiskit import QuantumCircuit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from .graph import NoiseProfile
from .budget import ResourceBudget
from iceberg.costs import DEFAULT_WORK_WEIGHTS
from iceberg.utils import logical_capacity
from .tables import translated_view
from .profiles import extract_profiles
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

from .cost import DEFAULT_DEPTH_WEIGHT, PackingCost
from .lanes import assign_lanes
from .search import DescentResult, Packing, descend, naive_seed, program_order, shuffled_seed
from .seed import DEFAULT_PARTITIONER, graph_seed

__all__ = [
    "PackingPlan",
    "plan_packing",
    "pack",
    "naive_packing",
    "measured_cost",
]

Placement = List[Tuple[int, int]]

#: Below this width a full descent costs a few milliseconds, so restarting from other
#: seeds is free. Above it, restarts measured no gain across QASMBench and cost 2-3x.
SMALL_ENOUGH_TO_RESTART = 16


def naive_packing(num_qubits: int, k: int, patches: Optional[int] = None) -> Placement:
    """
    Fill patches in program order. The baseline, and the fallback for `verify`.

    With `patches`, program order into exactly that many patches, full ones first
    (`search.program_order`) -- the naive packing at a fixed patch count, and the fallback
    for a plan made at one.
    """
    if patches is None:
        return [(index // k, index % k) for index in range(num_qubits)]
    placement: Placement = [(0, 0)] * num_qubits
    for patch, members in enumerate(program_order(num_qubits, k, patches)):
        for lane, qubit in enumerate(members):
            placement[qubit] = (patch, lane)
    return placement


def measured_cost(
    program: QuantumCircuit,
    placement: Placement,
    n_code: int,
    *,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    fault_tolerant: bool = False,
    h_commutation: bool = True,
    merging: bool = True,
    noise: Optional["NoiseProfile"] = None,
) -> Optional[float]:
    """
    Translate for real and price the result. `None` if the placement is rejected.

    Two currencies, chosen by whether `noise` is given:

        noise is None   weighted gates plus `depth_weight` per layer. What a caller who
                        asked for gate counting optimised, so it is what they get checked
                        on -- `plan_packing` passes the profile it was *handed*, not the
                        `NoiseProfile()` it defaults to, precisely so the check cannot
                        drift from the objective.
        noise given     expected detectable errors on the schedule the simulator will
                        really run, via `simulator.attribution.measured_noise_split`.

    A single `depth_weight` runs the gate-counting objective and its check; the old stack
    used 0.0 in `plan_placement` and 12.0 on the figure path, which meant the search and
    the measurement validating it wanted different things.

    **`noise` is what makes `verify` a real backstop, and without it the guarantee was
    vacuous.** This used to return gates and depth alone -- no movement term at any
    price -- so "keep the cheaper placement" was decided in a currency that omitted the
    dominant cost. Measured on `ising_n34` at k = 4, `distance_scalar = 128`: the old
    metric scored the packed placement 8096 against naive's 8568 and kept it, while the
    schedule charged it 27.66 expected errors against naive's 9.55, a 2.9x loss driven by
    3.4x the movement. Four of four benchmarks tested went the same way at that scalar,
    and `ising_n34` went the same way at `distance_scalar = 1` too, by 5.4%. Pricing the
    real schedule is what lets `verify` reject those.
    """
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    config.translator_config.fault_tolerant = fault_tolerant
    try:
        translated = translate_permutation(program, placement, config=config)
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    one, two = one_two_qubit_gate_totals(translated.circuit)
    w1, w2 = DEFAULT_WORK_WEIGHTS
    cost = w1 * one + w2 * two + depth_weight * translated.circuit.depth()
    if noise is None:
        return cost

    from simulator.attribution import measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    try:
        split = measured_noise_split(
            translated.circuit, StimNoiseParameters.neutral_atom(),
            idle_time_scalar=noise.idle_time_scalar,
            distance_scalar=noise.distance_scalar,
        )
    except Exception:  # noqa: BLE001 - attribution needs a layout; fall back to gates
        return cost
    # Expected detectable errors, and nothing else. `split["total"]` already carries the
    # gate and measurement error the weighted counts above stand in for, so returning it
    # alone is the whole cost rather than a term added to one.
    #
    # **Not gates + depth + movement + idle.** That mixture double-charges idling --
    # `depth_weight * depth` is a proxy for exactly what `split["idle"]` measures
    # directly -- and because it is not expected error, it ranks placements by something
    # the machine does not charge. Measured over 8 QASMBench proxies x `distance_scalar`
    # {0.5, 1, 2, 8} x k {2, 4}: the mixture regressed against naive in 4 of 8 cells,
    # worst -3.06%, while ranking on expected error alone regressed in 0 of 8 and held a
    # positive mean in all of them.
    #
    # The consequence, and it is deliberate: `depth_weight` is inert whenever a profile is
    # supplied. Depth is no longer something the planner will trade error for; it is an
    # outcome this objective happens to improve, and figures should report it as one.
    return split["total"]


@dataclass
class PackingPlan:
    """A placement, what it costs, and how it was reached."""

    placement: Placement
    n_code: int
    cost: float
    patches: int
    physical_qubits: int
    #: Whether the plan was built with H commutation on. `plan_packing` decides this per
    #: (circuit, code size) and prices everything under the answer, so a caller that
    #: translates `placement` without honouring it gets a different circuit than the one
    #: the plan costed -- measured 30% off, and worse than the naive arm that `verify`
    #: had just guaranteed it beat. Recording it is what makes the plan self-describing.
    h_commutation: bool = True
    descent: Optional[DescentResult] = None
    #: Set when `verify` ran; `fell_back` when it preferred the naive packing.
    verified: bool = False
    fell_back: bool = False
    considered: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def occupancy(self) -> List[int]:
        sizes: dict = {}
        for patch, _ in self.placement:
            sizes[patch] = sizes.get(patch, 0) + 1
        return sorted(sizes.values(), reverse=True)

    def summary(self) -> str:
        tail = ""
        if self.verified:
            tail = " [naive kept]" if self.fell_back else " [verified]"
        moves = self.descent.moves if self.descent else 0
        return (
            f"{self.patches} patches (sizes {self.occupancy}), "
            f"{self.physical_qubits} physical qubits, cost {self.cost:.0f}, "
            f"{moves} moves{tail}"
        )


def _plan_under(
    program: QuantumCircuit,
    n_code: int,
    *,
    commute: bool,
    budget: ResourceBudget,
    noise: NoiseProfile,
    depth_weight: float,
    spread: int,
    max_rounds: int,
    restarts: Optional[int],
    partitioner: str,
) -> PackingPlan:
    """
    One complete plan at a fixed commutation setting. No verify, no arm selection.

    Split out of `plan_packing` so the `auto` path can build both arms and compare them
    on the placements they actually produce. Everything here is a function of `commute`:
    the profile is extracted from the commuted view, so the objective, the seed graph and
    the descent are all pricing a different circuit in the two arms.
    """
    k = logical_capacity(n_code)
    view = translated_view(program, enable_h_commutation=commute)
    profiles = extract_profiles(view)
    cost = PackingCost(
        profiles=profiles,
        n_code=n_code,
        num_qubits=program.num_qubits,
        depth_weight=depth_weight,
        movement_cost=noise.movement_cost,
        idle_cost=noise.idle_cost,
        distance_scale=noise.distance_scale,
    )

    tightest = max(1, -(-program.num_qubits // k))
    slots = tightest + max(0, spread)
    ceiling = budget.patch_ceiling(n_code)
    if ceiling is not None:
        slots = max(tightest, min(slots, ceiling))

    packing = graph_seed(
        cost, profiles, slots, noise=noise, partitioner=partitioner
    )
    result = descend(packing, max_rounds=max_rounds)

    # The naive seed is always tried, at any width. The graph seed optimises cut weight,
    # which tracks the objective well when the objective is mostly gate counts and badly
    # when `noise` prices movement: at `movement_cost=52` -- an idle/distance ratio of
    # 1/16 -- the descent from the graph seed lands *above* naive on its own objective on
    # `ising_n26` (18,460 against 17,758) and cannot climb out, because a 1D chain is a
    # partition naive already nearly solves. Seeding from naive costs one descent and
    # makes "no worse than naive" hold in the model rather than only after `verify`
    # re-measures it. It ties naive where nothing better exists and beats both elsewhere
    # (`ising_n42` at ratio 1: 29,953 -> 29,485).
    #
    # The *shuffled* restarts stay size-gated, which is what `search.shuffled_seed`
    # measured: on small circuits the basins are few, so which one the seed lands in
    # decides the answer; on large ones they changed nothing and cost 2-3x.
    if restarts is None:
        restarts = 4 if program.num_qubits <= SMALL_ENOUGH_TO_RESTART else 0
    rng = random.Random(0)
    alternatives = [naive_seed(cost, slots)]
    alternatives += [shuffled_seed(cost, slots, rng) for _ in range(max(0, restarts))]
    for candidate in alternatives:
        descend(candidate, max_rounds=max_rounds)
        if candidate.value() < packing.value() - 1e-9:
            packing = candidate
            result.packing = candidate

    groups = packing.partition()
    return PackingPlan(
        placement=assign_lanes(groups, cost),
        n_code=n_code,
        cost=packing.value(),
        patches=len(groups),
        physical_qubits=len(groups) * (n_code + 2),
        h_commutation=commute,
        descent=result,
    )


def _plan_packing(
    program: QuantumCircuit,
    n_code: int,
    *,
    budget: Optional[ResourceBudget] = None,
    noise: Optional[NoiseProfile] = None,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    spread: int = 0,
    max_rounds: int = 8,
    restarts: Optional[int] = None,
    partitioner: str = DEFAULT_PARTITIONER,
    h_commutation: Union[bool, str] = "auto",
    verify: bool = True,
) -> PackingPlan:
    """
    Pack `program` into patches of the [[n, n-2, 2]] code.

    `spread` allocates that many empty patch slots above the tightest packing, which
    the descent may migrate into. Empty slots are free -- the objective charges live
    patches only -- so this bounds how far the search may spread, not what it costs.
    Zero is right on a gate-counting machine: an extra patch costs `ENCODE + DECODE`,
    234 weighted units at n=6, against the 12 units a crossing CNOT saves, so
    spreading needs a great many crossings to pay. Raise it when `noise` prices
    movement or idling, which is the regime where it does.

    `h_commutation="auto"` plans the circuit *both* ways and keeps whichever final
    placement measures cheaper. `commutation_pays` -- which this used to call -- answers
    the same question on the naive packing, and the two placements disagree: on
    `dnn_n51` at k=2 naive prefers the pass off (42,160 against 42,768 weighted) while
    the packed placement prefers it on (41,918 against 42,486). Asking about a placement
    the planner is about to discard got that circuit wrong by 1.3%, and left `verify`
    falling back to naive on a circuit where the packed arm was in fact better. The two
    passes interact through the partition, so the decision cannot be made before it.

    The cost is one extra descent and one fewer translation than the old path, because
    the arm comparison reuses the measurement `verify` needed anyway.
    """
    # Kept before the defaulting below, because `verify` must check on whatever the search
    # optimised. `NoiseProfile()` prices movement at zero, so a caller who passed nothing
    # asked for gate counting -- and scoring *that* search on expected error would put the
    # objective and its check back into disagreement, which is the failure the docstring on
    # `depth_weight` describes. Only a caller who supplied a profile gets the error-based
    # criterion, and they get the one they supplied.
    requested_noise = noise
    noise = noise or NoiseProfile()
    budget = budget or ResourceBudget()
    k = logical_capacity(n_code)
    shared = dict(
        budget=budget, noise=noise, depth_weight=depth_weight, spread=spread,
        max_rounds=max_rounds, restarts=restarts, partitioner=partitioner,
    )

    def score(candidate: PackingPlan) -> float:
        """
        What the placement really costs, translated. `inf` when it is rejected.

        Handed the same `noise` the search was given, so the check charges movement and
        idling on the real schedule instead of counting gates. Without it `verify` kept
        placements the simulator punished -- see `measured_cost`.
        """
        measured = measured_cost(
            program, candidate.placement, n_code, depth_weight=depth_weight,
            h_commutation=candidate.h_commutation, noise=requested_noise,
        )
        return float("inf") if measured is None else measured

    if h_commutation == "auto":
        arms = [
            _plan_under(program, n_code, commute=commute, **shared)
            for commute in (True, False)
        ]
        scored = [(score(candidate), candidate) for candidate in arms]
        # Model costs are not comparable across arms -- each is priced against its own
        # commuted view, so they describe different circuits. Only the measured numbers
        # are on one scale, which is why both arms get translated.
        ours, plan = min(scored, key=lambda pair: pair[0])
        plan.considered = [
            (f"commute={candidate.h_commutation}", cost) for cost, candidate in scored
        ]
    else:
        plan = _plan_under(program, n_code, commute=bool(h_commutation), **shared)
        ours = None

    if verify and plan.patches > 1:
        fallback = naive_packing(program.num_qubits, k)
        if ours is None:
            ours = score(plan)
        theirs = measured_cost(program, fallback, n_code, depth_weight=depth_weight,
                               h_commutation=plan.h_commutation, noise=requested_noise)
        plan.verified = True
        plan.considered = plan.considered + [
            ("packed", ours),
            ("naive", theirs if theirs is not None else float("inf")),
        ]
        if ours == float("inf") or (theirs is not None and theirs < ours):
            plan.placement = fallback
            plan.patches = max(1, -(-program.num_qubits // k))
            plan.physical_qubits = plan.patches * (n_code + 2)
            plan.fell_back = True
    return plan


def plan_packing(
    program: QuantumCircuit,
    n_code: int,
    *,
    budget: Optional[ResourceBudget] = None,
    noise: Optional[NoiseProfile] = None,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    spread: int = 0,
    max_rounds: int = 8,
    restarts: Optional[int] = None,
    partitioner: str = DEFAULT_PARTITIONER,
    h_commutation: Union[bool, str] = "auto",
    verify: bool = True,
    transversal_intra_two_qubit: bool = False,
) -> PackingPlan:
    """
    Pack `program` into patches of the [[n, n-2, 2]] code (see `_plan_packing`).

    `transversal_intra_two_qubit` plans for the transversal in-patch CZ / XCX at k = 2
    (`iceberg.gadget_options`): the cost tables the search prices, the commutation decision
    and every verify translation see that gadget. Off keeps the planner exactly as it was;
    the standard configuration (`packing.standard`) plans with the silent-aware planner and
    this on.
    """
    from iceberg.gadget_options import transversal_intra

    with transversal_intra(transversal_intra_two_qubit):
        return _plan_packing(
            program, n_code, budget=budget, noise=noise, depth_weight=depth_weight,
            spread=spread, max_rounds=max_rounds, restarts=restarts,
            partitioner=partitioner, h_commutation=h_commutation, verify=verify,
        )


def pack(program: QuantumCircuit, n_code: int, **kwargs) -> Placement:
    """`plan_packing` when only the placement is wanted."""
    return plan_packing(program, n_code, **kwargs).placement
