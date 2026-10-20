"""
standard.py -- the AQORN configuration the paper uses: w10T (since 2026-10-07).

    planner      `packing.silent.plan_silent_aware`, silent_weight = 10: AQORN's objective plus
                 10 x the expected silent logical errors. The weight sweep with the transversal
                 gadget on (W = 3..30, Fig 6 re-simulated with its own seeds; branch
                 v2-tvd-explore, reports/weight_sweep.md) found TVD on a plateau from W = 9-10
                 and LSR falling above it; picking the best weight on half the circuits gained
                 nothing over 10 on the other half. Above k = 2 the planner searches at weight 5
                 (`wide_silent_weight`, reports/selector_standard.md).
    gadgets      transversal in-patch CZ / XCX at k = 2 (`iceberg.gadget_options`)
    scheduling   every program layer-sorted before translation (`TranslatorConfig.asap_order`),
                 so a pass is credited with what it does, not with the order it emits in
    noise model  the planner prices movement and idling as Fig 6's machine charges them:
                 `packing.resource.noise.packing_profile` with default knobs, distance and idle
                 noise on at scalar 1 -- the profile every w10T plan was made with
    code size    `compile_circuit` plans every candidate code size with this planner and keeps
                 the one with the lowest standard objective (`standard_objective`: detected
                 errors read off the schedule + `ranking_weight()` x silent errors), so choosing
                 the code size, placing, and the final pick all optimise one thing
    inputs       since 2026-10-09 every program gets one verified round of logical optimisation
                 first (`translator.logical_optimization`: Qiskit O3 to a fixed point plus
                 Clifford / CNOT-block resynthesis, kept only if the same unitary), in the
                 translator, in the planner's view of the program and in the QASMBench loaders.
                 Process-wide, not a field here: AQORN_LOGICAL_OPT=0 turns it off everywhere
    solo patches since 2026-10-09 the planner prices a solo patch's Hadamards as the whole-patch
                 HH the translator emits for them (`solo_hh_aware`); plans recorded before then
                 priced them as single-lane gadgets, which differs for odd widths and fixed
                 patch counts (`STANDARD.with_(solo_hh_aware=False)` reproduces them)

The low-level classes keep their historical defaults (`TranslatorConfig()`, `plan_packing`,
`plan_silent_aware`), so every script that reproduces a submitted or archived figure is
untouched. New code asks for the standard here; `packing.compile_circuit` uses it by default.

    from packing.standard import plan_standard, standard_pipeline_config

    plan = plan_standard(program)                       # n_code = 4
    config = standard_pipeline_config(commute=plan.h_commutation)
    translated = translate_permutation(program, plan.placement, config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class AqornStandard:
    """The settings that make up the standard configuration. `STANDARD` is the paper's."""

    #: The objective's price on a silent logical error, in detected errors: 10 means a silent
    #: error costs 1 + 10 detected ones. Ranks code sizes (`compile_circuit`) and is the
    #: planner's search weight at k = 2.
    silent_weight: float = 10.0
    transversal_intra_two_qubit: bool = True
    asap_order: bool = True
    h_commutation: str = "auto"
    n_code: int = 4
    #: The planner's search weight above k = 2 (n_code > 4), where the transversal in-patch
    #: gadget does not exist, so every silent error avoided is paid for in shots. Chosen by
    #: `experiments/tvd_explore/weight_sweep_k4.py` (31 circuits at [[6,4,2]]): on Fig 6's
    #: machine silent error per kept shot plateaus from 3-5 (10 adds nothing), and on a
    #: balanced machine 10 gives up 38% of the shots for 5%; end to end on Fig 8's corpus
    #: (`exp6_selector/standard_selector.py`) 5 scores best of {1, 5, 10}. None: the same as
    #: `silent_weight`.
    wide_silent_weight: Optional[float] = 5.0
    #: The price of a silent error when `compile_circuit` *ranks* the candidates (code sizes,
    #: planned vs naive placement) on `standard_objective`, separate from the planners'
    #: search weights. Higher leans the choice towards accuracy (fewer silent errors per kept
    #: shot), lower towards shots. None: the same as `silent_weight`.
    selection_weight: Optional[float] = None
    #: Price a solo patch's Hadamards as the whole-patch HH the translator emits for them
    #: (`PackingCost.solo_hh`), in planning and in `standard_objective`. On since 2026-10-09.
    #: Off prices them as single-lane gadgets, as every w10T plan recorded before that date
    #: did (`STANDARD.with_(solo_hh_aware=False)` reproduces those): no different at the
    #: tightest packing of an even width -- no patch is solo -- but it misprices every placement
    #: with a solo patch: odd widths, and any fixed patch count above the tightest (`patches=`).
    solo_hh_aware: bool = True
    #: The planner's final check (`verify`: keep the naive packing when it prices lower) prices
    #: both on `standard_objective` -- the standard translation (layer-sorted, transversal) and
    #: a Clifford stand-in, so a program with rotations is priced on its schedule -- the same
    #: objective `compile_circuit` ranks on. On since 2026-10-10. Off: `plan_silent_aware`'s own
    #: check, which translates in program order and, on any program with rotations, falls back
    #: to gate counts + depth with no noise in it. Measured over Fig 6's 71 circuits x 7
    #: idle/distance ratios (reports/noise_ratio_packers.md): where movement dominates the old
    #: check fell back on plans 1.2-4.3x better in shots and kept plans 2-6x worse, and the
    #: standard objective sided with the frames on nearly every one of those.
    standard_verify: bool = True
    #: Decide H commutation ("auto") on `standard_objective`: plan both arms, price each on the
    #: standard translation and keep the lower, before the final check. On since 2026-10-11.
    #: Off: `plan_silent_aware`'s own decision, which prices the arms like its old check (program
    #: order, and gate counts + depth with no noise on any program with rotations). On the
    #: sweep above it changes 4-13 of 71 decisions per ratio, never significantly worse (Fig 6's
    #: machine: logical error x0.996, TVD x1.018, shots x1.017, all n.s.).
    standard_commutation: bool = True
    #: Random restarts of the planner's descent (`plan_silent_aware(restarts=...)`). 0 (since
    #: 2026-10-11): none -- the planner descends from its seed and from the naive packing and
    #: keeps the better. None: the library default, 4 more descents from shuffled seeds on
    #: programs of at most 16 qubits. Measured over Fig 6's 71 circuits x 7 idle/distance ratios
    #: (`noise_ratio_packers --variants`, reports/compiler_fixes_teddy.md): the restarts change
    #: 5-8 of the 33 small programs' plans per ratio and move logical error and TVD by at most
    #: 0.3% (geometric mean), never significantly.
    restarts: Optional[int] = 0
    #: Renumber the final plan's patches so the busiest sit at the centre of the grid
    #: (`packing.alignment.align_to_grid`), keeping the aligned plan only when it prices lower on
    #: `standard_objective`; the naive fallback is offered aligned too. Off: on the same sweep it
    #: was kept on 23 of 497 plans and changed no metric measurably at any ratio, so patch-to-site
    #: placement is left to the downstream router.
    grid_alignment: bool = False

    def with_(self, **changes) -> "AqornStandard":
        return replace(self, **changes)

    def silent_weight_for(self, n_code: int) -> float:
        """The planner's silent weight at this code size."""
        if n_code > 4 and self.wide_silent_weight is not None:
            return self.wide_silent_weight
        return self.silent_weight

    def ranking_weight(self) -> float:
        """The silent weight `standard_objective` ranks candidates with."""
        return self.silent_weight if self.selection_weight is None else self.selection_weight


STANDARD = AqornStandard()


def standard_noise_profile():
    """Fig 6's planner profile: movement and idling priced as the simulator charges them."""
    from packing.resource.noise import NoiseKnobs, packing_profile

    return packing_profile(NoiseKnobs(), movement_scale=1.0, idle_scale=1.0, distance=True)


def standard_translator_config(n_code: Optional[int] = None, *, commute: bool = True,
                               merging: bool = True, standard: AqornStandard = STANDARD):
    """A `TranslatorConfig` for the standard gadget set and scheduling."""
    from translator.config import TranslatorConfig

    return TranslatorConfig(
        n_code=standard.n_code if n_code is None else n_code,
        enable_h_commutation=commute,
        enable_merging=merging,
        asap_order=standard.asap_order,
        transversal_intra_two_qubit=standard.transversal_intra_two_qubit,
    )


def standard_pipeline_config(n_code: Optional[int] = None, *, commute: bool = True,
                             merging: bool = True, standard: AqornStandard = STANDARD, **kwargs):
    """A `PipelineConfig` whose translator is `standard_translator_config(...)`."""
    from pipeline.config import PipelineConfig

    config = PipelineConfig(verbose=False, save=False, **kwargs)
    config.translator_config = standard_translator_config(
        n_code, commute=commute, merging=merging, standard=standard)
    return config


def plan_standard(program: QuantumCircuit, n_code: Optional[int] = None, *, noise=None,
                  standard: AqornStandard = STANDARD, **kwargs):
    """
    Plan `program` with the standard planner: `plan_silent_aware` at the standard weight for
    this code size (`silent_weight_for`), under the standard gadget set, with Fig 6's noise
    profile unless `noise` is given. Remaining keyword arguments go to `plan_silent_aware`
    (budget, spread, verify, restarts, ...). `patches=P` plans at exactly P patches instead of
    letting the objective pick the count, which it always picks as the tightest.

    With `standard_commutation` an "auto" H-commutation decision plans both arms and keeps the
    one that prices lower on `standard_objective`; `grid_alignment` then offers the plan with
    its patches renumbered by `align_to_grid`; `standard_verify` is the final check against
    the naive packing. Every comparison is on `standard_objective` at the planner's weight.
    """
    from packing.silent import plan_silent_aware

    n_code = standard.n_code if n_code is None else n_code
    noise = noise if noise is not None else standard_noise_profile()
    kwargs.setdefault("solo_hh", standard.solo_hh_aware)
    kwargs.setdefault("restarts", standard.restarts)
    verify = kwargs.pop("verify", True)
    h_commutation = kwargs.pop("h_commutation", standard.h_commutation)
    on_standard = verify and standard.standard_verify

    def plan_under(commutation):
        return plan_silent_aware(
            program,
            n_code,
            silent_weight=standard.silent_weight_for(n_code),
            noise=noise,
            h_commutation=commutation,
            transversal_intra_two_qubit=standard.transversal_intra_two_qubit,
            verify=verify and not on_standard,
            **kwargs,
        )

    known = None
    if h_commutation == "auto" and standard.standard_commutation:
        arms = [plan_under(commute) for commute in (True, False)]
        values = [_standard_price(program, arm.placement, n_code, bool(arm.h_commutation),
                                  noise, standard) for arm in arms]
        best = min(range(len(arms)), key=values.__getitem__)    # a tie keeps commute=True
        plan, known = arms[best], values[best]
        plan.considered = [(f"commute={arm.h_commutation}", value)
                           for arm, value in zip(arms, values)] + list(plan.considered)
        plan.currency = "schedule"
    else:
        plan = plan_under(h_commutation)
    if on_standard or standard.grid_alignment:
        _finish_standard(program, plan, n_code, noise, standard, kwargs.get("patches"),
                         verify=on_standard, known=known)
    return plan


def _standard_price(program: QuantumCircuit, placement, n_code: int, commute: bool, noise,
                    standard: AqornStandard) -> float:
    """
    `standard_objective(...).total` of one placement under the standard translation, at the
    planner's silent weight for this code size -- not the selector's ranking weight: the plan
    handed to `compile_circuit` must not depend on how it will be ranked. A placement that
    cannot be translated or priced costs infinity.
    """
    from pipeline.tasks.translate_permutation import translate_permutation

    placement = [tuple(p) for p in placement]
    pricing = standard.with_(selection_weight=standard.silent_weight_for(n_code))
    config = standard_pipeline_config(n_code, commute=commute, standard=standard)
    try:
        circuit = translate_permutation(program, placement, config=config).circuit
        return standard_objective(program, placement, n_code, circuit, commute=commute,
                                  noise=noise, standard=pricing).total
    except Exception:  # noqa: BLE001 - a rejected or unpriceable placement loses
        return float("inf")


def _finish_standard(program: QuantumCircuit, plan, n_code: int, noise,
                     standard: AqornStandard, patches: Optional[int], *, verify: bool,
                     known: Optional[float] = None) -> None:
    """
    The stages after the search, on `standard_objective`, updating `plan` in place as
    `plan_silent_aware`'s own check does:

        grid alignment  (`grid_alignment`) the plan with its patches renumbered so the busiest
                        sit centrally -- the same groups and lanes, only the grid sites move
        final check     (`verify`, i.e. `standard_verify`) the naive packing, at the plan's
                        patch count when one was fixed, else the tightest; aligned as well
                        when alignment is on

    The lowest-priced candidate is kept, the plan on a tie; an untranslatable plan loses to
    the naive packing.
    """
    from iceberg.utils import logical_capacity
    from packing.alignment import align_to_grid
    from packing.plan import naive_packing

    if plan.patches <= 1:
        return
    k = logical_capacity(n_code)
    tightest = max(1, -(-program.num_qubits // k))
    commute = bool(plan.h_commutation)
    packed = [tuple(p) for p in plan.placement]
    candidates = [("packed", packed, False)]
    if verify:
        candidates.append(("naive", [tuple(p) for p in naive_packing(program.num_qubits, k,
                                                                     patches)], True))
    if standard.grid_alignment:
        for label, placement, naive in list(candidates):
            aligned = [tuple(p) for p in align_to_grid(placement, program)]
            if aligned != placement:
                candidates.append((f"{label} aligned", aligned, naive))
    values = [known if (index == 0 and known is not None)
              else _standard_price(program, placement, n_code, commute, noise, standard)
              for index, (_, placement, _) in enumerate(candidates)]
    best = min(range(len(candidates)), key=values.__getitem__)  # a tie keeps the earlier
    if verify and values[0] == float("inf"):
        best = 1                                                 # untranslatable: the naive
    plan.considered = plan.considered + [(label, value) for (label, _, _), value
                                         in zip(candidates, values)]
    label, placement, naive = candidates[best]
    plan.verified = plan.verified or verify
    plan.aligned = label.endswith("aligned")
    plan.placement = placement
    if naive:
        plan.patches = tightest if patches is None else patches
        plan.physical_qubits = plan.patches * (n_code + 2)
        plan.fell_back = True


@dataclass(frozen=True)
class StandardValue:
    """A translated placement priced on the standard objective: per shot, at alpha = 1."""

    #: `detected + ranking_weight * silent`, lower is better.
    total: float
    #: Expected detectable errors, read off the schedule the simulator will run
    #: (`measured_noise_split`), or the analytic estimate when `currency` says so.
    detected: float
    #: Expected silent logical errors, first order (`packing.silent.silent_errors`).
    silent: float
    #: "schedule", or "analytic" when the circuit could not be scheduled for pricing.
    currency: str = "schedule"


def standard_objective(program: QuantumCircuit, placement, n_code: int, circuit, *,
                       commute: bool, noise=None, params=None,
                       standard: AqornStandard = STANDARD,
                       analytic: Optional[float] = None) -> StandardValue:
    """
    Price one placement of `program` on the standard objective, given its translation.

    `circuit` is the physical circuit the placement translates to under the standard
    settings. Detected errors come from its schedule -- the measurement the planner's own
    verify uses, 7x closer to simulated selection rates than the selector's analytic
    estimate on Fig 8's corpus -- priced on a Clifford stand-in so rotations do not stop it
    (`simulator.attribution.clifford_stand_in`). `noise` is the planner's `NoiseProfile`,
    whose idle and distance scalars set the machine; `params` the `StimNoiseParameters`
    (default: the neutral-atom machine, as the planner prices). `analytic` is the fallback
    detected-error figure if the schedule cannot be read; every arm of one program shares a
    gate set, so a comparison never mixes the two currencies.
    """
    from iceberg.gadget_options import transversal_intra
    from packing.graph import NoiseProfile
    from packing.silent import load_silent_rates, silent_errors
    from simulator.attribution import clifford_stand_in, measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    profile = noise if noise is not None else NoiseProfile()
    currency = "schedule"
    try:
        detected = measured_noise_split(
            clifford_stand_in(circuit), params or StimNoiseParameters.neutral_atom(),
            idle_time_scalar=profile.idle_time_scalar,
            distance_scalar=profile.distance_scalar,
        )["total"]
    except Exception:  # noqa: BLE001 - an unschedulable gate: price analytically instead
        if analytic is None:
            raise
        detected, currency = float(analytic), "analytic"
    transversal = standard.transversal_intra_two_qubit
    with transversal_intra(transversal):
        silent = silent_errors(program, [tuple(p) for p in placement], n_code, commute=commute,
                               noise=profile, rates=load_silent_rates(
                                   n_code, transversal_intra_two_qubit=transversal),
                               solo_hh=standard.solo_hh_aware)
    return StandardValue(total=detected + standard.ranking_weight() * silent, detected=detected,
                         silent=silent, currency=currency)
