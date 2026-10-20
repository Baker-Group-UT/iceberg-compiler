"""
compiler.py — one call: a logical program in, an optimised physical circuit out.

    from packing import compile_circuit

    built = compile_circuit(circuit)
    built.circuit          # the physical circuit, ready to simulate
    print(built.summary())

Until now this package had two entry points and no way to run them together.
`plan_packing` answers *"given a code size and a patch count, which permutation?"*;
`select_resources` answers *"which code size, and how many patches?"*. They sit at
different levels and neither calls the other -- `select_resources` runs its own thinner
placement search rather than delegating down. A caller who wanted both had to choose the
resources with one, feed them to the other by hand, and then decide which of the two
answers to keep. This module is that composition, done once and with the decisions in it
stated rather than implied.

Three stages
------------
1. `select_resources` picks the code size and the patch count, and hands back the best
   placement it found at that shape -- ranked on expected detectable errors per shot.
2. `plan_packing` runs the real placement search at that code size, bounded by the chosen
   patch count (`spread`). This is the stronger search: a multilevel graph seed, a descent
   on the true objective, restarts on small circuits, and its own naive comparison. The
   selector's own placement is an affinity-greedy seed plus one descent.
3. Both placements -- plus the naive packing -- are exactly translated in the mode the
   caller asked for and ranked on the resource objective. The winner's translation *is*
   the returned circuit; nothing is translated twice.

Why stage 3 is not just "return whatever the packer said"
---------------------------------------------------------
Because the two stages minimise different things, and on real circuits they disagree
about the same three placements. `plan_packing` minimises weighted gates plus depth;
`select_resources` minimises expected detectable errors, which prices the space-time
volume and the atom movement a gate count cannot see. Measured on `knn_n25`, at the
`[[8,6,2]]x5` shape the selector chose:

                     weighted gates + depth      expected detectable errors
    planner                        13,304                            2.2870
    naive                          13,584                            2.2892
    selector                       13,700                            2.2759

Nearly opposite orders. The planner's placement is the best of the three on its own
objective and the middle one on the objective the code size was chosen for. On
`ising_n26` at `[[6,4,2]]x7` the gap is wider in the direction that matters: the planner
wins gates-and-depth 6,668 to 6,716 and loses E 1.400 to 1.341, so returning its answer
because it ran last would have cost 4.4% on the objective the resources were chosen for.

So the placements are compared on the objective that chose the shape, and by measurement
rather than by either model -- `resource/select.py` records the packing model ranking
`[[6,4,2]]x7` above a `[[4,2,2]]x13` point with *31% better survival* on `ising_n26`,
because it tallies *summed* per-gate depth where the volume term needs the critical path.

Naive is in that comparison for the reason it is in both stages already: it is what turns
"usually better" into "never worse". Neither stage's own floor carries here -- the
selector measures its naive arm in bare mode with merging on, and the packer scores its
own on weighted gates plus depth -- so a caller asking for fault tolerance, or for merging
off, would be outside both guarantees, and one extra translation restores it. It has not
been observed strictly beating both searched arms, which is the expected outcome of two
stages that each already compare against it; what it is doing is making that hold in the
mode actually returned rather than in the two modes the stages happened to search in.

The patch count is honoured by stage 3, not by stage 2
-----------------------------------------------------
`spread` bounds the planner by the resource decision and, as the model stands, never
binds. Every term in `PackingCost.price` is non-decreasing in the patch count -- the
per-patch `ENCODE + DECODE`, the crossings, the hop distance, and `idle_cost * depth *
patches * (n + 2)` -- except the `busiest` half of the depth estimate, and 234 weighted
units per patch at n=6 is not a thing half a depth term outbids. Measured at k=4 on
`ising_n26`, `knn_n25` and `qft_n18` with `spread` set to the tightest count again: the
descent returns the tightest packing under a zero profile, under the profile derived from
`NoiseKnobs`, and under that profile with idling priced 100x and 10,000x higher. This is
the same wall `resource/select.py` documents from the other side, and why patch count is
an output of the packing objective rather than a decision it makes.

It follows that the two stages only meet at the same shape when the selector also chose
the tightest packing -- which at the default `density_weight=0` it reliably does, by its
own docstring and on every circuit tried here. Above zero the selector spreads, the planner
cannot follow, and the arm comparison keeps the selector's placement because the planner's
is at the wrong shape and pays for it on E. The resource decision survives; the stronger
placement search does not reach it. Closing that needs a density term in `PackingCost`,
not a change here.

Why the noise profile is derived rather than defaulted
-----------------------------------------------------
`plan_packing` takes a `NoiseProfile` whose every cost defaults to zero, so a caller who
hands it nothing gets a planner pricing movement and idling at *nothing* while the
selector above it ranked code sizes on exactly those terms -- the two stages would be
describing different machines. So the `NoiseKnobs` the selector ranked on are converted to
weighted-gate units (`packing.resource.noise.packing_profile`) and the planner is charged
the same hardware. One knobs object drives both stages; `NoiseKnobs.gate_counting()` puts
both back on the paper's convention.

One commutation decision, by default
------------------------------------
`h_commutation=None` measures the decision once with `commutation_pays(program, n_code)`
-- the predicate `select_resources` already used internally, so it is a cache hit -- and
holds it fixed for the planner and for all three arms. That is what makes stage 3 a
ranking of placements rather than of translator settings. `"auto"` hands the decision
back to `plan_packing`, which plans the circuit both ways and keeps the better; the arms
are then translated under their own settings and still priced on their true circuits, so
the comparison remains ground truth -- it just no longer holds the translator fixed.

What this costs
---------------
Stage 1 dominates and always has: one profiling pass per code size, plus an exact
translation for every admissible candidate on the grid (`DEFAULT_MEASURE_TOP = None`,
and that default is load-bearing -- see `resource/select.py`). Stage 2 adds a descent
and its verify translations, stage 3 adds three. Pin `code_sizes` to one element to skip
the code-size question, or set `measure_top` to bound the sweep at the risk that
docstring describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from qiskit import QuantumCircuit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from iceberg.utils import logical_capacity
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation
from translator.core import TranslationResult

from .budget import ResourceBudget
from .cost import DEFAULT_DEPTH_WEIGHT, PackingCost
from .graph import NoiseProfile
from .passes import commutation_pays
from .plan import PackingPlan, naive_packing, plan_packing
from .profiles import extract_profiles
from .resource.noise import ErrorEstimate, NoiseKnobs, estimate, packing_profile
from .resource.select import (
    DEFAULT_CODE_SIZES,
    DEFAULT_MEASURE_TOP,
    Candidate,
    ResourcePlan,
    select_resources,
)

# Private on purpose in `select`, and imported anyway: the final ranking has to be
# computed by the same code that produced the frontier it is ranking against, or the
# three arms and the selector's own candidates would be on two different scales.
from .resource.select import _groups_of, _model_tallies
from .seed import DEFAULT_PARTITIONER
from .tables import mean_patch_hops, translated_view

__all__ = ["CompiledCircuit", "Arm", "compile_circuit"]

Placement = List[Tuple[int, int]]


@dataclass(frozen=True)
class Arm:
    """One candidate placement, translated for real and priced on the resource objective."""

    name: str
    placement: Placement
    n_code: int
    patches: int
    h_commutation: bool
    error: ErrorEstimate
    translation: TranslationResult

    @property
    def circuit(self) -> QuantumCircuit:
        return self.translation.circuit

    def score(self, density_weight: float) -> float:
        return self.error.score(density_weight)


@dataclass
class CompiledCircuit:
    """A physical circuit, the placement behind it, and how it was chosen."""

    circuit: QuantumCircuit
    placement: Placement
    n_code: int
    patches: int
    physical_qubits: int
    #: Expected detectable errors in a shot, for the circuit actually returned.
    error: ErrorEstimate
    #: The commutation setting the returned circuit was built under. A caller who
    #: re-translates `placement` without honouring it gets a different circuit than the
    #: one priced here -- the same trap `PackingPlan.h_commutation` exists for.
    h_commutation: bool
    #: Which arm won: `"packer"`, `"selector"` or `"naive"`.
    source: str
    resources: ResourcePlan
    plan: Optional[PackingPlan] = None
    translation: Optional[TranslationResult] = None
    #: `(arm, resource score)` for every arm that translated, best first. Under the standard
    #: the arm names carry the code size and the score is the standard objective.
    considered: List[Tuple[str, float]] = field(default_factory=list)
    #: Under the standard: the returned circuit's `packing.standard.StandardValue`, and every
    #: priced arm's, keyed like `considered`.
    objective: Optional[object] = None
    objectives: Dict[str, object] = field(default_factory=dict)
    #: Under the standard: every priced arm (placement, code size, commutation setting,
    #: translation) with its value, best first -- the candidates the choice was made among.
    priced: List[Tuple[Arm, object]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return logical_capacity(self.n_code)

    @property
    def patch_density(self) -> float:
        """Logical qubits per patch."""
        return len(self.placement) / self.patches if self.patches else 0.0

    def summary(self) -> str:
        losers = ", ".join(
            f"{name} {score:.3f}" for name, score in self.considered[1:]
        )
        return (
            f"[[{self.n_code},{self.k},2]] x{self.patches} "
            f"= {self.physical_qubits} physical qubits, "
            f"density {self.patch_density:.2f}, {self.error.summary()}; "
            f"kept {self.source}" + (f" over {losers}" if losers else "")
        )


def _translate(
    program: QuantumCircuit,
    placement: Placement,
    n_code: int,
    *,
    commute: bool,
    merging: bool,
    fault_tolerant: bool,
    asap_order: bool = False,
    transversal_intra_two_qubit: bool = False,
) -> Optional[TranslationResult]:
    """The real translation, or `None` if the translator rejects the placement."""
    config = PipelineConfig(verbose=False, save=False, fault_tolerant=fault_tolerant)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    config.translator_config.asap_order = asap_order
    config.translator_config.transversal_intra_two_qubit = transversal_intra_two_qubit
    try:
        return translate_permutation(program, placement, config=config)
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None


def _profiles(program: QuantumCircuit, commute: bool, cache: Dict[bool, Mapping]):
    """Per-qubit profiles of the view the translator will place, memoised per setting."""
    if commute not in cache:
        cache[commute] = extract_profiles(
            translated_view(program, enable_h_commutation=commute)
        )
    return cache[commute]


def _arm(
    program: QuantumCircuit,
    name: str,
    placement: Placement,
    n_code: int,
    *,
    commute: bool,
    merging: bool,
    fault_tolerant: bool,
    noise: NoiseKnobs,
    cache: Dict[bool, Mapping],
    asap_order: bool = False,
    transversal_intra_two_qubit: bool = False,
) -> Optional[Arm]:
    """
    Translate one placement and price it. `None` if the translator rejects it.

    `crossings` comes from the packing model rather than from the translated circuit,
    because it is a logical tally -- one grouped movement per cross-patch two-qubit gate,
    which is what `estimate` charges a round trip for. It is read off the same terms
    `select_resources` used, under the profiles for *this* arm's commutation setting:
    profiles are extracted from the commuted view, so pricing an arm's crossings against
    the other view's profiles would count gates the circuit does not contain.
    """
    translation = _translate(
        program, placement, n_code,
        commute=commute, merging=merging, fault_tolerant=fault_tolerant,
        asap_order=asap_order, transversal_intra_two_qubit=transversal_intra_two_qubit,
    )
    if translation is None:
        return None
    one, two = one_two_qubit_gate_totals(translation.circuit)
    cost = PackingCost(
        profiles=_profiles(program, commute, cache),
        n_code=n_code,
        num_qubits=program.num_qubits,
    )
    crossings, _ = _model_tallies(cost, _groups_of(placement))
    live = len({patch for patch, _ in placement})
    return Arm(
        name=name,
        placement=placement,
        n_code=n_code,
        patches=live,
        h_commutation=commute,
        translation=translation,
        error=estimate(
            noise,
            one_qubit=one,
            two_qubit=two,
            depth=translation.circuit.depth(),
            patches=live,
            n_code=n_code,
            logical_qubits=program.num_qubits,
            crossings=crossings,
            mean_hops=mean_patch_hops(live, n_code + 2),
        ),
    )


def compile_circuit(
    program: QuantumCircuit,
    *,
    budget: Optional[ResourceBudget] = None,
    noise: Optional[NoiseKnobs] = None,
    code_sizes: Sequence[int] = DEFAULT_CODE_SIZES,
    densities: Optional[Sequence[int]] = None,
    measure_top: Optional[int] = DEFAULT_MEASURE_TOP,
    packing_noise: Optional[NoiseProfile] = None,
    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
    spread: Optional[int] = None,
    max_rounds: int = 8,
    restarts: Optional[int] = None,
    partitioner: str = DEFAULT_PARTITIONER,
    h_commutation: Union[bool, str, None] = None,
    merging: bool = True,
    fault_tolerant: bool = False,
    verify: bool = True,
    planner: str = "standard",
    standard=None,
    pricing_params=None,
) -> CompiledCircuit:
    """
    Choose the resources, pack for them, and return the physical circuit.

    `planner="standard"` (the default since 2026-10-07) is the paper's configuration,
    `packing.standard.STANDARD` unless `standard` is given, and optimises one objective in
    every stage (`_compile_standard`): each code size in `code_sizes` is planned with the
    standard planner (silent-aware, transversal in-patch CZ / XCX at k = 2, layer-sorted),
    the planned placement and the naive one are translated and priced on
    `packing.standard.standard_objective` -- detected errors read off the schedule plus
    `ranking_weight()` x silent errors -- and the lowest wins. Each code size is packed at the
    planner's own (tightest) patch count, so `densities`, `measure_top` and `spread` do not
    apply. `pricing_params` (a `StimNoiseParameters`) sets the machine the schedule is priced
    on; default the neutral-atom machine, with the idle and distance scalars of the
    planner's noise profile.

    `planner="v4"` is the compiler as submitted: `select_resources` for the shape on the
    analytic error estimate, `plan_packing` at that shape, the arms ranked on that estimate,
    the original gadgets, the original emission order.

    `noise` is the one machine description: it ranks the resource candidates directly and,
    converted to weighted-gate units, tells the placement search what movement and idling
    cost. `budget` caps both stages. `code_sizes` pinned to a single element skips the
    code-size question and leaves the patch-count one.

    `spread` overrides how many patch slots above the tightest packing the placement
    search may migrate into; the default is however many the resource stage chose. That
    bound does not currently bind -- see the module docstring -- so it is plumbing for a
    packing objective that can spread, not a knob with a measured effect today.

    `fault_tolerant` and `merging` apply to the returned circuit and to the arm
    comparison that picks it. Both search stages rank in bare mode with merging on --
    `select_resources` has no fault-tolerant path -- so with `fault_tolerant=True` the
    *shape* was chosen on bare-mode measurements even though the circuit and the final
    ranking are fault-tolerant. `estimate` is also slightly off there: its intra-patch
    term nets out four physical gates per crossing, which is the bare-mode figure.

    Always returns something placeable, falling back to the naive packing at the chosen
    code size if both searched placements are rejected. Raises `ValueError` only when
    even that will not translate, which means no placement at that code size will.
    """
    from iceberg.gadget_options import transversal_intra

    if planner not in ("standard", "v4"):
        raise ValueError(f"planner must be 'standard' or 'v4', not {planner!r}")
    if planner == "v4":
        return _compile(
            program, budget=budget, noise=noise, code_sizes=code_sizes, densities=densities,
            measure_top=measure_top, packing_noise=packing_noise, depth_weight=depth_weight,
            spread=spread, max_rounds=max_rounds, restarts=restarts, partitioner=partitioner,
            h_commutation=h_commutation, merging=merging, fault_tolerant=fault_tolerant,
            verify=verify,
        )
    from .standard import STANDARD

    standard = standard or STANDARD
    with transversal_intra(standard.transversal_intra_two_qubit):
        return _compile_standard(
            program, budget=budget, noise=noise, code_sizes=code_sizes,
            packing_noise=packing_noise, depth_weight=depth_weight, max_rounds=max_rounds,
            restarts=restarts, partitioner=partitioner, h_commutation=h_commutation,
            merging=merging, fault_tolerant=fault_tolerant, verify=verify, standard=standard,
            pricing_params=pricing_params,
        )


def _compile_standard(program, *, budget, noise, code_sizes, packing_noise, depth_weight,
                      max_rounds, restarts, partitioner, h_commutation, merging, fault_tolerant,
                      verify, standard, pricing_params) -> CompiledCircuit:
    """
    The standard compiler: plan every code size, price planned and naive on one objective.

    Under the submitted compiler the three stages optimise different things -- the shape on
    the analytic estimate of detected errors, the placement on weighted gates and depth (or,
    under w10T, plus silent errors), the final pick on the analytic estimate again -- and with
    the silent-aware planner that split decided against the planner: on Fig 8's corpus the
    final pick kept the selector's own placement over the w10T plan on 11 of 13 circuits,
    for 0.86x the shots of the submitted compiler and the same silent error. Here the shape,
    the placement and the pick share `standard_objective`, and the shape is judged on the
    placement that will actually be built rather than on a greedy seed.
    """
    from .standard import plan_standard, standard_objective

    noise = noise or NoiseKnobs()
    budget = budget or ResourceBudget()
    profile = packing_noise or packing_profile(noise)
    setting = standard.h_commutation if h_commutation is None else h_commutation
    num_qubits = program.num_qubits
    cache: Dict[bool, Mapping] = {}

    scored: List[Tuple[float, str, Arm, object, PackingPlan]] = []
    plans: Dict[int, PackingPlan] = {}
    for n_code in sorted(code_sizes):
        k = logical_capacity(n_code)
        ceiling = budget.patch_ceiling(n_code)
        if ceiling is not None and -(-num_qubits // k) > ceiling:
            continue  # this code cannot hold the program within the budget
        try:
            plan = plan_standard(
                program, n_code, noise=profile, standard=standard, budget=budget,
                depth_weight=depth_weight, max_rounds=max_rounds, restarts=restarts,
                partitioner=partitioner, h_commutation=setting, verify=verify,
            )
        except Exception:  # noqa: BLE001 - a code size the planner cannot serve is skipped
            continue
        plans[n_code] = plan
        entries = [("packer", plan.placement, bool(plan.h_commutation))]
        if verify:
            entries.append(("naive", naive_packing(num_qubits, k), bool(plan.h_commutation)))
        for name, placement, commute in entries:
            arm = _arm(program, name, placement, n_code, commute=commute, merging=merging,
                       fault_tolerant=fault_tolerant, noise=noise, cache=cache,
                       asap_order=standard.asap_order,
                       transversal_intra_two_qubit=standard.transversal_intra_two_qubit)
            if arm is None:
                continue
            value = standard_objective(program, placement, n_code, arm.circuit, commute=commute,
                                       noise=profile, params=pricing_params, standard=standard,
                                       analytic=arm.error.total)
            scored.append((value.total, name, arm, value, plan))
    if not scored:
        raise ValueError("no code size in `code_sizes` admitted a placement under the budget")

    scored.sort(key=lambda entry: entry[0])
    _, name, best, value, plan = scored[0]
    frontier = [
        Candidate(n_code=arm.n_code, patches=arm.patches, placement=arm.placement,
                  error=arm.error, measured=True)
        for _, _, arm, _, _ in scored
    ]
    resources = ResourcePlan(placement=best.placement, n_code=best.n_code, patches=best.patches,
                             physical_qubits=best.error.physical_qubits, error=best.error,
                             frontier=frontier)
    return CompiledCircuit(
        circuit=best.circuit,
        placement=best.placement,
        n_code=best.n_code,
        patches=best.patches,
        physical_qubits=best.error.physical_qubits,
        error=best.error,
        h_commutation=best.h_commutation,
        source=name,
        resources=resources,
        plan=plan,
        translation=best.translation,
        considered=[(f"{entry_name} n={arm.n_code}", total)
                    for total, entry_name, arm, _, _ in scored],
        objective=value,
        objectives={f"{entry_name} n={arm.n_code}": entry_value
                    for _, entry_name, arm, entry_value, _ in scored},
        priced=[(arm, entry_value) for _, _, arm, entry_value, _ in scored],
    )


def _compile(program, *, budget, noise, code_sizes, densities, measure_top, packing_noise,
             depth_weight, spread, max_rounds, restarts, partitioner, h_commutation, merging,
             fault_tolerant, verify) -> CompiledCircuit:
    """The submitted compiler (`planner="v4"`), unchanged."""
    noise = noise or NoiseKnobs()
    budget = budget or ResourceBudget()

    resources = select_resources(
        program,
        budget=budget,
        noise=noise,
        code_sizes=code_sizes,
        densities=densities,
        measure_top=measure_top,
    )

    n_code = resources.n_code
    k = logical_capacity(n_code)
    tightest = max(1, -(-program.num_qubits // k))
    if spread is None:
        spread = max(0, resources.patches - tightest)

    # The selector measured this per code size and cached it by circuit content, so
    # asking again for the code size it chose costs nothing.
    setting = commutation_pays(program, n_code) if h_commutation is None else h_commutation

    search = dict(
        budget=budget,
        noise=packing_noise or packing_profile(noise),
        depth_weight=depth_weight,
        spread=spread,
        max_rounds=max_rounds,
        restarts=restarts,
        partitioner=partitioner,
        h_commutation=setting,
        verify=verify,
    )
    plan = plan_packing(program, n_code, **search)

    # A fixed setting -- measured or forced -- holds for every arm, which is what makes
    # the ranking a ranking of placements. Under "auto" the planner's arm keeps whatever
    # it decided and the other two stay on the selector's own decision, the setting they
    # were ranked under; each arm is still priced on the circuit it actually produces.
    baseline = (
        commutation_pays(program, n_code) if setting == "auto" else bool(setting)
    )
    candidates = [
        ("packer", plan.placement, plan.h_commutation),
        ("selector", resources.placement, baseline),
    ]
    if verify:
        candidates.append(("naive", naive_packing(program.num_qubits, k), baseline))

    cache: Dict[bool, Mapping] = {}

    def priced(entries) -> List[Arm]:
        return [
            arm for arm in (
                _arm(program, name, placement, n_code, commute=commute, merging=merging,
                     fault_tolerant=fault_tolerant, noise=noise, cache=cache)
                for name, placement, commute in entries
            )
            if arm is not None
        ]

    arms = priced(candidates)
    if not arms and not verify:
        # `verify=False` bought one fewer translation, not a promise to return nothing.
        # Both searched placements being rejected is a translator verdict on them, not on
        # the code size, so naive gets its chance here even though it was not competing.
        arms = priced([("naive", naive_packing(program.num_qubits, k), baseline)])
    if not arms:
        raise ValueError(
            f"every candidate placement was rejected at [[{n_code},{k},2]]"
            + (" under fault tolerance" if fault_tolerant else "")
        )

    arms.sort(key=lambda arm: arm.score(noise.density_weight))
    best = arms[0]
    return CompiledCircuit(
        circuit=best.circuit,
        placement=best.placement,
        n_code=best.n_code,
        patches=best.patches,
        physical_qubits=best.error.physical_qubits,
        error=best.error,
        h_commutation=best.h_commutation,
        source=best.name,
        resources=resources,
        plan=plan,
        translation=best.translation,
        considered=[(arm.name, arm.score(noise.density_weight)) for arm in arms],
    )
