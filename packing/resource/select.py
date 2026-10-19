"""
select.py — choose the code size and the patch density, then pack for them.

`packing.plan_packing` answers "given a code size and a patch count, which
permutation?". It cannot answer "which code size, and how many patches?", and not by
oversight: three mechanisms inside it are each independently monotone toward the
tightest packing.

    `pack_bins` scores affinity first, and an empty bin scores zero, so any occupied
    bin with a positive edge beats it and spare bins stay empty

    `refine` minimises cut weight, and spreading raises the cut, so it actively
    undoes any spread

    `descend` charges ENCODE + DECODE per live patch with no counterweight, because
    the movement and idle coefficients default to zero

Each is right for its own objective. Together they mean `spread=8` produced an
identical patch count on every circuit tried. So the resource decision belongs above
the packer, and it has to *drive* it -- the best partition at density 2.6 is not the
best partition at 3.7 with something split off, so every candidate is a fresh pack.

Which is affordable for one reason: `translated_view` and `extract_profiles` dominate
the cost and are independent of both knobs, so they are hoisted out and the whole
two-dimensional sweep is one profiling pass plus a re-pack per candidate.

The model generates candidates; it does not rank them
-----------------------------------------------------
The first version of this ranked on the packing model and translated only the top few.
That is wrong, and measurably so: on `ising_n26` it returned [[6,4,2]] x7 when the best
admissible point is [[4,2,2]] x13, which has **31% better survival**.

The cause is that `PackingCost` tallies *summed per-gate* depth, and the volume term
needs the critical path. Summed depth grows with the number of cross-patch gates, so
it systematically penalises spread placements -- precisely the ones that win here,
because spreading is what buys parallelism. It is wrong in both directions in absolute
terms too (`qft_n18` 918 against a true 768; `knn_n25` 336 against 380).

So every admissible candidate is translated. The grid is small by construction -- five
code sizes by about five patch counts -- and this runs once per circuit, not per
placement. A critical-path proxy in `PackingCost` (per-lane depth vectors reduced with
max rather than sum, still O(k) per move) would let the model pre-rank and shrink this
to a handful of translations; until that exists, measuring is the only honest option.
`measure_top` caps the translations if a caller needs the sweep bounded, at the risk
just described.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from qiskit import QuantumCircuit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from ..tables import mean_patch_hops, translated_view
from ..graph import build_graph
from ..budget import ResourceBudget
from ..profiles import extract_profiles
from iceberg.utils import logical_capacity
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

from ..cost import PackingCost
from ..lanes import assign_lanes
from ..plan import naive_packing
from ..search import Packing, descend
from ..passes import commutation_pays
from .noise import ErrorEstimate, NoiseKnobs, estimate

__all__ = ["ResourcePlan", "Candidate", "select_resources"]

Placement = List[Tuple[int, int]]
Group = FrozenSet[int]

#: Code sizes swept by default: the [[n, n-2, 2]] family from k=2 to k=10.
DEFAULT_CODE_SIZES: Tuple[int, ...] = (4, 6, 8, 10, 12)

#: How many candidates get an exact translation. `None` means all of them, which is
#: the only setting that reliably picks the best point -- see the module docstring.
DEFAULT_MEASURE_TOP: Optional[int] = None


@dataclass(frozen=True)
class Candidate:
    """One (code size, patch count) point, and what it is worth."""

    n_code: int
    patches: int
    placement: Placement
    error: ErrorEstimate
    measured: bool
    #: Cross-patch two-qubit gates. Carried from the model pass so the measured pass
    #: does not have to recount them; only depth changes between the two.
    crossings: int = 0

    @property
    def score_key(self) -> Tuple[int, int]:
        return (self.n_code, self.patches)


@dataclass
class ResourcePlan:
    """The chosen permutation, and the frontier it was chosen from."""

    placement: Placement
    n_code: int
    patches: int
    physical_qubits: int
    error: ErrorEstimate
    #: Every admissible candidate, measured ones first, best score first.
    frontier: List[Candidate] = field(default_factory=list)

    @property
    def k(self) -> int:
        return logical_capacity(self.n_code)

    @property
    def patch_density(self) -> float:
        """Logical qubits per patch."""
        return len(self.placement) / self.patches if self.patches else 0.0

    def summary(self) -> str:
        return (
            f"[[{self.n_code},{self.k},2]] x{self.patches} "
            f"= {self.physical_qubits} physical qubits, "
            f"density {self.patch_density:.2f}, {self.error.summary()}"
        )


def _forced_groups(
    num_qubits: int, weights, capacity: int, patches: int
) -> List[Group]:
    """
    Partition into *exactly* `patches` occupied groups, affinity-greedy.

    A balance cap of `ceil(m/patches)` is the whole trick: without a floor on
    occupancy the affinity-greedy assignment piles everything into the first bins and
    the patch count silently collapses back to tightest, which is what happens inside
    `pack_bins` today. Seeds are the highest-degree qubits, so the busiest qubits get
    separate patches rather than landing together by accident of order.
    """
    def weight(a: int, b: int) -> float:
        return weights.get((a, b) if a < b else (b, a), 0.0)

    cap = min(capacity, max(1, -(-num_qubits // patches)))
    bins: List[List[int]] = [[] for _ in range(patches)]
    order = sorted(
        range(num_qubits),
        key=lambda q: -sum(abs(weight(q, other)) for other in range(num_qubits)),
    )
    for index, qubit in enumerate(order[:patches]):
        bins[index].append(qubit)
    for qubit in order[patches:]:
        room = [i for i in range(patches) if len(bins[i]) < cap]
        if not room:
            room = [i for i in range(patches) if len(bins[i]) < capacity]
        if not room:
            raise ValueError(f"{num_qubits} qubits do not fit in {patches} patches")
        best = max(room, key=lambda i: sum(weight(qubit, other) for other in bins[i]))
        bins[best].append(qubit)
    return [frozenset(b) for b in bins if b]


def _measure(
    program: QuantumCircuit, placement: Placement, n_code: int, *, commute: bool = True
):
    """Exact translation. Returns (1q, 2q, depth) or None if the placement is rejected."""
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = True
    try:
        translated = translate_permutation(program, placement, config=config)
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    one, two = one_two_qubit_gate_totals(translated.circuit)
    return one, two, translated.circuit.depth()


def _densities(num_qubits: int, k: int, ceiling: Optional[int]) -> List[int]:
    """
    Patch counts worth trying, from tightest to one qubit per patch.

    Sampled rather than exhaustive: the two objectives are smooth in P -- accepted
    shots fall, patch density improves -- so a handful of points finds the knee, and
    every extra point costs a pack plus possibly a translation.
    """
    tightest = max(1, -(-num_qubits // k))
    loosest = num_qubits if ceiling is None else min(num_qubits, ceiling)
    if loosest <= tightest:
        return [tightest]
    span = loosest - tightest
    picks = {tightest, loosest}
    for fraction in (0.25, 0.5, 0.75):
        picks.add(tightest + max(1, round(span * fraction)))
    return sorted(p for p in picks if tightest <= p <= loosest)


def select_resources(
    program: QuantumCircuit,
    *,
    budget: Optional[ResourceBudget] = None,
    noise: Optional[NoiseKnobs] = None,
    code_sizes: Sequence[int] = DEFAULT_CODE_SIZES,
    densities: Optional[Sequence[int]] = None,
    measure_top: Optional[int] = DEFAULT_MEASURE_TOP,
) -> ResourcePlan:
    """
    Pick the code size, the patch count, and the permutation, under a budget.

    Sweeps `code_sizes` x patch counts, scores each on expected detectable errors plus
    `noise.density_weight` times expected corrupted logical qubits, translates the best
    few for real, and returns the winner. With the default `density_weight=0` this
    optimises accepted-shot rate alone and will reliably choose a tight packing; that
    is the correct answer to the question as posed, not a failure to explore.

    Falls back to the naive packing for any candidate the translator rejects, and to
    the tightest packing at the smallest admissible code size if every candidate is
    rejected, so this always returns something placeable.
    """
    noise = noise or NoiseKnobs()
    budget = budget or ResourceBudget()
    num_qubits = program.num_qubits

    # The one expensive step, and it depends on neither knob.
    # One decision per code size, shared by the profile and every translation below.
    commuting = {n: commutation_pays(program, n) for n in sorted(code_sizes)}
    profiles_for = {
        n: extract_profiles(translated_view(program, enable_h_commutation=commuting[n]))
        for n in sorted(code_sizes)
    }

    scored: List[Candidate] = []
    for n_code in sorted(code_sizes):
        k = logical_capacity(n_code)
        tightest = max(1, -(-num_qubits // k))
        ceiling = budget.patch_ceiling(n_code)
        if ceiling is not None and tightest > ceiling:
            continue  # this code cannot hold the program within the budget

        profiles = profiles_for[n_code]
        graph = build_graph(profiles, n_code, num_qubits=num_qubits)
        cost = PackingCost(profiles=profiles, n_code=n_code, num_qubits=num_qubits)
        counts = list(densities) if densities is not None else _densities(
            num_qubits, k, ceiling
        )
        for patches in counts:
            if patches < tightest or (ceiling is not None and patches > ceiling):
                continue
            try:
                groups = _forced_groups(num_qubits, graph.weights, k, patches)
            except ValueError:
                continue
            placement = assign_lanes(groups, cost)
            live = len({p for p, _ in placement})
            crossings, depth = _model_tallies(cost, groups)
            scored.append(
                Candidate(
                    n_code=n_code,
                    patches=live,
                    placement=placement,
                    crossings=crossings,
                    error=estimate(
                        noise,
                        one_qubit=0,
                        two_qubit=0,
                        depth=depth,
                        patches=live,
                        n_code=n_code,
                        logical_qubits=num_qubits,
                        crossings=crossings,
                        mean_hops=mean_patch_hops(live, n_code + 2),
                    ),
                    measured=False,
                )
            )

    if not scored:
        return _fallback(program, budget, noise, code_sizes)

    scored.sort(key=lambda c: c.error.score(noise.density_weight))
    final: List[Candidate] = []
    shortlist = scored if measure_top is None else scored[: max(1, measure_top)]
    for candidate in shortlist:
        metrics = _measure(program, candidate.placement, candidate.n_code,
                           commute=commuting[candidate.n_code])
        if metrics is None:
            continue
        one, two, depth = metrics
        final.append(
            Candidate(
                n_code=candidate.n_code,
                patches=candidate.patches,
                placement=candidate.placement,
                crossings=candidate.crossings,
                error=estimate(
                    noise,
                    one_qubit=one,
                    two_qubit=two,
                    depth=depth,
                    patches=candidate.patches,
                    n_code=candidate.n_code,
                    logical_qubits=num_qubits,
                    crossings=candidate.crossings,
                    mean_hops=mean_patch_hops(candidate.patches, candidate.n_code + 2),
                ),
                measured=True,
            )
        )

    if not final:
        return _fallback(program, budget, noise, code_sizes)

    final.sort(key=lambda c: c.error.score(noise.density_weight))
    best = _refine_and_verify(
        program, final[0], noise, num_qubits,
        profiles=profiles_for[final[0].n_code],
        commute=commuting[final[0].n_code],
    )
    return ResourcePlan(
        placement=best.placement,
        n_code=best.n_code,
        patches=best.patches,
        physical_qubits=best.error.physical_qubits,
        error=best.error,
        frontier=final + [c for c in scored if c.score_key not in
                          {f.score_key for f in final}],
    )


def _groups_of(placement: Placement) -> List[Group]:
    """Patch membership, recovered from a placement."""
    buckets: Dict[int, Set[int]] = {}
    for qubit, (patch, _) in enumerate(placement):
        buckets.setdefault(patch, set()).add(qubit)
    return [frozenset(buckets[patch]) for patch in sorted(buckets)]


def _score_placement(
    program: QuantumCircuit,
    placement: Placement,
    n_code: int,
    noise: NoiseKnobs,
    num_qubits: int,
    cost: PackingCost,
    *,
    commute: bool,
) -> Optional[Candidate]:
    """Translate a placement for real and price it on the resource objective."""
    metrics = _measure(program, placement, n_code, commute=commute)
    if metrics is None:
        return None
    one, two, depth = metrics
    live = len({patch for patch, _ in placement})
    crossings, _ = _model_tallies(cost, _groups_of(placement))
    return Candidate(
        n_code=n_code,
        patches=live,
        placement=placement,
        crossings=crossings,
        error=estimate(
            noise, one_qubit=one, two_qubit=two, depth=depth, patches=live,
            n_code=n_code, logical_qubits=num_qubits, crossings=crossings,
            mean_hops=mean_patch_hops(live, n_code + 2),
        ),
        measured=True,
    )


def _refine_and_verify(
    program: QuantumCircuit,
    best: Candidate,
    noise: NoiseKnobs,
    num_qubits: int,
    *,
    profiles,
    commute: bool,
) -> Candidate:
    """
    Descend on the winner, and check it against naive. Return whichever scores best.

    Every candidate above is a greedy seed plus lane assignment -- no local search, and
    no comparison against sequential. The *ranking* was honestly measured, but the thing
    returned had skipped both of the steps that make a packing safe to present as a
    gain, and it showed: on QASMBench small at the chosen shape the returned placement
    was worse than naive on 4 of 8 circuits (qpe_n9 +2.5%, simon_n6 +7.5%, adder_n10
    +2.6%, hhl_n7 +3.3%).

    Two arms are added here. `descend` optimises `PackingCost` -- weighted gates plus
    depth -- which is deliberately *not* this module's objective; it proposes, and the
    resource score decides. Naive is the floor, and including it is what turns "usually
    better" into "never worse". All three arms are exactly translated under the same
    commutation decision, so the comparison is like for like.
    """
    cost = PackingCost(profiles=profiles, n_code=best.n_code, num_qubits=num_qubits)
    arms: List[Candidate] = [best]

    packing = Packing(cost, [set(group) for group in _groups_of(best.placement)])
    try:
        descend(packing)
        descended = _score_placement(
            program, assign_lanes(packing.partition(), cost), best.n_code,
            noise, num_qubits, cost, commute=commute,
        )
    except Exception:  # noqa: BLE001 - a failed descent still leaves the other arms
        descended = None
    if descended is not None:
        arms.append(descended)

    naive = _score_placement(
        program, naive_packing(num_qubits, logical_capacity(best.n_code)),
        best.n_code, noise, num_qubits, cost, commute=commute,
    )
    if naive is not None:
        arms.append(naive)

    arms.sort(key=lambda candidate: candidate.error.score(noise.density_weight))
    return arms[0]


def _model_tallies(cost: PackingCost, groups: Sequence[Group]) -> Tuple[int, int]:
    """
    Cross-patch gate count and summed per-gate depth, straight from the cost terms.

    The depth here is a *sum*, not a critical path, which is why the top candidates get
    translated before the final ranking -- see the module docstring.
    """
    crossings = depth = 0
    for group in groups:
        depth += int(cost.unary(frozenset(group)).depth)
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            term = cost.binary(frozenset(left), frozenset(right))
            crossings += int(term.crossings)
            depth += int(term.depth)
    return crossings, depth


def _fallback(
    program: QuantumCircuit,
    budget: ResourceBudget,
    noise: NoiseKnobs,
    code_sizes: Sequence[int],
) -> ResourcePlan:
    """Tightest naive packing at the smallest admissible code, exactly measured."""
    for n_code in sorted(code_sizes):
        k = logical_capacity(n_code)
        patches = max(1, -(-program.num_qubits // k))
        ceiling = budget.patch_ceiling(n_code)
        if ceiling is not None and patches > ceiling:
            continue
        placement = naive_packing(program.num_qubits, k)
        # The per-code decision, not `_measure`'s default: pricing the fallback with
        # commutation forced on describes a circuit the pipeline would not build.
        metrics = _measure(program, placement, n_code,
                           commute=commutation_pays(program, n_code))
        if metrics is None:
            continue
        one, two, depth = metrics
        error = estimate(
            noise, one_qubit=one, two_qubit=two, depth=depth, patches=patches,
            n_code=n_code, logical_qubits=program.num_qubits,
        )
        return ResourcePlan(
            placement=placement, n_code=n_code, patches=patches,
            physical_qubits=error.physical_qubits, error=error,
        )
    raise ValueError("no code size in `code_sizes` admits this program under the budget")
