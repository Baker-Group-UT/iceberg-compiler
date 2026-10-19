"""
noise.py — what the hardware charges, and what a placement is worth under it.

The packing objective in `packing.cost` minimises weighted gates and depth. That is
the right thing to optimise once the resources are fixed, and it cannot choose the
resources, because it is nearly flat in the two knobs that matter: on `ising_n26`
gate counts move 6% across k = 2..10 while the space-time volume moves 30%.

This module supplies the missing scale. Everything here is one quantity -- the
expected number of *detectable* errors in a shot:

    E = n1*p1 + n2*p2                          gates
      + V * t_layer * r,   V = P*(n+2)*depth   idling through the gate schedule
      + M * (moved*p_move + trip*P*(n+2)*r)    movement: M grouped round trips
      + I * (p_move + trip_in*P*(n+2)*r)       intra-patch beyond-blockade carries

where `r` is the twirled thermal error rate the simulator itself applies
(`stim_backend.thermal_error_rate`, ~1/(2*T2) + 1/(4*T1)). Survival under
post-selection is `exp(-E)`, so shots needed is `exp(E)`; minimising E maximises
accepted shots.

Movement is priced the way `simulator.transport` charges it -- **per grouped
movement, not per physical gate**: one cross-patch logical gate is one round trip of
its control side (two atoms), costing `move_error` per carried atom plus the idling
the trip forces on every atom in the array. The old form (`movement_error` per
crossing per hop, an opaque coefficient defaulting to zero) priced four trips per
logical CNOT and n per macro, and disagreed with the clamped copy the placement
search used. The intra-patch term is the transport model's blockade threshold showing
up at resource-selection time: the fraction of a patch's pairs beyond the blockade
radius is fixed by the code size, so it belongs exactly here, where code sizes are
ranked -- the placement search cannot change it and never sees it.

Two objectives, not one
-----------------------
Fewer, fuller patches win on E: less volume, fewer cross-patch gates, less
encode/decode. More, emptier patches win on *patch density* -- more detectors per
logical qubit and fewer logical qubits behind each one, so an error that does slip
through corrupts less. Measured on `ising_n26` at k=4, going from 7 patches to 26
costs 12x the accepted shots and buys 4x fewer corrupted logical qubits.

There is no objective-free way to combine those. `density_weight` is the exchange
rate, and it is a property of the application rather than the machine: zero means a
discarded shot is the only thing you care about, large means a wrong answer is much
worse than a retry.

Trust boundaries, stated because they are close
-----------------------------------------------
`exp(-E)` is a Poisson approximation that assumes every error is independent and
detectable. At the measured E of 3-5 on QASMBench proxies it is an ordering, not a
rate. Worse, `corrupted_qubits` uses `E^2/(2P)` for the weight-2 failures a distance-2
code misses, and that expansion is only valid for E << 1. Both are here to *rank*
candidates; `iceberg_postselect` returns a real selection rate, and calibrating
against it is the way to make either number quotable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import cycle, and only the annotation needs it
    from ..graph import NoiseProfile

__all__ = [
    "NoiseKnobs",
    "ErrorEstimate",
    "estimate",
    "intra_patch_movement",
    "packing_profile",
]


def _neutral_atom_defaults() -> Dict[str, float]:
    """The simulator's own numbers, imported so these cannot drift."""
    from simulator.noise_models import BASE_MOVE_ERROR, NEUTRAL_ATOM

    return {**NEUTRAL_ATOM, "move_error": BASE_MOVE_ERROR}


_NA = _neutral_atom_defaults()


@dataclass(frozen=True)
class NoiseKnobs:
    """
    Hardware error rates and the one application-level preference.

    Defaults are `simulator.noise_models.NEUTRAL_ATOM` verbatim -- the machine this
    repo simulates. They were the superconducting calibration (p2=3.5e-3, T2=341us,
    88ns layers) long after the simulator moved to neutral atoms, so the selector was
    ranking code sizes for a machine nothing downstream measured on.
    """

    #: Per single-qubit gate.
    one_qubit_error: float = _NA["one_qubit_error"]
    #: Per two-qubit gate.
    two_qubit_error: float = _NA["two_qubit_error"]
    #: Seconds per circuit layer. On this platform the *single*-qubit gate is the
    #: binding one -- a 1 us Raman pulse against a 0.5 us Rydberg gate.
    layer_time: float = _NA["one_qubit_gate_time"]
    #: Relaxation and dephasing times. Both matter: the idle rate is the twirled
    #: thermal rate the simulator applies, not 1/T2.
    t1: float = _NA["T1"]
    t2: float = _NA["T2"]
    #: How many discarded shots one corrupted logical qubit is worth. Zero optimises
    #: accepted-shot rate alone and will always choose the tightest packing; raise it
    #: to buy patch density with throughput. Not a machine property -- an application
    #: one, so there is no defensible default other than "off".
    density_weight: float = 0.0

    # --- movement, in the transport model's own terms ---
    #: Infidelity per atom per carriage (pick-up/put-down). `ScaledNoiseModel`
    #: scales this with alpha like every other error rate.
    move_error: float = _NA["move_error"]
    #: Atoms carried per grouped cross-patch movement: the control side of a
    #: targeted logical gate (lane + anchor).
    moved_per_crossing: float = 2.0
    #: Geometry, matching `simulator.transport.Geometry` defaults.
    atom_pitch: float = 3e-6
    movement_speed: float = 1.0
    blockade_radius: float = 8e-6

    @property
    def idle_rate(self) -> float:
        """Error probability per idle second -- the simulator's twirled thermal rate."""
        from simulator.stim_backend import thermal_error_rate

        return thermal_error_rate(self.t1, self.t2)

    @property
    def idle_per_qubit_layer(self) -> float:
        """Dephasing probability for one physical qubit sitting out one layer."""
        return self.layer_time * self.idle_rate

    def round_trip_error_seconds(self, cells: float) -> float:
        """Seconds of there-and-back flight for a carry of `cells` Manhattan cells."""
        if self.movement_speed <= 0:
            return 0.0
        return 2.0 * cells * self.atom_pitch / self.movement_speed

    @classmethod
    def gate_counting(cls) -> "NoiseKnobs":
        """The paper's convention: only gate error, no idling, no movement."""
        return cls(layer_time=0.0, density_weight=0.0, move_error=0.0,
                   movement_speed=0.0)


#: (fraction of intra-patch pairs beyond blockade, mean Manhattan cells among them),
#: cached by (atoms, pitch, blockade). The geometry is `most_square_rectangle` filled
#: row-major -- `grid_constructor` verbatim.
_INTRA: Dict[Tuple[int, float, float], Tuple[float, float]] = {}


def intra_patch_movement(
    qubits_per_patch: int, *, pitch: float = 3e-6, blockade: float = 8e-6
) -> Tuple[float, float]:
    """
    How much of a patch's own two-qubit work needs a carry, and how far.

    Returns `(fraction beyond blockade, mean Manhattan cells among those pairs)` for
    a patch of `qubits_per_patch` atoms. This is the mechanism that keeps large
    patches honest -- 0.00 at k=2 rising to ~0.27 at k=10 at the default pitch -- and
    it is a property of the code size alone, which is why it is priced here rather
    than in the placement objective.
    """
    key = (qubits_per_patch, pitch, blockade)
    cached = _INTRA.get(key)
    if cached is not None:
        return cached

    from simulator.transport import most_square_rectangle

    rows, columns = most_square_rectangle(qubits_per_patch)
    cells = [
        (offset % columns, offset // columns) for offset in range(qubits_per_patch)
    ]
    beyond = []
    pairs = 0
    for a, b in itertools.combinations(cells, 2):
        pairs += 1
        if math.hypot(a[0] - b[0], a[1] - b[1]) * pitch > blockade:
            beyond.append(abs(a[0] - b[0]) + abs(a[1] - b[1]))
    result = (
        (len(beyond) / pairs, sum(beyond) / len(beyond)) if beyond and pairs else (0.0, 0.0)
    )
    _INTRA[key] = result
    return result


@dataclass(frozen=True)
class ErrorEstimate:
    """
    Expected detectable errors in one shot, split by where they come from.

    Attribution follows `measured_noise_split`: `movement` is distance-based noise
    only (the carriage handling on the atoms actually moved); the idling a trip
    forces on the rest of the array is idling, and lands in `idle`.
    """

    gate: float
    idle: float
    movement: float
    #: Expected logical qubits corrupted per *accepted* shot. See the module note --
    #: this one is the least trustworthy number here.
    corrupted: float
    patches: int
    physical_qubits: int
    depth: int

    @property
    def total(self) -> float:
        return self.gate + self.idle + self.movement

    @property
    def survival(self) -> float:
        """Fraction of shots that clear post-selection."""
        return math.exp(-self.total)

    def score(self, density_weight: float) -> float:
        """Lower is better. `density_weight` trades throughput against correctness."""
        return self.total + density_weight * self.corrupted

    def summary(self) -> str:
        return (
            f"{self.patches}p/{self.physical_qubits}q depth {self.depth}: "
            f"E={self.total:.2f} (gate {self.gate:.2f}, idle {self.idle:.2f}"
            f"{f', move {self.movement:.2f}' if self.movement else ''}), "
            f"survival {self.survival:.2e}, corrupted {self.corrupted:.2f}"
        )


def estimate(
    knobs: NoiseKnobs,
    *,
    one_qubit: int,
    two_qubit: int,
    depth: int,
    patches: int,
    n_code: int,
    logical_qubits: int,
    crossings: int = 0,
    mean_hops: float = 0.0,
) -> ErrorEstimate:
    """
    Price one measured placement.

    `crossings` is grouped movements (one per cross-patch logical gate, matching
    `packing.cost.Term.crossings`), and `mean_hops` is the mean centre-to-centre
    separation in **cells** (`packing.tables.mean_patch_hops`). Each movement carries
    `moved_per_crossing` atoms there and back: the carried atoms pay `move_error`
    each, and the whole array pays the trip's idling -- which is the dominant
    movement cost and scales with the candidate's own physical qubit count, so a
    16-patch layout pays more per trip than a 4-patch one twice over (longer trips,
    more bystanders).

    The intra-patch term charges the beyond-blockade fraction of the patch's own
    two-qubit gates a carry each -- `two_qubit` minus the four physical gates of each
    targeted crossing, clamped at zero for the model-tally pass where gate counts are
    not yet measured.
    """
    physical = patches * (n_code + 2)
    volume = physical * depth
    rate = knobs.idle_rate
    gate = one_qubit * knobs.one_qubit_error + two_qubit * knobs.two_qubit_error
    idle = volume * knobs.idle_per_qubit_layer

    # --- grouped cross-patch movements ---
    trip_seconds = knobs.round_trip_error_seconds(max(0.0, mean_hops))
    handling = crossings * knobs.moved_per_crossing * knobs.move_error
    idle += crossings * trip_seconds * physical * rate

    # --- intra-patch beyond-blockade carries, a property of the code size ---
    fraction, mean_cells = intra_patch_movement(
        n_code + 2, pitch=knobs.atom_pitch, blockade=knobs.blockade_radius
    )
    intra_gates = max(0, two_qubit - 4 * crossings)
    carries = fraction * intra_gates
    handling += carries * knobs.move_error
    idle += carries * knobs.round_trip_error_seconds(mean_cells) * physical * rate

    movement = handling

    # A distance-2 code detects one error and misses two. Spread the total error load
    # over the patches, square it for the coincidence, and multiply by how many logical
    # qubits sit behind each detector pair -- the patch density.
    load = gate + idle + movement
    density = logical_qubits / patches if patches else float(logical_qubits)
    corrupted = (load * load / (2.0 * patches)) * density if patches else 0.0

    return ErrorEstimate(
        gate=gate,
        idle=idle,
        movement=movement,
        corrupted=corrupted,
        patches=patches,
        physical_qubits=physical,
        depth=depth,
    )


#: The workload the two shape-dependent conversions in `packing_profile` are pinned to:
#: a k=2, 12-logical-qubit program, so 8 patches of 6 atoms. It is the reference the
#: weighted-gate constants were calibrated on, and a reference is needed because the
#: profile is built *before* any candidate's patch count exists.
_REFERENCE_PATCHES = 8
_REFERENCE_PATCH_ATOMS = 6


def packing_profile(
    knobs: Optional[NoiseKnobs] = None,
    *,
    movement_scale: float = 1.0,
    idle_scale: float = 1.0,
    distance: bool = True,
) -> "NoiseProfile":
    """
    The `packing.NoiseProfile` that charges what these `NoiseKnobs` charge.

    Two objects describe the same hardware at two different layers. `NoiseKnobs` is in
    physical units -- error probabilities, seconds, metres -- because the resource stage
    ranks candidates on expected detectable errors. `NoiseProfile` is in weighted-gate
    units, because the placement objective is a gate count. Without a conversion the two
    stages disagree about the machine: every cost in `NoiseProfile` defaults to zero, so
    a caller who hands `plan_packing` nothing gets a planner optimising gates and depth
    with movement and idling priced at *nothing*, while the selector above it was ranking
    code sizes on exactly those terms. That mismatch is also what makes `spread` a
    no-op -- `descend` charges `ENCODE + DECODE` per live patch and, at zero movement and
    idle cost, has no counterweight -- so composing the two stages needs this function,
    not merely benefits from it.

    The exchange rate comes from `estimate` above, whose whole content is

        E = n1*p1 + n2*p2 + V*t_layer*r,   V = patches * (n + 2) * depth

    so one qubit idling through one layer costs `t_layer * r` where a two-qubit gate
    costs `p2`. `DEFAULT_WORK_WEIGHTS` prices that gate at `w2` units, which fixes
    everything else: a probability converts at `w2 / p2`, one two-qubit gate's worth of
    error per `w2` units.

    `movement_cost` follows the same route from the grouped-transport charges: one
    crossing is one round trip of two atoms, paying `move_error` each plus the idling the
    flight forces on the whole array. Both of those depend on the layout -- how many
    atoms are sitting there, and how far the trip is -- which is why they are pinned to
    the reference workload above rather than left to a patch count that does not exist
    yet. The result is per *cell*, matching `packing.cost.price`, which multiplies it by
    `mean_patch_hops`.

    `distance=False` prices movement at nothing and leaves the model distance-agnostic;
    `idle_scale=0.0` switches idling off. `NoiseKnobs.gate_counting()` reaches the same
    place from the other side -- zero `layer_time`, `move_error` and `movement_speed` --
    and returns the paper's convention, `NoiseProfile()`.
    """
    from iceberg.costs import DEFAULT_WORK_WEIGHTS

    from ..graph import NoiseProfile
    from ..tables import mean_patch_hops

    knobs = knobs or NoiseKnobs()
    _, two_qubit_weight = DEFAULT_WORK_WEIGHTS
    units_per_probability = two_qubit_weight / knobs.two_qubit_error
    idle_units = two_qubit_weight * knobs.idle_per_qubit_layer / knobs.two_qubit_error

    reference_qubits = _REFERENCE_PATCHES * _REFERENCE_PATCH_ATOMS
    reference_cells = mean_patch_hops(_REFERENCE_PATCHES, _REFERENCE_PATCH_ATOMS)
    idle_per_cell = knobs.round_trip_error_seconds(1.0) * reference_qubits * knobs.idle_rate
    handling_per_cell = (
        knobs.moved_per_crossing * knobs.move_error / reference_cells
        if reference_cells > 0 else 0.0
    )
    movement_units_per_cell = (idle_per_cell + handling_per_cell) * units_per_probability

    return NoiseProfile(
        movement_cost=movement_scale * movement_units_per_cell if distance else 0.0,
        idle_cost=idle_units * idle_scale,
        distance_scale=1.0 if distance else 0.0,
        # The raw knobs, carried through unconverted. `movement_scale` arrives already
        # multiplied by the caller's `distance_scalar` (see `noise_profile_for`), so it
        # *is* the knob when `movement_scale` defaults to 1.0. Only `plan.measured_cost`
        # reads these, to reprice the schedule the simulator will really run.
        idle_time_scalar=idle_scale,
        distance_scalar=movement_scale if distance else 0.0,
    )
