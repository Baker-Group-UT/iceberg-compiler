"""
scheduler.py — when things happen, and who is idle while they do.

The old helper tracked a per-qubit `last_time` and charged a gate's own qubits for the
skew between them. Two things followed from that, and both were wrong in the same
direction. Atoms not involved in a gate were charged nothing, so the array was free while
one pair worked. And `time_to_move` never advanced the clock, so transport generated no
idle time for anyone — which meant movement and idling could not interact at all, and
sweeping them as independent axes described no machine.

This builds a real schedule instead:

    a *unit* is one thing the machine does -- a gate, or a transport group with the gates
    it carries. It occupies qubits for a duration.

    a *layer* is a set of units that run concurrently. Its duration is the **maximum** of
    its units, because the machine does not move on until the slowest finishes.

    every qubit accrues dephasing for the part of the layer it is not working: the full
    layer duration if it is in no unit, and the remainder if its unit finished early.

That last rule is what makes movement expensive in the way it is actually expensive. A
100 us transport in a layer is 100 us of dephasing for every atom in the array, not just
the two in flight. In-flight dephasing on the moved atoms is real but small — two atoms
against the whole register.

Units are laid out ASAP: a unit starts in the earliest layer where none of its qubits are
still busy. This reorders only units on disjoint qubits, which commute, so the sampled
distribution is unchanged. Within a layer units are emitted in original circuit order,
which keeps measurement results lined up with `circuit.data` — the sampler returns them in
append order and `run_stim_backend` maps them back positionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .transport import TransportGroup

__all__ = [
    "ScheduleUnit",
    "Layer",
    "build_layers",
    "idle_attribution",
    "idle_times_for_layer",
]


@dataclass
class ScheduleUnit:
    """One thing the machine does, and the qubits it ties up while doing it."""

    qubits: FrozenSet[int]
    duration: float
    #: Index of the earliest circuit instruction this unit covers. Emission order within
    #: a layer sorts on it, so relative order of measurements is preserved.
    order: int
    instructions: List[int] = field(default_factory=list)
    #: Set when this unit is a transport group; the gates ride along inside it.
    transport: Optional[TransportGroup] = None
    #: Seconds of the duration that is atom flight rather than gate time. Charged to the
    #: moved atoms on top of their idle share.
    move_time: float = 0.0
    #: The atoms carried during this unit: a transport group's control side, the mover of
    #: an intra-patch beyond-blockade gate, or both ends of a swap-by-exchange. They pay
    #: `move_error` and in-flight dephasing regardless of which kind of unit carried them.
    moved: Tuple[int, ...] = ()
    #: Seconds each qubit actually works, when that differs from `duration`. A unit may
    #: occupy qubits it does not use -- a transport group holds both whole patches so the
    #: scheduler blocks them -- and occupancy is not an exemption from decoherence: a
    #: qubit's idle share is `layer duration - busy`, with `busy` defaulting to the unit
    #: duration for every qubit not listed here.
    busy: Optional[Dict[int, float]] = None
    #: Fenced units batch into a single shared layer rather than being laid out
    #: independently. Measurements need this for two reasons. Physically, readout is one
    #: global image over the whole array, so a detection round costs `t_meas` once and not
    #: once per qubit measured -- laid out independently they staggered across seven
    #: layers and three quarters of the modelled circuit time was re-imaging. And
    #: mechanically, the sampler returns results in append order while `run_stim_backend`
    #: maps them back positionally against `circuit.data`, so a measurement floating into
    #: an earlier layer than one that precedes it silently relabels the bits.
    fence: bool = False
    #: What kind of work this is: "gate", "transport" or "readout". Only used to attribute
    #: idle time back to whatever made a layer long -- see `idle_attribution`.
    cause: str = "gate"


@dataclass
class Layer:
    """Units running concurrently. Duration is the slowest of them."""

    units: List[ScheduleUnit] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((unit.duration for unit in self.units), default=0.0)

    @property
    def busy(self) -> FrozenSet[int]:
        result: set = set()
        for unit in self.units:
            result |= unit.qubits
        return frozenset(result)


def build_layers(units: Sequence[ScheduleUnit], num_qubits: int) -> List[Layer]:
    """
    ASAP-schedule `units` into layers.

    A unit starts in the earliest layer where every qubit it needs is free. Qubits are
    held for exactly one layer, so a long unit does not block its qubits across several —
    the layer it sits in is simply long, and everyone else in that layer idles for the
    difference.
    """
    next_free: Dict[int, int] = {}
    slots: List[List[ScheduleUnit]] = []
    pending: List[ScheduleUnit] = []
    #: Earliest slot the next fenced batch may use. Without this floor a later batch on
    #: cold qubits could land in an *earlier* slot than a batch already flushed — and the
    #: sampler returns measurements in emission order while `run_stim_backend` maps them
    #: back positionally, so that reordering silently relabels bits. Same-slot is fine:
    #: units within a layer are emitted sorted by circuit order.
    fence_floor = 0

    def earliest(unit: ScheduleUnit) -> int:
        return max((next_free.get(q, 0) for q in unit.qubits), default=0)

    def place(unit: ScheduleUnit, start: int) -> None:
        while len(slots) <= start:
            slots.append([])
        slots[start].append(unit)
        for qubit in unit.qubits:
            next_free[qubit] = start + 1

    def flush() -> None:
        """Commit the pending fenced batch to one shared layer."""
        nonlocal fence_floor
        if not pending:
            return
        start = max(fence_floor, max(earliest(unit) for unit in pending))
        for unit in pending:
            place(unit, start)
        pending.clear()
        fence_floor = start

    for unit in units:
        if not unit.qubits:
            continue
        if unit.fence:
            pending.append(unit)
            continue
        # Anything that needs a qubit still waiting to be imaged has to let the image
        # happen first. With one detection round at the end this never fires; a
        # mid-circuit round splits here, and `fence_floor` keeps the split batches in
        # circuit order.
        if any(unit.qubits & other.qubits for other in pending):
            flush()
        place(unit, earliest(unit))
    flush()

    layers: List[Layer] = []
    for slot in slots:
        if not slot:
            continue
        slot.sort(key=lambda unit: unit.order)
        layers.append(Layer(units=slot))
    return layers


def _busy_seconds(unit: ScheduleUnit, qubit: int) -> float:
    """Seconds `qubit` actually works inside `unit`."""
    if unit.busy is None:
        return unit.duration
    return unit.busy.get(qubit, unit.duration)


def idle_times_for_layer(layer: Layer, num_qubits: int) -> List[Tuple[int, float]]:
    """
    `(qubit, seconds idle)` for every qubit with a nonzero share in this layer.

    A qubit in no unit idles for the whole layer. A qubit whose unit finished early idles
    for the remainder — and a qubit a unit merely *occupies* idles for everything but the
    seconds it actually works (`ScheduleUnit.busy`): the bystander atoms of a patch held
    by a transport group are standing still, and being unaddressable does not pause their
    decoherence. Transport flight time is *not* deducted here — an atom in transit is
    decohering, and its flight charge is applied separately from `move_time`.
    """
    duration = layer.duration
    if duration <= 0:
        return []

    idle: Dict[int, float] = {q: duration for q in range(num_qubits)}
    for unit in layer.units:
        for qubit in unit.qubits:
            idle[qubit] = max(0.0, duration - _busy_seconds(unit, qubit))

    return [(qubit, seconds) for qubit, seconds in idle.items() if seconds > 0]


def idle_attribution(
    layers: Sequence[Layer], num_qubits: int
) -> Dict[str, float]:
    """
    Qubit-seconds of idling, attributed to whatever made each layer long.

    Idling is not an independent noise source in this model -- it is what the array does
    while something slow happens, and the slow thing is usually transport or readout. So
    "how much of the idle error is really movement?" is the question that decides how the
    downstream figures should weight the two, and it cannot be read off the idle total.

    Attribution is **marginal**, not gross: a transport layer is credited only with the
    idling it adds *over what that layer would have cost anyway*. If a 300 us move happens
    alongside a 250 us intra-patch move, transport is charged 50 us of everyone's time and
    the remaining 250 goes to the gate that would have stalled the array regardless.
    Answering "if movement were free, how much idle would disappear?" is exactly what
    tuning an idle-versus-distance ratio needs.

    Returns qubit-seconds keyed by cause. Divide by T2 for the error contribution.
    """
    totals: Dict[str, float] = {}
    for layer in layers:
        duration = layer.duration
        if duration <= 0 or not layer.units:
            continue
        dominant = max(layer.units, key=lambda unit: unit.duration).cause
        counterfactual = max(
            (unit.duration for unit in layer.units if unit.cause != dominant),
            default=0.0,
        )

        own: Dict[int, float] = {}
        for unit in layer.units:
            for qubit in unit.qubits:
                own[qubit] = _busy_seconds(unit, qubit)

        charged = residual = 0.0
        for qubit in range(num_qubits):
            busy = own.get(qubit, 0.0)
            charged += max(0.0, duration - busy)
            residual += max(0.0, counterfactual - busy)

        totals[dominant] = totals.get(dominant, 0.0) + (charged - residual)
        if residual:
            # The idling that would have happened anyway belongs to the unit that would
            # have stalled the layer without the dominant cause.
            runner_up = max(
                (unit for unit in layer.units if unit.cause != dominant),
                key=lambda unit: unit.duration,
            ).cause
            totals[runner_up] = totals.get(runner_up, 0.0) + residual
    return totals
