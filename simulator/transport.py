"""
transport.py — where atoms sit, and what it costs to move them.

The model this replaces treated a patch as a dimensionless point at unit pitch. Every
qubit in a patch shared one `(x, y)`, so intra-patch gates were charged distance zero at
every code size, and the site pitch was 1 whether the patch held 6 atoms or 14. Both gave
large `k` an advantage it had not earned, and neither corresponds to a machine.

This module is the geometry plus the movement accounting. It targets a **neutral-atom**
architecture, and the modelling choices below are deliberate simplifications rather than
oversights — they are listed so a reader can see exactly which corners are cut.

Architecture
------------
*One grid, storage and computation together.* Patches sit on a plane and atoms move
between them. Real AOD machines often separate a storage zone from an entangling zone;
that is a different geometry and would change the distance distribution, not the shape of
the argument.

*A patch is the most-square rectangle that fits its `n + 2` atoms*, one atom per cell,
tiled at that rectangle's size plus `PATCH_GUTTER` cells of clearance. Most-square rather
than smallest-area: 14 atoms fit a 2x7 perfectly and waste two cells in a 4x4, but 2x7 is
an elongated patch whose internal distances grow like `n` instead of `sqrt(n)`, and the
k=8/k=10 ordering flips on that tie-break alone. The gutter exists so neighbouring
patches do not sit inside each other's blockade radius.

What movement costs
-------------------
*Blockade, not a fudge factor.* Two atoms within `BLOCKADE_RADIUS` gate without moving —
the laser reaches. Beyond it they must be brought together like anything else. So there
is no "intra-patch operations cost some fraction of a real move" constant: the rule is a
threshold on the actual separation, and the penalty for large `k` falls out of the
geometry. The fraction of intra-patch pairs that fall outside blockade, which is the
constant this replaces, runs 0.00 at k=2 to 0.27 at k=10 at the default pitch:

    k     atoms   rect    max sep   outside blockade
    2       6     2x3      6.7um          0.00
    4       8     3x3      8.5um          0.04
    6      10     3x4     10.8um          0.18
    10     14     4x4     12.7um          0.27
    12     16     4x4     12.7um          0.32

A patch small enough to sit inside one blockade sphere genuinely needs no internal
movement; one that is not, does. That is the mechanism that keeps large patches honest,
and it has no tuned parameter in it.

*One movement per group, not per physical gate.* A logical cross-patch CNOT emits four
physical CX, all four spanning the patches (`iceberg_set.CNOT`), and a transversal macro
emits `n`. Charging transport per physical gate — which is what the old model did — means
paying four round trips for one logical CNOT and `n` for one macro. Grouping recovers a
4x saving on plain crossings and `(k+2)x` on macros.

*A group ends when anything touches either participating patch.* This is the conservative
reading: an atom that has left cannot be addressed, and rather than track which specific
atoms went and whether a given gate needs them, the whole patch is treated as unavailable
until they return. It over-splits — a gate on a lane that never moved also ends the group
— so movement here is an upper bound. Both large savings survive it anyway, because the
gates within one logical operation are emitted back to back with nothing in between.

*Atoms return home afterwards.* The alternative, leaving them parked until needed
elsewhere, is a real compiler optimisation and is not modelled.

*Groups on disjoint patch pairs run concurrently.* Nothing serialises transport across
unrelated patches. This assumes a machine that can gate in one region while moving atoms
in another, which current hardware generally cannot; it is the "capable compiler"
assumption and it is what makes parallelism worth anything in the schedule.

Deliberately absent
-------------------
*Acceleration.* Transport time is linear in distance. Real AOD moves are acceleration
limited, so short moves go like `sqrt(d)`.

*Fixed per-move overhead.* Picking an atom up and putting it down takes time — trap
handoff, ramps, settling — that demonstrated systems spend tens of microseconds on,
which is comparable to the 30-60 us flights modelled here. Only the *infidelity* of the
handoff is charged (`move_error`); its duration is not, so transport durations, and the
idling they force on the array, are optimistic by roughly that constant per trip.

*Atom loss.* Around 1e-3 per move, and detectable — it would interact with post-selection
in an interesting way, but it is a second mechanism.

*AOD ordering constraints.* Rows and columns move together and cannot cross. Arbitrary
subsets are allowed to move here.

*Blockade exclusivity.* Two Rydberg gates whose pairs sit within a blockade radius of
each other cannot run simultaneously — the interaction that makes the gate work couples
the pairs too. The scheduler happily runs disjoint same-patch pairs concurrently 3 um
apart, so intra-patch two-qubit parallelism is generous; on these numbers it is really
~1 gate per patch at a time.

*Bystander crosstalk.* An atom flying over another imparts some dephasing. Not modelled.
Note this is not the same as bystanders *idling* while a move happens, which is modelled
and is the dominant cost — see `scheduler.build_layers`.

*Syndrome-ancilla placement.* Atoms fill their rectangle row-major, which leaves the two
parity atoms in the far corner — near-maximal distances for the decode round's data-to-
ancilla gates. A machine designer would park the ancilla centrally, or carry it along the
data row during the round instead of paying a round trip per gate. Both are compiler
optimisations this model does not take; the corner placement is the conservative reading.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "ATOM_PITCH",
    "BLOCKADE_RADIUS",
    "MOVEMENT_SPEED",
    "PATCH_GUTTER",
    "Geometry",
    "TransportGroup",
    "derive_transport_groups",
    "euclidean",
    "manhattan",
    "most_square_rectangle",
]


#: Metres between adjacent atom cells. Rubidium tweezer arrays run 3-5 um.
ATOM_PITCH = 3e-6

#: Metres. Two atoms closer than this gate without being moved. Depends on the Rydberg
#: state; 7-10 um is the usual range.
BLOCKADE_RADIUS = 8e-6

#: Metres per second. Equivalently ~1 us of flight per um travelled, which is the figure
#: quoted for current tweezer hardware.
#:
#: This was 0.2 m/s and that was too pessimistic by 5x. The arrays here are *small* -- six
#: patches of the [[4,2,2]] code span about 39 x 18 um in total, so a typical inter-patch
#: hop is 27 um one way and 54 um there and back. At 0.2 m/s that round trip took 270 us,
#: which made a single cross-patch operation longer than the entire rest of the circuit and
#: put transport at 29% of wall clock. At 1 m/s the same trip is 54 us. The distances were
#: never the problem; the speed was.
MOVEMENT_SPEED = 1.0

#: Empty cells between neighbouring patch rectangles, so patches do not sit inside each
#: other's blockade radius. The operative number is nearest-atom spacing, which is
#: `gutter + 1` cells: at the default pitch that is 9 um against an 8 um radius, so
#: cross-patch pairs always need transport — by 1 um, which `Geometry.__post_init__`
#: checks. The gutter changes the constant rather than the scaling, since inter-patch
#: distance is dominated by patch size times grid dimension.
PATCH_GUTTER = 2


@dataclass(frozen=True)
class Geometry:
    """Physical scale. Cell coordinates times `pitch` give metres."""

    pitch: float = ATOM_PITCH
    blockade_radius: float = BLOCKADE_RADIUS
    movement_speed: float = MOVEMENT_SPEED
    gutter: int = PATCH_GUTTER

    def __post_init__(self):
        # The gutter exists so cross-patch pairs always need transport: nearest atoms of
        # neighbouring patches sit `gutter + 1` cells apart. If the blockade reaches
        # across that, adjacent-patch pairs could physically gate in place while the
        # model still charges them a round trip — over-charging silently. Warn rather
        # than raise, because sweeps may do this deliberately.
        if self.blockade_radius >= (self.gutter + 1) * self.pitch > 0:
            warnings.warn(
                f"blockade radius {self.blockade_radius:g} m reaches across the patch "
                f"gutter ({self.gutter} cells + 1 at {self.pitch:g} m pitch): "
                "cross-patch pairs within blockade are still charged transport",
                stacklevel=2,
            )

    def metres(self, cells: float) -> float:
        return cells * self.pitch

    def needs_transport(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """
        True if two atoms have to be brought together before they can interact.

        Measured **Euclidean**, unlike everything else here: the blockade is a physical
        sphere around the excited atom, and the laser reaches in a straight line. Movement
        is Manhattan because the atom travels along rows and columns. Using Manhattan for
        both would shrink the effective blockade by up to sqrt(2) and start charging
        transport inside a k=2 patch, which is exactly the patch that should be free.
        """
        return self.metres(euclidean(a, b)) > self.blockade_radius

    def round_trip_time(self, cells: float) -> float:
        """Seconds to carry an atom `cells` of Manhattan travel away and back again."""
        if self.movement_speed <= 0:
            return 0.0
        return 2.0 * self.metres(cells) / self.movement_speed

    def one_way_time(self, cells: float) -> float:
        """Seconds for a one-way carry — a swap-by-exchange does not come back."""
        if self.movement_speed <= 0:
            return 0.0
        return self.metres(cells) / self.movement_speed


def most_square_rectangle(atoms: int) -> Tuple[int, int]:
    """
    Smallest near-square integer rectangle holding `atoms` cells.

    Minimises aspect ratio first and area second — see the module docstring for why that
    order matters. Returns `(rows, columns)` with `rows <= columns`.
    """
    if atoms <= 0:
        return (0, 0)
    best: Optional[Tuple[int, int, int, int]] = None
    for rows in range(1, atoms + 1):
        columns = math.ceil(atoms / rows)
        if rows > columns:
            continue
        candidate = (abs(rows - columns), rows * columns, rows, columns)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return (best[2], best[3])


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Travel distance: an atom moves along rows and columns."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Line-of-sight distance, which is what the blockade radius is measured in."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class TransportGroup:
    """
    One movement: some atoms go, a run of gates happens, the atoms come back.

    `instructions` indexes into the circuit the group was derived from. `moved` is the
    control-side atoms — the ones that travel. `distance_cells` is between patch centres.
    """

    patches: Tuple[int, int]
    instructions: List[int] = field(default_factory=list)
    moved: List[int] = field(default_factory=list)
    distance_cells: float = 0.0

    def duration(self, geometry: Geometry) -> float:
        return geometry.round_trip_time(self.distance_cells)


def derive_transport_groups(
    instruction_patches: Sequence[FrozenSet[int]],
    instruction_qubits: Sequence[Sequence[int]],
    patch_of_qubit: Dict[int, int],
    patch_centres: Dict[int, Tuple[int, int]],
    geometry: Geometry,
) -> List[TransportGroup]:
    """
    Group consecutive cross-patch gates into movements.

    Runs over an already-built encoded circuit, so it sees only gates that survived
    `translator.core`'s three cancellation rounds. That is deliberate: a marker emitted
    inline during construction would sit on no qubit wire, so it would neither block a
    cancellation nor be cleaned up when the gates it wrapped disappeared.

    Args:
        instruction_patches: per instruction, the set of patches it touches.
        instruction_qubits: per instruction, its physical qubit indices.
        patch_of_qubit: physical qubit -> patch, from `iceberg.layout`.
        patch_centres: patch -> its rectangle's centre cell.
        geometry: physical scale.

    Returns:
        Groups in circuit order. An instruction appears in at most one group; anything not
        in a group is either intra-patch or single-qubit and is charged separately.
    """
    open_groups: Dict[Tuple[int, int], TransportGroup] = {}
    closed: List[TransportGroup] = []

    def close(key: Tuple[int, int]) -> None:
        group = open_groups.pop(key, None)
        if group is not None:
            closed.append(group)

    for index, patches in enumerate(instruction_patches):
        qubits = instruction_qubits[index]
        crossing = len(patches) == 2 and len(qubits) == 2

        key: Optional[Tuple[int, int]] = None
        if crossing:
            ordered = sorted(patches)
            key = (ordered[0], ordered[1])

        # Anything touching a patch with atoms away brings them home first. That includes
        # a *different* crossing that shares one of the patches.
        for other in list(open_groups):
            if other == key:
                continue
            if patches & set(other):
                close(other)

        if not crossing or key is None:
            continue

        group = open_groups.get(key)
        if group is None:
            left, right = patch_centres.get(key[0]), patch_centres.get(key[1])
            if left is None or right is None:
                # Silently pricing this trip at zero would make the movement free.
                raise ValueError(
                    f"no centre for patch pair {key}: patch_centres covers "
                    f"{sorted(patch_centres)}"
                )
            group = TransportGroup(patches=key, distance_cells=float(manhattan(left, right)))
            open_groups[key] = group

        group.instructions.append(index)
        # The control side travels. Every gate in a cross-patch construction is emitted
        # control-first, so the first qubit's patch is the control patch throughout.
        control_patch = patch_of_qubit.get(qubits[0])
        for qubit in qubits:
            if patch_of_qubit.get(qubit) == control_patch and qubit not in group.moved:
                group.moved.append(qubit)

    for key in list(open_groups):
        close(key)

    closed.sort(key=lambda g: g.instructions[0] if g.instructions else 0)
    return closed
