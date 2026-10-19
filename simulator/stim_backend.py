# This function is for stim. It needs to look at the circuit and create a minimal
# paulichannel noise model
# based off of the qubit interactions. I.e.
# We should not create a fully connected graph of the different
# pauli error channels, just the ones that are actually used in the circuit.
# pylint: disable = ['E1101']

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List
import qiskit
import stim
import math

from iceberg.layout import patch_of_qubit

from .noise_models import (manhattan_distance)
from .noise_markers import NOISE_BEGIN, NOISE_END
from .grid_constructor import centres_for_circuit
from .scheduler import ScheduleUnit, build_layers, idle_times_for_layer
from .transport import Geometry, TransportGroup, derive_transport_groups

ONE_Q = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sx": "SQRT_X",
    "measure": "M",
}

TWO_Q = {
    "cx": "CX",
    "cz": "CZ",
    "swap": "SWAP",
}

#: Defaults are the neutral-atom machine (`noise_models.NEUTRAL_ATOM`), which is what this
#: compiler targets. They used to be the paper's Table 2 -- a superconducting calibration
#: on hardware that cannot move atoms at all.
NOISE_PARAMS = {
    "distance_scalar": 1.0,
    "p1": 2e-4,
    "p2": 3e-3,
    "p_meas": 3e-3,
    "T1": 10.0,
    "T2": 1.5,
    "idle_time_scalar": 1.0,
    "t_1q": 1e-6,
    "t_2q": 0.5e-6,
    "t_meas": 3e-3,
    # Geometry. The pitch used to be a hardcoded 1e-8 m -- 10 nm per hop, which is no
    # lattice constant, just a number that put transport on the same timescale as an
    # 88 ns gate. `distance_scalar` scales movement *cost* (see `_geometry`), not this.
    "atom_pitch": 3e-6,
    "blockade_radius": 8e-6,
    "movement_speed": 1.0,
    "patch_gutter": 2,
    # Per atom per round trip, and not dephasing -- see `StimNoiseParameters.move_error`.
    "move_error": 1e-3,
}

#: Per-thread active parameters. `NOISE_PARAMS` above is the *defaults*; a run never
#: writes into it.
#:
#: `run_stim_backend` used to copy its arguments straight into the module dict and never
#: restore them, which broke two ways at once. Sequentially, any later call that did not
#: pass explicit params silently inherited whatever ran last. Concurrently, two threads
#: sharing the dict raced -- and `experiments/.../stim_vs_aer.py` drives this from a
#: ThreadPoolExecutor, so the race was reachable. Now every read goes through `_params()`,
#: which sees only its own thread's scoped copy.
_LOCAL = threading.local()


def _params() -> Dict:
    """Parameters in force on this thread, falling back to the module defaults."""
    return getattr(_LOCAL, "params", NOISE_PARAMS)


@contextmanager
def _active_params(overrides: Dict):
    """Install `overrides` over the defaults for this thread only, then restore."""
    previous = getattr(_LOCAL, "params", None)
    _LOCAL.params = {**NOISE_PARAMS, **overrides}
    try:
        yield
    finally:
        if previous is None:
            del _LOCAL.params
        else:
            _LOCAL.params = previous


def _clamp_arr(vals: List[float]) -> List[float]:
    ret: List[float] = []
    for val in vals:
        val = max(0.0, min(1.0, val))
        ret.append(val)
    return ret

def _normalize_probs(vals: List[float]) -> List[float]:
    z = sum(vals)
    if z <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / z for v in vals]

# NOTE: Twirled symmetric depolarizing channel is a symmetric depolarizing channel
def _pauli_twirl_approx(t: float, T1: float, T2: float) -> List[float]:
    p_x = p_y = (1 - math.exp(-t / T1)) / 4
    p_z = (1 - math.exp(-t / T2)) / 2 - (1 - math.exp(-t / T1)) / 4
    p_i = 1 - p_x - p_y - p_z
    p = _normalize_probs(_clamp_arr([p_i, p_x, p_y, p_z]))
    return p


def thermal_error_rate(T1: float, T2: float) -> float:
    """
    Error probability per second of the twirled thermal channel, in the small-t limit.

    `_pauli_twirl_approx` gives a total non-identity weight of
    (1 - e^{-t/T2})/2 + (1 - e^{-t/T1})/4 ≈ t/(2·T2) + t/(4·T1), so the rate is
    1/(2·T2) + 1/(4·T1). Any accounting that prices idle seconds must use this rate —
    dividing by T2 alone overstates every idle-family term ~2x against the exact
    per-channel probabilities used for gates, measurement and move handling.
    """
    rate = 0.0
    if T2 > 0:
        rate += 0.5 / T2
    if T1 > 0:
        rate += 0.25 / T1
    return rate

def _geometry() -> Geometry:
    """
    Physical scale in force on this thread.

    `distance_scalar` divides the movement *speed* rather than multiplying the pitch.
    Scaling the pitch would move the atoms, and since blockade is a threshold on
    separation that made the sweep knob discontinuous -- past some scalar, gates inside a
    patch abruptly started needing transport. Slowing the atoms down costs exactly what
    moving further costs, on both channels it should (flight dephasing and the idling it
    forces on everyone else), and leaves the geometry alone.
    """
    scalar = _params()["distance_scalar"]
    return Geometry(
        pitch=_params()["atom_pitch"],
        blockade_radius=_params()["blockade_radius"],
        movement_speed=_params()["movement_speed"] / scalar if scalar > 0 else 0.0,
        gutter=int(_params()["patch_gutter"]),
    )


def _dephase(out: stim.Circuit, qubits: Iterable[int], seconds: float) -> None:
    """Apply a twirled thermal channel for `seconds` to each of `qubits`."""
    if seconds <= 0:
        return
    _, p_x, p_y, p_z = _pauli_twirl_approx(seconds, _params()["T1"], _params()["T2"])
    if p_x <= 0 and p_y <= 0 and p_z <= 0:
        return
    for qubit in qubits:
        out.append("PAULI_CHANNEL_1", [qubit], [p_x, p_y, p_z])


def add_two_qubit_gate_noise(stim_circuit: stim.Circuit, qubits: list[int]) -> None:
    """
    Depolarizing noise for one two-qubit gate.

    This used to also charge the gate a distance-dependent thermal channel, per *physical*
    gate -- so one logical cross-patch CNOT paid four round trips (`iceberg_set.CNOT`
    emits four gates, all spanning the patches) and a transversal macro paid n. Geometry
    now lives in grouped transport events instead; see `simulator.transport`.
    """
    if len(qubits) != 2:
        return
    p2 = _params()["p2"]
    if p2 > 0:
        stim_circuit.append("DEPOLARIZE2", [qubits[0], qubits[1]], [p2])

def add_single_qubit_noise(stim_circuit: stim.Circuit, qubits: list[int], gate_name: str) -> None:
    if gate_name == "measure":
        return
    p1 = _params()["p1"]
    if p1 > 0:
        stim_circuit.append("DEPOLARIZE1", qubits, [p1])

@dataclass
class _Record:
    """One emittable instruction, with the noise-marker region it fell in."""

    index: int          # position among records, which is also emission order
    name: str
    qubits: List[int]
    kind: str           # "one_q" | "two_q" | "measure"
    noise: bool


def _scan(circuit: qiskit.QuantumCircuit) -> List[_Record]:
    """Flatten the circuit, resolving noise-marker regions and dropping barriers."""
    has_markers = any(
        inst.operation.name.lower() in {NOISE_BEGIN, NOISE_END} for inst in circuit.data
    )
    enabled = not has_markers
    records: List[_Record] = []
    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name == NOISE_BEGIN:
            enabled = True
            continue
        if name == NOISE_END:
            enabled = False
            continue
        if name == "barrier":
            continue
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if name == "measure":
            kind = "measure"
        elif name in ONE_Q:
            kind = "one_q"
        elif name in TWO_Q:
            kind = "two_q"
        else:
            raise NotImplementedError(f"Unsupported gate for Stim: {name}")
        records.append(_Record(len(records), name, qubits, kind, enabled))
    return records


def _patch_context(circuit: qiskit.QuantumCircuit, geometry: Geometry):
    """`(qubit -> patch, patch -> centre cell)`, or empties for a non-encoded circuit."""
    try:
        return (
            patch_of_qubit(circuit),
            centres_for_circuit(circuit, geometry=geometry),
        )
    except (ValueError, AttributeError, KeyError):
        return {}, {}


@dataclass
class _Motion:
    """What one instruction costs: occupancy, and any atom carriage inside it."""

    duration: float
    move_time: float = 0.0
    moved: tuple = ()


def _motion(record: _Record, layout, patch_map: Dict[int, int], geometry: Geometry) -> _Motion:
    """
    Seconds this instruction occupies its qubits, and which atoms it carries.

    Three kinds of two-qubit instruction involve movement outside a transport group:

    * an intra-patch gate whose atoms sit farther apart than the blockade radius pays a
      round trip -- see `simulator.transport` for why that threshold replaces a tuned
      intra-patch discount. The first qubit travels, matching the control-side convention
      of cross-patch groups, and pays `move_error` plus flight dephasing like any other
      carried atom.
    * a `swap` is an **atom exchange**: the two atoms trade cells, each travelling the
      separation once, simultaneously — so the duration is one one-way flight and there
      is no gate pulse at all. Both atoms are carried, so both pay `move_error` and
      flight dephasing; the gate-noise path deliberately skips swaps because their noise
      *is* the movement. (Under the fixed index->cell layout a physical exchange is
      exactly what Stim's SWAP instruction says: the state at cell A becomes the state
      that was at cell B.)
    * cross-patch pairs are not charged here; their movement belongs to the group that
      carries them. A cross-patch `swap` would also be swallowed by a group and priced as
      a gate — nothing in `iceberg/` emits one.

    Without a layout (distance noise off, or a non-encoded circuit) every movement
    mechanism is off and a swap falls back to plain gate occupancy.
    """
    if record.kind == "measure":
        return _Motion(_params()["t_meas"])
    if record.kind == "one_q":
        return _Motion(_params()["t_1q"])

    gate_time = _params()["t_2q"]
    if layout and len(record.qubits) == 2:
        left, right = record.qubits
        if left in layout and right in layout:
            distance = manhattan_distance(layout[left], layout[right])
            same_patch = patch_map.get(left, -1) == patch_map.get(right, -2)
            if record.name == "swap" and same_patch:
                flight = geometry.one_way_time(distance)
                if flight > 0:
                    return _Motion(flight, move_time=flight, moved=(left, right))
                return _Motion(gate_time)
            # Blockade is line-of-sight; travel is along rows and columns.
            if same_patch and geometry.needs_transport(layout[left], layout[right]):
                trip = geometry.round_trip_time(distance)
                return _Motion(gate_time + trip, move_time=trip, moved=(left,))
    return _Motion(gate_time)


def _build_units(
    records: List[_Record],
    groups: List[TransportGroup],
    layout,
    patch_map: Dict[int, int],
    geometry: Geometry,
) -> List[ScheduleUnit]:
    """One unit per instruction, except that a transport group swallows its own gates."""
    qubits_of_patch: Dict[int, set] = {}
    for qubit, patch in patch_map.items():
        qubits_of_patch.setdefault(patch, set()).add(qubit)

    owner: Dict[int, TransportGroup] = {}
    for group in groups:
        for index in group.instructions:
            owner[index] = group

    units: List[ScheduleUnit] = []
    started: set = set()
    for record in records:
        group = owner.get(record.index)
        if group is None:
            motion = _motion(record, layout, patch_map, geometry)
            units.append(
                ScheduleUnit(
                    qubits=frozenset(record.qubits),
                    duration=motion.duration,
                    order=record.index,
                    instructions=[record.index],
                    move_time=motion.move_time,
                    moved=motion.moved,
                    fence=record.kind == "measure",
                    cause=(
                        "readout" if record.kind == "measure"
                        else "transport" if motion.move_time > 0
                        else "gate"
                    ),
                    # The stationary partner of an intra-patch move works only during the
                    # gate itself; it idles while its mate is in the air.
                    busy=(
                        {q: motion.duration - motion.move_time
                         for q in record.qubits if q not in motion.moved}
                        if motion.move_time > 0 else None
                    ),
                )
            )
            continue
        if id(group) in started:
            continue
        started.add(id(group))

        members = [records[i] for i in group.instructions]
        member_times = [_motion(m, layout, patch_map, geometry).duration for m in members]
        gate_time = sum(member_times)
        # Patch-level occupancy: atoms that left cannot be addressed, and rather than
        # track which specific ones went, the whole patch is held until they return.
        occupied: set = set()
        for patch in group.patches:
            occupied |= qubits_of_patch.get(patch, set())
        for member in members:
            occupied |= set(member.qubits)
        move = group.duration(geometry)

        # Occupancy blocks the patches; it does not pause their decoherence. Each atom
        # is *working* only while it gates (and, for the carried atoms, while it flies);
        # a bystander atom of a held patch works for none of it and idles the whole
        # unit, exactly like an atom outside the patches.
        busy: Dict[int, float] = {qubit: 0.0 for qubit in occupied}
        for member, member_time in zip(members, member_times):
            for qubit in member.qubits:
                busy[qubit] = busy.get(qubit, 0.0) + member_time
        for qubit in group.moved:
            busy[qubit] = busy.get(qubit, 0.0) + move

        units.append(
            ScheduleUnit(
                qubits=frozenset(occupied),
                duration=move + gate_time,
                order=group.instructions[0],
                instructions=list(group.instructions),
                transport=group,
                move_time=move,
                moved=tuple(group.moved),
                cause="transport",
                busy=busy,
            )
        )
    return units


def _emit_record(
    out: stim.Circuit, record: _Record, include_noise: bool, price_swaps_as_gates: bool = False
) -> None:
    noisy = include_noise and record.noise
    if record.kind == "measure":
        if noisy:
            p_meas = _params()["p_meas"]
            if p_meas > 0:
                out.append("X_ERROR", record.qubits, [p_meas])
        out.append("M", record.qubits)
    elif record.kind == "one_q":
        out.append(ONE_Q[record.name], record.qubits)
        if noisy:
            add_single_qubit_noise(out, record.qubits, record.name)
    else:
        out.append(TWO_Q[record.name], record.qubits)
        # A swap is an atom exchange, not a pulse: its noise is the movement charge on
        # the two carried atoms (`_motion`), not a two-qubit depolarizing channel. When
        # there is no layout to price the exchange with, it falls back to gate pricing
        # rather than being free — the Aer models charge swaps p2 for the same reason.
        if noisy and (record.name != "swap" or price_swaps_as_gates):
            add_two_qubit_gate_noise(out, record.qubits)


def qiskit_to_stim(
    circuit: qiskit.QuantumCircuit, layout=None, include_noise: bool = True
) -> stim.Circuit:
    """
    Convert to Stim, scheduling the circuit into layers and charging idle time honestly.

    The old version walked instructions one at a time and charged a gate's own qubits for
    the skew between them, so the rest of the array was free while one pair worked, and
    transport never advanced the clock at all. Now every instruction becomes a scheduling
    unit, cross-patch runs are bundled into transport groups, units are laid out ASAP, and
    each layer lasts as long as its slowest member -- with everyone not working charged
    dephasing for the difference. See `simulator.scheduler`.

    Reordering is confined to units on disjoint qubits, which commute, and within a layer
    units are emitted in original order, so measurement results stay lined up with
    `circuit.data`.
    """
    out = stim.Circuit()
    records = _scan(circuit)
    if not records:
        return out

    geometry = _geometry()
    patch_map, centres = _patch_context(circuit, geometry)

    groups: List[TransportGroup] = []
    if layout is not None and patch_map and centres:
        groups = derive_transport_groups(
            [frozenset(patch_map[q] for q in r.qubits if q in patch_map) for r in records],
            [r.qubits for r in records],
            patch_map,
            centres,
            geometry,
        )

    units = _build_units(records, groups, layout, patch_map, geometry)
    scalar = _params()["idle_time_scalar"]

    for layer in build_layers(units, circuit.num_qubits):
        noisy = include_noise and any(
            records[index].noise for unit in layer.units for index in unit.instructions
        )
        if noisy and scalar != 0:
            for qubit, seconds in idle_times_for_layer(layer, circuit.num_qubits):
                _dephase(out, [qubit], seconds * scalar)
        for unit in layer.units:
            if noisy and unit.moved:
                # The atoms actually in flight decohere on top of their idle share. This
                # is the small term: a few atoms against the whole register, and it
                # shrinks as T2 improves. `moved` covers every kind of carriage: a
                # group's control side, the mover of an intra-patch beyond-blockade
                # gate, and both ends of a swap-by-exchange.
                _dephase(out, unit.moved, unit.move_time)
                # Picking an atom up and putting it back down costs something that is not
                # dephasing -- heating, imperfect handoff -- and does *not* improve with
                # coherence. Charged once per carriage to the atoms actually carried;
                # atoms merely flown over are not modelled. See `move_error`.
                # `distance_scalar` scales this too, so both movement channels move
                # together when the ratio sweep dials movement up or down.
                move_error = min(1.0, _params()["move_error"] * _params()["distance_scalar"])
                if move_error > 0:
                    out.append("DEPOLARIZE1", list(unit.moved), [move_error])
            for index in unit.instructions:
                _emit_record(
                    out, records[index], include_noise,
                    price_swaps_as_gates=layout is None,
                )
    return out

def _quasi_from_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}

def _format_key_by_cregs(mem: List[int], circuit: qiskit.QuantumCircuit) -> str:
    """
    Return a space-delimited key per classical register, in circuit.cregs order,
    with position `i` of each substring holding bit `i` of that register.

    This is the *post-inversion* convention that everything downstream of the
    simulator assumes: `iceberg_postselect` reads position `d` of a data register
    as data qubit `d`, and `reduce_distribution` indexes the decoded logical
    string as `patch * k + slot`, left to right. Aer arrives at the same
    convention the long way round — it emits registers in reverse declaration
    order, each little-endian, and `Simulator._invert_endianness` reverses both —
    so producing it directly here is what makes the two backends comparable.

    Emitting each register in Qiskit's *printed* little-endian order instead
    (`reversed(reg_bits)`, which this function used to do) is invisible at n=4
    and corrupts the distribution at every larger code size. For a 4-bit
    even-parity codeword, `CodeLayout.decode_logical` is reversal-invariant:
    with b0^b1^b2^b3 = 0, decoding the reversed word gives
    [b2^b0, b1^b0] == [b1^b3, b2^b3], the same pair of logical bits. At n=6 the
    reversal both permutes the lanes and mixes in the wrong anchor, and the
    logical distribution comes back wrong rather than erroring — which is how a
    k=2 TVD at the sampling floor sat next to 0.775 at k=4.
    """
    if not circuit.cregs:
        return "".join(str(int(b)) for b in mem)

    clbit_to_idx = {cb: i for i, cb in enumerate(circuit.clbits)}
    parts = []
    for creg in circuit.cregs:
        reg_bits = [mem[clbit_to_idx[cb]] for cb in creg]
        parts.append("".join(str(int(b)) for b in reg_bits))
    return " ".join(parts)

def run_stim_backend(
    circuit: qiskit.QuantumCircuit,
    *,
    shots: int = 10_000,
    seed: int | None = None,
    include_noise: bool = True,
    noise_params=None,
    shots_per_chunk: int | None = None,
    progress_callback=None,
    layout=None,
    distance_scalar: float | None = None,
):
    overrides: Dict = {}
    if noise_params is not None:
        for key, attribute in (
            ("T1", "T1"),
            ("T2", "T2"),
            ("idle_time_scalar", "idle_time_scalar"),
            ("t_1q", "one_qubit_gate_time"),
            ("t_2q", "two_qubit_gate_time"),
            ("t_meas", "meas_time"),
            ("p1", "one_qubit_error"),
            ("p2", "two_qubit_error"),
            ("p_meas", "readout_error"),
            ("atom_pitch", "atom_pitch"),
            ("blockade_radius", "blockade_radius"),
            ("movement_speed", "movement_speed"),
            ("patch_gutter", "patch_gutter"),
            ("move_error", "move_error"),
        ):
            value = getattr(noise_params, attribute, None)
            if value is not None:
                overrides[key] = value
    if distance_scalar is not None:
        overrides["distance_scalar"] = distance_scalar

    with _active_params(overrides):
        return _run(circuit, shots, seed, include_noise, shots_per_chunk,
                    progress_callback, layout)


def _run(circuit, shots, seed, include_noise, shots_per_chunk, progress_callback, layout):
    sc = qiskit_to_stim(circuit, layout=layout, include_noise=include_noise)
    sampler = sc.compile_sampler(seed=seed)
    clbits = list(circuit.clbits)
    clbit_index = {cb: idx for idx, cb in enumerate(clbits)}
    meas_map = []
    for inst in circuit.data:
        if inst.operation.name.lower() != "measure":
            continue
        q = circuit.find_bit(inst.qubits[0]).index
        if inst.clbits:
            for cb in inst.clbits:
                meas_map.append((q, clbit_index.get(cb)))
        else:
            meas_map.append((q, None))

    num_clbits = len(clbits)
    if not meas_map or shots <= 0:
        return {}

    counts: dict[str, int] = {}
    chunk_size = max(1, shots_per_chunk or shots)
    remaining = shots
    while remaining > 0:
        current = min(chunk_size, remaining)
        samples = sampler.sample(shots=current)
        for row in samples:
            mem = [0] * max(num_clbits, 1)
            for k, (_, ci) in enumerate(meas_map):
                if ci is None or ci >= len(mem):
                    continue
                mem[ci] = int(row[k])
            key = _format_key_by_cregs(mem, circuit)
            counts[key] = counts.get(key, 0) + 1
        if progress_callback is not None:
            progress_callback(current)
        remaining -= current

    return _quasi_from_counts(counts)
