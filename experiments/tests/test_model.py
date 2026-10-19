"""
test_model.py -- print what the neutral-atom noise model is actually doing.

Not a pytest file despite the name; it is a terminal inspector for
`simulator.transport` + `simulator.scheduler`. The model has a lot of moving parts that
are individually simple and jointly hard to picture, so this dumps each of them:

    ATOM GRID     where every physical qubit sits, and which patch owns it
    BLOCKADE      which intra-patch pairs are close enough to gate without moving
    MOVEMENTS     how cross-patch gates grouped into round trips, and what split them
    SCHEDULE      the layers, what sets each one's duration, and who idles through it
    BUDGET        where the error actually comes from
    STIM          the emitted circuit with its noise instructions (--stim)

Run with:

    PYTHONPATH=. python experiments/tests/test_model.py
    PYTHONPATH=. python experiments/tests/test_model.py --n-code 12 --qubits 20
    PYTHONPATH=. python experiments/tests/test_model.py --stim --layers 12
    PYTHONPATH=. python experiments/tests/test_model.py --sweep
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import string
from typing import Dict, List

from iceberg.layout import patch_of_qubit
from iceberg.utils import logical_capacity
from permutations.generator import random_permutation_from_circuit
from simulator import stim_backend as sb
from simulator.grid_constructor import centres_for_circuit, positions_for_circuit
from simulator.noise_models import StimNoiseParameters
from simulator.scheduler import build_layers, idle_attribution, idle_times_for_layer
from simulator.transport import derive_transport_groups, euclidean
from tests.random_programs import random_clifford_program
from translator.config import TranslatorConfig
from translator.core import Translator

#: Patches are labelled A, B, C ... in the grid dump.
LABELS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def overrides_from(params: StimNoiseParameters) -> Dict:
    """The `_active_params` dict `run_stim_backend` would build from a parameter set."""
    return {
        "T1": params.T1,
        "T2": params.T2,
        "t_1q": params.one_qubit_gate_time,
        "t_2q": params.two_qubit_gate_time,
        "t_meas": params.meas_time,
        "p1": params.one_qubit_error,
        "p2": params.two_qubit_error,
        "p_meas": params.readout_error,
        "atom_pitch": params.atom_pitch,
        "blockade_radius": params.blockade_radius,
        "movement_speed": params.movement_speed,
        "patch_gutter": params.patch_gutter,
        "move_error": params.move_error,
        "idle_time_scalar": 1.0,
    }


def _drop_swaps(circuit):
    """The same circuit with program SWAPs removed. See `build_circuit`."""
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.operation.name.lower() != "swap":
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def build_circuit(n_code: int, num_qubits: int, seed: int, depth: int):
    """
    First random program and placement the translator accepts at this code size.

    **Why this retries, and why it falls back to a swap-free program.** A program SWAP is
    implemented as a *relabel* of two slots (`translator.handlers.swap`), which only works
    when both qubits sit in the same patch -- you cannot relabel across patches, so a
    cross-patch SWAP is rejected outright. Wide programs at small `k` are almost all
    cross-patch: 20 logical qubits at k=2 spreads over 12 patches holding under two qubits
    each, so nearly every SWAP straddles a boundary and every placement is refused. That is
    a translator limitation, not a property of the noise model, and it should not stop an
    inspector from running -- so if the full gate set will not place, this drops SWAPs and
    says so.
    """
    k = logical_capacity(n_code)
    translator = Translator(config=TranslatorConfig(n_code=n_code, fault_tolerant=False))
    patches = [n_code] * (num_qubits // k + 2)

    for swap_free in (False, True):
        for attempt in range(60):
            rng = random.Random(seed + attempt)
            program = random_clifford_program(
                num_qubits, depth, rng=rng, include_xcx=True, measure=True
            )
            if swap_free:
                program = _drop_swaps(program)
            permutation = random_permutation_from_circuit(
                program, patches, seed=seed + attempt
            )
            try:
                circuit = translator.translate(program, permutation).circuit
            except RuntimeWarning:
                continue
            if swap_free:
                print(
                    f"  note: no placement accepted with SWAPs at n={n_code}, "
                    f"q={num_qubits} -- a program SWAP is a slot relabel and cannot "
                    f"cross patches. Retried without them."
                )
            return circuit
    raise SystemExit(
        f"no placement accepted at n={n_code}, q={num_qubits}, depth={depth}, "
        f"even without SWAPs"
    )


class Model:
    """Everything `qiskit_to_stim` computes, held open for inspection."""

    def __init__(self, circuit, params: StimNoiseParameters):
        self.circuit = circuit
        self.params = params
        with sb._active_params(overrides_from(params)):
            self.geometry = sb._geometry()
            self.layout = positions_for_circuit(circuit, geometry=self.geometry)
            self.patch_of = patch_of_qubit(circuit)
            self.centres = centres_for_circuit(circuit, geometry=self.geometry)
            self.records = sb._scan(circuit)
            self.groups = derive_transport_groups(
                [
                    frozenset(self.patch_of[q] for q in r.qubits if q in self.patch_of)
                    for r in self.records
                ],
                [r.qubits for r in self.records],
                self.patch_of,
                self.centres,
                self.geometry,
            )
            self.units = sb._build_units(
                self.records, self.groups, self.layout, self.patch_of, self.geometry
            )
            self.layers = build_layers(self.units, circuit.num_qubits)

    @property
    def patches(self) -> List[int]:
        return sorted(set(self.patch_of.values()))

    @property
    def per_patch(self) -> int:
        return len(self.patch_of) // max(1, len(self.patches))

    def unit_of_layer(self, layer):
        return max(layer.units, key=lambda u: u.duration)

    def label_of(self, layer) -> str:
        """A short description of what sets this layer's duration."""
        slowest = self.unit_of_layer(layer)
        if slowest.transport is not None:
            left, right = slowest.transport.patches
            moved = len(slowest.transport.moved)
            gates = len(slowest.instructions)
            return (
                f"transport {LABELS[left]}<->{LABELS[right]} "
                f"({moved} atom{'s' if moved != 1 else ''}, "
                f"{slowest.transport.distance_cells:.0f} cells, "
                f"{gates} gate{'s' if gates != 1 else ''})"
            )
        record = self.records[slowest.instructions[0]]
        if record.kind == "measure":
            total = sum(
                len(u.instructions) for u in layer.units if u.fence
            )
            return f"readout ({total} qubits, one image)"
        if record.kind == "two_q" and slowest.duration > self.params.two_qubit_gate_time:
            return f"{record.name} + intra-patch move"
        return f"{record.name} x{len(layer.units)}"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def show_params(model: Model) -> None:
    params = model.params
    print(f"\n{'=' * 78}\nPARAMETERS")
    rows = (
        ("T1", params.T1, "s"),
        ("T2", params.T2, "s"),
        ("1q gate time", params.one_qubit_gate_time, "s"),
        ("2q gate time", params.two_qubit_gate_time, "s"),
        ("readout time", params.meas_time, "s"),
        ("1q error", params.one_qubit_error, ""),
        ("2q error", params.two_qubit_error, ""),
        ("readout error", params.readout_error, ""),
        ("atom pitch", params.atom_pitch, "m"),
        ("blockade radius", params.blockade_radius, "m"),
        ("movement speed", params.movement_speed, "m/s"),
        ("patch gutter", params.patch_gutter, "cells"),
        ("move error", params.move_error, "per atom per trip"),
    )
    for name, value, unit in rows:
        rendered = f"{value:g}" if isinstance(value, float) else str(value)
        print(f"  {name:>16}  {rendered:>10} {unit}")


def show_grid(model: Model) -> None:
    geometry = model.geometry
    print(f"\n{'=' * 78}\nATOM GRID")
    print(
        f"  {len(model.patches)} patches x {model.per_patch} atoms   "
        f"pitch {geometry.pitch * 1e6:.1f} um   gutter {geometry.gutter} cells   "
        f"blockade {geometry.blockade_radius * 1e6:.1f} um"
    )
    cells = {q: pos for q, pos in model.layout.items() if q in model.patch_of}
    if not cells:
        print("  (no patches)")
        return
    width = max(p[0] for p in cells.values()) + 1
    height = max(p[1] for p in cells.values()) + 1
    grid = [["." for _ in range(width)] for _ in range(height)]
    for qubit, (x, y) in cells.items():
        grid[y][x] = LABELS[model.patch_of[qubit] % len(LABELS)]

    print("\n        " + "".join(f"{x % 10}" for x in range(width)))
    for y, row in enumerate(grid):
        print(f"   {y:>3}  " + "".join(row))
    print()
    for patch in model.patches:
        owned = sorted(q for q, p in model.patch_of.items() if p == patch)
        centre = model.centres[patch]
        span = f"{owned[0]}-{owned[-1]}"
        print(
            f"   {LABELS[patch]} = patch {patch:<3} qubits {span:<8} "
            f"centre ({centre[0]:.1f}, {centre[1]:.1f})"
        )


def show_blockade(model: Model) -> None:
    geometry = model.geometry
    print(f"\n{'=' * 78}\nBLOCKADE  -- which intra-patch pairs gate without moving")
    patch = model.patches[0]
    owned = sorted(q for q, p in model.patch_of.items() if p == patch)
    pairs = list(itertools.combinations(owned, 2))
    beyond = [
        (a, b)
        for a, b in pairs
        if geometry.needs_transport(model.layout[a], model.layout[b])
    ]
    print(
        f"  patch {patch}: {len(pairs) - len(beyond)}/{len(pairs)} pairs within reach, "
        f"{len(beyond)} need a move  ->  effective alpha = "
        f"{len(beyond) / len(pairs):.2f}"
    )
    worst = max(
        pairs, key=lambda ab: euclidean(model.layout[ab[0]], model.layout[ab[1]])
    )
    span = euclidean(model.layout[worst[0]], model.layout[worst[1]])
    print(
        f"  widest pair q{worst[0]}-q{worst[1]}: {span:.2f} cells = "
        f"{span * geometry.pitch * 1e6:.1f} um "
        f"({'beyond' if span * geometry.pitch > geometry.blockade_radius else 'within'} "
        f"blockade)"
    )
    print(
        "  note: blockade is line-of-sight (Euclidean); travel is along rows and\n"
        "  columns (Manhattan), so a pair can be in reach but still cost a long move."
    )


def show_movements(model: Model) -> None:
    print(f"\n{'=' * 78}\nMOVEMENTS  -- cross-patch gates grouped into round trips")
    crossing = [
        r
        for r in model.records
        if len({model.patch_of.get(q) for q in r.qubits if q in model.patch_of}) == 2
    ]
    if not model.groups:
        print(f"  {len(crossing)} cross-patch gates, no movements derived")
        return
    saving = len(crossing) / len(model.groups)
    print(
        f"  {len(crossing)} cross-patch gates -> {len(model.groups)} movements "
        f"({saving:.1f}x fewer round trips than charging per gate)"
    )
    print(
        f"\n  {'#':>3}  {'patches':<9} {'gates':>6} {'atoms':>6} {'cells':>6} "
        f"{'round trip':>11}  split by"
    )
    for index, group in enumerate(model.groups):
        last = group.instructions[-1]
        following = model.records[last + 1] if last + 1 < len(model.records) else None
        reason = "-"
        if following is not None:
            touched = {
                model.patch_of.get(q) for q in following.qubits if q in model.patch_of
            }
            if touched & set(group.patches):
                reason = f"{following.name} on q{following.qubits[0]}"
        print(
            f"  {index:>3}  "
            f"{LABELS[group.patches[0]]}<->{LABELS[group.patches[1]]:<7} "
            f"{len(group.instructions):>6} {len(group.moved):>6} "
            f"{group.distance_cells:>6.0f} "
            f"{group.duration(model.geometry) * 1e6:>9.1f}us  {reason}"
        )


def show_schedule(model: Model, limit: int) -> None:
    total = sum(layer.duration for layer in model.layers)
    print(f"\n{'=' * 78}\nSCHEDULE  -- {len(model.layers)} layers, {total * 1e3:.2f} ms")
    print(
        "\n  'working' is qubits in some unit; 'dephasing' is qubits charged idle, which\n"
        "  includes working qubits whose unit finished before the layer did -- so the two\n"
        "  overlap and can sum past the register size."
    )
    print(
        f"\n  {'#':>4} {'duration':>10} {'working':>8} {'dephasing':>10}  "
        f"what sets the layer"
    )
    shown = model.layers[:limit]
    for index, layer in enumerate(shown):
        idle = idle_times_for_layer(layer, model.circuit.num_qubits)
        print(
            f"  {index:>4} {layer.duration * 1e6:>8.1f}us "
            f"{len(layer.busy):>8} {len(idle):>10}  {model.label_of(layer)}"
        )
    if len(model.layers) > limit:
        print(f"  ... {len(model.layers) - limit} more (use --layers to see them)")

    # Where the wall clock goes, which is not where the layer count goes.
    buckets: Dict[str, List[float]] = {}
    for layer in model.layers:
        slowest = model.unit_of_layer(layer)
        if slowest.transport is not None:
            tag = "transport"
        elif slowest.fence:
            tag = "readout"
        else:
            record = model.records[slowest.instructions[0]]
            tag = record.kind
            if tag == "two_q" and slowest.duration > model.params.two_qubit_gate_time:
                tag = "two_q + intra-patch move"
        entry = buckets.setdefault(tag, [0, 0.0])
        entry[0] += 1
        entry[1] += layer.duration
    print(f"\n  {'what sets the layer':<28} {'layers':>7} {'wall ms':>9} {'share':>7}")
    for tag, (count, wall) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
        print(f"  {tag:<28} {count:>7} {wall * 1e3:>9.3f} {100 * wall / total:>6.1f}%")


def budget(model: Model) -> Dict[str, float]:
    """
    Error by source. `idle` is split by *what made the layer long*, not by qubit.

    That split is the number worth having: idling is not an independent mechanism here,
    it is what the array does while something slow happens, so attributing it back to
    transport is how you find out what movement really costs. Idle seconds are priced at
    the simulator's own twirled thermal rate (`thermal_error_rate`), and the movement
    roll-up counts only distance-based charges -- see `measured_noise_split`.
    """
    from simulator.stim_backend import thermal_error_rate

    params = model.params
    attribution = idle_attribution(model.layers, model.circuit.num_qubits)
    flight_seconds = sum(
        unit.move_time * len(unit.moved)
        for layer in model.layers
        for unit in layer.units
        if unit.moved
    )
    moves = sum(
        len(unit.moved)
        for layer in model.layers
        for unit in layer.units
        if unit.moved
    )
    n1 = sum(1 for r in model.records if r.kind == "one_q" and r.noise)
    n2 = sum(1 for r in model.records if r.kind == "two_q" and r.noise)
    nm = sum(1 for r in model.records if r.kind == "measure" and r.noise)
    swaps = sum(1 for r in model.records if r.name == "swap" and r.noise)
    idle_rate = thermal_error_rate(params.T1, params.T2)
    return {
        "gates": n1 * params.one_qubit_error + (n2 - swaps) * params.two_qubit_error,
        "measurement": nm * params.readout_error,
        "idle (from transport)": attribution.get("transport", 0.0) * idle_rate,
        "idle (from readout)": attribution.get("readout", 0.0) * idle_rate,
        "idle (from gates)": attribution.get("gate", 0.0) * idle_rate,
        "move handling": moves * params.move_error,
        "flight dephasing": flight_seconds * idle_rate,
    }


def show_budget(model: Model) -> None:
    parts = budget(model)
    total = sum(parts.values())
    wall = sum(layer.duration for layer in model.layers)
    transport_wall = sum(
        unit.move_time
        for layer in model.layers
        for unit in layer.units
        if unit.transport is not None
    )
    print(f"\n{'=' * 78}\nBUDGET")
    for name, value in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>22}: {100 * value / total:>5.1f}%   E = {value:.4f}")
    print(
        f"  {'TOTAL':>22}: {'':>5}     E = {total:.4f}   "
        f"survival exp(-E) = {math.exp(-total):.3e}"
    )

    movement = parts["move handling"] + parts["flight dephasing"]
    idle = (
        parts["idle (from transport)"] + parts["idle (from gates)"]
        + parts["idle (from readout)"]
    )
    print(
        f"\n  distance-based noise (movement): {100 * movement / total:.1f}%  "
        f"(E = {movement:.4f})"
        f"\n  idling, all causes:              {100 * idle / total:.1f}%  "
        f"(E = {idle:.4f})"
    )
    print(
        f"    'flight dephasing' is only the atoms in the air, so it stays small and\n"
        f"    shrinks further as T2 improves. 'move handling' does not -- it is heating\n"
        f"    and handoff, charged per atom per carriage. 'idle (from transport)' is\n"
        f"    counted as idling, not movement, but it is transport's doing: transport is\n"
        f"    {100 * transport_wall / wall:.0f}% of the {wall * 1e3:.2f} ms wall clock, "
        f"and that is what the time does to the array."
    )


def show_stim(model: Model, limit: int) -> None:
    print(f"\n{'=' * 78}\nSTIM  -- emitted circuit with noise instructions")
    with sb._active_params(overrides_from(model.params)):
        emitted = sb.qiskit_to_stim(
            model.circuit, layout=model.layout, include_noise=True
        )
    lines = str(emitted).splitlines()
    noisy = {"PAULI_CHANNEL_1", "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR"}
    for index, line in enumerate(lines[:limit]):
        tag = "  <- noise" if line.strip().split(" ")[0] in noisy else ""
        print(f"  {index:>5}  {line}{tag}")
    if len(lines) > limit:
        print(f"  ... {len(lines) - limit} more instructions")
    counts: Dict[str, int] = {}
    for line in lines:
        head = line.strip().split(" ")[0]
        if head in noisy:
            counts[head] = counts.get(head, 0) + 1
    print("\n  noise instructions emitted:")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<18} {count}")


def show_sweep(args) -> None:
    """The budget across code sizes, and against a machine without decoupling."""
    print(f"\n{'=' * 78}\nSWEEP")
    na = StimNoiseParameters.neutral_atom()
    no_dd = StimNoiseParameters.neutral_atom()
    no_dd.T2 = 10e-3
    for label, params in (("T2 = 1.5 s (decoupled)", na), ("T2 = 10 ms (bare)", no_dd)):
        print(f"\n  {label}")
        print(
            f"    {'k':>3} {'patches':>8} {'moves':>6} {'wall ms':>9} "
            f"{'gates':>7} {'meas':>7} {'idle':>7} {'move':>7} {'E':>8}   "
            f"(move = distance-based only; idle includes what transport forces)"
        )
        for n_code in (4, 6, 8, 12):
            k = logical_capacity(n_code)
            circuit = build_circuit(n_code, args.qubits, args.seed, args.depth)
            model = Model(circuit, params)
            parts = budget(model)
            total = sum(parts.values())
            wall = sum(layer.duration for layer in model.layers)
            movement = parts["move handling"] + parts["flight dephasing"]
            columns = (
                parts["gates"],
                parts["measurement"],
                parts["idle (from gates)"] + parts["idle (from readout)"]
                + parts["idle (from transport)"],
                movement,
            )
            print(
                f"    {k:>3} {len(model.patches):>8} {len(model.groups):>6} "
                f"{wall * 1e3:>9.2f} "
                + "".join(f"{100 * value / total:>6.1f}%" for value in columns)
                + f" {total:>8.3f}"
            )


def show_balance(args) -> None:
    """
    Nominal `distance_scalar` against the movement share it actually buys.

    This is the honest way to build a noise-ratio axis. Sweeping a scalar and *labelling*
    the axis with that scalar asserts a balance the model does not have -- the mapping
    from scalar to balance depends on the circuit, the packing and `k`, so a nominal
    "ratio = 1" is a different physical point for every column. Sweeping the scalar to
    generate spread and then plotting against the **measured** movement share keeps the
    spread and drops the fiction.

    Averaged over the code sizes below, which is the regime the k figures compare across.
    Pick scalars whose mean movement share spans the range you want on the x axis.
    """
    from experiments.shared.simulation import measured_noise_split

    params = StimNoiseParameters.neutral_atom()
    sizes = [(n, args.qubits) for n in (4, 6, 8, 12)]
    circuits = [(n, build_circuit(n, q, args.seed, args.depth)) for n, q in sizes]

    print(f"\n{'=' * 78}\nBALANCE  -- nominal knob vs measured movement share")
    print(f"\n  {'distance_scalar':>16} {'mean move %':>12} {'spread over k':>28}")
    for scalar in (0.0625, 0.25, 1.0, 4.0, 16.0, 64.0):
        shares = []
        for _, circuit in circuits:
            split = measured_noise_split(circuit, params, distance_scalar=scalar)
            shares.append(split["movement"] / split["total"])
        mean = sum(shares) / len(shares)
        detail = "  ".join(
            f"k={logical_capacity(n)}:{100 * s:.0f}%"
            for (n, _), s in zip(sizes, shares)
        )
        print(f"  {scalar:>16g} {100 * mean:>11.1f}% {detail:>28}")
    print(
        "\n  The spread across k at a fixed scalar is the confound: movement share is\n"
        "  partly a restatement of patch count, and patch count is set by k. Averaging\n"
        "  over circuits does not remove it -- only plotting against the measured value\n"
        "  does, and even then the k columns sample different parts of the axis."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4, help="code size n (k = n - 2)")
    parser.add_argument("--qubits", type=int, default=12, help="logical qubits")
    parser.add_argument("--depth", type=int, default=14, help="program depth")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--layers", type=int, default=25, help="schedule rows to print")
    parser.add_argument("--stim", action="store_true", help="dump the emitted circuit")
    parser.add_argument("--stim-lines", type=int, default=60)
    parser.add_argument("--sweep", action="store_true", help="budget across code sizes")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="nominal distance_scalar vs the movement share it actually buys",
    )
    parser.add_argument(
        "--superconducting",
        action="store_true",
        help="use the legacy parameter defaults instead of the NA preset",
    )
    args = parser.parse_args()

    if args.sweep:
        show_sweep(args)
        return
    if args.balance:
        show_balance(args)
        return

    # `StimNoiseParameters()` would be the NA defaults again -- the flag needs the
    # named superconducting preset to actually change anything.
    params = (
        StimNoiseParameters.superconducting()
        if args.superconducting
        else StimNoiseParameters.neutral_atom()
    )
    circuit = build_circuit(args.n_code, args.qubits, args.seed, args.depth)
    model = Model(circuit, params)

    k = logical_capacity(args.n_code)
    print(
        f"\nk = {k}  (n = {args.n_code}), {args.qubits} logical qubits, "
        f"{circuit.num_qubits} physical, "
        f"{'superconducting' if args.superconducting else 'neutral atom'} parameters"
    )
    show_params(model)
    show_grid(model)
    show_blockade(model)
    show_movements(model)
    show_schedule(model, args.layers)
    show_budget(model)
    if args.stim:
        show_stim(model, args.stim_lines)
    print()


if __name__ == "__main__":
    main()
