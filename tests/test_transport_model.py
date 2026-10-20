"""
The neutral-atom transport model: geometry, grouping, and the schedule.

These cover the rules `simulator.transport` documents, and two regressions found while
building it — both in the schedule rather than the physics, and both silent.
"""

from __future__ import annotations

import itertools

import pytest

from simulator.grid_constructor import (
    build_patch_centres,
    build_physical_qubit_positions,
)
from simulator.noise_models import StimNoiseParameters
from simulator.scheduler import (
    ScheduleUnit,
    build_layers,
    idle_attribution,
    idle_times_for_layer,
)
from simulator.transport import (
    Geometry,
    derive_transport_groups,
    manhattan,
    most_square_rectangle,
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_rectangle_prefers_square_over_minimal_area():
    """
    14 atoms fit 2x7 exactly and waste two cells in 4x4, and the tie-break matters.

    Area-first gives k=10 an elongated patch whose internal distances grow like `n`; the
    k=8 / k=10 ordering flips on this choice alone. See the module docstring.
    """
    assert most_square_rectangle(14) == (4, 4)
    assert most_square_rectangle(6) == (2, 3)
    assert most_square_rectangle(8) == (3, 3)


@pytest.mark.parametrize("atoms", range(1, 33))
def test_rectangle_always_fits(atoms):
    rows, columns = most_square_rectangle(atoms)
    assert rows * columns >= atoms
    assert rows <= columns


def test_every_atom_gets_its_own_cell():
    """The bug this model exists to fix: a patch used to be a dimensionless point."""
    coords = build_physical_qubit_positions(4, qubits_per_patch=6)
    assert len(set(coords.values())) == len(coords) == 24


def test_patches_are_separated_by_the_gutter():
    geometry = Geometry(gutter=2)
    coords = build_physical_qubit_positions(2, qubits_per_patch=6, geometry=geometry)
    first = [coords[q] for q in range(6)]
    second = [coords[q] for q in range(6, 12)]
    gap = min(
        manhattan(a, b) for a, b in itertools.product(first, second)
    )
    # `gutter` counts *empty cells*, so nearest atoms are one step further than that.
    assert gap == geometry.gutter + 1


def test_blockade_penalises_large_patches_and_not_small_ones():
    """
    The mechanism that replaces a tuned intra-patch discount.

    A k=2 patch fits inside one blockade sphere and genuinely needs no internal movement.
    A k=10 patch does not, so a growing share of its own gates start paying transport.
    """
    geometry = Geometry()

    def outside_fraction(qubits_per_patch: int) -> float:
        coords = build_physical_qubit_positions(1, qubits_per_patch=qubits_per_patch)
        cells = list(coords.values())
        pairs = list(itertools.combinations(cells, 2))
        beyond = sum(1 for a, b in pairs if geometry.needs_transport(a, b))
        return beyond / len(pairs)

    assert outside_fraction(6) == 0.0           # k=2
    assert outside_fraction(14) > 0.2           # k=10
    assert outside_fraction(6) < outside_fraction(10) < outside_fraction(16)


def test_round_trip_time_is_there_and_back():
    geometry = Geometry(pitch=3e-6, movement_speed=0.2)
    assert geometry.round_trip_time(10) == pytest.approx(2 * 10 * 3e-6 / 0.2)


def test_blockade_is_line_of_sight_and_travel_is_not():
    """
    Using Manhattan for the blockade would shrink it by up to sqrt(2), which is enough to
    start charging transport inside a k=2 patch -- the one patch that should be free.
    """
    geometry = Geometry(pitch=3e-6, blockade_radius=8e-6)
    corner_to_corner = ((0, 0), (2, 1))
    assert manhattan(*corner_to_corner) * 3e-6 > 8e-6      # Manhattan would charge it
    assert not geometry.needs_transport(*corner_to_corner)  # line of sight does not


def test_patch_centres_sit_inside_their_rectangles():
    centres = build_patch_centres(4, qubits_per_patch=6)
    coords = build_physical_qubit_positions(4, qubits_per_patch=6)
    for patch, centre in centres.items():
        cells = [coords[q] for q in range(patch * 6, patch * 6 + 6)]
        assert min(c[0] for c in cells) <= centre[0] <= max(c[0] for c in cells)
        assert min(c[1] for c in cells) <= centre[1] <= max(c[1] for c in cells)


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


CENTRES = {0: (0.0, 0.0), 1: (5.0, 0.0), 2: (0.0, 5.0)}
#: Patch 0 owns qubits 0-1, patch 1 owns 2-3, patch 2 owns 4-5.
PATCH_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}


def _groups(instructions):
    """`instructions` is a list of qubit tuples; patches are derived from PATCH_OF."""
    patches = [frozenset(PATCH_OF[q] for q in qubits) for qubits in instructions]
    return derive_transport_groups(
        patches, instructions, PATCH_OF, CENTRES, Geometry()
    )


def test_consecutive_crossings_share_one_movement():
    """
    The whole point: a logical cross-patch CNOT emits four physical gates spanning the
    same two patches, and it should cost one round trip rather than four.
    """
    groups = _groups([(0, 2), (0, 3), (1, 2), (1, 3)])
    assert len(groups) == 1
    assert groups[0].instructions == [0, 1, 2, 3]
    assert groups[0].patches == (0, 1)


def test_a_gate_on_either_patch_ends_the_group():
    """The conservative rule the model commits to: atoms come home before anything else."""
    groups = _groups([(0, 2), (0, 3), (1,), (1, 2), (1, 3)])
    assert [g.instructions for g in groups] == [[0, 1], [3, 4]]


def test_work_on_an_unrelated_patch_does_not_end_the_group():
    """Disjoint patches run concurrently -- otherwise nothing is ever parallel."""
    groups = _groups([(0, 2), (4, 5), (0, 3)])
    assert [g.instructions for g in groups] == [[0, 2]]


def test_a_crossing_that_shares_a_patch_ends_the_group():
    groups = _groups([(0, 2), (0, 4), (0, 3)])
    assert [g.patches for g in groups] == [(0, 1), (0, 2), (0, 1)]


def test_control_side_atoms_are_the_ones_that_travel():
    """Every cross-patch construction emits control-first, so qubit 0 names the mover."""
    groups = _groups([(0, 2), (0, 3), (1, 2), (1, 3)])
    assert sorted(groups[0].moved) == [0, 1]


def test_distance_is_between_patch_centres():
    groups = _groups([(0, 2)])
    assert groups[0].distance_cells == manhattan((0, 0), (5, 0))


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def _unit(qubits, duration, order, **kwargs):
    return ScheduleUnit(
        qubits=frozenset(qubits), duration=duration, order=order, **kwargs
    )


def _encoded_circuit(n_code: int, num_qubits: int = 8, seed: int = 11):
    """First random program and placement the translator accepts at this code size."""
    import random

    from iceberg.utils import logical_capacity
    from permutations.generator import random_permutation_from_circuit
    from tests.random_programs import random_clifford_program
    from translator.config import TranslatorConfig
    from translator.core import Translator

    k = logical_capacity(n_code)
    translator = Translator(config=TranslatorConfig(n_code=n_code, fault_tolerant=False))
    for attempt in range(60):
        rng = random.Random(seed + attempt)
        program = random_clifford_program(
            num_qubits, 12, rng=rng, include_xcx=True, measure=True
        )
        permutation = random_permutation_from_circuit(
            program, [n_code] * (num_qubits // k + 2), seed=seed + attempt
        )
        try:
            return translator.translate(program, permutation).circuit
        except RuntimeWarning:
            continue
    pytest.skip(f"no placement accepted at n={n_code}")


def test_disjoint_units_share_a_layer_and_overlapping_ones_do_not():
    layers = build_layers(
        [_unit([0], 1.0, 0), _unit([1], 1.0, 1), _unit([0], 1.0, 2)], 2
    )
    assert [len(layer.units) for layer in layers] == [2, 1]


def test_layer_lasts_as_long_as_its_slowest_unit():
    layers = build_layers([_unit([0], 1.0, 0), _unit([1], 7.0, 1)], 2)
    assert layers[0].duration == 7.0


def test_bystanders_are_charged_the_whole_layer():
    """
    The term the old scheduler could not see at all.

    A 100 us transport is 100 us of dephasing for every atom in the array, not just the
    two in flight -- that is where the cost of movement actually lands.
    """
    layers = build_layers([_unit([0, 1], 10.0, 0)], 5)
    idle = dict(idle_times_for_layer(layers[0], 5))
    assert idle == {2: 10.0, 3: 10.0, 4: 10.0}


def test_a_unit_that_finishes_early_idles_for_the_remainder():
    layers = build_layers([_unit([0], 2.0, 0), _unit([1], 10.0, 1)], 2)
    idle = dict(idle_times_for_layer(layers[0], 2))
    assert idle == {0: 8.0}


def test_readout_batches_into_one_layer():
    """
    One global image covers the array, so a detection round costs `t_meas` once.

    Laid out independently these staggered across seven layers and three quarters of the
    modelled circuit time was re-imaging the same array.
    """
    units = [_unit([q], 3e-3, q, fence=True) for q in range(8)]
    layers = build_layers(units, 8)
    assert len(layers) == 1
    assert layers[0].duration == 3e-3


def test_readout_waits_for_the_qubits_it_images():
    units = [_unit([0], 1.0, 0)] + [
        _unit([q], 3e-3, q + 1, fence=True) for q in range(4)
    ]
    layers = build_layers(units, 4)
    assert len(layers) == 2
    assert all(unit.fence for unit in layers[1].units)


def test_measurement_order_survives_scheduling():
    """
    Regression. The sampler returns results in append order and `run_stim_backend` maps
    them back positionally, so a measurement floating into an earlier layer than one that
    precedes it silently relabels the bits -- it looks like a valid distribution.
    """
    busy = _unit([0], 5.0, 0)
    late_on_free_qubit = _unit([1], 1.0, 2, fence=True)
    early_on_busy_qubit = _unit([0], 1.0, 1, fence=True)
    layers = build_layers([busy, early_on_busy_qubit, late_on_free_qubit], 2)
    emitted = [unit.order for layer in layers for unit in layer.units if unit.fence]
    assert emitted == sorted(emitted)


@pytest.mark.parametrize("n_code", [4, 6, 8, 12])
def test_a_real_circuit_images_the_array_exactly_once(n_code):
    """
    End to end: this pipeline does one detection round, so it must cost one image.

    Scheduled independently these staggered across seven layers on a k=2 circuit, and
    because a layer is `t_meas` long whether it images one qubit or fifty, three quarters
    of the modelled circuit time was re-reading the same array -- which inflated every
    other qubit's idle charge along with it.
    """
    from iceberg.layout import patch_of_qubit
    from simulator import stim_backend as sb
    from simulator.grid_constructor import centres_for_circuit, positions_for_circuit
    from simulator.noise_models import StimNoiseParameters

    circuit = _encoded_circuit(n_code)
    params = StimNoiseParameters.neutral_atom()
    with sb._active_params({"t_meas": params.meas_time, "t_2q": params.two_qubit_gate_time}):
        geometry = sb._geometry()
        records = sb._scan(circuit)
        patch_map = patch_of_qubit(circuit)
        groups = derive_transport_groups(
            [frozenset(patch_map[q] for q in r.qubits if q in patch_map) for r in records],
            [r.qubits for r in records],
            patch_map,
            centres_for_circuit(circuit, geometry=geometry),
            geometry,
        )
        units = sb._build_units(
            records,
            groups,
            positions_for_circuit(circuit, geometry=geometry),
            patch_map,
            geometry,
        )
        layers = build_layers(units, circuit.num_qubits)

    measured = sum(1 for r in records if r.kind == "measure")
    readout_layers = [layer for layer in layers if any(u.fence for u in layer.units)]
    assert measured > 0
    assert len(readout_layers) == 1
    assert sum(len(u.instructions) for u in readout_layers[0].units) == measured
    # And it is the last thing that happens.
    assert readout_layers[0] is layers[-1]


def test_idle_attribution_is_marginal_not_gross():
    """
    A slow unit is charged only what it adds over what the layer would have cost anyway.

    Gross attribution would credit transport with the whole layer even when a gate in it
    was nearly as slow, which overstates what making movement free would buy -- and that
    counterfactual is the entire point of the split.
    """
    layers = build_layers(
        [
            _unit([0, 1], 10.0, 0, cause="transport"),
            _unit([2, 3], 8.0, 1, cause="gate"),
        ],
        6,
    )
    attributed = idle_attribution(layers, 6)
    # As scheduled: q4/q5 sit out all 10, the gate pair sits out 2 -> 24 qubit-seconds.
    # Without the transport the layer runs 8, so q4/q5 would still sit out 8 each -> 16.
    # Transport therefore owns the 8 it added, spread over the bystanders *and* the gate
    # pair it kept waiting -- gross attribution would have charged it all 24.
    assert attributed["transport"] == pytest.approx(8.0)
    assert attributed["gate"] == pytest.approx(16.0)


def test_idle_attribution_totals_the_idle_it_splits():
    layers = build_layers(
        [
            _unit([0], 5.0, 0, cause="transport"),
            _unit([1], 1.0, 1, cause="gate"),
        ],
        4,
    )
    attributed = idle_attribution(layers, 4)
    charged = sum(seconds for _, seconds in idle_times_for_layer(layers[0], 4))
    assert sum(attributed.values()) == pytest.approx(charged)


def test_a_layer_where_everyone_works_attributes_nothing():
    """Readout images the whole array at once, so a detection round charges no idle."""
    units = [_unit([q], 3e-3, q, fence=True, cause="readout") for q in range(4)]
    attributed = idle_attribution(build_layers(units, 4), 4)
    assert sum(attributed.values()) == pytest.approx(0.0)


def test_move_handling_error_lands_on_the_carried_atoms_only():
    """
    Heating and handoff, charged per atom per round trip -- and unlike flight dephasing
    it does not shrink as T2 improves, which is what keeps movement visible at T2 = 1.5 s.
    """
    from simulator import stim_backend as sb

    circuit = _encoded_circuit(4)
    params = StimNoiseParameters.neutral_atom()
    overrides = {
        "T1": params.T1,
        "T2": params.T2,
        "t_2q": params.two_qubit_gate_time,
        "t_meas": params.meas_time,
        "p1": 0.0,
        "p2": 0.0,
        "p_meas": 0.0,
        "move_error": 0.25,          # unmistakable in the emitted text
        "idle_time_scalar": 0.0,     # isolate the move charge from dephasing
    }
    from simulator.grid_constructor import positions_for_circuit

    with sb._active_params(overrides):
        layout = positions_for_circuit(circuit, geometry=sb._geometry())
        emitted = str(sb.qiskit_to_stim(circuit, layout=layout, include_noise=True))
    charges = [line for line in emitted.splitlines() if "DEPOLARIZE1(0.25)" in line]
    assert charges, "no move-handling error emitted"

    with sb._active_params({**overrides, "move_error": 0.0}):
        layout = positions_for_circuit(circuit, geometry=sb._geometry())
        without = str(sb.qiskit_to_stim(circuit, layout=layout, include_noise=True))
    assert "DEPOLARIZE1(0.25)" not in without


def test_nothing_to_schedule_is_not_an_error():
    assert build_layers([], 4) == []
    # A unit owning no qubits is dropped rather than occupying a layer.
    assert build_layers([_unit([], 1.0, 0)], 4) == []


def test_a_zero_length_layer_charges_no_idle():
    layers = build_layers([_unit([0], 0.0, 0)], 4)
    assert idle_times_for_layer(layers[0], 4) == []


# --------------------------------------------------------------------------
# noise-accounting regressions (the review round)
# --------------------------------------------------------------------------


def _two_patch_circuit(n: int = 4):
    """Two hand-built patches with the register names `patch_of_qubit` reads."""
    from qiskit import QuantumCircuit, QuantumRegister

    registers = []
    for patch in range(2):
        registers.append(QuantumRegister(n, f"data_{patch}"))
        registers.append(QuantumRegister(2, f"parity_{patch}"))
    return QuantumCircuit(*registers)


def _model_pieces(circuit, overrides=None):
    from iceberg.layout import patch_of_qubit
    from simulator import stim_backend as sb
    from simulator.grid_constructor import centres_for_circuit, positions_for_circuit
    from simulator.transport import derive_transport_groups

    with sb._active_params(overrides or {}):
        geometry = sb._geometry()
        layout = positions_for_circuit(circuit, geometry=geometry)
        patch_map = patch_of_qubit(circuit)
        records = sb._scan(circuit)
        groups = derive_transport_groups(
            [frozenset(patch_map[q] for q in r.qubits if q in patch_map) for r in records],
            [r.qubits for r in records],
            patch_map,
            centres_for_circuit(circuit, geometry=geometry),
            geometry,
        )
        units = sb._build_units(records, groups, layout, patch_map, geometry)
    return records, units, build_layers(units, circuit.num_qubits)


def test_bystanders_inside_a_held_patch_still_dephase():
    """
    Regression. A transport group occupies both whole patches so the scheduler blocks
    them -- but occupancy is not an exemption from decoherence. The stationary atoms of
    the held patches used to be charged zero idle for the longest layers in the schedule,
    which under-charged movement in proportion to patch size.
    """
    circuit = _two_patch_circuit()
    circuit.cx(0, 6)  # data_0[0] -> data_1[0]: one crossing, one group
    records, units, layers = _model_pieces(circuit)

    (unit,) = units
    assert unit.transport is not None and unit.move_time > 0
    idle = dict(idle_times_for_layer(layers[0], circuit.num_qubits))
    duration = layers[0].duration

    # The mover works the whole unit; its gate partner idles through the flight; every
    # other atom of the two held patches idles the entire layer.
    assert 0 not in idle
    assert idle[6] == pytest.approx(unit.move_time)
    for bystander in set(range(circuit.num_qubits)) - {0, 6}:
        assert idle[bystander] == pytest.approx(duration)


def test_intra_patch_move_charges_the_carried_atom():
    """
    Regression. An intra-patch beyond-blockade gate paid time but no error: the carried
    atom now pays `move_error` and flight dephasing exactly like a cross-patch mover.
    """
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    circuit = _two_patch_circuit(n=14)  # 4x4 patch: corner pairs sit beyond blockade
    circuit.cx(0, 15)  # data_0[0] at (0,0) to data_0[15] at (3,3)
    overrides = {"p1": 0.0, "p2": 0.0, "p_meas": 0.0, "move_error": 0.25,
                 "idle_time_scalar": 0.0}
    records, units, layers = _model_pieces(circuit, overrides)

    (unit,) = units
    assert unit.transport is None
    assert unit.moved == (0,)
    assert unit.move_time > 0
    assert unit.duration == pytest.approx(sb.NOISE_PARAMS["t_2q"] + unit.move_time)

    with sb._active_params(overrides):
        layout = positions_for_circuit(circuit, geometry=sb._geometry())
        emitted = str(sb.qiskit_to_stim(circuit, layout=layout, include_noise=True))
    assert "DEPOLARIZE1(0.25) 0" in emitted


def test_swap_is_an_atom_exchange():
    """
    Regression. A physical swap was a free, noiseless unitary. It is an atom exchange:
    both atoms are carried the separation once -- one-way flight, `move_error` each, and
    no two-qubit depolarizing because there is no pulse.
    """
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit
    from simulator.transport import manhattan

    circuit = _two_patch_circuit()
    circuit.swap(0, 3)
    overrides = {"p1": 0.0, "p2": 0.5, "p_meas": 0.0, "move_error": 0.25,
                 "idle_time_scalar": 0.0}
    records, units, layers = _model_pieces(circuit, overrides)

    (unit,) = units
    assert sorted(unit.moved) == [0, 3]
    with sb._active_params(overrides):
        geometry = sb._geometry()
        layout = positions_for_circuit(circuit, geometry=geometry)
        expected = geometry.one_way_time(manhattan(layout[0], layout[3]))
        emitted = str(sb.qiskit_to_stim(circuit, layout=layout, include_noise=True))
    assert unit.duration == pytest.approx(expected)
    assert unit.move_time == pytest.approx(expected)

    assert "DEPOLARIZE1(0.25) 0 3" in emitted   # the carriage
    assert "DEPOLARIZE2" not in emitted          # no pulse to depolarize

    # Without a layout the exchange cannot be priced, so it falls back to gate pricing
    # rather than being free.
    with sb._active_params(overrides):
        bare = str(sb.qiskit_to_stim(circuit, layout=None, include_noise=True))
    assert "DEPOLARIZE2(0.5) 0 3" in bare


def test_fenced_batches_never_reorder_across_flushes():
    """
    Regression. A mid-circuit flush pinned batch 1 at some slot; a later batch on cold
    qubits could land in an *earlier* slot, and the sampler's positional mapping then
    silently relabelled the bits. The fence floor keeps flushed batches in circuit order.
    """
    units = [
        _unit([0], 1.0, 0),               # busy q0
        _unit([0], 1.0, 1, fence=True),   # measure q0 (pends)
        _unit([0], 1.0, 2),               # gate on q0 -> flushes batch 1 at slot 1
        _unit([1], 1.0, 3, fence=True),   # measure q1: cold qubit, must not precede
    ]
    layers = build_layers(units, 2)
    emitted = [unit.order for layer in layers for unit in layer.units if unit.fence]
    assert emitted == sorted(emitted)


def test_move_error_scales_with_alpha():
    """
    Regression. `move_error` sat at its default whatever alpha was, so 'noiseless'
    models still depolarized carried atoms, and the per-circuit calibration -- which
    models selection rate as exponential in alpha -- stalled against the constant term.
    """
    from simulator.noise_models import BASE_MOVE_ERROR, ScaledNoiseModel

    assert ScaledNoiseModel(0.0)._stim_params.move_error == 0.0
    assert ScaledNoiseModel(1.0)._stim_params.move_error == pytest.approx(BASE_MOVE_ERROR)
    assert ScaledNoiseModel(4.0)._stim_params.move_error == pytest.approx(4 * BASE_MOVE_ERROR)


def test_a_noiseless_model_emits_no_noise_even_with_a_layout():
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit
    from simulator.noise_models import ScaledNoiseModel, stim_params_from_noise_model

    circuit = _two_patch_circuit()
    circuit.cx(0, 6)
    circuit.swap(0, 3)
    params = stim_params_from_noise_model(ScaledNoiseModel(0.0))
    overrides = {"p1": params.one_qubit_error, "p2": params.two_qubit_error,
                 "p_meas": params.readout_error, "T1": params.T1, "T2": params.T2,
                 "move_error": params.move_error, "idle_time_scalar": 1.0}
    with sb._active_params(overrides):
        layout = positions_for_circuit(circuit, geometry=sb._geometry())
        emitted = str(sb.qiskit_to_stim(circuit, layout=layout, include_noise=True))
    for channel in ("DEPOLARIZE", "PAULI_CHANNEL", "X_ERROR"):
        assert channel not in emitted


def test_thermal_error_rate_matches_the_emitted_channel():
    """The analytic split must price idle seconds at the simulator's own twirl rate."""
    from simulator.stim_backend import _pauli_twirl_approx, thermal_error_rate

    T1, T2, t = 10.0, 1.5, 1e-4
    _, p_x, p_y, p_z = _pauli_twirl_approx(t, T1, T2)
    assert p_x + p_y + p_z == pytest.approx(t * thermal_error_rate(T1, T2), rel=1e-3)


def test_split_movement_means_distance_based_noise_only():
    """`movement` = move handling + flight; transport-forced idling stays under idle."""
    from experiments.shared.simulation import measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    circuit = _two_patch_circuit()
    circuit.cx(0, 6)
    circuit.measure_all()
    split = measured_noise_split(circuit, StimNoiseParameters.neutral_atom())
    assert split["movement"] == pytest.approx(
        split["move_handling"] + split["flight_dephasing"]
    )
    assert split["idle"] == pytest.approx(
        split["idle_gates"] + split["idle_transport"] + split["idle_readout"]
    )
