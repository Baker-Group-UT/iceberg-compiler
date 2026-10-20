"""
attribution.py -- where an encoded circuit's error actually comes from.

Lives here rather than in `experiments/shared` because every line of it reads this
package's own internals -- `stim_backend`, `grid_constructor`, `scheduler`, `transport`,
`noise_models` -- and because `packing` needs it. The placement objective has to be able
to price the schedule the simulator will really run; when it could not, it optimised
crossing *counts* while the simulator charged *distance*, and `plan.measured_cost` signed
off on placements that cost up to 3.4x the movement they were credited with.

`experiments.shared.simulation` re-exports `measured_noise_split` so existing callers keep
their import path.
"""

from __future__ import annotations

from typing import Dict

def measured_noise_split(
    circuit, params=None, *, idle_time_scalar: float = 1.0,
    distance_scalar: float = 1.0,
) -> Dict[str, float]:
    """
    Where an encoded circuit's error actually comes from, in expected detectable errors.

    This is the honest replacement for the nominal idle/distance ratio: it reads the
    schedule the simulator will really run and attributes every charge to its source,
    rather than asserting a balance from two scalars. Use it to *label* a sweep, or as the
    sweep variable itself.

    Three conventions to know:

    * Idle seconds are priced at the simulator's own twirled thermal rate,
      `stim_backend.thermal_error_rate` -- dividing by T2 alone overstated every
      idle-family term ~2x against the exact probabilities used for gates.
    * Idle is split by what made each layer long, and marginally -- transport is credited
      only with the idling it adds over what the layer would have cost anyway. See
      `simulator.scheduler.idle_attribution`.
    * The ``movement`` roll-up is **distance-based noise only**: move handling plus
      flight dephasing -- the charges on the atoms actually carried. Idling that
      transport forces on the rest of the array stays in the ``idle`` roll-up under
      ``idle_transport``: it is idling, whatever made the layer long.

    Returns keys: ``gates``, ``measurement``, ``idle_gates``, ``idle_transport``,
    ``idle_readout``, ``move_handling``, ``flight_dephasing``, plus three roll-ups --
    ``movement``, ``idle`` and ``total``.
    """
    from iceberg.layout import patch_of_qubit
    from simulator import stim_backend as sb
    from simulator.grid_constructor import centres_for_circuit, positions_for_circuit
    from simulator.noise_models import StimNoiseParameters
    from simulator.scheduler import build_layers, idle_attribution
    from simulator.stim_backend import thermal_error_rate
    from simulator.transport import derive_transport_groups

    params = params or StimNoiseParameters()
    overrides = {
        "T1": params.T1, "T2": params.T2,
        "t_1q": params.one_qubit_gate_time, "t_2q": params.two_qubit_gate_time,
        "t_meas": params.meas_time, "p1": params.one_qubit_error,
        "p2": params.two_qubit_error, "p_meas": params.readout_error,
        "atom_pitch": params.atom_pitch, "blockade_radius": params.blockade_radius,
        "movement_speed": params.movement_speed, "patch_gutter": params.patch_gutter,
        "move_error": params.move_error,
        "idle_time_scalar": idle_time_scalar, "distance_scalar": distance_scalar,
    }

    with sb._active_params(overrides):
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
        layers = build_layers(units, circuit.num_qubits)
        attributed = idle_attribution(layers, circuit.num_qubits)
        move_error = min(1.0, params.move_error * distance_scalar)

    # `moved` covers every carriage: transport groups, intra-patch beyond-blockade
    # movers, and both ends of a swap-by-exchange.
    flight = sum(
        unit.move_time * len(unit.moved)
        for layer in layers for unit in layer.units if unit.moved
    )
    carried = sum(
        len(unit.moved)
        for layer in layers for unit in layer.units if unit.moved
    )
    # Count only records the simulator will actually make noisy. Translated circuits
    # carry no noise markers today, so this matches a plain count -- but the emission
    # honours `record.noise`, and this stays in lockstep if markers return.
    counts = {kind: sum(1 for r in records if r.kind == kind and r.noise)
              for kind in ("one_q", "two_q", "measure")}
    swaps = sum(1 for r in records if r.name == "swap" and r.noise)
    idle_rate = thermal_error_rate(params.T1, params.T2)

    split = {
        # Swaps are atom exchanges: no pulse, so no p2 -- their cost is in the carriage.
        "gates": (counts["one_q"] * params.one_qubit_error
                  + (counts["two_q"] - swaps) * params.two_qubit_error),
        "measurement": counts["measure"] * params.readout_error,
        "idle_gates": attributed.get("gate", 0.0) * idle_time_scalar * idle_rate,
        "idle_transport": attributed.get("transport", 0.0) * idle_time_scalar * idle_rate,
        "idle_readout": attributed.get("readout", 0.0) * idle_time_scalar * idle_rate,
        "move_handling": carried * move_error,
        "flight_dephasing": flight * idle_rate,
    }
    split["movement"] = split["move_handling"] + split["flight_dephasing"]
    split["idle"] = (
        split["idle_gates"] + split["idle_transport"] + split["idle_readout"]
    )
    split["total"] = sum(
        split[key] for key in
        ("gates", "measurement", "idle_gates", "idle_transport", "idle_readout",
         "move_handling", "flight_dephasing")
    )
    return split


def clifford_stand_in(circuit):
    """
    `circuit` with every one-qubit gate Stim cannot take (rz, rx, t, ...) replaced by S.

    The backend schedules and charges every one-qubit gate alike -- one `t_1q` slot and one
    `p1` depolarisation whatever the angle -- so the stand-in has the same schedule and the
    same noise charges, and `measured_noise_split` can price an encoded circuit that carries
    rotations instead of raising. For pricing only: it computes a different logical function.
    Returns `circuit` itself when nothing needs replacing.
    """
    from qiskit.circuit.library import SGate

    from simulator.stim_backend import ONE_Q

    def foreign(instruction) -> bool:
        name = instruction.operation.name.lower()
        return (len(instruction.qubits) == 1 and not instruction.clbits
                and name not in ONE_Q and name not in ("barrier", "reset"))

    if not any(foreign(instruction) for instruction in circuit.data):
        return circuit
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        if foreign(instruction):
            out.append(SGate(), instruction.qubits)
        else:
            out.append(instruction)
    return out
