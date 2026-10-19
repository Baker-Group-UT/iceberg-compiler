"""Mid-circuit syndrome rounds (`TranslatorConfig.syndrome_rounds`) and the lookup decoder."""

import numpy as np
import pytest
import stim
from qiskit import QuantumCircuit

from experiments.concatenation import construct as C
from experiments.concatenation import decode as D
from experiments.concatenation import lookup as L
from experiments.concatenation.concat_eval import _tvd
from experiments.revision.common import exact_ideal_distribution
from experiments.shared.simulation import ensure_measurements
from translator.core import round_cuts


def _program(seed: int = 1, gates: int = 30) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    program = QuantumCircuit(4)
    for _ in range(gates):
        g = rng.integers(4)
        a, b = (int(v) for v in rng.choice(4, 2, replace=False))
        if g == 0:
            program.h(a)
        elif g == 1:
            program.s(a)
        elif g == 2:
            program.x(a)
        else:
            program.cx(a, b)
    return program


def _noiseless(result, maps, kind, program):
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    circuit = result.circuit
    clean = sb.qiskit_to_stim(circuit, layout=positions_for_circuit(circuit, geometry=sb._geometry()),
                              include_noise=False)
    decoded = (D.decode if kind == "two" else D.decode_single)(
        clean.compile_sampler(seed=3).sample(20_000), maps)
    tvd, _ = _tvd(decoded["postselect"], decoded["postselect_out"], exact_ideal_distribution(program))
    return float(decoded["postselect"].mean()), tvd


def _two_level(program, rounds):
    two = C.compile_two_level(program, [(i // 2, i % 2) for i in range(4)], outer_commute=True,
                              outer_merging=True, inner_layout="stride", inner_commute=True,
                              inner_merging=True)
    two.inner = C.inner_translate(two.flat, two.inner_placement, commute=True, merging=True,
                                  rounds=rounds)
    return two


def test_round_cuts_split_layers_evenly():
    program = QuantumCircuit(1)
    for _ in range(9):
        program.h(0)
    assert round_cuts(program, 0) == set()
    assert round_cuts(program, 2) == {2, 5}          # boundaries at layers 3 and 6
    assert round_cuts(program, 20) == set(range(8))  # at most one per boundary, none at the end


@pytest.mark.parametrize("rounds", [1, 3])
def test_rounds_keep_every_shot_and_the_output_noiselessly(rounds):
    program = _program()
    one = C.translator(4, commute=True, merging=True, rounds=rounds).translate(
        ensure_measurements(program), [(i // 2, i % 2) for i in range(4)])
    registers = [r.name for r in one.circuit.cregs if r.name.startswith("m_parity_round")]
    assert len(registers) == 2 * rounds
    lsr, tvd = _noiseless(one, D.single_maps_for(one), "single", program)
    assert lsr == 1.0 and tvd < 0.03
    two = _two_level(program, rounds)
    lsr, tvd = _noiseless(two.inner, D.maps_for(two), "two", program)
    assert lsr == 1.0 and tvd < 0.03


def test_injected_faults_match_explicit_circuits_and_first_order_rates():
    program = _program()
    two = _two_level(program, 2)
    from experiments.tvd_explore.fastframes import at_alpha_one
    from experiments.concatenation.noise import rescale_text

    noisy = at_alpha_one(two.inner.circuit)
    mech = L.mechanisms(noisy)
    flips = L.single_fault_flips(mech, batch=4096)
    rng = np.random.default_rng(0)
    for k in rng.choice(len(mech.probability), 25, replace=False):
        circuit = mech.clean[:int(mech.position[k])].copy()
        for part, qubits in (("X_ERROR", mech.x_qubits[k]), ("Z_ERROR", mech.z_qubits[k])):
            for q in qubits:
                if q >= 0:
                    circuit.append(part, [int(q)], 1.0)
        circuit += mech.clean[int(mech.position[k]):]
        sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True,
                                 num_qubits=noisy.num_qubits)
        sim.do(circuit)
        assert (sim.get_measurement_flips(bit_packed=False)[:, 0] == flips[k]).all()
    maps = D.maps_for(two)
    table = L.Table.build(mech, flips, maps, "two")
    alpha = 0.005
    sim = stim.FlipSimulator(batch_size=200_000, disable_stabilizer_randomization=True,
                             num_qubits=noisy.num_qubits, seed=5)
    sim.do(rescale_text(str(noisy), alpha))
    checks, _ = L.syndromes(sim.get_measurement_flips(bit_packed=False).T, maps, "two")
    assert checks.any(axis=1).mean() == pytest.approx(alpha * table.summary["detected"], rel=0.05)


def test_lookup_corrects_a_lone_unanimous_fault():
    program = _program()
    two = _two_level(program, 1)
    from experiments.tvd_explore.fastframes import at_alpha_one

    noisy = at_alpha_one(two.inner.circuit)
    mech = L.mechanisms(noisy)
    flips = L.single_fault_flips(mech)
    maps = D.maps_for(two)
    table = L.Table.build(mech, flips, maps, "two")
    checks, outputs = L.syndromes(flips, maps, "two")
    keep, corrected = table.apply(flips, maps, "two", threshold=1.0)
    unanimous = np.array([checks[i].any() and table.confidence.get(
        bytes(L._keys(checks[i:i + 1])[0]), 0) >= 1.0 for i in range(len(flips))])
    assert unanimous.any()
    assert keep[unanimous].all() and not corrected[unanimous].any()
