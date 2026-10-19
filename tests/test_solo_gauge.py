"""
[[4,1,2]] gauge checks on solo patches (`experiments/solo_gauge/gauge.py`) and the two
opt-in translator switches that keep a solo patch's spare lane checkable.

The validity criterion is the part that must not be wrong: a check is admissible only if
its value is fixed by the spare's |0bar> preparation for *every* program input. Pinned on
hand-built circuits where the answer is known -- spare untouched, flipped to the X frame by
an odd number of HH, restored by a second HH, entangled with a program lane by a
transversal CNOT (deterministic at the |0> input, so a noiseless-only test would wrongly
accept it), spare as a CNOT control, moved by a routing SWAP, flipped to |1bar>, and a
product of two spares that is valid although neither spare is alone.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.solo_gauge import gauge
from iceberg.iceberg_postselect import iceberg_postselect
from iceberg.iceberg_set import IcebergSet
from translator import explore_switches
from translator.config import TranslatorConfig
from translator.core import Translator

THREE = [(0, 0), (0, 1), (1, 0)]  # patch 1 is solo; its spare is global lane 3


def _built(ops, n_program_lanes: int = 4):
    """A two-patch encoded circuit with `ops(iceberg_set)` as its body."""
    qc = QuantumCircuit()
    patches = IcebergSet(n_program_lanes, 4, qc)
    patches.encode()
    ops(patches)
    patches.decode()
    patches.measure()
    return qc


def _analyse(ops, permutation=THREE, final=None):
    circuit = _built(ops)
    final = permutation if final is None else final
    return circuit, gauge.analyse(circuit, permutation, final)


# ----------------------------------------------------------------------------------------
# the validity criterion
# ----------------------------------------------------------------------------------------


def test_clean_spare_is_a_valid_check():
    def ops(s):
        s.S(2)
        s.CNOT(2, 0)
        s.CZ(1, 2)
        s.SX(2)

    circuit, a = _analyse(ops)
    assert a.solo_patches == (1,) and a.candidates == (3,)
    assert a.status == {3: "valid"}
    assert a.checks == (gauge.GaugeCheck(lanes=(3,), expected=0),)
    gate = gauge.noiseless_gate(circuit, a)
    assert gate["noiseless_check_fired"] == 0 and gate["noiseless_syndrome_fired"] == 0
    assert gauge.randomized_input_gate(circuit, a, THREE)["random_input_check_fired"] == 0


def test_odd_hh_moves_the_spare_to_the_x_frame():
    circuit, a = _analyse(lambda s: (s.HH(1), s.CNOT(2, 0)))
    assert a.status == {3: "x_frame"} and a.checks == ()
    # And it really is random: the probe flips in a noiseless run.
    assert gauge.noiseless_gate(circuit, a)["rejected_but_deterministic_at_zero_input"] == 0


def test_even_hh_restores_the_check():
    circuit, a = _analyse(lambda s: (s.HH(1), s.S(2), s.HH(1)))
    assert a.status == {3: "valid"} and len(a.checks) == 1
    assert gauge.noiseless_gate(circuit, a)["noiseless_check_fired"] == 0


def test_spare_entangled_by_a_transversal_cnot_is_rejected():
    # CNOT_k_Transversal(0 -> 1) pairs lane 1 (a program qubit) with lane 3 (the spare):
    # the spare's Zbar becomes Zbar_1 Zbar_3. At the |0> input it is still deterministic,
    # which is exactly the post-selection-on-the-answer trap.
    circuit, a = _analyse(lambda s: s.CNOT_k_Transversal(0, 1))
    assert a.status == {3: "entangled"} and a.checks == ()
    assert gauge.noiseless_gate(circuit, a)["rejected_but_deterministic_at_zero_input"] == 1
    gate = gauge.randomized_input_gate(circuit, a, THREE, inputs=8)
    assert gate["rejected_random_under_random_inputs"] == 1


def test_spare_as_transversal_control_stays_valid():
    circuit, a = _analyse(lambda s: s.CNOT_k_Transversal(1, 0))
    assert a.status == {3: "valid"} and len(a.checks) == 1
    assert gauge.randomized_input_gate(circuit, a, THREE)["random_input_check_fired"] == 0


def test_routing_swap_moves_the_spare():
    # The program qubit of patch 1 moves from slot 0 to slot 1; the spare is now lane 2.
    final = [(0, 0), (0, 1), (1, 1)]
    circuit, a = _analyse(lambda s: (s.S(2), s.SWAP([[1, 0, 1]]), s.S(3)), final=final)
    assert a.candidates == (2,) and a.status == {2: "valid"}
    assert gauge.noiseless_gate(circuit, a)["noiseless_check_fired"] == 0


def test_sign_of_a_flipped_spare():
    # Logical X on the spare: |1bar>, so the check expects 1. HH.Z.HH = X as well.
    _, a = _analyse(lambda s: s.X(3))
    assert a.checks == (gauge.GaugeCheck(lanes=(3,), expected=1),)
    circuit, b = _analyse(lambda s: (s.HH(1), s.Z(3), s.HH(1)))
    assert b.checks == (gauge.GaugeCheck(lanes=(3,), expected=1),)
    assert gauge.noiseless_gate(circuit, b)["noiseless_check_fired"] == 0


def test_product_of_two_spares_found_by_elimination():
    # Two solo patches. HH puts spare 1 in |+bar>; the transversal CNOT then makes the two
    # spares a Bell pair: each alone is random, their Zbar product is fixed.
    perm = [(0, 0), (1, 0)]

    def ops(s):
        s.HH(0)
        s.CNOT_k_Transversal(0, 1)

    circuit = _built(ops)
    a = gauge.analyse(circuit, perm, perm)
    assert a.candidates == (1, 3)
    assert set(a.status.values()) <= {"x_frame", "entangled"}
    assert a.checks == (gauge.GaugeCheck(lanes=(1, 3), expected=0),)
    assert gauge.noiseless_gate(circuit, a)["noiseless_check_fired"] == 0
    assert gauge.randomized_input_gate(circuit, a, perm)["random_input_check_fired"] == 0


def test_only_solo_patches_are_candidates():
    perm = [(0, 0), (1, 0), (1, 1)]  # patch 0 solo, patch 1 full
    circuit = _built(lambda s: s.S(0))
    a = gauge.analyse(circuit, perm, perm)
    assert a.solo_patches == (0,) and a.candidates == (1,)


# ----------------------------------------------------------------------------------------
# the decoder
# ----------------------------------------------------------------------------------------


def _translated(program, permutation, **switches):
    explore_switches.configure(**switches)
    try:
        config = TranslatorConfig(enable_merging=False, enable_h_commutation=False)
        return Translator(config=config).translate(program, permutation)
    finally:
        explore_switches.configure(**{name: False for name in switches})


def _random_register_dist(key_widths, rng, count=400):
    """Random outcomes, biased towards clean syndromes and codewords so both paths keep some."""
    out = defaultdict(float)
    for _ in range(count):
        parts = []
        for width in key_widths:
            if width == 2:
                bits = [0, 0] if rng.random() < 0.8 else [rng.randrange(2) for _ in range(2)]
            else:
                bits = [rng.randrange(2) for _ in range(width - 1)]
                bits.append(sum(bits) % 2 if rng.random() < 0.85 else 1 - sum(bits) % 2)
            parts.append("".join(map(str, bits)))
        out[" ".join(parts)] += 1.0 / count
    return dict(out)


@pytest.mark.parametrize("seed", range(4))
def test_422_path_matches_the_library(seed):
    from analysis.postselection import reduce_distribution

    program = QuantumCircuit(3)
    program.h(0)
    program.cx(0, 2)
    program.s(2)
    program.measure_all()
    tr = _translated(program, THREE)
    a = gauge.analyse_translation(tr)
    names = [r.name for r in tr.circuit.cregs]
    widths = [len(r) for r in tr.circuit.cregs]
    dist = _random_register_dist(widths, random.Random(seed))
    decoded = gauge.decode_samples(dist, names, widths, tr.context.state_positions,
                                   a.checks, 4)
    got, lsr = decoded.distribution("422")
    want, want_lsr = iceberg_postselect(dist, names)
    want = reduce_distribution(want, tr.context)
    assert lsr == pytest.approx(want_lsr, abs=1e-12)
    assert set(got) == set(want)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-12)

    # [[4,1,2]] keeps exactly the [[4,2,2]] outcomes whose spare bit is the expected one.
    _, lsr_412 = decoded.distribution("412")
    (check,) = a.checks
    kept = 0.0
    for outcome, p in dist.items():
        registers = dict(zip(names, outcome.split()))
        if any("1" in registers[k] for k in names if "parity" in k):
            continue
        if any(registers[k].count("1") % 2 for k in names if "data" in k):
            continue
        bits = registers["m_data_1"]
        if (int(bits[2]) ^ int(bits[3])) == check.expected:
            kept += p
    assert lsr_412 == pytest.approx(kept, abs=1e-12)


def test_noiseless_decoding_is_unchanged():
    """With the noise off no valid check fires, so both rules give the same distribution."""
    from experiments.solo_gauge.gauge import _register_samples, to_stim

    program = QuantumCircuit(3)
    program.h(0)
    program.cx(0, 1)
    program.cx(1, 2)
    program.s(2)
    program.measure_all()
    tr = _translated(program, THREE)
    a = gauge.analyse_translation(tr)
    assert len(a.checks) == 1
    samples = to_stim(tr.circuit, measure=True).compile_sampler(seed=3).sample(shots=3000)
    registers = _register_samples(tr.circuit, samples)
    names = [r.name for r in tr.circuit.cregs]
    widths = [len(r) for r in tr.circuit.cregs]
    dist = defaultdict(float)
    for shot in range(samples.shape[0]):
        key = " ".join("".join(str(int(b)) for b in registers[name][shot]) for name in names)
        dist[key] += 1 / samples.shape[0]
    decoded = gauge.decode_samples(dict(dist), names, widths, tr.context.state_positions,
                                   a.checks, 4)
    d422, l422 = decoded.distribution("422")
    d412, l412 = decoded.distribution("412")
    assert l422 == pytest.approx(1.0) and l412 == pytest.approx(1.0)
    assert d422 == d412
    assert set(d422) == {"000", "111"}


# ----------------------------------------------------------------------------------------
# the opt-in translator switches
# ----------------------------------------------------------------------------------------


def test_switches_default_off():
    assert explore_switches.SOLO_H_TARGETED is False
    assert explore_switches.SOLO_GAUGE_PARITY_FIX is False


def _solo_h_program():
    program = QuantumCircuit(3)
    program.h(2)
    program.cx(2, 0)
    program.s(1)
    return program


def test_default_promotes_solo_h_to_hh():
    circuit = _translated(_solo_h_program(), THREE).circuit
    names = [inst.operation.name for inst in circuit.data]
    assert "swap" in names  # HH's anchor exchange
    reference = _translated(_solo_h_program(), THREE, SOLO_H_TARGETED=False,
                            SOLO_GAUGE_PARITY_FIX=False).circuit
    assert circuit == reference


def test_targeted_switch_keeps_the_spare_checkable():
    default = gauge.analyse_translation(_translated(_solo_h_program(), THREE))
    assert default.status == {3: "x_frame"}
    targeted_tr = _translated(_solo_h_program(), THREE, SOLO_H_TARGETED=True)
    assert "swap" not in [inst.operation.name for inst in targeted_tr.circuit.data]
    targeted = gauge.analyse_translation(targeted_tr)
    assert targeted.status == {3: "valid"}
    fixed = gauge.analyse_translation(
        _translated(_solo_h_program(), THREE, SOLO_GAUGE_PARITY_FIX=True))
    assert fixed.status == {3: "valid"}
    # Even HH count: the fix-up emits nothing.
    even = _solo_h_program()
    even.h(2)
    assert (_translated(even, THREE, SOLO_GAUGE_PARITY_FIX=True).circuit
            == _translated(even, THREE).circuit)


@pytest.mark.parametrize("switch", ["SOLO_H_TARGETED", "SOLO_GAUGE_PARITY_FIX"])
def test_switches_preserve_the_program(switch):
    from tests.random_programs import random_clifford_program
    from tests.test_pipeline_logical_action import check_program

    rng = random.Random(4242)
    checked = 0
    explore_switches.configure(**{switch: True})
    try:
        for _ in range(12):
            program = random_clifford_program(3, 14, rng=rng, include_xcx=True)
            for permutation in (THREE, [(0, 0), (1, 0), (1, 1)], [(1, 1), (0, 0), (1, 0)]):
                for merging in (False, True):
                    try:
                        check_program(program, permutation, 4, enable_merging=merging)
                    except RuntimeWarning:
                        continue
                    checked += 1
    finally:
        explore_switches.configure(**{switch: False})
    assert checked > 20
