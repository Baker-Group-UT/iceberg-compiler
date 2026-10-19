"""
The planner's price of a solo qubit's Hadamards (`PackingCost.solo_hh`,
`AqornStandard.solo_hh_aware`).

The translator emits every H of a qubit alone in its patch as a whole-patch HH, merging on or off
(`translator.handlers.hadamard`); the historical objective prices it as a single-lane gadget,
because it treats a qubit's own gates as costing the same wherever it sits. Pinned: off is that
objective exactly; on, a solo patch costs exactly what its Hadamards save as HH less -- in work,
and in silent error under the silent-aware cost -- and nothing else moves; the translator's
switch that turns the promotion off turns the price off; the saving per H is the translator's in
gates; and a planner given the price uses it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.costs import DEFAULT_WORK_WEIGHTS, work_table
from packing.cost import PackingCost
from packing.graph import NoiseProfile
from packing.profiles import extract_profiles
from packing.silent import SilentAwareCost, load_silent_rates
from packing.tables import translated_view
from translator import explore_switches
from translator.config import TranslatorConfig
from translator.core import Translator


def program(num_qubits: int, depth: int, seed: int = 5) -> QuantumCircuit:
    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            if rng.random() < 0.4:
                circuit.h(qubit)
        for _ in range(max(1, num_qubits // 3)):
            a, b = rng.sample(range(num_qubits), 2)
            rng.choice([circuit.cx, circuit.cz])(a, b)
    return circuit


def _profiles(circuit):
    return extract_profiles(translated_view(circuit, enable_h_commutation=False))


def _h_count(cost, qubit):
    return len(cost.layers["h"].get(qubit, frozenset()))


GROUPS = [frozenset({0, 1}), frozenset({2}), frozenset({3, 4}), frozenset({5})]


def test_off_is_the_historical_objective():
    profiles = _profiles(program(6, 6))
    historical = PackingCost(profiles=profiles, n_code=4, num_qubits=6)
    off = PackingCost(profiles=profiles, n_code=4, num_qubits=6, solo_hh=False)
    assert off.of(GROUPS) == historical.of(GROUPS)


def test_a_solo_patch_costs_exactly_its_hadamards_saving_less():
    profiles = _profiles(program(6, 6))
    off = PackingCost(profiles=profiles, n_code=4, num_qubits=6)
    on = PackingCost(profiles=profiles, n_code=4, num_qubits=6, solo_hh=True)
    saving = sum(_h_count(on, q) for q in (2, 5)) * (on._h_single - on._hh)
    assert saving > 0
    assert off.of(GROUPS) - on.of(GROUPS) == pytest.approx(saving, rel=1e-12)
    # No solo patch, no difference.
    paired = [frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})]
    assert off.of(paired) == pytest.approx(on.of(paired), rel=1e-12)


def test_the_silent_aware_cost_also_credits_the_leak_it_avoids():
    profiles = _profiles(program(6, 6))
    rates = load_silent_rates(4)
    noise = NoiseProfile()
    kwargs = dict(profiles=profiles, n_code=4, rates=rates, silent_weight=10.0, num_qubits=6,
                  movement_cost=noise.movement_cost, idle_cost=noise.idle_cost)
    off = SilentAwareCost(**kwargs)
    on = SilentAwareCost(solo_hh=True, **kwargs)
    count = sum(_h_count(on, q) for q in (2, 5))
    expected = count * ((on._h_single - on._hh)
                        + on.silent_scale * (rates.h_lane - rates.hh))
    assert off.of(GROUPS) - on.of(GROUPS) == pytest.approx(expected, rel=1e-12)


def test_the_translators_switch_turns_the_price_off():
    profiles = _profiles(program(6, 6))
    explore_switches.configure(SOLO_H_TARGETED=True)
    try:
        on = PackingCost(profiles=profiles, n_code=4, num_qubits=6, solo_hh=True)
    finally:
        explore_switches.configure(SOLO_H_TARGETED=False)
    off = PackingCost(profiles=profiles, n_code=4, num_qubits=6)
    assert on.solo_hh is False
    assert on.of(GROUPS) == off.of(GROUPS)


def test_the_saving_per_hadamard_is_the_translators():
    """One H on a solo qubit: targeted gadget against whole-patch HH, in gates."""
    circuit = QuantumCircuit(3)
    circuit.cx(1, 2)
    circuit.h(0)
    circuit.cz(1, 2)
    circuit.measure_all()
    placement = [(0, 0), (1, 0), (1, 1)]

    def counts(promote: bool):
        config = TranslatorConfig(enable_merging=False, enable_h_commutation=False,
                                  solo_hh_promotion=promote)
        out = Translator(config=config).translate(circuit, placement).circuit
        one = sum(1 for i in out.data if len(i.qubits) == 1
                  and i.operation.name not in ("measure", "barrier", "reset"))
        two = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
        return one, two

    targeted, promoted = counts(False), counts(True)
    table = work_table(4)
    assert targeted[0] - promoted[0] == table["H"][0] - table["HH"][0]
    # HH's anchor exchange is one SWAP in the circuit; the model prices it as movement
    # (`carriage_cost("HH")`, inside `_hh`), as it prices every HH macro, not as a table gate.
    assert targeted[1] - promoted[1] == table["H"][1] - table["HH"][1] - 1
    from iceberg.costs import carriage_cost

    cost = PackingCost(profiles=_profiles(circuit), n_code=4, num_qubits=3, solo_hh=True)
    w1, w2 = DEFAULT_WORK_WEIGHTS
    assert cost._hh == pytest.approx(table["HH"][0] * w1 + table["HH"][1] * w2
                                     + carriage_cost("HH"))
    assert carriage_cost("HH") > 0


def test_a_planner_given_the_price_leaves_the_hadamard_heavy_qubit_alone():
    """Three qubits in two patches: one must be alone, and it should be the one with the Hs."""
    from packing.standard import plan_standard

    circuit = QuantumCircuit(3)
    for _ in range(4):
        circuit.h(0)
        circuit.cx(1, 2)
        circuit.s(0)
        circuit.cz(1, 2)
    circuit.cx(0, 1)
    circuit.measure_all()
    plan = plan_standard(circuit, 4, patches=2, h_commutation=False, verify=False, solo_hh=True)
    patch_of = {qubit: patch for qubit, (patch, _) in enumerate(plan.placement)}
    assert patch_of[0] not in (patch_of[1], patch_of[2])
    assert patch_of[1] == patch_of[2]


def test_the_standard_prices_solo_patches_and_the_legacy_planner_is_one_switch_away():
    from packing.silent import plan_silent_aware
    from packing.standard import STANDARD, plan_standard, standard_noise_profile

    assert STANDARD.solo_hh_aware is True
    circuit = QuantumCircuit(3)
    for _ in range(3):
        circuit.h(0)
        circuit.cx(1, 2)
    circuit.measure_all()
    legacy = STANDARD.with_(solo_hh_aware=False)
    for standard, solo_hh in ((STANDARD, True), (legacy, False)):
        via_standard = plan_standard(circuit, 4, standard=standard, patches=2, verify=False,
                                     h_commutation=False)
        direct = plan_silent_aware(
            circuit, 4, silent_weight=standard.silent_weight_for(4),
            noise=standard_noise_profile(), h_commutation=False,
            transversal_intra_two_qubit=standard.transversal_intra_two_qubit,
            patches=2, verify=False, solo_hh=solo_hh)
        assert via_standard.placement == direct.placement
