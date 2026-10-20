"""
The standard configuration (w10T): the transversal in-patch CZ / XCX as a config option, the
silent-aware planner in the library, and `packing.standard`.

Pinned here: the option defaults off and emits the transversal gadget when on; translation stays
exact under it; the scope restores the process-wide flag (also on errors) and only ever turns it
on; the cost tables and the commutation-decision cache follow it; a scoped plan equals a plan made
under the global switch (how every measured w10T result was produced); the planner moved without
changing; and the standard is exactly weight 10 + the transversal gadget + layer order.
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

from iceberg.gadget_options import transversal_intra, transversal_intra_enabled
from translator import explore_switches
from translator.config import TranslatorConfig
from translator.core import Translator

PLACEMENT = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _cz_program() -> QuantumCircuit:
    program = QuantumCircuit(4)
    program.h(0)
    program.cz(0, 1)        # in-patch at PLACEMENT
    program.cx(1, 2)        # cross-patch
    program.measure_all()
    return program


def _two_qubit_count(circuit) -> int:
    return sum(1 for i in circuit.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def test_option_defaults_off_and_round_trips():
    assert TranslatorConfig().transversal_intra_two_qubit is False
    assert TranslatorConfig().to_dict()["transversal_intra_two_qubit"] is False
    assert transversal_intra_enabled() is False


def test_option_emits_the_transversal_in_patch_cz():
    def translate(on: bool):
        config = TranslatorConfig(enable_merging=False, enable_h_commutation=False,
                                  transversal_intra_two_qubit=on)
        return Translator(config=config).translate(_cz_program(), PLACEMENT).circuit

    legacy, transversal = translate(False), translate(True)
    # The in-patch CZ gadget is three two-qubit gates; transversally it is none.
    assert _two_qubit_count(legacy) - _two_qubit_count(transversal) == 3
    assert transversal_intra_enabled() is False          # restored after the call


@pytest.mark.parametrize("seed", range(4))
def test_translation_is_exact_under_the_option(seed):
    from tests.random_programs import permutations_for, random_clifford_program
    from tests.test_pipeline_logical_action import check_program

    rng = random.Random(4242 + seed)
    checked = 0
    for _ in range(6):
        program = random_clifford_program(4, 16, rng=rng, include_xcx=True)
        for permutation in permutations_for(4, 4, rng=rng, count=2):
            try:
                check_program(program, permutation, 4, transversal_intra_two_qubit=True)
            except RuntimeWarning:
                continue
            checked += 1
    assert checked > 0


def test_scope_restores_the_flag_and_never_turns_it_off():
    with transversal_intra(True):
        assert transversal_intra_enabled()
    assert not transversal_intra_enabled()
    with pytest.raises(RuntimeError):
        with transversal_intra(True):
            raise RuntimeError("boom")
    assert not transversal_intra_enabled()
    explore_switches.configure(TRANSVERSAL_INTRA_TWO_QUBIT=True)
    try:
        with transversal_intra(False):                    # a script that set the global keeps it
            assert transversal_intra_enabled()
    finally:
        explore_switches.configure(TRANSVERSAL_INTRA_TWO_QUBIT=False)


def test_cost_tables_follow_the_scope():
    from iceberg.costs import depth_table, work_table

    assert work_table(4)["CZ"] == (1, 3) and work_table(4)["XCX"] == (5, 3)
    with transversal_intra(True):
        assert work_table(4)["CZ"] == (6, 0) and work_table(4)["XCX"] == (6, 0)
        assert depth_table(4)["CZ"] == 2
        assert work_table(6)["CZ"] == (1, 3)              # k = 2 only
    assert work_table(4)["CZ"] == (1, 3)


def test_commutation_cache_keys_on_the_gadget_set():
    from packing import passes

    passes.clear_cache()
    program = _cz_program()
    passes.commutation_pays(program, 4)
    with transversal_intra(True):
        passes.commutation_pays(program, 4)
    flags = {key[-1] for key in passes._CACHE}
    assert flags == {False, True}
    passes.clear_cache()


def _small_program(seed: int = 3) -> QuantumCircuit:
    from tests.random_programs import random_clifford_program

    program = random_clifford_program(6, 30, rng=random.Random(seed), include_xcx=True)
    program.measure_all()
    return program


def test_scoped_plan_equals_a_plan_under_the_global_switch():
    """How every measured w10T plan was made: the switch on for the whole worker."""
    from packing import passes
    from packing.plan import plan_packing

    program = _small_program()
    passes.clear_cache()
    scoped = plan_packing(program, 4, transversal_intra_two_qubit=True)
    passes.clear_cache()
    explore_switches.configure(TRANSVERSAL_INTRA_TWO_QUBIT=True)
    try:
        global_ = plan_packing(program, 4)
    finally:
        explore_switches.configure(TRANSVERSAL_INTRA_TWO_QUBIT=False)
        passes.clear_cache()
    assert list(scoped.placement) == list(global_.placement)
    assert scoped.h_commutation == global_.h_commutation


def test_planner_moved_without_changing():
    import packing.silent as silent
    from experiments.revision.exp8_undetected import cost_model, gadgets

    assert cost_model.plan_silent_aware is silent.plan_silent_aware
    assert cost_model.SilentAwareCost is silent.SilentAwareCost
    assert gadgets.UndetectedRates is silent.UndetectedRates


def test_packaged_rates():
    from packing.silent import load_silent_rates

    plain = load_silent_rates(4)
    transversal = load_silent_rates(4, transversal_intra_two_qubit=True)
    assert transversal.intra["CZ"] == 0.0 and transversal.intra["XCX"] == 0.0
    assert plain.intra["CZ"] > 0.0 and plain.intra["CX"] == transversal.intra["CX"]
    assert plain.p2 == 0.003
    for n_code in (6, 8, 10, 12):                         # transversal only exists at k = 2
        assert load_silent_rates(n_code) == load_silent_rates(n_code, transversal_intra_two_qubit=True)


def test_standard_is_weight_ten_transversal_and_layer_order():
    from packing.resource.noise import NoiseKnobs, packing_profile
    from packing.silent import plan_silent_aware
    from packing.standard import (STANDARD, plan_standard, standard_noise_profile,
                                  standard_translator_config)

    assert (STANDARD.silent_weight, STANDARD.transversal_intra_two_qubit, STANDARD.asap_order) == (
        10.0, True, True)
    config = standard_translator_config()
    assert config.asap_order and config.transversal_intra_two_qubit and config.n_code == 4
    assert standard_noise_profile() == packing_profile(NoiseKnobs(), movement_scale=1.0,
                                                       idle_scale=1.0, distance=True)
    program = _small_program(5)
    # With the commutation decision and the final check on the planner's own currency, the
    # standard is plan_silent_aware.
    plan = plan_standard(program, standard=STANDARD.with_(standard_verify=False,
                                                          standard_commutation=False))
    reference = plan_silent_aware(program, 4, silent_weight=10.0, noise=standard_noise_profile(),
                                  transversal_intra_two_qubit=True, solo_hh=True)
    assert list(plan.placement) == list(reference.placement)
    assert plan.h_commutation == reference.h_commutation
    assert plan.fell_back == reference.fell_back
    assert not transversal_intra_enabled()


def _translatable_program() -> QuantumCircuit:
    """A program every placement translates (`_small_program`'s swaps / XCX do not always)."""
    from tests.test_packing_compiler import program as compiler_program

    return compiler_program(10, 3)


def _rotation_program(seed: int = 4) -> QuantumCircuit:
    """The same with non-Clifford rz on some qubits: the old check gate-counts these."""
    program = _translatable_program()
    rng = random.Random(seed)
    for _ in range(6):
        program.rz(rng.uniform(0.1, 1.4), rng.randrange(program.num_qubits))
    return program


@pytest.mark.parametrize("make", [_translatable_program, _rotation_program])
def test_standard_verify_keeps_the_lower_standard_objective(make):
    from packing.plan import naive_packing
    from packing.standard import (STANDARD, plan_standard, standard_noise_profile,
                                  standard_objective, standard_pipeline_config)
    from pipeline.tasks.translate_permutation import translate_permutation

    program = make()
    raw = plan_standard(program, verify=False)
    plan = plan_standard(program)
    assert plan.verified and not raw.verified
    commute = bool(plan.h_commutation)
    naive = naive_packing(program.num_qubits, 2)

    def value(placement):
        placement = [tuple(p) for p in placement]
        circuit = translate_permutation(program, placement, config=standard_pipeline_config(
            commute=commute)).circuit
        return standard_objective(program, placement, 4, circuit, commute=commute,
                                  noise=standard_noise_profile())

    ours, theirs = value(raw.placement), value(naive)
    # Priced on the schedule even with rotations in the program: no gate-counting fallback.
    assert ours.currency == theirs.currency == "schedule"
    assert dict(plan.considered)["packed"] == pytest.approx(ours.total)
    assert dict(plan.considered)["naive"] == pytest.approx(theirs.total)
    expected = naive if theirs.total < ours.total else raw.placement
    assert [tuple(p) for p in plan.placement] == [tuple(p) for p in expected]
    assert plan.fell_back == (theirs.total < ours.total)
    assert STANDARD.standard_verify and not transversal_intra_enabled()


def test_standard_verify_respects_a_fixed_patch_count():
    from packing.standard import plan_standard

    program = _translatable_program()                   # 10 qubits: tightest is 5 patches
    plan = plan_standard(program, patches=7)
    assert plan.verified and all(value < float("inf") for _, value in plan.considered[-2:])
    assert len({patch for patch, _ in plan.placement}) == 7
    assert plan.patches == 7 and plan.physical_qubits == 7 * 6


def _standard_value(program, placement, commute):
    from packing.standard import standard_noise_profile, standard_objective, standard_pipeline_config
    from pipeline.tasks.translate_permutation import translate_permutation

    placement = [tuple(p) for p in placement]
    circuit = translate_permutation(program, placement, config=standard_pipeline_config(
        commute=commute)).circuit
    return standard_objective(program, placement, 4, circuit, commute=commute,
                              noise=standard_noise_profile()).total


@pytest.mark.parametrize("make", [_translatable_program, _rotation_program])
def test_standard_commutation_keeps_the_arm_with_the_lower_standard_objective(make):
    from packing.standard import STANDARD, plan_standard

    program = make()
    plan = plan_standard(program, verify=False)
    arms = {commute: plan_standard(program, verify=False, h_commutation=commute)
            for commute in (True, False)}
    values = {commute: _standard_value(program, arm.placement, commute)
              for commute, arm in arms.items()}
    considered = dict(plan.considered)
    for commute in (True, False):
        assert considered[f"commute={commute}"] == pytest.approx(values[commute])
    chosen = min((True, False), key=values.__getitem__)
    assert bool(plan.h_commutation) == chosen
    assert [tuple(p) for p in plan.placement] == [tuple(p) for p in arms[chosen].placement]
    assert plan.currency == "schedule" and STANDARD.standard_commutation
    assert not transversal_intra_enabled()


def test_restarts_reach_the_planner():
    from packing.silent import plan_silent_aware
    from packing.standard import STANDARD, plan_standard, standard_noise_profile

    program = _translatable_program()
    assert STANDARD.restarts == 0
    for restarts in (0, 2):
        plan = plan_standard(program, verify=False, h_commutation=True,
                             standard=STANDARD.with_(restarts=restarts))
        reference = plan_silent_aware(program, 4, silent_weight=10.0,
                                      noise=standard_noise_profile(), h_commutation=True,
                                      transversal_intra_two_qubit=True, solo_hh=True,
                                      verify=False, restarts=restarts)
        assert list(plan.placement) == list(reference.placement)


def test_grid_alignment_keeps_the_groups_and_the_lowest_price():
    from packing.alignment import align_to_grid
    from packing.standard import STANDARD, plan_standard

    from tests.test_packing_compiler import program as compiler_program

    program = compiler_program(12, 4)                   # 6 patches on a 3 x 2 grid
    plain = plan_standard(program, verify=False, h_commutation=True)
    plan = plan_standard(program, verify=False, h_commutation=True,
                         standard=STANDARD.with_(grid_alignment=True))
    aligned = [tuple(p) for p in align_to_grid(plain.placement, program)]
    considered = dict(plan.considered)
    assert considered["packed"] == pytest.approx(_standard_value(program, plain.placement, True))
    assert aligned != [tuple(p) for p in plain.placement]   # alignment moves this plan
    value = _standard_value(program, aligned, True)
    assert considered["packed aligned"] == pytest.approx(value)
    keep = aligned if value < considered["packed"] else plain.placement
    assert [tuple(p) for p in plan.placement] == [tuple(p) for p in keep]
    assert plan.aligned == (keep is aligned)

    def groups(placement):
        return sorted(sorted(q for q, (patch, _) in enumerate(placement) if patch == p)
                      for p in {patch for patch, _ in placement})

    assert groups(plan.placement) == groups(plain.placement)
    assert not STANDARD.grid_alignment


def test_compile_circuit_defaults_to_the_standard_planner():
    from packing.compiler import compile_circuit
    from tests.test_packing_compiler import program as compiler_program

    program = compiler_program(10, 3)
    built = compile_circuit(program, code_sizes=(4,))
    assert getattr(built.plan, "currency", None) is not None   # set only by the silent-aware planner
    legacy = compile_circuit(program, code_sizes=(4,), planner="v4")
    assert getattr(legacy.plan, "currency", None) is None
    with pytest.raises(ValueError):
        compile_circuit(program, code_sizes=(4,), planner="w3")
    assert not transversal_intra_enabled()


def test_clifford_stand_in_prices_rotations_like_any_one_qubit_gate():
    from qiskit.circuit.library import XGate

    from simulator.attribution import clifford_stand_in, measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    program = QuantumCircuit(4)
    program.h(0)
    program.rz(0.3, 1)
    program.rx(0.2, 2)
    program.cx(0, 2)
    program.cz(1, 3)
    program.measure_all()
    config = TranslatorConfig(enable_merging=False, enable_h_commutation=False)
    encoded = Translator(config=config).translate(program, PLACEMENT).circuit
    rotations = {"rz", "rx"}
    assert {i.operation.name for i in encoded.data} & rotations
    params = StimNoiseParameters.neutral_atom()
    with pytest.raises(Exception):
        measured_noise_split(encoded, params)                    # Stim cannot take rotations
    stand_in = clifford_stand_in(encoded)
    assert len(stand_in.data) == len(encoded.data)
    assert not {i.operation.name for i in stand_in.data} & rotations
    assert clifford_stand_in(stand_in) is stand_in               # nothing left to replace
    # The premise: every one-qubit gate is charged alike, so any Clifford stand-in prices the same.
    twin = encoded.copy_empty_like()
    for instruction in encoded.data:
        if instruction.operation.name in rotations:
            twin.append(XGate(), instruction.qubits)
        else:
            twin.append(instruction)
    priced, other = measured_noise_split(stand_in, params), measured_noise_split(twin, params)
    assert set(priced) == set(other)
    for key in priced:
        assert priced[key] == pytest.approx(other[key], rel=1e-12, abs=1e-15)


def test_silent_weight_is_per_code_size():
    from packing.standard import STANDARD, AqornStandard

    assert STANDARD.silent_weight_for(4) == STANDARD.silent_weight == 10.0
    assert STANDARD.silent_weight_for(6) == STANDARD.silent_weight_for(12) == 5.0
    custom = AqornStandard(wide_silent_weight=2.0)
    assert custom.silent_weight_for(4) == 10.0 and custom.silent_weight_for(6) == 2.0
    assert AqornStandard(wide_silent_weight=None).silent_weight_for(8) == 10.0


def test_standard_objective_is_schedule_detected_plus_weighted_silent():
    from packing.standard import STANDARD, standard_objective, standard_noise_profile
    from packing.standard import standard_pipeline_config
    from pipeline.tasks.translate_permutation import translate_permutation
    from simulator.attribution import measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    from packing.standard import plan_standard
    from tests.test_packing_compiler import program as compiler_program

    program = compiler_program(10, 3)
    plan = plan_standard(program)
    placement, commute = [tuple(p) for p in plan.placement], bool(plan.h_commutation)
    with transversal_intra(True):
        circuit = translate_permutation(program, placement,
                                        config=standard_pipeline_config(commute=commute)).circuit
    value = standard_objective(program, placement, 4, circuit, commute=commute,
                               noise=standard_noise_profile())
    detected = measured_noise_split(circuit, StimNoiseParameters.neutral_atom())["total"]
    assert value.currency == "schedule"
    assert value.detected == pytest.approx(detected)
    assert value.total == pytest.approx(value.detected + STANDARD.silent_weight * value.silent)
    assert value.silent > -1e-12
    # Program-order packing keeps in-patch gadgets the plan moved apart: it pays silent error.
    from packing.plan import naive_packing

    naive = naive_packing(program.num_qubits, 2)
    with transversal_intra(True):
        naive_circuit = translate_permutation(
            program, naive, config=standard_pipeline_config(commute=commute)).circuit
    naive_value = standard_objective(program, naive, 4, naive_circuit, commute=commute,
                                     noise=standard_noise_profile())
    assert naive_value.silent > value.silent + 1e-6


def test_standard_compile_keeps_the_lowest_objective_over_code_sizes():
    from packing.compiler import compile_circuit
    from tests.test_packing_compiler import program as compiler_program

    program = compiler_program(10, 3)
    built = compile_circuit(program, code_sizes=(4, 6))
    totals = [total for _, total in built.considered]
    assert totals == sorted(totals)
    assert built.objective.total == pytest.approx(totals[0])
    assert {name.split()[1] for name, _ in built.considered} == {"n=4", "n=6"}
    assert built.source in ("packer", "naive")
    assert built.objectives[built.considered[0][0]] is built.objective
    assert not transversal_intra_enabled()


def test_ranking_weight_is_its_own_knob():
    from packing.standard import STANDARD

    assert STANDARD.selection_weight is None
    assert STANDARD.ranking_weight() == STANDARD.silent_weight == 10.0
    custom = STANDARD.with_(selection_weight=50.0)
    assert custom.ranking_weight() == 50.0
    assert custom.silent_weight_for(4) == 10.0 and custom.silent_weight_for(6) == 5.0


def test_selection_weight_moves_the_choice_along_the_shots_accuracy_trade():
    """Weight 0 ranks on detected errors alone, a huge weight on silent errors alone."""
    from packing.compiler import compile_circuit
    from packing.standard import STANDARD
    from tests.test_packing_compiler import program as compiler_program

    program = compiler_program(12, 3, seed=11)
    default = compile_circuit(program, code_sizes=(4, 6))
    shots = compile_circuit(program, code_sizes=(4, 6),
                            standard=STANDARD.with_(selection_weight=0.0))
    accuracy = compile_circuit(program, code_sizes=(4, 6),
                               standard=STANDARD.with_(selection_weight=1e6))
    # The planners' weights are untouched, so all three price the same candidates ...
    assert set(default.objectives) == set(shots.objectives) == set(accuracy.objectives)
    for name, value in default.objectives.items():
        assert shots.objectives[name].detected == pytest.approx(value.detected)
        assert accuracy.objectives[name].silent == pytest.approx(value.silent)
    # ... and each keeps the best of its own weighting of the same two components.
    detected = {name: value.detected for name, value in default.objectives.items()}
    silent = {name: value.silent for name, value in default.objectives.items()}
    assert detected[shots.considered[0][0]] == pytest.approx(min(detected.values()))
    assert silent[accuracy.considered[0][0]] == pytest.approx(min(silent.values()), abs=1e-12)
    assert shots.objective.detected <= accuracy.objective.detected + 1e-12
    assert accuracy.objective.silent <= shots.objective.silent + 1e-12
    for built, weight in ((default, 10.0), (shots, 0.0), (accuracy, 1e6)):
        value = built.objective
        assert value.total == pytest.approx(value.detected + weight * value.silent)
