"""
Two translator options for an ablation that turns merging off completely and then back on in
two steps: `solo_hh_promotion` (a solo qubit's H as a whole-patch HH, which the Hadamard handler
did in every arm) and `merge_two_qubit` (the merging pass's transversal CXT / CZT / XCXT).

Pinned: both default to the shipped behaviour; each changes what it says and nothing else;
translation stays exact under each, on placements that have solo qubits; and the solo-H scope
restores the process-wide switch (also on errors) and only ever turns it on.
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

from iceberg.gadget_options import solo_h_targeted
from translator import explore_switches
from translator.config import TranslatorConfig
from translator.core import Translator
from translator.merging import preprocess_program_for_merging_with_stats

#: Three program qubits, each alone in its patch.
SOLO = [(0, 0), (1, 0), (2, 0)]
#: Four program qubits, two to a patch: lane-aligned pairs, so transversal macros can fire.
PAIRED = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _counts(circuit):
    one = sum(1 for i in circuit.data if len(i.qubits) == 1 and i.operation.name not in
              ("measure", "barrier", "reset"))
    two = sum(1 for i in circuit.data if len(i.qubits) == 2 and i.operation.name != "barrier")
    return one, two


def _solo_h_program() -> QuantumCircuit:
    program = QuantumCircuit(3)
    program.h(0)
    program.cx(0, 1)
    program.h(2)
    program.cz(1, 2)
    program.h(1)
    program.measure_all()
    return program


def _transversal_program() -> QuantumCircuit:
    """CX(0,2) and CX(1,3) in one layer: a CXT between the two patches of `PAIRED`."""
    program = QuantumCircuit(4)
    program.h(0)
    program.h(1)                # with H(0): an HH on patch 0
    program.cx(0, 2)
    program.cx(1, 3)
    program.measure_all()
    return program


def _translate(program, placement, **config):
    base = dict(enable_merging=False, enable_h_commutation=False)
    base.update(config)
    return Translator(config=TranslatorConfig(**base)).translate(program, placement).circuit


def test_options_default_to_the_shipped_behaviour():
    config = TranslatorConfig()
    assert config.solo_hh_promotion is True
    assert config.merge_two_qubit is True
    assert config.to_dict()["solo_hh_promotion"] is True
    assert config.to_dict()["merge_two_qubit"] is True


def test_solo_h_is_targeted_without_promotion():
    promoted = _translate(_solo_h_program(), SOLO)
    targeted = _translate(_solo_h_program(), SOLO, solo_hh_promotion=False)
    # The targeted single-lane H is a (7, 4) gadget; the whole-patch HH has no two-qubit gate
    # beyond its anchor exchange, so the targeted circuit has strictly more of both.
    assert _counts(targeted)[1] > _counts(promoted)[1]
    assert _counts(targeted)[0] > _counts(promoted)[0]
    assert explore_switches.SOLO_H_TARGETED is False      # restored after the call


def test_option_equals_the_global_switch():
    scoped = _translate(_solo_h_program(), SOLO, solo_hh_promotion=False)
    explore_switches.configure(SOLO_H_TARGETED=True)
    try:
        global_ = _translate(_solo_h_program(), SOLO)
    finally:
        explore_switches.configure(SOLO_H_TARGETED=False)
    assert str(scoped) == str(global_)


def test_promotion_only_touches_solo_patches():
    """With every qubit paired there is no solo H, so the option changes nothing."""
    program = _transversal_program()
    assert str(_translate(program, PAIRED)) == str(
        _translate(program, PAIRED, solo_hh_promotion=False))


def test_scope_restores_the_switch_and_never_turns_it_off():
    with solo_h_targeted(True):
        assert explore_switches.SOLO_H_TARGETED is True
    assert explore_switches.SOLO_H_TARGETED is False
    with pytest.raises(RuntimeError):
        with solo_h_targeted(True):
            raise RuntimeError("boom")
    assert explore_switches.SOLO_H_TARGETED is False
    explore_switches.configure(SOLO_H_TARGETED=True)
    try:
        with solo_h_targeted(False):              # a script that set the global keeps it
            assert explore_switches.SOLO_H_TARGETED is True
    finally:
        explore_switches.configure(SOLO_H_TARGETED=False)


def _macro_names(circuit):
    return {i.operation.name.lower() for i in circuit.data}


def test_merge_two_qubit_off_keeps_single_qubit_macros_only():
    program = _transversal_program()
    full, *_ = preprocess_program_for_merging_with_stats(program, PAIRED, n_code=4)
    single, *_ = preprocess_program_for_merging_with_stats(program, PAIRED, n_code=4,
                                                            two_qubit=False)
    assert any("cxt" in name for name in _macro_names(full)), _macro_names(full)
    assert not any(name.endswith("t") and name in ("cxt", "czt", "xcxt")
                   for name in _macro_names(single))
    assert any("hh" in name for name in _macro_names(single)), _macro_names(single)


def test_merge_two_qubit_off_through_the_translator():
    program = _transversal_program()
    full = _translate(program, PAIRED, enable_merging=True)
    single = _translate(program, PAIRED, enable_merging=True, merge_two_qubit=False)
    unmerged = _translate(program, PAIRED)
    # The CXT saves two-qubit gates over two targeted CXs; HH saves over two single-lane Hs.
    assert _counts(full)[1] < _counts(single)[1] <= _counts(unmerged)[1]


def _sparse_placements(num_qubits: int, rng: random.Random, count: int):
    """Placements at k = 2 with at least one qubit alone in its patch."""
    out = []
    for _ in range(count):
        patches = rng.randint(-(-num_qubits // 2) + 1, num_qubits)
        pairs = num_qubits - patches
        order = list(range(num_qubits))
        rng.shuffle(order)
        placement = [None] * num_qubits
        for index, qubit in enumerate(order):
            if index < 2 * pairs:
                placement[qubit] = (index // 2, index % 2)
            else:
                placement[qubit] = (pairs + index - 2 * pairs, rng.randint(0, 1))
        out.append(placement)
    return out


@pytest.mark.parametrize("config", [
    {"solo_hh_promotion": False},
    {"solo_hh_promotion": False, "enable_h_commutation": True},
    {"merge_two_qubit": False},
    {"merge_two_qubit": False, "transversal_intra_two_qubit": False},
])
@pytest.mark.parametrize("seed", range(3))
def test_translation_is_exact_under_the_options(config, seed):
    from tests.random_programs import random_clifford_program
    from tests.test_pipeline_logical_action import check_program

    rng = random.Random(9090 + seed)
    checked = 0
    for _ in range(5):
        program = random_clifford_program(4, 16, rng=rng, include_xcx=True)
        for placement in _sparse_placements(4, rng, count=2) + [PAIRED]:
            try:
                check_program(program, placement, 4, **config)
            except RuntimeWarning:
                continue
            checked += 1
    assert checked > 0
