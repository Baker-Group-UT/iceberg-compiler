"""
`TranslatorConfig.asap_order`: every program layer-sorted before translation.

The field exists so that Fig 6's stages can be compared on equal schedules: shipped, only a
program the merging pass rewrote reaches the translator in layer order. These tests pin
three things -- the default is off, turning it on never changes what a circuit computes
(exact tableau comparison, merging on and off), and it is exactly a re-ordering of the
program, nothing else.
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

from iceberg.utils import logical_capacity
from tests.random_programs import permutations_for, random_clifford_program
from tests.test_pipeline_logical_action import check_program
from translator.config import TranslatorConfig
from translator.core import Translator, _layer_sorted

CODE_SIZES = [4, 6, 8]


def test_default_is_off():
    assert TranslatorConfig().asap_order is False
    assert TranslatorConfig().to_dict()["asap_order"] is False


@pytest.mark.parametrize("n_code", CODE_SIZES)
@pytest.mark.parametrize("enable_merging", [True, False])
def test_asap_order_preserves_the_logical_action(n_code, enable_merging):
    rng = random.Random(9090 + n_code + (1 if enable_merging else 0))
    k = logical_capacity(n_code)
    checked = 0
    for _ in range(10):
        program = random_clifford_program(2 * k, 16, rng=rng, include_xcx=True)
        for permutation in permutations_for(2 * k, n_code, rng=rng, count=2):
            try:
                check_program(program, permutation, n_code, enable_merging=enable_merging,
                              asap_order=True)
            except RuntimeWarning:
                continue  # placement refused by the translator's rules, not a wrong answer
            checked += 1
    assert checked > 0


def _program(seed: int, num_qubits: int = 4, gates: int = 12) -> QuantumCircuit:
    """A random Clifford program without SWAPs, which these placements always accept."""
    from tests.random_programs import _append_two_qubit

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(gates):
        if rng.random() < 0.5:
            a, b = rng.sample(range(num_qubits), 2)
            _append_two_qubit(circuit, rng.choice(["cx", "cz", "xcx"]), a, b)
        else:
            getattr(circuit, rng.choice(["h", "s", "sx", "x", "z"]))(rng.randrange(num_qubits))
    return circuit


PLACEMENT = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _translate(circuit, asap):
    config = TranslatorConfig(enable_merging=False, enable_h_commutation=False, asap_order=asap)
    return Translator(config=config).translate(circuit, PLACEMENT).circuit


@pytest.mark.parametrize("seed", range(6))
def test_layer_sort_is_the_same_dag_in_layer_order(seed):
    """A re-ordering and nothing else: same DAG, emitted with non-decreasing layer."""
    from qiskit.converters import circuit_to_dag

    from translator.merging.layer import layer_by_node

    program = _program(seed, gates=30)
    ordered = _layer_sorted(program)
    assert circuit_to_dag(ordered) == circuit_to_dag(program)
    dag = circuit_to_dag(ordered)
    layers, _ = layer_by_node(dag)
    by_position = [layers[node] for node in sorted(dag.topological_op_nodes(), key=lambda n: n._node_id)]
    assert by_position == sorted(by_position)


def test_asap_order_is_what_translation_receives(monkeypatch):
    """
    The field layer-sorts the program the translator is handed, after the passes.

    Not a comparison against translating a pre-sorted program: the translator's first step,
    `cancel_inverse_gates`, round-trips the circuit through a DAG and Qiskit rebuilds it
    in its own canonical order, so "program order" never survives to translation anyway.
    """
    import translator.core as core
    from translator.passes import cancel_inverse_gates

    seen = []
    original = core.Translation

    def capture(*args, **kwargs):
        seen.append(kwargs["program_circuit"])
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "Translation", capture)
    program = _program(2, gates=30)
    _translate(program, True)
    _translate(program, False)
    assert seen[0] == _layer_sorted(cancel_inverse_gates(program))
    assert seen[1] == cancel_inverse_gates(program)


def test_asap_order_shortens_a_program_emitted_out_of_layer_order():
    """
    The case it exists for. Gadgets on one patch serialize on its anchors, so a gate the
    program writes late but could run early holds up whatever follows it on the anchor.
    Seed 8 is one of 74 in the first 200 that layer order shortens (43 -> 39 layers).
    """
    program = _program(8)
    assert _translate(program, True).depth() < _translate(program, False).depth()
