"""
Tests for `packing.passes` — is H commutation worth running on this circuit?

The pass is not free: it removes a Hadamard by carrying a basis flip forward and pays
for that by rewriting the two-qubit gates the frame passes through. When the Hadamards
do not actually go away the rewrite is pure cost.
"""

from __future__ import annotations

import random

import pytest
from qiskit import QuantumCircuit

from packing.passes import clear_cache, commutation_pays
from packing.plan import measured_cost, naive_packing, plan_packing
from iceberg.utils import logical_capacity
from translator.passes import cancel_inverse_gates, commutation_gain, commute_hadamards


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


def hadamard_sandwich(num_qubits: int = 6) -> QuantumCircuit:
    """H on every qubit, a CNOT chain, H back. Commutation should absorb all of them."""
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.h(qubit)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    for qubit in range(num_qubits):
        circuit.h(qubit)
    return circuit


def cnot_heavy(num_qubits: int = 8, layers: int = 6) -> QuantumCircuit:
    """CNOTs with a Hadamard the frame cannot absorb, so the rewrite buys nothing."""
    rng = random.Random(4)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(layers):
        order = list(range(num_qubits))
        rng.shuffle(order)
        for index in range(0, num_qubits - 1, 2):
            circuit.cx(order[index], order[index + 1])
        circuit.h(rng.randrange(num_qubits))
    return circuit


# --- the decision agrees with the thing it is deciding ------------------


@pytest.mark.parametrize("n_code", [4, 6, 8])
@pytest.mark.parametrize(
    "builder", [hadamard_sandwich, cnot_heavy], ids=["sandwich", "cnot_heavy"]
)
def test_decision_matches_the_measurement(builder, n_code):
    """`commutation_pays` must agree with translating both ways. It *is* that."""
    circuit = builder()
    naive = naive_packing(circuit.num_qubits, logical_capacity(n_code))
    with_pass = measured_cost(circuit, naive, n_code, h_commutation=True)
    without = measured_cost(circuit, naive, n_code, h_commutation=False)
    if with_pass is None or without is None:
        pytest.skip("placement rejected at this code size")
    assert commutation_pays(circuit, n_code) == (with_pass <= without)


def test_never_worse_than_always_commuting():
    """Whatever it decides, the result cannot be worse than forcing the pass on."""
    for builder in (hadamard_sandwich, cnot_heavy):
        circuit = builder()
        for n_code in (4, 6):
            naive = naive_packing(circuit.num_qubits, logical_capacity(n_code))
            forced = measured_cost(circuit, naive, n_code, h_commutation=True)
            chosen = measured_cost(
                circuit, naive, n_code,
                h_commutation=commutation_pays(circuit, n_code),
            )
            if forced is None or chosen is None:
                continue
            assert chosen <= forced + 1e-9


# --- behaviour ----------------------------------------------------------


def test_cache_returns_the_same_answer():
    circuit = cnot_heavy()
    first = commutation_pays(circuit, 6)
    assert commutation_pays(circuit, 6) is first
    clear_cache()
    assert commutation_pays(circuit, 6) is first


def test_decision_can_differ_by_code_size():
    """It is a per-(circuit, code size) question, so the signature must carry both."""
    circuit = cnot_heavy(10, 8)
    answers = {n: commutation_pays(circuit, n) for n in (4, 6, 8, 10, 12)}
    assert set(answers) == {4, 6, 8, 10, 12}
    assert all(isinstance(v, bool) for v in answers.values())


def test_empty_circuit_is_handled():
    assert isinstance(commutation_pays(QuantumCircuit(2), 4), bool)


# --- plan_packing honours it -------------------------------------------


def test_plan_packing_auto_is_no_worse_than_either_forced_arm():
    """
    `auto` must beat both fixed settings, measured on the placements they produce.

    Stronger than "no worse than forced on", and it is the property that broke when the
    decision was taken from `commutation_pays`: that answers the question on the naive
    packing, so on `dnn_n51` at k=2 it chose off (42,160 against 42,768 weighted) while
    the placement it was choosing for wanted on (41,918 against 42,486). Comparing
    against both arms is what catches an arm never being explored.

    Note each arm is measured under *its own* commutation setting -- a placement priced
    against the commuted view and translated without the pass describes no circuit.
    """
    circuit = cnot_heavy(8, 6)
    for n_code in (4, 6):
        auto = plan_packing(circuit, n_code)
        got = measured_cost(
            circuit, auto.placement, n_code, h_commutation=auto.h_commutation
        )
        assert got is not None
        for flag in (True, False):
            forced = plan_packing(circuit, n_code, h_commutation=flag)
            ref = measured_cost(
                circuit, forced.placement, n_code, h_commutation=flag
            )
            if ref is not None:
                assert got <= ref + 1e-9


def test_plan_packing_auto_records_both_arms():
    """The arm comparison is in `considered`, so a plan says why it chose what it chose."""
    plan = plan_packing(cnot_heavy(8, 6), 6)
    arms = {label for label, _ in plan.considered if label.startswith("commute=")}
    assert arms == {"commute=True", "commute=False"}


@pytest.mark.parametrize("flag", [True, False])
def test_plan_packing_explicit_flag_overrides_auto(flag):
    circuit = cnot_heavy()
    plan = plan_packing(circuit, 6, h_commutation=flag)
    assert len(plan.placement) == circuit.num_qubits


# --- the cheap estimate is kept honest ---------------------------------


def test_rewriting_a_cnot_is_invisible_to_gate_counts():
    """
    The reason `commutation_gain` has to price depth at all.

    Commuting a Hadamard through a CNOT rewrites it to an XCX. On gate counts that
    looks like an *improvement* -- intra-patch CX is (0, 4) = 52 weighted units against
    XCX's (5, 3) = 44 -- and the entire cost is the two extra layers. Price depth at
    zero and the estimate concludes the rewrite always pays.
    """
    from iceberg.costs import DEFAULT_WORK_WEIGHTS, depth_table, work_table

    w1, w2 = DEFAULT_WORK_WEIGHTS
    work, depths = work_table(6), depth_table(6)
    cx = w1 * work["CX"][0] + w2 * work["CX"][1]
    xcx = w1 * work["XCX"][0] + w2 * work["XCX"][1]
    assert xcx < cx, "gate counts prefer XCX"
    assert depths["XCX"] > depths["CX"], "depth is where the cost actually is"


def test_commutation_gain_responds_to_depth_weight():
    """Depth has to move the number, or the knob is doing nothing."""
    circuit = cnot_heavy()
    assert commutation_gain(circuit, 6, depth_weight=0.0) != commutation_gain(
        circuit, 6, depth_weight=12.0
    )


def test_commutation_gain_is_zero_when_nothing_changes():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    assert commutation_gain(circuit, 6) == pytest.approx(0.0)


def test_absorbed_hadamards_register_as_a_gain():
    circuit = hadamard_sandwich()
    before = cancel_inverse_gates(circuit)
    after = cancel_inverse_gates(commute_hadamards(before))
    absorbed = sum(1 for i in before.data if i.operation.name.lower() == "h") - sum(
        1 for i in after.data if i.operation.name.lower() == "h"
    )
    assert absorbed > 0
    assert commutation_gain(circuit, 6) > 0
