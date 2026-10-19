"""
The compiler's first pass, one verified round of logical optimisation
(`translator.logical_optimization`): what it returns, that it always returns the same thing for
the same program, and that the planner, the translator and the QASMBench loaders all see the
one optimised program. The rest of the suite runs with the pass off (conftest.py); every test
here turns it on, with records in a temporary folder so the shared records are never touched.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from translator import logical_optimization as lo

REPO = Path(__file__).resolve().parents[1]
ADDER = REPO / "benchmarks" / "qasmbench" / "small" / "adder_n4" / "adder_n4.qasm"


@pytest.fixture(autouse=True)
def pass_on(monkeypatch, tmp_path):
    from packing import passes

    monkeypatch.setenv(lo.ENV, "1")
    monkeypatch.setenv(lo.CACHE_ENV, str(tmp_path / "records"))
    monkeypatch.setenv(lo.TIMEOUT_ENV, "0")  # in-process; the budget has its own tests
    lo.clear_memory()
    passes.clear_cache()
    yield tmp_path / "records"
    lo.clear_memory()
    passes.clear_cache()


def redundant() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qc.rz(0.3, 2)
    qc.rz(0.4, 2)
    qc.cx(1, 2)
    qc.s(1)
    qc.s(1)
    qc.h(0)
    qc.cx(0, 2)
    return qc


def ops(circuit: QuantumCircuit):
    return [(inst.operation.name, tuple(circuit.find_bit(q).index for q in inst.qubits),
             tuple(float(p) for p in inst.operation.params))
            for inst in circuit.data]


def unitary_part(circuit: QuantumCircuit) -> QuantumCircuit:
    return circuit.remove_final_measurements(inplace=False)


# --- what a round returns ---------------------------------------------------------------


def test_redundant_program_gets_smaller_and_stays_the_same_unitary():
    program = redundant()
    out = lo.logically_optimized(program)
    assert out is not program
    assert out.size() < program.size()
    assert sum(1 for i in out.data if i.operation.num_qubits == 2) <= 2
    assert Operator(out).equiv(Operator(program))
    assert out.metadata[lo.METADATA_KEY]["version"] == lo.OPTIMIZER_VERSION
    assert ops(program)[0][0] == "h"  # the caller's circuit is never touched


def test_final_measurements_come_back_on_the_same_clbits():
    program = redundant()
    program.measure_all()
    out = lo.logically_optimized(program)
    measured = [(out.find_bit(i.qubits[0]).index, out.find_bit(i.clbits[0]).index)
                for i in out.data if i.operation.name == "measure"]
    assert measured == [(0, 0), (1, 1), (2, 2)]
    assert [r.name for r in out.cregs] == [r.name for r in program.cregs]
    assert Operator(unitary_part(out)).equiv(Operator(unitary_part(program)))
    # measured and unmeasured copies of one program share one record
    assert ops(unitary_part(out)) == ops(lo.logically_optimized(redundant()))


def test_a_program_and_its_measured_copy_share_one_record():
    from experiments.shared.simulation import ensure_measurements

    proxy = QuantumCircuit(1)
    proxy.s(0)
    proxy.s(0)
    lo.register_optimized(proxy)
    measured = ensure_measurements(proxy)  # measure_all: a barrier, then the measurements
    assert lo.logically_optimized(measured) is measured
    program = redundant()
    assert ops(unitary_part(lo.logically_optimized(ensure_measurements(program)))) == ops(
        lo.logically_optimized(program))


def test_clifford_angle_rotations_come_back_as_named_gates():
    import math

    qc = QuantumCircuit(2)
    qc.rz(math.pi / 2, 0)
    qc.rx(math.pi, 1)
    qc.cx(0, 1)
    qc.rz(-math.pi / 2, 1)
    qc.rx(math.pi / 2, 0)
    qc.rz(0.3, 0)
    out = lo.logically_optimized(qc)
    clifford_rotations = [i for i in out.data if i.operation.name in ("rz", "rx")
                          and abs((float(i.operation.params[0]) / (math.pi / 2)) % 1) < 1e-9]
    assert not clifford_rotations
    assert {i.operation.name for i in out.data} <= {"h", "x", "z", "s", "sx", "rz", "rx", "cx", "cz"}
    assert Operator(out).equiv(Operator(qc))


def test_a_clifford_program_never_gains_a_non_clifford_rotation():
    from experiments.optimality.exhaustive_silent import Settings, _random_circuit
    from translator.logical_optimizer import non_clifford_rotations

    # rand:6q:06 of the exhaustive suites: plain O3 writes it with rz(pi/4), rx(-3pi/4)
    program = _random_circuit(6, 20260730 + 6, Settings())
    out = lo.logically_optimized(program)
    assert non_clifford_rotations(out) == 0
    assert Operator(out).equiv(Operator(program))


def test_no_program_gains_a_non_clifford_rotation(monkeypatch):
    """
    No new rotations, for programs that already carry some too: when the optimiser's pick has
    more rotations than the program (as O3's is for qugan_n71, 519 -> 635), the best verified
    candidate that adds none is returned.
    """
    import dataclasses

    from translator import logical_optimizer
    from translator.logical_optimizer import non_clifford_rotations, score

    program = redundant()  # rz(0.3), rz(0.4): two rotations
    real, check = logical_optimizer.optimize_verified(program, seed=0)
    assert real.changed and non_clifford_rotations(real.circuit) <= 2
    greedy = real.circuit.copy()
    for angle in (0.1, 0.2, 0.3):
        greedy.rz(angle, 0)
    rigged = dataclasses.replace(real, circuit=greedy, route="greedy",
                                 alternatives=[(score(real.circuit), 0, real.circuit, "o3")])
    monkeypatch.setattr(logical_optimizer, "optimize_verified",
                        lambda body, seed=0, **kwargs: (rigged, check))
    out = lo.logically_optimized(program)
    assert non_clifford_rotations(out) <= non_clifford_rotations(program)
    assert ops(out) == ops(lo.translator_cliffords(real.circuit))
    assert Operator(out).equiv(Operator(program))


def test_the_search_never_steps_to_more_rotations(monkeypatch):
    """
    `optimize(max_rotations=)`, the pass's limit: a route that cuts two-qubit gates by adding
    rotations is not taken, and one that cuts fewer and adds none still is -- as for
    basis_trotter_n4, where the limited search beats the study's answer on both counts.
    """
    from translator import logical_optimizer
    from translator.logical_optimizer import non_clifford_rotations, optimize

    program = QuantumCircuit(2)
    program.rz(0.3, 0)
    for _ in range(3):
        program.cx(0, 1)

    def greedy(_):
        out = QuantumCircuit(2)
        out.rz(0.3, 0)
        out.rz(0.1, 1)
        out.rz(0.2, 1)
        return out

    def careful(_):
        out = QuantumCircuit(2)
        out.rz(0.3, 0)
        out.cx(0, 1)
        return out

    monkeypatch.setattr(logical_optimizer, "route_o3", lambda circuit: None)
    routes = {"greedy": greedy, "careful": careful}
    assert optimize(program, routes=routes).route == "greedy"
    limited = optimize(program, routes=routes, max_rotations=non_clifford_rotations(program))
    assert limited.route == "careful"
    assert non_clifford_rotations(limited.circuit) == 1


def test_a_recorded_answer_that_adds_rotations_is_not_used():
    program = redundant()
    body, _ = lo._body_and_tail(program)
    rigged = program.copy()
    rigged.rz(0.1, 0)
    rigged.rz(-0.1, 0)  # the same unitary, two rotations more
    lo._write_record(lo.fingerprint(body), rigged)
    lo.clear_memory()
    assert lo.logically_optimized(program) is program


def test_seeding_refuses_an_answer_that_adds_rotations(pass_on):
    program = redundant()
    rigged = program.copy()
    rigged.rz(0.1, 0)
    rigged.rz(-0.1, 0)
    assert lo.seed_record(program, rigged) == "refused"
    assert not list(pass_on.glob("*.qpy"))


def test_already_minimal_program_comes_back_as_the_same_object():
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    assert lo.logically_optimized(bell) is bell


def test_mid_circuit_measurement_is_left_alone():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(0)
    qc.measure(0, 0)
    qc.h(0)
    qc.measure(0, 1)
    assert lo.logically_optimized(qc) is qc


# --- one answer per program -------------------------------------------------------------


def test_same_answer_every_time_in_memory_and_from_disk(pass_on):
    first = lo.logically_optimized(redundant())
    assert ops(lo.logically_optimized(redundant())) == ops(first)
    lo.clear_memory()
    assert ops(lo.logically_optimized(redundant())) == ops(first)
    assert list(pass_on.glob("*.qpy")) and list(pass_on.glob("*.fixed"))


def test_an_optimised_program_is_a_fixed_point():
    out = lo.logically_optimized(redundant())
    assert lo.logically_optimized(out) is out
    lo.clear_memory()  # the fixed-point record is on disk too
    assert lo.logically_optimized(out) is out


def test_registered_circuit_is_never_reoptimised():
    proxy = QuantumCircuit(1)
    proxy.s(0)
    proxy.s(0)  # the optimiser would make this one Z
    lo.register_optimized(proxy)
    assert lo.logically_optimized(proxy) is proxy


def _optimize_in_worker(_):
    return ops(lo.logically_optimized(redundant()))


def test_two_worker_processes_get_the_same_circuit():
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        first, second = pool.map(_optimize_in_worker, [0, 1])
    assert first == second == ops(lo.logically_optimized(redundant()))


def test_round_in_a_child_process_within_its_budget(monkeypatch):
    monkeypatch.setenv(lo.TIMEOUT_ENV, "120")
    program = redundant()
    out = lo.logically_optimized(program)
    assert out.size() < program.size()
    assert Operator(out).equiv(Operator(program))


def test_a_round_that_runs_out_of_budget_is_recorded_unchanged(monkeypatch, pass_on):
    monkeypatch.setenv(lo.TIMEOUT_ENV, "0.001")
    program = redundant()
    assert lo.logically_optimized(program) is program
    assert [p.stat().st_size for p in pass_on.glob("*.qpy")] == [0]
    monkeypatch.setenv(lo.TIMEOUT_ENV, "0")
    lo.clear_memory()
    assert lo.logically_optimized(program) is program  # the record stands


# --- the switch ---------------------------------------------------------------------------


def test_off_returns_the_program_untouched_and_reaches_workers():
    program = redundant()
    with lo.disabled():
        assert os.environ[lo.ENV] == "0"
        assert lo.logically_optimized(program) is program
        assert "logical_optimization" not in __import__(
            "translator.config", fromlist=["TranslatorConfig"]).TranslatorConfig().to_dict()
    assert os.environ[lo.ENV] == "1"
    assert lo.state() == lo.OPTIMIZER_VERSION


def test_provenance_records_the_pass():
    import pandas as pd

    from experiments.revision.common import fingerprint
    from experiments.shared.figures import LOGICAL_COLUMN, stamp_provenance
    from translator.config import TranslatorConfig

    assert TranslatorConfig().to_dict()["logical_optimization"] == lo.OPTIMIZER_VERSION
    assert stamp_provenance(pd.DataFrame({"x": [1]}))[LOGICAL_COLUMN].item() == lo.OPTIMIZER_VERSION
    on = fingerprint({"a": 1})
    with lo.disabled():
        assert fingerprint({"a": 1}) != on


# --- the planner, the translator and the loaders see one program ---------------------------


def test_planner_view_and_translation_are_of_the_optimised_program():
    from packing.tables import translated_view
    from translator.config import TranslatorConfig
    from translator.core import Translator

    program = redundant()
    program.measure_all()
    optimised = lo.logically_optimized(program)
    assert ops(translated_view(program)) == ops(translated_view(optimised))

    placement = [(0, 0), (0, 1), (1, 0)]
    translator = Translator(config=TranslatorConfig(n_code=4))
    raw = translator.translate(program, placement).circuit
    assert ops(raw) == ops(translator.translate(optimised, placement).circuit)
    with lo.disabled():
        unoptimised = translator.translate(program, placement).circuit
    assert raw.size() < unoptimised.size()


def test_plans_are_made_on_the_optimised_program():
    from packing.standard import plan_standard

    program = redundant()
    program.measure_all()
    plan = plan_standard(program)
    again = plan_standard(lo.logically_optimized(program))
    assert [tuple(p) for p in plan.placement] == [tuple(p) for p in again.placement]


@pytest.mark.skipif(not ADDER.exists(), reason="QASMBench not checked out")
def test_loader_optimises_before_making_the_proxy():
    import qiskit

    from experiments.shared.benchmark_circuits import create_proxy_from_qasm
    from translator.logical_optimizer import proxy_of

    original = create_proxy_from_qasm(ADDER, timeout_seconds=None, replace_rz_rx_with_s=False)
    proxy = create_proxy_from_qasm(ADDER, timeout_seconds=None, replace_rz_rx_with_s=True)
    with lo.disabled():
        level1 = create_proxy_from_qasm(ADDER, timeout_seconds=None,
                                        replace_rz_rx_with_s=False)
    reference = qiskit.transpile(QuantumCircuit.from_qasm_file(str(ADDER)),
                                 basis_gates=["h", "x", "z", "cx", "cz", "s", "rz", "rx"],
                                 optimization_level=1)
    assert Operator(original).equiv(Operator(unitary_part(reference)))
    assert original.size() <= level1.size()
    assert ops(proxy) == ops(proxy_of(original))
    assert lo.logically_optimized(proxy) is proxy
    # and the translator leaves both as they are
    assert lo.logically_optimized(original) is original
