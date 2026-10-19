"""
The cost model has to agree with the circuits actually emitted, and at n=4 it has
to reproduce the numbers the packer and the merging pass were calibrated against.
Both are asserted here, so `iceberg.costs` cannot drift from either the
constructions or the tuned weights that depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.costs import (
    DEFAULT_WORK_WEIGHTS,
    depth_table,
    flagged_overhead,
    gate_cost,
    pauli_macro_cost,
    work_table,
)
from iceberg.iceberg_module import IcebergModule
from iceberg.iceberg_set import IcebergSet
from iceberg.utils import SharedInt
from archive.packer.work_table import DEFAULT_WORK_TABLE, DEFAULT_WORK_WEIGHTS as PACKER_WEIGHTS
from translator.merging.conflicts import (
    DEFAULT_MERGE_WORK_TABLE,
    DEFAULT_MERGE_WORK_WEIGHTS,
)

CODE_SIZES = [4, 6, 8]


def emitted_counts(circuit: QuantumCircuit) -> tuple[int, int]:
    """(1q, 2q) gate counts, ignoring barriers and measurements."""
    one_qubit = two_qubit = 0
    for instruction in circuit.data:
        if instruction.operation.name in ("barrier", "measure"):
            continue
        if len(instruction.qubits) == 1:
            one_qubit += 1
        else:
            two_qubit += 1
    return one_qubit, two_qubit


def emit(n: int, apply) -> tuple[int, int]:
    circuit = QuantumCircuit()
    module = IcebergModule(n, circuit, SharedInt())
    module._encoded = True
    apply(module)
    return emitted_counts(circuit)


def emit_set(n: int, n_program: int, apply) -> tuple[int, int]:
    circuit = QuantumCircuit()
    iceberg = IcebergSet(n_program=n_program, n=n, circuit=circuit)
    iceberg._encoded = True
    for module in iceberg.modules:
        module._encoded = True
    apply(iceberg)
    return emitted_counts(circuit)


# --------------------------------------------------------------------------
# the table must match the circuits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", CODE_SIZES)
def test_single_patch_costs_match_emitted_circuits(n):
    table = work_table(n)
    cases = {
        "X": lambda m: m.X(0),
        "Z": lambda m: m.Z(0),
        # XX/ZZ are the *whole-patch* macros X/Z on all k lanes; partial
        # fusions are covered by test_partial_pauli_fusion_costs below.
        "XX": lambda m: m.X(list(range(m.k))),
        "ZZ": lambda m: m.Z(list(range(m.k))),
        "S": lambda m: m.S(0),
        "SX": lambda m: m.SX(0),
        "H": lambda m: m.H(0),
        "RZ": lambda m: m.RZ(0, 0.3),
        "RX": lambda m: m.RX(0, 0.3),
        "RZZ": lambda m: m.RZZ(0, 1, 0.3),
        "CX": lambda m: m.CNOT(0, 1),
        "CZ": lambda m: m.CZ(0, 1),
        "XCX": lambda m: m.XCX(0, 1),
    }
    for gate, apply in cases.items():
        assert emit(n, apply) == table[gate], f"n={n} {gate}"


@pytest.mark.parametrize("n", CODE_SIZES)
def test_partial_pauli_fusion_costs(n):
    """
    A fused group of m coincident X's (or Z's) emits m + (m mod 2) gates: one per
    lane, plus the anchor when the anchor factors do not cancel. The work table
    only carries the whole-patch m = k value, so the merging pass prices partial
    groups through `pauli_macro_cost` -- pinned against the emitted circuits here
    for every group size the code family admits.
    """
    k = n - 2
    for m in range(1, k + 1):
        expected = pauli_macro_cost(m)
        got_x = emit(n, lambda mod, m=m: mod.X(list(range(m))))
        got_z = emit(n, lambda mod, m=m: mod.Z(list(range(m))))
        assert got_x == expected, f"n={n} X fusion m={m}"
        assert got_z == expected, f"n={n} Z fusion m={m}"
    # The whole-patch entry and the m = k fusion must agree (k is even).
    assert work_table(n)["XX"] == pauli_macro_cost(k)
    assert work_table(n)["ZZ"] == pauli_macro_cost(k)


@pytest.mark.parametrize("n", CODE_SIZES)
def test_whole_patch_hadamard_cost_is_linear_in_n(n):
    """
    HH is n Hadamards plus a swap. The swap contributes no *pulses* -- the work table
    stays (n, 0) -- but it is not free: it is an atom exchange, priced as two
    carriages in the movement column.
    """
    from iceberg.costs import CARRIAGES, carriage_cost, carriage_weight

    one_qubit, two_qubit = emit(n, lambda m: m.HH())
    assert (one_qubit, two_qubit) == (n, 1)
    assert work_table(n)["HH"] == (n, 0)
    assert CARRIAGES["HH"] == CARRIAGES["SWAP"] == 2
    # Derived from the simulator's own rates: move_error / p1 = 1e-3 / 2e-4.
    assert carriage_weight() == pytest.approx(5.0)
    assert carriage_cost("HH") == pytest.approx(10.0)
    assert carriage_cost("CX") == 0.0


@pytest.mark.parametrize("n", CODE_SIZES)
def test_encode_and_decode_costs(n):
    circuit = QuantumCircuit()
    module = IcebergModule(n, circuit, SharedInt())
    module.encode()
    assert emitted_counts(circuit) == work_table(n)["ENCODE"]

    circuit = QuantumCircuit()
    module = IcebergModule(n, circuit, SharedInt())
    module._encoded = True
    module.decode("0")
    assert emitted_counts(circuit) == work_table(n)["DECODE"]


@pytest.mark.parametrize("n", CODE_SIZES)
def test_transversal_costs_match_emitted_circuits(n):
    table = work_table(n)
    k = n - 2
    cases = {
        "CXT": lambda ib: ib.CNOT_k_Transversal(0, 1),
        "CZT": lambda ib: ib.CZ_k_Transversal(0, 1),
        "XCXT": lambda ib: ib.XCX_k_Transversal(0, 1),
    }
    for gate, apply in cases.items():
        assert emit_set(n, 2 * k, apply) == table[gate], f"n={n} {gate}"


@pytest.mark.parametrize("n", CODE_SIZES)
def test_targeted_cross_patch_costs_are_constant_in_n(n):
    """
    The claim the packing story rests on: targeted cross-patch gates cost the
    same at every code size, while their transversal counterparts grow with n.
    """
    k = n - 2
    assert emit_set(n, 2 * k, lambda ib: ib.CNOT(0, k)) == (0, 4)
    assert emit_set(n, 2 * k, lambda ib: ib.CZ(0, k)) == (0, 4)
    assert emit_set(n, 2 * k, lambda ib: ib.XCX(0, k)) == (4, 4)


@pytest.mark.parametrize("n", CODE_SIZES)
def test_flagged_overhead_is_independent_of_n(n):
    """One flagged gate costs the same extra work at any code size."""
    bare = emit(n, lambda m: m.CNOT(0, 1))
    flagged = emit(n, lambda m: m.CNOT(0, 1, fault_tolerant=True))
    extra_1q, extra_2q, _ = flagged_overhead()
    gates_wrapped = bare[1]
    assert flagged[0] - bare[0] == extra_1q * gates_wrapped
    assert flagged[1] - bare[1] == extra_2q * gates_wrapped


# --------------------------------------------------------------------------
# the table must reproduce the compiler's calibrated n=4 numbers
# --------------------------------------------------------------------------


def test_matches_packer_table_at_n4():
    table = work_table(4)
    for gate, cost in DEFAULT_WORK_TABLE.items():
        assert table[gate] == cost, f"packer table disagrees on {gate}"


def test_matches_merge_table_at_n4():
    table = work_table(4)
    for gate, cost in DEFAULT_MERGE_WORK_TABLE.items():
        assert table[gate] == cost, f"merge table disagrees on {gate}"


def test_weights_agree_across_consumers():
    assert DEFAULT_WORK_WEIGHTS == PACKER_WEIGHTS == DEFAULT_MERGE_WORK_WEIGHTS


def test_gate_cost_uses_weights():
    w1, w2 = DEFAULT_WORK_WEIGHTS
    assert gate_cost("H", 4) == 7 * w1 + 4 * w2
    assert gate_cost("HH", 6) == 6 * w1


@pytest.mark.parametrize("bad", [3, 5, 2, 0, -4])
def test_work_table_rejects_invalid_code_sizes(bad):
    with pytest.raises(ValueError):
        work_table(bad)


# --------------------------------------------------------------------------
# the depth table must match the circuits too
# --------------------------------------------------------------------------


def emitted_depth(circuit: QuantumCircuit) -> int:
    return circuit.depth()


def emit_depth(n: int, apply) -> int:
    circuit = QuantumCircuit()
    module = IcebergModule(n, circuit, SharedInt())
    module._encoded = True
    apply(module)
    return emitted_depth(circuit)


def emit_set_depth(n: int, n_program: int, apply) -> int:
    circuit = QuantumCircuit()
    iceberg = IcebergSet(n_program=n_program, n=n, circuit=circuit)
    iceberg._encoded = True
    for module in iceberg.modules:
        module._encoded = True
    apply(iceberg)
    return emitted_depth(circuit)


#: key -> how to emit it on a single patch.
_SINGLE_PATCH_DEPTHS = {
    "X": lambda m: m.X(0),
    "Z": lambda m: m.Z(0),
    "XX": lambda m: m.X(list(range(m.k))),
    "ZZ": lambda m: m.Z(list(range(m.k))),
    "S": lambda m: m.S(0),
    "SX": lambda m: m.SX(0),
    "H": lambda m: m.H(0),
    "HH": lambda m: m.HH(),
    "RZ": lambda m: m.RZ(0, 0.3),
    "RX": lambda m: m.RX(0, 0.3),
    "CX": lambda m: m.CNOT(0, 1),
    "CZ": lambda m: m.CZ(0, 1),
    "XCX": lambda m: m.XCX(0, 1),
    "SWAP": lambda m: m.SWAP(0, 1),
}


@pytest.mark.parametrize("n", CODE_SIZES)
def test_single_patch_depths_match_emitted_circuits(n):
    table = depth_table(n)
    for key, apply in _SINGLE_PATCH_DEPTHS.items():
        assert emit_depth(n, apply) == table[key], f"{key} depth at n={n}"


@pytest.mark.parametrize("n", CODE_SIZES)
def test_cross_patch_and_transversal_depths_match_emitted_circuits(n):
    table = depth_table(n)
    k = n - 2
    cases = {
        "CX_CROSS": lambda s: s.CNOT(0, k),
        "CZ_CROSS": lambda s: s.CZ(0, k),
        "XCX_CROSS": lambda s: s.XCX(0, k),
        "CXT": lambda s: s.CNOT_k_Transversal(0, 1),
        "CZT": lambda s: s.CZ_k_Transversal(0, 1),
        "XCXT": lambda s: s.XCX_k_Transversal(0, 1),
    }
    for key, apply in cases.items():
        assert emit_set_depth(n, 2 * k, apply) == table[key], f"{key} depth at n={n}"


@pytest.mark.parametrize("n", CODE_SIZES)
def test_depths_are_constant_in_n(n):
    """
    Nothing's depth grows with the code size, unlike its gate count.

    HH is one transversal layer plus the anchor swap; the transversal two-qubit
    macros are n gates in parallel. So the table at any n equals the table at 4, and
    the whole n-dependence of the cost model lives in `work_table`.
    """
    assert depth_table(n) == depth_table(4)


def test_cnot_prefers_splitting_on_depth_but_is_indifferent_on_gates():
    """
    The fact that makes cut-minimisation the wrong objective for CNOT.

    Intra- and cross-patch CNOT cost the same gates, so a gate-count packer sees no
    reason to prefer either -- but cross-patch is a layer shallower, so a
    depth-sensitive one should actively split CNOT partners. A packer minimising cut
    weight does the opposite.
    """
    table = work_table(6)
    depths = depth_table(6)
    assert table["CX"] == (0, 4)
    assert depths["CX"] == 4
    assert depths["CX_CROSS"] == 3


def test_transversal_wins_on_both_axes_when_lanes_align():
    """One CXT beats k separate cross-patch CNOTs in gates *and* in depth."""
    for n in CODE_SIZES:
        k = n - 2
        table = work_table(n)
        depths = depth_table(n)
        w1, w2 = DEFAULT_WORK_WEIGHTS
        macro = w1 * table["CXT"][0] + w2 * table["CXT"][1]
        separate = k * (w1 * table["CX"][0] + w2 * table["CX"][1])
        assert macro < separate, f"CXT should be cheaper than {k} CNOTs at n={n}"
        assert depths["CXT"] < depths["CX_CROSS"]
