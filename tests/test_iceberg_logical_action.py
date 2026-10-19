"""
Keystone suite: assert the *logical* action of every physical construction in
`iceberg/` at n = 4, 6, 8, in both bare and fault-tolerant modes.

Each test builds a patch (or a pair of patches), applies one logical operation,
and compares the induced logical Clifford against a plain reference circuit on
the logical qubits — see `tests/logical_action.py` for how that comparison works.
A test failing here means the construction is not the logical gate it claims to
be, or it leaks out of the code space, or it leaves a flag ancilla excited.

The n=4 cases are the regression gate for the k=2 behaviour the compiler was
built against; n=6 and n=8 are the generalization claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.iceberg_module import IcebergModule
from iceberg.iceberg_set import IcebergSet
from iceberg.layout import CodeLayout
from iceberg.utils import SharedInt
from tests.logical_action import assert_logical_action, logical_clifford_of

CODE_SIZES = [4, 6, 8]
FT_MODES = [False, True]


def make_patch(n: int) -> tuple[QuantumCircuit, IcebergModule]:
    """A single encoded patch, with no encoding circuit emitted."""
    circuit = QuantumCircuit()
    module = IcebergModule(n, circuit, SharedInt())
    module._encoded = True
    return circuit, module


def make_set(n: int, n_program: int) -> tuple[QuantumCircuit, IcebergSet]:
    circuit = QuantumCircuit()
    iceberg = IcebergSet(n_program=n_program, n=n, circuit=circuit)
    iceberg._encoded = True
    for module in iceberg.modules:
        module._encoded = True
    return circuit, iceberg


def reference(k: int) -> QuantumCircuit:
    return QuantumCircuit(k)


# --------------------------------------------------------------------------
# single-lane operations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize(
    "op,gate",
    [("X", "x"), ("Z", "z"), ("S", "s"), ("SX", "sx")],
)
def test_single_lane_pauli_and_phase_gates(n, op, gate):
    layout = CodeLayout(n)
    for lane in layout.lanes():
        circuit, module = make_patch(n)
        getattr(module, op)(lane)
        want = reference(layout.k)
        getattr(want, gate)(lane)
        assert_logical_action(circuit, [module], want, label=f"n={n} {op}({lane})")


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize("fault_tolerant", FT_MODES)
def test_single_lane_hadamard(n, fault_tolerant):
    layout = CodeLayout(n)
    for lane in layout.lanes():
        circuit, module = make_patch(n)
        module.H(lane, fault_tolerant=fault_tolerant)
        want = reference(layout.k)
        want.h(lane)
        assert_logical_action(circuit, [module], want, label=f"n={n} H({lane})")


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize("fault_tolerant", FT_MODES)
def test_sx_is_flag_transparent(n, fault_tolerant):
    """SX in FT mode must act identically and return its flags to |0>."""
    layout = CodeLayout(n)
    circuit, module = make_patch(n)
    module.SX(0, fault_tolerant=fault_tolerant)
    want = reference(layout.k)
    want.sx(0)
    assert_logical_action(circuit, [module], want, label=f"n={n} SX ft={fault_tolerant}")


@pytest.mark.parametrize("n", CODE_SIZES)
def test_multi_lane_pauli(n):
    """An even number of lanes must leave the anchor untouched, an odd one flip it."""
    layout = CodeLayout(n)
    for count in range(1, layout.k + 1):
        lanes = list(range(count))
        circuit, module = make_patch(n)
        module.X(lanes)
        want = reference(layout.k)
        for lane in lanes:
            want.x(lane)
        assert_logical_action(circuit, [module], want, label=f"n={n} X{lanes}")


# --------------------------------------------------------------------------
# whole-patch Hadamard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize("fault_tolerant", FT_MODES)
def test_whole_patch_hadamard(n, fault_tolerant):
    layout = CodeLayout(n)
    circuit, module = make_patch(n)
    module.HH(fault_tolerant=fault_tolerant)
    want = reference(layout.k)
    for lane in layout.lanes():
        want.h(lane)
    assert_logical_action(circuit, [module], want, label=f"n={n} HH")


@pytest.mark.parametrize("n", CODE_SIZES)
def test_hadamard_on_every_lane_routes_to_hh(n):
    """H(all lanes) must equal HH, and cost n+1 gates rather than k gadgets."""
    layout = CodeLayout(n)
    circuit, module = make_patch(n)
    module.H(list(layout.lanes()))
    want = reference(layout.k)
    for lane in layout.lanes():
        want.h(lane)
    assert_logical_action(circuit, [module], want, label=f"n={n} H(all)")
    assert sum(circuit.count_ops().values()) == n + 1


@pytest.mark.parametrize("n", CODE_SIZES)
def test_transversal_hadamard_alone_is_not_a_logical_hadamard(n):
    """
    Transversal H is H^k composed with C_A, not H^k.

    This is the trap the k=2 code hid: at k=2 the difference is a lane swap.
    """
    layout = CodeLayout(n)
    circuit, module = make_patch(n)
    for qubit in module.data_qubits:
        circuit.h(qubit)

    want = reference(layout.k)
    for lane in layout.lanes():
        want.h(lane)
    got = logical_clifford_of(circuit, [module])
    from qiskit.quantum_info import Clifford

    assert got != Clifford(want), "transversal H should differ from H^k"

    # ... and it differs by exactly C_A: x -> x XOR parity(x).
    with_correction = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    with_correction.compose(circuit, inplace=True)
    with_correction.swap(module.x_anchor, module.z_anchor)
    assert_logical_action(
        with_correction, [module], want, label=f"n={n} transversal H + C_A"
    )


# --------------------------------------------------------------------------
# two-lane operations within a patch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize("fault_tolerant", FT_MODES)
@pytest.mark.parametrize("op", ["CNOT", "CZ", "XCX", "SWAP"])
def test_two_lane_ops(n, fault_tolerant, op):
    layout = CodeLayout(n)
    lane_pairs = [(0, 1)] if layout.k == 2 else [(0, 1), (1, 3), (layout.k - 1, 0)]
    for i, j in lane_pairs:
        circuit, module = make_patch(n)
        getattr(module, op)(i, j, fault_tolerant=fault_tolerant)
        want = reference(layout.k)
        _apply_reference_two_qubit(want, op, i, j)
        assert_logical_action(
            circuit, [module], want, label=f"n={n} {op}({i},{j}) ft={fault_tolerant}"
        )


def _apply_reference_two_qubit(circuit: QuantumCircuit, op: str, i: int, j: int) -> None:
    if op == "CNOT":
        circuit.cx(i, j)
    elif op == "CZ":
        circuit.cz(i, j)
    elif op == "SWAP":
        circuit.swap(i, j)
    elif op == "XCX":
        # XCX is CZ conjugated by H on both qubits.
        circuit.h(i)
        circuit.h(j)
        circuit.cz(i, j)
        circuit.h(i)
        circuit.h(j)
    else:
        raise ValueError(op)


@pytest.mark.parametrize("n", CODE_SIZES)
def test_intra_patch_ops_reject_bad_lanes(n):
    _, module = make_patch(n)
    with pytest.raises(ValueError):
        module.CNOT(0, 0)
    with pytest.raises(ValueError):
        module.CZ(0, module.k)
    with pytest.raises(ValueError):
        module.H(module.k)
    with pytest.raises(ValueError):
        module.X([0, 0])


# --------------------------------------------------------------------------
# cross-patch operations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize("fault_tolerant", FT_MODES)
@pytest.mark.parametrize("op", ["CNOT", "CZ", "XCX"])
def test_cross_patch_targeted_ops_any_slot_pair(n, fault_tolerant, op):
    """
    Targeted cross-patch gates work for *any* pair of slots.

    Only the two weight-2 supports enter the construction, so the slots need not
    line up — the k=2 code's slot-equality requirement was an artifact.
    """
    layout = CodeLayout(n)
    k = layout.k
    slot_pairs = [(0, 0), (0, k - 1), (k - 1, 1)]
    for qi, qj in slot_pairs:
        circuit, iceberg = make_set(n, 2 * k)
        getattr(iceberg, op)(qi, k + qj, fault_tolerant=fault_tolerant)
        want = reference(2 * k)
        _apply_reference_two_qubit(want, op, qi, k + qj)
        assert_logical_action(
            circuit,
            iceberg.modules,
            want,
            label=f"n={n} cross {op}(p0s{qi}, p1s{qj}) ft={fault_tolerant}",
        )


@pytest.mark.parametrize("n", CODE_SIZES)
@pytest.mark.parametrize("op", ["CNOT_k_Transversal", "CZ_k_Transversal", "XCX_k_Transversal"])
def test_transversal_ops_are_lane_wise(n, op):
    """The transversal variants apply the same gate to every lane pair."""
    layout = CodeLayout(n)
    k = layout.k
    circuit, iceberg = make_set(n, 2 * k)
    getattr(iceberg, op)(0, 1)

    want = reference(2 * k)
    kind = {"CNOT_k_Transversal": "CNOT", "CZ_k_Transversal": "CZ", "XCX_k_Transversal": "XCX"}[op]
    for lane in range(k):
        _apply_reference_two_qubit(want, kind, lane, k + lane)
    assert_logical_action(circuit, iceberg.modules, want, label=f"n={n} {op}")


@pytest.mark.parametrize("n", CODE_SIZES)
def test_intra_patch_dispatch_from_set(n):
    """A set-level gate on two slots of one patch must use the intra-patch path."""
    layout = CodeLayout(n)
    circuit, iceberg = make_set(n, layout.k)
    iceberg.CNOT(0, 1)
    want = reference(layout.k)
    want.cx(0, 1)
    assert_logical_action(circuit, iceberg.modules, want, label=f"n={n} intra CNOT")
