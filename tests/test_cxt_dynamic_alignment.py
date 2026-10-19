from __future__ import annotations

import sys
from pathlib import Path

import qiskit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translator.context import TranslationContext
from translator.handlers.controlled_not import apply_logical_cxt, place_logical_cnot
from translator.handlers.controlled_x import apply_logical_xcxt, place_logical_xcx
from translator.handlers.controlled_z import apply_logical_czt, place_logical_cz
from translator.merging.gates import CXTGate
from translator.merging.gates import CZTGate, XCXTGate


class _MacroOp:
    def __init__(self, gate, qubits):
        self.gate = gate
        self.qubits = list(qubits)


def test_cxt_uses_current_logical_positions_after_prior_cx_swap():
    qc = qiskit.QuantumCircuit(4)
    permutation = [(0, 0), (0, 1), (1, 0), (1, 1)]
    context = TranslationContext.from_config(
        permutation,
        qc,
        encode_override=True,
    )

    recorded_ops: list[tuple[str, object]] = []

    def record_swap(swaps, fault_tolerant=False):
        recorded_ops.append(("swap", (swaps, fault_tolerant)))

    def record_cxt(control_patch, target_patch):
        recorded_ops.append(("cxt", (control_patch, target_patch)))

    context.iceberg_set.SWAP = record_swap
    context.iceberg_set.CNOT_k_Transversal = record_cxt
    context.iceberg_set.CNOT = lambda *args, **kwargs: None

    # CX(0,3) is misaligned under the tight packing, so the control patch swaps.
    assert place_logical_cnot(0, 3, context) is True
    assert context.current_pos_of(0) == (0, 1)
    assert context.current_pos_of(1) == (0, 0)
    recorded_ops.clear()

    op = _MacroOp(
        CXTGate(
            control_patch=1,
            target_patch=0,
            control_count=2,
            logical_qubits=(2, 3, 0, 1),
        ),
        [qc.qubits[2], qc.qubits[3], qc.qubits[0], qc.qubits[1]],
    )

    assert apply_logical_cxt(op, context) is True
    assert recorded_ops == [
        ("swap", ([[0, 0, 1]], False)),
        ("cxt", (1, 0)),
    ]
    assert context.current_pos_of(0) == (0, 0)
    assert context.current_pos_of(1) == (0, 1)


def test_czt_uses_current_logical_positions_after_prior_cz_swap():
    qc = qiskit.QuantumCircuit(4)
    permutation = [(0, 0), (0, 1), (1, 0), (1, 1)]
    context = TranslationContext.from_config(
        permutation,
        qc,
        encode_override=True,
    )

    recorded_ops: list[tuple[str, object]] = []

    def record_swap(swaps, fault_tolerant=False):
        recorded_ops.append(("swap", (swaps, fault_tolerant)))

    def record_czt(control_patch, target_patch):
        recorded_ops.append(("czt", (control_patch, target_patch)))

    context.iceberg_set.SWAP = record_swap
    context.iceberg_set.CZ_k_Transversal = record_czt
    context.iceberg_set.CZ = lambda *args, **kwargs: None

    assert place_logical_cz(0, 3, context) is True
    assert context.current_pos_of(0) == (0, 1)
    assert context.current_pos_of(1) == (0, 0)
    recorded_ops.clear()

    op = _MacroOp(
        CZTGate(
            control_patch=1,
            target_patch=0,
            control_count=2,
            logical_qubits=(2, 3, 0, 1),
        ),
        [qc.qubits[2], qc.qubits[3], qc.qubits[0], qc.qubits[1]],
    )

    assert apply_logical_czt(op, context) is True
    assert recorded_ops == [
        ("swap", ([[0, 0, 1]], False)),
        ("czt", (1, 0)),
    ]
    assert context.current_pos_of(0) == (0, 0)
    assert context.current_pos_of(1) == (0, 1)


def test_xcxt_uses_current_logical_positions_after_prior_xcx_swap():
    qc = qiskit.QuantumCircuit(4)
    permutation = [(0, 0), (0, 1), (1, 0), (1, 1)]
    context = TranslationContext.from_config(
        permutation,
        qc,
        encode_override=True,
    )

    recorded_ops: list[tuple[str, object]] = []

    def record_swap(swaps, fault_tolerant=False):
        recorded_ops.append(("swap", (swaps, fault_tolerant)))

    def record_xcxt(control_patch, target_patch):
        recorded_ops.append(("xcxt", (control_patch, target_patch)))

    context.iceberg_set.SWAP = record_swap
    context.iceberg_set.XCX_k_Transversal = record_xcxt
    context.iceberg_set.XCX = lambda *args, **kwargs: None

    assert place_logical_xcx(0, 3, context) is True
    assert context.current_pos_of(0) == (0, 1)
    assert context.current_pos_of(1) == (0, 0)
    recorded_ops.clear()

    op = _MacroOp(
        XCXTGate(
            control_patch=1,
            target_patch=0,
            control_count=2,
            logical_qubits=(2, 3, 0, 1),
        ),
        [qc.qubits[2], qc.qubits[3], qc.qubits[0], qc.qubits[1]],
    )

    assert apply_logical_xcxt(op, context) is True
    assert recorded_ops == [
        ("swap", ([[0, 0, 1]], False)),
        ("xcxt", (1, 0)),
    ]
    assert context.current_pos_of(0) == (0, 0)
    assert context.current_pos_of(1) == (0, 1)
