"""
The parametric rotations are not Clifford, so the tableau machinery in
`tests/logical_action.py` does not apply. They are checked directly instead:
each construction must equal `exp(-i rads P / 2)` for the physical Pauli `P` that
represents the logical operator it rotates about. That is a stronger statement
than "correct on the code space" — it holds as a full unitary identity — and it
pins the sign of the angle, which a code-space-only check would leave ambiguous.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iceberg.iceberg_module import IcebergModule
from iceberg.layout import CodeLayout
from iceberg.utils import SharedInt
from tests.logical_action import assert_logical_action, physical_pauli, unitary_part

CODE_SIZES = [4, 6, 8]
ANGLE = 0.7137  # generic: not a multiple of pi/2, so Clifford aliasing can't hide errors


def make_patch(n: int) -> tuple[QuantumCircuit, IcebergModule]:
    circuit = QuantumCircuit()
    module = IcebergModule(n, circuit, SharedInt())
    module._encoded = True
    return circuit, module


def pauli_exponential(num_qubits: int, sites: dict[int, str], rads: float) -> np.ndarray:
    """exp(-i rads P / 2) = cos(rads/2) I - i sin(rads/2) P."""
    pauli = physical_pauli(num_qubits, sites).to_matrix()
    identity = np.eye(2**num_qubits)
    return np.cos(rads / 2) * identity - 1j * np.sin(rads / 2) * pauli


def rotation_cases(n: int):
    """(label, apply, pauli sites) for every rotation the patch supports."""
    layout = CodeLayout(n)
    cases = []
    for lane in layout.lanes():
        cases.append(
            (
                f"RZ({lane})",
                lambda m, lane=lane: m.RZ(lane, ANGLE),
                {layout.z_anchor: "Z", layout.lane(lane): "Z"},
            )
        )
        cases.append(
            (
                f"RX({lane})",
                lambda m, lane=lane: m.RX(lane, ANGLE),
                {layout.x_anchor: "X", layout.lane(lane): "X"},
            )
        )
    pairs = [(0, 1)] if layout.k == 2 else [(0, 1), (1, layout.k - 1)]
    for i, j in pairs:
        cases.append(
            (
                f"RZZ({i},{j})",
                lambda m, i=i, j=j: m.RZZ(i, j, ANGLE),
                {layout.lane(i): "Z", layout.lane(j): "Z"},
            )
        )
    return cases


@pytest.mark.parametrize("n", CODE_SIZES)
def test_rotations_are_pauli_exponentials(n):
    """Each rotation must equal exp(-i rads P / 2) exactly, as a full unitary."""
    for label, apply, sites in rotation_cases(n):
        circuit, module = make_patch(n)
        apply(module)
        got = Operator(circuit).data
        want = pauli_exponential(circuit.num_qubits, sites, ANGLE)
        assert np.allclose(got, want, atol=1e-9), f"n={n} {label} is not exp(-i t P / 2)"


@pytest.mark.parametrize("n", CODE_SIZES)
def test_rotation_at_quarter_turn_reproduces_s_and_sx(n):
    """
    RZ(pi/2) is S and RX(pi/2) is SX, up to global phase.

    This cross-checks the two families against each other: the rotations are
    verified above as Pauli exponentials, so agreement here confirms that S and
    SX rotate about the intended logical operator with the intended sign — which
    is exactly what the old anchor mix-up got wrong.
    """
    layout = CodeLayout(n)
    for lane in layout.lanes():
        for op, gate in (("RZ", "s"), ("RX", "sx")):
            circuit, module = make_patch(n)
            getattr(module, op)(lane, np.pi / 2)
            want = QuantumCircuit(layout.k)
            getattr(want, gate)(lane)
            assert_logical_action(
                circuit, [module], want, label=f"n={n} {op}(pi/2) vs {gate}"
            )


@pytest.mark.parametrize("n", [4, 6])
def test_fault_tolerant_rotations_are_flag_transparent(n):
    """
    In fault-tolerant mode the flagged gadgets must leave the rotation unchanged
    and return every flag ancilla to |0>.
    """
    for label, apply, sites in rotation_cases(n):
        bare_circuit, bare_module = make_patch(n)
        apply(bare_module)
        bare = Operator(bare_circuit).data

        circuit, module = make_patch(n)
        _apply_fault_tolerant(module, label)
        flagged = Operator(unitary_part(circuit)).data

        num_flags = circuit.num_qubits - bare_circuit.num_qubits
        assert num_flags > 0, f"{label} emitted no flag ancillas in FT mode"

        # |psi> (x) |0...0>_flags  must map to  (bare |psi>) (x) |0...0>_flags.
        block = _zero_flag_block(flagged, bare_circuit.num_qubits, num_flags)
        assert np.allclose(block, bare, atol=1e-9), f"n={n} {label} FT gadget is not transparent"


def _apply_fault_tolerant(module: IcebergModule, label: str) -> None:
    name, args = label.split("(")
    indices = [int(a) for a in args.rstrip(")").split(",")]
    getattr(module, name)(*indices, ANGLE, fault_tolerant=True)


def _zero_flag_block(matrix: np.ndarray, num_system: int, num_flags: int) -> np.ndarray:
    """
    The block of `matrix` on flags-in-|0> to flags-in-|0>.

    Flag registers are added after the data and parity registers, so in Qiskit's
    little-endian index order they occupy the high-order bits; the |0>-flag
    subspace is therefore the leading 2**num_system x 2**num_system block.
    """
    dim = 2**num_system
    assert matrix.shape[0] == dim * 2**num_flags
    return matrix[:dim, :dim]
