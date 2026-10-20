"""
Render logical and translated Iceberg gate diagrams as SVGs.

Single-patch cases use a 2-logical-qubit / 1-patch layout. Inter-patch and
transversal cases use a 4-logical-qubit / 2-patch layout. Physical outputs are
filtered to data-qubit operations only, so parity qubits, flag ancillas,
measurements, and barriers are omitted from the drawings.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure matplotlib/font caches are writable inside the CLI sandbox.
MPL_CACHE_DIR = THIS_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

FONT_CACHE_DIR = THIS_DIR / ".cache"
(FONT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib
from qiskit import QuantumCircuit, QuantumRegister

from translator.context import TranslationContext
from translator.gate_handlers import translate_and_apply
from translator.merging.gates import CXTGate, CZTGate, XCXTGate

matplotlib.use("agg")


SINGLE_PATCH_PERMUTATION = [(0, 0), (0, 1)]
TWO_PATCH_PERMUTATION = [(0, 0), (0, 1), (1, 0), (1, 1)]
OUTPUT_DIR = THIS_DIR / "gates"
LOGICAL_OUTPUT_DIR = THIS_DIR / "logical_gates"
ROTATION_ANGLE = math.pi / 4
UNICODE_SUBSCRIPTS = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")


@dataclass(frozen=True)
class GateCase:
    filename: str
    builder: Callable[[], QuantumCircuit]
    permutation: Sequence[tuple[int, int]]


def _single_patch_qc() -> QuantumCircuit:
    return QuantumCircuit(2, name="logical_gate")


def _two_patch_qc() -> QuantumCircuit:
    return QuantumCircuit(4, name="logical_gate")


def _one_qubit_cases() -> List[GateCase]:
    cases: List[GateCase] = []

    single_qubit_builders = {
        "h": lambda qc, q: qc.h(q),
        "x": lambda qc, q: qc.x(q),
        "z": lambda qc, q: qc.z(q),
        "s": lambda qc, q: qc.s(q),
        "sx": lambda qc, q: qc.sx(q),
        "rx": lambda qc, q: qc.rx(ROTATION_ANGLE, q),
        "rz": lambda qc, q: qc.rz(ROTATION_ANGLE, q),
    }

    for gate_name, gate_builder in single_qubit_builders.items():
        for logical_index in (0, 1):
            def _builder(
                gate_builder: Callable[[QuantumCircuit, int], None] = gate_builder,
                logical_index: int = logical_index,
            ) -> QuantumCircuit:
                qc = _single_patch_qc()
                gate_builder(qc, logical_index)
                return qc

            cases.append(
                GateCase(
                    filename=f"{gate_name}_logical_{logical_index}.svg",
                    builder=_builder,
                    permutation=SINGLE_PATCH_PERMUTATION,
                )
            )

    return cases


def _two_qubit_cases() -> List[GateCase]:
    cases: List[GateCase] = []

    local_gate_builders = {
        "swap": lambda qc: qc.swap(0, 1),
        "cx": lambda qc: qc.cx(0, 1),
        "cz": lambda qc: qc.cz(0, 1),
        "xcx": lambda qc: qc.append(_xcx_gate(), [0, 1]),
    }

    cross_patch_builders = {
        "cx": lambda qc: qc.cx(0, 2),
        "cz": lambda qc: qc.cz(0, 2),
        "xcx": lambda qc: qc.append(_xcx_gate(), [0, 2]),
    }

    transversal_gate_builders = {
        "cxt": lambda qc: qc.append(
            CXTGate(
                control_patch=0,
                target_patch=1,
                control_count=2,
                logical_qubits=(0, 1, 2, 3),
            ),
            [0, 1, 2, 3],
        ),
        "czt": lambda qc: qc.append(
            CZTGate(
                control_patch=0,
                target_patch=1,
                control_count=2,
                logical_qubits=(0, 1, 2, 3),
            ),
            [0, 1, 2, 3],
        ),
        "xcxt": lambda qc: qc.append(
            XCXTGate(
                control_patch=0,
                target_patch=1,
                control_count=2,
                logical_qubits=(0, 1, 2, 3),
            ),
            [0, 1, 2, 3],
        ),
    }

    for name, build in local_gate_builders.items():
        def _builder(build: Callable[[QuantumCircuit], None] = build) -> QuantumCircuit:
            qc = _single_patch_qc()
            build(qc)
            return qc

        cases.append(
            GateCase(
                filename=f"{name}_intra_patch.svg",
                builder=_builder,
                permutation=SINGLE_PATCH_PERMUTATION,
            )
        )

    for name, build in cross_patch_builders.items():
        def _builder(build: Callable[[QuantumCircuit], None] = build) -> QuantumCircuit:
            qc = _two_patch_qc()
            build(qc)
            return qc

        cases.append(
            GateCase(
                filename=f"{name}_inter_patch_0_to_0.svg",
                builder=_builder,
                permutation=TWO_PATCH_PERMUTATION,
            )
        )

    for name, build in transversal_gate_builders.items():
        def _builder(build: Callable[[QuantumCircuit], None] = build) -> QuantumCircuit:
            qc = _two_patch_qc()
            build(qc)
            return qc

        cases.append(
            GateCase(
                filename=f"{name}_transversal.svg",
                builder=_builder,
                permutation=TWO_PATCH_PERMUTATION,
            )
        )

    return cases


def _xcx_gate():
    gate_circuit = QuantumCircuit(2, name="xcx")
    return gate_circuit.to_gate(label="xcx")


def _all_cases() -> List[GateCase]:
    return _one_qubit_cases() + _two_qubit_cases()


def _draw_svg(circuit: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit.draw("mpl", filename=str(path), fold=-1)


def _subscript_text(value: int) -> str:
    return str(value).translate(UNICODE_SUBSCRIPTS)


def _data_only_circuit(
    circuit: QuantumCircuit,
    data_qubits_by_module: Sequence[Sequence],
) -> QuantumCircuit:
    registers: List[QuantumRegister] = []
    qubit_map = {}
    single_patch = len(data_qubits_by_module) == 1

    for module_index, module_qubits in enumerate(data_qubits_by_module):
        for slot, old_qubit in enumerate(module_qubits):
            if single_patch:
                register = QuantumRegister(1, name=f"d{_subscript_text(slot)}")
            else:
                register = QuantumRegister(1, name=f"d{module_index}{_subscript_text(slot)}")
            registers.append(register)
            qubit_map[old_qubit] = register[0]

    filtered = QuantumCircuit(*registers, name=circuit.name)
    data_qubits = set(qubit_map)

    for instruction in circuit.data:
        if hasattr(instruction, "operation") and hasattr(instruction, "qubits"):
            inst = instruction.operation
            qargs = list(instruction.qubits)
            cargs = list(getattr(instruction, "clbits", ()))
        else:
            inst = instruction[0]
            qargs = list(instruction[1])
            cargs = list(instruction[2] if len(instruction) > 2 else ())

        if cargs or getattr(inst, "name", "").lower() == "barrier":
            continue
        if not qargs or any(qubit not in data_qubits for qubit in qargs):
            continue

        filtered.append(inst.copy(), [qubit_map[qubit] for qubit in qargs], [])

    return filtered


def _translate_case(case: GateCase, program_circuit: QuantumCircuit) -> QuantumCircuit:
    context = TranslationContext.from_config(
        list(case.permutation),
        program_circuit,
        encode_override=True,
    )

    for instruction in program_circuit.data:
        translate_and_apply(context, instruction)

    data_qubits_by_module = [module.data_qubits for module in context.iceberg_set.modules]
    return _data_only_circuit(context.iceberg_set.circuit, data_qubits_by_module)


def _encoding_only_circuit(
    permutation: Sequence[tuple[int, int]],
    num_qubits: int,
    name: str,
) -> QuantumCircuit:
    context = TranslationContext.from_config(
        list(permutation),
        QuantumCircuit(num_qubits, name=name),
        encode_override=False,
    )
    data_qubits_by_module = [module.data_qubits for module in context.iceberg_set.modules]
    return _data_only_circuit(context.iceberg_set.circuit, data_qubits_by_module)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGICAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _draw_svg(
        _encoding_only_circuit(SINGLE_PATCH_PERMUTATION, 2, "encoding_single_patch"),
        OUTPUT_DIR / "encoding_single_patch.svg",
    )
    _draw_svg(
        _encoding_only_circuit(TWO_PATCH_PERMUTATION, 4, "encoding_two_patch"),
        OUTPUT_DIR / "encoding_two_patch.svg",
    )

    for case in _all_cases():
        logical_circuit = case.builder()
        _draw_svg(logical_circuit, LOGICAL_OUTPUT_DIR / case.filename)

        translated = _translate_case(case, logical_circuit)
        _draw_svg(translated, OUTPUT_DIR / case.filename)


if __name__ == "__main__":
    main()
