"""
Helpers for marking circuit regions that should inject Stim noise.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction

NOISE_BEGIN = "noise_begin"
NOISE_END = "noise_end"
NOISE_MARKERS = {NOISE_BEGIN, NOISE_END}


def noise_begin_instruction() -> Instruction:
    return Instruction(name=NOISE_BEGIN, num_qubits=0, num_clbits=0, params=[])


def noise_end_instruction() -> Instruction:
    return Instruction(name=NOISE_END, num_qubits=0, num_clbits=0, params=[])


def append_noise_begin(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit.append(noise_begin_instruction(), [], [])
    return circuit


def append_noise_end(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit.append(noise_end_instruction(), [], [])
    return circuit


def is_noise_marker_name(name: str) -> bool:
    return name.lower() in NOISE_MARKERS


def strip_noise_markers(circuit: QuantumCircuit) -> QuantumCircuit:
    stripped = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name in NOISE_MARKERS:
            continue
        stripped.append(inst.operation, inst.qubits, inst.clbits)
    return stripped
