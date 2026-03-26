from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.noise_markers import append_noise_begin, append_noise_end
from simulator.stim_backend import qiskit_to_stim


def test_noise_markers_only_apply_noise_inside_marked_regions():
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    append_noise_begin(circuit)
    circuit.x(0)
    append_noise_end(circuit)
    circuit.z(0)
    append_noise_begin(circuit)
    circuit.x(0)
    append_noise_end(circuit)
    circuit.measure(0, 0)

    stim_circuit = qiskit_to_stim(
        circuit,
        include_noise=True,
    )
    stim_text = str(stim_circuit)

    assert stim_text.count("DEPOLARIZE1") == 2
    assert "X_ERROR" not in stim_text


def test_noise_end_keeps_measurement_noiseless():
    circuit = QuantumCircuit(1, 1)
    append_noise_begin(circuit)
    circuit.x(0)
    append_noise_end(circuit)
    circuit.measure(0, 0)

    stim_circuit = qiskit_to_stim(
        circuit,
        include_noise=True,
    )
    stim_text = str(stim_circuit)

    assert "DEPOLARIZE1" in stim_text
    assert "X_ERROR" not in stim_text
