# simulator/transpile.py
# FIXME: ignore this file in the current state, don't use it
from __future__ import annotations
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers
import cirq
from typing import Sequence
import qiskit
from qiskit import transpile
from experiments_superstaq.converters.qiskit import cirq_to_qiskit, qiskit_to_cirq
# FIXME: also consider: cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset()) where we change the gateset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_repetition_ids.*")

def transpile_for_sqale(circuit: cirq.Circuit) -> cirq.Circuit:
    # 1) Cirq -> Qiskit
    qs: Sequence[cirq.Qid] = sorted(circuit.all_qubits())
    qk_circ: qiskit.QuantumCircuit = cirq_to_qiskit(circuit, qubits=qs)

    # 2) Transpile in Qiskit.
    #    Include 'sx' so BasisTranslator can unroll H and do CX<->CZ conjugations.
    #    Keep 'measure' so measurements survive (fine for simulators).
    # TODO: Add global rotations
    tqc = transpile(
        qk_circ,
        basis_gates=["rz", "cz", "sx", "measure"],
        optimization_level=3,
        # If you have connectivity constraints, pass a CouplingMap/Target here.
    )

    # 2b) Clean up passes that can confuse QASM import
    #     (Remove barriers; we’re not scheduling so delays shouldn’t appear.)
    pm = PassManager([RemoveBarriers()])
    tqc = pm.run(tqc)

    # 3) Qiskit -> Cirq (preserve global phase if your importer supports it)
    compiled: cirq.Circuit = qiskit_to_cirq(tqc, keep_global_phase=True)

    return compiled
