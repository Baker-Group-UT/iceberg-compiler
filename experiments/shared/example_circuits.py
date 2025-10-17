"""
shared.example_circuits.py - various Qiskit circuits (converted from Cirq)
"""

import math
from qiskit import QuantumCircuit


def unscalable_shors_3():
    label = "unscalable_shors_3"
    qc = QuantumCircuit(3, name=label)  # qA=0, qB=1, qC=2
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4]}


def square_cluster_4():
    label = "square_cluster_4"
    qc = QuantumCircuit(4, name=label)  # qA=0, qB=1, qC=2, qD=3
    qc.h(0); qc.h(1); qc.h(2); qc.h(3)
    qc.cz(0, 1)  # A-B
    qc.cz(1, 3)  # B-D
    qc.cz(3, 2)  # D-C
    qc.cz(2, 0)  # C-A
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4, 4]}


def grover_3():
    label = "grover_3"
    qc = QuantumCircuit(3, name=label)  # qA=0, qB=1, qC=2
    qc.h(0); qc.h(1); qc.h(2)
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.h(0); qc.h(1); qc.h(2)
    qc.x(0); qc.x(1); qc.x(2)
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(0); qc.x(1); qc.x(2)
    qc.h(0); qc.h(1); qc.h(2)
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4]}


def ghz_3():
    label = "ghz_3"
    qc = QuantumCircuit(3, name=label)  # qA=0, qB=1, qC=2
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4]}


def shor_toy_3():
    label = "unscalable_shor_3"
    qc = QuantumCircuit(3, name=label)  # qA=0, qB=1, qC=2
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4]}


def half_adder():
    label = "half_adder"
    # A=0, B=1, S=2, C=3 (S and C start |0>)
    qc = QuantumCircuit(4, name=label)
    qc.cx(0, 2)        # CNOT A→S
    qc.cx(1, 2)        # CNOT B→S
    qc.ccx(0, 1, 3)    # CCX_{A,B->C}
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4, 4]}


def ripple_inc_3():
    label = "ripple_incrementer_3"
    # q0=0, q1=1, q2=2, c1=3, c2=4
    qc = QuantumCircuit(5, name=label)
    qc.cx(0, 3)        # 0
    qc.cx(3, 0)        # 1
    qc.ccx(0, 3, 1)    # 2
    qc.cx(1, 4)        # 3
    qc.cx(4, 1)        # 4
    qc.ccx(1, 4, 2)    # 5
    qc.cx(4, 1)        # 6
    qc.cx(1, 4)        # 7
    qc.ccx(0, 3, 1)    # 8
    qc.cx(3, 0)        # 9
    qc.cx(0, 3)        # 10
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4, 4, 4]}


def toffoli_like_snippet():
    label = "toffoli_like_snippet"
    qc = QuantumCircuit(3, name=label)  # qA=0, qB=1, qC=2
    qc.h(2)                      # H_C
    qc.cx(1, 2)                  # CNOT B→C
    qc.rz(-math.pi / 4, 2)       # RZ(-π/4)_C
    qc.cx(0, 2)                  # CNOT A→C
    qc.rz(math.pi / 4, 2)        # RZ(π/4)_C
    qc.cx(1, 2)                  # CNOT B→C
    qc.rz(math.pi / 4, 1)        # RZ(π/4)_B
    qc.rz(-math.pi / 4, 2)       # RZ(-π/4)_C
    qc.cx(0, 2)                  # CNOT A→C
    qc.cx(0, 1)                  # CNOT A→B
    qc.rz(math.pi / 4, 2)        # RZ(π/4)_C
    qc.h(2)                      # H_C
    qc.rz(math.pi / 4, 0)        # RZ(π/4)_A
    qc.rz(-math.pi / 4, 1)       # RZ(-π/4)_B
    qc.cx(0, 1)                  # CNOT A→B
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4, 4, 4]}

def n_ghz_state(n: int):
    label = "CNOT Ladder/Chain"
    qc = QuantumCircuit(n, label)
    qc.h(0)
    for i in range(1, n-1):
        qc.cx(i - 1, i)
    qc.measure_all()
    return {"label": label, "program": qc, "codes": [4] * n}

ALL_FACTORIES = {
    "unscalable_shors_3": unscalable_shors_3,
    "square_cluster_4": square_cluster_4,
    "grover_3": grover_3,
    "ghz_3": ghz_3,
    "unscalable_shor_3": shor_toy_3,  # keep original key -> function
    "half_adder": half_adder,
    "ripple_incrementer_3": ripple_inc_3,
    "toffoli_like_snippet": toffoli_like_snippet,
    "n_ghz_state": n_ghz_state,
}