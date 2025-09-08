import cirq
import math

def square_cluster_4():
    qA, qB, qC, qD = cirq.LineQubit.range(4)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.H(qB),                        # 1)  H_B
        cirq.H(qC),                        # 2)  H_C
        cirq.H(qD),                        # 3)  H_D
        cirq.CZ(qA, qB),                   # 4)  CZ_A,B
        cirq.CZ(qB, qD),                   # 5)  CZ_B,D
        cirq.CZ(qD, qC),                   # 6)  CZ_D,C
        cirq.CZ(qC, qA),                   # 7)  CZ_C,A
    )
    return {"label": "square_cluster_4", "logical": logical, "codes": [4,4,4,4]}

def grover_3():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.H(qB),                        # 1)  H_B
        cirq.H(qC),                        # 2)  H_C
        cirq.H(qC),                        # 3)  H_C
        cirq.CCX(qA, qB, qC),              # 4)  CCX_{A,B->C}
        cirq.H(qC),                        # 5)  H_C
        cirq.H(qA),                        # 6)  H_A
        cirq.H(qB),                        # 7)  H_B
        cirq.H(qC),                        # 8)  H_C
        cirq.X(qA),                        # 9)  X_A
        cirq.X(qB),                        # 10) X_B
        cirq.X(qC),                        # 11) X_C
        cirq.H(qC),                        # 12) H_C
        cirq.CCX(qA, qB, qC),              # 13) CCX_{A,B->C}
        cirq.H(qC),                        # 14) H_C
        cirq.X(qA),                        # 15) X_A
        cirq.X(qB),                        # 16) X_B
        cirq.X(qC),                        # 17) X_C
        cirq.H(qA),                        # 18) H_A
        cirq.H(qB),                        # 19) H_B
        cirq.H(qC),                        # 20) H_C
    )
    return {"label": "grover_3", "logical": logical, "codes": [4,4,4]}

# def random_bad_circuit_7_qubits():
#     qA, qB, qC, qD, qE, qF = cirq.LineQubit.range(6)
#     logical = cirq.Circuit(
#         cirq.H(qA),
#         cirq.CNOT(qA, qB),
#         cirq.CZ(qB, qC),
#         # cirq.CCX(qA, qB, qC),
#         cirq.rz(math.pi/4)(qD),
#         cirq.H(qE),
#         cirq.CNOT(qE, qF),
#         cirq.CZ(qF, qA),
#         cirq.CCX(qD, qE, qF),
#         cirq.rz(-math.pi/4)(qA),
#         cirq.H(qB),
#         cirq.CNOT(qC, qD),
#         cirq.CZ(qD, qE),
#         # cirq.CCX(qE, qF, qB),
#         cirq.rz(math.pi/4)(qC),
#         cirq.H(qC),
#         cirq.CNOT(qA, qD),
#         cirq.CZ(qB, qF),
#         # cirq.CCX(qA, qE, qF),
#         cirq.rz(-math.pi/4)(qF),
#     )
#     return {"label": "random_bad_circuit_7_qubits", "logical": logical, "codes": [4,4,4,4,4,4]}

def ghz_3():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.CNOT(qA, qB),                 # 1)  CNOT_A→B
        cirq.CNOT(qA, qC),                 # 2)  CNOT_A→C
    )
    return {"label": "ghz_3", "logical": logical, "codes": [4,4,4]}

def shor_toy_3():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.CNOT(qA, qB),                 # 1)  CNOT_A→B
        cirq.CNOT(qA, qC),                 # 2)  CNOT_A→C
        cirq.H(qA),                        # 3)  H_A
    )
    return {"label": "unscalable_shor_3", "logical": logical, "codes": [4,4,4]}

def half_adder():
    qA, qB, qS, qC = cirq.LineQubit.range(4)  # A,B,S(|0>),C(|0>)
    logical = cirq.Circuit(
        cirq.CNOT(qA, qS),                 # 0)  CNOT_A→S
        cirq.CNOT(qB, qS),                 # 1)  CNOT_B→S
        cirq.CCX(qA, qB, qC),              # 2)  CCX_{A,B->C}
    )
    return {"label": "half_adder", "logical": logical, "codes": [4,4,4,4]}

def ripple_inc_3():
    q0, q1, q2, c1, c2 = cirq.LineQubit.range(5)
    logical = cirq.Circuit(
        cirq.CNOT(q0, c1),                 # 0
        cirq.CNOT(c1, q0),                 # 1
        cirq.CCX(q0, c1, q1),              # 2

        cirq.CNOT(q1, c2),                 # 3
        cirq.CNOT(c2, q1),                 # 4
        cirq.CCX(q1, c2, q2),              # 5

        cirq.CNOT(c2, q1),                 # 6
        cirq.CNOT(q1, c2),                 # 7
        cirq.CCX(q0, c1, q1),              # 8
        cirq.CNOT(c1, q0),                 # 9
        cirq.CNOT(q0, c1),                 # 10
    )
    return {"label": "ripple_incrementer_3", "logical": logical, "codes": [4,4,4,4,4]}

def toffoli_like_snippet():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qC),                        # 0)  H_C
        cirq.CNOT(qB, qC),                 # 1)  CNOT_B→C
        cirq.rz(-math.pi / 4)(qC),         # 2)  RZ(-π/4)_C
        cirq.CNOT(qA, qC),                 # 3)  CNOT_A→C
        cirq.rz(math.pi / 4)(qC),          # 4)  RZ(π/4)_C
        cirq.CNOT(qB, qC),                 # 5)  CNOT_B→C
        cirq.rz(math.pi / 4)(qB),          # 6)  RZ(π/4)_B
        cirq.rz(-math.pi / 4)(qC),         # 7)  RZ(-π/4)_C
        cirq.CNOT(qA, qC),                 # 8)  CNOT_A→C
        cirq.CNOT(qA, qB),                 # 9)  CNOT_A→B
        cirq.rz(math.pi / 4)(qC),          # 10) RZ(π/4)_C
        cirq.H(qC),                        # 11) H_C
        cirq.rz(math.pi / 4)(qA),          # 12) RZ(π/4)_A
        cirq.rz(-math.pi / 4)(qB),         # 13) RZ(-π/4)_B
        cirq.CNOT(qA, qB),                 # 14) CNOT_A→B
    )
    return {"label": "toffoli_like_snippet", "logical": logical, "codes": [4,4,4]}