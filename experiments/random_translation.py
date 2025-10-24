"""
run_scan.py - high level test file
"""
from numpy._typing._array_like import NDArray


from numpy import floating


from typing import Any


from qiskit.circuit.quantumcircuit import QuantumCircuit


import numpy as np
import qiskit

from pipeline.config import PipelineConfig
from pipeline import run_specific_scan
from simulator.config_presets import StandardSimulator
from simulator.noise_models import ScaledNoiseModel
import matplotlib.pyplot as plt
from qiskit.circuit.random import random_circuit, random_clifford_circuit

permGroup = [(0, 0), (0, 1), (1, 0), (1, 1)]
permDiff =  [(1, 0), (0, 0), (1, 1), (0, 1)]
# permDiff =  [(0, 0), (1, 0), (2, 0), (3, 0)]

for seed in range(100, 200):
    program_circuit = random_circuit(num_qubits=4, depth=10, seed=seed)

    program_circuit.measure_all()

    target_basis = ['x', 'z', 'h', 'rz', 'cx', 'cz', 'measure']

    tcirc = qiskit.transpile(
        program_circuit,
        basis_gates=target_basis,
        optimization_level=0,
    )

    results = []

    config = PipelineConfig(
        sim_config=StandardSimulator(repetitions=5_000).with_noiseless(),
        verbose=False,
        save=False,
    )

    record = run_specific_scan(
        program_circuit=tcirc,
        permutations=[permGroup, permDiff],
        config=config,
    )

    results.append((-1, record))

    f_group = []
    f_diff = []
    lsr_group = []
    lsr_diff = []

    for scalar, record in results:
        df = record.dataframe.df
        f_group.append(1- df.loc[0, "logical_total_variation_distance"])
        f_diff.append(1- df.loc[1, "logical_total_variation_distance"])
        lsr_group.append(df.loc[0, "logical_selection_rate"])
        lsr_diff.append(df.loc[1, "logical_selection_rate"])

    if f_group[0] < 0.94:
        print(seed)
        break
