"""
random_translation.py - high level test file
"""

from pipeline.config import PipelineConfig
from pipeline import run_random_scan, run_specific_scan
from simulator.config_presets import StandardSimulator
from qiskit.circuit.random import random_clifford_circuit

permGroup = [(0, 0), (0, 1)]
# permGroup = [(0, 0), (0, 1)]
# permDiff =  [(0, 0), (1, 0), (2, 0), (3, 0)]
# permDiff =  [(0, 0), (1, 0), (2, 0), (3, 0)]

for seed in range(72, 200):
    # S gate seems a lil broken. Also more than like 10 qubits will eventually fail. Not sure why.
    # program_circuit = random_clifford_circuit(num_qubits=10, num_gates=30, seed=seed, gates=['x', 'z', 'h', 'cx', 'cz'])
    program_circuit = random_clifford_circuit(num_qubits=2, num_gates=25, seed=seed, gates=['x', 'z', 's', 'h', 'cx', 'cz'])

    program_circuit.measure_all()

    results = []

    config = PipelineConfig(
        sim_config=StandardSimulator(repetitions=25_000).with_noiseless(),#.with_stim(),
        verbose=False,
        save=False,
    )

    record = run_specific_scan(
        program_circuit=program_circuit,
        permutations=[permGroup],
        config=config,
    )

    # record = run_random_scan(program_circuit=program_circuit, m=2, config=config)

    results.append((-1, record))

    f_group = []
    f_diff = []
    lsr_group = []
    lsr_diff = []

    for scalar, record in results:
        df = record.dataframe.df
        f_group.append(1 - df.loc[0, "logical_total_variation_distance"])
        # f_diff.append(1 - df.loc[1, "logical_total_variation_distance"])
        lsr_group.append(df.loc[0, "logical_selection_rate"])
        # lsr_diff.append(df.loc[1, "logical_selection_rate"])

    if f_group[0] < 0.94:
        print(seed)
        print("fid:", f_group[0])
        break
