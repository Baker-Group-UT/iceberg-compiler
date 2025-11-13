"""
random_translation.py - high level test file
"""

from pipeline.config import PipelineConfig
from pipeline import run_random_scan, run_specific_scan
from simulator.config_presets import ScaledSimulator, StandardSimulator
from qiskit.circuit.random import random_clifford_circuit

permGroup = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
# permGroup = [(0, 0), (0, 1)]
# permDiff =  [(0, 0), (1, 0), (2, 0), (3, 0)]
# permDiff =  [(0, 0), (1, 0), (2, 0), (3, 0)]

for seed in range(131, 200):
    # S gate seems a lil broken. Also more than like 10 qubits will eventually fail. Not sure why.
    # program_circuit = random_clifford_circuit(num_qubits=10, num_gates=30, seed=seed, gates=['x', 'z', 'h', 'cx', 'cz'])
    program_circuit = random_clifford_circuit(num_qubits=30, num_gates=100, seed=seed, gates=['x', 'z', 'h', 's', 'cx', 'cz'])

    program_circuit.measure_all()

    results = []

    config = PipelineConfig(
        sim_config=ScaledSimulator(repetitions=50_000, alpha=0.1).with_stim(), #.with_noiseless(),
        verbose=False,
        save=False,
    )

    record = run_random_scan(
        program_circuit=program_circuit,
        # permutations=[permGroup],
        m=1,
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

    if f_group[0] < 0.6:
        print(seed)
        print("fid:", f_group[0])
        print(program_circuit)
        break
