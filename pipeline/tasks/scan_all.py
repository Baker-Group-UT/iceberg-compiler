"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations
import qiskit
import pandas as pd
from tqdm import trange

from pipeline.config                    import PipelineConfig
from pipeline.analysis.postselection    import postselect_against_reference
from pipeline.analysis.result_row       import result_to_row
from pipeline.analysis.print_df         import print_df_table
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations
from translator.core                    import Translator
from simulator.simulator                import Simulator
from repository.run_builder             import RunBuilder
from repository.repository              import RunRepository


def run_full_scan(program_circuit: qiskit.QuantumCircuit, config: PipelineConfig = PipelineConfig()):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a finalized RunRecord and saves it via RunRepository.
    """
    # Step 1: Generate permutation space
    q = program_circuit.num_qubits  # Assumes 1 logical qubit per qubit
    code_sets = generate_uniform_code_sets(q)
    permutations = generate_permutations(code_sets, q)

    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate_batch(program_circuit, permutations)

    for t in translated:
        print(t.circuit)

    # # Step 3: Simulate
    simulator = Simulator(config.sim_config)
    ideal_sim_res = simulator.ideal(circuit=program_circuit)
    simulation_results = []
    for i in trange(len(translated), desc="Simulating circuits", unit=" sim"):
        simulation_results.append(simulator.noisy(translated[i].circuit))

    # # Step 4: Postselection
    simulation_results = postselect_against_reference(
        translated,
        ideal_sim_res,
        simulation_results
    )

    # Step 5: Extract DataFrame
    rows = [result_to_row(sim_res, config) for sim_res in simulation_results]
    df = pd.DataFrame(rows)

    # Step 6: Build and save run record
    builder = RunBuilder(
        circuit=program_circuit,
        circuit_name="r",
        repetitions=config.sim_config.repetitions,
    )

    builder.attach_dataframe(df)
    builder.attach_results([res.to_dict() for res in simulation_results])
    record = builder.finalize()
    if config.save:
        RunRepository(config.save_dir).save_run(record)
    if config.verbose:
        print_df_table(df)

    return record
