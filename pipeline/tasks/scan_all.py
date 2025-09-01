"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations
import cirq
import pandas as pd

from pipeline.config                    import PipelineConfig
from pipeline.analysis.postselection    import apply_iceberg_postselection
from pipeline.analysis.result_row       import result_to_row
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations
from translator.core                    import Translator
from simulator.simulator                import Simulator
from repository.run_builder             import RunBuilder
from repository.repository              import RunRepository


def run_full_scan(program_circuit: cirq.Circuit, config: PipelineConfig = PipelineConfig()):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a finalized RunRecord and saves it via RunRepository.
    """
    # Step 1: Generate permutation space
    q = len(program_circuit.all_qubits())  # Assumes 1 logical qubit per Cirq qubit
    code_sets = generate_uniform_code_sets(q)
    permutations = generate_permutations(code_sets, q)
    print(permutations)
    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate_batch(program_circuit, permutations)

    # Step 3: Simulate
    sim_config = config.sim_config
    simulator = Simulator(sim_config)
    simulation_results = [simulator.run(t.circuit) for t in translated]

    # Step 4: Postselection
    simulation_results = apply_iceberg_postselection(translated, simulation_results)

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
        print(df)

    return record
