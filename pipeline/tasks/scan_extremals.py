"""
pipeline.tasks.scan_extremals.py - Lightweight permutation scan from program circuit.
Analyzes shortest and longest physical circuits per number of codes used. 
"""

from __future__ import annotations
import cirq
import pandas as pd

from pipeline.config                        import PipelineConfig
from pipeline.analysis.postselection        import apply_iceberg_postselection
from pipeline.analysis.result_row           import result_to_row
from pipeline.analysis.extremal_selection   import extremal_translation_results_per_code
from permutations                           import generate_permutations, generate_uniform_code_sets
from translator                             import Translator
from simulator                              import Simulator
from repository                             import RunBuilder, RunRepository

def run_extremal_scan(
        program_circuit: cirq.Circuit,
        config: PipelineConfig = PipelineConfig(),
        *,
        extremals_per_code: int = 1
    ):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a finalized RunRecord and saves it via RunRepository.
    """
    # Step 1: Generate permutation space
    q = len(program_circuit.all_qubits())  # Assumes 1 logical qubit per Cirq qubit
    code_sets = generate_uniform_code_sets(q)
    permutations = generate_permutations(code_sets, q)

    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate_batch(program_circuit, permutations)

    # Step 3: Determine extremal permutations of translations
    extremals = extremal_translation_results_per_code(
        translated,
        extremals_per_code=extremals_per_code,
        score_fn=lambda tr: len(tr.circuit)  # could later use total_gates or custom metric
    )

    # Step 4: Simulate
    simulator = Simulator(config=config.sim_config)
    simulation_results = [simulator.run(e.circuit) for e in extremals]

    # Step 5: Postselection
    simulation_results = apply_iceberg_postselection(extremals, simulation_results)

    # Step 6: Extract DataFrame
    rows = [result_to_row(sim_res, config) for sim_res in simulation_results]
    df = pd.DataFrame(rows)

    # Step 7: Build and save run record
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
