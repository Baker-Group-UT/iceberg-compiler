"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations
from typing import List
import qiskit
import pandas as pd

from pipeline.config                    import PipelineConfig
from analysis.result_types     import ScanResult
from analysis.postselection    import postselect_against_reference
from analysis.result_row       import result_to_row
from analysis.print_df         import print_df_table
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations
from translator.core                    import Translator
from simulator.simulator                import simulate_batch


def run_full_scan(program_circuit: qiskit.QuantumCircuit, config: PipelineConfig = PipelineConfig(), added_perms: List[List[tuple]] = None):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a ScanResult holding the per-permutation DataFrame.
    """
    # Step 1: Generate permutation space
    q = program_circuit.num_qubits  # Assumes 1 logical qubit per qubit
    # Code size comes from the translator config, so one knob selects the
    # [[n, n-2, 2]] code for the whole pipeline.
    code_sets = generate_uniform_code_sets(q, n=config.translator_config.n_code)
    permutations = generate_permutations(code_sets, q)

    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate_batch(program_circuit, permutations)

    # # Step 3: Simulate
    ideal_sim_res, simulation_results = simulate_batch(
        program_circuit,
        translated,
        config.sim_config,
    )

    # # Step 4: Postselection
    simulation_results = postselect_against_reference(
        translated,
        ideal_sim_res,
        simulation_results
    )

    # Step 5: Extract DataFrame
    rows = [result_to_row(sim_res, config) for sim_res in simulation_results]
    df = pd.DataFrame(rows)

    record = ScanResult.of(df, [res.to_dict() for res in simulation_results])
    if config.verbose:
        print_df_table(df)

    return record
