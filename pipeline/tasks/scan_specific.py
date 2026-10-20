"""
pipeline.tasks.scan_specific.py - permutation scan from program circuit for a specific permutation.
"""

from __future__ import annotations
import qiskit
import pandas as pd
from typing import List

from pipeline.config                    import PipelineConfig
from analysis.result_types     import ScanResult
from analysis.postselection    import postselect_against_reference
from analysis.result_row       import result_to_row
from analysis.print_df         import print_df_table
from translator.core                    import Translator
from simulator.simulator                import simulate_batch


def run_specific_scan(program_circuit: qiskit.QuantumCircuit, permutations: List[List[tuple]], config: PipelineConfig = PipelineConfig()):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a ScanResult holding the per-permutation DataFrame.
    """
    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate_batch(program_circuit, permutations)

    # Step 3: Simulate
    ideal_sim_res, simulation_results = simulate_batch(
        program_circuit,
        translated,
        config.sim_config,
    )

    # Step 4: Postselection
    simulation_results = postselect_against_reference(translated, ideal_sim_res, simulation_results)

    # Step 5: Extract DataFrame
    rows = [result_to_row(sim_res, config) for sim_res in simulation_results]
    df = pd.DataFrame(rows)

    record = ScanResult.of(df, [res.to_dict() for res in simulation_results])
    if config.verbose:
        print_df_table(df)

    return record
