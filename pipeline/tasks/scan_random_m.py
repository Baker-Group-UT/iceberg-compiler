"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations
from logging import warning
import random
import qiskit
import pandas as pd
from typing import List

from pipeline.config                    import PipelineConfig
from analysis.result_types     import ScanResult
from analysis.postselection    import postselect_against_reference
from analysis.result_row       import result_to_row
from analysis.print_df         import print_df_table
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import random_permutation_from_circuit
from translator.core                    import Translator
from simulator.simulator                import simulate_batch

REPEAT_LIMIT = 2500

def run_random_scan(program_circuit: qiskit.QuantumCircuit, m: int = 100, config: PipelineConfig = PipelineConfig(), added_perms: List[List[tuple]] = None):
    """
    Runs a permutation scan for m random permutations of the provided logical circuit using the given config.
    Returns a ScanResult holding the per-permutation DataFrame.
    """
    # Step 1: Generate permutation set
    q = program_circuit.num_qubits  # Assumes 1 logical qubit per qubit
    # Code size comes from the translator config, so one knob selects the
    # [[n, n-2, 2]] code for the whole pipeline.
    code_sets = generate_uniform_code_sets(q, n=config.translator_config.n_code)
    failures = 0
    permutations = []
    rng = random.Random(config.seed)
    while(len(permutations) < m and failures < REPEAT_LIMIT):
        perm_seed = rng.randrange(2**63 - 1)
        perm = random_permutation_from_circuit(program_circuit, code_sets, seed=perm_seed)
        if perm not in permutations:
            permutations.append(perm)
        else:
            failures += 1
    if len(permutations) < m:
        if failures == REPEAT_LIMIT:
            warning("Permutation set incomplete, reached 2,500 addition failures to permutation set.")
        else:
            warning(f"Permutation set incomplete. For some reason, the permutation set if not of length m but length {len(permutations)}")

    if added_perms is not None:
        for perm in added_perms:
            permutations.append(perm)

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
