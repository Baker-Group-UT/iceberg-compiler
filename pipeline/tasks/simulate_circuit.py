"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations
import cirq
import pandas as pd

from pipeline.config                    import PipelineConfig
from pipeline.analysis.postselection    import calculate_metrics_for_bare_circuit
from pipeline.analysis.result_row       import result_to_row
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations
from translator.core                    import Translator
from simulator.simulator                import Simulator
from repository.run_builder             import RunBuilder
from repository.repository              import RunRepository



def simulate_circuit(circuit: cirq.Circuit, config: PipelineConfig = PipelineConfig()):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a finalized RunRecord and saves it via RunRepository.
    """
    # Step 3: Simulate
    sim_config = config.sim_config
    simulator = Simulator(sim_config)
    simulation_results = simulator.run(circuit)

    # Step 4: Postselection
    sim_res = calculate_metrics_for_bare_circuit(circuit, simulation_results)

    # Step 5: Extract DataFrame
    row = result_to_row(sim_res, config)
    df = pd.DataFrame(row)

    # Step 6: Build and save run record
    builder = RunBuilder(
        circuit=circuit,
        circuit_name="r",
        repetitions=config.sim_config.repetitions,
    )

    builder.attach_dataframe(df)
    builder.attach_results(sim_res.to_dict())
    record = builder.finalize()
    if config.save:
        RunRepository(config.save_dir).save_run(record)
    if config.verbose:
        print(df)

    return record
