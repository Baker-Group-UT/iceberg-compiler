"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations
import qiskit
import pandas as pd

from pipeline.config                    import PipelineConfig
from pipeline.analysis.postselection    import calculate_metrics_for_bare_circuit
from pipeline.analysis.result_row       import result_to_row
from pipeline.analysis.print_df         import print_df_table
from simulator.simulator                import Simulator
from repository.run_builder             import RunBuilder
from repository.repository              import RunRepository

def simulate_circuit(circuit: qiskit.QuantumCircuit, config: PipelineConfig = PipelineConfig()):
    """
    Simulates a specific circuit.
    """
    # Step 3: Simulate
    sim_config = config.sim_config
    simulator = Simulator(sim_config)
    simulation_results = simulator.run(circuit)

    # Step 4: Postselection
    sim_res = calculate_metrics_for_bare_circuit(circuit, simulation_results)

    return sim_res
