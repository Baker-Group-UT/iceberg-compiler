"""
pipeline.tasks.scan_all.py - Full permutation scan from program circuit.
"""

from __future__ import annotations

import qiskit

from pipeline.config                    import PipelineConfig
from pipeline.analysis.postselection    import calculate_metrics_for_bare_circuit
from pipeline.analysis.result_types import PostselectedSimulationResult
from simulator.simulator                import Simulator

def simulate_circuit(circuit: qiskit.QuantumCircuit,
    config: PipelineConfig = PipelineConfig()
) -> PostselectedSimulationResult:
    """
    Simulates a specific circuit.
    """
    # Step 3: Simulate
    sim_config = config.sim_config
    simulator = Simulator(sim_config)
    simulation_results = simulator.run(circuit)

    # Step 4: Postselection
    sim_res: PostselectedSimulationResult = calculate_metrics_for_bare_circuit(circuit, simulation_results)

    return sim_res
