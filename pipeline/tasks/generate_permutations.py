"""
pipeline.tasks.generate_permutations.py
"""

from __future__ import annotations
import cirq

from pipeline.config                    import PipelineConfig
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations


def generate_permutation_set(program_circuit: cirq.Circuit, config: PipelineConfig = PipelineConfig()):
    """
    Generates the set of permutations for a circuit
    """
    # Step 1: Generate permutation space
    q = len(program_circuit.all_qubits())  # Assumes 1 logical qubit per Cirq qubit
    code_sets = generate_uniform_code_sets(q)
    permutations = generate_permutations(code_sets, q)

    if config.verbose:
        for perm in permutations:
            print(perm)

    return permutations
