"""
pipeline.tasks.generate_permutations.py
"""

from __future__ import annotations
import qiskit
import random

from pipeline.config                    import PipelineConfig
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations



def generate_permutation_set(program_circuit: qiskit.QuantumCircuit, config: PipelineConfig = PipelineConfig(), return_length: int = None):
    """
    Generates the set of permutations for a circuit.

    If return_length is specified, returns a random subset of that many permutations.
    """
    # Step 1: Generate permutation space
    q = program_circuit.num_qubits
    # Code size comes from the translator config, so one knob selects the
    # [[n, n-2, 2]] code for the whole pipeline.
    code_sets = generate_uniform_code_sets(q, n=config.translator_config.n_code)
    permutations = generate_permutations(code_sets, q)

    # Step 2: Optionally limit the number of permutations
    if return_length is not None:
        if return_length <= 0:
            raise ValueError("return_length must be positive.")
        if return_length > len(permutations):
            # Allow all if fewer permutations exist than requested
            if config.verbose:
                print(f"[WARN] Requested {return_length} permutations, but only {len(permutations)} available. Returning all.")
        else:
            permutations = random.sample(permutations, return_length)

    # Step 3: Optionally print
    if config.verbose:
        print(f"Generated {len(permutations)} permutations for {q}-qubit circuit.")
        for perm in permutations:
            print(perm)

    return permutations
