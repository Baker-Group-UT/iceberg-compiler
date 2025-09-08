"""
pipeline.tasks.generate_circuits.py
"""

from __future__ import annotations
import cirq
import pandas as pd

from pipeline.config                    import PipelineConfig
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import generate_permutations
from translator.core                    import Translator


def generate_circuits(program_circuit: cirq.Circuit, config: PipelineConfig = PipelineConfig()):
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

    for t in translated:
        print(t.permutation)
        print(t.circuit)
        print("\n\n")

    return translated
