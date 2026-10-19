"""
pipeline.tasks.generate_circuit_random.py
"""

from __future__ import annotations
import qiskit

from pipeline.config                    import PipelineConfig
from permutations.code_sets             import generate_uniform_code_sets
from permutations.generator             import random_permutation_from_circuit
from translator.core                    import Translator


def generate_random_circuit(program_circuit: qiskit.QuantumCircuit, config: PipelineConfig = PipelineConfig()):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a finalized RunRecord and saves it via RunRepository.
    """
    q = program_circuit.num_qubits
    code_sets = generate_uniform_code_sets(q)
    permutation = random_permutation_from_circuit(
        circuit=program_circuit,
        code_sizes=code_sets,
        seed=config.seed
    )

    translator = Translator(config=config.translator_config)
    translated = translator.translate_batch(program_circuit, [permutation])

    return translated
