"""
pipeline.tasks.generate_circuit.py
"""

from __future__ import annotations
from typing import List, Tuple
import qiskit

from pipeline.config                    import PipelineConfig
from translator.core                    import Translator


def generate_circuit(program_circuit: qiskit.QuantumCircuit, permutation: List[Tuple[int, int]], config: PipelineConfig = PipelineConfig()):
    """
    Runs a full permutation scan for the provided logical circuit using the given config.
    Returns a finalized RunRecord and saves it via RunRepository.
    """
    translator = Translator(config=config.translator_config)
    translated = translator.translate(program_circuit, permutation)

    return translated
