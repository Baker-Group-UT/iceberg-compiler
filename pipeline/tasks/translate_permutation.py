"""
pipeline.tasks.translate_permutation.py
"""

from __future__ import annotations
import qiskit
from typing import List

from pipeline.config                    import PipelineConfig
from translator.core                    import Translator


def translate_permutation(program_circuit: qiskit.QuantumCircuit, permutation: List[tuple], config: PipelineConfig = PipelineConfig()):
    """
    Translates a specific circuit given a permutation.
    """
    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate(program_circuit, permutation)
    
    return translated