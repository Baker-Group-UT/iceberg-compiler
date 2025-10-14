"""
pipeline.tasks.translate_permutation.py
"""

from __future__ import annotations
import cirq
from typing import List

from pipeline.config                    import PipelineConfig
from translator.core                    import Translator


def translate_permutation(program_circuit: cirq.Circuit, permutation: List[tuple], config: PipelineConfig = PipelineConfig()):
    """
    Translates a specific circuit given a permutation.
    """
    # Step 2: Translate logical to physical
    translator = Translator(config=config.translator_config)
    translated = translator.translate(program_circuit, permutation)
    
    return translated