# translator/entrypoints.py
"""
translator.entrypoints.py - public facing accessible functions.
Mostly used for translate_circuit
"""

from __future__ import annotations
from typing import List, Tuple
import cirq

from .core import Translator, Translation, TranslationResult
from .config import TranslatorConfig

__all__ = [
    "translate_circuit",
    "translate_gate",
    "TranslationResult",
]

def translate_circuit(
    program_circuit: cirq.Circuit,
    permutation: List[Tuple[int, int]],
    *,
    fault_tolerant: bool = False,
    verbose: bool = False,
) -> TranslationResult:
    """
    Translates a logical circuit using the given permutation into a physical Iceberg circuit.
    Returns a TranslationResult object with the final circuit, logs, and settings.
    """
    config = TranslatorConfig(
        fault_tolerant= fault_tolerant,
        verbose=verbose,
        n_code=4
    )
    translator = Translator(config=config)
    return translator.translate(program_circuit, permutation)

def translate_gate(
    op: cirq.Operation,
    permutation: List[Tuple[int, int]],
    program_circuit: cirq.Circuit,
    *,
    fault_tolerant: bool = False,
    verbose: bool = False,
    lookahead: cirq.Operation | None = None,
):
    """
    Applies a single logical operation using the translator’s logic, for testing/debugging.
    """
    translation = Translation(
        permutation=permutation,
        program_circuit=program_circuit,
        fault_tolerant=fault_tolerant,
        verbose=verbose,
    )
    return translation.translate_gate(op, next_op=lookahead)
