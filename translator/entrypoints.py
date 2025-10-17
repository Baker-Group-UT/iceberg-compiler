"""
translator.entrypoints.py - public-facing functions (Qiskit).
Mostly used for translate_circuit / translate_gate.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Any

from qiskit import QuantumCircuit

from .core import Translator, Translation, TranslationResult
from .config import TranslatorConfig

__all__ = [
    "translate_circuit",
    "translate_gate",
    "TranslationResult",
]


def translate_circuit(
    program_circuit: QuantumCircuit,
    permutation: List[Tuple[int, int]],
    *,
    fault_tolerant: bool = False,
    verbose: bool = False,
) -> TranslationResult:
    """
    Translate a logical circuit using the given permutation into a physical Iceberg circuit.
    Returns a TranslationResult with the final circuit, logs, and settings.
    """
    config = TranslatorConfig(
        fault_tolerant=fault_tolerant,
        verbose=verbose,
        n_code=4,
    )
    translator = Translator(config=config)
    return translator.translate(program_circuit, permutation)


def translate_gate(
    op: Any,  # Qiskit op: typically a (inst, qargs, cargs) tuple from QuantumCircuit.data
    permutation: List[Tuple[int, int]],
    program_circuit: QuantumCircuit,
    *,
    fault_tolerant: bool = False,
    verbose: bool = False,
    lookahead: Optional[Any] = None,  # same shape as `op` if provided
):
    """
    Apply a single logical operation using the translator’s logic (testing/debugging).
    """
    translation = Translation(
        permutation=permutation,
        program_circuit=program_circuit,
        fault_tolerant=fault_tolerant,
        verbose=verbose,
    )
    return translation.translate_gate(op, next_op=lookahead)
