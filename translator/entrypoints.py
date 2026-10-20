"""
translator.entrypoints.py - public-facing functions (Qiskit).
Mostly used for translate_circuit / translate_gate.
"""

from __future__ import annotations
from typing import List, Tuple, Any

from qiskit import QuantumCircuit

from .core import Translator, Translation, TranslationResult
from .config import TranslatorConfig
from .utils import DEFAULT_CODE_SIZE

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
    n_code: int = DEFAULT_CODE_SIZE,
) -> TranslationResult:
    """
    Translate a logical circuit using the given permutation into a physical Iceberg circuit.
    Returns a TranslationResult with the final circuit, logs, and settings.

    `n_code` selects the [[n, n-2, 2]] code size. Merging only runs at n_code=4
    (see `Translator._merging_available`).
    """
    config = TranslatorConfig(
        fault_tolerant=fault_tolerant,
        verbose=verbose,
        n_code=n_code,
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
    n_code: int = DEFAULT_CODE_SIZE,
):
    """
    Apply a single logical operation using the translator’s logic (testing/debugging).
    """
    translation = Translation(
        permutation=permutation,
        program_circuit=program_circuit,
        fault_tolerant=fault_tolerant,
        verbose=verbose,
        n_code=n_code,
    )
    return translation.translate_gate(op)
