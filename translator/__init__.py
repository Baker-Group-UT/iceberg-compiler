"""
Public package interface for the translator module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from .entrypoints import translate_circuit, translate_gate, TranslationResult
from .core import Translator, Translation
from .context import TranslationContext

__all__ = [
    "translate_circuit",
    "translate_gate",
    "TranslationResult",
    "Translator",
    "Translation",
    "TranslationContext",
]