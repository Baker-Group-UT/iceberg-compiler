"""
extremal_selection.py - Heuristic selection of best and worst logical-to-physical translations.

Provides tools for selecting extremal TranslationResult objects based on a scoring function.
Used in the pipeline phase to avoid simulating the full permutation space by targeting the 
most and least efficient circuits per code count group (e.g., based on circuit depth).

Typical use case:
    - Given a list of TranslationResult objects from the translator,
      select the top/bottom k per number of codes used for targeted simulation.

Functionality:
    - extremal_translation_results_per_code(): returns best and worst scoring circuits
      by score_fn within each code count group.
"""

from collections import defaultdict
from typing import List, Callable

from translator.core import TranslationResult

def extremal_translation_results_per_code(
    translations: List[TranslationResult],
    score_fn: Callable[[TranslationResult], float] = lambda tr: len(tr.circuit),
    extremals_per_code: int = 1
) -> List[TranslationResult]:
    """
    Filters to best and worst scoring circuits per code count based on score_fn.
    
    Args:
        translations: List of TranslationResult objects.
        score_fn: Function to compute a score from a TranslationResult.
        extremals_per_code: Number of top and bottom results to return per code count.
    
    Returns:
        Filtered list of TranslationResult objects.
    """
    buckets = defaultdict(list)
    for tr in translations:
        codes_used = len({p for p, _ in tr.permutation})
        buckets[codes_used].append(tr)

    filtered = []
    for code_count, trs in buckets.items():
        sorted_trs = sorted(trs, key=score_fn)
        filtered.extend(sorted_trs[:extremals_per_code])               # best
        filtered.extend(sorted_trs[-extremals_per_code:])              # worst
        
    return filtered