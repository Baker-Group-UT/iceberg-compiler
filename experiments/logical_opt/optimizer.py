"""
optimizer.py -- moved to `translator.logical_optimizer` (2026-10-09), when a round of logical
optimisation became the compiler's first pass (`translator.logical_optimization`). Re-exported
here so the logical-optimisation study and its cached results keep their import path.
"""

from translator.logical_optimizer import *  # noqa: F401,F403
from translator.logical_optimizer import (  # noqa: F401  (named, for readers and linters)
    CORPUS_BASIS,
    ROUTES,
    TRANSLATOR_BASIS,
    Optimized,
    check_equivalence,
    is_clifford,
    logical_counts,
    non_clifford_rotations,
    optimize,
    optimize_verified,
    proxy_of,
    score,
)
