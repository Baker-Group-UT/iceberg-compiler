"""
Public package interface for the pipeline module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from .config import PipelineConfig
from .tasks.scan_all import run_full_scan
from .tasks.scan_extremals import run_extremal_scan
from .tasks.scan_specific import run_specific_scan
from .tasks.simulate_circuit import simulate_circuit
from .tasks.generate_circuits import generate_circuits
from .tasks.generate_permutations import generate_permutation_set

__all__ = [
    "PipelineConfig",
    "run_full_scan",
    "run_extremal_scan",
    "run_specific_scan",
    "simulate_circuit",
    "generate_circuits",
    "generate_permutation_set",
]