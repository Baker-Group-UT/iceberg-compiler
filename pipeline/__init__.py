"""
Public package interface for the pipeline module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from .config import PipelineConfig
from .tasks.scan_all import run_full_scan
from .tasks.scan_extremals import run_extremal_scan

__all__ = [
    "PipelineConfig",
    "run_full_scan",
    "run_extremal_scan",
]