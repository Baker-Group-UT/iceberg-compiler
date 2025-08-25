"""
Public package interface for the simulator module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from .simulator import Simulator, SimulatorConfig

__all__ = [
    "Simulator",
    "SimulatorConfig",
]