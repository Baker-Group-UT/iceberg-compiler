"""
Public package interface for the simulator module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from .noise_models import (
    DefaultNoiseModel,
    DistanceDependentNoiseModel,
    ScaledNoiseModel,
    StochasticNoiseModel,
    distance_noise_function,
)
from .simulator import Simulator, SimulatorConfig

__all__ = [
    "Simulator",
    "SimulatorConfig",
    "DefaultNoiseModel",
    "StochasticNoiseModel",
    "ScaledNoiseModel",
    "DistanceDependentNoiseModel",
    "distance_noise_function",
]

