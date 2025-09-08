"""
Public package interface for the simulator module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from experiments_superstaq.simulation.sqale_leakage_model import HilbertNoiseParams, DEFAULT_PARAMS

from .simulator import Simulator, SimulatorConfig
from .noise_models import DefaultNoiseModel, StochasticNoiseModel, ScaledNoiseModel
from .utils import scale_noise_params, alpha_scale_noise_params

__all__ = [
    "HilbertNoiseParams",
    "DEFAULT_PARAMS",
    "Simulator",
    "SimulatorConfig",
    "DefaultNoiseModel",
    "StochasticNoiseModel",
    "ScaledNoiseModel",
    "scale_noise_params",
    "alpha_scale_noise_params",
]