"""
Public package interface for the simulator module.

Expose high-level entrypoints and core classes for convenient imports.
"""

from .noise_models import (
    NEUTRAL_ATOM,
    ScaledNoiseModel,
    StimNoiseParameters,
    StochasticNoiseModel,
)
from .noise_markers import (
    append_noise_begin,
    append_noise_end,
    noise_begin_instruction,
    noise_end_instruction,
    strip_noise_markers,
)
from .simulator import Simulator, SimulatorConfig

__all__ = [
    "Simulator",
    "SimulatorConfig",
    "NEUTRAL_ATOM",
    "StimNoiseParameters",
    "StochasticNoiseModel",
    "ScaledNoiseModel",
    "append_noise_begin",
    "append_noise_end",
    "noise_begin_instruction",
    "noise_end_instruction",
    "strip_noise_markers",
]
