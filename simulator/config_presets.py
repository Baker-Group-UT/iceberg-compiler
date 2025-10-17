"""
simulator.config_presets.py - Quick presets for the Simulator class.
"""
# pylint: disable=[C0103, C0116]

from .config import SimulatorConfig
from .noise_models import StochasticNoiseModel, ScaledNoiseModel


def TestSimulator(noiseless: bool = False, seed: int | None = None):
    return SimulatorConfig(
        repetitions=1,
        noiseless=noiseless,
        seed=seed,
    )

def QuickSimulator(seed: int | None = None):
    return SimulatorConfig(repetitions=100, seed=seed)

def StandardSimulator(repetitions: int = 1000, seed: int | None = None):
    return SimulatorConfig(repetitions=repetitions, seed=seed)

def ScaledSimulator(repetitions: int = 1000, seed: int | None = None, alpha: float = 1.0):
    return SimulatorConfig(
        repetitions=repetitions,
        noise_model=ScaledNoiseModel(alpha=alpha),
        seed=seed,
    )

def FinalizedSimulator(seed: int | None = None):
    return SimulatorConfig(repetitions=200_000, seed=seed)

def StochasticSimulator(seed: int | None = None):
    return SimulatorConfig(
        repetitions=1000,
        noise_model=StochasticNoiseModel(),
        noiseless=False,
        seed=seed,
    )
