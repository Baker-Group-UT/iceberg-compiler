"""
simulator.config_presets.py - Quick presets for the Simulator class.
"""
# pylint: disable=[C0103, C0116]

from .config import SimulatorConfig
from .noise_models import StochasticNoiseModel

def TestSimulator(noiseless: bool = False):
    return SimulatorConfig(
        repetitions=1,
        noiseless=noiseless,
    )

def QuickSimulator():
    return SimulatorConfig(repetitions=100)

def StandardSimulator(repetitions: int = 1000):
    return SimulatorConfig(repetitions=repetitions)

def OversampledSimulator():
    return SimulatorConfig(repetitions=20, oversample=50)

def FinalizedSimulator():
    return SimulatorConfig(
        repetitions=10000,
        oversample=20,
    )

def StochasticSimulator():
    return SimulatorConfig(
        noise_model=StochasticNoiseModel
    )
