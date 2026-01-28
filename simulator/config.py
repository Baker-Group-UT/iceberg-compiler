"""
simulator/config.py
Defines configuration defaults for the Qiskit-based Simulator class.
"""
# pylint: disable=[R0902]

from dataclasses import dataclass
from typing import Optional, Self

from qiskit_aer.noise import NoiseModel
from .noise_models import (
    noise_model_to_dict,
    DefaultNoiseModel,
    StimParameterScaling,
)


@dataclass
class SimulatorConfig:
    """
    Dataclass for storing and quickly instantiating the Simulator class.
    """
    repetitions: int = 1000
    distance_noise = False
    distance_scalar = 1.0
    idle_noise = False
    idle_time_scalar = 1.0
    noise_model: Optional[NoiseModel] = None
    noiseless: bool = False
    seed: Optional[int] = None
    stim: bool = False
    stim_scaling: Optional[StimParameterScaling] = None

    def __post_init__(self):
        # Default to a realistic noise model if none specified and not noiseless
        if self.noise_model is None and not self.noiseless:
            self.noise_model = DefaultNoiseModel()

    def to_dict(self) -> dict:
        """
        Returns the simulator config as a dictionary.
        """
        return {
            "repetitions": self.repetitions,
            "distance_noise": self.distance_noise,
            "distance_scalar": self.distance_scalar,
            "idle_noise": self.idle_noise,
            "idle_time_scalar": self.idle_time_scalar,
            "noise_model": noise_model_to_dict(self.noise_model)
            if self.noise_model
            else None,
            "noiseless": self.noiseless,
            "seed": self.seed,
            "stim_scaling": self.stim_scaling.__dict__ if self.stim_scaling else None,
        }

    def with_distance_noise(self, distance_scalar: float = 1, distance_noise: bool = True) -> Self:
        self.distance_scalar = distance_scalar
        self.distance_noise = distance_noise
        return self

    def with_idle_noise(self, idle_time_scalar: float = 1.0,  idle_noise: bool = True) -> Self:
        self.idle_time_scalar = idle_time_scalar
        self.idle_noise = idle_noise
        return self

    def with_noiseless(self, noiseless: bool = True) -> Self:
        self.noiseless = noiseless
        return self

    def with_seed(self, seed: Optional[int] = None) -> Self:
        self.seed = seed
        return self

    def with_noise_model(self, noise_model: NoiseModel) -> Self:
        self.noise_model = noise_model
        return self

    def with_stim(self, stim: bool = True) -> Self:
        self.stim = stim
        return self

    def with_stim_scaling(self, scaling: StimParameterScaling = None) -> Self:
        if scaling is None:
            scaling = StimParameterScaling()
        self.stim_scaling = scaling
        return self
