"""
simulator/config.py
Defines configuration defaults for the Qiskit-based Simulator class.
"""
# pylint: disable=[R0902]

from dataclasses import dataclass
from typing import Optional, Self

from qiskit_aer.noise import NoiseModel
from .noise_models import noise_model_to_dict, DefaultNoiseModel


@dataclass
class SimulatorConfig:
    """
    Dataclass for storing and quickly instantiating the Simulator class.
    """
    repetitions: int = 1000
    noise_model: Optional[NoiseModel] = None
    noiseless: bool = False
    seed: Optional[int] = None
    stim: bool = False

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
            "noise_model": noise_model_to_dict(self.noise_model)
            if self.noise_model
            else None,
            "noiseless": self.noiseless,
            "seed": self.seed,
        }

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
