"""
simulator.config.py - This file is responsible for setting 
the class variables for the Simulator class. 
"""
# pylint: disable=[R0902]

from dataclasses import dataclass
from typing import Optional, Self
import cirq

from experiments_superstaq.simulation.sqale_leakage_model import SqaleLeakageModel
from .noise_models import noise_model_to_dict

@dataclass
class SimulatorConfig:
    """
    Dataclass for storing and quickly instantiating the Simulator class.
    """
    repetitions: int = 1000
    oversample: int = 1
    ch_form: bool = False
    preserve_leakage: bool = False
    noise_model: Optional[cirq.NoiseModel] = cirq.NoiseModel.from_noise_model_like(
        cirq.depolarize(0.0)  # no-op by default
    )    
    noiseless: bool = False
    seed: Optional[int] = None
    progressbar: bool = False
    sqale: bool = True

    def to_dict(self) -> dict:
        """
        Returns the simulator config as a dictionary.
        """
        return {
            "repetitions": self.repetitions,
            "oversample": self.oversample,
            "ch_form": self.ch_form,
            "preserve_leakage": self.preserve_leakage,
            "noise_model": noise_model_to_dict(self.noise_model),
            "noiseless": self.noiseless,
            "seed": self.seed,
            "sqale": self.sqale,
        }

    def with_noiseless(self, noiseless: bool = True) -> Self:
        self.noiseless = noiseless
        return self

    def with_seed(self, seed: int = None) -> Self:
        self.seed = seed
        return self