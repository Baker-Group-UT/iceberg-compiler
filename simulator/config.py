"""
simulator.config.py - This file is responsible for setting 
the class variables for the Simulator class. 
Contains multiple configs set for quick simulator setup.
"""
# pylint: disable=[R0902]

from dataclasses import dataclass
from typing import Optional

from experiments_superstaq.simulation.sqale_leakage_model import SqaleLeakageModel
from .noise_models import DefaultNoiseModel, noise_model_to_dict

@dataclass
class SimulatorConfig:
    """
    Dataclass for storing and quickly instantiating the Simulator class.
    """
    repetitions: int = 1000
    noiseless_repetitions: Optional[int] = None
    oversample: int = 1
    ch_form: bool = False
    preserve_leakage: bool = False
    noise_model: Optional[SqaleLeakageModel] = DefaultNoiseModel()
    noiseless: bool = False
    seed: Optional[int] = None
    progressbar: bool = True

    def to_dict(self) -> dict:
        """
        Returns the simulator config as a dictionary.
        """
        return {
            "repetitions": self.repetitions,
            "noiseless_repetitions": self.noiseless_repetitions,
            "oversample": self.oversample,
            "ch_form": self.ch_form,
            "preserve_leakage": self.preserve_leakage,
            "noise_model": noise_model_to_dict(self.noise_model),
            "noiseless": self.noiseless,
            "seed": self.seed,
            "progressbar": self.progressbar,
        }
