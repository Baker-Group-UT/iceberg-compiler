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
    ScaledNoiseModel,
)


@dataclass
class SimulatorConfig:
    """
    Dataclass for storing and quickly instantiating the Simulator class.
    """
    repetitions: int = 1000
    # Annotated so these are real dataclass fields. Without annotations they were class
    # attributes: the constructor rejected them as keywords, and `dataclasses.replace`
    # or `asdict` silently dropped them.
    distance_noise: bool = False
    distance_scalar: float = 1.0
    idle_noise: bool = False
    idle_time_scalar: float = 1.0
    noise_model: Optional[NoiseModel] = None
    noiseless: bool = False
    seed: Optional[int] = None
    stim: bool = True
    #: Per-shot tqdm bar. Set False for any run fanned across processes: workers inherit
    #: one stdout, so their bar updates serialise on a single pipe and the workers block
    #: on writes instead of computing. Measured at 12 workers, 148 KB of log per work item
    #: against 15 KB throttled, and 2.6x the throughput. `Simulator.with_shot_progress`
    #: sets the same flag fluently; this exists so it survives a `SimulatorConfig` round
    #: trip, which is the path `pipeline.tasks.simulate_circuit` actually takes.
    show_shot_progress: bool = True
    #: Aer backend only, and opt-in: the `method` handed to `AerSimulator` (for example
    #: "density_matrix"). None -- the default -- builds `AerSimulator()` exactly as before
    #: and keeps the chunked shot loop. When set, all shots run in one `run` call, because
    #: a density-matrix evolution is paid once per call and sampling from it is free; the
    #: chunked loop would re-evolve it ~100 times. Added for the v2 revision's exact
    #: small-circuit validation (experiments/revision/exp3_proxy). Ignored by Stim.
    aer_method: Optional[str] = None

    def __post_init__(self):
        # Default to a realistic noise model if none specified and not noiseless
        if self.noise_model is None and not self.noiseless:
            # alpha=1 is the NEUTRAL_ATOM baseline -- see noise_models.ScaledNoiseModel.
            self.noise_model = ScaledNoiseModel(alpha=1.0)

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

