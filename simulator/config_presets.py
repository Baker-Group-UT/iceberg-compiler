"""
simulator.config_presets.py - Quick presets for the Simulator class.

**Every preset installs `ScaledNoiseModel`, whose alpha=1 baseline is `NEUTRAL_ATOM`.**
Two drifts to know about, both deliberate:

* Presets used to install `DefaultNoiseModel`, a hybrid -- IBM-Pittsburgh rates and times
  with the paper's T1/T2 -- which did not match the paper's Table 2. The same encoded
  circuit measured LSR 0.827 under that hybrid against 0.590 under Table 2, so which
  preset a script picked decided its headline number.
* `ScaledNoiseModel`'s alpha=1 then moved again, from Table 2 (superconducting) to
  `NEUTRAL_ATOM`, when the transport model landed -- this compiler targets a machine
  that can move atoms. To reproduce a Table 2 number, pass
  `StimNoiseParameters.paper_table_2()`-derived values explicitly.

`alpha` scales every error rate together -- p1, p2, readout, 1/T1, 1/T2, and
`move_error` -- so alpha=1.0 is the hardware as `NEUTRAL_ATOM` describes it and the
calibration in `experiments/shared/simulation.py` moves it from there.
"""
# pylint: disable=[C0103, C0116]

from .config import SimulatorConfig
from .noise_models import StochasticNoiseModel, ScaledNoiseModel


def TestSimulator(noiseless: bool = False, seed: int | None = None):
    return SimulatorConfig(
        repetitions=1,
        noiseless=noiseless,
        noise_model=ScaledNoiseModel(alpha=1.0),
        seed=seed,
    )

def QuickSimulator(seed: int | None = None):
    return SimulatorConfig(
        repetitions=100, noise_model=ScaledNoiseModel(alpha=1.0), seed=seed
    )

def StandardSimulator(repetitions: int = 1000, seed: int | None = None):
    return SimulatorConfig(
        repetitions=repetitions, noise_model=ScaledNoiseModel(alpha=1.0), seed=seed
    )

def ScaledSimulator(repetitions: int = 1000, seed: int | None = None, alpha: float = 1.0):
    return SimulatorConfig(
        repetitions=repetitions,
        noise_model=ScaledNoiseModel(alpha=alpha),
        seed=seed,
    )

def FinalizedSimulator(seed: int | None = None):
    return SimulatorConfig(
        repetitions=200_000, noise_model=ScaledNoiseModel(alpha=1.0), seed=seed
    )

def StochasticSimulator(seed: int | None = None):
    return SimulatorConfig(
        repetitions=1000,
        noise_model=StochasticNoiseModel(),
        noiseless=False,
        seed=seed,
    )
