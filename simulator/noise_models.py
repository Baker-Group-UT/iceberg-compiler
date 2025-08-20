"""
simulator/noise_models.py

Predefined leakage-aware noise models for use with the Simulator.
"""
# pylint: disable=[C0103]

from experiments_superstaq.simulation.sqale_leakage_model import (
    SqaleLeakageModel,
    DEFAULT_PARAMS,
    STOCHASTIC_PARAMS,
)

def noise_model_to_dict(model) -> dict:
    """
    To dict function for the SqaleLeakageModel class
    """
    if hasattr(model, "model_dump"):  # Comes from Pydantic
        return model.model_dump()
    elif hasattr(model, "to_dict"):
        return model.to_dict()
    else:
        return {"__str__": str(model)}

def DefaultNoiseModel() -> SqaleLeakageModel:
    """
    Returns the default calibrated SQALE leakage noise model.

    Uses realistic gate-level parameters extracted from hardware calibration.
    """
    params = DEFAULT_PARAMS.model_copy()
    return SqaleLeakageModel(**params.model_dump())


def StochasticNoiseModel() -> SqaleLeakageModel:
    """
    Returns the stochastic variant of the SQALE leakage noise model.

    Typically used for randomized benchmarking or quick sweeps.
    """
    params = STOCHASTIC_PARAMS.model_copy()
    return SqaleLeakageModel(**params.model_dump())
