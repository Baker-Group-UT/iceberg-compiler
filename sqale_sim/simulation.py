from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import cirq
import experiments_superstaq as ess

from fidelity import bhattacharyya
from postselection import post_select_no_leak

# silence tqdm globally for Superstaq helpers
os.environ.setdefault("TQDM_DISABLE", "1")

__all__ = ["SqaleSimulationResult", "run_sqale_simulation"]


@dataclass
class SqaleSimulationResult:
    fidelity: float
    ideal_dist: Dict[str, float]
    noisy_dist: Dict[str, float]
    post_selected_dist: Dict[str, float]


def run_sqale_simulation(
    circuit: cirq.Circuit,
    *,
    repetitions: int = 1_000,
    oversample: int = 1,
    ch_form: bool = False,
    post_select: bool = True,
    noise_model: Optional[
        ess.simulation.sqale_leakage_model.SqaleLeakageModel
    ] = None,
    noiseless: bool = False,
) -> SqaleSimulationResult:
    """
    Simulate *circuit* with Superstaq’s SQALE leakage model and compare to the
    ideal 2-level outcome.
    """
    # -- 0. ideal reference --------------------------------------------------
    ideal_dist = ess.simulation.leakage_sim.simulate_true_distribution(
        circuit, dimension=2
    )

    # -- 1. attach / build noise model --------------------------------------
    if noise_model is None:
        params = ess.simulation.sqale_leakage_model.STOCHASTIC_PARAMS
        noise_model = ess.simulation.sqale_leakage_model.SqaleLeakageModel(
            **params.model_dump()
        )

    noisy_circuit = circuit if noiseless else circuit.with_noise(noise_model)

    # -- 2. Monte-Carlo ------------------------------------------------------
    noisy_dist = ess.simulation.leakage_sim.estimate_distribution(
        noisy_circuit,
        repetitions=repetitions,
        oversample=oversample,
        ch_form=ch_form,
    )

    # -- 3. optional post-selection -----------------------------------------
    post_selected_dist = (
        post_select_no_leak(noisy_dist) if post_select else noisy_dist
    )
    
    # -- 4. compare ----------------------------------------------------------
    fidelity = bhattacharyya(ideal_dist, post_selected_dist)

    return SqaleSimulationResult(
        fidelity=fidelity,
        ideal_dist=ideal_dist,
        noisy_dist=noisy_dist,
        post_selected_dist=post_selected_dist,
    )