"""
simulator/simulator.py

Defines the core Simulator class for executing quantum circuits with leakage-aware noise models.
Follows a fluent interface pattern to configure simulation parameters.
"""
# pylint: disable=[C0116]

from __future__ import annotations

from typing import Optional, Union
import cirq
from experiments_superstaq.simulation.sqale_leakage_model import (
    SqaleLeakageModel,
    HilbertNoiseParams,
)
from experiments_superstaq.simulation.leakage_sim import estimate_distribution

from .config import SimulatorConfig
from .utils import remove_leakage_states, bhattacharyya, remove_barriers
from .result_types import SqaleSimulationResult
from .transpile import transpile_for_sqale

class Simulator:
    """
    Leakage-aware quantum circuit simulator.

    Provides a fluent interface to configure simulation parameters and run circuits under
    a SQALE noise model. Leakage post-selection is applied by default.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        if config is None:
            config = SimulatorConfig()
        self._repetitions = config.repetitions
        self._noiseless_repetitions = config.noiseless_repetitions
        self._oversample = config.oversample
        self._ch_form = config.ch_form
        self._preserve_leakage = config.preserve_leakage
        self._noise_model = config.noise_model
        self._noiseless = config.noiseless
        self._seed = config.seed
        self._progressbar = config.progressbar
        
        self._compile_with_superstaq = config.compile_with_superstaq
        self._compile_target = config.compile_target
        self._compile_enforce_canonical = config.compile_enforce_canonical
        self._allow_global_rotations = config.allow_global_rotations
        self._compile_service = config.compile_service

    # -------- Fluent Configuration Methods --------

    def with_repetitions(self, repetitions: int) -> Simulator:
        self._repetitions = repetitions
        return self

    def with_noiseless_repetitions(self, noiseless_repetitions: int) -> Simulator:
        self._noiseless_repetitions = noiseless_repetitions
        return self

    def with_oversample(self, oversample: int) -> Simulator:
        self._oversample = oversample
        return self

    def with_ch_form(self, ch_form: bool = True) -> Simulator:
        self._ch_form = ch_form
        return self

    def with_leakage_preserved(self, preserve: bool = True) -> Simulator:
        self._preserve_leakage = preserve
        return self

    def with_noise_model(self,
            noise_model: Union[SqaleLeakageModel, HilbertNoiseParams]) -> Simulator:
        if isinstance(noise_model, HilbertNoiseParams):
            self._noise_model = SqaleLeakageModel(**noise_model.model_dump())
        else:
            self._noise_model = noise_model
        return self

    def with_seed(self, seed: int) -> Simulator:
        self._seed = seed
        return self

    def with_progressbar(self, enabled: bool = True) -> Simulator:
        self._progressbar = enabled
        return self

    def with_noiseless(self, noiseless: bool = True) -> Simulator:
        self._noiseless = noiseless
        return self

    def clone(self) -> Simulator:
        return Simulator()\
            .with_repetitions(self._repetitions)\
            .with_noiseless_repetitions(self._noiseless_repetitions or self._repetitions)\
            .with_oversample(self._oversample)\
            .with_ch_form(self._ch_form)\
            .with_leakage_preserved(self._preserve_leakage)\
            .with_noise_model(self._noise_model)\
            .with_seed(self._seed)\
            .with_progressbar(self._progressbar)\
            .with_noiseless(self._noiseless)\
            .with_compile_with_superstaq(self._compile_with_superstaq)\
            .with_compile_target(self._compile_target)\
            .with_compile_enforce_canonical(self._compile_enforce_canonical)\
            .with_allow_global_rotations(self._allow_global_rotations)\
            .with_compile_service(self._compile_service)

    def with_compile_target(self, target: str) -> Simulator:
        self._compile_target = target
        return self

    def with_compile_with_superstaq(self, enable: bool = True) -> Simulator:
        self._compile_with_superstaq = enable
        return self

    def with_compile_enforce_canonical(self, enable: bool = True) -> Simulator:
        self._compile_enforce_canonical = enable
        return self

    def with_allow_global_rotations(self, allow: bool = True) -> Simulator:
        self._allow_global_rotations = allow
        return self

    def with_compile_service(self, service) -> Simulator:
        self._compile_service = service
        return self

    # -------- Core Simulation Method --------

    def _prep_for_sim(self, circuit: cirq.Circuit) -> cirq.Circuit:
        clean = remove_barriers(circuit=circuit)
        transpiled = transpile_for_sqale(
            clean,
            target=self._compile_target,
            use_superstaq=self._compile_with_superstaq,
            enforce_canonical=self._compile_enforce_canonical,
            allow_global=self._allow_global_rotations,
            service=self._compile_service,
        )
        return transpiled

    def run(self, circuit: cirq.Circuit) -> SqaleSimulationResult:
        """
        Simulates the given Cirq circuit with the current configuration.

        Returns:
            SqaleSimulationResult: object containing post-leakage distributions and raw fidelity.
        """
        clean_circuit = remove_barriers(circuit=circuit)

        ideal_raw_dist = estimate_distribution(
            clean_circuit,
            repetitions=self._noiseless_repetitions or self._repetitions,
            oversample=self._oversample,
            ch_form=self._ch_form,
            seed=self._seed,
            progressbar=self._progressbar,
        )

        noise_model = self._noise_model
        noisy_circuit = clean_circuit if self._noiseless else clean_circuit.with_noise(noise_model)

        noisy_raw_dist = estimate_distribution(
            noisy_circuit,
            repetitions=self._repetitions,
            oversample=self._oversample,
            ch_form=self._ch_form,
            seed=self._seed,
            progressbar=self._progressbar,
        )

        noisy_dist, percent_shots_kept, percent_shots_dropped = remove_leakage_states(
            noisy_raw_dist, track_stats=True)
        ideal_dist = remove_leakage_states(ideal_raw_dist)

        raw_fidelity = bhattacharyya(ideal_raw_dist, noisy_raw_dist)

        return SqaleSimulationResult(
            raw_fidelity=raw_fidelity,
            ideal_dist=ideal_dist,
            noisy_dist=noisy_dist,
            ideal_raw_dist=ideal_raw_dist,
            noisy_raw_dist=noisy_raw_dist,
            percent_shots_kept=percent_shots_kept,
            percent_shots_dropped=percent_shots_dropped
        )

    def noisy_without_comparison(self, circuit: cirq.Circuit) -> SqaleSimulationResult:
        """
        Run a simulation without comparing it to the ideal
        """
        base = self._prep_for_sim(circuit)
        noise_model = self._noise_model
        noisy_circuit = base if self._noiseless else base.with_noise(noise_model)
        noisy_raw_dist = estimate_distribution(
            noisy_circuit,
            repetitions=self._repetitions,
            oversample=self._oversample,
            ch_form=self._ch_form,
            seed=self._seed,
            progressbar=self._progressbar,
        )
        noisy_dist, kept, dropped = remove_leakage_states(noisy_raw_dist, track_stats=True)
        return SqaleSimulationResult(
            raw_fidelity=None,
            ideal_dist=None,
            noisy_dist=noisy_dist,
            ideal_raw_dist=None,
            noisy_raw_dist=noisy_raw_dist,
            percent_shots_kept=kept,
            percent_shots_dropped=dropped,
        )
    
    def ideal_without_comparison(self, circuit: cirq.Circuit) -> SqaleSimulationResult:
        """
        Run an ideal simulation without comparing it to noisy.
        """
        clean_circuit = remove_barriers(circuit=circuit)
        print(clean_circuit)
        ideal_raw_dist = estimate_distribution(
            clean_circuit,
            repetitions=self._noiseless_repetitions or self._repetitions,
            oversample=self._oversample,
            ch_form=self._ch_form,
            seed=self._seed,
            progressbar=self._progressbar,
        )


        ideal_dist = remove_leakage_states(ideal_raw_dist)

        return SqaleSimulationResult(
            raw_fidelity=None,
            ideal_dist=ideal_dist,
            noisy_dist=None,
            ideal_raw_dist=ideal_raw_dist,
            noisy_raw_dist=None,
            percent_shots_kept=None,
            percent_shots_dropped=None
        )
