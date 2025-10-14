"""
simulator/simulator.py

Defines the core Simulator class for executing quantum circuits with leakage-aware noise models.
Follows a fluent interface pattern to configure simulation parameters.
"""
# pylint: disable=[C0116]

from __future__ import annotations

import time
from typing import Optional, Union
import cirq
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from experiments_superstaq.simulation.sqale_leakage_model import (
    SqaleLeakageModel,
    HilbertNoiseParams,
)
from experiments_superstaq.simulation.leakage_sim import estimate_distribution
from iceberg.iceberg_postselect import ordered_measurement_keys
from .config import SimulatorConfig
from .utils import remove_leakage_states, bhattacharyya, remove_barriers
from .result_types import SimulationResult

NoiseLike = Union[cirq.NoiseModel, cirq.NOISE_MODEL_LIKE, SqaleLeakageModel, HilbertNoiseParams, None]

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
        self._oversample = config.oversample
        self._ch_form = config.ch_form
        self._preserve_leakage = config.preserve_leakage
        self._noiseless = config.noiseless
        self._seed = config.seed
        self._progressbar = config.progressbar
        self._sqale = config.sqale
        self._noise_model: NoiseLike = config.noise_model

    # -------- Fluent Configuration Methods --------

    def with_repetitions(self, repetitions: int) -> Simulator:
        self._repetitions = repetitions
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

    def with_noise_model(self, noise_model: NoiseLike) -> Simulator:
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
    
    def with_sqale(self, sqale: bool = False) -> Simulator:
        self._sqale = sqale
        return self

    def with_noiseless(self, noiseless: bool = True) -> Simulator:
        self._noiseless = noiseless
        return self

    def clone(self) -> Simulator:
        return Simulator()\
            .with_repetitions(self._repetitions)\
            .with_oversample(self._oversample)\
            .with_ch_form(self._ch_form)\
            .with_leakage_preserved(self._preserve_leakage)\
            .with_noise_model(self._noise_model)\
            .with_seed(self._seed)\
            .with_progressbar(self._progressbar)\
            .with_noiseless(self._noiseless)\
            .with_sqale(self._sqale)\

    # -------- Core Simulation Method --------

    def _use_sqale(self) -> bool:
        # SQALE if  enabled OR noise_model is SQALE
        return bool(self._sqale or isinstance(self._noise_model, SqaleLeakageModel))

    def _prep_for_sim(self, circuit: cirq.Circuit) -> cirq.Circuit:
        clean = remove_barriers(circuit=circuit)
        return clean

    def _add_noise(self, circuit: cirq.Circuit) -> cirq.Circuit:
        if self._noiseless or self._use_sqale():
            return circuit
        return circuit.with_noise(self._noise_model)

    def _simulate(self, circuit: cirq.Circuit, track_stats:bool = False) -> dict(str, float):
        if self._use_sqale():
            dist_raw = estimate_distribution(
                circuit,
                repetitions=self._repetitions,
                oversample=self._oversample,
                ch_form=self._ch_form,
                seed=self._seed,
                progressbar=self._progressbar,
            )
            result = remove_leakage_states(dist_raw, track_stats=track_stats) # specific to sqale, removes 2s
            
            if track_stats:
                dist_clean, kept, dropped = result
            else:
                dist_clean = result
                kept, dropped = None, None            
            
            return dist_clean, dist_raw, kept, dropped, None, None

        dist_raw, key_order, key_widths = self._simulate_with_cirq(circuit)

        # Return both distributions + stats + layout via attributes on self (or return tuple and use locally)
        return dist_raw, dist_raw, None, None, key_order, None

    def run(self, circuit: cirq.Circuit) -> SimulationResult:
        """
        Simulates the given Cirq circuit with the current configuration.

        Returns:
            SqaleSimulationResult: object containing post-leakage distributions and raw fidelity.
        """
        clean_circuit = remove_barriers(circuit=circuit)

        ideal_dist, ideal_raw_dist, _, _, key_order, key_widths = self._simulate(circuit=clean_circuit, track_stats=False)
        noisy_circuit = self._add_noise(circuit=clean_circuit)
        noisy_dist, noisy_raw_dist, kept, dropped, key_order, key_widths = self._simulate(circuit=noisy_circuit, track_stats=True)

        raw_fidelity = bhattacharyya(ideal_raw_dist, noisy_raw_dist)

        return SimulationResult(
            raw_fidelity=raw_fidelity,
            ideal_dist=ideal_dist,
            noisy_dist=noisy_dist,
            ideal_raw_dist=ideal_raw_dist,
            noisy_raw_dist=noisy_raw_dist,
            percent_shots_kept=kept,
            percent_shots_dropped=dropped,
            # TODO: update key stuff
        )

    def noisy(self, circuit: cirq.Circuit) -> SimulationResult:
        """
        Run a simulation without comparing it to the ideal
        """
        clean_circuit = self._prep_for_sim(circuit)
        noisy_circuit = self._add_noise(circuit=clean_circuit)
        noisy_dist, noisy_raw_dist, kept, dropped, key_order, key_widths = self._simulate(circuit=noisy_circuit, track_stats=True)
        return SimulationResult(
            noisy_dist=noisy_dist,
            noisy_raw_dist=noisy_raw_dist,
            percent_shots_kept=kept,
            percent_shots_dropped=dropped,
            key_order=key_order if key_order is not None else ordered_measurement_keys(circuit=circuit),
        )

    def ideal(self, circuit: cirq.Circuit) -> SimulationResult:
        """
        Run an ideal simulation without comparing it to noisy.
        """
        clean_circuit = remove_barriers(circuit=circuit)
        ideal_dist, ideal_raw_dist, _, _, key_order, key_widths = self._simulate(circuit=clean_circuit, track_stats=False)

        reordered_ideal_dist = self._normalize_bitstrings_by_key_order(ideal_dist, key_order)
        return SimulationResult(
            ideal_dist=reordered_ideal_dist,
            ideal_raw_dist=reordered_ideal_dist,
            key_order=key_order if key_order is not None else ordered_measurement_keys(circuit=circuit),
            # key_widths=None,
         )

    # FIXME: This function is extremely scuffed and needs to be updated to a qiskit aer convention in the future.
    def _simulate_with_cirq(self, circuit: cirq.Circuit) -> tuple[dict[str, float], list[str], None]:
        """
        Cirq-based sampler that:
        • synchronizes to a single terminal measurement moment,
        • orders measurement keys alphanumerically from that final moment,
        • trades the *bits* among data_* registers by reversing their segment order,
        • concatenates per-key bit rows in that (unchanged) sorted key_order (no spaces),
        • returns {bitstring -> probability}, the sorted key_order, and None for widths.
        """
        # 1) Keep measurements aligned (but noise may remain after them)
        circuit = cirq.synchronize_terminal_measurements(circuit)

        # 2) Backend
        simulator = (cirq.Simulator(seed=self._seed)
                    if self._noiseless else cirq.CliffordSimulator(seed=self._seed))
        total_reps = max(1, int(self._repetitions) * int(self._oversample))

        if len(circuit) == 0:
            return {"": 1.0}, [], None

        # 3) Find the last moment that actually contains MeasurementGate ops
        last_meas_idx = None
        for mi in range(len(circuit) - 1, -1, -1):
            if any(isinstance(getattr(op, "gate", None), cirq.MeasurementGate) for op in circuit[mi]):
                last_meas_idx = mi
                break

        if last_meas_idx is None:
            qubits = sorted(circuit.all_qubits())
            if not qubits:
                return {"": 1.0}, [], None
            circuit = circuit + cirq.Moment([cirq.measure(q, key=str(q)) for q in qubits])
            last_meas_idx = len(circuit) - 1

        # 4) Take keys from that moment only, alphanumeric order
        meas_ops = [op for op in circuit[last_meas_idx]
                    if isinstance(getattr(op, "gate", None), cirq.MeasurementGate)]
        key_order = sorted([cirq.measurement_key_name(op) for op in meas_ops], key=str)

        # 5) Run shots
        start = time.time()
        trial = simulator.run(circuit, repetitions=total_reps)
        end = time.time()
        print("cirq time: ", end - start)


        # NOTE: swaps the bits in data registers, for example data_0 -> data_2, data_2 -> data_0
        data_keys = [k for k in key_order if k.startswith("data_")]
        counts: dict[str, int] = {}
        for r in range(total_reps):
            row_bits = {k: "".join(str(int(b)) for b in trial.measurements[k][r]) for k in key_order}
            if len(data_keys) > 1:
                rev = list(reversed([row_bits[k] for k in data_keys]))
                for i, k in enumerate(data_keys):
                    row_bits[k] = rev[i]
            bitstring = "".join(row_bits[k] for k in key_order)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        dist_raw = {s: c / float(total_reps) for s, c in counts.items()}
        return dist_raw, key_order, None

    def _normalize_bitstrings_by_key_order(
        self, 
        dist: dict[str, float],
        key_order: list[str]
    ) -> dict[str, float]:
        """
        Reorders bitstrings in `dist` so that registers are concatenated in
        alphanumeric-sorted order of `key_order`.
        Example:
            key_order = ['q(2)', 'q(1)', 'q(0)']
            sorted -> ['q(0)', 'q(1)', 'q(2)']
        """
        if not key_order:
            return dist

        # Sort the keys alphanumerically
        sorted_keys = sorted(key_order, key=lambda k: str(k))

        # Compute segment lengths per key from one example string
        example = next(iter(dist.keys()))
        lengths: dict[str, int] = {}
        idx = 0
        for k in key_order:
            # Grab length from the first distribution entry (split according to original order)
            seg_len = len(example[idx: idx + 1]) if example else 1
            # safer: infer from original trial result—but here assume consistent length
            lengths[k] = len(example[idx: idx + 1])
            idx += lengths[k]

        new_dist: dict[str, float] = {}
        for bits, prob in dist.items():
            # Break old string into segments by original key_order
            segments: dict[str, str] = {}
            idx = 0
            for k in key_order:
                seg_len = lengths[k]
                segments[k] = bits[idx: idx + seg_len]
                idx += seg_len
            # Rebuild bitstring in sorted key order
            reordered = "".join(segments[k] for k in sorted_keys)
            new_dist[reordered] = new_dist.get(reordered, 0.0) + prob

        return new_dist
