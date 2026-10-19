"""
simulator/simulator.py

Defines the core Simulator class for executing quantum circuits (Qiskit).
Next step (separate change): implement AerSimulator execution and
counts→probability distribution utilities.
"""
# pylint: disable=[C0116]

from __future__ import annotations

import dataclasses

import random
from collections import Counter
from contextlib import contextmanager
from logging import warning
from typing import Iterator, Optional, Callable, Sequence, Union, TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel  # kept for future use with AerSimulator
from tqdm.auto import trange

from simulator.grid_constructor import positions_for_circuit

from .config import SimulatorConfig
from .transport import Geometry
from .noise_markers import strip_noise_markers
from .result_types import SimulationResult
from .utils import bhattacharyya

from .stim_backend import run_stim_backend
from .noise_models import stim_params_from_noise_model

if TYPE_CHECKING:  # pragma: no cover
    from translator.core import TranslationResult

TranslationLike = Union[QuantumCircuit, "TranslationResult"]

class Simulator:
    """
    Quantum circuit simulator (Qiskit).

    Provides a fluent interface to configure simulation parameters.
    Noise is expected to be supplied as a Qiskit Aer `NoiseModel` and will be
    wired in when we add the AerSimulator in the next step.
    """

    _DEFAULT_SHOT_CHUNK = 50_000
    _SHOT_BAR_TARGET_UPDATES = 100

    def __init__(self, config: Optional[SimulatorConfig] = None):
        if config is None:
            config = SimulatorConfig()
        self._repetitions: int = config.repetitions
        self._noiseless: bool = config.noiseless
        self._seed: Optional[int] = config.seed
        self._distance_noise = config.distance_noise
        self._distance_scalar = config.distance_scalar
        self._idle_noise = config.idle_noise
        self._idle_time_scalar = config.idle_time_scalar
        self._noise_model: Optional[NoiseModel] = config.noise_model
        self._stim: bool = config.stim
        self._show_shot_progress: bool = config.show_shot_progress
        self._shot_chunk_size: int = self._DEFAULT_SHOT_CHUNK
        # Opt-in Aer `method`; None keeps `AerSimulator()` and the chunked loop. See
        # `SimulatorConfig.aer_method`.
        self._aer_method: Optional[str] = getattr(config, "aer_method", None)

    # -------- Fluent Configuration Methods --------

    def with_repetitions(self, repetitions: int) -> "Simulator":
        self._repetitions = repetitions
        return self

    def with_noise_model(self, noise_model: NoiseModel | None) -> "Simulator":
        self._noise_model = noise_model
        return self

    def with_seed(self, seed: int) -> "Simulator":
        self._seed = seed
        return self

    def with_noiseless(self, noiseless: bool = True) -> "Simulator":
        self._noiseless = noiseless
        return self

    def with_stim(self, stim: bool = True) -> "Simulator":
        self._stim = stim
        return self

    def with_shot_progress(self, show: bool = True) -> "Simulator":
        self._show_shot_progress = show
        return self

    def with_shot_chunk_size(self, chunk_size: int) -> "Simulator":
        self._shot_chunk_size = max(1, chunk_size)
        return self

    def clone(self) -> "Simulator":
        """
        A field-for-field copy. Copies *every* setting -- the fluent-method version this
        replaces dropped the distance/idle configuration and turned an unseeded simulator
        (fresh entropy per run) into one pinned at seed 0.
        """
        other = Simulator()
        other._repetitions = self._repetitions
        other._noiseless = self._noiseless
        other._seed = self._seed
        other._distance_noise = self._distance_noise
        other._distance_scalar = self._distance_scalar
        other._idle_noise = self._idle_noise
        other._idle_time_scalar = self._idle_time_scalar
        other._noise_model = self._noise_model
        other._stim = self._stim
        other._show_shot_progress = self._show_shot_progress
        other._shot_chunk_size = self._shot_chunk_size
        other._aer_method = self._aer_method
        return other

    # -------------- ( Helpers ) -------------

    def _counts_to_probs(self, counts: dict[str, int]) -> dict[str, float]:
        """Convert counts to normalized probability distribution."""
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    def _key_meta(
        self,
        circuit: QuantumCircuit,
        result: Optional[Result] = None,
    ) -> tuple[list[str], list[int]]:
        """
        Return (key_order, key_widths) from Qiskit's metadata.
        Prefer the simulator result header if provided; otherwise
        fall back to circuit.cregs.
        """
        # --- Prefer post-run metadata (exact output ordering) ---
        if result is not None and result.results:
            header = getattr(result.results[0], "header", None)
            if header and hasattr(header, "creg_sizes"):
                creg_sizes = (
                    header.creg_sizes
                )  # e.g. [['m_parity_0', 2], ['m_data_0', 4]]
                key_order = [name for name, _ in creg_sizes]
                key_widths = [size for _, size in creg_sizes]
                return key_order, key_widths

        # --- Fallback: use circuit registers directly ---
        key_order = [reg.name for reg in circuit.cregs]
        key_widths = [len(reg) for reg in circuit.cregs]
        return key_order, key_widths

    # NOTE: We invert the endianness of the AerSim output from little to big endianness.
    # this is so we can print out bitstrings and the left most character or [0] is the 0th bit.
    def _invert_endianness(self, counts: dict[str, int | float]) -> dict[str, int]:
        """
        Reverse the bit order of each key in `counts`.
        e.g. {'010 11': 5} → {'11 010': 5}

        Works for both counts and probability dicts.
        """
        flipped = {}
        for k, v in counts.items():
            # Reverse each register separately (split by spaces)
            parts = k.split()
            flipped_key = " ".join(p[::-1] for p in parts[::-1])
            flipped[flipped_key] = v
        return flipped

    def _shot_chunk_size_for(self, total: int) -> int:
        """Compute an appropriate chunk size that keeps the shot bar responsive."""
        adaptive_target = max(total // self._SHOT_BAR_TARGET_UPDATES, 1)
        return max(1, min(self._shot_chunk_size, adaptive_target))

    def _chunk_seed_stream(self) -> Iterator[Optional[int]]:
        """
        An endless stream of per-chunk simulator seeds derived from the configured seed.

        Aer's `seed_simulator` seeds a *run*, not a simulator object: calling
        `sim.run(...)` twice on one `AerSimulator(seed_simulator=s)` replays the same
        samples byte for byte. Shots are simulated in chunks so the progress bar can
        move, so a single fixed seed made every chunk after the first a duplicate of
        the first. The aggregate then carried `repetitions / _SHOT_BAR_TARGET_UPDATES`
        distinct samples while the counts summed to `repetitions` -- 1% of the
        requested shots at the default chunking, with nothing in the output to say so.
        Distributions came back quantised to 1/40 for a 4000-shot run, and every
        sampling-error bound computed from the nominal shot count was too tight by 10x.

        Chunk seeds come from a PRNG keyed on the configured seed rather than from
        `seed + index`, because consecutive base seeds would otherwise share almost
        their entire stream and correlate any experiment that sweeps the seed.

        Yields `None` forever when unseeded, which leaves Aer to draw fresh entropy
        per run -- the behaviour callers already relied on to dodge this bug.
        """
        if self._seed is None:
            while True:
                yield None
        rng = random.Random(self._seed)
        while True:
            yield rng.randrange(2**31 - 1)

    def _shot_chunk_generator(self, total: int) -> Iterator[tuple[int, Optional[int]]]:
        """Yield (shots, seed) per chunk for incremental simulation."""
        chunk_size = self._shot_chunk_size_for(total)
        seeds = self._chunk_seed_stream()
        remaining = total
        while remaining > 0:
            current = min(chunk_size, remaining)
            yield current, next(seeds)
            remaining -= current

    @contextmanager
    def _shot_progress_bar(self, desc: str) -> Callable[[int], None] | None:
        """
        Provide a nested tqdm bar that disappears on completion.
        Returns a callback that updates the bar, or None if disabled.
        """
        if (not self._show_shot_progress) or self._repetitions <= 0:
            yield None
            return

        bar = trange(
            self._repetitions,
            desc=desc,
            leave=False,
            unit="shot",
            position=1,
            dynamic_ncols=True,
        )
        try:
            yield bar.update
        finally:
            bar.close()

    # -------- Core Simulation Method --------

    def _stimulate(self, circuit: QuantumCircuit, add_noise: bool = True, layout=None):
        """
        Execute `circuit` and return a (probs, key_order, key_widths) tuple.
        This version uses the Stim backend (quasi-probabilities), replacing Aer sim.run().
        """

        # If there are no classical bits, nothing to count
        tcirc = circuit  # preserve naming from your original
        if len(tcirc.clbits) == 0:
            warning("Did you forget to add measurement registers?")
            return {}, [], []

        # ---- Run via Stim ----
        include_noise = add_noise and (not self._noiseless)

        # `replace`, not attribute assignment. `stim_params_from_noise_model` hands back
        # the object hanging off the noise model, so assigning here mutated state shared
        # with every other simulator holding that model.
        stim_params = stim_params_from_noise_model(self._noise_model)
        stim_params = dataclasses.replace(
            stim_params,
            idle_time_scalar=self._idle_time_scalar if self._idle_noise else 0.0,
        )

        with self._shot_progress_bar("Shots") as progress_update:
            probs = run_stim_backend(
                circuit=tcirc,
                shots=self._repetitions,
                seed=self._seed,
                include_noise=include_noise,
                noise_params=stim_params,
                shots_per_chunk=self._shot_chunk_size_for(self._repetitions),
                progress_callback=progress_update,
                layout=layout,
                distance_scalar=self._distance_scalar if layout is not None else None,
            )
        # `run_stim_backend` already emits the post-inversion convention Aer reaches
        # via `_invert_endianness`: registers in declaration order, position i = bit i.
        # No further reordering here, and in particular none conditioned on
        # `add_noise` -- this branch used to invert only the ideal call, which
        # compensated for a reversal in `_format_key_by_cregs` on the one code path
        # (the unencoded reference circuit) where a single creg made the register
        # reordering a no-op, and left the encoded noisy distribution reversed.
        key_order, key_widths = self._key_meta(tcirc)

        return probs, key_order, key_widths


    def _simulate(self, circuit: QuantumCircuit, add_noise: bool = True):
        """
        Execute `circuit` with Qiskit's AerSimulator and return a counts dict.
        Uses class config: repetitions (shots), seed, noiseless, and noise_model.
        """

        if self._distance_noise and add_noise:
            if not self._stim:
                # Aer never had a transport model, only a per-physical-gate distance
                # charge that priced one logical cross-patch CNOT as four round trips.
                # That is gone rather than ported: geometry now lives in grouped
                # transport events, which need the layered schedule in
                # `simulator.scheduler`, and Stim is what this project simulates on.
                # Failing loudly beats two backends quietly modelling different machines.
                raise NotImplementedError(
                    "Distance-dependent noise is Stim-only. Set `stim=True`, or turn "
                    "distance noise off with `with_distance_noise(distance_noise=False)`."
                )
            # Build the layout with the geometry the run will actually use -- gutter and
            # pitch come from the noise parameters, not the module defaults, so a swept
            # `patch_gutter` moves the atoms it claims to move.
            params = stim_params_from_noise_model(self._noise_model)
            geometry = Geometry(
                pitch=params.atom_pitch,
                blockade_radius=params.blockade_radius,
                movement_speed=params.movement_speed,
                gutter=int(params.patch_gutter),
            )
            return self._stimulate(
                circuit,
                add_noise=add_noise,
                layout=positions_for_circuit(circuit, geometry=geometry),
            )

        if self._stim:
            return self._stimulate(circuit, add_noise=add_noise)

        # Choose simulator backend. The seed is supplied per `run` below rather than
        # here -- see `_chunk_seed_stream` for why a constructor seed silently
        # collapsed a chunked run to one chunk's worth of samples.
        method = {} if self._aer_method is None else {"method": self._aer_method}
        if (not add_noise) or (self._noise_model is None) or self._noiseless:
            sim = AerSimulator(**method)
        else:
            sim = AerSimulator(noise_model=self._noise_model, **method)

        # Transpile for the chosen simulator (ensures basis/optimization compatible)
        # tcirc = transpile(circuit, sim, optimization_level=0, seed_transpiler=self._seed)
        tcirc = strip_noise_markers(circuit)

        # If there are no classical bits, nothing to count
        if len(tcirc.clbits) == 0:
            warning("Did you forget to add measurement registers?")
            return {}, [], []

        # Run and return raw counts
        aggregated_counts: Counter[str] = Counter()
        first_result: Result | None = None

        with self._shot_progress_bar("Shots") as progress_update:
            # One run when an Aer method is pinned: a density matrix is evolved once per
            # `run`, so chunking would redo it ~100 times. Default: the chunked loop.
            chunks = (
                self._shot_chunk_generator(self._repetitions)
                if self._aer_method is None
                else [(self._repetitions, next(self._chunk_seed_stream()))]
            )
            for chunk, chunk_seed in chunks:
                job = sim.run(tcirc, shots=chunk, seed_simulator=chunk_seed)
                result = job.result()
                if first_result is None:
                    first_result = result

                # <+======================================================================================+>*
                # NOTE: EXTREMELY important Note. AerSim returns are little endian.                         |   HEY READ THE NOTE!!!
                #  This also includes register order.                                                       |             ,\
                #  We reverse the bitstring here so we can directly read                                    |             \\\,_
                #  bit strings into their correct register from L to R                                      |              \` ,\
                # For example:                                                                              |         __,.-" =__)
                #   bits B -> L 0101 1111 00 01 != ['m_parity_0', 'm_parity_1', 'm_data_0', 'm_data_1']     |       ."        )
                #        L -> B 10 00 1111 1010 == ['m_parity_0', 'm_parity_1', 'm_data_0', 'm_data_1']     |    ,_/   ,    \/\_
                # NOTE: In summary, we invert bit strings to match registers                                |    \_|    )_-\ \_-`
                #       and so we can print(bitstring[i]) to print the actual ith bit                       |       `-----` `--`
                #       i.e. 0,1,1,0,1,0,1 -> bits 0,1,2,3,4,5,6                                            |
                # <+======================================================================================+>*
                little_endian_counts = result.get_counts(
                    0
                )  # first (and typically only) circuit
                counts = self._invert_endianness(little_endian_counts)
                aggregated_counts.update(counts)

                if progress_update is not None:
                    progress_update(chunk)

        key_order, key_widths = self._key_meta(tcirc, first_result)

        # counts: Dict[str, int], keys are space-delimited per creg in qc.cregs order
        probs = self._counts_to_probs(dict(aggregated_counts))
        return probs, key_order, key_widths

    # ---------------------- Public APIs -----------------------

    def run(self, circuit: QuantumCircuit) -> SimulationResult:
        """
        Simulate ideal and noisy distributions and compute Bhattacharyya fidelity.
        """
        ideal_dist, _, _ = self._simulate(circuit, add_noise=False)
        noisy_dist, key_order, key_widths = self._simulate(circuit, add_noise=True)

        raw_fidelity = bhattacharyya(ideal_dist, noisy_dist)

        return SimulationResult(
            raw_fidelity=raw_fidelity,
            ideal_dist=ideal_dist,
            noisy_dist=noisy_dist,
            key_order=key_order,
            key_widths=key_widths,
        )

    def noisy(self, circuit: QuantumCircuit) -> SimulationResult:
        """
        Simulate only the noisy distribution.
        """
        noisy_dist, key_order, key_widths = self._simulate(circuit, add_noise=True)

        return SimulationResult(
            noisy_dist=noisy_dist,
            key_order=key_order,
            key_widths=key_widths,
        )

    def ideal(self, circuit: QuantumCircuit) -> SimulationResult:
        """
        Simulate only the ideal (noiseless) distribution.
        """
        ideal_dist, key_order, key_widths = self._simulate(circuit, add_noise=False)

        return SimulationResult(
            ideal_dist=ideal_dist,
            key_order=key_order,
            key_widths=key_widths,
        )


def simulate_batch(
    program_circuit: QuantumCircuit,
    translations: Sequence[TranslationLike],
    sim_config: SimulatorConfig,
    *,
    desc: str = "Simulating circuits",
) -> tuple[SimulationResult, list[SimulationResult]]:
    """
    Run the ideal simulation for the logical program circuit and noisy simulations
    for each translated circuit in `translations`.

    Returns a tuple of (ideal_simulation_result, noisy_results_list).
    """
    simulator = Simulator(sim_config)
    ideal_sim_res = simulator.ideal(circuit=program_circuit)

    simulation_results: list[SimulationResult] = []
    for idx in trange(len(translations), desc=desc, unit=" sim"):
        translation = translations[idx]
        circuit = translation.circuit if hasattr(translation, "circuit") else translation
        simulation_results.append(simulator.noisy(circuit))

    return ideal_sim_res, simulation_results
