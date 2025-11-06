"""
simulator/simulator.py

Defines the core Simulator class for executing quantum circuits (Qiskit).
Next step (separate change): implement AerSimulator execution and
counts→probability distribution utilities.
"""
# pylint: disable=[C0116]

from __future__ import annotations

from logging import warning
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit_aer.noise import NoiseModel  # kept for future use with AerSimulator
from qiskit_aer import AerSimulator

from .config import SimulatorConfig
from .utils import bhattacharyya
from .result_types import SimulationResult

from .stim_backend import run_stim_backend

class Simulator:
    """
    Quantum circuit simulator (Qiskit).

    Provides a fluent interface to configure simulation parameters.
    Noise is expected to be supplied as a Qiskit Aer `NoiseModel` and will be
    wired in when we add the AerSimulator in the next step.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        if config is None:
            config = SimulatorConfig()
        self._repetitions: int = config.repetitions
        self._noiseless: bool = config.noiseless
        self._seed: Optional[int] = config.seed
        self._noise_model: Optional[NoiseModel] = config.noise_model
        self._stim: bool = config.stim

    # -------- Fluent Configuration Methods --------

    def with_repetitions(self, repetitions: int) -> "Simulator":
        self._repetitions = repetitions
        return self

    def with_noise_model(self, noise_model: NoiseModel) -> "Simulator":
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

    def clone(self) -> "Simulator":
        return (
            Simulator()
            .with_repetitions(self._repetitions)
            .with_noise_model(self._noise_model)
            .with_seed(self._seed if self._seed is not None else 0)
            .with_noiseless(self._noiseless)
            .with_stim(self._stim)
        )

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
                creg_sizes = header.creg_sizes  # e.g. [['m_parity_0', 2], ['m_data_0', 4]]
                key_order = [name for name, _ in creg_sizes]
                key_widths = [size for _, size in creg_sizes]
                return key_order, key_widths

        # --- Fallback: use circuit registers directly ---
        key_order = [reg.name for reg in circuit.cregs]
        key_widths = [len(reg) for reg in circuit.cregs]
        return key_order, key_widths

    # NOTE: We invert the endianness of the AerSim output from little to big endianness.
    # this is so we can print out bitstrings and the left most character or [0] is the 0th bit.
    def _invert_endianness(self, counts: dict[str, int | float]) -> dict[str, int | float]:
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

    # -------- Core Simulation Method --------

    def _stimulate(self, circuit: QuantumCircuit, add_noise: bool = True):
        """
        Execute `circuit` and return a (probs, key_order, key_widths) tuple.
        This version uses the Stim backend (quasi-probabilities), replacing Aer sim.run().
        """

        # If there are no classical bits, nothing to count
        tcirc = circuit  # preserve naming from your original
        if len(tcirc.clbits) == 0:
            warning("Did you forget to add measurement registers?")
            return {}

        # ---- Run via Stim ----
        # NOTE: This returns quasi-probabilities directly (normalized to sum ~1.0).
        #       It also returns bitstrings in conventional printed order already,
        #       so DO NOT call _invert_endianness here.
        probs = run_stim_backend(
            circuit=tcirc,
            shots=self._repetitions,
            seed=self._seed,
        )

        key_order = [creg.name for creg in tcirc.cregs]
        key_widths = [creg.size for creg in tcirc.cregs]

        return probs, key_order, key_widths


    def _simulate(self, circuit: QuantumCircuit, add_noise: bool = True):
        """
        Execute `circuit` with Qiskit's AerSimulator and return a counts dict.
        Uses class config: repetitions (shots), seed, noiseless, and noise_model.
        """

        if self._stim:
            return self._stimulate(circuit, add_noise=add_noise)

        # Choose simulator backend
        if (not add_noise) or (self._noise_model is None) or self._noiseless:
            sim = AerSimulator(seed_simulator=self._seed)
        else:
            sim = AerSimulator(noise_model=self._noise_model, seed_simulator=self._seed)

        # Transpile for the chosen simulator (ensures basis/optimization compatible)
        # tcirc = transpile(circuit, sim, optimization_level=0, seed_transpiler=self._seed)
        tcirc = circuit

        # If there are no classical bits, nothing to count
        if len(tcirc.clbits) == 0:
            warning("Did you forget to add measurement registers?")
            return {}

        # Run and return raw counts
        job = sim.run(tcirc, shots=self._repetitions)
        result = job.result()

        key_order, key_widths = self._key_meta(tcirc, result)

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
        little_endian_counts = result.get_counts(0)  # first (and typically only) circuit
        counts = self._invert_endianness(little_endian_counts)
        # ^^^^^^^^^^ IMPORTANT THING ^^^^^^^^^^ #
        # NOTE: When we invert this here we also HAVE to invert the post selection bitstrings

        # counts: Dict[str, int], keys are space-delimited per creg in qc.cregs order
        probs = self._counts_to_probs(counts)
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
