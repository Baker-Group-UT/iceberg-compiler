"""
packer.calibrate
=================

Monte‑Carlo style calibration utilities for estimating logical error weights
for individual Iceberg logical gates.  The workflow for one calibration sample
is:

1. Build a random logical state-preparation circuit (Qiskit) for the requested
   number of logical qubits.
2. Append the logical gate under test (X, Z, H, RZ, CX, CZ, S, HH, CXT, CZT…).
3. Translate the logical circuit into an Iceberg physical circuit via the
   translator package (encode → logical ops → decode → measure).
4. Noiselessly simulate encoding + prep, noisily simulate only the logical gate
   under test (using any of the simulator noise models), then noiselessly
   simulate the remainder (decode + measurements).
5. Convert the resulting probability distribution into the same format that the
   pipeline expects and run the standard Iceberg postselection analysis to
   obtain logical TVD/LSR metrics.
6. Repeat for many random preparations to build an empirical distribution and
   derive an expected error weight per logical gate.

The main entrypoint is :func:`calibrate_gates`, which returns a dictionary of
`GateCalibrationSummary` objects keyed by gate name.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.random import random_clifford_circuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetDensityMatrix
from qiskit_aer.noise import NoiseModel
from tqdm.auto import trange

from pipeline.analysis.postselection import apply_iceberg_postselection
from pipeline.analysis.result_types import PostselectedSimulationResult
from simulator.noise_models import DefaultNoiseModel
from simulator.result_types import SimulationResult
from translator.config import TranslatorConfig
from translator.core import Translation, TranslationResult
from translator.merging import HHGate
from translator.merging import preprocess_program_for_merging
from translator.passes import cancel_inverse_gates

os.environ.setdefault("QISKIT_AER_SHM_DISABLE", "1")

LSR_FLOOR = 1e-6
DEFAULT_PREP_GATES = ("x", "z", "h", "s", "rz", "cx", "cz")
DEFAULT_GATES_TO_CALIBRATE = ("x", "z", "h", "s", "rz", "cx", "cz", "hh")
# DEFAULT_GATES_TO_CALIBRATE = ["cx", "cxt"]


def default_permutation(
    gate: Optional[str] = None,
    num_patches: int = 2,
    k_log: int = 2,
) -> List[Tuple[int, int]]:
    """
    Build a default logical-to-physical permutation for ``gate``.  Non-CXT/CZT
    gates default to a single patch while transversal CXT/CZT gates retain two
    patches.  Each patch contributes ``k_log`` logical slots.
    """
    normalized_gate = gate.lower() if gate else None
    patch_count = num_patches if normalized_gate in {"cxt", "czt"} else 1
    return [(patch, slot) for patch in range(patch_count) for slot in range(k_log)]


def _slice_circuit(source: QuantumCircuit, start: int, stop: int) -> QuantumCircuit:
    """
    Return a shallow copy of ``source`` containing operations in [start, stop).
    Registers are shared so the resulting circuit can be simulated without
    re-binding qubits.
    """
    sub = QuantumCircuit(*source.qregs, *source.cregs)
    for inst in source.data[start:stop]:
        sub.append(inst.operation, inst.qubits, inst.clbits)
    return sub


def _strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Copy ``circ`` but drop measurement instructions (used so we can evolve the
    unitary portion and handle measurement probabilities analytically).
    """
    stripped = QuantumCircuit(*circ.qregs, *circ.cregs)
    for inst in circ.data:
        if inst.operation.name == "measure":
            continue
        stripped.append(inst.operation, inst.qubits, inst.clbits)
    return stripped


def _evolve_segment(
    segment: QuantumCircuit,
    state: Optional[DensityMatrix],
    noise_model: Optional[NoiseModel],
    seed: Optional[int],
) -> DensityMatrix:
    """
    Execute ``segment`` starting from ``state`` (density matrix) and return the
    resulting density matrix.  If ``state`` is None the segment starts in
    |0...0>.  When ``noise_model`` is None the evolution is noiseless.
    """
    if len(segment.data) == 0:
        return state.copy() if state is not None else DensityMatrix.from_label("0" * segment.num_qubits)

    working = QuantumCircuit(*segment.qregs, *segment.cregs)
    if state is not None:
        working.append(SetDensityMatrix(state), working.qubits)
    working.compose(segment, inplace=True)
    # pylint: disable=no-member
    working.save_density_matrix(label="rho_out")

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model, seed_simulator=seed)
    tcirc = transpile(working, simulator, optimization_level=0, seed_transpiler=seed)
    result = simulator.run(tcirc, shots=1).result()
    rho = result.data(0)["rho_out"]
    return DensityMatrix(rho)


@dataclass(frozen=True)
class MeasurementLayout:
    key_order_raw: List[str]
    key_widths_raw: List[int]
    reg_qubit_indices: List[List[int]]

    @property
    def key_order(self) -> List[str]:
        return list(reversed(self.key_order_raw))

    @property
    def key_widths(self) -> List[int]:
        return list(reversed(self.key_widths_raw))


def _build_measurement_layout(circuit: QuantumCircuit) -> MeasurementLayout:
    """
    Inspect measurement operations in ``circuit`` and build:
      * register order/width metadata (matching Simulator._key_meta fallback)
      * mapping from each classical register bit to the underlying qubit index
    """
    qubit_index = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    meas_map: Dict[object, int] = {}
    for inst in circuit.data:
        if inst.operation.name != "measure":
            continue
        for qubit, clbit in zip(inst.qubits, inst.clbits):
            meas_map[clbit] = qubit_index[qubit]

    reg_qubit_indices: List[List[int]] = []
    key_order = []
    key_widths = []

    for creg in circuit.cregs:
        indices: List[int] = []
        for bit in creg:
            if bit not in meas_map:
                raise ValueError(f"Classical bit {bit} has no measurement mapping; circuit lacks measure on this register.")
            indices.append(meas_map[bit])
        reg_qubit_indices.append(indices)
        key_order.append(creg.name)
        key_widths.append(len(creg))

    if not key_order:
        raise ValueError("Circuit has no classical registers; calibration requires measurements.")

    return MeasurementLayout(key_order_raw=key_order, key_widths_raw=key_widths, reg_qubit_indices=reg_qubit_indices)


def _density_to_distribution(
    density: DensityMatrix,
    layout: MeasurementLayout,
) -> Dict[str, float]:
    """
    Convert a density matrix into the probability distribution over the
    classical registers specified by ``layout``.  Keys match the
    Simulator-produced format (register order reversed, bits reversed within
    each register).
    """
    probs = density.probabilities_dict()
    dist: Dict[str, float] = defaultdict(float)

    reg_qubit_indices = layout.reg_qubit_indices
    for key, p in probs.items():
        bitstring = str(key)[::-1]  # flip so index 0 corresponds to qubit 0
        pre_parts: List[str] = []
        for indices in reg_qubit_indices:
            reg_bits = "".join(bitstring[idx] for idx in indices)
            pre_parts.append(reg_bits)
        flipped_parts = ["".join(reversed(part)) for part in reversed(pre_parts)]
        final_key = " ".join(flipped_parts)
        dist[final_key] += float(p)

    return dict(dist)


@dataclass
class GateCalibrationConfig:
    """
    Runtime knobs for gate calibration.
    """

    permutation: List[Tuple[int, int]] = field(default_factory=default_permutation)
    trials_per_gate: int = 1000
    prep_depth_range: Tuple[int, int] = (2, 6)
    prep_gate_set: Sequence[str] = field(default_factory=lambda: DEFAULT_PREP_GATES)
    gates: Sequence[str] = field(default_factory=lambda: DEFAULT_GATES_TO_CALIBRATE)
    noise_model: NoiseModel = field(default_factory=DefaultNoiseModel)
    translator_config: TranslatorConfig = field(default_factory=TranslatorConfig)
    rz_angle_sampler: Callable[[np.random.Generator], float] = field(
        default_factory=lambda: (lambda rng: float(rng.uniform(0.0, 2 * math.pi)))
    )
    rng_seed: Optional[int] = None
    lsr_floor: float = LSR_FLOOR
    max_attempts_per_trial: int = 5


@dataclass(frozen=True)
class GateCalibrationSample:
    gate: str
    trial: int
    result: PostselectedSimulationResult

    @property
    def tvd(self) -> float:
        return float(self.result.logical_total_variation_distance)

    @property
    def lsr(self) -> float:
        return float(self.result.logical_selection_rate)


@dataclass
class GateCalibrationSummary:
    gate: str
    samples: List[GateCalibrationSample]
    lsr_floor: float = LSR_FLOOR

    def tvd_values(self) -> List[float]:
        return [sample.tvd for sample in self.samples]

    def lsr_values(self) -> List[float]:
        return [sample.lsr for sample in self.samples]

    def expected_error(self) -> float:
        values = []
        for sample in self.samples:
            denom = max(sample.lsr, self.lsr_floor)
            values.append(sample.tvd / denom)
        return float(np.mean(values)) if values else 0.0

    def stats(self) -> Dict[str, float]:
        tvd = np.array(self.tvd_values(), dtype=float)
        lsr = np.array(self.lsr_values(), dtype=float)
        return {
            "tvd_mean": float(np.mean(tvd)) if len(tvd) else 0.0,
            "tvd_std": float(np.std(tvd)) if len(tvd) else 0.0,
            "lsr_mean": float(np.mean(lsr)) if len(lsr) else 0.0,
            "lsr_std": float(np.std(lsr)) if len(lsr) else 0.0,
            "expected_error": self.expected_error(),
        }

    def to_dict(self) -> Dict[str, object]:
        payload = self.stats()
        payload["gate"] = self.gate
        payload["num_samples"] = len(self.samples)
        payload["samples"] = [
            {
                "trial": sample.trial,
                "logical_total_variation_distance": sample.tvd,
                "logical_selection_rate": sample.lsr,
            }
            for sample in self.samples
        ]
        return payload


class GateCalibrator:
    """
    Helper object that orchestrates translation, segmented simulation, and
    postselection for logical gate calibration.
    """

    def __init__(self, config: Optional[GateCalibrationConfig] = None):
        self.config = config or GateCalibrationConfig()
        self.rng = np.random.default_rng(self.config.rng_seed)
        self._noise_model = self.config.noise_model
        self._translator_cfg = self.config.translator_config
        self._patch_slots = self._build_patch_slots(self.config.permutation)

    @staticmethod
    def _build_patch_slots(permutation: Sequence[Tuple[int, int]]) -> Dict[int, Dict[int, int]]:
        slots: Dict[int, Dict[int, int]] = {}
        for logical_i, (patch, slot) in enumerate(permutation):
            slots.setdefault(patch, {})[slot] = logical_i
        return slots

    @property
    def num_logical_qubits(self) -> int:
        return len(self.config.permutation)

    def calibrate_gate(self, gate: str) -> GateCalibrationSummary:
        samples: List[GateCalibrationSample] = []
        for trial in trange(
            self.config.trials_per_gate,
            desc=f"Calibrating {gate}",
            unit=" trial",
        ):
            sample = self._run_trial(gate, int(trial))
            samples.append(sample)
        return GateCalibrationSummary(gate=gate, samples=samples, lsr_floor=self.config.lsr_floor)

    def calibrate(self, gates: Optional[Iterable[str]] = None) -> Dict[str, GateCalibrationSummary]:
        gates = list(gates) if gates is not None else list(self.config.gates)
        summaries: Dict[str, GateCalibrationSummary] = {}
        for gate in gates:
            self.config.permutation = default_permutation(gate=gate)
            self._patch_slots = self._build_patch_slots(self.config.permutation)
            summaries[gate] = self.calibrate_gate(gate)
        return summaries

    # ------------------------------------------------------------------
    # Trial orchestration
    # ------------------------------------------------------------------

    def _run_trial(self, gate: str, trial_index: int) -> GateCalibrationSample:
        last_error: Optional[Exception] = None
        for _ in range(self.config.max_attempts_per_trial):
            try:
                translation, segments, measurement_layout = self._build_translated_circuit(gate)
                sim_result = self._simulate_segments(translation.circuit, segments, measurement_layout)
                post_results = apply_iceberg_postselection([translation], [sim_result])
                sample = GateCalibrationSample(gate=gate, trial=trial_index, result=post_results[0])
                return sample
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"Failed to build a valid {gate.upper()} calibration trial after "
                           f"{self.config.max_attempts_per_trial} attempts.") from last_error

    # ------------------------------------------------------------------
    # Logical circuit preparation
    # ------------------------------------------------------------------

    def _random_prep_circuit(self) -> QuantumCircuit:
        """
        Generate a random Clifford preparation circuit using Qiskit's helper.
        """
        num_qubits = self.num_logical_qubits
        if num_qubits == 0:
            return QuantumCircuit(0)

        depth_low, depth_high = self.config.prep_depth_range
        low = min(depth_low, depth_high)
        high = max(depth_low, depth_high)
        if high <= 0:
            return QuantumCircuit(num_qubits)
        low = max(low, 1)
        num_gates = int(self.rng.integers(low, high + 1))

        gate_set = [g.lower() for g in self.config.prep_gate_set]
        supported = ["x", "z", "h", "s", "cx", "cz"]
        filtered_gates = [gate for gate in gate_set if gate in supported]
        if not filtered_gates:
            filtered_gates = supported

        seed = int(self.rng.integers(0, 2**32 - 1))
        return random_clifford_circuit(
            num_qubits=num_qubits,
            num_gates=num_gates,
            seed=seed,
            gates=filtered_gates,
        )

    def _gate_circuit(self, gate: str) -> QuantumCircuit:
        """
        Build a single-gate logical circuit for the gate under test.
        """
        gate = gate.lower()
        qc = QuantumCircuit(self.num_logical_qubits)

        if gate in ("x", "z", "h", "s"):
            target = int(self.rng.integers(0, self.num_logical_qubits))
            getattr(qc, gate)(target)
        elif gate == "rz":
            target = int(self.rng.integers(0, self.num_logical_qubits))
            theta = self.config.rz_angle_sampler(self.rng)
            qc.rz(theta, target)
        elif gate in ("cx", "cz"):
            control, target = self._random_pair()
            getattr(qc, gate)(control, target)
        elif gate == "hh":
            patch, logicals = self._random_patch_with_two_slots()
            hh_gate = HHGate()
            qc.append(hh_gate, [logicals[0], logicals[1]])
        elif gate in ("cxt", "czt"):
            control_patch, target_patch = self._random_patch_pair()
            control_logical = self._random_logical_from_patch(control_patch)
            target_logical = self._random_logical_from_patch(target_patch)
            macro = Gate(gate, 2, [])
            qc.append(macro, [control_logical, target_logical])
        else:
            raise ValueError(f"Unsupported gate '{gate}'.")

        return qc

    def _process_logical_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Apply optional HH/CXT/CZT merging followed by local inverse cancellation.
        """
        processed = circuit
        if self._translator_cfg.enable_hh_merging:
            processed = preprocess_program_for_merging(
                processed,
                self.config.permutation,
                allow_transversal_swap_merges=self._translator_cfg.allow_transversal_swap_merges,
            )
        processed = cancel_inverse_gates(processed)
        return processed

    def _build_program(self, gate: str) -> Tuple[QuantumCircuit, int, int]:
        prep = self._process_logical_circuit(self._random_prep_circuit())
        gate_circ = self._process_logical_circuit(self._gate_circuit(gate))

        program = QuantumCircuit(self.num_logical_qubits)
        program.compose(prep, inplace=True)
        program.barrier()
        program.compose(gate_circ, inplace=True)

        return program, len(prep.data), len(gate_circ.data)

    # ------------------------------------------------------------------
    # Translation and segmentation
    # ------------------------------------------------------------------

    @dataclass
    class _Segments:
        pre_stop: int
        gate_stop: int
        total_ops: int

    def _build_translated_circuit(
        self,
        gate: str,
    ) -> Tuple[TranslationResult, "_Segments", MeasurementLayout]:
        program, prep_len, gate_len = self._build_program(gate)

        # print(program)
        translation = Translation(
            permutation=self.config.permutation,
            # permutation=[(0,0),(1,0),(0,1),(1,1)],
            program_circuit=program,
            fault_tolerant=self._translator_cfg.fault_tolerant,
            verbose=self._translator_cfg.verbose,
        )

        pre_stop = gate_stop = None
        iceberg_circuit = translation.context.iceberg_set.circuit

        prep_consumed = 0
        gate_consumed = 0

        for inst in program.data:
            flag = translation.translate_gate(inst)
            if flag is False:
                raise RuntimeError(f"Failed to translate instruction {inst.operation.name} during {gate.upper()} calibration.")
            if inst.operation.name == "barrier":
                continue
            if prep_consumed < prep_len:
                prep_consumed += 1
                if prep_consumed == prep_len:
                    pre_stop = len(iceberg_circuit.data)
                continue
            gate_consumed += 1
            if gate_consumed == gate_len:
                gate_stop = len(iceberg_circuit.data)

        if pre_stop is None:
            pre_stop = len(iceberg_circuit.data)
        if gate_stop is None:
            gate_stop = len(iceberg_circuit.data)
        # print(translation.context.iceberg_set.circuit)
        translation.context.iceberg_set.decode()
        translation.context.iceberg_set.measure()

        total_ops = len(iceberg_circuit.data)
        segments = GateCalibrator._Segments(pre_stop=pre_stop, gate_stop=gate_stop, total_ops=total_ops)

        measurement_layout = _build_measurement_layout(iceberg_circuit)

        translation_result = TranslationResult(
            circuit=iceberg_circuit,
            permutation=self.config.permutation,
            context=translation.context,
            contains_faults=(len(translation.context.fault_log) == 0),
            fault_log=translation.context.fault_log,
            verbose=self._translator_cfg.verbose,
            fault_tolerant=self._translator_cfg.fault_tolerant,
        )

        return translation_result, segments, measurement_layout

    # ------------------------------------------------------------------
    # Segmented simulation
    # ------------------------------------------------------------------

    def _simulate_segments(
        self,
        circuit: QuantumCircuit,
        segments: "_Segments",
        layout: MeasurementLayout,
    ) -> SimulationResult:
        pre_segment = _slice_circuit(circuit, 0, segments.pre_stop)
        gate_segment = _slice_circuit(circuit, segments.pre_stop, segments.gate_stop)
        post_segment = _slice_circuit(circuit, segments.gate_stop, segments.total_ops)
        post_unitary = _strip_measurements(post_segment)

        seed = int(self.rng.integers(0, 2**32))

        rho_pre = _evolve_segment(pre_segment, state=None, noise_model=None, seed=seed)
        rho_gate_ideal = _evolve_segment(gate_segment, rho_pre, noise_model=None, seed=seed)
        rho_gate_noisy = _evolve_segment(gate_segment, rho_pre, noise_model=self._noise_model, seed=seed)

        rho_post_ideal = _evolve_segment(post_unitary, rho_gate_ideal, noise_model=None, seed=seed)
        rho_post_noisy = _evolve_segment(post_unitary, rho_gate_noisy, noise_model=None, seed=seed)

        ideal_dist = _density_to_distribution(rho_post_ideal, layout)
        noisy_dist = _density_to_distribution(rho_post_noisy, layout)

        return SimulationResult(
            key_order=layout.key_order,
            key_widths=layout.key_widths,
            ideal_dist=ideal_dist,
            noisy_dist=noisy_dist,
        )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _random_pair(self) -> Tuple[int, int]:
        if self.num_logical_qubits < 2:
            raise ValueError("Need at least two logical qubits for a two-qubit gate.")
        # i, j = self.rng.choice(self.num_logical_qubits, size=2, replace=False)
        i, j = 0, 1
        return int(i), int(j)

    def _random_patch_with_two_slots(self) -> Tuple[int, List[int]]:
        candidates = [(patch, slots) for patch, slots in self._patch_slots.items() if len(slots) >= 2]
        if not candidates:
            raise ValueError("No patch contains two logical qubits; cannot place HH gate.")
        patch, slots = candidates[self.rng.integers(0, len(candidates))]
        ordered = [slots[k] for k in sorted(slots)]
        return patch, ordered[:2]

    def _random_patch_pair(self) -> Tuple[int, int]:
        patches = list(self._patch_slots.keys())
        if len(patches) < 2:
            raise ValueError("Need at least two patches to place transversal gates.")
        selected = self.rng.choice(patches, size=2, replace=False)
        return int(selected[0]), int(selected[1])

    def _random_logical_from_patch(self, patch: int) -> int:
        logicals = list(self._patch_slots[patch].values())
        choice = self.rng.choice(logicals)
        return int(choice)


def calibrate_gates(
    gates: Optional[Iterable[str]] = None,
    config: Optional[GateCalibrationConfig] = None,
) -> Dict[str, GateCalibrationSummary]:
    """
    Convenience entrypoint for one-off calibrations.
    """
    calibrator = GateCalibrator(config)
    return calibrator.calibrate(gates)


def _save_calibration_results(
    summaries: Dict[str, GateCalibrationSummary],
    output_dir: str = "results",
) -> str:
    """
    Persist calibration summaries as JSON in ``output_dir`` and return the path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"gate_calibration_{timestamp}.json")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gates": {gate: summary.to_dict() for gate, summary in summaries.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
def _print_calibration_results(summaries: Dict[str, GateCalibrationSummary]) -> None:
    """
    Pretty-print aggregate stats for each calibrated gate.
    """
    if not summaries:
        print("No calibration results to display.")
        return
    print("Gate calibration summary:")
    for gate in sorted(summaries):
        summary = summaries[gate]
        stats = summary.stats()
        print(
            f"  {gate.upper():>4} | samples={len(summary.samples):3d} | "
            f"tvd_mean={stats['tvd_mean']:.4e} ± {stats['tvd_std']:.2e} | "
            f"lsr_mean={stats['lsr_mean']:.4e} ± {stats['lsr_std']:.2e} | "
            f"expected_error={stats['expected_error']:.4e}"
        )


__all__ = [
    "GateCalibrationConfig",
    "GateCalibrationSample",
    "GateCalibrationSummary",
    "GateCalibrator",
    "calibrate_gates",
]

if __name__ == "__main__":
    _summaries = calibrate_gates()
    _print_calibration_results(_summaries)
    _output_path = _save_calibration_results(_summaries)
    print(f"Saved calibration results to {_output_path}")
