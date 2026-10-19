"""
frames.py -- Experiment8: measure detected and silent error on a translated circuit directly.

The cost model *predicts* silent error from per-gadget rates. This measures it, on the
exact noisy Stim circuit the simulator runs (same `qiskit_to_stim`, layout, idle and
movement noise, alpha), with no reference distribution needed.

`stim.FlipSimulator` with stabilizer randomization disabled propagates only the Pauli
frames the noise injects, so each shot's measurement flips are exactly the flips noise
caused relative to the noiseless run. Post-selection is then read off the flips the way
`iceberg.iceberg_postselect` reads it off the bits:

    detected   any `m_parity_*` / `flag*` bit flipped, or any patch's data parity flipped
    silent     not detected, but some decoded logical bit -- the parity of Zbar_i's two
               measured qubits (`CodeLayout.z_support`) -- flipped

`p_silent` is therefore the probability that a shot survives post-selection carrying a
logical error, which is what TVD after post-selection is made of; `p_detected` is
1 - LSR to first approximation. Monte Carlo over the noise, all orders, so these are not
first-order quantities.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.revision.exp8_undetected.gadgets import stim_overrides


def stim_params(alpha: float):
    """`StimNoiseParameters` of Fig 6's machine at `alpha` (what `ScaledSimulator` installs)."""
    from simulator.noise_models import ScaledNoiseModel, stim_params_from_noise_model

    return stim_params_from_noise_model(ScaledNoiseModel(alpha=alpha))


def _records(circuit) -> List[Tuple[str, int]]:
    """(register name, bit index) of every measurement, in Stim's record order."""
    out: List[Tuple[str, int]] = []
    for instruction in circuit.data:
        if instruction.operation.name.lower() != "measure":
            continue
        for clbit in instruction.clbits:
            register, index = circuit.find_bit(clbit).registers[0]
            out.append((register.name, index))
    return out


def detection_split(circuit, *, alpha: float = 1.0, idle_time_scalar: float = 1.0,
                    distance_scalar: float = 1.0, shots: int = 20_000,
                    seed: Optional[int] = 0) -> Dict[str, float]:
    """P(detected), P(silent logical error) and P(silent | kept) for one encoded circuit."""
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    params = stim_params(alpha)
    with sb._active_params(stim_overrides(params, idle_time_scalar=idle_time_scalar,
                                          distance_scalar=distance_scalar)):
        layout = positions_for_circuit(circuit, geometry=sb._geometry())
        noisy = sb.qiskit_to_stim(circuit, layout=layout, include_noise=True)
    return split_from_stim(circuit, noisy, shots=shots, seed=seed)


@contextmanager
def capture_noisy_stim():
    """
    Record every noisy circuit the simulator converts, as (qiskit circuit, Stim circuit).

    Converting is the slow step (seconds per circuit; the frame simulation is
    milliseconds), and the simulator already does it once per run. Wrapping the module
    function `stim_backend._run` calls lets the frame measurement reuse that conversion
    instead of repeating it. Worker-local and undone on exit; the library is untouched.
    """
    from simulator import stim_backend as sb

    original = sb.qiskit_to_stim
    captured: List[Tuple[object, object]] = []

    def recording(circuit, layout=None, include_noise=True):
        out = original(circuit, layout=layout, include_noise=include_noise)
        if include_noise:
            captured.append((circuit, out))
        return out

    sb.qiskit_to_stim = recording
    try:
        yield captured
    finally:
        sb.qiskit_to_stim = original


def split_from_stim(circuit, noisy, *, shots: int = 20_000,
                    seed: Optional[int] = 0) -> Dict[str, float]:
    """`detection_split` on an already-converted noisy Stim circuit."""
    import stim

    from iceberg.layout import CodeLayout

    simulator = stim.FlipSimulator(batch_size=shots, disable_stabilizer_randomization=True,
                                   num_qubits=noisy.num_qubits, seed=seed)
    simulator.do(noisy)
    flips = simulator.get_measurement_flips(bit_packed=False)

    rows: Dict[str, Dict[int, int]] = {}
    for row, (name, index) in enumerate(_records(circuit)):
        rows.setdefault(name, {})[index] = row
    if flips.shape[0] != sum(len(v) for v in rows.values()):
        raise ValueError("measurement records and Stim flips disagree in count")

    detected = np.zeros(shots, dtype=bool)
    logical = np.zeros(shots, dtype=bool)
    for name, by_index in rows.items():
        block = flips[[by_index[i] for i in sorted(by_index)]]
        if "parity" in name or "flag" in name:
            detected |= block.any(axis=0)
        elif "data" in name:
            detected |= (block.sum(axis=0) % 2).astype(bool)
            code = CodeLayout(block.shape[0])
            for lane in range(code.k):
                anchor, qubit = code.z_support(lane)
                logical |= block[anchor] ^ block[qubit]
    silent = logical & ~detected
    kept = max(1, int((~detected).sum()))
    return {
        "p_detected": float(detected.mean()),
        "p_silent": float(silent.mean()),
        "p_silent_given_kept": float(silent.sum() / kept),
        "frame_shots": int(shots),
    }


def translate(program, placement, n_code: int, *, commute: bool, merging: bool = True):
    """The encoded circuit the simulator would run for this placement."""
    from experiments.shared.simulation import ensure_measurements
    from pipeline.config import PipelineConfig
    from translator.core import Translator

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    translated = Translator(config=config.translator_config).translate_batch(
        ensure_measurements(program), [[tuple(p) for p in placement]])
    return translated[0].circuit
