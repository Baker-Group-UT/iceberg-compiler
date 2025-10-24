"""
simulator/noise_models.py

Predefined Qiskit Aer noise models for use with the Simulator.
"""
# pylint: disable=[C0103]
from __future__ import annotations
from typing import Dict, Optional

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

# ---------- utils ----------

def noise_model_to_dict(model: NoiseModel) -> Dict:
    out: Dict[str, Dict] = {"quantum_errors": {}, "readout_errors": {}}
    for instr, errs in getattr(model, "_local_quantum_errors", {}).items():
        out["quantum_errors"][instr] = [repr(e) for e in errs]
    for instr, err in getattr(model, "_default_quantum_errors", {}).items():
        out["quantum_errors"].setdefault(instr, []).append(repr(err))
    for q, err in getattr(model, "_local_readout_errors", {}).items():
        try:
            mat = err.assignments
        except Exception:
            try:
                mat = err.assignment_matrix()
            except Exception:
                mat = "<?>"
        out["readout_errors"][str(q)] = mat
    return out

# ---------- presets ----------

def DefaultNoiseModel() -> NoiseModel:
    nm = NoiseModel()

    oneq_ops = ["x", "sx", "rz", "h"]
    twoq_ops = ["cx", "cz"]

    p1, p2 = 1e-3, 5e-3
    p_meas = 0.02
    T1, T2 = 100e-6, 80e-6
    t_1q, t_2q = 50e-9, 300e-9

    dep1 = depolarizing_error(p1, 1)
    thr1 = thermal_relaxation_error(T1, T2, t_1q)
    err1 = dep1.compose(thr1)
    nm.add_all_qubit_quantum_error(err1, oneq_ops)

    dep2 = depolarizing_error(p2, 2)
    thr2 = thermal_relaxation_error(T1, T2, t_2q).tensor(
        thermal_relaxation_error(T1, T2, t_2q)
    )
    err2 = dep2.compose(thr2)
    nm.add_all_qubit_quantum_error(err2, twoq_ops)

    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(ro)

    return nm


def StochasticNoiseModel() -> NoiseModel:
    nm = NoiseModel()
    oneq_ops = ["x", "sx", "rz", "h"]
    twoq_ops = ["cx", "cz"]

    p1, p2 = 5e-3, 2e-2
    p_meas = 0.05

    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), oneq_ops)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), twoq_ops)

    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(ro)
    return nm


def ScaledNoiseModel(
    alpha: float,
    *,
    base_p1: float = 1e-3,
    base_p2: float = 5e-3,
    base_p_meas: float = 0.02,
    base_T1: float = 100e-6,
    base_T2: float = 80e-6,
    t_1q: float = 50e-9,
    t_2q: float = 300e-9,
    scale_readout: bool = True,
    oneq_ops: Optional[list[str]] = None,
    twoq_ops: Optional[list[str]] = None,
) -> NoiseModel:
    """
    Rebuild a Qiskit Aer NoiseModel whose severity increases with `alpha`.

    alpha = 0  -> noiseless (empty model)
    alpha = 1  -> equivalent to DefaultNoiseModel()
    alpha > 1  -> linearly increases depolarizing + readout errors and shortens T1/T2.

    Args:
        alpha: Non-negative scalar controlling noise severity.
        base_*: Baseline parameters (match DefaultNoiseModel at alpha=1).
        scale_readout: If True, readout error scales with alpha as well.
        oneq_ops, twoq_ops: Gate name lists to attach 1q/2q errors to.

    Returns:
        NoiseModel with depolarizing + thermal + (optionally) readout errors scaled by alpha.
    """
    if alpha <= 0:
        return NoiseModel()

    if oneq_ops is None:
        oneq_ops = ["x", "sx", "rz", "h"]
    if twoq_ops is None:
        twoq_ops = ["cx", "cz"]

    # Scale probabilities linearly and clamp to reasonable maxima
    # (kept conservative to avoid singular matrices or non-CP maps).
    p1 = base_p1 * alpha
    p2 = base_p2 * alpha
    p_meas = base_p_meas * alpha if scale_readout else base_p_meas

    # Shorten T1/T2 as alpha grows (alpha=1 -> baseline; alpha=10 -> 10x shorter)
    T1 = base_T1 / alpha
    T2 = base_T2 / alpha

    nm = NoiseModel()

    # 1q: depol ∘ thermal
    dep1 = depolarizing_error(p1, 1)
    thr1 = thermal_relaxation_error(T1, T2, t_1q)
    err1 = dep1.compose(thr1)
    nm.add_all_qubit_quantum_error(err1, oneq_ops)

    # 2q: depol ∘ (thermal ⊗ thermal)
    dep2 = depolarizing_error(p2, 2)
    thr2 = thermal_relaxation_error(T1, T2, t_2q).tensor(
        thermal_relaxation_error(T1, T2, t_2q)
    )
    err2 = dep2.compose(thr2)
    nm.add_all_qubit_quantum_error(err2, twoq_ops)

    # Readout
    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(ro)

    return nm
