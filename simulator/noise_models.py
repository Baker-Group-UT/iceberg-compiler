"""
simulator/noise_models.py

Predefined Qiskit Aer noise models for use with the Simulator.
"""
# pylint: disable=[C0103]

from __future__ import annotations
from typing import Dict

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

# ---------- utils ----------

def noise_model_to_dict(model: NoiseModel) -> Dict:
    """
    Lightweight serialization for inspection/logging.
    (Qiskit NoiseModel has no stable public to_dict API.)
    """
    out: Dict[str, Dict] = {"quantum_errors": {}, "readout_errors": {}}
    # Quantum errors (per instruction)
    # NOTE: using private fields for introspection; stable enough for diagnostics.
    for instr, errs in getattr(model, "_local_quantum_errors", {}).items():
        out["quantum_errors"][instr] = [repr(e) for e in errs]
    for instr, err in getattr(model, "_default_quantum_errors", {}).items():
        out["quantum_errors"].setdefault(instr, []).append(repr(err))
    # Readout errors
    for q, err in getattr(model, "_local_readout_errors", {}).items():
        try:
            mat = err.assignments  # newer
        except Exception:
            try:
                mat = err.assignment_matrix()  # older
            except Exception:
                mat = "<?>"
        out["readout_errors"][str(q)] = mat
    return out

# ---------- presets ----------

def DefaultNoiseModel() -> NoiseModel:
    """
    Hardware-ish, moderate noise:
      - 1q depolarizing p=1e-3 on {x,sx,rz,h}
      - 2q depolarizing p=5e-3 on {cx,cz}
      - Readout error p=0.02
      - Thermal relaxation with T1=100us, T2=80us, gate times 50ns/300ns (composed with depol)
    """
    nm = NoiseModel()

    # Gate sets
    oneq_ops = ["x", "sx", "rz", "h"]
    twoq_ops = ["cx", "cz"]

    # Parameters
    p1, p2 = 1e-3, 5e-3
    p_meas = 0.02
    T1, T2 = 100e-6, 80e-6   # seconds
    t_1q, t_2q = 50e-9, 300e-9

    # 1q: depol ∘ thermal
    dep1 = depolarizing_error(p1, 1)
    thr1 = thermal_relaxation_error(T1, T2, t_1q)
    err1 = dep1.compose(thr1)
    nm.add_all_qubit_quantum_error(err1, oneq_ops)

    # 2q: depol ∘ (thermal ⊗ thermal)
    dep2 = depolarizing_error(p2, 2)
    thr2 = thermal_relaxation_error(T1, T2, t_2q).tensor(thermal_relaxation_error(T1, T2, t_2q))
    err2 = dep2.compose(thr2)
    nm.add_all_qubit_quantum_error(err2, twoq_ops)

    # Readout
    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(ro)

    return nm


def StochasticNoiseModel() -> NoiseModel:
    """
    Faster, purely stochastic model (no T1/T2):
      - 1q depolarizing p=5e-3 on {x,sx,rz,h}
      - 2q depolarizing p=2e-2 on {cx,cz}
      - Readout error p=0.05
    """
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


def ScaledNoiseModel(alpha: float, base: NoiseModel = None) -> NoiseModel:
    """
    Return a copy of `base` with quantum-error probabilities scaled by `alpha`.
    (Readout errors left unchanged for simplicity.)

    Note: Uses private NoiseModel fields to reattach errors; future Qiskit may change these.
    """
    if base is None:
        base = DefaultNoiseModel()
    if alpha <= 0:
        # Degenerate: no noise
        return NoiseModel()

    scaled = NoiseModel()

    # Copy and scale quantum errors
    for instr, errs in getattr(base, "_local_quantum_errors", {}).items():
        for err in errs:
            e = err.copy()
            if hasattr(e, "params"):
                e.params = [min(max(float(p) * alpha, 0.0), 1.0) if isinstance(p, (int, float)) else p for p in e.params]
            scaled.add_all_qubit_quantum_error(e, [instr])

    for instr, err in getattr(base, "_default_quantum_errors", {}).items():
        e = err.copy()
        if hasattr(e, "params"):
            e.params = [min(max(float(p) * alpha, 0.0), 1.0) if isinstance(p, (int, float)) else p for p in e.params]
        scaled.add_all_qubit_quantum_error(e, [instr])

    # Copy readout errors as-is
    for q, err in getattr(base, "_local_readout_errors", {}).items():
        scaled.add_readout_error(err, [q])

    return scaled