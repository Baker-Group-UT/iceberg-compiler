"""
simulator/noise_models.py

Predefined Qiskit Aer noise models for use with the Simulator.
"""

# pylint: disable=[C0103]
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


@dataclass
class StimNoiseParameters:

    p1, p2 = 1.623e-4, 3.5e-3
    p_meas = 3.906e-3
    T1, T2 = 314.29e-6, 340.61e-6
    t_1q, t_2q = 32e-9, 88e-9

    T1: float = 314.29e-6
    T2: float = 340.61e-6
    one_qubit_gate_time: float = 32e-9
    two_qubit_gate_time: float = 88e-9
    one_qubit_error: float = 1.623e-4
    two_qubit_error: float = 3.5e-3
    meas_time: float = 2584e-9
    readout_error: float = 3.906e-3


@dataclass
class StimParameterScaling:
    T1: float = 1 # 0.118848
    T2: float = 1 # 0.736701
    one_qubit_gate_time: float = 1 # 0.137947
    two_qubit_gate_time: float = 1 # 0.151854
    one_qubit_error: float = 1 # 0.856380
    two_qubit_error: float = 1 # 0.75098
    meas_time: float = 1 # 0.176850
    readout_error: float = 1 # 0.279562


def _attach_stim_params(model: NoiseModel, params: StimNoiseParameters) -> NoiseModel:
    setattr(model, "_stim_params", params)
    return model


def stim_params_from_noise_model(model: Optional[NoiseModel]) -> StimNoiseParameters:
    if model is not None:
        params = getattr(model, "_stim_params", None)
        if isinstance(params, StimNoiseParameters):
            return params
    return StimNoiseParameters()


def apply_stim_scaling(
    params: StimNoiseParameters, scaling: Optional[StimParameterScaling]
) -> StimNoiseParameters:
    if scaling is None:
        return params
    return StimNoiseParameters(
        T1=params.T1 * scaling.T1,
        T2=params.T2 * scaling.T2,
        one_qubit_gate_time=params.one_qubit_gate_time * scaling.one_qubit_gate_time,
        two_qubit_gate_time=params.two_qubit_gate_time * scaling.two_qubit_gate_time,
        one_qubit_error=params.one_qubit_error * scaling.one_qubit_error,
        two_qubit_error=params.two_qubit_error * scaling.two_qubit_error,
        meas_time=params.meas_time * scaling.meas_time,
        readout_error=params.readout_error * scaling.readout_error,
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

    oneq_ops = ["x", "z", "s", "rz", "h"]
    twoq_ops = ["cx", "cz"]

    # Old Defaults
    # p1, p2 = 1e-3, 5e-3
    # p_meas = 0.02
    T1, T2 = 100e-6, 80e-6
    # t_1q, t_2q = 50e-9, 300e-9
    # IBM Pittsburgh
    p1, p2 = 1.623e-4, 3.5e-3
    p_meas = 3.906e-3
    # T1, T2 = 300e-6, 350e-6
    t_1q, t_2q = 32e-9, 88e-9

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

    params = StimNoiseParameters(
        T1=T1,
        T2=T2,
        one_qubit_gate_time=t_1q,
        two_qubit_gate_time=t_2q,
        one_qubit_error=p1,
        two_qubit_error=p2,
        meas_time=1e-6,
        readout_error=p_meas,
    )

    return _attach_stim_params(nm, params)


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

    params = StimNoiseParameters(
        T1=100e-6,
        T2=80e-6,
        one_qubit_gate_time=50e-9,
        two_qubit_gate_time=300e-9,
        one_qubit_error=p1,
        two_qubit_error=p2,
        meas_time=1e-6,
        readout_error=p_meas,
    )

    return _attach_stim_params(nm, params)


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
        # Return an explicitly noiseless model, but still attach stim params so the
        # Stim backend doesn't fall back to nonzero default parameters.
        nm = NoiseModel()
        params = StimNoiseParameters(
            T1=float("inf"),
            T2=float("inf"),
            one_qubit_gate_time=0.0,
            two_qubit_gate_time=0.0,
            one_qubit_error=0.0,
            two_qubit_error=0.0,
            meas_time=0.0,
            readout_error=0.0,
        )
        return _attach_stim_params(nm, params)

    if oneq_ops is None:
        oneq_ops = ["x", "sx", "rz", "h"]
    if twoq_ops is None:
        twoq_ops = ["cx", "cz"]

    # Scale probabilities linearly and clamp to reasonable maxima
    # (kept conservative to avoid singular matrices or non-CP maps).
    p1 = min(base_p1 * alpha, 1)
    p2 = min(base_p2 * alpha, 1)
    if scale_readout:
        p_meas = min(base_p_meas * alpha, 1.0 - 1e-9)
    else:
        p_meas = base_p_meas
    # p1 = base_p1 * alpha
    # p2 = base_p2 * alpha
    # p_meas = base_p_meas * alpha if scale_readout else base_p_meas

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

    params = StimNoiseParameters(
        T1=T1,
        T2=T2,
        one_qubit_gate_time=t_1q,
        two_qubit_gate_time=t_2q,
        one_qubit_error=p1,
        two_qubit_error=p2,
        meas_time=1e-6,
        readout_error=p_meas,
    )

    return _attach_stim_params(nm, params)


def DistanceDependentNoiseModel(
    *,
    layout: Dict[int, tuple[int, int]] = None,
    distance_scalar: float = 1,
    distance_function: Callable[
        [float, float, float, float],
        Callable[[tuple[int, int], tuple[int, int]], QuantumError],
    ] = None,
    p1: float = 1e-3,
    p2: float = 5e-3,
    p_meas: float = 0.02,
    T1: float = 100e-6,
    T2: float = 80e-6,
    t_1q: float = 50e-9,
    t_2q: float = 300e-9,
    oneq_ops: Optional[list[str]] = None,
    twoq_ops: Optional[list[str]] = None,
) -> NoiseModel:
    """
    Build a Qiskit Aer NoiseModel with two-qubit errors dependent on qubit distance.
    Args:
        layout: Mapping of qubit index to (x, y) position.
        distance_function: Function that takes T1, T2, t_2q, p2 and returns a function
            which takes two positions and returns a QuantumError for those qubits.
        p1, p2: Depolarizing error probabilities for 1q and 2q gates.
        p_meas: Readout error probability.
        T1, T2: Relaxation and dephasing times.
        t_1q, t_2q: Gate times for 1q and 2q gates.
        oneq_ops, twoq_ops: Gate name lists to attach 1q/2q errors to.
    """
    if layout is None:
        raise ValueError("DistanceDependentNoiseModel requires a qubit layout.")

    if twoq_ops is None:
        twoq_ops = ["cx", "cz"]
    if oneq_ops is None:
        oneq_ops = ["x", "sx", "rz", "h"]

    if distance_function is None:
        distance_function = distance_noise_function

    nm = NoiseModel()

    dep1 = depolarizing_error(p1, 1)
    thr1 = thermal_relaxation_error(T1, T2, t_1q)
    err1 = dep1.compose(thr1)
    nm.add_all_qubit_quantum_error(err1, oneq_ops)

    df = distance_function(T1, T2, t_2q, p2, distance_scalar)
    for q1 in layout:
        for q2 in layout:
            if q1 >= q2:
                continue
            pos1 = layout[q1]
            pos2 = layout[q2]
            err2 = df(pos1, pos2)
            nm.add_quantum_error(err2, twoq_ops, [q1, q2])

    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(ro)

    params = StimNoiseParameters(
        T1=T1,
        T2=T2,
        one_qubit_gate_time=t_1q,
        two_qubit_gate_time=t_2q,
        one_qubit_error=p1,
        two_qubit_error=p2,
        meas_time=1e-6,
        readout_error=p_meas,
    )

    return _attach_stim_params(nm, params)


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def distance_noise_function(
    T1: float,
    T2: float,
    t_2q: float,
    p2: float,
    distance_scalar: float = 1,
    dist: Callable[[tuple[int, int], tuple[int, int]], float] = manhattan_distance,
    movement_speed: float = 0.5,
) -> Callable[[tuple[int, int], tuple[int, int]], QuantumError]:
    """
    Returns a function that computes a QuantumError based on Manhattan distance.
    The error probability increases with distance.
    """

    def error_function(pos1: tuple[int, int], pos2: tuple[int, int]) -> QuantumError:
        distance = dist(pos1, pos2) * 1e-8 * distance_scalar
        time_to_move = 2 * distance / movement_speed
        time_2q = t_2q + time_to_move
        # print(distance, time_to_move, time_2q, p2)
        dep2 = depolarizing_error(p2, 2)
        thr2 = thermal_relaxation_error(T1, T2, time_2q).tensor(
            thermal_relaxation_error(T1, T2, time_2q)
        )
        return dep2.compose(thr2)

    return error_function
