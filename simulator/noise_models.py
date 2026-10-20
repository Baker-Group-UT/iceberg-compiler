"""
simulator/noise_models.py

Predefined Qiskit Aer noise models for use with the Simulator.
"""

# pylint: disable=[C0103]
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

#: Single-qubit gates a *translated* circuit can contain. The paper-accurate models used
#: to list ["x", "sx", "rz", "h"], which left `s` and `z` completely noiseless -- 6 of 28
#: single-qubit gates in one translated Clifford program got no error at all, while `sx`
#: in that list never appears. Encoded circuits are full of `s` and `z`, because that is
#: what the logical Pauli and phase gadgets are built from, so those two omissions made
#: every simulated number optimistic. Every model here now shares this one list.
TRANSLATED_ONE_QUBIT_OPS = ["x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "rx", "rz"]


#: A rubidium tweezer-array machine -- the architecture this compiler targets, and the
#: default everywhere since `simulator.transport` landed.
#:
#: The dominant difference against the superconducting set below is T2: 1.5 s against
#: 341 us, a factor of ~4400, while the two-qubit gate only slows 88 ns to 500 ns. Idle
#: per layer therefore falls about three orders of magnitude, which relocates the whole
#: idle/distance balance -- expect this to move results more than the transport model does.
#:
#: T2 = 1.5 s assumes dynamical decoupling on the hyperfine qubit. Without it T2 is
#: milliseconds and nothing survives a circuit of this depth, so it is an assumption about
#: the machine rather than a knob to sweep.
#:
#: Readout is slow and batched: one 3 ms image covers the whole array, so a detection round
#: costs `meas_time` once no matter how many qubits it reads. This pipeline does one round
#: at the end.
#:
#: **The single-qubit gate is slower than the two-qubit gate, and that is not a typo.** It
#: is a real inversion on this platform and it catches everyone. A two-qubit gate is a
#: Rydberg blockade pulse -- the interaction is enormous, so the gate is fast, a few hundred
#: ns. A single-qubit gate is a Raman transition driven through a tightly focused addressing
#: beam, which is weak by comparison and takes ~1 us. Superconducting hardware has the
#: opposite ordering (32 ns against 88 ns in `SUPERCONDUCTING`), which is where the
#: expectation comes from.
NEUTRAL_ATOM = {
    "T1": 10.0,
    "T2": 1.5,
    "one_qubit_gate_time": 1e-6,
    "two_qubit_gate_time": 0.5e-6,
    "one_qubit_error": 2e-4,
    "two_qubit_error": 3e-3,
    "meas_time": 3e-3,
    "readout_error": 3e-3,
}

#: Infidelity per atom per carriage at alpha = 1, from heating and imperfect handoff.
#: Lives outside `NEUTRAL_ATOM` because the Aer dicts predate movement; it is the
#: baseline that `ScaledNoiseModel` multiplies by alpha like every other error rate.
BASE_MOVE_ERROR = 1e-3

#: The calibration that was default before the neutral-atom model: T1/T2 of 314/341 us
#: against an 88 ns two-qubit gate, which is a **superconducting** machine and cannot move
#: atoms at all. Kept so every number this repo published before the switch stays
#: reproducible -- `StimNoiseParameters.superconducting()`.
SUPERCONDUCTING = {
    "T1": 314.29e-6,
    "T2": 340.61e-6,
    "one_qubit_gate_time": 32e-9,
    "two_qubit_gate_time": 88e-9,
    "one_qubit_error": 1.623e-4,
    "two_qubit_error": 3.5e-3,
    "meas_time": 2584e-9,
    "readout_error": 3.906e-3,
}

#: Table 2 of the paper, which `ScaledNoiseModel(alpha=1.0)` reproduced exactly before the
#: switch. Retained for the same reason as `SUPERCONDUCTING`.
PAPER_TABLE_2 = {
    "T1": 100e-6,
    "T2": 80e-6,
    "one_qubit_gate_time": 50e-9,
    "two_qubit_gate_time": 300e-9,
    "one_qubit_error": 1e-3,
    "two_qubit_error": 5e-3,
    "meas_time": 1e-6,
    "readout_error": 0.02,
}


@dataclass
class StimNoiseParameters:
    """
    Hardware rates and times. Defaults are `NEUTRAL_ATOM`.

    Two named alternatives exist for reproducing pre-transport-model numbers:
    `superconducting()` and `paper_table_2()`.
    """

    T1: float = NEUTRAL_ATOM["T1"]
    T2: float = NEUTRAL_ATOM["T2"]
    one_qubit_gate_time: float = NEUTRAL_ATOM["one_qubit_gate_time"]
    two_qubit_gate_time: float = NEUTRAL_ATOM["two_qubit_gate_time"]
    one_qubit_error: float = NEUTRAL_ATOM["one_qubit_error"]
    two_qubit_error: float = NEUTRAL_ATOM["two_qubit_error"]
    meas_time: float = NEUTRAL_ATOM["meas_time"]
    readout_error: float = NEUTRAL_ATOM["readout_error"]
    idle_time_scalar: float = 1.0

    #: Geometry, in the units `simulator.transport.Geometry` expects. Defaults match that
    #: module's constants; they live here too so a run can carry one parameter object.
    atom_pitch: float = 3e-6
    blockade_radius: float = 8e-6
    #: ~1 us of flight per um travelled. See `simulator.transport.MOVEMENT_SPEED`.
    movement_speed: float = 1.0
    patch_gutter: int = 2

    #: Infidelity per atom per carriage, from heating and imperfect handoff. Unlike the
    #: dephasing charged over flight time this does *not* shrink as T2 improves, so it is
    #: the term that keeps movement visible on a well-decoupled machine -- flight dephasing
    #: alone is a few tenths of a percent of the budget at T2 = 1.5 s. Charged only to the
    #: atoms actually carried; atoms flown *over* are not modelled. Error-based, so
    #: `ScaledNoiseModel` scales it with alpha like p1/p2/p_meas.
    move_error: float = BASE_MOVE_ERROR

    @classmethod
    def neutral_atom(cls, **overrides) -> "StimNoiseParameters":
        """The defaults, named explicitly. Kept so call sites can be self-documenting."""
        return cls(**{**NEUTRAL_ATOM, **overrides})

    @classmethod
    def superconducting(cls, **overrides) -> "StimNoiseParameters":
        """The pre-transport-model defaults. Atoms cannot move; `move_error` is off."""
        return cls(**{**SUPERCONDUCTING, "move_error": 0.0, **overrides})

    @classmethod
    def paper_table_2(cls, **overrides) -> "StimNoiseParameters":
        """What `ScaledNoiseModel(alpha=1.0)` used to mean."""
        return cls(**{**PAPER_TABLE_2, "move_error": 0.0, **overrides})


def _attach_stim_params(model: NoiseModel, params: StimNoiseParameters) -> NoiseModel:
    setattr(model, "_stim_params", params)
    return model


def stim_params_from_noise_model(model: Optional[NoiseModel]) -> StimNoiseParameters:
    if model is not None:
        params = getattr(model, "_stim_params", None)
        if isinstance(params, StimNoiseParameters):
            return params
    return StimNoiseParameters()


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


def StochasticNoiseModel() -> NoiseModel:
    """
    A deliberately harsh machine, for stress-testing rather than for quoting.

    Rates are an order of magnitude worse than `NEUTRAL_ATOM` on purpose -- this is what
    `StochasticSimulator` installs when a test wants errors to actually show up in a few
    hundred shots. The *times* are the neutral-atom ones, so it is the same architecture
    run badly rather than a different machine; it used to carry superconducting timings,
    which meant a "stress" run also silently changed the idle/transport balance.
    """
    nm = NoiseModel()
    oneq_ops = list(TRANSLATED_ONE_QUBIT_OPS)
    twoq_ops = ["cx", "cz", "swap"]

    p1, p2 = 5e-3, 2e-2
    p_meas = 0.05

    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), oneq_ops)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), twoq_ops)

    ro = ReadoutError([[1 - p_meas, p_meas], [p_meas, 1 - p_meas]])
    nm.add_all_qubit_readout_error(ro)

    params = StimNoiseParameters(
        one_qubit_error=p1,
        two_qubit_error=p2,
        readout_error=p_meas,
        # Same machine run badly: every *error* is an order of magnitude worse,
        # including the handling error; the times stay neutral-atom.
        move_error=10 * BASE_MOVE_ERROR,
    )

    return _attach_stim_params(nm, params)


def ScaledNoiseModel(
    alpha: float,
    *,
    base_p1: float = NEUTRAL_ATOM["one_qubit_error"],
    base_p2: float = NEUTRAL_ATOM["two_qubit_error"],
    base_p_meas: float = NEUTRAL_ATOM["readout_error"],
    base_T1: float = NEUTRAL_ATOM["T1"],
    base_T2: float = NEUTRAL_ATOM["T2"],
    t_1q: float = NEUTRAL_ATOM["one_qubit_gate_time"],
    t_2q: float = NEUTRAL_ATOM["two_qubit_gate_time"],
    t_meas: float = NEUTRAL_ATOM["meas_time"],
    base_move_error: float = BASE_MOVE_ERROR,
    scale_readout: bool = True,
    oneq_ops: Optional[list[str]] = None,
    twoq_ops: Optional[list[str]] = None,
) -> NoiseModel:
    """
    Rebuild a Qiskit Aer NoiseModel whose severity increases with `alpha`.

    alpha = 0  -> noiseless (empty model)
    alpha = 1  -> the baseline hardware, which is now `NEUTRAL_ATOM`
    alpha > 1  -> linearly increases depolarizing + readout errors and shortens T1/T2.

    **Everything error-based scales with alpha, including `move_error`.** It did not
    always: `move_error` used to sit at its 1e-3 default at every alpha, which broke the
    per-circuit calibration in `experiments.shared.simulation` — selection rate is
    modelled there as exponential in alpha, and a constant term makes it flatten instead.
    On large benchmark circuits the movement term dominated at the small calibrated
    alphas, the log-space correction stalled, and baselines landed at LSR ~0.55-0.64
    against a 0.12 target — which compressed every relative improvement built on them.
    Times and geometry (gate times, T-times aside, movement speed, pitch) are machine
    shape, not error, and do not scale.

    **The baseline moved.** `alpha = 1` used to be the paper's Table 2 exactly, and every
    preset in `simulator.config_presets` calls this with `alpha=1.0`, so this one default
    change relocates every simulation in the repo. That is intended -- the compiler models
    a neutral-atom machine and Table 2 is a superconducting one, which cannot move atoms
    at all -- but it means alpha solved by `experiments.shared.simulation` against the old
    baseline no longer means the same thing. Pass `**PAPER_TABLE_2`-derived values, or use
    `StimNoiseParameters.paper_table_2()`, to reproduce a pre-switch number.

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
        # Stim backend doesn't fall back to nonzero default parameters. `move_error`
        # is zeroed explicitly: it used to ride through on its dataclass default, so a
        # "noiseless" run with a layout still depolarized every carried atom at 1e-3.
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
            move_error=0.0,
        )
        return _attach_stim_params(nm, params)

    if oneq_ops is None:
        oneq_ops = list(TRANSLATED_ONE_QUBIT_OPS)
    if twoq_ops is None:
        # `swap` is charged like a gate here. The Stim backend prices it as an atom
        # exchange (movement) instead; Aer has no movement channel, and free swaps made
        # every logical H and every routing swap costless.
        twoq_ops = ["cx", "cz", "swap"]

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
        meas_time=t_meas,
        readout_error=p_meas,
        move_error=min(base_move_error * alpha, 1.0),
    )

    return _attach_stim_params(nm, params)


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
