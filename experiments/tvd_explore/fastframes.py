"""
fastframes.py -- convert an encoded circuit to noisy Stim once, at alpha = 1, and rescale.

Conversion (`qiskit_to_stim`: scheduling, transport, idle accounting) is the slow step and
does not depend on alpha: `ScaledNoiseModel` changes error rates only, never gate times or
the schedule. Every rate scales exactly:

    DEPOLARIZE1 / DEPOLARIZE2 / X_ERROR    p -> alpha * p      (p1, p2, readout, move_error)
    PAULI_CHANNEL_1 (twirled thermal)       T1, T2 -> T1/alpha, T2/alpha, so with
                                            a = 4 px = 1 - e^{-t/T1}, b = 2 (px + pz) = 1 - e^{-t/T2}:
                                            a -> 1 - (1 - a)^alpha,  b -> 1 - (1 - b)^alpha

so a circuit converted at alpha = 1 rescales to any alpha with no second conversion.
`check()` compares against a direct conversion.
"""

from __future__ import annotations

from typing import Dict, Iterable


def at_alpha_one(encoded):
    from experiments.revision.exp8_undetected.frames import stim_params
    from experiments.revision.exp8_undetected.gadgets import stim_overrides
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    with sb._active_params(stim_overrides(stim_params(1.0), idle_time_scalar=1.0,
                                          distance_scalar=1.0)):
        layout = positions_for_circuit(encoded, geometry=sb._geometry())
        return sb.qiskit_to_stim(encoded, layout=layout, include_noise=True)


def rescale(noisy, alpha: float):
    import stim

    if alpha == 1.0:
        return noisy
    out = stim.Circuit()
    for op in noisy.flattened():
        name, args = op.name, op.gate_args_copy()
        if name in ("DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"):
            args = [min(alpha * args[0], 0.75 if name == "DEPOLARIZE1" else 1.0)]
        elif name == "DEPOLARIZE2":
            args = [min(alpha * args[0], 15 / 16)]
        elif name == "PAULI_CHANNEL_1":
            px, py, pz = args
            a = 1 - (1 - min(4 * px, 1.0)) ** alpha
            b = 1 - (1 - min(2 * (px + pz), 1.0)) ** alpha
            args = [a / 4, a / 4, max(b / 2 - a / 4, 0.0)]
        out.append(stim.CircuitInstruction(name, op.targets_copy(), args))
    return out


def splits(encoded, alphas: Iterable[float], *, shots: int = 20_000, seed: int = 11) -> Dict[float, Dict]:
    from experiments.revision.exp8_undetected.frames import split_from_stim

    base = at_alpha_one(encoded)
    return {a: split_from_stim(encoded, rescale(base, a), shots=shots, seed=seed) for a in alphas}


def check(encoded, alpha: float) -> float:
    """Largest absolute difference in any noise argument against a direct conversion."""
    from experiments.revision.exp8_undetected.frames import stim_params
    from experiments.revision.exp8_undetected.gadgets import stim_overrides
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    with sb._active_params(stim_overrides(stim_params(alpha), idle_time_scalar=1.0,
                                          distance_scalar=1.0)):
        layout = positions_for_circuit(encoded, geometry=sb._geometry())
        direct = sb.qiskit_to_stim(encoded, layout=layout, include_noise=True)
    scaled = rescale(at_alpha_one(encoded), alpha)
    worst = 0.0
    for left, right in zip(direct.flattened(), scaled.flattened()):
        assert left.name == right.name and left.targets_copy() == right.targets_copy()
        for x, y in zip(left.gate_args_copy(), right.gate_args_copy()):
            worst = max(worst, abs(x - y))
    assert len(direct) == len(scaled)
    return worst
