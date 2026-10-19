"""
noise.py -- `tvd_explore.fastframes.rescale`, on the circuit's text.

Same exact rescaling rules (see fastframes' docstring), applied by rewriting each noise channel's
arguments in the Stim program text and parsing it once, instead of rebuilding the circuit one
instruction at a time from Python -- which costs ~1 ms per instruction on these circuits (17 s for
a 17k-instruction concatenated circuit, per alpha). `check()` compares against `rescale`.
"""

from __future__ import annotations

import re

_CHANNEL = re.compile(r"^(DEPOLARIZE1|DEPOLARIZE2|X_ERROR|Y_ERROR|Z_ERROR|PAULI_CHANNEL_1)"
                      r"\(([^)]*)\)", re.MULTILINE)


def _scaled(name: str, args, alpha: float):
    if name in ("DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"):
        return [min(alpha * args[0], 0.75 if name == "DEPOLARIZE1" else 1.0)]
    if name == "DEPOLARIZE2":
        return [min(alpha * args[0], 15 / 16)]
    px, py, pz = args
    a = 1 - (1 - min(4 * px, 1.0)) ** alpha
    b = 1 - (1 - min(2 * (px + pz), 1.0)) ** alpha
    return [a / 4, a / 4, max(b / 2 - a / 4, 0.0)]


def rescale_text(text: str, alpha: float):
    import stim

    if alpha == 1.0:
        return stim.Circuit(text)

    def sub(match):
        name = match.group(1)
        args = [float(x) for x in match.group(2).split(",")]
        return f"{name}({', '.join(repr(v) for v in _scaled(name, args, alpha))})"

    return stim.Circuit(_CHANNEL.sub(sub, text))


def check(noisy, alpha: float) -> float:
    """Largest absolute argument difference against `fastframes.rescale`."""
    from experiments.tvd_explore.fastframes import rescale

    slow = rescale(noisy, alpha)
    fast = rescale_text(str(noisy), alpha)
    worst = 0.0
    for left, right in zip(slow.flattened(), fast.flattened()):
        assert left.name == right.name and left.targets_copy() == right.targets_copy()
        for x, y in zip(left.gate_args_copy(), right.gate_args_copy()):
            worst = max(worst, abs(x - y))
    assert len(slow) == len(fast)
    return worst
