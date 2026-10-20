"""
Seeding must not cost shots.

`Simulator` splits a run into `repetitions / _SHOT_BAR_TARGET_UPDATES` chunks so the
progress bar can move. Aer's `seed_simulator` seeds a *run*, not a simulator object, so
constructing `AerSimulator(seed_simulator=s)` once and calling `sim.run(...)` per chunk
replayed the first chunk byte for byte. Counts summed to `repetitions` while carrying
one chunk's worth of independent samples -- 1% of the requested shots at the default
chunking, invisible in the output.

The consequence was not a small bias. A 4000-shot seeded run returned probabilities
quantised to 1/40, so any sampling-error bound derived from the nominal shot count was
too tight by a factor of ten, and comparisons between two such runs showed TVDs around
0.2 with no cause but noise.

These tests pin the three properties that matter: every shot counts, a seed still
reproduces a run exactly, and neighbouring seeds are independent.
"""

from __future__ import annotations

import sys
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional

from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.simulator import Simulator, SimulatorConfig

#: Enough shots that the default chunking splits the run into many chunks, which is the
#: condition the bug needed. `_shot_chunk_size_for` targets 100 updates, so this is 100
#: chunks of 40 shots.
SHOTS = 4_000
CHUNKS = Simulator._SHOT_BAR_TARGET_UPDATES

#: Uniform over 8 outcomes, so every count is well away from zero at 4000 shots and the
#: greatest common divisor below is a meaningful measurement rather than an accident.
def uniform_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    for qubit in range(3):
        circuit.h(qubit)
    circuit.measure_all()
    return circuit


def ideal_distribution(seed: Optional[int], shots: int = SHOTS) -> Dict[str, float]:
    simulator = Simulator(
        SimulatorConfig(repetitions=shots, noiseless=True, seed=seed)
    ).with_shot_progress(False)
    return simulator.ideal(uniform_circuit()).ideal_dist


def shot_counts(distribution: Dict[str, float], shots: int) -> List[int]:
    return [round(probability * shots) for probability in distribution.values()]


def test_seeded_run_uses_every_shot_not_just_the_first_chunk():
    """
    The granularity of the returned probabilities must match the shots requested.

    If each chunk replays the first, every count is exactly `CHUNKS` copies of that
    chunk's count, so `gcd` over the counts is a multiple of `CHUNKS`. Independent
    chunks leave no such common factor.
    """
    counts = shot_counts(ideal_distribution(seed=7), SHOTS)
    assert sum(counts) == SHOTS
    common = reduce(gcd, counts)
    assert common < CHUNKS, (
        f"counts share a factor of {common} with {CHUNKS} chunks in the run, so only "
        f"{SHOTS // common} of {SHOTS} shots were independent"
    )


def test_sampling_error_matches_the_shot_count_requested():
    """
    The practical symptom: the error has to be the one the shot count promises.

    Note that "error shrinks as shots grow" is *not* the property to test. Chunk size
    is `repetitions / 100`, so under the bug the effective sample count grew with
    `repetitions` too -- the error did shrink, just from ten times too high a floor. It
    has to be checked against an absolute expectation instead.

    For a uniform distribution over 8 outcomes the binomial standard error on each
    probability is `sqrt(p(1-p)/N)`. At 40000 shots that is 0.00165, so 5 sigma is
    0.0083. Under the bug the effective N was 400, putting the same bound at 0.0165 and
    the observed maximum around 0.03.
    """
    shots = 40_000
    probability = 0.125
    sigma = (probability * (1.0 - probability) / shots) ** 0.5

    distribution = ideal_distribution(seed=11, shots=shots)
    worst = max(abs(value - probability) for value in distribution.values())
    assert worst < 5.0 * sigma, (
        f"worst deviation {worst:.5f} exceeds 5 sigma ({5.0 * sigma:.5f}) for "
        f"{shots} shots, so the run did not use {shots} independent samples"
    )


def test_a_seed_still_reproduces_a_run_exactly():
    assert ideal_distribution(seed=7) == ideal_distribution(seed=7)


def test_neighbouring_seeds_are_independent():
    """
    Chunk seeds come from a PRNG keyed on the base seed, not from `base + index`.

    Under `base + index` a run seeded 7 and a run seeded 8 would share every chunk but
    the first and last, so a sweep over consecutive seeds would resample almost the
    same shots each time and understate its own variance.
    """
    first, second = ideal_distribution(seed=7), ideal_distribution(seed=8)
    assert first != second
    outcomes = set(first) | set(second)
    spread = sum(
        abs(first.get(key, 0.0) - second.get(key, 0.0)) for key in outcomes
    ) / len(outcomes)
    assert spread > 1e-4, f"seeds 7 and 8 agree to {spread:.6f}; streams look shared"


def test_unseeded_runs_still_vary():
    assert ideal_distribution(seed=None) != ideal_distribution(seed=None)
