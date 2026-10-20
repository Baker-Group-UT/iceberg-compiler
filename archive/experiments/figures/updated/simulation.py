"""
simulation.py — shared LSR/TVD measurement for the figures that need it.

Gate counts come from translation alone and are fast. Logical selection rate and
total variation distance need the circuit actually simulated, postselected and
compared against an unencoded reference, which is orders of magnitude slower and is
why the figures that use it have smaller defaults.

Two things to know before reading any number this produces:

**Noise has to be calibrated per circuit.** Postselection is an exponentially strict
filter, so a fixed noise scale either keeps everything (and measures nothing) or
keeps nothing (and measures noise). Each circuit gets a pilot run, and alpha is
solved so the baseline selection rate lands near `target_selection_rate`. Same
scheme as `density_ablation_sweep`, which is where it was worked out.

**Seeded runs used to sample less than they claimed.** `Simulator._shot_chunk_size_for`
caps a chunk at `repetitions // 100`, and a single `seed_simulator` on the Aer backend
made every chunk a replay of the first -- `repetitions / 100` independent samples, so
roughly ten times the sampling error the shot count implied. Fixed: chunks now draw
their own seeds from the configured one, pinned by `tests/test_simulator_seeding.py`.
Seeding is safe again, though `seed=None` remains the default here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import qiskit

from pipeline.config import PipelineConfig
from pipeline.tasks.scan_specific import run_specific_scan
from simulator.config_presets import ScaledSimulator

Perm = list


@dataclass(frozen=True)
class SimSettings:
    """Noise calibration and shot budget."""

    alpha_reference: float = 0.1
    reference_gate_count: float = 200.0
    target_selection_rate: float = 0.12
    pilot_repetitions: int = 4_000
    target_kept_shots: int = 20_000
    min_repetitions: int = 20_000
    max_repetitions: int = 400_000
    stim: bool = True
    distance_noise: bool = True
    idle_noise: bool = True
    #: Left at None deliberately -- see the module docstring on seeded runs.
    seed: Optional[int] = None


def build_config(
    settings: SimSettings,
    *,
    n_code: int,
    alpha: float,
    repetitions: int,
    h_commutation: bool,
    merging: bool,
) -> PipelineConfig:
    sim = ScaledSimulator(alpha=alpha, repetitions=repetitions, seed=settings.seed)
    if settings.stim:
        sim = sim.with_stim()
    if settings.distance_noise:
        sim = sim.with_distance_noise(distance_scalar=1.0)
    if settings.idle_noise:
        sim = sim.with_idle_noise(idle_time_scalar=1.0)
    config = PipelineConfig(verbose=False, save=False, sim_config=sim)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    return config


def evaluate(
    circuit: qiskit.QuantumCircuit, permutation: Perm, config: PipelineConfig
) -> Optional[Dict[str, float]]:
    """Simulate one placement and return its LSR and TVD, or None if it failed."""
    try:
        record = run_specific_scan(
            program_circuit=circuit, permutations=[permutation], config=config
        )
    except Exception:  # noqa: BLE001
        return None
    if record.dataframe is None or record.dataframe.df.empty:
        return None
    row = record.dataframe.df.iloc[0]
    return {
        "lsr": float(row["logical_selection_rate"]),
        "tvd": float(row["logical_total_variation_distance"]),
    }


def calibrate(
    circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    settings: SimSettings,
    *,
    n_code: int,
    baseline_gate_count: int,
) -> Tuple[float, int]:
    """
    Pick a noise scale and shot count that put this circuit in a measurable regime.

    Two pilots: one to see where the default alpha lands, one to confirm the
    correction. Selection rate falls exponentially in noise, so the correction is
    solved in log space.
    """
    alpha = settings.alpha_reference * (
        settings.reference_gate_count / max(1.0, float(baseline_gate_count))
    )
    pilot = evaluate(
        circuit,
        permutation,
        build_config(
            settings, n_code=n_code, alpha=alpha,
            repetitions=settings.pilot_repetitions,
            h_commutation=False, merging=False,
        ),
    )
    if pilot is None:
        return alpha, settings.min_repetitions

    observed = float(np.clip(pilot["lsr"], 1e-6, 1 - 1e-6))
    target = float(np.clip(settings.target_selection_rate, 1e-6, 1 - 1e-6))
    alpha = alpha * math.log(target) / math.log(observed)

    confirm = evaluate(
        circuit,
        permutation,
        build_config(
            settings, n_code=n_code, alpha=alpha,
            repetitions=settings.pilot_repetitions,
            h_commutation=False, merging=False,
        ),
    )
    rate = confirm["lsr"] if confirm else settings.target_selection_rate
    repetitions = int(
        np.clip(
            math.ceil(settings.target_kept_shots / max(rate, 1e-4)),
            settings.min_repetitions,
            settings.max_repetitions,
        )
    )
    return alpha, repetitions
