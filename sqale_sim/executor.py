"""executor.py – Monte‑Carlo benchmarking helpers and batch runner

Updated 2025‑07‑11
------------------
* Fixed import paths (simulation, postselection, state_prep, fidelity, IcebergGrid).
* Added robust *interactive plotting* handling in `run_sim_groups`:
    – If `show=True` we attempt to switch to an interactive backend
      (TkAgg → Qt5Agg) when the current one is non‑GUI (e.g. Agg).
    – Figures are displayed **non‑blocking** with `fig.show()` and
      `plt.pause(0.1)`, so the loop continues to the next simulation.
    – If no GUI backend is available we fall back to saving PNGs only and
      warn the user.
* `run_sim_groups` signature now includes `show: bool = False`.
"""
from __future__ import annotations


from simulation import run_sqale_simulation
from postselection import iceberg_post_select, ordered_measurement_keys
from state_prep import prepare_random_basis_state
from fidelity import bhattacharyya

import time
from pathlib import Path
import concurrent.futures as cf
from typing import Callable, List, Tuple, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cirq
import experiments_superstaq as ess

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from iceberg.IcebergCode import IcebergGrid

__all__ = ["logical_error_rate", "run_sim_groups"]

# -----------------------------------------------------------------------------
#  Single‑gate logical error rate  --------------------------------------------
# -----------------------------------------------------------------------------

def _one_state(args) -> float:
    gate_func, shots, seed = args
    rng = np.random.default_rng(seed)

    circuit = cirq.Circuit()
    grid = IcebergGrid(n_program=4, n=4, circuit=circuit)

    grid.encode()
    prepare_random_basis_state(grid, rng)
    gate_func(grid)
    grid.decode()
    grid.measure()

    ref_dist = ess.simulation.leakage_sim.simulate_true_distribution(
        circuit, dimension=2
    )
    keys = ordered_measurement_keys(circuit)

    sim = run_sqale_simulation(circuit, repetitions=shots)
    return bhattacharyya(
        iceberg_post_select(ref_dist, keys),
        iceberg_post_select(sim.post_selected_dist, keys),
    )


def logical_error_rate(
    label: str,
    gate_func: Callable[[IcebergGrid], None],
    *,
    N_states: int = 4_000,
    shots: int = 20,
    batch: int = 200,
) -> float:
    """Estimate average logical‑error rate for *gate_func* over random states."""
    start = time.time()
    seeds = np.random.SeedSequence().spawn(N_states)
    tasks = [(gate_func, shots, s.generate_state(1)[0]) for s in seeds]

    with cf.ProcessPoolExecutor() as pool:
        fids = []
        for i in range(0, N_states, batch):
            fids.extend(pool.map(_one_state, tasks[i : i + batch]))

    err = 1.0 - sum(fids) / N_states
    print(
        f"{label:<6s}: logical-error ≈ {err:6.2%}  "
        f"({N_states} states × {shots} shots, {time.time() - start:.1f}s)"
    )
    return err


# -----------------------------------------------------------------------------
#  Batch helper – run & plot gate groups --------------------------------------
# -----------------------------------------------------------------------------

def _ensure_interactive_backend() -> bool:
    """Switch to a GUI backend if the current one is non‑interactive.

    Returns True if an interactive backend is active after the call, else False.
    """
    gui_backends = ["TkAgg", "Qt5Agg", "QtAgg", "MacOSX", "GTK3Agg"]
    current = matplotlib.get_backend()

    if current.lower().endswith("agg"):
        for be in gui_backends:
            try:
                matplotlib.use(be, force=True)
                return True
            except (ImportError, RuntimeError):
                continue
        return False  # no GUI backend available
    return True  # already GUI


def run_sim_groups(
    sim_groups: List[Dict],
    *,
    N_states: int = 1,
    shots: int = 200,
    batch: int = 200,
    out_dir: str | Path = "results",
    show: bool = False,
) -> List[Dict]:
    """Benchmark & visualise *groups* of gates.

    Each element in *sim_groups* must contain::

        {
            "title": "…",
            "gates": [(label, gate_func), …],
            # optional:
            "figsize": (6,4),
            "filename": "custom.png",
        }

    Returns the same list augmented with a ``"results"`` key.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    have_gui = _ensure_interactive_backend() if show else False

    for group in sim_groups:
        title = group["title"]
        gates = group["gates"]
        figsize = group.get("figsize", (6, 4))
        filename = group.get("filename", f"{title.lower().replace(' ', '_')}.png")
        filename = out_dir / filename

        # -- run benchmarks --------------------------------------------------
        results = [
            (lbl, logical_error_rate(lbl, func, N_states=N_states, shots=shots, batch=batch))
            for lbl, func in gates
        ]
        group["results"] = results

        # -- plot ------------------------------------------------------------
        labels, errs = zip(*results)
        plt.figure(figsize=figsize)
        plt.bar(labels, errs)
        plt.ylabel("Logical Error Rate")
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"[saved] {filename}")

        if show and have_gui:
            # non‑blocking display
            plt.show(block=False)
            plt.pause(0.1)  # give the window time to draw
        else:
            plt.close()

    return sim_groups
