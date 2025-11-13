"""
experiments.error_analysis.calibration_sweep
============================================

Utility helpers for studying how the Iceberg calibration metrics converge as we
increase the number of random logical states (trials) per gate.

Typical usage::

    from experiments.error_analysis.calibration_sweep import (
        sweep_gate_vs_states,
        plot_tvd_lsr_vs_states,
    )

    results = sweep_gate_vs_states("x", [4, 8, 16, 32])
    plot_tvd_lsr_vs_states(results)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from packer.calibrate import GateCalibrationConfig, GateCalibrator


def sweep_gate_vs_states(
    gate: str,
    trial_counts: Sequence[int],
    *,
    base_config: Optional[GateCalibrationConfig] = None,
) -> List[dict]:
    """
    Calibrate ``gate`` repeatedly while sweeping the number of random logical
    states (``trial_counts``).  Each element in ``trial_counts`` becomes the
    ``trials_per_gate`` value for that sweep.

    Returns a list of dicts containing the summary statistics for each sweep,
    augmented with the number of states used.
    """
    if not trial_counts:
        return []

    config_template = base_config or GateCalibrationConfig()
    results: List[dict] = []

    for count in trial_counts:
        count = int(count)
        if count <= 0:
            raise ValueError("trial_counts entries must be positive integers.")

        cfg = replace(config_template, trials_per_gate=count)
        calibrator = GateCalibrator(cfg)
        summary = calibrator.calibrate_gate(gate)
        stats = summary.stats()
        tvd_values = summary.tvd_values()
        stats["tvd_max"] = max(tvd_values) if tvd_values else 0.0
        stats["tvd_mean"] = stats["tvd_max"]
        stats["states"] = count
        stats["gate"] = gate
        stats["num_samples"] = len(summary.samples)
        results.append(stats)

    return results


def plot_tvd_lsr_vs_states(
    results: Iterable[dict],
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot TVD/LSR means vs. the number of calibrated states.
    """
    data = sorted(results, key=lambda r: r["states"])
    if not data:
        raise ValueError("No results provided.")

    states = [entry["states"] for entry in data]
    tvd = [entry.get("tvd_max", entry.get("tvd_mean", 0.0)) for entry in data]
    lsr = [entry["lsr_mean"] for entry in data]

    if ax is None:
        _, ax = plt.subplots()

    # TVD on left axis
    ax.plot(states, tvd, marker="o", color="tab:red", label="TVD max")
    ax.set_xlabel("Number of calibrated states")
    ax.set_ylabel("TVD", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))

    # LSR on right axis
    ax2 = ax.twinx()
    ax2.plot(states, lsr, marker="s", color="tab:blue", label="LSR mean")
    ax2.set_ylabel("Logical selection rate", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))

    gate_name = data[0].get("gate", "")
    ax.set_title(title or f"TVD/LSR convergence for gate '{gate_name}'")

    # Build a combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    ax.grid(True, which="both", alpha=0.2)
    return ax


__all__ = [
    "sweep_gate_vs_states",
    "plot_tvd_lsr_vs_states",
]

if __name__ == "__main__":
    default_counts = [1, 2, 3, 4, 5]
    gate = "x"
    sweep_results = sweep_gate_vs_states(gate, default_counts)
    print(f"Calibration sweep for gate '{gate}'")
    for entry in sweep_results:
        print(
            f"  states={entry['states']:<3} "
            f"TVD={entry.get('tvd_max', entry['tvd_mean']):.8f} "
            f"LSR={entry['lsr_mean']:.8f} "
            f"expected_error={entry['expected_error']:.8f}"
        )
    plot_tvd_lsr_vs_states(sweep_results)
    plt.show()
