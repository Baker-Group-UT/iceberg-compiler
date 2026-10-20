"""
fig11_density_sweep.py — arXiv Figure 11, rebuilt.

"Packing and merging improvement over random permutations by packing density."
Two panels: (a) fixed depth, varying qubit count; (b) fixed qubit count, varying
depth. Each point compares the compiler's placement against the mean of several
random permutations at the same patch count.

Packing density is `qubits / patches`, so density k means every patch is full and
density 1 means every qubit is solo. The original swept 2.0 down to 1.0 because k
was 2; here the top of the range is k, which is the generalization the figure gains.

Gate counts only. The random-permutation baseline is what makes this expensive --
`--random-trials` controls it -- but scoring is analytic, so it is still minutes.

    python -m experiments.figures.updated.fig11_density_sweep
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.figures.updated.common import (
    GATE_METRICS,
    METRICS,
    OUTPUT_DIR,
    config_for,
    measure,
    optimized_placement,
    pct_reduction,
    quiet,
    random_circuit,
    save,
    use_paper_style,
)
from iceberg.utils import logical_capacity

PANELS = {
    "qubits": {
        "title": "(a) Fixed depth (10 gates per qubit), varying qubit count",
        "xlabel": "Packing density (logical qubits per patch)",
    },
    "gates": {
        "title": "(b) Fixed qubit count (50), varying gates per qubit",
        "xlabel": "Packing density (logical qubits per patch)",
    },
}


def random_permutation(num_qubits: int, patches: int, k: int, rng) -> List:
    slots = [(p, s) for p in range(patches) for s in range(k)][: num_qubits]
    rng.shuffle(slots)
    return slots


def sweep_one(
    circuit,
    n_code: int,
    random_trials: int,
    rng,
) -> List[Dict]:
    """Compare the placer against random permutations across every patch count."""
    k = logical_capacity(n_code)
    config = config_for(n_code, h_commutation=True, merging=True)
    tightest = max(1, -(-circuit.num_qubits // k))
    rows: List[Dict] = []

    for patches in range(tightest, circuit.num_qubits + 1):
        density = circuit.num_qubits / patches
        placed = measure(
            circuit, optimized_placement(circuit, n_code, num_patches=patches), config
        )
        if placed is None:
            continue
        samples = []
        for _ in range(random_trials):
            metrics = measure(
                circuit, random_permutation(circuit.num_qubits, patches, k, rng), config
            )
            if metrics is not None:
                samples.append(metrics.as_tuple())
        if not samples:
            continue
        reference = np.mean(np.array(samples), axis=0)
        row = {"density": density, "patches": patches, "n_code": n_code,
               "num_qubits": circuit.num_qubits}
        for key, ref, got in zip(GATE_METRICS, reference, placed.as_tuple()):
            row[METRICS[key][0]] = pct_reduction(ref, got)
        rows.append(row)
    return rows


def collect(n_code: int, circuits_per_point: int, random_trials: int) -> pd.DataFrame:
    rows: List[Dict] = []
    rng = random.Random(20260727)

    for label, qubit_counts, gates_per_qubit in (
        ("qubits", (10, 20, 30, 40, 50), (10,)),
        ("gates", (30,), (2, 5, 10, 15, 20)),
    ):
        for num_qubits in qubit_counts:
            for depth in gates_per_qubit:
                for trial in tqdm(
                    range(circuits_per_point),
                    desc=f"  {label} q={num_qubits} g={depth}",
                    leave=False,
                ):
                    circuit = random_circuit(num_qubits, depth, seed=trial * 31 + num_qubits)
                    for row in sweep_one(circuit, n_code, random_trials, rng):
                        row.update(panel=label, gates_per_qubit=depth, trial=trial)
                        rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, n_code: int, name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    k = logical_capacity(n_code)
    panels = [p for p in PANELS if (df["panel"] == p).any()]
    fig, axes = plt.subplots(
        nrows=len(panels), ncols=1, figsize=(7.2, 4.2 * len(panels))
    )
    axes = np.atleast_1d(axes)

    colours = {"1q": "#3b4c56", "2q": "#0097fd", "depth": "#c0736a"}
    markers = {"1q": "o", "2q": "s", "depth": "^"}

    for ax, panel in zip(axes, panels):
        sub = df[df.panel == panel]
        # Bin densities so points from different qubit counts line up.
        bins = np.linspace(1.0, k, 8)
        sub = sub.assign(bin=pd.cut(sub.density, bins=bins, include_lowest=True))
        for key in GATE_METRICS:
            grouped = sub.groupby("bin", observed=True)[METRICS[key][0]].mean()
            centres = [interval.mid for interval in grouped.index]
            ax.plot(centres, grouped.to_numpy(), marker=markers[key],
                    color=colours[key], label=METRICS[key][1], linewidth=1.6,
                    markersize=5)
        ax.axhline(0, color="#636363", linestyle="--", linewidth=0.9)
        ax.set_title(PANELS[panel]["title"], fontsize=13)
        ax.set_xlabel(PANELS[panel]["xlabel"])
        ax.set_ylabel("Percent reduction")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=12)

    fig.suptitle(
        f"Improvement over random permutations, $[[{n_code},\\,{k},\\,2]]$ code",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--circuits", type=int, default=4)
    parser.add_argument("--random-trials", type=int, default=12)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    name = f"fig11_density_sweep_n{args.n_code}"
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.n_code, args.circuits, args.random_trials)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    if df.empty:
        raise RuntimeError("no data collected")
    plot(df, args.n_code, name)
    print("\n[fig11] mean improvement at the tightest and loosest densities")
    for panel in df.panel.unique():
        sub = df[df.panel == panel]
        tight = sub[sub.density >= sub.density.max() - 0.1]
        loose = sub[sub.density <= sub.density.min() + 0.1]
        for label, part in (("tightest", tight), ("loosest", loose)):
            parts = [
                f"{METRICS[key][1]}={part[METRICS[key][0]].mean():5.1f}%"
                for key in GATE_METRICS
            ]
            print(f"  {panel:7s} {label:9s} {'  '.join(parts)}")


if __name__ == "__main__":
    main()
