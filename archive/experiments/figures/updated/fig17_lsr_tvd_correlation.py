"""
fig17_lsr_tvd_correlation.py — arXiv Figure 17, rebuilt.

"Logical selection rate and total variation distance are strongly anticorrelated
across parameter regimes." Four panels, each sweeping one parameter: patch count,
qubit count, repetitions, and noise scale alpha.

The claim matters because LSR is observable on hardware and TVD is not: if they
track each other, LSR is a usable proxy for output quality. The rebuild adds a fifth
sweep the original could not run -- code size -- since every panel now works at any
k.

Simulation-bound. Defaults are modest; raise `--permutations` and `--circuits` for a
publication run.

    python -m experiments.figures.updated.fig17_lsr_tvd_correlation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from experiments.figures.updated.common import (
    OUTPUT_DIR,
    optimized_placement,
    quiet,
    random_circuit,
    save,
    sequential_packing,
    use_paper_style,
)
from experiments.figures.updated.simulation import SimSettings, build_config, evaluate

#: panel key -> (axis label, the values swept)
SWEEPS = {
    "patches": ("Patch count", None),
    "qubits": ("Logical qubit count", (6, 8, 10, 12)),
    "repetitions": ("Repetitions", (20_000, 50_000, 100_000)),
    "alpha": ("Noise scalar $\\alpha$", (0.02, 0.05, 0.1, 0.2)),
    "code_size": ("Logical qubits per patch, $k$", (2, 4, 6)),
}


def sample_points(
    sweep: str,
    value,
    settings: SimSettings,
    *,
    base_qubits: int,
    base_alpha: float,
    base_reps: int,
    base_k: int,
    permutations: int,
    seed: int,
) -> List[Dict]:
    """Simulate several placements at one point of one sweep."""
    import random

    num_qubits = value if sweep == "qubits" else base_qubits
    alpha = value if sweep == "alpha" else base_alpha
    repetitions = value if sweep == "repetitions" else base_reps
    k = value if sweep == "code_size" else base_k
    n_code = k + 2

    # Clifford-only: the Stim backend this simulates through rejects Rx/Rz.
    circuit = random_circuit(num_qubits, 10, seed=seed, clifford_only=True)
    tightest = max(1, -(-num_qubits // k))
    patches = value if sweep == "patches" else tightest

    rng = random.Random(seed * 31 + 5)
    slots = [(p, s) for p in range(patches) for s in range(k)][:num_qubits]
    candidates = [optimized_placement(circuit, n_code, num_patches=patches),
                  sequential_packing(num_qubits, k)]
    for _ in range(max(0, permutations - len(candidates))):
        shuffled = slots[:]
        rng.shuffle(shuffled)
        candidates.append(shuffled)

    circuit.measure_all()

    rows: List[Dict] = []
    for placement in candidates:
        result = evaluate(
            circuit, placement,
            build_config(settings, n_code=n_code, alpha=alpha,
                         repetitions=repetitions, h_commutation=True, merging=True),
        )
        if result is None:
            continue
        rows.append({
            "sweep": sweep, "value": float(value if value is not None else patches),
            "lsr": result["lsr"], "tvd": result["tvd"],
            "num_qubits": num_qubits, "k": k, "patches": patches,
        })
    return rows


def collect(
    sweeps: List[str], permutations: int, circuits: int, settings: SimSettings
) -> pd.DataFrame:
    rows: List[Dict] = []
    base = dict(base_qubits=10, base_alpha=0.1, base_reps=20_000, base_k=2)

    for sweep in sweeps:
        _, values = SWEEPS[sweep]
        if values is None:  # patch count depends on the base qubit count
            k = base["base_k"]
            tightest = max(1, -(-base["base_qubits"] // k))
            values = tuple(range(tightest, base["base_qubits"] + 1, 2))
        for value in tqdm(values, desc=f"  {sweep}"):
            for trial in range(circuits):
                rows.extend(
                    sample_points(
                        sweep, value, settings, permutations=permutations,
                        seed=trial * 17 + 3, **base
                    )
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    sweeps = [s for s in SWEEPS if (df.sweep == s).any()]
    columns = 2
    rows_n = -(-len(sweeps) // columns)
    fig, axes = plt.subplots(
        nrows=rows_n, ncols=columns, figsize=(9.5, 3.6 * rows_n), squeeze=False
    )
    flat = [ax for row in axes for ax in row]

    for ax, sweep in zip(flat, sweeps):
        sub = df[df.sweep == sweep]
        values = sorted(sub.value.unique())
        pearson, spearman = [], []
        for value in values:
            at = sub[sub.value == value]
            if len(at) < 3 or at.lsr.nunique() < 2 or at.tvd.nunique() < 2:
                pearson.append(np.nan)
                spearman.append(np.nan)
                continue
            pearson.append(pearsonr(at.lsr, at.tvd)[0])
            spearman.append(spearmanr(at.lsr, at.tvd)[0])
        ax.plot(values, pearson, marker="o", color="#3b4c56", label="Pearson $r$")
        ax.plot(values, spearman, marker="s", color="#0097fd", label=r"Spearman $\rho$")
        ax.axhline(0.0, color="#636363", linestyle="--", linewidth=0.9)
        ax.set_xlabel(SWEEPS[sweep][0])
        ax.set_ylabel("LSR-TVD correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=11)

    for ax in flat[len(sweeps):]:
        ax.set_visible(False)
    fig.suptitle(
        "Logical selection rate is anticorrelated with total variation distance",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    parser.add_argument("--permutations", type=int, default=8)
    parser.add_argument("--circuits", type=int, default=2)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    name = "fig17_lsr_tvd_correlation"
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.sweeps, args.permutations, args.circuits, SimSettings())
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    if df.empty:
        raise RuntimeError("no data collected")
    plot(df, name)
    print("\n[fig17] pooled correlation per sweep")
    for sweep in df.sweep.unique():
        sub = df[df.sweep == sweep]
        if sub.lsr.nunique() < 2 or sub.tvd.nunique() < 2:
            continue
        print(
            f"  {sweep:12s} pearson={pearsonr(sub.lsr, sub.tvd)[0]:+.3f} "
            f"spearman={spearmanr(sub.lsr, sub.tvd)[0]:+.3f}  ({len(sub)} points)"
        )


if __name__ == "__main__":
    main()
