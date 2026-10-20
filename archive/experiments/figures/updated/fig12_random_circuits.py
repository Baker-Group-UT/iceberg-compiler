"""
fig12_random_circuits.py — arXiv Figure 12, rebuilt.

"Compilation techniques significantly improve resource overhead for random circuits."
One stacked bar per size class, split into H commutation / + merging / + packing.

Random circuits carry fewer immutable gates (S, Rx, Rz) than the benchmarks, so a
larger share of the circuit is available to merging -- that is the paper's point and
`--immutable-ratio` is the knob that controls it.

Gate counts only; runs in under a minute. `--n-code` sweeps the code size, which the
original could not do because merging was [[4,2,2]]-only.

    python -m experiments.figures.updated.fig12_random_circuits
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.figures.updated.common import (
    COMPONENTS,
    GATE_METRICS,
    METRICS,
    OUTPUT_DIR,
    ablate,
    component_legend,
    quiet,
    stack_heights,
    random_circuit,
    save,
    use_paper_style,
)
from iceberg.utils import logical_capacity

#: label -> (qubits, gates per qubit) ranges, matching the draft's x-axis captions.
SIZE_CLASSES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "Small": ((5, 10), (10, 50)),
    "Medium": ((15, 25), (30, 150)),
    "Large": ((30, 100), (60, 300)),
}


def collect(n_code: int, trials: int, immutable_ratio: float) -> pd.DataFrame:
    import random

    rows: List[Dict] = []
    for label, ((q_lo, q_hi), (g_lo, g_hi)) in SIZE_CLASSES.items():
        rng = random.Random(hash(label) % 10_000 + n_code)
        for trial in tqdm(range(trials), desc=f"  {label}"):
            num_qubits = rng.randint(q_lo, q_hi)
            total_gates = rng.randint(g_lo, g_hi)
            per_qubit = max(1, total_gates // num_qubits)
            circuit = random_circuit(
                num_qubits, per_qubit, seed=trial * 97 + n_code,
                immutable_ratio=immutable_ratio,
            )
            result = ablate(circuit, n_code)
            if result is None:
                continue
            rows.extend(
                result.rows(
                    size_class=label,
                    trial=trial,
                    num_qubits=num_qubits,
                    n_code=n_code,
                )
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, n_code: int, name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    classes = [c for c in SIZE_CLASSES if (df["size_class"] == c).any()]
    fig, axes = plt.subplots(
        nrows=len(GATE_METRICS), ncols=1, figsize=(6.4, 7.6), sharex=True
    )

    for ax, key in zip(axes, GATE_METRICS):
        column = METRICS[key][0]
        xs = np.arange(len(classes))
        bottom = np.zeros(len(classes))
        per_class = {c: stack_heights(df[df.size_class == c], column) for c in classes}
        for index, (component, label, colour) in enumerate(COMPONENTS):
            values = np.array([per_class[c][index] for c in classes])
            ax.bar(xs, values, width=0.62, bottom=bottom, color=colour,
                   edgecolor="white", linewidth=0.6, label=label)
            bottom += values
        ax.set_ylabel(METRICS[key][1])
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    component_legend(axes[0], loc="upper left")
    axes[-1].set_xticks(np.arange(len(classes)))
    axes[-1].set_xticklabels(
        [
            f"{c}\n{SIZE_CLASSES[c][0][0]}-{SIZE_CLASSES[c][0][1]}q, "
            f"{SIZE_CLASSES[c][1][0]}-{SIZE_CLASSES[c][1][1]}g"
            for c in classes
        ]
    )
    k = logical_capacity(n_code)
    axes[-1].set_xlabel(f"Random circuit size   ($[[{n_code},\\,{k},\\,2]]$ code)")
    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--immutable-ratio", type=float, default=0.08)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    name = f"fig12_random_circuits_n{args.n_code}"
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.n_code, args.trials, args.immutable_ratio)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    plot(df, args.n_code, name)
    print("\n[fig12] mean reduction by size class")
    for label in SIZE_CLASSES:
        sub = df[df.size_class == label]
        if sub.empty:
            continue
        parts = [
            f"{METRICS[key][1]}={sum(stack_heights(sub, METRICS[key][0])):5.1f}%"
            for key in GATE_METRICS
        ]
        print(f"  {label:7s} {'  '.join(parts)}")


if __name__ == "__main__":
    main()
