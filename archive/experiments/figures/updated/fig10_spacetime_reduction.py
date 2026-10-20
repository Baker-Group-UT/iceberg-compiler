"""
fig10_spacetime_reduction.py — arXiv Figure 10, rebuilt.

"Our framework enables significant resource reduction for 71 QASMBench circuits."
One stacked bar per benchmark, split into H commutation / + merging / + packing,
grouped by size, with an average bar on the right.

What differs from the original
------------------------------
Placement is the `grouping` stack rather than the pairwise packer, merging runs at
every code size, and placements are scored analytically. At the paper's k=2 the
first two change little; the point of rebuilding is that the same script now runs at
any k, which `--n-code` exposes.

Gate counts and depth only -- no simulation, so this is minutes rather than hours.

    python -m experiments.figures.updated.fig10_spacetime_reduction
    python -m experiments.figures.updated.fig10_spacetime_reduction --n-code 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.figures.updated.common import (
    COMPONENTS,
    GATE_METRICS,
    METRICS,
    OUTPUT_DIR,
    SIZE_GROUPS,
    ablate,
    component_legend,
    stack_heights,
    load_group,
    quiet,
    save,
    use_paper_style,
)
from iceberg.utils import logical_capacity


def collect(n_code: int, groups: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for group in groups:
        circuits = load_group(group)
        if not circuits:
            print(f"[fig10] no circuits for {group}")
            continue
        print(f"[fig10] {group}: {len(circuits)} circuits")
        for name, circuit in tqdm(circuits, desc=f"  {group}"):
            result = ablate(circuit, n_code)
            if result is None:
                print(f"[fig10] skipped {name}: placement rejected")
                continue
            rows.extend(
                result.rows(
                    size_group=group,
                    circuit_name=name,
                    num_qubits=circuit.num_qubits,
                    n_code=n_code,
                )
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, n_code: int, name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    groups = [g for g in SIZE_GROUPS if (df["size_group"] == g).any()]
    fig, axes = plt.subplots(
        nrows=len(GATE_METRICS), ncols=1, figsize=(19, 9.5), sharex=True
    )

    # One ordering for every row, so the shared x labels stay correct. Sorted by
    # total 2q reduction, which is the metric the paper orders on.
    order: Dict[str, List[str]] = {}
    for group in groups:
        sub = df[df.size_group == group]
        totals = {
            circuit: sum(
                stack_heights(sub[sub.circuit_name == circuit], METRICS["2q"][0])
            )
            for circuit in sub.circuit_name.unique()
        }
        order[group] = sorted(totals, key=totals.get)

    positions_all: List[float] = []
    labels_all: List[str] = []
    for ax_index, (ax, key) in enumerate(zip(axes, GATE_METRICS)):
        column = METRICS[key][0]
        cursor = 0.0
        positions_all, labels_all = [], []
        for group in groups:
            names = order[group]
            sub = df[df.size_group == group]
            per_circuit = {
                circuit: stack_heights(sub[sub.circuit_name == circuit], column)
                for circuit in names
            }
            xs = np.arange(len(names)) + cursor
            bottom = np.zeros(len(names))
            for index, (component, label, colour) in enumerate(COMPONENTS):
                values = np.array([per_circuit[c][index] for c in names])
                ax.bar(xs, values, width=0.78, bottom=bottom, color=colour,
                       edgecolor="white", linewidth=0.4, label=label)
                bottom += values
            positions_all.extend(xs.tolist())
            labels_all.extend(names)
            cursor = xs[-1] + 4.0 if len(xs) else cursor + 4.0

        # Overall average, as its own wider bar.
        avg_x = cursor + 1.0
        bottom = 0.0
        for value, (component, label, colour) in zip(
            stack_heights(df, column), COMPONENTS
        ):
            ax.bar(avg_x, value, width=3.6, bottom=bottom, color=colour,
                   edgecolor="white", linewidth=0.4)
            bottom += value
        ax.axvline(cursor - 2.0, color="#bdbdbd", linestyle=":", linewidth=1.0)
        positions_all.append(avg_x)
        labels_all.append("average")

        ax.set_ylabel(METRICS[key][1])
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if ax_index == 0:
            component_legend(ax, loc="upper left")

    axes[-1].set_xticks(positions_all)
    axes[-1].set_xticklabels(labels_all, rotation=60, ha="right", fontsize=8)
    axes[-1].set_xlabel("Benchmark circuit (grouped by size)")
    fig.supylabel("Percent reduction")
    k = logical_capacity(n_code)
    fig.suptitle(
        f"Resource reduction on QASMBench, $[[{n_code},\\,{k},\\,2]]$ code",
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.985))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--groups", nargs="+", default=SIZE_GROUPS, choices=SIZE_GROUPS)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    name = f"fig10_spacetime_reduction_n{args.n_code}"
    csv = OUTPUT_DIR / f"{name}.csv"
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.n_code, args.groups)
        if df.empty:
            raise RuntimeError("no data collected")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"[fig10] wrote {csv}")

    plot(df, args.n_code, name)
    print("\n[fig10] mean reduction by size group (clipped at zero, as plotted)")
    for group in SIZE_GROUPS:
        sub = df[df.size_group == group]
        if sub.empty:
            continue
        parts = [
            f"{METRICS[key][1]}={sum(stack_heights(sub, METRICS[key][0])):5.1f}%"
            for key in GATE_METRICS
        ]
        print(f"  {group:7s} {'  '.join(parts)}  ({sub.circuit_name.nunique()} circuits)")
    parts = [
        f"{METRICS[key][1]}={sum(stack_heights(df, METRICS[key][0])):5.1f}%"
        for key in GATE_METRICS
    ]
    print(f"  {'overall':7s} {'  '.join(parts)}")


if __name__ == "__main__":
    main()
