"""
fig_k_sweep.py — how each technique's benefit moves with the code size. New figure.

The arXiv draft evaluates one code, [[4,2,2]]. Every figure in it is a slice at k=2.
This one sweeps k and shows what the paper could only conjecture: "we expect the
efficacy of these techniques to become more significant as k increases."

Two panels:

*techniques*  stacked bars per k, one group per benchmark size, split into
              H commutation / + merging / + packing. The same decomposition as
              Figure 12 of the draft, swept over the code family.
*resources*   physical qubits and weighted gate cost against k, which is the
              trade-off a machine designer actually faces: a larger code needs
              `patches x (n + 2)` atoms and pays `ENCODE + DECODE` per patch, but
              holds more logical qubits per patch.

The second panel is only affordable because placements are now scored analytically
-- it evaluates every patch count for every circuit at every k.

    python -m experiments.figures.updated.fig_k_sweep
    python -m experiments.figures.updated.fig_k_sweep --groups small medium large
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
    SIZE_LABELS,
    ablate,
    analytic_model,
    component_legend,
    load_group,
    quiet,
    save,
    sequential_packing,
    stack_heights,
    tightest_patches,
    use_paper_style,
)

DEFAULT_K = [2, 4, 6, 8, 10]


def collect(k_values: List[int], groups: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for group in groups:
        circuits = load_group(group)
        if not circuits:
            continue
        print(f"[k-sweep] {group}: {len(circuits)} circuits")
        for name, circuit in tqdm(circuits, desc=f"  {group}"):
            for k in k_values:
                n_code = k + 2
                result = ablate(circuit, n_code)
                if result is None:
                    continue
                model = analytic_model(circuit, n_code)
                sequential_cost = model.cost(
                    sequential_packing(circuit.num_qubits, k)
                )
                patches = result.patches
                rows.extend(
                    result.rows(
                        size_group=group,
                        circuit_name=name,
                        num_qubits=circuit.num_qubits,
                        k=k,
                        n_code=n_code,
                        physical_qubits=patches * (n_code + 2),
                        analytic_cost=sequential_cost,
                        tightest=tightest_patches(circuit.num_qubits, k),
                    )
                )
    return pd.DataFrame(rows)


def plot_techniques(df: pd.DataFrame, name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    groups = [g for g in SIZE_GROUPS if (df["size_group"] == g).any()]
    k_values = sorted(df.k.unique())

    width = 0.8
    step = width * 1.02
    k_step = len(groups) * step + 1.6
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2.0) * step

    fig, axes = plt.subplots(
        nrows=len(GATE_METRICS), ncols=1, figsize=(8.2, 8.6), sharex=True
    )
    for ax, key in zip(axes, GATE_METRICS):
        column = METRICS[key][0]
        tops = []
        for k_index, k in enumerate(k_values):
            for g_index, group in enumerate(groups):
                x = k_index * k_step + offsets[g_index]
                sub = df[(df.k == k) & (df.size_group == group)]
                if sub.empty:
                    continue
                bottom = 0.0
                for value, (_, _, colour) in zip(stack_heights(sub, column), COMPONENTS):
                    ax.bar(x, value, width=width, bottom=bottom, color=colour,
                           edgecolor="white", linewidth=0.6)
                    bottom += value
                tops.append(bottom)
        ax.set_ylabel(METRICS[key][1])
        ax.set_ylim(0, (max(tops) if tops else 1) * 1.18)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for k_index in range(len(k_values)):
            for g_index, group in enumerate(groups):
                ax.annotate(
                    SIZE_LABELS[group],
                    xy=(k_index * k_step + offsets[g_index], 0.0),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="white",
                )

    component_legend(axes[0], loc="upper right")
    axes[-1].set_xticks([i * k_step for i in range(len(k_values))])
    axes[-1].set_xticklabels([str(k) for k in k_values])
    axes[-1].set_xlabel("Logical qubits per patch, $k$   ($[[k{+}2,\\,k,\\,2]]$ code)")
    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
    return save(fig, name)


def plot_resources(df: pd.DataFrame, name: str) -> Path:
    """Physical qubits and modelled cost against k -- the machine designer's view."""
    import matplotlib.pyplot as plt

    use_paper_style()
    groups = [g for g in SIZE_GROUPS if (df["size_group"] == g).any()]
    k_values = sorted(df.k.unique())
    colours = {"small": "#3b4c56", "medium": "#0097fd", "large": "#c0736a"}

    fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4.0))
    per_circuit = df[df.component == "h_commutation"]

    for group in groups:
        sub = per_circuit[per_circuit.size_group == group]
        atoms = [sub[sub.k == k].physical_qubits.mean() for k in k_values]
        axes[0].plot(k_values, atoms, marker="o", color=colours[group],
                     label=group.capitalize(), linewidth=1.7)

        # Normalise cost to the k=2 value so groups of different size share an axis.
        costs = np.array([sub[sub.k == k].analytic_cost.mean() for k in k_values])
        axes[1].plot(k_values, costs / costs[0], marker="s", color=colours[group],
                     label=group.capitalize(), linewidth=1.7)

    axes[0].set_ylabel("Physical qubits (mean)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Modelled cost, relative to $k=2$")
    axes[1].axhline(1.0, color="#636363", linestyle="--", linewidth=0.9)
    for ax in axes:
        ax.set_xlabel("Logical qubits per patch, $k$")
        ax.set_xticks(k_values)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=12)
    fig.suptitle("Resource cost of a larger code", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K)
    parser.add_argument("--groups", nargs="+", default=["small", "medium"],
                        choices=SIZE_GROUPS)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.k_values, args.groups)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / "fig_k_sweep.csv", index=False)

    if df.empty:
        raise RuntimeError("no data collected")
    plot_techniques(df, "fig_k_sweep_techniques")
    plot_resources(df, "fig_k_sweep_resources")

    print("\n[k-sweep] full-pipeline reduction by group and k")
    for group in SIZE_GROUPS:
        sub = df[df.size_group == group]
        if sub.empty:
            continue
        print(f"  {group}:")
        for k in sorted(sub.k.unique()):
            at_k = sub[sub.k == k]
            parts = [
                f"{METRICS[key][1]}={sum(stack_heights(at_k, METRICS[key][0])):5.1f}%"
                for key in GATE_METRICS
            ]
            fired = int(at_k[at_k.component == "merging"].merging_fired.sum())
            total = int((at_k.component == "merging").sum())
            print(f"    k={k:<3d} {'  '.join(parts)}   [merging fired {fired}/{total}]")


if __name__ == "__main__":
    main()
