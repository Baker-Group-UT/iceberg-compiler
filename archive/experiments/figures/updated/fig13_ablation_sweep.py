"""
fig13_ablation_sweep.py — arXiv Figure 13, rebuilt.

"Ablation of techniques over packing density", with all five metric rows: 1q gates,
2q gates, depth, LSR and TVD. The draft's headline claim from this figure is that
packing contributes disproportionately to LSR and TVD relative to its gate-count
contribution, attributed to grid alignment reducing communication cost.

Simulation makes this the slowest script here -- budget hours, not minutes. Use
`--no-sim` for a gate-only version in seconds while iterating on the plot.

    python -m experiments.figures.updated.fig13_ablation_sweep --no-sim
    python -m experiments.figures.updated.fig13_ablation_sweep --groups small
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
    component_legend,
    config_for,
    load_group,
    measure,
    optimized_placement,
    pct_improvement,
    pct_reduction,
    quiet,
    save,
    sequential_packing,
    use_paper_style,
)
from experiments.figures.updated.simulation import (
    SimSettings,
    build_config,
    calibrate,
    evaluate,
)
from iceberg.utils import logical_capacity

ALL_METRICS = GATE_METRICS + ["lsr", "tvd"]

VARIANTS = (
    ("base", False, False),
    ("h", True, False),
    ("hm", True, True),
    ("pack", True, True),
)


def density_points(num_qubits: int, k: int, samples: int) -> List[int]:
    """Patch counts spanning tightest (density k) to solo (density 1)."""
    tightest = max(1, -(-num_qubits // k))
    raw = np.linspace(tightest, num_qubits, samples)
    return sorted({int(round(value)) for value in raw})


def spread_packing(num_qubits: int, patches: int, k: int) -> List:
    """
    Index-order fill across exactly `patches` patches, honouring capacity.

    The naive baseline at a given density: qubits go in program order, spread as
    evenly as the patch count allows.
    """
    placement: List = []
    for index in range(num_qubits):
        patch = index % patches
        slot = index // patches
        placement.append((patch, min(slot, k - 1)))
    seen = set()
    for index, (patch, slot) in enumerate(placement):
        while (patch, slot) in seen:
            slot += 1
            if slot >= k:
                patch = (patch + 1) % patches
                slot = 0
        seen.add((patch, slot))
        placement[index] = (patch, slot)
    return placement


def collect(
    n_code: int, groups: List[str], samples: int, simulate: bool, settings: SimSettings
) -> pd.DataFrame:
    k = logical_capacity(n_code)
    rows: List[Dict] = []

    for group in groups:
        circuits = load_group(group)
        if not circuits:
            continue
        print(f"[fig13] {group}: {len(circuits)} circuits")
        for name, circuit in tqdm(circuits, desc=f"  {group}"):
            trivial = sequential_packing(circuit.num_qubits, k)
            here = measure(
                circuit, trivial, config_for(n_code, h_commutation=False, merging=False)
            )
            if here is None:
                continue

            alpha, repetitions = 0.0, 0
            if simulate:
                alpha, repetitions = calibrate(
                    circuit, trivial, settings, n_code=n_code,
                    baseline_gate_count=here.one_qubit + here.two_qubit,
                )

            for patches in density_points(circuit.num_qubits, k, samples):
                naive = spread_packing(circuit.num_qubits, patches, k)
                packed = optimized_placement(circuit, n_code, num_patches=patches)
                perms = {"base": naive, "h": naive, "hm": naive, "pack": packed}

                counts = {
                    label: measure(
                        circuit, perms[label],
                        config_for(n_code, h_commutation=h, merging=m),
                    )
                    for label, h, m in VARIANTS
                }
                if any(value is None for value in counts.values()):
                    continue

                sims: Dict[str, Dict[str, float]] = {}
                if simulate:
                    for label, h, m in VARIANTS:
                        result = evaluate(
                            circuit, perms[label],
                            build_config(settings, n_code=n_code, alpha=alpha,
                                         repetitions=repetitions,
                                         h_commutation=h, merging=m),
                        )
                        if result is None:
                            sims = {}
                            break
                        sims[label] = result

                cumulative: Dict[str, List[float]] = {}
                for label in ("h", "hm", "pack"):
                    values = [
                        pct_reduction(b, c)
                        for b, c in zip(counts["base"].as_tuple(), counts[label].as_tuple())
                    ]
                    if sims:
                        values.append(pct_improvement(sims["base"]["lsr"], sims[label]["lsr"]))
                        values.append(pct_reduction(sims["base"]["tvd"], sims[label]["tvd"]))
                    cumulative[label] = values

                zero = [0.0] * len(cumulative["h"])
                for component, later, earlier in (
                    ("h_commutation", cumulative["h"], zero),
                    ("merging", cumulative["hm"], cumulative["h"]),
                    ("packing", cumulative["pack"], cumulative["hm"]),
                ):
                    row = {
                        "size_group": group, "circuit_name": name, "n_code": n_code,
                        "num_qubits": circuit.num_qubits, "patches": patches,
                        "density": circuit.num_qubits / patches, "component": component,
                    }
                    for key, value in zip(
                        ALL_METRICS, [a - b for a, b in zip(later, earlier)]
                    ):
                        row[METRICS[key][0]] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, n_code: int, metrics: List[str], name: str) -> Path:
    import matplotlib.pyplot as plt

    use_paper_style()
    groups = [g for g in SIZE_GROUPS if (df["size_group"] == g).any()]
    k = logical_capacity(n_code)
    bins = np.linspace(1.0, k, 7)
    df = df.assign(bin=pd.cut(df.density, bins=bins, include_lowest=True))
    centres = sorted(
        (b for b in df.bin.dropna().unique()), key=lambda i: i.mid, reverse=True
    )

    width, step = 0.8, 0.82
    bin_step = len(groups) * step + 1.5
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2.0) * step

    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=1,
        figsize=(8.6, 2.3 * len(metrics) + 1.0), sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, metrics):
        column = METRICS[key][0]
        for b_index, interval in enumerate(centres):
            for g_index, group in enumerate(groups):
                sub = df[(df.bin == interval) & (df.size_group == group)]
                if sub.empty:
                    continue
                x = b_index * bin_step + offsets[g_index]
                bottom = 0.0
                for component, _, colour in COMPONENTS:
                    value = float(
                        sub[sub.component == component][column].clip(lower=0).mean()
                    )
                    value = 0.0 if np.isnan(value) else value
                    ax.bar(x, value, width=width, bottom=bottom, color=colour,
                           edgecolor="white", linewidth=0.5)
                    bottom += value
                ax.annotate(SIZE_LABELS[group], xy=(x, 0), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, color="white")
        ax.set_ylabel(METRICS[key][1])
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    component_legend(axes[0], loc="upper right")
    axes[-1].set_xticks([i * bin_step for i in range(len(centres))])
    axes[-1].set_xticklabels([f"{i.mid:.2f}" for i in centres])
    axes[-1].set_xlabel(
        f"Packing density (logical qubits per patch), $[[{n_code},\\,{k},\\,2]]$ code"
    )
    fig.supylabel("Percent improvement / reduction")
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--groups", nargs="+", default=["small", "medium"],
                        choices=SIZE_GROUPS)
    parser.add_argument("--density-samples", type=int, default=6)
    parser.add_argument("--no-sim", action="store_true", help="Skip LSR/TVD (fast)")
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    simulate = not args.no_sim
    name = f"fig13_ablation_sweep_n{args.n_code}" + ("" if simulate else "_gates")
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(
            args.n_code, args.groups, args.density_samples, simulate, SimSettings()
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    if df.empty:
        raise RuntimeError("no data collected")
    metrics = [
        key for key in ALL_METRICS
        if METRICS[key][0] in df.columns and df[METRICS[key][0]].notna().any()
    ]
    plot(df, args.n_code, metrics, name)

    print("\n[fig13] mean contribution per technique, pooled over density")
    for component, label, _ in COMPONENTS:
        sub = df[df.component == component]
        parts = [
            f"{METRICS[key][1]}={sub[METRICS[key][0]].clip(lower=0).mean():5.1f}%"
            for key in metrics
        ]
        print(f"  {label:16s} {'  '.join(parts)}")


if __name__ == "__main__":
    main()
