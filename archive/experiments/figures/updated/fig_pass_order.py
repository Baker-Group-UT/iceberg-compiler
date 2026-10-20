"""
fig_pass_order.py — what merging and packing are worth without commutation first.

The draft says, of Figure 10: "Because Hadamard commutation is applied first, it
captures the largest share of gate reductions and leaves fewer opportunities for
merging and packing to merge Hadamards. If merging and packing were applied before
commutation, their relative contributions would increase accordingly." That is an
argument, not a measurement. This figure measures it.

Three conditions, selected by `TranslatorConfig.h_commutation_stage`:

    commutation first   the paper's order. Commutation runs, then merging.
    commutation last    merging runs first and sees the Hadamards; commutation then
                        works on what is left. Macros are opaque to it -- it flushes
                        its frame at any instruction it does not recognise -- so
                        whole-patch and transversal macros act as barriers.
    no commutation      merging and packing alone, which is the honest ceiling for
                        what they can do unaided.

Each bar is a *cumulative* reduction against the same naive baseline, so the three
conditions are directly comparable and the difference between them is exactly the
attribution effect the draft describes.

Gate counts and depth only, so this is minutes.

    python -m experiments.figures.updated.fig_pass_order
    python -m experiments.figures.updated.fig_pass_order --n-code 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.figures.updated.common import (
    GATE_METRICS,
    METRICS,
    OUTPUT_DIR,
    SIZE_GROUPS,
    load_group,
    measure,
    optimized_placement,
    pct_reduction,
    quiet,
    save,
    sequential_packing,
    use_paper_style,
)
from iceberg.utils import logical_capacity
from pipeline.config import PipelineConfig

#: label -> (h_commutation_stage, enable_h_commutation, colour)
CONDITIONS: Dict[str, Tuple[str, bool, str]] = {
    "Commutation first\n(paper order)": ("before_merging", True, "#3b4c56"),
    "Commutation last": ("after_merging", True, "#c0736a"),
    "No commutation": ("never", False, "#0097fd"),
}

#: What each stacked segment measures, in order.
STAGES = [
    ("merging", "Merging", 0.55),
    ("packing", "+ Packing", 0.8),
    ("commutation", "+ Commutation", 1.0),
]


def config_for_stage(
    n_code: int, stage: str, commutation: bool, *, merging: bool
) -> PipelineConfig:
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_merging = merging
    config.translator_config.enable_h_commutation = commutation
    config.translator_config.h_commutation_stage = stage
    return config


def collect(n_code: int, groups: List[str]) -> pd.DataFrame:
    k = logical_capacity(n_code)
    rows: List[Dict] = []

    for group in groups:
        circuits = load_group(group)
        if not circuits:
            continue
        print(f"[pass-order] {group}: {len(circuits)} circuits")
        for name, circuit in tqdm(circuits, desc=f"  {group}"):
            trivial = sequential_packing(circuit.num_qubits, k)
            # One baseline for all three conditions: no technique at all.
            base = measure(
                circuit, trivial, config_for_stage(n_code, "never", False, merging=False)
            )
            if base is None:
                continue
            placed = optimized_placement(circuit, n_code)

            for label, (stage, commutation, _) in CONDITIONS.items():
                variants = {
                    # merging alone, on the naive placement
                    "merging": measure(
                        circuit, trivial,
                        config_for_stage(n_code, "never", False, merging=True),
                    ),
                    # merging plus packing
                    "packing": measure(
                        circuit, placed,
                        config_for_stage(n_code, "never", False, merging=True),
                    ),
                    # everything, with commutation wherever this condition puts it
                    "commutation": measure(
                        circuit, placed,
                        config_for_stage(n_code, stage, commutation, merging=True),
                    ),
                }
                if any(value is None for value in variants.values()):
                    continue
                for key, metrics in variants.items():
                    row = {
                        "condition": label, "stage": key, "size_group": group,
                        "circuit_name": name, "n_code": n_code,
                        "num_qubits": circuit.num_qubits,
                    }
                    for metric, ref, got in zip(
                        GATE_METRICS, base.as_tuple(), metrics.as_tuple()
                    ):
                        row[f"cumulative_{METRICS[metric][0]}"] = pct_reduction(ref, got)
                    rows.append(row)
    return pd.DataFrame(rows)


def _monotone(frame, column: str) -> List[float]:
    """Segment heights from clipped cumulative means, so nothing exceeds the total."""
    running = 0.0
    out: List[float] = []
    for key, _, _ in STAGES:
        values = frame[frame.stage == key][f"cumulative_{column}"]
        mean = float(values.mean()) if len(values) else running
        mean = running if mean != mean else mean
        clipped = max(running, mean)
        out.append(clipped - running)
        running = clipped
    return out


def plot(df: pd.DataFrame, n_code: int, name: str) -> Path:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    use_paper_style()
    conditions = [c for c in CONDITIONS if (df.condition == c).any()]
    fig, axes = plt.subplots(
        nrows=len(GATE_METRICS), ncols=1, figsize=(7.4, 8.0), sharex=True
    )

    for ax, key in zip(axes, GATE_METRICS):
        column = METRICS[key][0]
        xs = np.arange(len(conditions))
        heights = {c: _monotone(df[df.condition == c], column) for c in conditions}
        bottom = np.zeros(len(conditions))
        for index, (_, _, alpha) in enumerate(STAGES):
            values = np.array([heights[c][index] for c in conditions])
            colours = [CONDITIONS[c][2] for c in conditions]
            ax.bar(xs, values, width=0.6, bottom=bottom, color=colours,
                   alpha=alpha, edgecolor="white", linewidth=0.7)
            bottom += values
        ax.set_ylabel(METRICS[key][1])
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor="#666666", alpha=alpha, edgecolor="white", label=label)
            for _, label, alpha in STAGES
        ],
        loc="upper right",
        frameon=False,
        title="Cumulative",
        title_fontsize=11,
    )
    axes[-1].set_xticks(np.arange(len(conditions)))
    axes[-1].set_xticklabels(conditions)
    k = logical_capacity(n_code)
    axes[-1].set_xlabel(f"Pass order   ($[[{n_code},\\,{k},\\,2]]$ code)")
    fig.supylabel("Percent reduction vs. no techniques")
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    return save(fig, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--groups", nargs="+", default=SIZE_GROUPS, choices=SIZE_GROUPS)
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()

    quiet()
    name = f"fig_pass_order_n{args.n_code}"
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.n_code, args.groups)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    if df.empty:
        raise RuntimeError("no data collected")
    plot(df, args.n_code, name)

    print("\n[pass-order] cumulative reduction by condition and stage")
    for condition in CONDITIONS:
        sub = df[df.condition == condition]
        if sub.empty:
            continue
        print(f"  {condition.splitlines()[0]}:")
        for key, label, _ in STAGES:
            at = sub[sub.stage == key]
            parts = [
                f"{METRICS[m][1]}={at[f'cumulative_{METRICS[m][0]}'].mean():6.2f}%"
                for m in GATE_METRICS
            ]
            print(f"    {label:16s} {'  '.join(parts)}")


if __name__ == "__main__":
    main()
