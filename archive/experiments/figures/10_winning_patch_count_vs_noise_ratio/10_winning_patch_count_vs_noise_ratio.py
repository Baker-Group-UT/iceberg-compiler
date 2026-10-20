"""
10_winning_patch_count_vs_noise_ratio.py -- how thin to spread, as the noise changes shape.

**Claim.** The number of patches a program should occupy is set by which noise mechanism
the machine is limited by. Where gates pay to travel between patches, the best
configuration is few and full; where qubits pay to sit idle, it is many and thin. The
winning patch count rises monotonically along the ratio axis, and it does so on every
circuit size in the benchmark set.

This is the same finding as `1_best_k_phase_diagram`, said in the units a hardware person
actually plans in. That figure answers "which code size" -- a compiler question. This one
answers "how many patches", which is the question a machine's qubit budget poses, and the
two are not interchangeable: `k` and patch count are related by the program's width, so a
fixed `k` is a different patch count on every circuit.

x = idle/distance noise contribution ratio, geometric mean pinned at one
(`SimSettings.with_noise_ratio`). y = the patch count of the best-scoring code size at that
point, as a median over the circuits in each size bin, with the interquartile range.

**No simulation of its own.** It reads `1_best_k_phase_diagram`'s CSV, which already
records the patch count each `(circuit, ratio, k)` arm actually occupied. One simulation,
two readings, and they cannot disagree.

**Median and IQR, not mean and SD.** The winning patch count is a small integer drawn from
a short ladder (a 26-qubit program at k = 2..10 can only be 13, 7, 5, 4 or 3 patches), so
its distribution across circuits is discrete and skewed. A mean would report values no
circuit can occupy and an SD would imply a symmetry the ladder does not have. The
interquartile range says what it means: half the circuits sit inside this band.

**Binned by program size, because the answer depends on it.** Pooling every circuit into
one line would average across a real interaction: the largest programs turn over to many
thin patches earlier and harder than the smallest. That interaction is the reason
`1_best_k_phase_diagram` needs two axes, and flattening it here would contradict it.

Run with:

    PYTHONPATH=. python experiments/figures/10_winning_patch_count_vs_noise_ratio/10_winning_patch_count_vs_noise_ratio.py
    PYTHONPATH=. python .../10_winning_patch_count_vs_noise_ratio.py --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

NAME = "10_winning_patch_count_vs_noise_ratio"
HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()

#: The phase-diagram sweep this reads. See the module docstring: no simulation of its own.
DEFAULT_SOURCE = (
    SCRIPT.parents[1] / "1_best_k_phase_diagram" / "1_best_k_phase_diagram.csv"
)

DEFAULT_SIZE_BINS = 3


@dataclass
class Settings:
    source: Path = field(default_factory=lambda: DEFAULT_SOURCE)
    size_bins: int = DEFAULT_SIZE_BINS
    output_override: Optional[Path] = None


def winners(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The best-scoring arm per (circuit, ratio), and the patch count it occupied.

    Selection rate is the score, because that is what the source sweep can measure at
    these widths -- TVD is not estimable on a 26-qubit entangled state at any affordable
    shot budget, which is why the sweep records it and never plots it.
    """
    rows: List[Dict] = []
    for (circuit, ratio), block in frame.groupby(["circuit", "ratio"]):
        if block.empty:
            continue
        best = block.loc[block["lsr"].idxmax()]
        rows.append({
            "circuit": circuit,
            "ratio": ratio,
            "measured_ratio_ref": float(block["measured_ratio_ref"].iloc[0]),
            "program_gates": int(best["program_gates"]),
            "program_qubits": int(best["program_qubits"]),
            "best_k": int(best["k"]),
            "patches": int(best["patches"]),
            "lsr": float(best["lsr"]),
            # What the ladder allowed, so "many patches" can be read against its ceiling.
            "max_patches": int(block["patches"].max()),
            "min_patches": int(block["patches"].min()),
        })
    return pd.DataFrame(rows)


def assign_bins(table: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Quantile bins over program gates, so every line carries comparable weight."""
    per_circuit = table.groupby("circuit")["program_gates"].first().sort_values()
    labels = pd.qcut(per_circuit, q=bins, labels=False, duplicates="drop")
    edges = pd.qcut(per_circuit, q=bins, duplicates="drop").cat.categories
    table = table.copy()
    table["size_bin"] = table["circuit"].map(labels)
    table["size_label"] = table["size_bin"].map(
        {i: f"{int(math.ceil(c.left))}-{int(c.right)} gates" for i, c in enumerate(edges)}
    )
    return table


def curves(table: pd.DataFrame) -> pd.DataFrame:
    """Median winning patch count and its interquartile range, per (size bin, ratio)."""
    rows: List[Dict] = []
    for (size_bin, label), block in table.groupby(["size_bin", "size_label"]):
        for ratio in sorted(block["ratio"].unique()):
            column = block[block["ratio"] == ratio]
            if column.empty:
                continue
            finite = column[np.isfinite(column["measured_ratio_ref"])]
            rows.append({
                "size_bin": size_bin,
                "size_label": label,
                "ratio": ratio,
                "measured_ratio": float(
                    np.exp(np.log(finite["measured_ratio_ref"]).mean())
                ) if not finite.empty else float("nan"),
                "patches": float(column["patches"].median()),
                "low": float(column["patches"].quantile(0.25)),
                "high": float(column["patches"].quantile(0.75)),
                "best_k": float(column["best_k"].median()),
                "circuits": int(column["circuit"].nunique()),
            })
    return pd.DataFrame(rows)


def _tick_label(ratio: float) -> str:
    """Powers of four read as fractions below one and integers above it."""
    if ratio >= 1:
        return f"{int(round(ratio))}"
    return f"1/{int(round(1 / ratio))}"


def print_summary(table: pd.DataFrame, curve: pd.DataFrame) -> None:
    ratios = sorted(curve["ratio"].unique())
    print(f"\n[{NAME}] median winning patch count by size bin and noise ratio")
    header = "  size".ljust(22) + "".join(f"{_tick_label(r):>9}" for r in ratios)
    print(header)
    for size_bin in sorted(curve["size_bin"].unique()):
        block = curve[curve["size_bin"] == size_bin]
        label = block["size_label"].iloc[0]
        cells = []
        for ratio in ratios:
            row = block[block["ratio"] == ratio]
            cells.append(f"{'-':>9}" if row.empty else f"{row.iloc[0]['patches']:>9.0f}")
        print(f"  {label:<20}" + "".join(cells))

    print(f"\n[{NAME}] median winning k, same layout")
    print(header)
    for size_bin in sorted(curve["size_bin"].unique()):
        block = curve[curve["size_bin"] == size_bin]
        label = block["size_label"].iloc[0]
        cells = []
        for ratio in ratios:
            row = block[block["ratio"] == ratio]
            cells.append(f"{'-':>9}" if row.empty else f"{row.iloc[0]['best_k']:>9.0f}")
        print(f"  {label:<20}" + "".join(cells))

    low, high = ratios[0], ratios[-1]
    first = table[table["ratio"] == low]["patches"].median()
    last = table[table["ratio"] == high]["patches"].median()
    print(
        f"\n[{NAME}] pooled: the winning configuration goes from {first:.0f} patches at "
        f"ratio {_tick_label(low)} (movement-dominated) to {last:.0f} at "
        f"{_tick_label(high)} (idle-dominated), a factor of {last / max(first, 1):.1f}"
    )

    monotone = []
    for size_bin in sorted(curve["size_bin"].unique()):
        series = curve[curve["size_bin"] == size_bin].sort_values("ratio")["patches"]
        monotone.append(bool((series.diff().dropna() >= 0).all()))
    print(
        f"[{NAME}] {sum(monotone)}/{len(monotone)} size bins are monotone non-decreasing "
        f"in the ratio"
    )


def plot(curve: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.78))

    for size_bin in sorted(curve["size_bin"].unique()):
        block = curve[curve["size_bin"] == size_bin].sort_values("measured_ratio")
        if block.empty:
            continue
        x = block["measured_ratio"].to_numpy()
        y = block["patches"].to_numpy()
        line, = axis.plot(x, y, label=block["size_label"].iloc[0])
        axis.fill_between(
            x, block["low"].to_numpy(), block["high"].to_numpy(),
            color=line.get_color(), alpha=0.15, linewidth=0,
        )

    axis.set_xscale("log")
    axis.set_xlabel("measured idle / movement noise ratio")
    axis.set_ylabel("patches in the best configuration")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    paths = save_figure(fig, NAME, SCRIPT, settings.output_override, versioned=True)
    plt.close(fig)
    for path in paths:
        print(f"[{NAME}] wrote {path}")
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--from-csv", type=Path, default=defaults.source,
        help="The phase-diagram sweep's CSV. No simulation of its own.",
    )
    parser.add_argument("--size-bins", type=int, default=defaults.size_bins)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    return Settings(
        source=args.from_csv,
        size_bins=args.size_bins,
        output_override=args.output_dir,
    )


def main() -> None:
    settings = parse_args()
    if not settings.source.exists():
        raise FileNotFoundError(
            f"{settings.source} not found. Run 1_best_k_phase_diagram.py first."
        )
    frame = pd.read_csv(settings.source)
    if "measured_ratio_ref" not in frame.columns:
        raise SystemExit(
            f"[{NAME}] {settings.source} predates the measured-regime axis -- "
            "regenerate 1_best_k_phase_diagram first."
        )
    table = winners(frame)
    if table.empty:
        raise RuntimeError(f"no winning arm found in {settings.source}")
    table = assign_bins(table, settings.size_bins)
    curve = curves(table)
    print_summary(table, curve)
    plot(curve, settings)


if __name__ == "__main__":
    main()
