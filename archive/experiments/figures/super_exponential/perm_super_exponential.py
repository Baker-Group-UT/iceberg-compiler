"""Visualization helpers for showing how permutation search spaces explode."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica",
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }
)
from matplotlib import font_manager as fm


def compute_totals(num_slots: int, q: int) -> tuple[int, int]:
    """Return (total_combos, est_total) for the provided slots and q."""
    total_combos = math.comb(num_slots, q)
    est_total = total_combos * math.factorial(q)
    return total_combos, est_total


def combinations_for_fixed_slots(
    slot_count: int, q_values: Sequence[int]
) -> dict[str, List[float]]:
    """Return combination counts for a constant slot_count as q varies."""
    totals = {"q": [], "total_combos": []}
    for q in q_values:
        if q < 0:
            raise ValueError("q must be non-negative.")
        if q > slot_count:
            break
        total_combos, _ = compute_totals(slot_count, q)
        totals["q"].append(q)
        totals["total_combos"].append(total_combos)
    return totals


def fractional_q_est_totals(
    slot_counts: Sequence[int], fraction: float
) -> dict[str, List[float]]:
    """Compute est_total when q = floor(fraction * slots)."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    stats = {"slots": [], "q": [], "est_total": []}
    for slots in slot_counts:
        q = max(1, min(slots, math.floor(fraction * slots)))
        _, est_total = compute_totals(slots, q)
        stats["slots"].append(slots)
        stats["q"].append(q)
        stats["est_total"].append(est_total)
    return stats


def plot_total_combos(ax: plt.Axes, data: dict[str, List[float]], slot_count: int) -> None:
    """Plot total_combos as q varies while holding slot_count constant."""
    ax.plot(data["q"], data["total_combos"], color="#265a31", linewidth=3.6)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Total permutations")
    ax.set_title(f"Permutation count as qubit count increases for {slot_count} logical slots")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_fractional_est_totals(
    ax: plt.Axes, stats: dict[str, List[float]], fraction: float
) -> None:
    """Plot est_total on a log scale for q tied to slots via fraction."""
    ax.plot(stats["slots"], stats["est_total"], color="#44a932", linewidth=3.6)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.set_xlabel("Number of slots (75% filled with qubits)")
    ax.set_ylabel("Total permutations")
    ax.set_title(f"Total permutation count as slot and qubit counts increase")
    ax.grid(True, which="major", linestyle="--", linewidth=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def generate_plot(
    slot_counts: Iterable[int],
    q_values: Iterable[int],
    output_path: Path,
    fractional_q_ratio: float,
    fixed_slot_count: int,
) -> None:
    """Generate a two-panel figure visualizing total_combos and est_total."""
    slot_counts = list(slot_counts)
    q_values = sorted(set(q_values))
    totals = combinations_for_fixed_slots(fixed_slot_count, q_values)
    fractional_stats = fractional_q_est_totals(slot_counts, fractional_q_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=False, sharex=False)
    plot_total_combos(axes[0], totals, fixed_slot_count)
    plot_fractional_est_totals(axes[1], fractional_stats, fractional_q_ratio)

    fig.suptitle("Super-exponential growth of permutation search spaces")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot log-space total combinations as slots and q grow."
    )
    parser.add_argument(
        "--min-slots",
        type=int,
        default=5,
        help="Minimum slot count to include for the fractional subplot.",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=75,
        help="Maximum slot count to include.",
    )
    parser.add_argument(
        "--q-values",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="q values to evaluate for the fixed-slot subplot.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.75,
        help=(
            "Fraction of slots used to define q for the second subplot "
            "(q = floor(fraction * slots))."
        ),
    )
    parser.add_argument(
        "--fixed-slots",
        type=int,
        default=10,
        help="Slot count used when plotting total combinations vs q.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".svg"),
        help="Path to save the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.q_values = sorted(set(args.q_values))
    if any(q < 0 for q in args.q_values):
        raise ValueError("q-values must be non-negative.")
    max_q = max(args.q_values)
    if args.fixed_slots < max_q:
        raise ValueError(
            f"fixed_slots ({args.fixed_slots}) must be >= largest q ({max_q}) "
            "so that combinations are defined."
        )
    if args.min_slots < 1:
        raise ValueError("min_slots must be >= 1.")
    if args.max_slots <= args.min_slots:
        raise ValueError("max_slots must be greater than min_slots.")

    slot_counts = range(args.min_slots, args.max_slots + 1)
    generate_plot(
        slot_counts,
        args.q_values,
        args.output,
        args.fraction,
        args.fixed_slots,
    )


if __name__ == "__main__":
    main()
