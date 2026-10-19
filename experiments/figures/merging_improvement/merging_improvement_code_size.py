"""
merging_improvement_code_size.py
--------------------------------

How do the compiler's benefits change as the code holds more logical qubits?

Sweeps the code size over the ``[[k+2, k, 2]]`` family -- k = 2 (the [[4,2,2]]
Iceberg code the paper evaluates) through k = 10 -- and, at each k, measures the
cumulative percent reduction from three techniques against a naive baseline at the
same k:

    1. H commutation
    2. + merging
    3. + packing

Styled after Figure 12 of the paper draft (``technique_benefits_random_hg.pdf``):
one row per metric, stacked bars whose segments are the incremental benefit of each
technique, with one bar per benchmark size group at every x position.

Merging now runs at every code size. It used to be [[4,2,2]]-only, which made the
k >= 4 bars a lower bound of unknown tightness; ``translator/merging`` is general in
k as of this revision, so all three segments are measured at every point. What
changes with k is how *reachable* a merge is rather than whether the pass exists:
``hh`` and the transversal macros are whole-patch operations, so they need a gate on
every occupied lane at once, and that gets rarer as k grows. ``xx``/``zz`` are
per-lane and fuse on any subset.

One thing about this sweep is still not symmetric across k, and it is recorded rather
than hidden:

**Packing gets less to decide as k grows.** The comparison holds packing density
fixed at its tightest, ceil(q/k) patches, so patch count falls as k rises and a
small circuit eventually fits in a single patch -- at which point there is no
placement decision left and the packing segment is zero for a structural reason
rather than an algorithmic one. ``patches_packed`` is recorded per row, the summary
prints how many circuits are degenerate at each k, and ``--min-patches 2`` reruns
the aggregate over just the circuits that still pose a packing problem.

The packer is the ``packing`` package -- profile, signed-graph seed, one descent on
the real objective, exact lane assignment, then a two-translation verify against the
naive packing. It is general in k, unlike ``packer/merge_packer.py``, which assigned
qubits in pairs and so could not express a placement at k > 2 at all; that packer and
the ``grouping`` stack this script used to call are both archived. The objective
prices weighted gate count *and* depth, so all three plotted rows are optimised rather
than two of them plus a bystander.

LSR and TVD are wired through the CSV and the plot but require simulation; they are
off unless ``--metrics`` asks for them.

**Cost.** This used to be the slow script in the repo, and the reason was the
objective rather than the search: every candidate placement was scored by
*translating the circuit*, so a sweep cost ``candidates x translation``. The swap
neighbourhood is ``O(m^2)``, which made the whole thing ``O(m^3)`` with a large
constant -- and, worse, wildly uneven. Runtime tracked neither qubit count nor the
progress bar: a point finished instantly whenever the circuit collapsed to a single
patch (local search skipped entirely) and then jumped to minutes as soon as there
were two, scaled by gate count rather than width. ``dnn_n8``, 8 qubits, took 8.5s at
k=6 while ``qaoa_n6`` took 0.00s.

``packing.cost`` scores candidates instead: ~0.1ms apiece rather than ~30-700ms, no
translation in the search loop. That needs no wall-clock cap, so no point is truncated
and every placement is a real local optimum of the stated objective. Only the final
candidate is measured exactly, against the naive placement -- two translations -- which
is what keeps the packing segment from reading as a gain that is not one. ``--packer
raw`` turns that check off and reports the search unguarded; it answers whether the
*algorithm* improved, where the default answers what a caller actually gets.

Expect a couple of minutes for ``--groups small medium`` and well under an hour with
``large`` added.

Run with:

    python -m experiments.figures.merging_improvement.merging_improvement_code_size \
        --groups small medium --no-show

    # replot from saved data without recomputing
    python -m experiments.figures.merging_improvement.merging_improvement_code_size \
        --from-csv experiments/figures/merging_improvement/merging_improvement_code_size.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.figures.merging_improvement.common import stack_heights
from experiments.shared.benchmark_circuits import load_proxy_circuits
from packing.cost import DEFAULT_DEPTH_WEIGHT
from packing.graph import NoiseProfile
from packing.plan import plan_packing
from packing.seed import DEFAULT_PARTITIONER, PARTITIONERS
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

Perm = List[Tuple[int, int]]

# Figure 12 is set in Times New Roman; match it so the figure drops into the paper
# without a font mismatch.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 13,
        "legend.fontsize": 15,
        "figure.titlesize": 16,
    }
)


# ---------------------------------------------------------------------------
# Metrics, components, groups
# ---------------------------------------------------------------------------

#: key -> (csv column, y-axis label, needs simulation)
METRICS: Dict[str, Tuple[str, str, bool]] = {
    "1q": ("pct_reduction_one_qubit_gates", "#1q gates", False),
    "2q": ("pct_reduction_two_qubit_gates", "#2q gates", False),
    "depth": ("pct_reduction_circuit_depth", "depth", False),
    "lsr": ("pct_improvement_logical_selection_rate", "LSR", True),
    "tvd": ("pct_reduction_logical_total_variation_distance", "TVD", True),
}

DEFAULT_METRICS = ["1q", "2q", "depth"]

# Colours taken from the existing stylized figures so the sweep matches Figure 12.
COMPONENTS = [
    ("h_commutation", "H commutation", "#3b4c56"),
    ("merging", "+ Merging", "#cdc2c0"),
    ("packing", "+ Packing", "#0097fd"),
]

SIZE_GROUPS = ["small", "medium", "large"]
SIZE_GROUP_LABELS = {"small": "S", "medium": "M", "large": "L"}

DEFAULT_K_VALUES = [2, 4, 6, 8, 10]

#: Depth weight for the placement objective. `packing` and this sweep have to agree on
#: it: the search optimises weighted gates plus `depth_weight` per layer, and the depth
#: row plots what that produced, so a second number here would mean the search and the
#: measurement validating it wanted different things. That is the mistake the old stack
#: made -- 0.0 in `plan_placement`, 12.0 on the figure path -- so take the package's.
DEFAULT_ANALYTIC_DEPTH_WEIGHT = DEFAULT_DEPTH_WEIGHT


@dataclass
class CodeSizeSettings:
    qasm_root: Path = Path("benchmarks/qasmbench")
    groups: Sequence[str] = ("small", "medium", "large")
    k_values: Sequence[int] = tuple(DEFAULT_K_VALUES)
    metrics: Sequence[str] = tuple(DEFAULT_METRICS)
    max_proxy_gates: Optional[int] = 2500
    max_proxy_qubits: Optional[int] = 150
    proxy_timeout_seconds: Optional[float] = 1.0
    packer: str = "packing"
    partitioner: str = DEFAULT_PARTITIONER
    min_patches: int = 1
    output_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_code_size.svg"
        )
    )
    csv_path: Path = field(
        default_factory=lambda: Path(
            "experiments/figures/merging_improvement/merging_improvement_code_size.csv"
        )
    )
    from_csv: Optional[Path] = None
    show_plot: bool = False


DEFAULT_SETTINGS = CodeSizeSettings()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def tight_trivial_packing(num_qubits: int, k: int) -> Perm:
    """Fill patches in program order, k slots at a time -- the naive baseline."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return [(index // k, index % k) for index in range(num_qubits)]


def patches_for(num_qubits: int, k: int) -> int:
    """Tightest packing: the fewest patches that hold `num_qubits` at k slots each."""
    return max(1, -(-num_qubits // k))


def build_config(*, n_code: int, h_commutation: bool, merging: bool) -> PipelineConfig:
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    return config


def compute_metrics(
    program_circuit: qiskit.QuantumCircuit,
    permutation: Perm,
    config: PipelineConfig,
) -> Optional[Tuple[int, int, int]]:
    """(1q, 2q, depth) for the translated circuit, or None if placement is rejected."""
    try:
        translated = translate_permutation(program_circuit, permutation, config=config)
    except Exception:  # noqa: BLE001 - rejected under the placement rules
        return None
    circuit = translated.circuit
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    return one_q, two_q, circuit.depth()


# ---------------------------------------------------------------------------
# Packing strategies
# ---------------------------------------------------------------------------


def packed_permutation(
    program_circuit: qiskit.QuantumCircuit,
    n_code: int,
    settings: CodeSizeSettings,
    *,
    depth_weight: float = DEFAULT_ANALYTIC_DEPTH_WEIGHT,
    noise: Optional[NoiseProfile] = None,
) -> Tuple[Optional[Perm], str]:
    """
    Place with the `packing` package: graph seed, descent, lanes, verify.

    This is the reason the script is no longer the slow one. Nothing translates the
    circuit inside the search loop -- candidates are scored by `packing.cost`, a cached
    sum of per-patch and per-pair terms costing roughly 0.1ms, so the descent is
    affordable rather than the dominant expense. That matters because the graph is a
    *pairwise* proxy for an objective that is not pairwise: whole-patch HH and the
    transversal macros only pay when every lane of a patch pair cooperates, and no edge
    weight can express a conjunction. The graph seeds; the descent optimises the real
    thing.

    Three details the call has to get right, all of them ways the number could quietly
    stop meaning what the axis says:

    **The raw program goes in.** `plan_packing` applies the H-commutation passes itself
    through `translated_view`. Handing it a preprocessed circuit runs them twice, which
    was worth a 13-49% cost error when this script last made that mistake.

    **`h_commutation` is pinned, not `"auto"`.** The package decides per circuit whether
    commutation pays and skips it where it does not, which is right for the compiler and
    wrong here: the packed row is translated with `enable_h_commutation=True`, so the
    plan must be built for that same circuit or the packing segment absorbs the
    commutation decision too.

    **`depth_weight` matches what the rows are scored with.** One number runs the search
    and the measurement.

    `verify` translates the winner and the naive packing and keeps the cheaper, so the
    packing segment can never read as a gain that is not one -- the model over-counts
    merging by 2-3% and on a few circuits that is enough to prefer a placement genuinely
    worse than sequential. `--packer raw` is that check turned off.
    """
    try:
        plan = plan_packing(
            program_circuit,
            n_code,
            noise=noise,
            depth_weight=depth_weight,
            partitioner=settings.partitioner,
            h_commutation=True,
            verify=settings.packer != "raw",
        )
    except Exception as error:  # noqa: BLE001 - a failed plan is data, not a crash
        return None, f"packing failed: {type(error).__name__}"

    note = f"{settings.packer}:{plan.patches}p"
    if plan.descent is not None:
        note += f" descend:{plan.descent.moves}m"
    if plan.fell_back:
        note += " naive-kept"
    return plan.placement, note


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------


def load_group_circuits(
    group: str, settings: CodeSizeSettings
) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    timeout = settings.proxy_timeout_seconds
    if timeout is not None and timeout <= 0:
        timeout = None
    circuits = load_proxy_circuits(
        settings.qasm_root / group,
        gate_limit=2000,
        qubit_limit=settings.max_proxy_qubits,
        timeout_seconds=timeout,
        replace_rz_rx_with_s=False,
    )
    if settings.max_proxy_gates is not None:
        circuits = [
            (name, circuit)
            for name, circuit in circuits
            if circuit.size() <= settings.max_proxy_gates
        ]
    # `*_n2` circuits have two qubits, so every k >= 2 puts them in one patch and
    # there is no placement question to ask.
    return [
        (name, circuit)
        for name, circuit in circuits
        if not name.lower().endswith("_n2")
    ]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("one_qubit_gates", "two_qubit_gates", "circuit_depth")


def run_experiment(
    circuits: Sequence[Tuple[str, qiskit.QuantumCircuit]],
    size_group: str,
    settings: CodeSizeSettings,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for circuit_name, program_circuit in tqdm(
        list(circuits), desc=f"circuits ({size_group})"
    ):
        num_qubits = program_circuit.num_qubits

        for k in settings.k_values:
            n_code = k + 2

            baseline_cfg = build_config(n_code=n_code, h_commutation=False, merging=False)
            h_only_cfg = build_config(n_code=n_code, h_commutation=True, merging=False)
            h_merge_cfg = build_config(n_code=n_code, h_commutation=True, merging=True)

            trivial = tight_trivial_packing(num_qubits, k)

            started = time.perf_counter()
            packed, note = packed_permutation(program_circuit, n_code, settings)
            pack_seconds = time.perf_counter() - started

            base = compute_metrics(program_circuit, trivial, baseline_cfg)
            h_only = compute_metrics(program_circuit, trivial, h_only_cfg)
            h_merge = compute_metrics(program_circuit, trivial, h_merge_cfg)
            if base is None or h_only is None or h_merge is None:
                print(
                    f"[code_size] skip {size_group}/{circuit_name} k={k}: "
                    "trivial placement rejected"
                )
                continue

            # A packer that cannot place falls back to the trivial permutation, so
            # its component reads as zero benefit rather than as missing data.
            packed_metrics = (
                compute_metrics(program_circuit, packed, h_merge_cfg)
                if packed is not None
                else None
            )
            if packed_metrics is None:
                packed_metrics = h_merge
                packed = trivial
                note = (note + " -> fell back to trivial").strip()

            cumulative = {
                "h_commutation": [
                    pct_reduction(b, c) for b, c in zip(base, h_only)
                ],
                "merging": [pct_reduction(b, c) for b, c in zip(base, h_merge)],
                "packing": [pct_reduction(b, c) for b, c in zip(base, packed_metrics)],
            }
            # Segments are differences between successive cumulative totals, so the
            # stack height is the full-pipeline reduction.
            segments = {
                "h_commutation": cumulative["h_commutation"],
                "merging": [
                    later - earlier
                    for later, earlier in zip(
                        cumulative["merging"], cumulative["h_commutation"]
                    )
                ],
                "packing": [
                    later - earlier
                    for later, earlier in zip(
                        cumulative["packing"], cumulative["merging"]
                    )
                ],
            }

            shared = {
                "size_group": size_group,
                "circuit_name": circuit_name,
                "num_qubits": num_qubits,
                "program_gates": program_circuit.size(),
                "k": k,
                "n_code": n_code,
                # Whether the pass actually changed anything, rather than whether it
                # is implemented -- it now runs at every k, so "did it fire" is the
                # question worth recording. Whole-patch macros need a gate on every
                # occupied lane at once, which gets rarer as k grows.
                "merging_fired": h_merge != h_only,
                "patches_trivial": len({patch for patch, _ in trivial}),
                "patches_packed": len({patch for patch, _ in packed}),
                "packer": settings.packer,
                "pack_note": note,
                "pack_seconds": round(pack_seconds, 3),
                "baseline_one_qubit_gates": base[0],
                "baseline_two_qubit_gates": base[1],
                "baseline_circuit_depth": base[2],
                "full_one_qubit_gates": packed_metrics[0],
                "full_two_qubit_gates": packed_metrics[1],
                "full_circuit_depth": packed_metrics[2],
            }
            for component, values in segments.items():
                row = dict(shared, component=component)
                for metric_key, value in zip(_METRIC_KEYS, values):
                    row[f"pct_reduction_{metric_key}"] = value
                # The cumulative total this segment was differenced out of. Recorded
                # because plotting needs it: clipping *segments* at zero inflates the
                # stack whenever one technique hurts and a later one recovers. See
                # `aggregate`.
                for metric_key, value in zip(_METRIC_KEYS, cumulative[component]):
                    row[f"cumulative_pct_reduction_{metric_key}"] = value
                # Placeholders so the CSV schema is stable once simulation lands.
                row["pct_improvement_logical_selection_rate"] = np.nan
                row["pct_reduction_logical_total_variation_distance"] = np.nan
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "size_group", "circuit_name", "num_qubits", "program_gates", "k", "n_code",
    "merging_fired", "patches_trivial", "patches_packed", "packer", "pack_note",
    "pack_seconds", "component",
    "baseline_one_qubit_gates", "baseline_two_qubit_gates", "baseline_circuit_depth",
    "full_one_qubit_gates", "full_two_qubit_gates", "full_circuit_depth",
    "pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates",
    "pct_reduction_circuit_depth",
    "cumulative_pct_reduction_one_qubit_gates",
    "cumulative_pct_reduction_two_qubit_gates",
    "cumulative_pct_reduction_circuit_depth",
    "pct_improvement_logical_selection_rate",
    "pct_reduction_logical_total_variation_distance",
]


def append_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df[CSV_COLUMNS].to_csv(csv_path, mode="a", index=False, header=write_header)
    print(f"[code_size] appended {len(df)} rows to {csv_path}")


#: What identifies one measurement. `circuit_name` alone is not enough -- QASMBench's
#: `small` group holds two distinct `basis_trotter_n4` files (92 and 1888 gates), and
#: `load_proxy_circuits` gives both the same proxy name.
ROW_KEY = ["size_group", "circuit_name", "program_gates", "num_qubits", "k", "component"]


def read_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the sweep, dropping duplicate measurements.

    Rows are appended per group so that a long run does not lose everything if it is
    interrupted, which means two runs sharing a `--csv` path silently merge instead of
    one overwriting the other: the startup `unlink` only fires once, at whichever run
    started later. That happened during development and produced a file with the large
    group in it twice, which does not move the per-group means but does weight that
    group twice in the pooled per-technique table.

    Keeping the last occurrence makes a rerun of one group behave the way a rerun
    ought to -- it replaces that group rather than doubling it.
    """
    df = pd.read_csv(csv_path)
    if not set(ROW_KEY).issubset(df.columns):
        return df
    deduped = df.drop_duplicates(subset=ROW_KEY, keep="last")
    dropped = len(df) - len(deduped)
    if dropped:
        print(
            f"[code_size] dropped {dropped} duplicate rows from {csv_path} "
            "(kept the most recent measurement of each)"
        )
    return deduped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def active_metrics(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    """Requested metrics that actually carry data, so LSR/TVD rows appear only once simulated."""
    active = []
    for key in requested:
        column = METRICS[key][0]
        if column in df.columns and df[column].notna().any():
            active.append(key)
    return active


def aggregate(df: pd.DataFrame, metrics: Sequence[str], min_patches: int) -> pd.DataFrame:
    """
    Segment heights per (size group, k), from clipped *cumulative* means.

    A stack segment is a benefit and cannot be drawn downwards, so something has to
    be clipped. Clipping each segment -- which this script used to do, and the
    draft's own scripts still do -- is the wrong choice: when one technique hurts and
    a later one more than recovers, dropping the negative segment leaves the
    recovery at full height and the bar exceeds the real reduction. On `ising_n98`
    that produces a 179% bar on a percent-reduction axis.

    Clipping the cumulative series and differencing it instead keeps every segment
    non-negative *and* keeps the stack height equal to the true full-pipeline
    reduction; a technique that hurts contributes zero rather than inflating the next
    one. Raw per-segment values stay in the CSV either way.
    """
    columns = [METRICS[key][0] for key in metrics]
    subset = df.loc[df["patches_packed"] >= min_patches]
    order = [component for component, _, _ in COMPONENTS]

    rows: List[Dict] = []
    for (group, k), frame in subset.groupby(["size_group", "k"]):
        heights = {column: stack_heights(frame, column, order) for column in columns}
        for index, component in enumerate(order):
            row: Dict = {"size_group": group, "k": k, "component": component}
            for column in columns:
                row[column] = heights[column][index]
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, metrics: Sequence[str], settings: CodeSizeSettings) -> None:
    columns = [METRICS[key][0] for key in metrics]
    order = [component for component, _, _ in COMPONENTS]

    print("\n[code_size] full-pipeline reduction, mean over circuits")
    print("            (plotted = stack height; true = mean cumulative reduction)")
    for group in SIZE_GROUPS:
        group_df = df.loc[df["size_group"] == group]
        if group_df.empty:
            continue
        print(f"  {group}:")
        for k in sorted(group_df["k"].unique()):
            k_df = group_df.loc[group_df["k"] == k]
            circuits = k_df["circuit_name"].nunique()
            degenerate = (
                k_df.loc[k_df["component"] == "h_commutation", "patches_packed"] < 2
            ).sum()
            parts = []
            for key, column in zip(metrics, columns):
                plotted = sum(stack_heights(k_df, column, order))
                cumulative_column = f"cumulative_{column}"
                true = (
                    float(k_df.loc[k_df["component"] == "packing", cumulative_column].mean())
                    if cumulative_column in k_df.columns
                    else sum(
                        float(k_df.loc[k_df["component"] == c, column].mean())
                        for c in order
                    )
                )
                parts.append(f"{METRICS[key][1]}={plotted:5.1f}% (true {true:5.1f}%)")
            mean_patches = k_df.loc[
                k_df["component"] == "h_commutation", "patches_packed"
            ].mean()
            fell_back = int(
                k_df.loc[k_df["component"] == "packing", "pack_note"]
                .astype(str)
                .str.contains("naive-kept")
                .sum()
            )
            print(
                f"    k={k:<3d} {'  '.join(parts)}"
                f"   [{circuits} circuits, {mean_patches:.1f} patches avg, "
                f"{degenerate} single-patch"
                + (f", {fell_back} verified back to naive" if fell_back else "")
                + "]"
            )

    print("\n[code_size] per-technique contribution at each k (all groups pooled)")
    for k in sorted(df["k"].unique()):
        k_df = df.loc[df["k"] == k]
        fired = int(k_df.loc[k_df["component"] == "merging", "merging_fired"].sum())
        heights = {column: stack_heights(k_df, column, order) for column in columns}
        for index, (component, label, _) in enumerate(COMPONENTS):
            parts = [
                f"{METRICS[key][1]}={heights[column][index]:5.1f}%"
                for key, column in zip(metrics, columns)
            ]
            suffix = ""
            if component == "merging":
                total = int((k_df["component"] == "merging").sum())
                suffix = f"   [fired on {fired}/{total} circuits]"
            print(f"    k={k:<3d} {label:<15s} {'  '.join(parts)}{suffix}")

    never_fired = sorted(
        int(k)
        for k in df["k"].unique()
        if not df.loc[(df["k"] == k) & (df["component"] == "merging"), "merging_fired"].any()
    )
    if never_fired:
        print(
            f"\n[code_size] NOTE: merging fired on no circuit at k={never_fired}. The "
            "pass runs at every code size; whole-patch macros just need a gate on "
            "every occupied lane at once, which gets rarer as k grows."
        )
    if settings.min_patches > 1:
        print(
            f"[code_size] NOTE: aggregate restricted to placements with "
            f">= {settings.min_patches} patches."
        )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_results(
    df: pd.DataFrame, metrics: Sequence[str], settings: CodeSizeSettings
) -> None:
    groups = [group for group in SIZE_GROUPS if (df["size_group"] == group).any()]
    agg = aggregate(df, metrics, settings.min_patches)
    k_values = sorted(agg["k"].unique())

    bar_width = 0.8
    group_step = bar_width * 1.02
    k_step = len(groups) * group_step + 1.6
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2.0) * group_step

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(8.2, 2.5 * len(metrics) + 1.1),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, metrics):
        metric_column, metric_label, _ = METRICS[key]
        tops: List[float] = []
        for k_index, k in enumerate(k_values):
            for group_index, group in enumerate(groups):
                x = k_index * k_step + offsets[group_index]
                sub = agg.loc[(agg["size_group"] == group) & (agg["k"] == k)]
                if sub.empty:
                    continue
                bottom = 0.0
                for component, _, color in COMPONENTS:
                    match = sub.loc[sub["component"] == component, metric_column]
                    value = float(match.iloc[0]) if not match.empty else 0.0
                    ax.bar(
                        x,
                        value,
                        width=bar_width,
                        bottom=bottom,
                        color=color,
                        edgecolor="white",
                        linewidth=0.6,
                    )
                    bottom += value
                tops.append(bottom)

        ax.set_ylabel(metric_label)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Size-group initial inside the foot of each bar, as Figure 13 does. Placed
        # in axes-fraction y so it sits just above the spine at any y-scale.
        headroom = max(tops) if tops else 1.0
        ax.set_ylim(top=headroom * 1.18 if headroom > 0 else 1.0)
        for k_index in range(len(k_values)):
            for group_index, group in enumerate(groups):
                ax.annotate(
                    SIZE_GROUP_LABELS[group],
                    xy=(k_index * k_step + offsets[group_index], 0.0),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="white",
                )

    axes[-1].set_xticks([k_index * k_step for k_index in range(len(k_values))])
    axes[-1].set_xticklabels([str(k) for k in k_values])
    # Code name in plain text: mathtext commas render as semicolons under
    # `mathtext.fontset: cm` + `svg.fonttype: none`, so this read `[[k+2; k; 2]]` in the
    # SVG. `$k$` alone is verified safe. See `9_code_size_tradeoff` for the full diagnosis.
    axes[-1].set_xlabel("Logical qubits per patch, $k$   ([[k+2, k, 2]] code)")

    handles = [
        mpatches.Patch(facecolor=color, edgecolor="white", label=label)
        for _, label, color in COMPONENTS
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False, ncol=1)

    fig.supylabel("Percent reduction")
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(settings.output_path, dpi=200)
    pdf_path = settings.output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"[code_size] saved plot to {settings.output_path} and {pdf_path}")

    if settings.show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> CodeSizeSettings:
    parser = argparse.ArgumentParser(
        description="Sweep the [[k+2, k, 2]] code size and plot technique benefits."
    )
    parser.add_argument("--qasm-root", type=Path, default=DEFAULT_SETTINGS.qasm_root)
    parser.add_argument(
        "--groups", nargs="+", choices=SIZE_GROUPS, default=list(DEFAULT_SETTINGS.groups)
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_SETTINGS.k_values),
        help="Logical qubits per patch to sweep; code size is k + 2.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRICS),
        default=list(DEFAULT_SETTINGS.metrics),
        help="Metric rows to plot. lsr/tvd need simulation and are not yet wired.",
    )
    parser.add_argument("--max-gates", type=int, default=DEFAULT_SETTINGS.max_proxy_gates)
    parser.add_argument("--max-qubits", type=int, default=DEFAULT_SETTINGS.max_proxy_qubits)
    parser.add_argument(
        "--proxy-timeout", type=float, default=DEFAULT_SETTINGS.proxy_timeout_seconds
    )
    parser.add_argument(
        "--packer",
        choices=["packing", "raw"],
        default=DEFAULT_SETTINGS.packer,
        help="packing = the full pipeline with the two-translation verify against the "
        "naive placement, so the packing segment is what a caller actually gets (the "
        "default); raw = the same search with verify off, which is the number that says "
        "whether the algorithm improved. The 'analytic', 'measured' and 'legacy' packers "
        "went with the grouping and packer packages; see archive/README.md.",
    )
    parser.add_argument(
        "--partitioner",
        choices=sorted(PARTITIONERS),
        default=DEFAULT_SETTINGS.partitioner,
        help="Seed partitioner inside packing. See packing/seed.py, which records what "
        "each arm measured.",
    )
    parser.add_argument(
        "--min-patches",
        type=int,
        default=DEFAULT_SETTINGS.min_patches,
        help="Drop placements with fewer patches than this from the aggregate. "
        "Use 2 to exclude circuits that fit in a single patch.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_SETTINGS.output_path)
    parser.add_argument("--csv", type=Path, default=DEFAULT_SETTINGS.csv_path)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Replot from a saved CSV without recomputing.",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    return CodeSizeSettings(
        qasm_root=args.qasm_root,
        groups=args.groups,
        k_values=args.k_values,
        metrics=args.metrics,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        packer=args.packer,
        partitioner=args.partitioner,
        min_patches=args.min_patches,
        output_path=args.output,
        csv_path=args.csv,
        from_csv=args.from_csv,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()

    # Placement rejections and encode/decode warnings are expected across a sweep this
    # wide and are reported through `pack_note` and the summary, so keep them out of
    # the per-circuit output.
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if settings.from_csv is not None:
        df = read_csv(settings.from_csv)
        metrics = active_metrics(df, settings.metrics)
        print_summary(df, metrics, settings)
        plot_results(df, metrics, settings)
        return

    if settings.csv_path.exists():
        settings.csv_path.unlink()

    for group in settings.groups:
        circuits = load_group_circuits(group, settings)
        if not circuits:
            print(f"[code_size] no circuits loaded for {group}, skipping")
            continue
        qubit_range = (
            min(c.num_qubits for _, c in circuits),
            max(c.num_qubits for _, c in circuits),
        )
        print(
            f"[code_size] {group}: {len(circuits)} circuits, "
            f"{qubit_range[0]}-{qubit_range[1]} qubits"
        )
        group_df = run_experiment(circuits, group, settings)
        if not group_df.empty:
            append_csv(group_df, settings.csv_path)

    if not settings.csv_path.exists():
        raise RuntimeError(
            f"no data produced -- check --qasm-root ({settings.qasm_root}) and filters"
        )

    df = read_csv(settings.csv_path)
    metrics = active_metrics(df, settings.metrics)
    missing = [key for key in settings.metrics if key not in metrics]
    if missing:
        print(
            f"[code_size] metrics {missing} have no data (simulation not run); "
            "omitting those rows"
        )
    print_summary(df, metrics, settings)
    plot_results(df, metrics, settings)


if __name__ == "__main__":
    main()
