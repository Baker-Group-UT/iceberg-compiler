"""
ablation_vs_k.py -- what each compiler technique is worth as the code holds more qubits.

**Claim.** All three techniques keep paying across the [[k+2, k, 2]] family, and the mix
shifts: H commutation is roughly flat in k, merging thins out as whole-patch macros get
harder to fire, and packing has more placement freedom to exploit at small k.

One stacked bar per k, segments in pipeline order:

    1. H commutation
    2. + merging
    3. + packing

against a naive index-order packing at the *same* k with both passes off, so every bar
is a within-k ablation rather than a comparison across code families.

Random circuits rather than QASMBench, deliberately. The benchmark corpus is not
matched across widths, so a k sweep over it confounds "what k does" with "which circuits
happen to sit at this width". Seeded random circuits hold width, depth and gate mix fixed
while k moves, which is the only way the x axis means one thing.

**Width is chosen so the packing question survives the sweep.** At a fixed qubit count a
large enough k puts the whole program in one patch, at which point there is no placement
decision left and the packing segment reads zero for a structural reason rather than an
algorithmic one -- the archived code-size sweep hit exactly this and papered over it with
a `--min-patches` filter. The default width of 24 needs 3 patches even at k = 10, so
every point in the sweep is a real packing problem. `patches` is recorded per row and the
summary prints it, so the reader can check.

Translation only -- no simulation, so this is the cheap figure in the set.

**Structured circuits, no immutable gates.** Uniform-random two-qubit pairs give an
Erdos-Renyi interaction graph with no communities, so every partition is equally good and
the packing segment reads zero for a structural reason -- see
`experiments/shared/random_circuits.structured_program_circuit`. Blocks are planted on
shuffled labels, so recovering them is an achievement rather than a coincidence.
`block_size` is held fixed across the sweep rather than tracking k: the circuit has to be
the same at every point or the x axis confounds "what k does" with "what circuit we drew".
`immutable_ratio=0` because S and the rotations are inert here -- they neither merge nor
move with the placement, so they only add variance to a figure about k.

Run with:

    python -m experiments.figures.ablation_vs_k.ablation_vs_k
    python -m experiments.figures.ablation_vs_k.ablation_vs_k --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.figures import (
    COLUMN_WIDTH,
    save_figure,
    stack_heights,
    use_style,
)
from experiments.shared.random_circuits import (
    plant_blocks,
    structured_program_circuit,
)
from packing.passes import commutation_pays
from packing.plan import plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

NAME = "ablation_vs_k"
HERE = Path(__file__).resolve().parent

Perm = List[Tuple[int, int]]

#: (csv suffix, y-axis label). Depth is a count of layers, not of gates, so it gets its
#: own label rather than sharing the gate one.
METRICS: Dict[str, str] = {
    "one_qubit_gates": "1q gates",
    "two_qubit_gates": "2q gates",
    "circuit_depth": "depth",
}

#: Pipeline order, with the palette position each segment takes. The indices are not
#: 0/1/2: the published ablation figure draws H commutation dark, merging tan and packing
#: blue, so the accent lands on the technique this package is responsible for rather than
#: on the first one in the stack. Pinning the positions here keeps that agreement instead
#: of inheriting whatever order the cycler happens to be in.
#:
#: 1 = #3b4c56 dark slate, 2 = #cdc2c0 warm grey, 0 = #0097fd accent blue.
COMPONENTS: Sequence[Tuple[str, str, int]] = (
    ("h_commutation", "H commutation", 1),
    ("merging", "+ merging", 2),
    ("packing", "+ packing", 0),
)

DEFAULT_K_VALUES = (2, 4, 6, 8, 10)


@dataclass
class Settings:
    k_values: Sequence[int] = DEFAULT_K_VALUES
    num_qubits: int = 24
    gates_per_qubit: int = 14
    immutable_ratio: float = 0.0
    block_size: int = 4
    locality: float = 0.85
    circuits: int = 24
    seed: int = 20260729
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False


def build_config(*, n_code: int, h_commutation: bool, merging: bool) -> PipelineConfig:
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    return config


def naive_packing(num_qubits: int, k: int) -> Perm:
    """Fill patches in program order, k slots at a time -- the within-k baseline."""
    return [(index // k, index % k) for index in range(num_qubits)]


def measure(
    circuit: qiskit.QuantumCircuit, permutation: Perm, config: PipelineConfig
) -> Optional[Tuple[int, int, int]]:
    """(1q, 2q, depth) of the translated circuit, or None if the placement is rejected."""
    try:
        translated = translate_permutation(circuit, permutation, config=config)
    except Exception:  # noqa: BLE001 - rejection under the placement rules is data
        return None
    one_q, two_q = one_two_qubit_gate_totals(translated.circuit)
    return one_q, two_q, translated.circuit.depth()


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def compute(settings: Settings) -> pd.DataFrame:
    rows: List[Dict] = []
    points = [
        (seed_index, k)
        for seed_index in range(settings.circuits)
        for k in settings.k_values
    ]

    for seed_index, k in tqdm(points, desc="circuit x k"):
        n_code = k + 2
        blocks = plant_blocks(
            settings.num_qubits, settings.block_size, seed=settings.seed + seed_index
        )
        circuit = structured_program_circuit(
            settings.num_qubits,
            settings.gates_per_qubit,
            seed=settings.seed + seed_index,
            blocks=blocks,
            locality=settings.locality,
            immutable_ratio=settings.immutable_ratio,
            clifford_only=False,
        )
        naive = naive_packing(settings.num_qubits, k)

        # The pipeline decides per (circuit, code size) whether H commutation pays, and
        # skips it where it does not -- on QASMBench proxies that is 27 of 64 circuits at
        # n=4. Take that decision once here and thread it through all four rows, so the
        # ablation describes the compiler as it ships.
        #
        # This keeps the decomposition cumulative, which is the thing that would break if
        # only the packed row honoured the decision. When `commute` is False the
        # H-commutation row collapses onto the baseline and its segment is legitimately
        # zero -- the shipped pipeline did not run the pass on that circuit -- and merging
        # and packing are then both measured on the uncommuted branch. Forcing the pass on
        # instead would credit a technique the compiler declined to use.
        commute = commutation_pays(circuit, n_code)

        base = measure(circuit, naive, build_config(
            n_code=n_code, h_commutation=False, merging=False))
        h_only = measure(circuit, naive, build_config(
            n_code=n_code, h_commutation=commute, merging=False))
        h_merge = measure(circuit, naive, build_config(
            n_code=n_code, h_commutation=commute, merging=True))
        if base is None or h_only is None or h_merge is None:
            continue

        try:
            plan = plan_packing(circuit, n_code, h_commutation=commute, verify=True)
            packed_perm = plan.placement
        except Exception:  # noqa: BLE001
            packed_perm = naive
        packed = measure(circuit, packed_perm, build_config(
            n_code=n_code, h_commutation=commute, merging=True))
        if packed is None:
            packed, packed_perm = h_merge, naive

        cumulative = {
            "h_commutation": [pct_reduction(b, c) for b, c in zip(base, h_only)],
            "merging": [pct_reduction(b, c) for b, c in zip(base, h_merge)],
            "packing": [pct_reduction(b, c) for b, c in zip(base, packed)],
        }
        order = [component for component, _, _ in COMPONENTS]
        segments = {order[0]: cumulative[order[0]]}
        for earlier, later in zip(order, order[1:]):
            segments[later] = [
                after - before
                for after, before in zip(cumulative[later], cumulative[earlier])
            ]

        shared = {
            "seed": settings.seed + seed_index,
            "num_qubits": settings.num_qubits,
            "program_gates": circuit.size(),
            "k": k,
            "n_code": n_code,
            "patches_naive": len({patch for patch, _ in naive}),
            "patches_packed": len({patch for patch, _ in packed_perm}),
            "merging_fired": h_merge != h_only,
            "h_commutation_chosen": commute,
        }
        for component, values in segments.items():
            row = dict(shared, component=component)
            for metric, value in zip(METRICS, values):
                row[f"pct_reduction_{metric}"] = value
            for metric, value in zip(METRICS, cumulative[component]):
                row[f"cumulative_pct_reduction_{metric}"] = value
            rows.append(row)

    return pd.DataFrame(rows)


def print_summary(frame: pd.DataFrame) -> None:
    order = [component for component, _, _ in COMPONENTS]
    print(f"\n[{NAME}] stack heights, mean over circuits")
    for k in sorted(frame["k"].unique()):
        block = frame.loc[frame["k"] == k]
        circuits = block["seed"].nunique()
        patches = block.loc[block["component"] == "packing", "patches_packed"].mean()
        parts = []
        for metric, label in METRICS.items():
            column = f"pct_reduction_{metric}"
            heights = stack_heights(block, column, order)
            parts.append(f"{label}={sum(heights):5.1f}%")
        fired = int(block.loc[block["component"] == "merging", "merging_fired"].sum())
        commuted = int(block.loc[block["component"] == "merging", "h_commutation_chosen"].sum())
        print(
            f"  k={k:<3d} {'  '.join(parts)}"
            f"   [{circuits} circuits, {patches:.1f} patches, "
            f"merging fired {fired}/{circuits}, commutation chosen {commuted}/{circuits}]"
        )


def plot(frame: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    order = [component for component, _, _ in COMPONENTS]
    k_values = sorted(frame["k"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=1,
        # Three stacked panels at column width. Height is per-panel plus room for the
        # shared x label; the style's default 2.4in is for a single-panel figure.
        figsize=(COLUMN_WIDTH, 1.15 * len(METRICS) + 0.85),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for axis, (metric, label) in zip(axes, METRICS.items()):
        column = f"pct_reduction_{metric}"
        cumulative_column = f"cumulative_{column}"
        tops: List[float] = []
        for position, k in enumerate(k_values):
            block = frame.loc[frame["k"] == k]
            heights = stack_heights(block, column, order)
            bottom = 0.0
            for index, (component, series_label, colour) in enumerate(COMPONENTS):
                axis.bar(
                    position,
                    heights[index],
                    width=0.68,
                    bottom=bottom,
                    color=palette[colour],
                    edgecolor="white",
                    linewidth=0.5,
                    label=series_label if position == 0 else None,
                )
                bottom += heights[index]
            tops.append(bottom)

            # One error bar on the total, not per segment: the segments are differences
            # between clipped cumulative means and their spreads are not independent,
            # so stacking three of them would overstate what is known. The spread that
            # means something is the full-pipeline reduction across circuits.
            spread = float(
                block.loc[block["component"] == "packing", cumulative_column].std(ddof=0)
            )
            if np.isfinite(spread):
                axis.errorbar(
                    position, bottom, yerr=spread,
                    fmt="none", ecolor="#636363", elinewidth=0.8, capsize=2,
                )

        axis.set_ylabel(label)
        axis.set_ylim(bottom=0.0)
        axis.set_axisbelow(True)

    axes[-1].set_xticks(range(len(k_values)))
    axes[-1].set_xticklabels([str(k) for k in k_values])
    axes[-1].set_xlabel("logical qubits per patch, $k$   ($[[k{+}2,\\,k,\\,2]]$ code)")

    handles = [
        mpatches.Patch(facecolor=palette[colour], edgecolor="white", label=label)
        for _, label, colour in COMPONENTS
    ]
    # Outside the axes rather than inside panel 0: the tallest stack is at the left of
    # the top panel, which is exactly where an inside legend wants to sit. Constrained
    # layout reserves the strip, so this cannot collide at any data range.
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.2)
    fig.supylabel("reduction against naive packing (%)")

    paths = save_figure(fig, NAME, Path(__file__), settings.output_override)
    plt.close(fig)
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values),
                        help="Logical qubits per patch to sweep; code size is k + 2.")
    parser.add_argument("--num-qubits", type=int, default=defaults.num_qubits,
                        help="Program width. Must exceed max(k) or the largest code "
                             "fits the whole program in one patch and packing has "
                             "nothing to decide.")
    parser.add_argument("--gates-per-qubit", type=int, default=defaults.gates_per_qubit)
    parser.add_argument("--immutable-ratio", type=float, default=defaults.immutable_ratio,
                        help="Share of S/Rx/Rz, which merging cannot fuse. Zero here: "
                             "they are inert for a k sweep and only add variance.")
    parser.add_argument("--block-size", type=int, default=defaults.block_size,
                        help="Planted community size, held fixed across the sweep so "
                             "the circuit does not change with k.")
    parser.add_argument("--locality", type=float, default=defaults.locality,
                        help="Probability a 2q gate stays inside its block. 1.0 is "
                             "perfectly separable; ~0.27 reproduces uniform-random.")
    parser.add_argument("--circuits", type=int, default=defaults.circuits,
                        help="Random circuits averaged per k.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="Replot from cached data without recomputing.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/. Worth it only if the "
                             "run was expensive; this one takes minutes, so usually not.")
    args = parser.parse_args()
    return Settings(
        k_values=args.k_values,
        num_qubits=args.num_qubits,
        gates_per_qubit=args.gates_per_qubit,
        immutable_ratio=args.immutable_ratio,
        block_size=args.block_size,
        locality=args.locality,
        circuits=args.circuits,
        seed=args.seed,
        csv_path=args.csv,
        from_csv=args.from_csv,
        output_override=args.output_dir,
        persist=args.persist,
    )


def main() -> None:
    settings = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if settings.from_csv is not None:
        frame = pd.read_csv(settings.from_csv)
    else:
        if settings.num_qubits <= max(settings.k_values):
            raise ValueError(
                f"--num-qubits {settings.num_qubits} is not larger than max k "
                f"{max(settings.k_values)}; the largest code would need one patch and "
                "the packing segment would be structurally zero."
            )
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every placement was rejected")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            from experiments.shared.figures import data_dir

            target = data_dir(Path(__file__)) / f"{NAME}.csv"
            frame.to_csv(target, index=False)
            print(f"[{NAME}] persisted to {target}")

    print_summary(frame)
    for path in plot(frame, settings):
        print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
