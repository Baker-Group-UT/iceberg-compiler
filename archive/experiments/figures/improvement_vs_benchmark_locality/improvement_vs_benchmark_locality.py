"""
improvement_vs_benchmark_locality.py -- the packing claim on real circuits, not synthetic ones.

**Claim, and it is not the one this figure was built to make.** The mean improvement
survives contact with real programs -- +6.9% selection rate, +3.0% two-qubit gates, +7.7%
depth over 32 QASMBench circuits -- but *locality does not predict which circuit benefits*.

This is the benchmark counterpart to `improvement_vs_locality_and_k`, and it was built
expecting the synthetic figure's locality gradient to reappear. It does not:

                     synthetic r(locality)   benchmark r(locality)   partial r | width
    selection rate          0.91                    0.04                  0.01
    2q gates                0.93                    0.35                  0.26
    depth                   0.86                    0.16                  0.19
    TVD                    -0.89                    0.08                  0.11

Only 2q keeps a weak positive, and it is the mechanism metric, so the direction is at
least right. Everything else is flat.

The reading that fits both figures: the synthetic sweep holds width, depth and gate mix
fixed by construction and moves locality alone, so it measures locality's effect *all else
equal* -- which is real, and is what a controlled sweep is for. Real benchmarks differ in
all of those at once, and locality explains little of the variance that results. So the
synthetic figure supports "locality is what packing exploits"; it does not support
"locality tells you which benchmark will benefit". Only this figure can speak to the
second question, and its answer is no.

**Locality is confounded with width in this corpus**, at r = -0.69: narrow circuits are
trivially local (mean 0.86 at n = 4-6, against 0.41 at n = 17-24), and six circuits are
no wider than the grouping capacity, so their locality is 1.0 by definition. Controlling
for width does not rescue the correlation -- the partial column above -- so the confound
is real but it is not what is hiding the effect. There is no effect to hide.

Same comparison, same pipeline, same five metrics as the sibling; the only difference is
that locality is *measured* here rather than requested.

**Not a heatmap, and that is forced by the corpus rather than chosen.** QASMBench yields 32
circuits that are Clifford-proxyable, wide enough to need more than one patch, and small
enough to simulate. Against a 6x5 grid that is about one circuit per cell, and cells in the
high-locality column empty out entirely at large k because the circuits up there are narrow.
A grid needs samples per cell; a scatter needs samples per axis, and 32 circuits x 5 code
sizes is a comfortable scatter. So the panels keep the metrics and the k series of the
sibling figure and drop the second spatial dimension. `--heatmap` renders the coarse binned
grid anyway, for direct visual comparison against the synthetic figure -- read the printed
cell counts before believing any cell of it.

**Locality is capacity-constrained intra-patch fraction, and the obvious metric is wrong.**
The natural measure -- run community detection, ask what share of two-qubit gates stay
inside a community -- calls a star graph perfectly local. `bv_n19` is one ancilla talking to
eighteen data qubits: Louvain puts the whole thing in one community and scores it 1.000,
when a star is the *least* packable graph there is, because the hub has to cross to every
patch. Measured, on this corpus:

                     louvain-intra   capacity-4 intra
    bv_n19               1.000            0.111
    qft_n18              1.000            0.131
    bv_n14               1.000            0.231
    ising_n42            0.878            0.634
    simon_n6             0.929            0.929

So locality here is the best intra-group two-qubit share achievable with groups of at most
`--locality-capacity` qubits, which is what "can this circuit be packed" actually means.
Groups come from recursive Kernighan-Lin bisection in networkx, deliberately *not* from
`packing`'s own partitioner: the x axis must not be produced by the thing the y axis
evaluates, or the figure is scoring the packer against its own opinion of the circuit.

The capacity is fixed at 4 rather than tracked to `k`, so one circuit has one locality and
the x axis does not shift under the series. It is a structural property of the program --
"how much of this circuit's traffic confines to groups of four" -- not a statement about any
particular code.

**Filtering is done on the loaded circuit, not the README.** `find_qasm_files` reads its
gate and qubit limits out of each benchmark's README and treats a missing or unparseable
one as "no limit", so `qubit_limit=20` still returns `n=100` and a 294k-gate circuit. Every
bound here is applied to the transpiled proxy instead.

**This is the figure that can carry the TVD claim, and the sibling is not.** The sibling
disclaims TVD because at 20 qubits its noiseless floor sits near 0.08 and swamps the
effect. That is a fact about width against shot budget, not about TVD: this corpus has a
13-qubit median and a noiseless floor of 0.003 against a measured TVD of 0.102, so only
about 3% of the signal is sampling. Over the 102 pairs where the floor is demonstrably
below the effect:

    measured                 +11.85%
    shot-count artefact       +1.94%   (the two arms accept different shot counts)
    residual                  +9.91%  +/- 1.83   (t = +5.4)

So AQORN improves TVD by roughly 10% on real circuits, and that survives the same
correction that reduced the synthetic figure's apparent gain to nothing.

Two things had to be fixed before this number was trustworthy, and both moved it a lot.
Including the miscompiled `qpe_n9` pair put the residual at +0.8% +/- 8.4 -- one bad point
with a -922% ratio dominating 111 others. Applying a blanket floor warning, as the sibling
does, would have suppressed a real result. `_report_tvd` therefore measures the floor per
point and decides from the data whether the number can be quoted, rather than assuming.

**This figure found a miscompilation bug, and guards against it.** On `qpe_n9` at k = 2 the
combination (AQORN placement, merging on, H commutation off) produces a circuit whose
noiseless TVD is 0.87 against naive's 0.01 -- it computes the wrong distribution -- while
the selection rate stays at 1.0, so post-selection does not catch it. Stim and Aer agree,
so it is translation rather than simulation, and `plan_packing(verify=True)` does not catch
it either because that compares modelled cost, not correctness. It is specific to that
combination: the same placement with H commutation *on*, or with merging off, is fine.

22% of this corpus' AQORN rows run that configuration, so `noiseless_tvd` simulates every
point at zero noise first and `paired_improvement` drops pairs where AQORN departs from
naive, printing which. The comparison is *differential* and the first version of it was
not, which cost nine good pairs: an absolute cutoff mistakes a wide circuit's sampling
floor for a fault, and 8k shots over `qft_n18`'s 18-qubit output leaves both arms at 0.97
with nothing wrong. Both arms share that floor exactly, so only a gap between them means
anything.

That guard belongs in `experiments/shared/simulation.py` next to `evaluate`, so every
simulated figure gets it; it is local here only because that file is being edited
concurrently.

Run with:

    python -m experiments.figures.improvement_vs_benchmark_locality.improvement_vs_benchmark_locality
    python -m experiments.figures.improvement_vs_benchmark_locality.improvement_vs_benchmark_locality --from-csv <path>
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import qiskit
from tqdm.auto import tqdm

from experiments.shared.benchmark_circuits import create_proxy_from_qasm, find_qasm_files
from experiments.shared.figures import FULL_WIDTH, save_figure, use_style
from experiments.shared.simulation import (
    CORRECTNESS_MARGIN,
    CORRECTNESS_RATIO,
    CORRECTNESS_REPETITIONS,
    TVD_RESOLUTION_LIMIT,
    METRICS,
    SimSettings,
    add_sim_arguments,
    baseline_gate_count,
    build_config,
    calibrate,
    evaluate,
    naive_packing,
    noise_profile_for,
    noiseless_tvd,
    settings_from_args,
)
from packing.plan import plan_packing

NAME = "improvement_vs_benchmark_locality"
HERE = Path(__file__).resolve().parent

Perm = List[Tuple[int, int]]

#: Same metrics, same signs, same panel names as `improvement_vs_locality_and_k`, so the
#: two figures can be read against each other without translating.
PANELS: Dict[str, Tuple[str, bool]] = {
    "lsr": ("selection rate", True),
    "tvd": ("TVD", False),
    "one_qubit_gates": ("1q gates", False),
    "two_qubit_gates": ("2q gates", False),
    "circuit_depth": ("depth", False),
}

_UNKNOWN = set(PANELS) - set(METRICS)
if _UNKNOWN:
    raise ImportError(f"PANELS names metrics evaluate() does not return: {sorted(_UNKNOWN)}")

LAYOUTS: Dict[str, Tuple[int, int, Sequence[str]]] = {
    "": (1, 2, ("lsr", "tvd")),
    "_depth": (1, 3, ("lsr", "tvd", "circuit_depth")),
    "_quad": (2, 2, ("lsr", "tvd", "two_qubit_gates", "circuit_depth")),
}

DEFAULT_K_VALUES = (2, 4, 6, 8, 10)

#: Bin edges for `--heatmap` and for the printed summary. Coarser than the synthetic
#: figure's six columns because there are 32 circuits to spread across them.
LOCALITY_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)


@dataclass
class Settings:
    benchmark_root: Path = field(default_factory=lambda: Path("benchmarks/qasmbench"))
    k_values: Sequence[int] = DEFAULT_K_VALUES
    #: Applied to the loaded proxy. Lower bound so there is a placement to make at all,
    #: upper bound so the simulation finishes.
    min_qubits: int = 4
    max_qubits: int = 24
    max_gates: int = 3000
    min_two_qubit_gates: int = 8
    locality_capacity: int = 4
    sim: SimSettings = field(default_factory=SimSettings)
    heatmap: bool = False
    csv_path: Path = field(default_factory=lambda: HERE / f"{NAME}.csv")
    from_csv: Optional[Path] = None
    output_override: Optional[Path] = None
    persist: bool = False


# --- structure analysis -------------------------------------------------------------

def interaction_graph(circuit: qiskit.QuantumCircuit) -> nx.Graph:
    """Qubits as nodes, two-qubit gate counts as edge weights. Self-pairs dropped."""
    weights: Counter = Counter()
    for instruction in circuit.data:
        if len(instruction.qubits) != 2:
            continue
        left, right = sorted(circuit.find_bit(q).index for q in instruction.qubits)
        if left != right:
            weights[(left, right)] += 1
    graph = nx.Graph()
    graph.add_nodes_from(range(circuit.num_qubits))
    for (left, right), count in weights.items():
        graph.add_edge(left, right, weight=count)
    return graph


def capacity_intra_fraction(graph: nx.Graph, capacity: int, seed: int = 7) -> float:
    """
    Best share of two-qubit gates confinable to groups of at most `capacity` qubits.

    Recursive Kernighan-Lin bisection: split, then split each half, until every part fits
    the capacity. KL is a local-refinement heuristic, so this is a lower bound on what is
    achievable rather than the optimum -- which is the right side to err on, since an
    over-optimistic locality would flatter the correlation this figure is measuring.

    Deliberately networkx rather than `packing.seed.graph_seed`: using the packer's own
    partitioner to define the axis it is then scored against would make the figure
    self-referential.
    """
    if graph.number_of_edges() == 0:
        return 0.0

    def split(nodes: List[int]) -> List[set]:
        if len(nodes) <= capacity:
            return [set(nodes)]
        subgraph = graph.subgraph(nodes)
        try:
            left, right = nx.community.kernighan_lin_bisection(
                subgraph, weight="weight", seed=seed
            )
        except Exception:  # noqa: BLE001 - disconnected or too small to bisect
            ordered = list(nodes)
            half = len(ordered) // 2
            left, right = set(ordered[:half]), set(ordered[half:])
        if not left or not right:
            ordered = list(nodes)
            half = max(1, len(ordered) // 2)
            left, right = set(ordered[:half]), set(ordered[half:])
        return split(list(left)) + split(list(right))

    owner = {node: index for index, group in enumerate(split(list(graph.nodes))) for node in group}
    total = sum(data["weight"] for _, _, data in graph.edges(data=True))
    if total <= 0:
        return 0.0
    inside = sum(
        data["weight"] for a, b, data in graph.edges(data=True) if owner[a] == owner[b]
    )
    return inside / total


#: The guard moved to `experiments.shared.simulation`, which is where this figure's
#: docstring had been saying it belonged: 8 of the 10 figures that pass a noise profile
#: had no guard at all. `noiseless_tvd` now takes a `SimSettings` rather than this
#: module's `Settings`, so the call sites pass `settings.sim`.


@dataclass(frozen=True)
class Benchmark:
    name: str
    circuit: qiskit.QuantumCircuit
    locality: float
    two_qubit_gates: int


def load_benchmarks(settings: Settings) -> List[Benchmark]:
    """
    Proxy every QASMBench circuit and keep the ones this figure can actually use.

    Bounds are checked against the loaded proxy rather than `find_qasm_files`' README
    limits, which silently pass anything whose README is missing or unparseable.
    """
    found: List[Benchmark] = []
    paths = find_qasm_files(settings.benchmark_root)
    for path in tqdm(sorted(paths), desc="loading benchmarks"):
        circuit = create_proxy_from_qasm(path, timeout_seconds=10.0)
        if circuit is None:
            continue
        if not settings.min_qubits <= circuit.num_qubits <= settings.max_qubits:
            continue
        if circuit.size() > settings.max_gates:
            continue
        graph = interaction_graph(circuit)
        two_qubit = int(sum(data["weight"] for _, _, data in graph.edges(data=True)))
        if two_qubit < settings.min_two_qubit_gates:
            continue
        found.append(
            Benchmark(
                name=path.stem,
                circuit=circuit,
                locality=capacity_intra_fraction(graph, settings.locality_capacity),
                two_qubit_gates=two_qubit,
            )
        )
    return found


# --- measurement --------------------------------------------------------------------

def compute(settings: Settings) -> pd.DataFrame:
    benchmarks = load_benchmarks(settings)
    if not benchmarks:
        raise RuntimeError("no usable benchmark circuits found")
    print(f"[{NAME}] {len(benchmarks)} usable circuits, "
          f"locality {min(b.locality for b in benchmarks):.2f}"
          f"-{max(b.locality for b in benchmarks):.2f}")

    # Only pairs with a placement to make. A circuit narrower than the code sits in one
    # patch, where naive and AQORN are the same permutation and the row is a tautology.
    points = [
        (bench, k)
        for bench in benchmarks
        for k in settings.k_values
        if ceil(bench.circuit.num_qubits / k) >= 2
    ]
    skipped = len(benchmarks) * len(settings.k_values) - len(points)
    if skipped:
        print(f"[{NAME}] skipping {skipped} (circuit, k) pairs that need only one patch")

    rows: List[Dict] = []
    for bench, k in tqdm(points, desc="benchmark x k"):
        n_code = k + 2
        naive = naive_packing(bench.circuit.num_qubits, k)
        calibration = calibrate(
            bench.circuit, naive, settings.sim,
            n_code=n_code,
            baseline_gate_count=baseline_gate_count(bench.circuit, naive, n_code),
        )
        try:
            plan = plan_packing(
                bench.circuit, n_code, h_commutation="auto", verify=True,
                noise=noise_profile_for(settings.sim),
            )
            aqorn, commute = plan.placement, plan.h_commutation
        except Exception:  # noqa: BLE001
            aqorn, commute = naive, True

        config = build_config(
            settings.sim, n_code=n_code,
            alpha=calibration.alpha, repetitions=calibration.repetitions,
            h_commutation=commute, merging=True,
        )
        measured = {
            "naive": evaluate(bench.circuit, naive, config),
            "aqorn": evaluate(bench.circuit, aqorn, config),
        }
        if any(value is None for value in measured.values()):
            continue

        # Check the compile before trusting the measurement, not after.
        clean = {
            "naive": noiseless_tvd(bench.circuit, naive, settings.sim,
                                   n_code=n_code, h_commutation=commute),
            "aqorn": noiseless_tvd(bench.circuit, aqorn, settings.sim,
                                   n_code=n_code, h_commutation=commute),
        }

        for series, result in measured.items():
            placement = naive if series == "naive" else aqorn
            rows.append({
                "circuit": bench.name,
                "num_qubits": bench.circuit.num_qubits,
                "program_gates": bench.circuit.size(),
                "program_two_qubit_gates": bench.two_qubit_gates,
                "locality": bench.locality,
                "locality_capacity": settings.locality_capacity,
                "k": k,
                "n_code": n_code,
                "series": series,
                "patches": len({patch for patch, _ in placement}),
                "alpha": calibration.alpha,
                "repetitions": calibration.repetitions,
                "h_commutation_chosen": commute,
                "noiseless_tvd": clean[series],
                **{metric: result[metric] for metric in PANELS},
            })
    return pd.DataFrame(rows)


def drop_miscompiled(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove pairs whose AQORN arm miscompiles, and flag pairs whose TVD is unresolvable.

    Applied once, in `main`, so the summary statistics and the plotted points describe
    the same set of rows -- running it inside `paired_improvement` left the TVD
    decomposition quoting 112 pairs under a header that said 111.
    """
    if "noiseless_tvd" in frame.columns:
        clean = frame.pivot_table(index=["circuit", "k"], columns="series",
                                  values="noiseless_tvd")
        if {"naive", "aqorn"}.issubset(clean.columns):
            base = clean["naive"]
            broken = clean["aqorn"] > np.maximum(
                CORRECTNESS_RATIO * base, base + CORRECTNESS_MARGIN
            )
            if broken.any():
                names = ", ".join(f"{c}@k={k}" for c, k in clean.index[broken])
                print(f"  MISCOMPILED, dropped {int(broken.sum())} pair(s): {names}")
                print(f"    (AQORN's noiseless TVD exceeds naive's by more than "
                      f"{CORRECTNESS_RATIO}x or {CORRECTNESS_MARGIN}; a placement cannot "
                      f"change the noiseless result, so the translation is wrong)")
                keep = set(clean.index[~broken])
                frame = frame[[(c, k) in keep
                               for c, k in zip(frame["circuit"], frame["k"])]]
            unresolved = clean.loc[
                clean.index.isin({(c, k) for c, k in zip(frame["circuit"], frame["k"])})
            ]
            coarse = unresolved[unresolved.min(axis=1) > TVD_RESOLUTION_LIMIT]
            if len(coarse):
                print(f"  TVD unresolved for {len(coarse)} pair(s) "
                      f"({', '.join(sorted({c for c, _ in coarse.index}))}): the noiseless "
                      f"reference is itself at its sampling floor, so read gates/depth/LSR "
                      f"there and ignore TVD. Rows kept.")

    return frame


def paired_improvement(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(circuit, k) percent improvement of AQORN over naive, signed so up is better.

    Assumes `drop_miscompiled` has already run: a miscompiled arm is not a bad result,
    it is an absent one, and `qpe_n9` at k = 2 would otherwise contribute a -922% TVD
    point that is entirely a translation fault.
    """
    wide = frame.pivot_table(
        index=["circuit", "locality", "num_qubits", "k"],
        columns="series",
        values=list(PANELS),
    )
    rows: List[Dict] = []
    for (circuit, locality, num_qubits, k), row in wide.iterrows():
        record = {"circuit": circuit, "locality": locality,
                  "num_qubits": num_qubits, "k": k}
        for metric, (_, higher_is_better) in PANELS.items():
            naive, aqorn = row[(metric, "naive")], row[(metric, "aqorn")]
            if not naive:
                record[metric] = np.nan
                continue
            gain = (aqorn - naive) if higher_is_better else (naive - aqorn)
            record[metric] = 100.0 * gain / naive
        rows.append(record)
    return pd.DataFrame(rows)


def _report_tvd(frame: pd.DataFrame) -> None:
    """
    Decompose the TVD result, and decide from the data whether it can be quoted.

    The sibling figure has to disclaim TVD entirely: at 20 qubits its noiseless floor is
    around 0.08, most of what it calls TVD is sampling error, and the two arms sit on
    different floors because they accept different shot counts. None of that is a fact
    about TVD -- it is a fact about width against shot budget. This corpus is mostly
    narrow (13-qubit median) and its noiseless floor is near 0.005, so the same budget
    resolves it comfortably and the number here is real.

    So test resolution rather than assume it either way: `noiseless_tvd` is the floor,
    measured per point, and the verdict follows from comparing it to the effect.
    """
    accepted = frame["lsr"] * frame["repetitions"]
    by_series = frame.assign(accepted=accepted).groupby("series")["accepted"].mean()
    print("  accepted shots/point: "
          + ", ".join(f"{s} {v:,.0f}" for s, v in by_series.items()))

    values = ["tvd", "accepted"] + (["noiseless_tvd"] if "noiseless_tvd" in frame else [])
    wide = frame.assign(accepted=accepted).pivot_table(
        index=["circuit", "k"], columns="series", values=values
    )
    if "noiseless_tvd" in frame:
        wide = wide[wide[("noiseless_tvd", "naive")] <= TVD_RESOLUTION_LIMIT]
    if wide.empty:
        return
    measured = 100.0 * (wide[("tvd", "naive")] - wide[("tvd", "aqorn")]) / wide[("tvd", "naive")]
    artefact = 100.0 * (1.0 - np.sqrt(wide[("accepted", "naive")] / wide[("accepted", "aqorn")]))
    residual = (measured - artefact).replace([np.inf, -np.inf], np.nan).dropna()
    if residual.empty:
        return
    spread = residual.std(ddof=1) / max(np.sqrt(len(residual)), 1.0)
    print(f"  TVD over {len(residual)} well-resolved pairs: measured {measured.mean():+.2f}%, "
          f"shot-count artefact {artefact.mean():+.2f}%, "
          f"residual {residual.mean():+.2f}% +/- {spread:.2f} "
          f"(t = {residual.mean() / spread:+.1f})")

    floor = wide[("noiseless_tvd", "naive")].median() if "noiseless_tvd" in frame else None
    if floor is not None:
        typical = frame.loc[frame["series"] == "naive", "tvd"].median()
        print(f"    noiseless floor {floor:.4f} against a measured TVD of {typical:.4f} "
              f"-- {100 * floor / max(typical, 1e-9):.0f}% of the signal is sampling")
        if abs(residual.mean()) > 2 * spread and floor < 0.25 * typical:
            print("    TVD is resolved on this corpus and the residual is significant; "
                  "this one is quotable, unlike the sibling figure's.")
        else:
            print("    WARNING: too close to the floor to quote.")


def print_summary(frame: pd.DataFrame, improvement: pd.DataFrame) -> None:
    circuits = improvement["circuit"].nunique()
    print(f"\n[{NAME}] AQORN improvement over naive (%), {circuits} QASMBench circuits, "
          f"{len(improvement)} (circuit, k) pairs")
    _report_tvd(frame)

    print("\n  correlation with measured locality, and mean by k:")
    print(f"{'metric':<16}{'mean':>8}{'r(locality)':>13}   " +
          "".join(f"{'k=' + str(k):>8}" for k in sorted(improvement['k'].unique())))
    for metric, (label, _) in PANELS.items():
        values = improvement[[metric, "locality", "k"]].dropna()
        r = (np.corrcoef(values["locality"], values[metric])[0, 1]
             if len(values) > 2 else float("nan"))
        by_k = [values.loc[values["k"] == k, metric].mean()
                for k in sorted(improvement["k"].unique())]
        print(f"{label:<16}{values[metric].mean():>8.2f}{r:>13.2f}   " +
              "".join(f"{v:>8.1f}" for v in by_k))

    print("\n  circuits per locality bin (what a heatmap cell would average over):")
    counts = pd.cut(improvement.drop_duplicates("circuit")["locality"],
                    bins=list(LOCALITY_BINS), right=False).value_counts().sort_index()
    for interval, count in counts.items():
        print(f"    [{interval.left:.1f}, {interval.right:.1f})  {count:>3}  {'#' * int(count)}")


# --- plotting -----------------------------------------------------------------------

def plot_layout(
    improvement: pd.DataFrame, settings: Settings, suffix: str,
    nrows: int, ncols: int, metrics: Sequence[str],
) -> List[Path]:
    k_values = sorted(improvement["k"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = plt.rcParams["axes.prop_cycle"].by_key()["marker"]

    panel = (FULL_WIDTH - 0.6) / ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(FULL_WIDTH, nrows * panel * 0.85 + 0.75),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for index, (axis, metric) in enumerate(zip(axes, metrics)):
        label, _ = PANELS[metric]
        # Zero first and underneath: "no better than naive" is the reference the eye
        # measures against, as in every other figure in this suite.
        axis.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0, zorder=1)
        for position, k in enumerate(k_values):
            block = improvement.loc[improvement["k"] == k, ["locality", metric]].dropna()
            if block.empty:
                continue
            axis.scatter(
                block["locality"], block[metric],
                s=14, color=palette[position % len(palette)],
                marker=markers[position % len(markers)],
                alpha=0.75, linewidths=0.0, zorder=3,
                label=f"$k={k}$" if index == 0 else None,
            )
        # One pooled trend across all k, because the claim is about locality and the
        # per-k subsets are too small to fit separately without overstating them.
        allk = improvement[["locality", metric]].dropna()
        if len(allk) > 2:
            slope, intercept = np.polyfit(allk["locality"], allk[metric], 1)
            grid = np.linspace(allk["locality"].min(), allk["locality"].max(), 50)
            axis.plot(grid, slope * grid + intercept, color="#3b4c56",
                      linewidth=1.2, linestyle="-", marker="", zorder=4)
            r = np.corrcoef(allk["locality"], allk[metric])[0, 1]
            # Opaque background: a gridline running through the label reads as a
            # strikethrough, which on a number is actively confusing.
            axis.annotate(
                f"$r={r:.2f}$", xy=(0.04, 0.92), xycoords="axes fraction",
                fontsize=plt.rcParams["font.size"] * 0.8, color="#3b4c56",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            )
        # A single point can be an order of magnitude past the rest -- one bad placement
        # on a small-TVD circuit -- and letting it set the limits compresses every other
        # point into a band. Clip to a robust range and say how many fell outside, rather
        # than dropping them from the data or rescaling the panel around them.
        column = improvement[metric].dropna()
        if len(column) > 4:
            low, high = np.percentile(column, [2, 98])
            pad = 0.08 * max(high - low, 1e-9)
            low, high = min(low - pad, 0.0), high + pad
            outside = int(((column < low) | (column > high)).sum())
            axis.set_ylim(low, high)
            if outside:
                axis.annotate(
                    f"{outside} off scale", xy=(0.96, 0.06), xycoords="axes fraction",
                    ha="right", fontsize=plt.rcParams["font.size"] * 0.8, color="#636363",
                )
        axis.set_title(f"({chr(ord('a') + index)})", loc="left", fontweight="bold")
        axis.set_title(label, loc="center")
        axis.set_axisbelow(True)

    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    fig.supxlabel(f"measured locality   (2q share confinable to groups of "
                  f"{settings.locality_capacity})")
    fig.supylabel("AQORN improvement over naive packing (%)")
    # Legend above the panels, not below. "outside lower center" still collides with
    # supxlabel -- constrained layout reserves a strip for each but does not stack them --
    # and the bottom strip is spoken for. Above the titles it reads as a key for the
    # figure and has the row to itself.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=len(handles))

    paths = save_figure(fig, f"{NAME}{suffix}", Path(__file__), settings.output_override)
    plt.close(fig)
    return paths


def heatmap_layout(
    improvement: pd.DataFrame, settings: Settings, suffix: str,
    nrows: int, ncols: int, metrics: Sequence[str],
) -> List[Path]:
    """
    The sibling figure's grid, on binned benchmarks. Provided for direct comparison.

    Every cell is annotated with the number of circuits behind it, not the value, because
    that is the number a reader needs in order to discount this rendering correctly: the
    corpus gives roughly one circuit per cell, and a one-circuit cell drawn in the same
    ink as a ten-circuit cell is the whole reason the scatter is the default.
    """
    from experiments.figures.improvement_vs_locality_and_k.improvement_vs_locality_and_k import (
        _edges, _scale,
    )

    centres = [0.5 * (LOCALITY_BINS[i] + LOCALITY_BINS[i + 1])
               for i in range(len(LOCALITY_BINS) - 1)]
    k_values = sorted(improvement["k"].unique())
    binned = improvement.assign(
        bin=pd.cut(improvement["locality"], bins=list(LOCALITY_BINS),
                   right=False, labels=range(len(centres)))
    )

    def grid_for(metric: str) -> Tuple[np.ndarray, np.ndarray]:
        values = np.full((len(k_values), len(centres)), np.nan)
        counts = np.zeros_like(values)
        for row, k in enumerate(k_values):
            for column in range(len(centres)):
                cell = binned.loc[(binned["k"] == k) & (binned["bin"] == column), metric].dropna()
                counts[row, column] = len(cell)
                if len(cell):
                    values[row, column] = float(cell.mean())
        return values, counts

    panel = (FULL_WIDTH - 0.9) / ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(FULL_WIDTH, nrows * panel * 0.92 + 0.55),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x_edges, y_edges = _edges(centres), _edges(k_values)

    for index, (axis, metric) in enumerate(zip(axes, metrics)):
        values, counts = grid_for(metric)
        cmap, norm = _scale(values)
        mesh = axis.pcolormesh(x_edges, y_edges, values, cmap=cmap, norm=norm,
                               shading="flat", edgecolors="white", linewidth=0.4)
        for row, k in enumerate(k_values):
            for column, centre in enumerate(centres):
                count = int(counts[row, column])
                value = values[row, column]
                shade = (0.0 if value != value
                         else 0.299 * cmap(float(norm(value)))[0]
                              + 0.587 * cmap(float(norm(value)))[1]
                              + 0.114 * cmap(float(norm(value)))[2])
                axis.text(centre, k, f"n={count}", ha="center", va="center",
                          fontsize=plt.rcParams["font.size"] * 0.8,
                          color="white" if (value == value and shade < 0.55) else "#1a1a1a")
        bar = fig.colorbar(mesh, ax=axis, pad=0.03, aspect=18)
        bar.ax.tick_params(length=2, width=0.6)
        bar.outline.set_visible(False)
        axis.set_title(f"({chr(ord('a') + index)})", loc="left", fontweight="bold")
        axis.set_title(PANELS[metric][0], loc="center")
        axis.set_xticks(centres)
        axis.set_xticklabels([f"{LOCALITY_BINS[i]:.1f}" for i in range(len(centres))])
        axis.set_yticks(k_values)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)

    for axis in axes[len(metrics):]:
        axis.set_visible(False)

    fig.suptitle("AQORN improvement over naive packing (%), cells labelled by circuit count")
    fig.supxlabel("measured locality bin")
    fig.supylabel("logical qubits per patch, $k$")
    paths = save_figure(fig, f"{NAME}{suffix}", Path(__file__), settings.output_override)
    plt.close(fig)
    return paths


def plot(improvement: pd.DataFrame, settings: Settings) -> List[Path]:
    use_style()
    draw = heatmap_layout if settings.heatmap else plot_layout
    paths: List[Path] = []
    for suffix, (nrows, ncols, metrics) in LAYOUTS.items():
        paths.extend(draw(improvement, settings, suffix, nrows, ncols, metrics))
    return paths


def parse_args() -> Settings:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--benchmark-root", type=Path, default=defaults.benchmark_root)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(defaults.k_values))
    parser.add_argument("--min-qubits", type=int, default=defaults.min_qubits)
    parser.add_argument("--max-qubits", type=int, default=defaults.max_qubits,
                        help="Applied to the loaded proxy, not the README.")
    parser.add_argument("--max-gates", type=int, default=defaults.max_gates)
    parser.add_argument("--min-two-qubit-gates", type=int, default=defaults.min_two_qubit_gates)
    parser.add_argument("--locality-capacity", type=int, default=defaults.locality_capacity,
                        help="Group size the locality metric partitions into. Fixed rather "
                             "than tracked to k so one circuit has one locality.")
    parser.add_argument("--heatmap", action="store_true",
                        help="Render the coarse binned grid instead of the scatter. Check "
                             "the printed per-bin counts first; the corpus is thin.")
    add_sim_arguments(parser, defaults.sim)
    parser.add_argument("--csv", type=Path, default=defaults.csv_path)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    return Settings(
        benchmark_root=args.benchmark_root,
        k_values=args.k_values,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        max_gates=args.max_gates,
        min_two_qubit_gates=args.min_two_qubit_gates,
        locality_capacity=args.locality_capacity,
        sim=settings_from_args(args, defaults.sim),
        heatmap=args.heatmap,
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
        frame = compute(settings)
        if frame.empty:
            raise RuntimeError("no data produced -- every simulation failed")
        settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.csv_path, index=False)
        print(f"[{NAME}] wrote {len(frame)} rows to {settings.csv_path}")
        if settings.persist:
            from experiments.shared.figures import data_dir

            target = data_dir(Path(__file__)) / f"{NAME}.csv"
            frame.to_csv(target, index=False)
            print(f"[{NAME}] persisted to {target}")

    frame = drop_miscompiled(frame)
    improvement = paired_improvement(frame)
    print_summary(frame, improvement)
    for path in plot(improvement, settings):
        print(f"[{NAME}] wrote {path}")


if __name__ == "__main__":
    main()
