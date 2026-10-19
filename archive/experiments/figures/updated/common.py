"""
common.py — shared harness for the rebuilt arXiv data figures.

Every script in this folder recreates a data figure from AQORN_arXiv.pdf using the
current compiler rather than the one the paper was written against. Three things
differ from the original scripts, and they are the reason the numbers move:

1. **Merging runs at every code size.** It used to be [[4,2,2]]-only.
2. **Placement is the `grouping` stack**, not `packer/merge_packer.py`: a signed
   interaction graph, multilevel partitioning, then exact Hungarian slot alignment.
   The old packer assigns qubits in pairs and cannot express a placement at k > 2.
3. **Placements are scored analytically**, so a candidate costs microseconds instead
   of a full translation. That is what makes the large benchmarks runnable at all --
   the translate-per-candidate search needed ~5.5 hours for one 140-qubit point.

Figures that need simulation (LSR, TVD) are far slower than the gate-count ones and
say so in their own docstrings.

Style follows Figure 12 of the draft: Times New Roman, stacked metric rows, the same
three technique colours.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from iceberg.costs import DEFAULT_WORK_WEIGHTS
from archive.grouping.analytic import AnalyticCost, translated_view
from archive.grouping.graph import NoiseProfile, build_graph
from archive.grouping.profiles import extract_profiles
from iceberg.utils import logical_capacity
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

#: Depth weight used when scoring placements. Set from the grid search in
#: grouping/tune.py, not by hand.
DEFAULT_ANALYTIC_DEPTH_WEIGHT = 12.0

Perm = List[Tuple[int, int]]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("experiments/figures/updated")
QASM_ROOT = Path("benchmarks/qasmbench")

# --- style, matching Figure 12 of the draft ---

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 15,
}

COMPONENTS = [
    ("h_commutation", "H commutation", "#3b4c56"),
    ("merging", "+ Merging", "#cdc2c0"),
    ("packing", "+ Packing", "#0097fd"),
]

SIZE_GROUPS = ["small", "medium", "large"]
SIZE_LABELS = {"small": "S", "medium": "M", "large": "L"}

#: key -> (csv column, axis label, needs simulation)
METRICS: Dict[str, Tuple[str, str, bool]] = {
    "1q": ("pct_reduction_one_qubit_gates", "#1q gates", False),
    "2q": ("pct_reduction_two_qubit_gates", "#2q gates", False),
    "depth": ("pct_reduction_circuit_depth", "depth", False),
    "lsr": ("pct_improvement_logical_selection_rate", "LSR", True),
    "tvd": ("pct_reduction_logical_total_variation_distance", "TVD", True),
}

GATE_METRICS = ["1q", "2q", "depth"]


def use_paper_style() -> None:
    plt.rcParams.update(PAPER_STYLE)


def save(fig, name: str, *, output_dir: Path = OUTPUT_DIR) -> Path:
    """Write both an SVG for editing and a PDF for the paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    svg = output_dir / f"{name}.svg"
    fig.savefig(svg, dpi=200)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"[updated] saved {svg} and {svg.with_suffix('.pdf')}")
    return svg


# --- circuits ---


def load_group(
    group: str,
    *,
    qasm_root: Path = QASM_ROOT,
    max_gates: Optional[int] = 2500,
    max_qubits: Optional[int] = 150,
    timeout: Optional[float] = 1.0,
) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """QASMBench proxy circuits for one size group, filtered as the paper does."""
    from experiments.shared.benchmark_circuits import load_proxy_circuits

    circuits = load_proxy_circuits(
        qasm_root / group,
        gate_limit=2000,
        qubit_limit=max_qubits,
        timeout_seconds=timeout,
        replace_rz_rx_with_s=False,
    )
    if max_gates is not None:
        circuits = [(n, c) for n, c in circuits if c.size() <= max_gates]
    return [(n, c) for n, c in circuits if not n.lower().endswith("_n2")]


def random_circuit(
    num_qubits: int,
    gates_per_qubit: int,
    seed: int,
    *,
    immutable_ratio: float = 0.15,
    clifford_only: bool = False,
) -> qiskit.QuantumCircuit:
    """
    A random circuit over the translator's gate set.

    `immutable_ratio` is the share of S/Rx/Rz -- gates merging cannot touch. The
    paper's point about random circuits is that they carry fewer of these than the
    benchmarks do, which is why merging and packing find more to work with.

    `clifford_only` restricts the immutable gates to S. Set it for anything that
    will be simulated: the Stim backend rejects Rx and Rz outright, which is the
    same reason the paper simulates rotation-bearing benchmarks through Clifford
    proxy circuits rather than directly.
    """
    import random

    rng = random.Random(seed)
    circuit = qiskit.QuantumCircuit(num_qubits)
    for _ in range(num_qubits * gates_per_qubit):
        roll = rng.random()
        if roll < 0.4:
            left, right = rng.sample(range(num_qubits), 2)
            getattr(circuit, rng.choice(("cx", "cz")))(left, right)
        elif roll < 0.4 + immutable_ratio:
            _append_immutable(circuit, rng, clifford_only)
        else:
            getattr(circuit, rng.choice(("h", "x", "z")))(rng.randrange(num_qubits))
    return circuit


def _append_immutable(circuit: qiskit.QuantumCircuit, rng, clifford_only: bool) -> None:
    """An S or a rotation: gates merging cannot fuse, so they bound its reach."""
    qubit = rng.randrange(circuit.num_qubits)
    choice = "s" if clifford_only else rng.choice(("s", "rz", "rx"))
    if choice == "s":
        circuit.s(qubit)
    else:
        getattr(circuit, choice)(rng.uniform(0.1, 3.0), qubit)


# --- configurations and placement ---


def config_for(
    n_code: int, *, h_commutation: bool, merging: bool, fault_tolerant: bool = False
) -> PipelineConfig:
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    config.translator_config.fault_tolerant = fault_tolerant
    return config


def sequential_packing(num_qubits: int, k: int) -> Perm:
    """The naive baseline: fill patches in program order."""
    return [(index // k, index % k) for index in range(num_qubits)]


def tightest_patches(num_qubits: int, k: int) -> int:
    return max(1, math.ceil(num_qubits / k))


def optimized_placement(
    circuit: qiskit.QuantumCircuit,
    n_code: int,
    *,
    num_patches: Optional[int] = None,
    noise: Optional[NoiseProfile] = None,
    depth_weight: float = 0.0,
) -> Perm:
    """
    Place with the current stack: signed graph, multilevel partition, exact alignment.

    Scores the circuit the translator will actually see -- `translated_view` applies
    the H-commutation passes first -- because placing against the raw program was
    worth a 13-49% cost error.
    """
    from archive.grouping.plan import best_placement

    k = logical_capacity(n_code)
    target = num_patches or tightest_patches(circuit.num_qubits, k)
    profiles = extract_profiles(translated_view(circuit))
    graph = build_graph(
        profiles, n_code, noise=noise, num_qubits=circuit.num_qubits
    )
    model = AnalyticCost(
        profiles=profiles,
        n_code=n_code,
        movement_cost=noise.movement_cost if noise else 0.0,
        depth_weight=depth_weight if depth_weight else DEFAULT_ANALYTIC_DEPTH_WEIGHT,
    )
    # Verify the chosen placement against the naive one by exact measurement, so a
    # 2-3% model error can never turn the packing stage into a regression. Two
    # translations; everything else here costs more.
    full = config_for(n_code, h_commutation=True, merging=True)
    w1, w2 = DEFAULT_WORK_WEIGHTS

    def exact(candidate: Perm):
        metrics = measure(circuit, candidate, full)
        if metrics is None:
            return None
        return (
            w1 * metrics.one_qubit
            + w2 * metrics.two_qubit
            + DEFAULT_ANALYTIC_DEPTH_WEIGHT * metrics.depth
        )

    from archive.grouping.plan import _rounds_for

    placement, _ = best_placement(
        graph, profiles, n_code, model, max_patches=target,
        refine_rounds=_rounds_for(circuit.num_qubits),
        verify=exact, fallback=sequential_packing(circuit.num_qubits, k),
    )
    return placement


def analytic_model(circuit: qiskit.QuantumCircuit, n_code: int, **kwargs) -> AnalyticCost:
    return AnalyticCost.for_program(circuit, n_code, **kwargs)


# --- measurement ---


@dataclass(frozen=True)
class GateMetrics:
    one_qubit: int
    two_qubit: int
    depth: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.one_qubit, self.two_qubit, self.depth)


def measure(
    circuit: qiskit.QuantumCircuit, permutation: Perm, config: PipelineConfig
) -> Optional[GateMetrics]:
    """Translate and count, or None if the placement is rejected under the FT rules."""
    try:
        translated = translate_permutation(circuit, permutation, config=config)
    except Exception:  # noqa: BLE001
        return None
    one, two = one_two_qubit_gate_totals(translated.circuit)
    return GateMetrics(one, two, translated.circuit.depth())


def pct_reduction(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (reference - challenger) / reference


def pct_improvement(reference: float, challenger: float) -> float:
    if reference <= 0:
        return 0.0
    return 100.0 * (challenger - reference) / reference


@dataclass
class Ablation:
    """
    The three-technique decomposition, as incremental percent reductions.

    Both the incremental segments and the cumulative totals they came from are
    recorded. Plots need the cumulative form -- see `stack_heights` for why the
    obvious alternative produces bars above 100%.
    """

    h_commutation: Tuple[float, float, float]
    merging: Tuple[float, float, float]
    packing: Tuple[float, float, float]
    cumulative: Optional[Dict[str, Tuple[float, float, float]]] = None
    baseline: Optional[GateMetrics] = None
    full: Optional[GateMetrics] = None
    patches: int = 0
    merging_fired: bool = False

    def rows(self, **shared) -> List[Dict]:
        out = []
        for component, values in (
            ("h_commutation", self.h_commutation),
            ("merging", self.merging),
            ("packing", self.packing),
        ):
            row = dict(shared, component=component)
            for key, value in zip(GATE_METRICS, values):
                row[METRICS[key][0]] = value
            if self.cumulative:
                for key, value in zip(GATE_METRICS, self.cumulative[component]):
                    row[f"cumulative_{METRICS[key][0]}"] = value
            row["patches"] = self.patches
            row["merging_fired"] = self.merging_fired
            out.append(row)
        return out


def stack_heights(frame, column: str) -> List[float]:
    """
    Segment heights for a stacked bar, in COMPONENTS order, guaranteed sane.

    Clipping each *segment* at zero -- which the draft's scripts do, and which this
    folder inherited -- breaks when one technique hurts and a later one more than
    recovers. On `ising_n98` H commutation costs 88% of depth and merging returns
    179%; the true total is 90%, but dropping the negative segment leaves a 179% bar
    on a percent-reduction axis.

    So clip the *cumulative* series instead and difference that. Each segment stays
    non-negative, the stack height is the real full-pipeline reduction, and a
    technique that hurts simply contributes nothing rather than inflating the one
    after it.

    Falls back to per-segment means when the cumulative columns are absent, so
    older CSVs still plot.
    """
    cumulative_column = f"cumulative_{column}"
    order = [component for component, _, _ in COMPONENTS]
    if cumulative_column not in getattr(frame, "columns", ()):
        return [
            max(0.0, float(frame[frame.component == c][column].mean() or 0.0))
            for c in order
        ]

    running = 0.0
    heights: List[float] = []
    for component in order:
        values = frame[frame.component == component][cumulative_column]
        mean = float(values.mean()) if len(values) else running
        mean = 0.0 if mean != mean else mean  # NaN guard
        clipped = max(running, mean)
        heights.append(clipped - running)
        running = clipped
    return heights


def ablate(
    circuit: qiskit.QuantumCircuit,
    n_code: int,
    *,
    num_patches: Optional[int] = None,
    noise: Optional[NoiseProfile] = None,
) -> Optional[Ablation]:
    """
    Measure the incremental benefit of H commutation, then merging, then packing.

    Each stage is compared against the same-k naive baseline, so the stack height is
    the full-pipeline reduction. The packing stage uses the `grouping` placement; the
    other two use the naive permutation, which is what makes the segments additive.
    """
    k = logical_capacity(n_code)
    trivial = sequential_packing(circuit.num_qubits, k)

    base = measure(circuit, trivial, config_for(n_code, h_commutation=False, merging=False))
    h_only = measure(circuit, trivial, config_for(n_code, h_commutation=True, merging=False))
    h_merge = measure(circuit, trivial, config_for(n_code, h_commutation=True, merging=True))
    if base is None or h_only is None or h_merge is None:
        return None

    placed_perm = optimized_placement(
        circuit, n_code, num_patches=num_patches, noise=noise
    )
    full_config = config_for(n_code, h_commutation=True, merging=True)
    packed = measure(circuit, placed_perm, full_config) or h_merge

    def cumulative(metrics: GateMetrics) -> Tuple[float, float, float]:
        return tuple(
            pct_reduction(b, c) for b, c in zip(base.as_tuple(), metrics.as_tuple())
        )

    h_cum, m_cum, p_cum = cumulative(h_only), cumulative(h_merge), cumulative(packed)
    return Ablation(
        h_commutation=h_cum,
        merging=tuple(m - h for m, h in zip(m_cum, h_cum)),
        packing=tuple(p - m for p, m in zip(p_cum, m_cum)),
        cumulative={
            "h_commutation": h_cum,
            "merging": m_cum,
            "packing": p_cum,
        },
        baseline=base,
        full=packed,
        patches=len({p for p, _ in placed_perm}),
        merging_fired=h_merge.as_tuple() != h_only.as_tuple(),
    )


# --- plotting ---


def stacked_metric_rows(
    axes,
    metrics: Sequence[str],
    draw,
    *,
    ylabel: str = "Percent reduction",
) -> None:
    """
    Apply the shared row styling and call `draw(ax, metric_key)` for each row.

    Keeps every figure in this folder visually identical to Figure 12: dotted y grid,
    no top or right spine, metric name on the y-axis.
    """
    for ax, key in zip(axes, metrics):
        draw(ax, key)
        ax.set_ylabel(METRICS[key][1])
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#b0b0b0")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    del ylabel


def component_legend(ax, *, loc: str = "upper right") -> None:
    import matplotlib.patches as mpatches

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=color, edgecolor="white", label=label)
            for _, label, color in COMPONENTS
        ],
        loc=loc,
        frameon=False,
    )


def quiet() -> None:
    """Silence the translator's expected warnings during a long sweep."""
    import logging
    import warnings

    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
