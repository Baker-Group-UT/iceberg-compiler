"""
tune.py — set the graph's two free parameters from measurement.

`graph.build_graph` has two numbers that are not derivable from the cost tables:

`depth_weight`           what one layer of depth is worth in weighted-gate units.
                         Sets the exchange rate between the two axes a placement
                         trades off. Too low and CNOT placement is invisible; too
                         high and it overrides real gate savings on CZ and XCX.
`transversal_confidence` how much of a speculative transversal macro's per-lane gain
                         to bank when only two of its k lanes are known to align.

Guessing them is how a cost model ends up pointing the wrong way, which has already
happened once in this codebase (the pairwise packer's estimate, r = -0.17 against
measured cost). So this module grid-searches them against *translated* gate counts
and depth on a benchmark sample, and reports the frontier rather than a single
winner -- the right point depends on whether the target machine is gate- or
depth-limited, and that is the caller's choice to make.

    python -m grouping.tune                       # default grid, small benchmarks
    python -m grouping.tune --groups small medium
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from iceberg.utils import logical_capacity
from translator.config import TranslatorConfig
from translator.core import Translator

from .analytic import translated_view
from .graph import build_graph
from .multilevel import align_slots, partition
from .profiles import extract_profiles

__all__ = ["TuningResult", "score_parameters", "grid_search"]

DEFAULT_DEPTH_WEIGHTS = (0.0, 1.0, 3.0, 6.0, 12.0)
DEFAULT_CONFIDENCES = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class TuningResult:
    depth_weight: float
    transversal_confidence: float
    #: Mean percent reduction against the sequential baseline, per metric.
    one_qubit: float
    two_qubit: float
    depth: float
    circuits: int

    @property
    def combined(self) -> float:
        """
        A single number to rank by: weighted gates plus depth, both as reductions.

        Gates and depth are averaged with equal weight rather than by the 1:13
        gate ratio, because the figures report them as three equal rows and the
        paper's headline quotes all three.
        """
        return (self.one_qubit + self.two_qubit + self.depth) / 3.0

    def summary(self) -> str:
        return (
            f"depth_weight={self.depth_weight:<5.1f} conf={self.transversal_confidence:<4.2f} "
            f"1q={self.one_qubit:6.2f}% 2q={self.two_qubit:6.2f}% "
            f"depth={self.depth:6.2f}%  mean={self.combined:6.2f}%"
        )


def _measure(
    circuit: QuantumCircuit, placement, n_code: int
) -> Optional[Tuple[int, int, int]]:
    try:
        result = Translator(config=TranslatorConfig(n_code=n_code)).translate(
            circuit, list(placement)
        )
    except Exception:  # noqa: BLE001 - rejected under the placement rules
        return None
    one, two = one_two_qubit_gate_totals(result.circuit)
    return one, two, result.circuit.depth()


def score_parameters(
    circuits: Sequence[Tuple[str, QuantumCircuit]],
    n_code: int,
    *,
    depth_weight: float,
    transversal_confidence: float,
) -> Optional[TuningResult]:
    """
    Mean percent reduction these parameters achieve against sequential placement.

    Measured by translating, not modelled -- the point of tuning is to check the
    model's parameters against reality, so using the model to score them would be
    circular.
    """
    k = logical_capacity(n_code)
    totals = [0.0, 0.0, 0.0]
    counted = 0

    for _, circuit in circuits:
        sequential = [(i // k, i % k) for i in range(circuit.num_qubits)]
        baseline = _measure(circuit, sequential, n_code)
        if baseline is None:
            continue

        profiles = extract_profiles(translated_view(circuit))
        graph = build_graph(
            profiles,
            n_code,
            depth_weight=depth_weight,
            transversal_confidence=transversal_confidence,
            num_qubits=circuit.num_qubits,
        )
        result = partition(
            graph, n_code, max_patches=max(1, -(-circuit.num_qubits // k))
        )
        placed = _measure(
            circuit, align_slots(result.groups, profiles, n_code), n_code
        )
        if placed is None:
            continue

        for index in range(3):
            reference = baseline[index]
            if reference > 0:
                totals[index] += 100.0 * (reference - placed[index]) / reference
        counted += 1

    if not counted:
        return None
    return TuningResult(
        depth_weight=depth_weight,
        transversal_confidence=transversal_confidence,
        one_qubit=totals[0] / counted,
        two_qubit=totals[1] / counted,
        depth=totals[2] / counted,
        circuits=counted,
    )


def grid_search(
    circuits: Sequence[Tuple[str, QuantumCircuit]],
    n_code: int,
    *,
    depth_weights: Sequence[float] = DEFAULT_DEPTH_WEIGHTS,
    confidences: Sequence[float] = DEFAULT_CONFIDENCES,
    verbose: bool = True,
) -> List[TuningResult]:
    results: List[TuningResult] = []
    for depth_weight, confidence in itertools.product(depth_weights, confidences):
        result = score_parameters(
            circuits,
            n_code,
            depth_weight=depth_weight,
            transversal_confidence=confidence,
        )
        if result is None:
            continue
        results.append(result)
        if verbose:
            print(f"  {result.summary()}")
    return sorted(results, key=lambda r: -r.combined)


def main() -> None:
    import logging
    import warnings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-code", type=int, default=4)
    parser.add_argument("--groups", nargs="+", default=["small"])
    parser.add_argument("--qasm-root", type=Path, default=Path("benchmarks/qasmbench"))
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    from experiments.figures.updated.common import load_group

    circuits: List[Tuple[str, QuantumCircuit]] = []
    for group in args.groups:
        circuits.extend(load_group(group, qasm_root=args.qasm_root))
    print(f"[tune] {len(circuits)} circuits at n={args.n_code}\n")

    results = grid_search(circuits, args.n_code)
    print("\n[tune] ranked by mean reduction across the three metrics")
    for result in results[:8]:
        print(f"  {result.summary()}")
    if results:
        best = results[0]
        print(
            f"\n[tune] best: depth_weight={best.depth_weight}, "
            f"transversal_confidence={best.transversal_confidence}"
        )
        gate_only = next(
            (r for r in results if r.depth_weight == 0.0 and r.transversal_confidence == 0.0),
            None,
        )
        if gate_only:
            print(f"[tune] gate-count-only baseline: {gate_only.summary()}")
            print(
                f"[tune] tuning is worth "
                f"{best.combined - gate_only.combined:+.2f} points of mean reduction"
            )


if __name__ == "__main__":
    main()
