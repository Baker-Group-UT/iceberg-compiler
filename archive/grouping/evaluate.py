"""
evaluate.py — do these groupings actually produce cheaper circuits?

Nothing above is trustworthy on its own: the cost model is a prediction and the
similarity kernel is a heuristic. This module closes the loop by translating each
candidate placement and measuring the real physical circuit, so every claim in the
report is backed by emitted gate counts rather than by the model that proposed them.

Strategies compared
    exhaustive    the true optimum, by trying every placement. Only feasible for
                  small qubit counts, and included precisely so the heuristics can be
                  scored against the actual best rather than against each other.
    cost          agglomerative clustering with the physical cost-model linkage
    similarity    agglomerative clustering with the frame/cast similarity linkage
    packer        the existing pairwise merge packer (n=4 only)
    sequential    qubit i to patch i // k -- the naive baseline
    random        best of N random placements, to show the spread the others beat

Run it:

    python -m grouping.evaluate                 # default sweep
    python -m grouping.evaluate --n 6 --qubits 8
    python -m grouping.evaluate --exhaustive-limit 6 --circuits 20
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import QuantumCircuit

from iceberg.costs import DEFAULT_WORK_WEIGHTS
from iceberg.utils import logical_capacity
from translator.config import TranslatorConfig
from translator.core import Translator

from .agglomerative import cluster_qubits
from .profiles import extract_profiles

__all__ = ["measure", "compare_strategies", "StrategyResult"]


@dataclass
class StrategyResult:
    name: str
    permutation: Optional[List[Tuple[int, int]]]
    one_qubit: int
    two_qubit: int
    weighted: float
    patches: int
    placeable: bool
    note: str = ""


def measure(
    program: QuantumCircuit,
    permutation: Sequence[Tuple[int, int]],
    n_code: int,
    *,
    fault_tolerant: bool = False,
) -> Optional[Tuple[int, int, float]]:
    """
    Translate and count. Returns (1q, 2q, weighted) or None if unplaceable.

    Unplaceable is a real outcome, not an error: the translator rejects placements
    that violate its fault-tolerance rules, and a grouping strategy that proposes
    many of those is worse for it.
    """
    translator = Translator(
        config=TranslatorConfig(n_code=n_code, fault_tolerant=fault_tolerant)
    )
    try:
        result = translator.translate(program, list(permutation))
    except Exception:  # noqa: BLE001 - placement rejected under the FT rules
        return None

    one = two = 0
    for instruction in result.circuit.data:
        if instruction.operation.name in ("barrier", "measure"):
            continue
        if len(instruction.qubits) == 1:
            one += 1
        else:
            two += 1
    w1, w2 = DEFAULT_WORK_WEIGHTS
    return one, two, w1 * one + w2 * two


def _result(
    name: str,
    program: QuantumCircuit,
    permutation: Optional[Sequence[Tuple[int, int]]],
    n_code: int,
    *,
    note: str = "",
    fault_tolerant: bool = False,
) -> StrategyResult:
    if permutation is None:
        return StrategyResult(name, None, 0, 0, float("inf"), 0, False, note or "no placement")
    counts = measure(program, permutation, n_code, fault_tolerant=fault_tolerant)
    patches = len({patch for patch, _ in permutation})
    if counts is None:
        return StrategyResult(
            name, list(permutation), 0, 0, float("inf"), patches, False, note or "rejected"
        )
    one, two, weighted = counts
    return StrategyResult(name, list(permutation), one, two, weighted, patches, True, note)


# --- strategies ---


def sequential_permutation(num_qubits: int, k: int) -> List[Tuple[int, int]]:
    return [(index // k, index % k) for index in range(num_qubits)]


def exhaustive_best(
    program: QuantumCircuit,
    n_code: int,
    *,
    fault_tolerant: bool = False,
    limit: int = 5040,
) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """
    The cheapest placement, by measuring every one. Returns (best, tried).

    Enumerates assignments of qubits to *slots*, not just groupings. Slot order
    within a patch matters — the transversal macros pair lane i with lane i — so
    enumerating groupings alone gives a bound that random slot shuffling can beat,
    which is exactly what an earlier version of this function did.
    """
    k = logical_capacity(n_code)
    num_patches = -(-program.num_qubits // k)
    slots = [(patch, slot) for patch in range(num_patches) for slot in range(k)]
    slots = slots[: program.num_qubits]

    best: Optional[List[Tuple[int, int]]] = None
    best_cost = float("inf")
    tried = 0
    for assignment in itertools.permutations(slots):
        if tried >= limit:
            break
        counts = measure(program, list(assignment), n_code, fault_tolerant=fault_tolerant)
        tried += 1
        if counts is None:
            continue
        if counts[2] < best_cost:
            best_cost = counts[2]
            best = list(assignment)
    return best, tried


def packer_permutation(program: QuantumCircuit, n_code: int) -> Optional[List[Tuple[int, int]]]:
    """
    The existing pairwise packer, which only supports two slots per patch.

    It requires the patch count up front, so it is given the tightest packing that
    fits — the same budget the clustering strategies are compared at.
    """
    if n_code != 4:
        return None
    from archive.packer.merge_packer import MergeMaximizingPacker

    num_codes = -(-program.num_qubits // 2)
    try:
        packer = MergeMaximizingPacker(patch_capacity=2, rng=random.Random(0))
        return list(packer.pack(program, num_codes).permutation)
    except Exception:  # noqa: BLE001
        return None


def random_best(
    program: QuantumCircuit,
    n_code: int,
    *,
    trials: int,
    rng: random.Random,
    fault_tolerant: bool = False,
) -> Tuple[Optional[List[Tuple[int, int]]], List[float]]:
    """Best of `trials` random placements, plus the spread of all of them."""
    k = logical_capacity(n_code)
    num_patches = -(-program.num_qubits // k)
    slots = [(patch, slot) for patch in range(num_patches) for slot in range(k)]
    best: Optional[List[Tuple[int, int]]] = None
    best_cost = float("inf")
    costs: List[float] = []
    for _ in range(trials):
        picked = slots[: program.num_qubits]
        candidate = list(picked)
        rng.shuffle(candidate)
        counts = measure(program, candidate, n_code, fault_tolerant=fault_tolerant)
        if counts is None:
            continue
        costs.append(counts[2])
        if counts[2] < best_cost:
            best_cost = counts[2]
            best = candidate
    return best, costs


def compare_strategies(
    program: QuantumCircuit,
    n_code: int,
    *,
    rng: random.Random,
    exhaustive_limit: int = 6,
    random_trials: int = 40,
    fault_tolerant: bool = False,
) -> List[StrategyResult]:
    k = logical_capacity(n_code)
    profiles = extract_profiles(program)
    results: List[StrategyResult] = []

    for linkage in ("cost", "similarity", "interaction"):
        grouping = cluster_qubits(
            program, n_code=n_code, linkage=linkage, profiles=profiles
        )
        results.append(
            _result(
                linkage,
                program,
                grouping.permutation(),
                n_code,
                note=f"{grouping.num_patches} patches",
                fault_tolerant=fault_tolerant,
            )
        )

    results.append(
        _result(
            "sequential",
            program,
            sequential_permutation(program.num_qubits, k),
            n_code,
            fault_tolerant=fault_tolerant,
        )
    )

    packer = packer_permutation(program, n_code)
    results.append(
        _result(
            "packer",
            program,
            packer,
            n_code,
            note="" if packer else "n=4 only",
            fault_tolerant=fault_tolerant,
        )
    )

    best_random, spread = random_best(
        program, n_code, trials=random_trials, rng=rng, fault_tolerant=fault_tolerant
    )
    note = ""
    if spread:
        note = f"median {statistics.median(spread):.0f} of {len(spread)}"
    results.append(
        _result("random-best", program, best_random, n_code, note=note, fault_tolerant=fault_tolerant)
    )

    # Local search seeded from each grouping, on the measured objective. This is
    # what tells us whether clustering is a useful *seed* even where it is not a
    # useful final answer.
    from .refine import measured_objective, model_objective, refine_placement

    objective = measured_objective(program, n_code, fault_tolerant=fault_tolerant)
    for seed_name in ("interaction", "sequential"):
        seed = next((r for r in results if r.name == seed_name and r.permutation), None)
        if seed is None:
            continue
        refined = refine_placement(
            seed.permutation, objective, n_code=n_code, max_evaluations=1500
        )
        results.append(
            _result(
                f"refined<{seed_name}",
                program,
                refined.placement,
                n_code,
                note=f"{refined.moves} moves, {refined.evaluations} evals",
                fault_tolerant=fault_tolerant,
            )
        )

    # Same local search, but driven by the cost model instead of by translation.
    # If this lands near the measured optimum, the model is a usable surrogate and
    # the search scales; if not, the model is the bottleneck.
    surrogate = model_objective(profiles, n_code)
    seed = next((r for r in results if r.name == "sequential" and r.permutation), None)
    if seed is not None:
        predicted = refine_placement(
            seed.permutation, surrogate, n_code=n_code, max_evaluations=4000
        )
        results.append(
            _result(
                "refined<model",
                program,
                predicted.placement,
                n_code,
                note=f"{predicted.moves} moves (model-scored)",
                fault_tolerant=fault_tolerant,
            )
        )

    if program.num_qubits <= exhaustive_limit:
        best, tried = exhaustive_best(program, n_code, fault_tolerant=fault_tolerant)
        results.append(
            _result(
                "exhaustive",
                program,
                best,
                n_code,
                note=f"{tried} placements",
                fault_tolerant=fault_tolerant,
            )
        )
    return results


# --- CLI ---


def _structured_programs(
    num_qubits: int,
    count: int,
    rng: random.Random,
    *,
    k: int,
    locality: float = 0.85,
) -> List[QuantumCircuit]:
    """
    Random programs with community structure in their interaction graph.

    The uniformly random programs above have no structure for a clustering method to
    find: every pair of qubits interacts about equally often, so a min-cut has
    essentially no signal and any grouping is as good as any other. Real circuits are
    not like that -- adders, QFT blocks and ansatz layers interact locally -- so this
    generator plants blocks of `k` qubits that interact mostly among themselves, with
    a `1 - locality` chance of reaching outside. If clustering is worth anything, this
    is where it shows.
    """
    from tests.random_programs import CLIFFORD_1Q

    # Relabel the qubits so the planted blocks are *not* the sequential grouping.
    # Without this, `sequential` already is the right answer and a clustering method
    # that recovers the structure perfectly still shows no improvement -- which is
    # exactly what the first version of this experiment measured.
    labels = list(range(num_qubits))
    rng.shuffle(labels)
    blocks = [
        [labels[index] for index in range(start, min(start + k, num_qubits))]
        for start in range(0, num_qubits, k)
    ]
    programs = []
    for _ in range(count):
        circuit = QuantumCircuit(num_qubits)
        for _ in range(6 * num_qubits):
            if rng.random() < 0.45:
                block = rng.choice(blocks)
                if len(block) >= 2 and rng.random() < locality:
                    left, right = rng.sample(block, 2)
                else:
                    left, right = rng.sample(range(num_qubits), 2)
                getattr(circuit, rng.choice(["cx", "cz"]))(left, right)
            else:
                getattr(circuit, rng.choice(CLIFFORD_1Q))(rng.randrange(num_qubits))
        programs.append(circuit)
    return programs


def _default_programs(
    num_qubits: int, count: int, rng: random.Random, *, include_swap: bool = False
) -> List[QuantumCircuit]:
    """
    Random Clifford programs, by default without SWAP gates.

    SWAPs are excluded because the translator rejects any *cross-patch* SWAP
    outright, and with more than a couple of patches almost every placement then has
    one. Measured: 20/20 placeable without SWAPs at 4, 6 and 8 qubits; 5/20, 1/20 and
    0/20 with them. Leaving them in would make the comparison a study of which
    strategy happens to avoid cross-patch SWAPs rather than of grouping quality.
    See the report for why lifting that restriction now looks cheap.
    """
    from tests.random_programs import CLIFFORD_1Q

    programs = []
    two_qubit = ["cx", "cz", "swap"] if include_swap else ["cx", "cz"]
    for _ in range(count):
        circuit = QuantumCircuit(num_qubits)
        for _ in range(6 * num_qubits):
            if rng.random() < 0.45:
                left, right = rng.sample(range(num_qubits), 2)
                getattr(circuit, rng.choice(two_qubit))(left, right)
            else:
                getattr(circuit, rng.choice(CLIFFORD_1Q))(rng.randrange(num_qubits))
        programs.append(circuit)
    return programs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--circuits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20250727)
    parser.add_argument("--exhaustive-limit", type=int, default=6)
    parser.add_argument("--random-trials", type=int, default=40)
    parser.add_argument("--fault-tolerant", action="store_true")
    parser.add_argument("--include-swap", action="store_true",
                        help="allow SWAP gates; most placements then get rejected")
    parser.add_argument("--structured", action="store_true",
                        help="use circuits with community structure in their "
                             "interaction graph, where clustering can matter")
    parser.add_argument("--locality", type=float, default=0.85)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    totals: Dict[Tuple[int, int, str], List[float]] = {}
    wins: Dict[Tuple[int, int, str], int] = {}

    for n_code in args.n:
        for num_qubits in args.qubits:
            k = logical_capacity(n_code)
            if num_qubits < k:
                continue
            if args.structured:
                programs = _structured_programs(
                    num_qubits, args.circuits, rng, k=k, locality=args.locality
                )
            else:
                programs = _default_programs(
                    num_qubits, args.circuits, rng, include_swap=args.include_swap
                )
            print()
            print("=" * 78)
            kind = "structured" if args.structured else "uniform-random"
            print(f" [[{n_code},{k},2]]   {num_qubits} logical qubits   "
                  f"{len(programs)} {kind} Clifford circuits")
            print("=" * 78)

            per_strategy: Dict[str, List[float]] = {}
            for index, program in enumerate(programs):
                results = compare_strategies(
                    program,
                    n_code,
                    rng=rng,
                    exhaustive_limit=args.exhaustive_limit,
                    random_trials=args.random_trials,
                    fault_tolerant=args.fault_tolerant,
                )
                placeable = [r for r in results if r.placeable]
                best = min((r.weighted for r in placeable), default=float("inf"))
                for result in results:
                    per_strategy.setdefault(result.name, []).append(result.weighted)
                    if result.placeable and result.weighted == best:
                        wins[(n_code, num_qubits, result.name)] = (
                            wins.get((n_code, num_qubits, result.name), 0) + 1
                        )
                if index == 0:
                    for result in results:
                        marker = "*" if result.placeable and result.weighted == best else " "
                        cost = (
                            f"{result.weighted:8.0f}" if result.placeable else "       -"
                        )
                        print(
                            f"  {marker} {result.name:12s} weighted={cost} "
                            f"1q={result.one_qubit:4d} 2q={result.two_qubit:4d} "
                            f"patches={result.patches:2d} {result.note}"
                        )

            print(f"  -- across {len(programs)} circuits --")
            baseline = per_strategy.get("sequential", [])
            for name, costs in sorted(per_strategy.items()):
                finite = [c for c in costs if c != float("inf")]
                if not finite:
                    print(f"     {name:12s} never placeable")
                    continue
                mean = statistics.mean(finite)
                relative = ""
                if baseline:
                    paired = [
                        (c / b)
                        for c, b in zip(costs, baseline)
                        if c != float("inf") and b not in (0, float("inf"))
                    ]
                    if paired:
                        relative = f"  {statistics.mean(paired):.3f} x sequential"
                print(
                    f"     {name:12s} mean={mean:8.1f}  placed {len(finite)}/{len(costs)}"
                    f"  wins={wins.get((n_code, num_qubits, name), 0)}{relative}"
                )
                totals[(n_code, num_qubits, name)] = finite
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
