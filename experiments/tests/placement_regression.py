"""
placement_regression.py -- does AQORN beat naive, everywhere we claim it does?

The gate to run *before* committing a machine to an overnight data run. Asserts one
invariant across the whole design space:

    the planner never returns a placement whose expected error exceeds naive's,
    and the mean improvement in every cell is positive

**Shot-free, which is the entire point.** `simulator.attribution.measured_noise_split`
reads expected detectable errors straight off the translated circuit's schedule -- the same
schedule the simulator would sample -- so a point costs one plan plus two translations
instead of ~300k shots. That is seconds rather than minutes, which buys hundreds of points
in the time one simulated cell used to take, and it is why this can sweep dimensions the
figures cannot afford to.

It exists because the figures could not catch what it catches. `improvement_vs_noise_and_k`
found the movement-pricing bug only after 8.3 hours of simulation, and found it as a
*negative result* that looked publishable until the cost model was inspected. Expected error
is the quantity selection rate and TVD are downstream of, so a regression here is a
regression there, detectable in seconds.

What it sweeps, and why each axis is here:

    k                 2..10. The code family. Patch count is `ceil(n/k)`, so k sets how
                      many placement decisions exist at all.
    distance_scalar   0.5..8, the span the cost model is calibrated for. This is the axis
                      that exposed the movement bug; regressions concentrated at its
                      movement-dominated end.
    structure         measured, not looked up: chain-like, star-like or irregular, from the
                      interaction graph's degree profile. The failures were all on chains
                      and stars -- `ising`, `cat`, `ghz` at -98% -- because index order is
                      already near distance-optimal there, so naive is a strong baseline and
                      a careless objective loses to it. Classifying by structure rather than
                      by benchmark name makes that a property of the circuit and lets a new
                      corpus land in the right bucket automatically.
    width             reported as a band. Wide circuits have more patches and more chance to
                      get a placement wrong.
    merging           on and off. Merging changes which crossings exist, so it changes what
                      the placement is optimising over.

Run with:

    PYTHONPATH=. python experiments/tests/placement_regression.py
    PYTHONPATH=. python experiments/tests/placement_regression.py --quick
    PYTHONPATH=. python experiments/tests/placement_regression.py --k-values 2 4 --circuits 8

Exit status is 1 on any regression or any non-positive cell mean, so it can gate a script.
"""

from __future__ import annotations

import argparse
import logging
import statistics as st
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import qiskit

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import (
    SimSettings,
    ensure_measurements,
    naive_packing,
    noise_profile_for,
)
from packing.plan import plan_packing
from packing.version import COST_MODEL_VERSION
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation
from simulator.attribution import measured_noise_split
from simulator.noise_models import StimNoiseParameters

#: The span the cost model is calibrated for, as `distance_scalar`. `with_noise_ratio` takes
#: a nominal ratio, and `distance_scalar = 8 / sqrt(nominal)`, so these invert to nominal
#: 1, 4, 16, 64, 256. Matching `improvement_vs_noise_and_k.DEFAULT_RATIOS`.
DEFAULT_DISTANCE_SCALARS: Tuple[float, ...] = (8.0, 4.0, 2.0, 1.0, 0.5)
DEFAULT_K_VALUES: Tuple[int, ...] = (2, 4, 6, 8, 10)

#: Below this the improvement is zero to measurement precision and the point is counted as
#: a tie rather than either a win or a regression. The quantity is a ratio of expected
#: errors computed in double precision from the same schedule, so this only has to clear
#: floating-point noise, not sampling noise -- there is none.
TIE = 1e-9


def structure_of(circuit: qiskit.QuantumCircuit) -> str:
    """
    Classify the interaction graph: `chain`, `star`, or `irregular`.

    Measured from the degree profile rather than read off the benchmark name, because the
    property that matters is "is index order already a good placement", and that is a fact
    about the graph. A chain gets `q[i]-q[i+1]` for free from program order; a star has one
    hub that must cross to every patch however you place it. Both make naive a strong
    baseline, and both are where a placement objective that counts crossings instead of
    measuring distance loses -- `ising`, `cat` and `ghz` were the -98% cells.
    """
    degree: Dict[int, int] = defaultdict(int)
    edges = set()
    for instruction in circuit.data:
        qubits = instruction.qubits
        if len(qubits) != 2:
            continue
        left, right = sorted(circuit.find_bit(q).index for q in qubits)
        if left != right:
            edges.add((left, right))
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    if not degree:
        return "irregular"
    top = max(degree.values())
    width = circuit.num_qubits
    if top >= max(3, width // 2):
        return "star"
    if top <= 2:
        return "chain"
    return "irregular"


def band_of(width: int) -> str:
    if width <= 20:
        return "n<=20"
    if width <= 40:
        return "n21-40"
    if width <= 80:
        return "n41-80"
    return "n>80"


@dataclass(frozen=True)
class Point:
    """One cell of the sweep. Module scope and picklable, for `spawn`."""

    name: str
    circuit: qiskit.QuantumCircuit
    k: int
    distance_scalar: float
    merging: bool


def _expected_error(
    circuit: qiskit.QuantumCircuit,
    placement,
    *,
    n_code: int,
    h_commutation: bool,
    merging: bool,
    params: StimNoiseParameters,
    sim: SimSettings,
) -> Optional[Tuple[float, int, int]]:
    """Expected detectable errors, depth and 2q count of the translated circuit."""
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    try:
        translated = translate_permutation(
            ensure_measurements(circuit), placement, config=config
        ).circuit
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    try:
        split = measured_noise_split(
            translated, params,
            idle_time_scalar=sim.idle_time_scalar,
            distance_scalar=sim.distance_scalar,
        )
    except Exception:  # noqa: BLE001 - attribution needs a layout
        return None
    two_qubit = sum(1 for inst in translated.data if len(inst.qubits) == 2)
    return split["total"], translated.depth(), two_qubit


def run_point(point: Point) -> Optional[Dict]:
    """Plan, translate both arms, and compare expected error. `None` if unusable."""
    n_code = point.k + 2
    naive = naive_packing(point.circuit.num_qubits, point.k)
    # `with_noise_ratio` is parameterised by the nominal ratio; invert the axis.
    nominal = (8.0 / point.distance_scalar) ** 2
    sim = SimSettings(shot_progress=False).with_noise_ratio(nominal)
    profile = noise_profile_for(sim)
    params = StimNoiseParameters.neutral_atom()

    try:
        plan = plan_packing(
            point.circuit, n_code, h_commutation="auto", verify=True, noise=profile,
        )
    except Exception:  # noqa: BLE001 - a rejected plan is a result, not a crash
        return None

    shared = dict(n_code=n_code, h_commutation=plan.h_commutation,
                  merging=point.merging, params=params, sim=sim)
    left = _expected_error(point.circuit, naive, **shared)
    right = _expected_error(point.circuit, plan.placement, **shared)
    if left is None or right is None or not left[0]:
        return None

    naive_error, naive_depth, naive_two = left
    aqorn_error, aqorn_depth, aqorn_two = right
    return {
        "circuit": point.name,
        "num_qubits": point.circuit.num_qubits,
        "band": band_of(point.circuit.num_qubits),
        "structure": structure_of(point.circuit),
        "k": point.k,
        "distance_scalar": point.distance_scalar,
        "merging": point.merging,
        "patches": len({patch for patch, _ in plan.placement}),
        "fell_back": list(plan.placement) == list(naive),
        "error_improvement": 100.0 * (naive_error - aqorn_error) / naive_error,
        "depth_improvement": (100.0 * (naive_depth - aqorn_depth) / naive_depth
                              if naive_depth else 0.0),
        "two_qubit_improvement": (100.0 * (naive_two - aqorn_two) / naive_two
                                  if naive_two else 0.0),
    }


def load_corpus(settings) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """The merging figure's loader and bounds, so this tests the corpus the paper uses."""
    pool: Dict[str, qiskit.QuantumCircuit] = {}
    for suite in settings.suites:
        for name, circuit in load_proxy_circuits(
            settings.benchmark_root / suite,
            gate_limit=2_000, qubit_limit=150, timeout_seconds=1.0,
            replace_rz_rx_with_s=True,
        ):
            if circuit.size() <= settings.max_gates:
                pool[name] = circuit
    ordered = sorted(pool.items(), key=lambda item: (item[1].num_qubits, item[0]))
    if settings.circuits is not None:
        # Spread the sample over the width range rather than taking the narrowest, so a
        # trimmed run still covers the wide circuits where placement matters most.
        step = max(1, len(ordered) // settings.circuits)
        ordered = ordered[::step][: settings.circuits]
    return ordered


def report(rows: Sequence[Dict], group: str, label: str) -> List[str]:
    """One table, grouped by a key. Returns the names of cells that failed."""
    cells: Dict[object, List[Dict]] = defaultdict(list)
    for row in rows:
        cells[row[group]].append(row)
    print(f"\n  === by {label} ===")
    print(f"  {label:>12s} {'n':>4s} {'mean err':>9s} {'worst':>8s} {'regress':>8s} "
          f"{'fellback':>9s} {'depth':>8s} {'2q':>7s}")
    failed: List[str] = []
    for key in sorted(cells, key=lambda k: (isinstance(k, str), k)):
        bucket = cells[key]
        gains = [r["error_improvement"] for r in bucket]
        regressions = [r for r in gains if r < -TIE]
        mean = st.mean(gains)
        flag = ""
        if regressions:
            flag = "  <== REGRESSION"
            failed.append(f"{label}={key}: {len(regressions)} regression(s), "
                          f"worst {min(gains):+.2f}%")
        elif mean <= TIE:
            flag = "  <== no gain"
            failed.append(f"{label}={key}: mean {mean:+.2f}% is not positive")
        print(f"  {str(key):>12s} {len(bucket):>4d} {mean:>8.2f}% {min(gains):>7.2f}% "
              f"{len(regressions):>8d} "
              f"{sum(1 for r in bucket if r['fell_back']):>9d} "
              f"{st.mean([r['depth_improvement'] for r in bucket]):>7.2f}% "
              f"{st.mean([r['two_qubit_improvement'] for r in bucket]):>6.2f}%"
              f"{flag}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/qasmbench"))
    parser.add_argument("--suites", nargs="+", default=["medium", "large"])
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--distance-scalars", nargs="+", type=float,
                        default=list(DEFAULT_DISTANCE_SCALARS),
                        help="Movement error multiplier. The calibrated span is 0.5 to 8.")
    parser.add_argument("--circuits", type=int, default=None,
                        help="Sample this many circuits, spread across the width range.")
    parser.add_argument("--max-gates", type=int, default=2_500)
    parser.add_argument("--merging", nargs="+", type=int, default=[1],
                        help="1 for merging on, 0 for off. Pass '1 0' to sweep both.")
    parser.add_argument("--quick", action="store_true",
                        help="8 circuits, k 2 and 6, distance_scalar 1 and 8. A smoke test.")
    add_jobs_argument(parser)
    args = parser.parse_args()

    if args.quick:
        args.circuits = args.circuits or 8
        args.k_values = [2, 6]
        args.distance_scalars = [8.0, 1.0]

    corpus = load_corpus(args)
    if not corpus:
        print("no usable circuits", file=sys.stderr)
        return 1

    points = [
        Point(name=name, circuit=circuit, k=k, distance_scalar=scalar,
              merging=bool(flag))
        for name, circuit in corpus
        for k in args.k_values
        for scalar in args.distance_scalars
        for flag in args.merging
        # A circuit narrower than one patch has no placement to make: naive and AQORN are
        # the same permutation and the point is a tautology reading exactly 0.
        if ceil(circuit.num_qubits / k) >= 2
    ]
    print(f"cost model version {COST_MODEL_VERSION}")
    print(f"{len(corpus)} circuits x {len(args.k_values)} k x "
          f"{len(args.distance_scalars)} distance_scalar x {len(args.merging)} merging "
          f"= {len(points)} points (tautologies dropped)")
    structures = defaultdict(int)
    for name, circuit in corpus:
        structures[structure_of(circuit)] += 1
    print(f"structures: {dict(structures)}")

    results = parallel_map(run_point, points, jobs=args.jobs, label="placement")
    rows = [row for row in results if row]
    dropped = len(points) - len(rows)
    print(f"\n{len(rows)} points measured" + (f", {dropped} unusable" if dropped else ""))
    if not rows:
        print("nothing measured", file=sys.stderr)
        return 1

    failed: List[str] = []
    failed += report(rows, "distance_scalar", "dist_scalar")
    failed += report(rows, "k", "k")
    failed += report(rows, "structure", "structure")
    failed += report(rows, "band", "width")

    gains = [row["error_improvement"] for row in rows]
    regressions = [row for row in rows if row["error_improvement"] < -TIE]
    print(f"\n  === overall ===")
    print(f"  {len(rows)} points, mean {st.mean(gains):+.2f}%, median "
          f"{st.median(gains):+.2f}%, worst {min(gains):+.2f}%, best {max(gains):+.2f}%")
    print(f"  fell back to naive: {sum(1 for r in rows if r['fell_back'])}/{len(rows)}")
    print(f"  regressions: {len(regressions)}")
    for row in sorted(regressions, key=lambda r: r["error_improvement"])[:10]:
        print(f"    {row['circuit']:<18s} k={row['k']:<3d} "
              f"dist={row['distance_scalar']:<5.4g} {row['error_improvement']:+.2f}%")

    if failed:
        print(f"\nFAILED -- {len(failed)} cell(s) did not hold the invariant:")
        for line in failed:
            print(f"  {line}")
        return 1
    print("\nPASSED -- no regressions, positive mean in every cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
