"""
bench.py — measure this package against the incumbent, by exact translation.

    python -m packing.bench --code-sizes 4 6 8 12

Nothing here is scored by the model. Every number is a real translation of the real
circuit with h-commutation and merging on, priced as weighted gates plus
`DEFAULT_DEPTH_WEIGHT` per layer -- the same figure of merit the search optimises, so a
win here is a win on the thing the search was aiming at rather than on a proxy for it.

The corpus is the QASMBench Clifford proxies `experiments/figures/merging_improvement`
uses, loaded through the same `create_proxy_from_qasm`, so the comparison is against the
circuits the committed figures were built from.
"""

from __future__ import annotations

import argparse
import glob
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import qiskit

from iceberg.utils import logical_capacity

from .plan import PackingPlan, measured_cost, naive_packing, plan_packing

Perm = List[Tuple[int, int]]

#: Proxies larger than this are dominated by translation time, not planning time.
DEFAULT_MAX_OPS = 4000


def load_corpus(
    root: str = "benchmarks/qasmbench",
    *,
    min_qubits: int = 4,
    max_qubits: int = 80,
    max_ops: int = DEFAULT_MAX_OPS,
) -> List[Tuple[str, qiskit.QuantumCircuit]]:
    """Every QASMBench proxy in the size band, transpiled to the supported basis."""
    from experiments.shared.benchmark_circuits import create_proxy_from_qasm

    out: List[Tuple[str, qiskit.QuantumCircuit]] = []
    for path in sorted(glob.glob(f"{root}/*/*/*.qasm")):
        if "transpiled" in path:
            continue
        circuit = create_proxy_from_qasm(Path(path), timeout_seconds=15.0)
        if circuit is None:
            continue
        if not min_qubits <= circuit.num_qubits <= max_qubits:
            continue
        if len(circuit.data) > max_ops:
            continue
        out.append((Path(path).stem, circuit))
    return out


def planners() -> Dict[str, Callable[[qiskit.QuantumCircuit, int], PackingPlan]]:
    """
    What to measure. The `grouping` incumbent used to be here and went with that
    package; its numbers are kept in the header so the comparison survives the code.

    These return the whole plan rather than just its placement, because the plan carries
    the H-commutation decision it was priced under and every measurement below has to
    honour it -- see `_measure_pair`.
    """
    return {
        "placement": lambda c, n: plan_packing(c, n),
        # `verify=False` is the raw search, with the safety net off. Reported separately
        # because it is the number that says whether the *algorithm* improved, while the
        # verified one says what a caller actually gets.
        "placement_raw": lambda c, n: plan_packing(c, n, verify=False),
    }


def _measure_pair(
    circuit: qiskit.QuantumCircuit,
    plan: PackingPlan,
    n_code: int,
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Cost of the plan and of naive, both under the plan's own commutation decision.

    Measuring at the default `h_commutation=True` instead is the bug this replaced.
    Since `plan_packing` began deciding the pass per (circuit, code size), 27 of 64
    corpus circuits at n=4 chose *off* -- and re-measuring those with it on described a
    circuit the pipeline never builds. `ising_n10` at n=4 came out 11198 against the
    verified 10158, +10.2%, and because both arms were re-measured under the wrong
    config the improvement could change sign outright.
    """
    ours = measured_cost(circuit, plan.placement, n_code,
                         h_commutation=plan.h_commutation)
    theirs = measured_cost(circuit, naive_packing(circuit.num_qubits, k), n_code,
                           h_commutation=plan.h_commutation)
    return ours, theirs


#: Cumulative percent reduction against a naive placement at the same code size, for
#: each technique in turn. `techniques()` measures these; the packing segment is what
#: this package is responsible for.
TECHNIQUES: Tuple[Tuple[str, bool, bool], ...] = (
    ("baseline", False, False),
    ("h_commutation", True, False),
    ("merging", True, True),
)


def techniques(
    code_sizes: Sequence[int] = (4, 6, 8, 12),
    *,
    max_qubits: int = 80,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    The three-technique decomposition: h commutation, then merging, then packing.

    This is what `experiments/figures/merging_improvement/merging_improvement_code_size`
    plotted, reduced to the numbers and dropped into the live tree, because that figure
    directory is archived and its script cannot run any more. Same measurement: every
    row is an exact translation, the first three share the naive placement and differ
    only in which translator passes are on, and the fourth adds this package's
    permutation on top of both passes.
    """
    from .plan import measured_cost as _cost

    corpus = load_corpus(max_qubits=max_qubits)
    if limit:
        corpus = corpus[:limit]
    rows: List[Dict] = []
    for name, circuit in corpus:
        for n_code in code_sizes:
            k = logical_capacity(n_code)
            naive = naive_packing(circuit.num_qubits, k)
            row: Dict = {"circuit": name, "qubits": circuit.num_qubits, "n_code": n_code}
            for label, commute, merge in TECHNIQUES:
                row[label] = _cost(
                    circuit, naive, n_code, h_commutation=commute, merging=merge
                )
            try:
                # `h_commutation=True` is pinned rather than left to `plan_packing`'s
                # per-circuit decision, and the plan is then measured under the same
                # setting -- so the plan and the measurement agree, which is what A4 was
                # about, without breaking what this table is.
                #
                # This is a *cumulative* ablation: each row turns on exactly one more
                # technique than the row above. Letting the planner choose `False` here
                # and honouring that would put the packing row on a different branch
                # from the merging row it is differenced against, so the segment would
                # no longer be the packing contribution. `run()` below does honour the
                # planner's choice, because there the question is what a caller gets
                # rather than what each technique is worth.
                plan = plan_packing(circuit, n_code, h_commutation=True)
                row["packing"] = _cost(
                    circuit, plan.placement, n_code, h_commutation=plan.h_commutation
                )
                row["packing_commuted"] = plan.h_commutation
            except Exception:  # noqa: BLE001
                row["packing"] = None
                row["packing_commuted"] = None
            rows.append(row)
    return rows


def summarize_techniques(rows: Sequence[Dict]) -> None:
    """
    Incremental reduction from each technique, meaned over circuits, per code size.

    Cumulative means are differenced rather than each segment meaned on its own, which
    is the same choice the archived figure made and for the same reason: when one
    technique hurts and a later one recovers, meaning the segments separately leaves the
    recovery at full height and the total exceeds the real reduction.
    """
    order = ["h_commutation", "merging", "packing"]
    print(f"\n{'n':>3} {'circuits':>9} " + "".join(f"{name:>16}" for name in order))
    for n_code in sorted({r["n_code"] for r in rows}):
        sel = [
            r for r in rows
            if r["n_code"] == n_code and r.get("baseline")
            and all(r.get(name) is not None for name in order)
        ]
        if not sel:
            continue
        cumulative = [
            statistics.fmean((1.0 - r[name] / r["baseline"]) * 100.0 for r in sel)
            for name in order
        ]
        segments = [cumulative[0]] + [
            cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))
        ]
        print(f"{n_code:>3} {len(sel):>9} "
              + "".join(f"{value:>15.2f}%" for value in segments))


def run(
    code_sizes: Sequence[int] = (4, 6, 8, 12),
    *,
    max_qubits: int = 80,
    limit: Optional[int] = None,
    quiet: bool = False,
) -> List[Dict]:
    corpus = load_corpus(max_qubits=max_qubits)
    if limit:
        corpus = corpus[:limit]
    engines = planners()
    rows: List[Dict] = []
    for name, circuit in corpus:
        for n_code in code_sizes:
            k = logical_capacity(n_code)
            row: Dict = {
                "circuit": name,
                "qubits": circuit.num_qubits,
                "n_code": n_code,
            }
            # `naive` is per-planner rather than a single column: each plan is scored
            # against naive translated under *that plan's* commutation decision, so the
            # delta is placement and nothing else.
            for label, planner in engines.items():
                start = time.perf_counter()
                try:
                    plan = planner(circuit, n_code)
                    ours, theirs = _measure_pair(circuit, plan, n_code, k)
                    row[label] = ours
                    row[f"{label}_naive"] = theirs
                    row[f"{label}_commuted"] = plan.h_commutation
                except Exception as exc:  # noqa: BLE001
                    row[label] = None
                    row[f"{label}_naive"] = None
                    row[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
                row[f"{label}_ms"] = (time.perf_counter() - start) * 1000.0
            row["naive"] = row.get("placement_naive")
            rows.append(row)
            if not quiet:
                cells = []
                for label in engines:
                    got = row.get(label)
                    base = row.get(f"{label}_naive")
                    if got is None or not base:
                        cells.append(f"{label}: --")
                    else:
                        cells.append(
                            f"{label}: {(got - base) / base * 100:+6.2f}% "
                            f"{row[f'{label}_ms']:>7.1f}ms"
                        )
                print(f"{name:<22} n={n_code:<3} m={circuit.num_qubits:<3} "
                      + "  ".join(cells), flush=True)
    return rows


def summarize(rows: Sequence[Dict], labels: Sequence[str]) -> None:
    print(f"\n{'planner':<16} {'n':>3} {'circuits':>9} {'mean':>8} {'median':>8} "
          f"{'wins':>5} {'ties':>5} {'losses':>7} {'mean ms':>9}")
    for label in labels:
        for n_code in sorted({r["n_code"] for r in rows}):
            sel = [r for r in rows
                   if r["n_code"] == n_code and r.get(f"{label}_naive")
                   and r.get(label) is not None]
            if not sel:
                continue
            deltas = [
                (r[label] - r[f"{label}_naive"]) / r[f"{label}_naive"] * 100
                for r in sel
            ]
            wins = sum(1 for d in deltas if d < -1e-9)
            losses = sum(1 for d in deltas if d > 1e-9)
            times = [r[f"{label}_ms"] for r in sel]
            print(f"{label:<16} {n_code:>3} {len(sel):>9} "
                  f"{statistics.fmean(deltas):>7.2f}% {statistics.median(deltas):>7.2f}% "
                  f"{wins:>5} {len(deltas) - wins - losses:>5} {losses:>7} "
                  f"{statistics.fmean(times):>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-sizes", type=int, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--max-qubits", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--techniques", action="store_true",
        help="Report the h-commutation / merging / packing decomposition instead of "
             "comparing planners. Replaces the archived code-size figure script.",
    )
    args = parser.parse_args()

    if args.techniques:
        summarize_techniques(
            techniques(args.code_sizes, max_qubits=args.max_qubits, limit=args.limit)
        )
        return

    rows = run(
        args.code_sizes,
        max_qubits=args.max_qubits,
        limit=args.limit,
        quiet=args.quiet,
    )
    summarize(rows, list(planners()))


if __name__ == "__main__":  # pragma: no cover
    main()
