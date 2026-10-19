"""
testexp2.py -- see what merging actually did to a circuit.

Prints, for one program, three things side by side:

    1. the logical (program) circuit as written
    2. the translated physical circuit with merging OFF
    3. the same translation with merging ON

plus a gate-by-gate tally of what changed, so a macro that fired shows up as a drop in
the gates it replaced. `testexp.py` fuzzes for correctness; this one is for looking.

The point is the structured random circuits. Their gate mix is a knob
(`--two-qubit-ratio`, `--immutable-ratio`, `--locality`) and it is not obvious from a
percentage how much of it merging can reach, because a whole-patch macro needs a gate on
*every* occupied lane at once. This shows whether the mix being generated leaves merging
anything to do.

Run with:

    python -m experiments.tests.testexp2
    python -m experiments.tests.testexp2 --qubits 8 --n-code 6 --immutable-ratio 0.15
    python -m experiments.tests.testexp2 --benchmark adder_n4      # a real circuit instead
"""

from __future__ import annotations

import argparse
import glob
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import qiskit

from analysis.gate_count_stats import one_two_qubit_gate_totals
from experiments.shared.random_circuits import (
    intra_block_fraction,
    plant_blocks,
    structured_program_circuit,
)
from iceberg.utils import logical_capacity
from packing.passes import commutation_pays
from packing.plan import naive_packing, plan_packing
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation

#: Macros merging can produce. Seeing one of these appear is the whole point.
MACROS = ("hh", "cxt", "czt", "xcxt", "xx", "zz")


def build_config(n_code: int, *, h_commutation: bool, merging: bool) -> PipelineConfig:
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = h_commutation
    config.translator_config.enable_merging = merging
    return config


def load_benchmark(name: str) -> Optional[qiskit.QuantumCircuit]:
    from experiments.shared.benchmark_circuits import create_proxy_from_qasm

    for path in sorted(glob.glob("benchmarks/qasmbench/*/*/*.qasm")):
        if "transpiled" in path or Path(path).stem != name:
            continue
        return create_proxy_from_qasm(Path(path), timeout_seconds=10.0)
    return None


def tally(circuit: qiskit.QuantumCircuit) -> Counter:
    return Counter(instruction.operation.name.lower() for instruction in circuit.data)


def show(label: str, circuit: qiskit.QuantumCircuit, *, fold: int) -> None:
    one_q, two_q = one_two_qubit_gate_totals(circuit)
    print(f"\n{'=' * 78}\n{label}")
    print(
        f"  {circuit.num_qubits} qubits | {len(circuit.data)} ops | "
        f"1q {one_q} | 2q {two_q} | depth {circuit.depth()}"
    )
    print("=" * 78)
    print(circuit.draw(output="text", fold=fold))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--gates-per-qubit", type=int, default=6)
    parser.add_argument("--n-code", type=int, default=6, help="Code size; k = n_code - 2.")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Planted community size. Defaults to k, so blocks fit a patch.")
    parser.add_argument("--locality", type=float, default=1.0,
                        help="Share of 2q gates staying inside a block. 1.0 = fully "
                             "separable, which is where merging has the most to work with.")
    parser.add_argument("--two-qubit-ratio", type=float, default=0.40)
    parser.add_argument("--immutable-ratio", type=float, default=0.0,
                        help="Share of S/rotations. These are what BLOCK merging, so "
                             "raising this is how you watch the macros stop firing.")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Use a QASMBench proxy by name instead of a random circuit.")
    parser.add_argument("--packed", action="store_true",
                        help="Place with AQORN instead of the naive packing.")
    parser.add_argument("--fold", type=int, default=140,
                        help="Character width before the drawing wraps.")
    parser.add_argument("--no-draw", action="store_true",
                        help="Tallies only. Useful on anything wide.")
    args = parser.parse_args()

    k = logical_capacity(args.n_code)
    block_size = args.block_size or k

    if args.benchmark:
        program = load_benchmark(args.benchmark)
        if program is None:
            raise SystemExit(f"no QASMBench circuit named {args.benchmark!r}")
        blocks = None
        print(f"[testexp2] {args.benchmark}: {program.num_qubits} qubits, "
              f"{len(program.data)} ops")
    else:
        blocks = plant_blocks(args.qubits, block_size, seed=args.seed)
        program = structured_program_circuit(
            args.qubits, args.gates_per_qubit, seed=args.seed,
            blocks=blocks, locality=args.locality,
            two_qubit_ratio=args.two_qubit_ratio,
            immutable_ratio=args.immutable_ratio,
            clifford_only=True,
        )
        mix = tally(program)
        print(f"[testexp2] structured random, {args.qubits} qubits, k={k}, "
              f"blocks {blocks}")
        print(f"           requested locality {args.locality:.2f} -> measured "
              f"{intra_block_fraction(program, blocks):.3f}")
        print(f"           gate mix: {dict(sorted(mix.items()))}")

    commute = commutation_pays(program, args.n_code)
    if args.packed:
        plan = plan_packing(program, args.n_code, verify=True)
        placement, commute = plan.placement, plan.h_commutation
        print(f"           placement: AQORN, {plan.patches} patches "
              f"({'naive kept' if plan.fell_back else 'packed'})")
    else:
        placement = naive_packing(program.num_qubits, k)
        print(f"           placement: naive, "
              f"{len({p for p, _ in placement})} patches")
    print(f"           H commutation: {'on' if commute else 'off'} "
          "(the planner's own decision)")

    if not args.no_draw:
        show("1. LOGICAL (program) CIRCUIT", program, fold=args.fold)

    variants = {}
    for label, merging in (("merging OFF", False), ("merging ON", True)):
        config = build_config(args.n_code, h_commutation=commute, merging=merging)
        try:
            variants[label] = translate_permutation(
                program, placement, config=config
            ).circuit
        except Exception as error:  # noqa: BLE001
            print(f"\n!! translation with {label} was rejected: "
                  f"{type(error).__name__}: {error}")
            return

    for index, (label, circuit) in enumerate(variants.items(), start=2):
        if not args.no_draw:
            show(f"{index}. TRANSLATED, {label}", circuit, fold=args.fold)

    off, on = variants["merging OFF"], variants["merging ON"]
    before, after = tally(off), tally(on)
    print(f"\n{'=' * 78}\nWHAT MERGING DID\n{'=' * 78}")
    print(f"  {'gate':<12} {'off':>7} {'on':>7} {'change':>8}")
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name, 0), after.get(name, 0)
        if b == a:
            continue
        flag = "  <- macro" if name in MACROS else ""
        print(f"  {name:<12} {b:>7} {a:>7} {a - b:>+8}{flag}")

    fired = {name: after.get(name, 0) for name in MACROS if after.get(name, 0)}
    print()
    if fired:
        print(f"  macros fired: {fired}")
    else:
        print("  macros fired: NONE -- merging found nothing it could fuse. A whole-patch "
              "macro\n                needs a gate on every occupied lane at once, so this "
              "is the\n                expected outcome when k is large or the immutable "
              "share is high.")

    one_off, two_off = one_two_qubit_gate_totals(off)
    one_on, two_on = one_two_qubit_gate_totals(on)
    print(f"\n  1q {one_off} -> {one_on}   2q {two_off} -> {two_on}   "
          f"depth {off.depth()} -> {on.depth()}")


if __name__ == "__main__":
    main()
