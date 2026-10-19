"""
seed_records.py -- make the compiler's logical optimisation pass return, for Fig 6's 71
circuits, exactly the circuits the logical-optimisation study chose, verified and measured
(reports/logical_opt.md), rather than a fresh O3 draw (O3 is not deterministic).

The pass's last step writes rotations at Clifford angles as named gates
(`translator_cliffords`), so each seeded answer is the study's circuit after that step.

For every Fig 6 circuit: the original as the study saw it (Fig 6's loader with the pass off --
level-1 transpile, no measurements), and the study's answer -- its saved optimised circuit
(`<study>/circuits/<name>.qpy`) where Stage A changed it, "unchanged" where it did not. Each
answer is checked to be the original's unitary again (`seed_record`) and recorded as the pass's
answer for that program (`translator.logical_optimization`; first writer wins, so a program
already on record keeps its record and is reported). Then, with the pass on, Fig 6's loader is
run again and every original and proxy must come back as the study's circuit, gate for gate.

No new rotations (2026-10-10): 7 of the study's answers carry more non-Clifford rotations than
their programs (basis_trotter_n4, qaoa_n6, variational_n4, dnn_n33, dnn_n51, qugan_n39,
qugan_n71), which the pass no longer returns. Their records are dropped and a round under the
rule decides -- the best verified candidate that adds none, or the program as it is -- recorded
like any other; set `AQORN_LOGICAL_OPT_TIMEOUT` (seconds) generously. Those programs must then
come back as that answer, and no program may come back with a rotation it did not have.

    AQORN_LOGICAL_OPT_TIMEOUT=900 PYTHONPATH=. venv/bin/python -m experiments.logical_opt.seed_records
    ... --study /path/to/results/logical_opt          # default: the main checkout's
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd


def _ops(circuit):
    return [(inst.operation.name, tuple(circuit.find_bit(q).index for q in inst.qubits),
             tuple(float(p) for p in inst.operation.params))
            for inst in circuit.data if inst.operation.name not in ("measure", "barrier")]


def main() -> None:
    from qiskit import qpy

    from experiments.revision.common import load_fig6_circuits
    from translator import logical_optimization as lo
    from translator.logical_optimizer import proxy_of

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--study", type=Path,
                        default=lo._main_checkout() / "results" / "logical_opt")
    parser.add_argument("--replace", nargs="*", default=[],
                        help="drop these circuits' existing records first, so the study's answer "
                             "replaces one recorded by mistake (nothing else may be running)")
    args = parser.parse_args()

    stage_a = pd.read_csv(args.study / "stage_a.csv").set_index("name")
    with lo.disabled():
        originals = {c.name: c.original for c in load_fig6_circuits()}
    print(f"[seed] {len(originals)} Fig 6 originals; records in {lo.cache_dir()}")

    answers: Dict[str, object] = {}
    outcome: Dict[str, List[str]] = {"seeded": [], "agrees": [], "kept": [], "refused": []}
    for name, original in originals.items():
        changed = bool(stage_a.loc[name, "changed"])
        optimized = None
        if changed:
            with (args.study / "circuits" / f"{name}.qpy").open("rb") as stream:
                optimized = qpy.load(stream)[0]
        # what the pass returns: the study's circuit (or the original) with its Clifford-angle
        # rotations as named gates (`translator_cliffords`)
        answers[name] = lo.translator_cliffords(optimized if optimized is not None else original)
        if name in args.replace:
            print(f"[seed] dropping the existing record of {name}: {lo.drop_record(original)}")
        outcome[lo.seed_record(original, optimized)].append(name)
    for key, names in outcome.items():
        print(f"[seed] {key:7s} {len(names):3d}  {' '.join(names[:8])}{' ...' if len(names) > 8 else ''}")

    # No new rotations: a refused answer's older record goes, and a round under the rule decides.
    from translator.logical_optimizer import non_clifford_rotations, score

    for name in outcome["refused"]:
        original = originals[name]
        dropped = lo.drop_record(original)
        with lo.switched(True):
            answers[name] = lo.logically_optimized(original)
        with (args.study / "circuits" / f"{name}.qpy").open("rb") as stream:
            study = lo.translator_cliffords(qpy.load(stream)[0])
        print(f"[seed] {name}: study answer refused ({non_clifford_rotations(original)} -> "
              f"{non_clifford_rotations(study)} rotations; 2q {score(original)[0]} -> "
              f"{score(study)[0]}){'; its record dropped' if dropped else ''}; under the rule: "
              f"{non_clifford_rotations(answers[name])} rotations, 2q {score(answers[name])[0]}"
              f"{' (unchanged)' if answers[name] is original else ''}")

    # What the pipeline now loads: every original and proxy must be the study's circuit (or, for
    # a refused one, the rule's), and none may carry a rotation its program did not.
    lo.clear_memory()
    with lo.switched(True):
        loaded = load_fig6_circuits()
    from experiments.shared.simulation import ensure_measurements

    wrong = []
    with lo.switched(True):
        for circuit in loaded:
            expected = answers[circuit.name]
            # what the loader returns, and what the translator makes of the unoptimised
            # original once an experiment has added its measurements
            measured = lo.logically_optimized(ensure_measurements(originals[circuit.name]))
            if _ops(circuit.original) != _ops(expected) or \
                    _ops(circuit.proxy) != _ops(proxy_of(expected)) or \
                    _ops(measured) != _ops(expected) or \
                    non_clifford_rotations(circuit.original) > \
                    non_clifford_rotations(originals[circuit.name]):
                wrong.append(circuit.name)
    print(f"[seed] Fig 6's loader with the pass on returns the study's circuit for "
          f"{len(loaded) - len(wrong)}/{len(loaded)}" + (f"; differs: {wrong}" if wrong else ""))
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
