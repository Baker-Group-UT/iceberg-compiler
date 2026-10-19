"""
standard_compile_time.py -- how long the final compiler takes on Fig 6's 71 circuits.

Compile time: the submitted "compiles in seconds" does not hold at 128
qubits. This times `plan_standard` (the whole planner: both H-commutation arms, the search,
the standard-objective checks) and the standard translation of its plan, once per circuit, each
in a fresh single-threaded worker after a warm-up plan on a small program. CPU seconds
(`process_time`) are the primary number; wall seconds are recorded beside them. Run it on a
quiet machine with far fewer workers than cores, so workers do not share a core.

Prints the per-size-group medians and a log-log fit of CPU seconds against width (n >= 16).

    PYTHONPATH=. venv/bin/python -m experiments.scaling.standard_compile_time --jobs 24
Writes results/scaling/standard_compile_time.csv.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("results/scaling/standard_compile_time.csv")


def _time(job):
    import warnings

    warnings.filterwarnings("ignore")
    from packing.standard import STANDARD, plan_standard, standard_pipeline_config
    from pipeline.tasks.translate_permutation import translate_permutation
    from qiskit import QuantumCircuit

    name, group, original = job
    warm = QuantumCircuit(4)
    warm.h(0)
    warm.cx(0, 1)
    warm.cx(2, 3)
    plan_standard(warm)
    cpu, wall = time.process_time(), time.perf_counter()
    plan = plan_standard(original)
    plan_cpu, plan_wall = time.process_time() - cpu, time.perf_counter() - wall
    cpu, wall = time.process_time(), time.perf_counter()
    translate_permutation(original, [tuple(p) for p in plan.placement],
                          config=standard_pipeline_config(commute=bool(plan.h_commutation)))
    return dict(circuit=name, size_group=group, num_qubits=original.num_qubits,
                gates=sum(1 for _ in original.data), patches=plan.patches,
                fell_back=bool(plan.fell_back), commute=bool(plan.h_commutation),
                plan_cpu_s=plan_cpu, plan_wall_s=plan_wall,
                translate_cpu_s=time.process_time() - cpu,
                translate_wall_s=time.perf_counter() - wall,
                restarts=STANDARD.restarts, load_after=os.getloadavg()[0])


def main() -> None:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    circuits = load_fig6_circuits()
    jobs = sorted(((c.name, c.size_group, c.original) for c in circuits),
                  key=lambda j: -j[2].num_qubits)
    rows = [r for r in parallel_map(_time, jobs, jobs=args.jobs, label="compile_time") if r]
    frame = pd.DataFrame(rows).sort_values("num_qubits")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(frame.groupby("size_group")[["num_qubits", "plan_cpu_s", "translate_cpu_s"]]
          .median().round(2).to_string())
    big = frame[frame.num_qubits >= 16]
    slope, intercept = np.polyfit(np.log(big.num_qubits), np.log(big.plan_cpu_s), 1)
    print(f"plan CPU seconds ~ n^{slope:.2f} (n >= 16, {len(big)} circuits); "
          f"max {frame.plan_cpu_s.max():.1f} s at n = "
          f"{int(frame.loc[frame.plan_cpu_s.idxmax(), 'num_qubits'])}; "
          f"median {frame.plan_cpu_s.median():.2f} s; wrote {out}")


if __name__ == "__main__":
    main()
