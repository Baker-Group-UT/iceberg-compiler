"""
noise_budget.py -- where Fig 6's error goes, stage by stage: the reductions in
movement and idle time behind the LSR gains.

For each of Fig 6's 71 circuits and its four cumulative arms (baseline: naive packing, both
passes off; +H commutation; +merging; +packing = AQORN's plan), the original is translated under
the standard settings (layer-sorted, transversal in-patch CZ / XCX in the merging arms), its
rotations are given Clifford stand-ins of the same duration, and the simulator's own schedule is
priced by `simulator.attribution.measured_noise_split` on Fig 6's machine (alpha = 1, idle and
distance scalars 1): expected detectable errors from gates, idling and atom movement (handling +
flight). The amounts scale with alpha together, so the percentages hold at any alpha.

Plans and commutation come from Fig 6's final CSV (`ablation_standard_logical.csv`).

    PYTHONPATH=. venv/bin/python -m experiments.revision.noise_budget --jobs 71
Writes results/revision/noise_budget.csv and prints per-stage median / geometric-mean changes.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SOURCE = Path("experiments/figures/ablation_standard/ablation_standard_logical.csv")
OUT = Path("results/revision/noise_budget.csv")
TERMS = ("gates", "idle", "movement", "total")


def _budget(job):
    import warnings

    warnings.filterwarnings("ignore")
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing
    from packing.standard import standard_translator_config
    from simulator.attribution import clifford_stand_in, measured_noise_split
    from simulator.noise_models import StimNoiseParameters
    from translator.core import Translator

    name, original, placement, commute = job
    original = ensure_measurements(original)
    naive = [tuple(p) for p in naive_packing(original.num_qubits, 2)]
    arms = (("baseline", naive, False, False), ("h_only", naive, commute, False),
            ("h_merge", naive, commute, True), ("packed", placement, commute, True))
    rows = []
    for arm, perm, h, merge in arms:
        config = standard_translator_config(4, commute=h, merging=merge)
        config.transversal_intra_two_qubit = config.transversal_intra_two_qubit and merge
        circuit = Translator(config=config).translate(original, perm).circuit
        split = measured_noise_split(clifford_stand_in(circuit), StimNoiseParameters.neutral_atom())
        rows.append(dict(circuit=name, num_qubits=original.num_qubits, arm=arm,
                         **{k: float(v) for k, v in split.items()}))
    return rows


def main() -> None:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--jobs", type=int, default=71)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    plans = pd.read_csv(args.source).drop_duplicates("circuit_name").set_index("circuit_name")
    circuits = {c.name: c for c in load_fig6_circuits(names=list(plans.index))}
    jobs = [(name, circuits[name].original,
             [tuple(int(v) for v in p) for p in json.loads(plans.loc[name, "placement"])],
             bool(plans.loc[name, "commute"])) for name in plans.index]
    results = parallel_map(_budget, jobs, jobs=args.jobs, label="noise_budget")
    frame = pd.DataFrame([row for result in results if result for row in result])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[noise_budget] {frame.circuit.nunique()} circuits -> {out}")
    wide = {term: frame.pivot_table(index="circuit", columns="arm", values=term) for term in TERMS}
    share = frame[frame.arm == "baseline"][list(TERMS)].sum()
    print("baseline error budget (corpus sum): " + ", ".join(
        f"{t} {100 * share[t] / share['total']:.1f}%" for t in TERMS[:3]))
    for left, right, label in (("packed", "h_merge", "packing (AQORN vs +merging)"),
                               ("h_merge", "baseline", "passes (+merging vs baseline)"),
                               ("packed", "baseline", "all stages (AQORN vs baseline)")):
        parts = []
        for term in TERMS:
            w = wide[term]
            ok = (w[left] > 0) & (w[right] > 0)
            r = np.log(w.loc[ok, left] / w.loc[ok, right])
            moved = (w[right] > 0) & (w[left] == 0)
            parts.append(f"{term} median {100 * (np.exp(r.median()) - 1):+.1f}% / geomean "
                         f"{100 * (np.exp(r.mean()) - 1):+.1f}% (n {ok.sum()}"
                         + (f", to zero {int(moved.sum())}" if moved.any() else "") + ")")
        print(f"  {label}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
