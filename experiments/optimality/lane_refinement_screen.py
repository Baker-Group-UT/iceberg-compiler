"""
lane_refinement_screen.py -- would a second, placement-level round help AQORN? Grid alignment
(the paper's section 4.3.4, implemented in `packing.alignment.align_to_grid` but called by no
planner) and a noise-aware lane order (section 4.3.1 says lane order does not affect quality;
the exhaustive optimality sweep says it decides most of AQORN's logical-error gap at 5 qubits).

For every Fig 6 circuit, AQORN's placement as Fig 6 under the standard + logical optimisation
used it (`ablation_standard_logical_v2.csv`), four variants:

    A    as planned
    G    + `align_to_grid` (patches renumbered so the busiest sit centrally; groups and lanes
         untouched, so gate counts and depth cannot change)
    L    + lane refinement from A: patches in order of cross-patch traffic, flip each patch's
         lanes (a solo qubit moves to the other lane), keep the flip if the measured objective
         falls; one pass, at most `--budget` flips tried
    GL   lane refinement from G

The objective is the standard's own, measured instead of modelled: detected errors per shot +
silent_weight x silent logical errors per shot (occupied lanes), by Pauli frames on the noisy
Stim circuit the simulator runs (Fig 6's machine, alpha = 1), every candidate of a circuit on
the same seed (common random numbers). The four results are then re-measured on fresh seeds
with more shots, so a variant is not credited with the luck of its selection.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.lane_refinement_screen --jobs 12
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

NAME = "lane_refinement_screen"
PLANS = "experiments/figures/ablation_standard/ablation_standard_logical_v2.csv"


def _evaluate(program, placement, commute: bool, shots: int, seed: int, weight: float):
    """(objective, lsr, ler) of one placement, measured by frames, and the noisy circuit."""
    from experiments.optimality.exhaustive_front import frame_metrics
    from experiments.tvd_explore.fastframes import at_alpha_one
    from packing.standard import standard_translator_config
    from translator.core import Translator

    result = Translator(config=standard_translator_config(4, commute=commute)).translate(
        program, [tuple(p) for p in placement])
    positions = [tuple(p) for p in result.context.state_positions]
    noisy = at_alpha_one(result.circuit)
    m = frame_metrics(result.circuit, noisy, positions, [], shots=shots, seed=seed)
    return (1 - m["lsr"]) + weight * m["ler"] * m["lsr"], m, (result.circuit, noisy, positions)


def _flip(placement, patch: int):
    return [[p, 1 - lane] if p == patch else [p, lane] for p, lane in placement]


def _refine(program, placement, commute, order, budget, shots, seed, weight):
    best, metrics, built = _evaluate(program, placement, commute, shots, seed, weight)
    tried = accepted = 0
    for patch in order[:budget]:
        candidate = _flip(placement, patch)
        value, m, b = _evaluate(program, candidate, commute, shots, seed, weight)
        tried += 1
        if value < best:
            best, metrics, built, placement = value, m, b, candidate
            accepted += 1
    return placement, built, tried, accepted


def _run(job) -> List[Dict]:
    from experiments.optimality.exhaustive_front import frame_metrics
    from experiments.shared.simulation import ensure_measurements
    from packing.alignment import align_to_grid, patch_traffic
    from packing.standard import STANDARD

    name, n, original, proxy, placement, commute, settings = job
    weight = STANDARD.silent_weight_for(4)
    program = ensure_measurements(proxy)
    began = time.time()
    seed = 1000 + n

    def order_of(perm):
        traffic: Dict[int, int] = {}
        for (a, b), count in patch_traffic([tuple(p) for p in perm], original).items():
            traffic[a] = traffic.get(a, 0) + count
            traffic[b] = traffic.get(b, 0) + count
        patches = sorted({p for p, _ in perm})
        return sorted(patches, key=lambda p: -traffic.get(p, 0))

    aligned = [list(p) for p in align_to_grid([tuple(p) for p in placement], original)]
    variants: Dict[str, Tuple[list, tuple, int, int]] = {}
    for label, start in (("A", placement), ("G", aligned)):
        if label == "G" and aligned == placement:  # alignment kept the order: G = A, GL = L
            variants["G"], variants["GL"] = variants["A"], variants["L"]
            continue
        _, _, built = _evaluate(program, start, commute, settings["shots"], seed, weight)
        variants[label] = (start, built, 0, 0)
        refined, built_r, tried, accepted = _refine(program, start, commute, order_of(start),
                                                    settings["budget"], settings["shots"], seed,
                                                    weight)
        variants[label + "L" if label == "G" else "L"] = (refined, built_r, tried, accepted)
    row = {"circuit": name, "num_qubits": n, "patches": len({p for p, _ in placement}),
           "commute": commute, "aligned_changed": aligned != placement}
    for label, (perm, (encoded, noisy, positions), tried, accepted) in variants.items():
        m = frame_metrics(encoded, noisy, positions, [], shots=settings["final_shots"],
                          seed=seed + 7919)
        row.update({f"{label}_lsr": m["lsr"], f"{label}_ler": m["ler"],
                    f"{label}_objective": (1 - m["lsr"]) + weight * m["ler"] * m["lsr"],
                    f"{label}_tried": tried, f"{label}_accepted": accepted,
                    f"{label}_placement": json.dumps(perm)})
    row["seconds"] = round(time.time() - began, 1)
    return [row]


def main() -> None:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--final-shots", type=int, default=1_000_000)
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--out", default="results/optimality_front/lane_refinement_screen.csv")
    args = parser.parse_args()
    settings = {"budget": args.budget, "shots": args.shots, "final_shots": args.final_shots}

    plans = pd.read_csv(PLANS).drop_duplicates("circuit_name").set_index("circuit_name")
    circuits = load_fig6_circuits()
    jobs = [(c.name, c.original.num_qubits, c.original, c.proxy,
             json.loads(plans.loc[c.name, "placement"]), bool(plans.loc[c.name, "commute"]),
             settings)
            for c in circuits if (not args.names or c.name in args.names)]
    jobs.sort(key=lambda j: -j[1])
    results = parallel_map(_run, jobs, jobs=args.jobs, label=NAME)
    rows = [r for result in results if result for r in result]
    failed = [j[0] for j, r in zip(jobs, results) if r is None]
    frame = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"[{NAME}] {len(frame)} circuits -> {args.out}; failed: {failed}")
    for label in ("G", "L", "GL"):
        d_ler = 100 * (frame[f"{label}_ler"] / frame["A_ler"] - 1)
        d_lsr = 100 * (frame[f"{label}_lsr"] / frame["A_lsr"] - 1)
        d_obj = 100 * (frame[f"{label}_objective"] / frame["A_objective"] - 1)
        print(f"  {label:2s} vs A: objective median {d_obj.median():+.2f}% (better on "
              f"{(d_obj < 0).mean():.0%}), LER median {d_ler.median():+.2f}% (mean {d_ler.mean():+.2f}), "
              f"LSR median {d_lsr.median():+.2f}% (mean {d_lsr.mean():+.2f})")


if __name__ == "__main__":
    main()
