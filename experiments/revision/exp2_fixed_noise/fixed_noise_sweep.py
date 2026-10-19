"""
fixed_noise_sweep.py -- Experiment2b + Experiment4: one physical machine for every circuit.

Question: do the LSR and TVD gains hold on one fixed physical machine for every circuit,
with absolute values and uncertainties? Also: controlled accepted-shot counts, an
equal-accepted-shot comparison, the finite-sample floor, raw/accepted counts, and a
feasibility count.

Fig 6 calibrates the noise scale per circuit so the baseline keeps ~6% of its shots,
which makes every circuit measurable but describes ~71 different machines (alpha spans
0.22-20.8; see Experiment2a). Here alpha is the *same* for every circuit:

    alpha = 1     the NEUTRAL_ATOM machine itself (Table 2's model). All four Fig 6 arms:
                  baseline (naive, both passes off), +H commutation, +merging, AQORN.
    alpha = 0.5, 2, 4   baseline and AQORN only.

A fixed raw shot budget per point (200k by default), Stim, distance and idle noise on at
scalar 1, the Fig 6 proxies, and AQORN's placement exactly as Fig 6 computed it (planned
on the original with Fig 6's noise profile; taken from Experiment1a's CSV when present).
Commutation follows the planner's per-circuit decision for every arm, as in Fig 6.

Experiment4 statistics, per simulated point: raw and accepted shots; a 200-sample
bootstrap 95% CI on TVD (resampling accepted counts and the sampled ideal reference); the
noiseless finite-sample TVD floor at that point's accepted count; and TVD at equal
accepted shots for the pairs (baseline, AQORN) and (+merging, AQORN) -- the arm with more
accepted shots is downsampled without replacement to the other's count, 20 draws.

A point is "post-selection infeasible" when it keeps fewer than 1,000 shots of 200k: the
summary counts how many baselines are infeasible on the fixed machine and how many of
those AQORN brings back above the line.

Histograms (accepted counts and the ideal reference) are saved to a JSONL sidecar for
circuits of at most 12 qubits, where they are small enough to keep.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import (
    RESULTS,
    NoiseSetup,
    ResultTable,
    add_common_arguments,
    equal_shot_tvd,
    histogram_json,
    load_fig6_circuits,
    log,
    out_dir,
    sim_config,
    simulate_arm,
    stream_map,
    tvd_statistics,
)

NAME = "exp2b"
SEED_BASE = 20261001
GROUP_OFFSET = {"small": 0, "medium": 100, "large": 200}
INFEASIBLE_BELOW = 1000
HISTOGRAM_MAX_QUBITS = 12


@dataclass
class Settings:
    alphas: Sequence[float] = (0.5, 1.0, 2.0, 4.0)
    full_stack_alpha: float = 1.0
    shots: int = 200_000
    bootstrap: int = 200
    equal_draws: int = 20
    static_csv: str = str(RESULTS / "exp1" / "packer_static_o1.csv")
    #: The standard compiler (`packing.standard`): AQORN planned with `plan_standard` (no
    #: Experiment1a presets), every arm layer-sorted, the merging arms' in-patch CZ / XCX
    #: transversal -- Fig 6 as it runs now.
    standard: bool = False
    smoke: bool = False
    names: Sequence[str] = ()
    jobs: int = 12
    out: str = str(RESULTS / "exp2")


SMOKE_NAMES = ("hs4_n4", "adder_n10", "ising_n42")


def seed_for(circuit, alpha_index: int, offset: int) -> int:
    global_index = GROUP_OFFSET.get(circuit.size_group, 300) + circuit.circuit_idx
    return SEED_BASE + 1_000_003 * global_index + 1009 * alpha_index + offset


def aqorn_from_static(path: str) -> Dict[str, Tuple[bool, List]]:
    p = Path(path)
    if not p.exists():
        return {}
    frame = pd.read_csv(p)
    frame = frame[(frame["n_code"] == 4) & (frame["arm"] == "aqorn") & (frame["status"] == "ok")]
    return {
        row.name: (bool(row.commute), [tuple(x) for x in json.loads(row.placement)])
        for row in frame.itertuples(index=False)
    }


def _run_circuit(job) -> Dict[str, List]:
    from experiments.revision.common import fig6_noise_profile
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing, plan_packing

    settings_dict, circuit, preset = job
    settings = Settings(**settings_dict)
    if settings.standard:
        from packing.standard import plan_standard

        plan = plan_standard(ensure_measurements(circuit.original))
        commute, aqorn = bool(plan.h_commutation), [tuple(p) for p in plan.placement]
    elif preset is None:
        plan = plan_packing(ensure_measurements(circuit.original), 4, h_commutation="auto",
                            verify=True, noise=fig6_noise_profile())
        commute, aqorn = bool(plan.h_commutation), [tuple(p) for p in plan.placement]
    else:
        commute, aqorn = preset
    naive = [tuple(p) for p in naive_packing(circuit.num_qubits, 2)]
    proxy = ensure_measurements(circuit.proxy)
    shots = 2000 if settings.smoke else settings.shots
    rng = np.random.default_rng(seed_for(circuit, 99, 99))

    arms_full = [("baseline", naive, False, False, 1), ("h_only", naive, commute, False, 2),
                 ("h_merge", naive, commute, True, 3), ("aqorn", aqorn, commute, True, 4)]
    arms_pair = [arms_full[0], arms_full[3]]

    rows: List[Dict] = []
    histograms: List[Dict] = []
    for alpha_index, alpha in enumerate(settings.alphas):
        arms = arms_full if abs(alpha - settings.full_stack_alpha) < 1e-12 else arms_pair
        noise = NoiseSetup(alpha=float(alpha))
        results: Dict[str, Dict] = {}
        for series, placement, h, merge, offset in arms:
            began = time.time()
            seed = seed_for(circuit, alpha_index, offset)
            config = sim_config(noise, n_code=4, commute=h, merging=merge,
                                repetitions=shots, seed=seed)
            if settings.standard:
                from packing.standard import STANDARD

                config.translator_config.asap_order = STANDARD.asap_order
                config.translator_config.transversal_intra_two_qubit = (
                    STANDARD.transversal_intra_two_qubit and merge)
            result = simulate_arm(proxy, placement, config)
            accepted = int(round(result["lsr"] * shots))
            result.update(tvd_statistics(result["noisy"], result["ideal"], accepted=accepted,
                                         ideal_shots=shots, rng=rng,
                                         bootstrap=settings.bootstrap))
            result.update(seed=seed, sim_seconds=time.time() - began, h_commutation=h,
                          merging=merge)
            results[series] = result

        def matched(a: str, b: str) -> Tuple[float, float, int]:
            """TVD of a and b, both at min(accepted_a, accepted_b)."""
            na, nb = results[a]["accepted_shots"], results[b]["accepted_shots"]
            target = min(na, nb)
            ta, _ = equal_shot_tvd(results[a]["noisy"], results[a]["ideal"], accepted=na,
                                   target=target, rng=rng, draws=settings.equal_draws)
            tb, _ = equal_shot_tvd(results[b]["noisy"], results[b]["ideal"], accepted=nb,
                                   target=target, rng=rng, draws=settings.equal_draws)
            return ta, tb, target

        pairs = {"baseline": ("baseline", "aqorn")}
        if "h_merge" in results:
            pairs["h_merge"] = ("h_merge", "aqorn")
        equal = {ref: matched(*pair) for ref, pair in pairs.items()}

        for series, result in results.items():
            row = {
                "name": circuit.name, "size_group": circuit.size_group,
                "circuit_idx": circuit.circuit_idx, "family": circuit.family,
                "num_qubits": circuit.num_qubits, "alpha": float(alpha), "series": series,
                "commute": commute, "h_commutation": result["h_commutation"],
                "merging": result["merging"], "seed": result["seed"], "raw_shots": shots,
                "accepted_shots": result["accepted_shots"], "lsr": result["lsr"],
                "tvd": result["tvd"], "tvd_ci_lo": result["tvd_ci_lo"],
                "tvd_ci_hi": result["tvd_ci_hi"], "tvd_boot_sd": result["tvd_boot_sd"],
                "tvd_floor": result["tvd_floor"], "tvd_floor_sd": result["tvd_floor_sd"],
                "support_ideal": result.get("support_ideal"),
                "support_noisy": result.get("support_noisy"),
                "infeasible": bool(result["accepted_shots"] < INFEASIBLE_BELOW),
                "one_qubit_gates": result["one_qubit_gates"],
                "two_qubit_gates": result["two_qubit_gates"], "depth": result["depth"],
                "physical_qubits": result["physical_qubits"],
                "sim_seconds": result["sim_seconds"],
                "placement": json.dumps([list(p) for p in
                                         dict((a[0], a[1]) for a in arms)[series]]),
            }
            for ref, (ta, tb, target) in equal.items():
                if series in pairs[ref]:
                    row[f"tvd_equal_vs_{ref}_pair"] = ta if series == pairs[ref][0] else tb
                    row[f"equal_shots_{ref}_pair"] = target
            rows.append(row)
            if circuit.num_qubits <= HISTOGRAM_MAX_QUBITS:
                histograms.append({
                    "name": circuit.name, "alpha": float(alpha), "series": series,
                    "raw_shots": shots, "accepted_shots": result["accepted_shots"],
                    "noisy_counts": json.loads(histogram_json(result["noisy"],
                                                              result["accepted_shots"])),
                    "ideal_counts": json.loads(histogram_json(result["ideal"], shots)),
                })
    return {"rows": rows, "histograms": histograms}


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / "fixed_noise_sweep.csv", key_columns=["name", "alpha", "series"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    hist_path = out / "fixed_noise_histograms.jsonl"
    names = list(settings.names) or (list(SMOKE_NAMES) if settings.smoke else None)
    circuits = load_fig6_circuits(names=names)
    presets = {} if settings.standard else aqorn_from_static(settings.static_csv)
    last_alpha = float(settings.alphas[-1])
    jobs = [(asdict(settings), c, presets.get(c.name)) for c in circuits
            if not table.is_done((c.name, last_alpha, "aqorn"))]
    jobs.sort(key=lambda j: -j[1].num_qubits)
    log(NAME, f"{len(circuits)} circuits, alphas {list(settings.alphas)}, "
              f"{settings.shots:,} raw shots/point; {len(jobs)} to run "
              f"({sum(1 for j in jobs if j[2] is not None)} with 1a placements)")

    def on_result(_index: int, result: Dict[str, List]) -> None:
        with hist_path.open("a") as handle:
            for record in result["histograms"]:
                handle.write(json.dumps(record) + "\n")
        table.append(result["rows"])

    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME, on_result=on_result)
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    report(table.frame())
    return table.path


def report(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    pivot = frame.pivot_table(index=["name", "alpha"], columns="series",
                              values=["lsr", "accepted_shots", "tvd"])
    for alpha in sorted(frame["alpha"].unique()):
        block = pivot.xs(alpha, level="alpha")
        lsr = block["lsr"]
        acc = block["accepted_shots"]
        improved = (lsr["aqorn"] > lsr["baseline"]).mean() * 100
        ratio = np.exp(np.nanmean(np.log(lsr["aqorn"].clip(lower=1e-9) /
                                         lsr["baseline"].clip(lower=1e-9))))
        infeasible = acc["baseline"] < INFEASIBLE_BELOW
        rescued = (infeasible & (acc["aqorn"] >= INFEASIBLE_BELOW)).sum()
        print(f"[{NAME}] alpha={alpha}: baseline LSR mean {lsr['baseline'].mean():.3f}, "
              f"AQORN {lsr['aqorn'].mean():.3f}; AQORN > baseline on {improved:.0f}% of "
              f"circuits; geometric-mean LSR ratio {ratio:.3f}; baseline infeasible "
              f"(<{INFEASIBLE_BELOW} accepted) on {int(infeasible.sum())}, AQORN rescues "
              f"{int(rescued)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp2")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--static-csv", default=str(RESULTS / "exp1" / "packer_static_o1.csv"))
    parser.add_argument("--standard", action="store_true",
                        help="the standard compiler (Settings.standard); use its own --out")
    args = parser.parse_args()
    settings = Settings(alphas=tuple(args.alphas), shots=args.shots, static_csv=args.static_csv,
                        standard=args.standard,
                        smoke=args.smoke, names=tuple(args.names), jobs=args.jobs,
                        out=str(out_dir(args, "exp2")))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
