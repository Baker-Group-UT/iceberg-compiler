"""
planner_simulated.py -- Experiment8b: silent-aware placements simulated against AQORN's.

Fig 6's machine and calibration, as Experiment1c uses them: each circuit's `alpha_final`
and `final_repetitions`, Stim with distance and idle noise at scalar 1, and Fig 6's seed
derivation. Per circuit, for every circuit where some weight changed the plan:

    naive          offset 5 -- re-draws Fig 6's +H+merge samples (the TVD-vs-naive reference)
    aqorn          offset 6 -- re-draws Fig 6's AQORN samples; must match Fig 6 (the gate)
    silent_w<w>    offset 300 + i, simulated only when its (placement, commutation) differs
                   from every arm already simulated; otherwise it reuses that arm's result
                   (`duplicate_of`)

Circuits where no weight changed anything are exact ties and are not re-simulated.

Every simulated arm also gets the frame measurement of its own noisy circuit
(`frames.split_from_stim`, captured from the simulator's conversion): P(detected),
P(silent), P(silent | kept). That measures what the cost model tried to reduce at any
width, including circuits too wide for TVD to resolve.

Experiment4 statistics per arm as in 1c (bootstrap CI on TVD, the noiseless floor,
equal-accepted-shot TVD against naive), plus equal-shot TVD against AQORN directly.

`--alpha A` runs the same arms on one fixed machine instead (Common-3's question), at
`--fixed-shots` raw shots and seeds offset away from Fig 6's, so there is no gate. It
matters here more than elsewhere: Fig 6 calibrates alpha per circuit to a baseline LSR of
~0.06, which puts the 3-4 qubit circuits at alpha 6-21 -- two-qubit error up to 6% --
where most silent errors are *pairs* of faults whose syndromes cancel, a regime no
per-gadget model describes. At alpha = 1 the first-order picture the cost model is built
on should hold. Circuits whose AQORN LSR at that alpha would extrapolate below
`min_expected_lsr` (-ln LSR scales linearly in alpha) are skipped: nothing would survive.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    fig6_reference,
    load_fig6_circuits,
    log,
    out_dir,
    percent_change,
    sim_config,
    simulate_arm,
    stream_map,
    tvd_statistics,
)
from experiments.revision.exp1_packers.packer_simulated import check_gate, fig6_seed

NAME = "exp8b"
GATE_CIRCUITS = ("adder_n10", "dnn_n8", "hs4_n4", "bell_n4")


@dataclass
class Settings:
    static_csv: str = str(RESULTS / "exp8" / "planner_static.csv")
    bootstrap: int = 200
    frame_shots: int = 50_000
    alpha: Optional[float] = None  # None: Fig 6's per-circuit alpha and shots
    fixed_shots: int = 100_000
    min_expected_lsr: float = 0.005
    smoke: bool = False
    smoke_shots: int = 2000
    names: Sequence[str] = ()
    skip_gate: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp8")


def arm_plans(static: pd.DataFrame) -> Dict[str, List[Dict]]:
    """{circuit: [{series, placement, commute, offset}]}, naive and AQORN first."""
    from packing.plan import naive_packing

    plans: Dict[str, List[Dict]] = {}
    for name, block in static.groupby("name"):
        block = block.sort_values("silent_weight")
        zero = block[block["silent_weight"] == 0.0].iloc[0]
        shipped = (zero["placement"], bool(zero["commute"]))
        naive = json.dumps([list(p) for p in naive_packing(int(zero["num_qubits"]), 2)])
        arms = [
            {"series": "naive", "placement": naive, "commute": bool(zero["commute"]), "offset": 5},
            {"series": "aqorn", "placement": shipped[0], "commute": shipped[1], "offset": 6},
        ]
        for index, row in enumerate(block[block["silent_weight"] > 0].itertuples()):
            arms.append({"series": f"silent_w{row.silent_weight:g}", "placement": row.placement,
                         "commute": bool(row.commute), "offset": 300 + index,
                         "weight": float(row.silent_weight)})
        plans[name] = arms
    return plans


def changed(arms: List[Dict]) -> bool:
    shipped = (arms[1]["placement"], arms[1]["commute"])
    return any((a["placement"], a["commute"]) != shipped for a in arms[2:])


def _run_circuit(job) -> List[Dict]:
    from experiments.revision.exp8_undetected.frames import capture_noisy_stim, split_from_stim
    from experiments.shared.simulation import ensure_measurements

    settings_dict, circuit, reference, arms = job
    settings = Settings(**settings_dict)
    proxy = ensure_measurements(circuit.proxy)
    fixed = settings.alpha is not None
    alpha = float(settings.alpha) if fixed else float(reference["alpha_final"])
    shots = (settings.smoke_shots if settings.smoke
             else settings.fixed_shots if fixed else int(reference["final_repetitions"]))
    #: Fixed-alpha runs draw away from Fig 6's samples; Fig 6's offsets stay Fig 6's.
    shift = 2000 if fixed else 0
    noise = NoiseSetup(alpha=alpha)
    rng = np.random.default_rng(fig6_seed(circuit.circuit_idx, 998 + shift))

    simulated: Dict[tuple, tuple] = {}
    results: Dict[str, Dict] = {}
    rows: List[Dict] = []
    for arm in arms:
        placement = [tuple(p) for p in json.loads(arm["placement"])]
        key = (arm["placement"], arm["commute"])
        duplicate_of = ""
        if key in simulated and arm["series"] not in {"naive", "aqorn"}:
            result, duplicate_of = simulated[key]
        else:
            seed = fig6_seed(circuit.circuit_idx, arm["offset"] + shift)
            began = time.time()
            config = sim_config(noise, n_code=4, commute=arm["commute"], merging=True,
                                repetitions=shots, seed=seed)
            with capture_noisy_stim() as captured:
                result = simulate_arm(proxy, placement, config)
            result["sim_seconds"] = time.time() - began
            result["seed"] = seed
            encoded, noisy = captured[-1]
            result.update(split_from_stim(encoded, noisy, shots=settings.frame_shots,
                                          seed=seed))
            accepted = int(round(result["lsr"] * shots))
            result.update(tvd_statistics(result["noisy"], result["ideal"], accepted=accepted,
                                         ideal_shots=shots, rng=rng,
                                         bootstrap=settings.bootstrap))
            simulated.setdefault(key, (result, arm["series"]))
        results[arm["series"]] = result
        rows.append({**arm, "duplicate_of": duplicate_of, **result})

    naive_result, shipped_result = results["naive"], results["aqorn"]
    out: List[Dict] = []
    for row in rows:
        accepted = int(row["accepted_shots"])
        versus = {}
        for label, other in (("naive", naive_result), ("aqorn", shipped_result)):
            target = min(accepted, int(other["accepted_shots"]))
            mine, mine_sd = equal_shot_tvd(row["noisy"], row["ideal"], accepted=accepted,
                                           target=target, rng=rng)
            theirs, _ = equal_shot_tvd(other["noisy"], other["ideal"],
                                       accepted=int(other["accepted_shots"]), target=target,
                                       rng=rng)
            versus[f"tvd_eq_vs_{label}"] = mine
            versus[f"tvd_eq_vs_{label}_sd"] = mine_sd
            versus[f"{label}_tvd_at_matched_shots"] = theirs
        out.append({
            "name": circuit.name, "size_group": circuit.size_group,
            "circuit_idx": circuit.circuit_idx, "family": circuit.family,
            "num_qubits": circuit.num_qubits, "alpha": alpha, "raw_shots": shots,
            "series": row["series"], "silent_weight": row.get("weight", 0.0),
            "duplicate_of": row["duplicate_of"], "commute": row["commute"],
            "seed": row.get("seed"), "lsr": row["lsr"], "tvd": row["tvd"],
            "accepted_shots": accepted, "tvd_ci_lo": row["tvd_ci_lo"],
            "tvd_ci_hi": row["tvd_ci_hi"], "tvd_boot_sd": row["tvd_boot_sd"],
            "tvd_floor": row["tvd_floor"], "tvd_floor_sd": row["tvd_floor_sd"],
            "support_ideal": row.get("support_ideal"), **versus,
            "p_detected": row["p_detected"], "p_silent": row["p_silent"],
            "p_silent_given_kept": row["p_silent_given_kept"],
            "frame_shots": row["frame_shots"],
            "lsr_impr_vs_aqorn_pct": percent_change(shipped_result["lsr"], row["lsr"]),
            "tvd_red_vs_aqorn_pct": -percent_change(shipped_result["tvd"], row["tvd"]),
            "lsr_impr_vs_naive_pct": percent_change(naive_result["lsr"], row["lsr"]),
            "tvd_red_vs_naive_pct": -percent_change(naive_result["tvd"], row["tvd"]),
            "one_qubit_gates": row["one_qubit_gates"], "two_qubit_gates": row["two_qubit_gates"],
            "depth": row["depth"], "sim_seconds": row.get("sim_seconds", 0.0),
            "fig6_naive_lsr": float(reference["hmerge_lsr"]),
            "fig6_packed_lsr": float(reference["packed_lsr"]),
            "fig6_baseline_tvd": float(reference["baseline_tvd"]),
            "placement": row["placement"],
        })
    return out


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    reference = fig6_reference().set_index("name")
    static = pd.read_csv(settings.static_csv)
    plans = arm_plans(static)
    names = list(settings.names) or (list(GATE_CIRCUITS) if settings.smoke else sorted(plans))
    names = [n for n in names if n in plans]
    fixed = settings.alpha is not None
    if fixed:
        # -ln LSR is linear in alpha, so Fig 6's AQORN LSR at alpha_final extrapolates.
        expected = {n: float(reference.loc[n, "packed_lsr"]) ** (
            settings.alpha / float(reference.loc[n, "alpha_final"])) for n in names}
        skipped = sorted(n for n in names if expected[n] < settings.min_expected_lsr)
        names = [n for n in names if n not in skipped]
        log(NAME, f"alpha = {settings.alpha:g}: skipping {len(skipped)} circuits whose AQORN "
                  f"LSR would extrapolate below {settings.min_expected_lsr}")
    circuits = {c.name: c for c in load_fig6_circuits(names=names)}
    stem = "planner_simulated" + (f"_alpha{settings.alpha:g}" if fixed else "")
    table = ResultTable(out / f"{stem}.csv", key_columns=["name", "series"],
                        experiment=NAME, settings=asdict(settings), resume=resume)

    if not settings.skip_gate and not settings.smoke and not fixed:
        log(NAME, f"sanity gate: re-drawing Fig 6's naive and AQORN samples on {list(GATE_CIRCUITS)}")
        gate_jobs = [(asdict(settings), circuits[n], reference.loc[n].to_dict(), plans[n][:2])
                     for n in GATE_CIRCUITS if n in circuits]
        gate_rows: List[Dict] = []
        stream_map(_run_circuit, gate_jobs, jobs=min(settings.jobs, len(gate_jobs)),
                   label=f"{NAME}-gate", on_result=lambda _i, r: gate_rows.extend(r))
        problems = check_gate(gate_rows)
        hard = [p for p in problems if not p.startswith("(within noise")]
        gate_dir = out / "_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        (gate_dir / "GATE.txt").write_text(
            ("PASS" if not hard else "FAIL") + "\n" + "\n".join(problems) + "\n")
        if hard or len(gate_rows) < 2 * len(gate_jobs):
            table.write_sidecar("stopped_at_sanity_gate", extra={"problems": problems})
            log(NAME, "SANITY GATE FAILED: Experiment8b stopped; see _gate/GATE.txt")
            sys.exit(3)
        log(NAME, "sanity gate passed: Fig 6's naive and AQORN samples reproduced")

    unchanged = [n for n in names if not changed(plans[n])]
    jobs = [(asdict(settings), circuits[n], reference.loc[n].to_dict(), plans[n])
            for n in names if changed(plans[n]) and n in circuits
            and not table.is_done((n, "aqorn"))]
    jobs.sort(key=lambda j: -j[1].num_qubits * float(j[2]["final_repetitions"]))
    log(NAME, f"{len(jobs)} circuits to simulate; {len(unchanged)} unchanged by every weight "
              f"(exact ties, not re-simulated)")
    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"], "unchanged": unchanged})
    frame = table.frame()
    if not frame.empty:
        if not fixed:
            problems = check_gate(frame.to_dict("records"))
            log(NAME, f"Fig 6 reproduction across simulated circuits: {len(problems)} "
                      f"discrepancies")
        print(frame.groupby("series")[["lsr_impr_vs_aqorn_pct", "tvd_red_vs_aqorn_pct",
                                       "p_silent_given_kept"]].mean().round(3).to_string())
    return table.path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp8")
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--static-csv", default=None)
    parser.add_argument("--alpha", type=float, default=None,
                        help="One fixed machine for every circuit instead of Fig 6's "
                             "per-circuit calibration.")
    parser.add_argument("--fixed-shots", type=int, default=100_000)
    args = parser.parse_args()
    out = out_dir(args, "exp8")
    settings = Settings(static_csv=args.static_csv or str(out / "planner_static.csv"),
                        alpha=args.alpha, fixed_shots=args.fixed_shots,
                        smoke=args.smoke, names=tuple(args.names), skip_gate=args.skip_gate,
                        jobs=args.jobs, out=str(out))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
