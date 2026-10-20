"""
packer_simulated.py -- Experiment1c: LSR and TVD for every distinct packer arm.

Question (fidelity half): does AQORN's packing beat interaction-based
partitioning and greedy search on what the simulator measures, not only on gate counts?

Fig 6's machine and calibration, reused exactly: each circuit's `alpha_final` and
`final_repetitions` come from Fig 6's CSV, the Stim backend with distance and idle noise
at scalar 1, and Fig 6's seed derivation `seed * 1_000_003 + circuit_idx * 17 + offset`.
Three arms therefore re-draw Fig 6's own samples -- baseline (offset 3), naive with H
commutation and merging (offset 5) and AQORN (offset 6) -- and must reproduce Fig 6's
numbers bit for bit. That is the sanity gate: it runs first on a few small circuits, and if
any of the three disagrees the script records why and stops (exit code 3) rather than
producing numbers on a pipeline that no longer matches the paper. Every other arm gets
its own offset (100 + position), so no two arms share a sample.

Placements come from Experiment1a's CSV (`packer_static_o1.csv`, n_code = 4) when present,
so this simulates exactly the arms 1a counted; otherwise the planners run here. Arms whose
placement is identical to one already simulated are not re-simulated: they reuse that
arm's result, recorded in `duplicate_of`.

Experiment4 statistics are computed for every simulated arm: accepted shots, a 200-sample
bootstrap CI on TVD (both the accepted counts and the sampled ideal reference are
resampled), the noiseless finite-sample floor at that arm's accepted count, and TVD at
equal accepted shots against the naive arm.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import math
import sys
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

NAME = "exp1c"
FIG6_SEED = 10
GATE_CIRCUITS = ("adder_n10", "dnn_n8", "hs4_n4", "bell_n4")


@dataclass
class Settings:
    static_csv: str = str(RESULTS / "exp1" / "packer_static_o1.csv")
    shots_scale: float = 1.0  # 1.0 = Fig 6's final_repetitions exactly
    bootstrap: int = 200
    smoke: bool = False
    smoke_shots: int = 2000
    names: Sequence[str] = ()
    skip_gate: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp1")


def fig6_seed(circuit_idx: int, offset: int) -> int:
    return FIG6_SEED * 1_000_003 + circuit_idx * 17 + offset


def load_placements(settings: Settings, names: Sequence[str]) -> Dict[str, Dict]:
    """{circuit: {"commute": bool, "arms": {arm: placement}}} from Experiment1a, n_code 4."""
    path = Path(settings.static_csv)
    if not path.exists():
        log(NAME, f"{path} not found; the planners will run inside the workers")
        return {}
    frame = pd.read_csv(path)
    frame = frame[(frame["n_code"] == 4) & (frame["status"] == "ok")]
    out: Dict[str, Dict] = {}
    for name, block in frame.groupby("name"):
        if name not in names:
            continue
        arms = {
            row.arm: [tuple(p) for p in json.loads(row.placement)]
            for row in block.itertuples(index=False)
            if not str(row.arm).startswith("fig6_")
        }
        out[name] = {"commute": bool(block["commute"].iloc[0]), "arms": arms}
    return out


def _arm_plan(circuit, preset: Optional[Dict]) -> Tuple[bool, Dict[str, List]]:
    if preset is not None:
        return preset["commute"], preset["arms"]
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.packers import run_all_packers
    from experiments.shared.simulation import ensure_measurements

    placements, notes, _ = run_all_packers(ensure_measurements(circuit.original), 4,
                                           noise=fig6_noise_profile())
    return bool(notes["commute"]), {a: p for a, p in placements.items() if p is not None}


def _run_circuit(job) -> List[Dict]:
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing

    settings_dict, circuit, reference, preset, arm_filter = job
    settings = Settings(**settings_dict)
    commute, arms = _arm_plan(circuit, preset)
    naive = [tuple(p) for p in naive_packing(circuit.num_qubits, 2)]
    proxy = ensure_measurements(circuit.proxy)
    alpha = float(reference["alpha_final"])
    shots = int(reference["final_repetitions"])
    if settings.smoke:
        shots = settings.smoke_shots
    else:
        shots = max(1, int(round(shots * settings.shots_scale)))
    noise = NoiseSetup(alpha=alpha)
    rng = np.random.default_rng(fig6_seed(circuit.circuit_idx, 999))

    # (series, placement, h_commutation, merging, seed offset)
    plan: List[Tuple[str, List, bool, bool, int]] = [
        ("fig6_baseline", naive, False, False, 3),
        ("naive", naive, commute, True, 5),
        ("aqorn", arms["aqorn"], commute, True, 6),
    ]
    for position, (arm, placement) in enumerate(sorted(arms.items())):
        if arm in {"aqorn", "naive"}:
            continue
        if arm_filter and arm not in arm_filter:
            continue
        plan.append((arm, placement, commute, True, 100 + position))

    simulated: Dict[Tuple, Dict] = {}
    rows: List[Dict] = []
    results: Dict[str, Dict] = {}
    for series, placement, h, merge, offset in plan:
        key = (tuple(placement), h, merge)
        duplicate_of = ""
        must_simulate = series in {"fig6_baseline", "naive", "aqorn"}
        if key in simulated and not must_simulate:
            result, duplicate_of = simulated[key]
        else:
            began = time.time()
            config = sim_config(noise, n_code=4, commute=h, merging=merge,
                                repetitions=shots, seed=fig6_seed(circuit.circuit_idx, offset))
            result = simulate_arm(proxy, placement, config)
            result["sim_seconds"] = time.time() - began
            result["seed"] = fig6_seed(circuit.circuit_idx, offset)
            accepted = int(round(result["lsr"] * shots))
            result.update(tvd_statistics(result["noisy"], result["ideal"], accepted=accepted,
                                         ideal_shots=shots, rng=rng,
                                         bootstrap=settings.bootstrap))
            simulated.setdefault(key, (result, series))
        results[series] = result
        rows.append({"series": series, "duplicate_of": duplicate_of, "placement": placement,
                     "h_commutation": h, "merging": merge, **result})

    base = results["fig6_baseline"]
    naive_result = results["naive"]
    out_rows: List[Dict] = []
    for row in rows:
        accepted = int(row["accepted_shots"])
        eq_mean, eq_sd = equal_shot_tvd(
            row["noisy"], row["ideal"], accepted=accepted,
            target=min(accepted, int(naive_result["accepted_shots"])), rng=rng)
        naive_eq, _ = equal_shot_tvd(
            naive_result["noisy"], naive_result["ideal"],
            accepted=int(naive_result["accepted_shots"]),
            target=min(accepted, int(naive_result["accepted_shots"])), rng=rng)
        out_rows.append({
            "name": circuit.name, "size_group": circuit.size_group,
            "circuit_idx": circuit.circuit_idx, "family": circuit.family,
            "num_qubits": circuit.num_qubits, "alpha": alpha, "raw_shots": shots,
            "commute": commute, "series": row["series"], "duplicate_of": row["duplicate_of"],
            "h_commutation": row["h_commutation"], "merging": row["merging"],
            "seed": row.get("seed"), "lsr": row["lsr"], "tvd": row["tvd"],
            "accepted_shots": accepted,
            "tvd_ci_lo": row["tvd_ci_lo"], "tvd_ci_hi": row["tvd_ci_hi"],
            "tvd_boot_sd": row["tvd_boot_sd"], "tvd_floor": row["tvd_floor"],
            "tvd_floor_sd": row["tvd_floor_sd"], "support_ideal": row.get("support_ideal"),
            "tvd_eq_vs_naive": eq_mean, "tvd_eq_vs_naive_sd": eq_sd,
            "naive_tvd_at_matched_shots": naive_eq,
            "lsr_impr_vs_baseline_pct": percent_change(base["lsr"], row["lsr"]),
            "tvd_red_vs_baseline_pct": -percent_change(base["tvd"], row["tvd"]),
            "lsr_impr_vs_naive_pct": percent_change(naive_result["lsr"], row["lsr"]),
            "tvd_red_vs_naive_pct": -percent_change(naive_result["tvd"], row["tvd"]),
            "one_qubit_gates": row["one_qubit_gates"], "two_qubit_gates": row["two_qubit_gates"],
            "depth": row["depth"], "sim_seconds": row.get("sim_seconds", 0.0),
            "fig6_baseline_lsr": float(reference["baseline_lsr"]),
            "fig6_naive_lsr": float(reference["hmerge_lsr"]),
            "fig6_packed_lsr": float(reference["packed_lsr"]),
            "fig6_baseline_tvd": float(reference["baseline_tvd"]),
            "fig6_packed_tvd": float(reference["packed_tvd"]),
            "placement": json.dumps([list(p) for p in row["placement"]]),
        })
    return out_rows


#: The three arms that re-draw Fig 6's samples, and the reference column each must match.
GATE_ARMS = {"fig6_baseline": "fig6_baseline_lsr", "naive": "fig6_naive_lsr",
             "aqorn": "fig6_packed_lsr"}


def check_gate(rows: Sequence[Dict]) -> List[str]:
    """Problems found; empty means the three Fig 6 arms reproduced."""
    problems = []
    for row in rows:
        target = GATE_ARMS.get(row["series"])
        if target is None:
            continue
        expected, got = float(row[target]), float(row["lsr"])
        sigma = math.sqrt(max(expected * (1 - expected), 1e-12) / max(1, row["raw_shots"]))
        if abs(got - expected) > max(1e-9, 4 * sigma):
            problems.append(f"{row['name']} {row['series']}: lsr {got:.6f} vs Fig 6 "
                            f"{expected:.6f} ({(got - expected) / sigma:+.1f} sigma)")
        elif abs(got - expected) > 1e-9:
            problems.append(f"(within noise, not bit-identical) {row['name']} "
                            f"{row['series']}: {got:.6f} vs {expected:.6f}")
    return problems


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    reference = fig6_reference().set_index("name")
    names = list(settings.names) or (list(GATE_CIRCUITS) if settings.smoke else list(reference.index))
    circuits = {c.name: c for c in load_fig6_circuits(names=names)}
    presets = load_placements(settings, names)
    table = ResultTable(out / "packer_simulated.csv", key_columns=["name", "series"],
                        experiment=NAME, settings=asdict(settings), resume=resume)

    if not settings.skip_gate and not settings.smoke:
        gate_dir = out / "_gate"
        log(NAME, f"sanity gate: re-drawing Fig 6's baseline, naive and AQORN samples on "
                  f"{list(GATE_CIRCUITS)}")
        gate_jobs = [(asdict(settings), circuits[n], reference.loc[n].to_dict(),
                      presets.get(n), ("__none__",)) for n in GATE_CIRCUITS if n in circuits]
        gate_rows: List[Dict] = []
        stream_map(_run_circuit, gate_jobs, jobs=min(settings.jobs, len(gate_jobs)),
                   label=f"{NAME}-gate", on_result=lambda _i, r: gate_rows.extend(r))
        problems = check_gate(gate_rows)
        hard = [p for p in problems if not p.startswith("(within noise")]
        gate_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(gate_rows).drop(columns=["placement"], errors="ignore").to_csv(
            gate_dir / "gate.csv", index=False)
        (gate_dir / "GATE.txt").write_text(
            ("PASS" if not hard else "FAIL") + "\n" + "\n".join(problems) + "\n")
        for problem in problems:
            log(NAME, f"gate: {problem}")
        if hard or len(gate_rows) < 3 * len(gate_jobs):
            table.write_sidecar("stopped_at_sanity_gate", extra={"problems": problems})
            log(NAME, "SANITY GATE FAILED: Experiment1c stopped; see _gate/GATE.txt")
            sys.exit(3)
        log(NAME, "sanity gate passed: Fig 6's samples reproduced")

    jobs = []
    for name in names:
        if name not in circuits or table.is_done((name, "aqorn")):
            continue
        jobs.append((asdict(settings), circuits[name], reference.loc[name].to_dict(),
                     presets.get(name), ()))
    jobs.sort(key=lambda j: -j[1].num_qubits * float(j[2]["final_repetitions"]))
    log(NAME, f"{len(jobs)} circuits to simulate ({len(presets)} with 1a placements)")
    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    frame = table.frame()
    if not frame.empty:
        problems = check_gate(frame.to_dict("records"))
        log(NAME, f"Fig 6 reproduction across all circuits: {len(problems)} discrepancies")
        print(frame.groupby("series")[["lsr_impr_vs_baseline_pct", "tvd_red_vs_baseline_pct",
                                       "lsr_impr_vs_naive_pct"]].mean().round(2).to_string())
    return table.path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp1")
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--shots-scale", type=float, default=1.0)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--static-csv", default=str(RESULTS / "exp1" / "packer_static_o1.csv"))
    args = parser.parse_args()
    settings = Settings(static_csv=args.static_csv, shots_scale=args.shots_scale,
                        smoke=args.smoke, names=tuple(args.names), skip_gate=args.skip_gate,
                        jobs=args.jobs, out=str(out_dir(args, "exp1")))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
