"""
proxy_simulated.py -- Experiment3b: simulate the non-Clifford originals, not just proxies.

Question: with the rotation angles left in, does AQORN keep the TVD ranking seen on the
Clifford proxies? Small original circuits are checked with an exact or trajectory simulator.

Stim cannot run rz/rx, so the original and its proxy are both run on **Aer**, with the
same noise model, placement and passes; the rotations are the only difference. On Aer the
channels are: depolarizing composed with thermal relaxation on every gate (for the gate's
duration) and readout error, all scaled by alpha. There is **no** distance/transport
channel (Aer raises if asked) and **no** idle-period noise (`Simulator._simulate` never
reads `idle_noise` on the Aer path; idling between gates is not charged). So this is a
gate-noise-only machine, and the comparison is between original and proxy *on that
machine* -- not a reproduction of Fig 6's numbers, which include movement and idling.

Exact: Aer's density-matrix method (the opt-in `SimulatorConfig.aer_method`) on circuits of
4 logical qubits, which compile to 12 physical qubits at k = 2. 5-6 logical qubits need 18
physical qubits -- a 2^36-entry density matrix -- so those run as statevector trajectories
(`--trajectories`), at alpha = 1 only and a smaller shot count, because a trajectory costs
~36 s per 1,000 shots there.

Circuits: every 4-qubit QASMBench original with a rotation at a non-Clifford angle, plus 10
random 4-qubit Clifford+Rz/Rx programs with arbitrary angles (`random_program_circuit`,
`clifford_only=False`). Arms: Fig 6's baseline (naive, both passes off), naive with both
passes, and AQORN (planned on the original, as Fig 6 plans). Noise: alpha calibrated per
circuit on the naive *original* baseline to LSR ~0.06 (Fig 6's two-pilot rule), and alpha = 1.

TVD is reported twice: against the pipeline's sampled ideal reference, and against the
exact ideal distribution (`Statevector`), which removes the reference's sampling noise.
Cross-check: the proxy arms also run on Stim with distance and idle noise off, which should
agree with Aer on TVD (to ~0.003, per `experiments/shared/simulation.py`) though not on LSR.
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
from typing import Dict, List, Sequence

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import (
    RESULTS,
    NoiseSetup,
    ResultTable,
    add_common_arguments,
    exact_ideal_distribution,
    load_small_qasmbench,
    log,
    non_clifford_rotations,
    out_dir,
    percent_change,
    sim_config,
    simulate_arm,
    stream_map,
)

NAME = "exp3b"
SEED_BASE = 20261009
TRAJECTORY_NAMES = ("pea_n5", "qaoa_n6", "simon_n6", "qec_en_n5")


@dataclass
class Settings:
    random_circuits: int = 10
    random_seed: int = 20261010
    random_gates_per_qubit: int = 10
    random_immutable_ratio: float = 0.3
    dm_shots: int = 200_000
    pilot_shots: int = 50_000
    tiers: Sequence[str] = ("exact",)
    table: str = "proxy_simulated"
    trajectory_shots: int = 10_000
    target_selection_rate: float = 0.06
    stim_crosscheck: bool = True
    #: The standard compiler (`packing.standard`): AQORN planned with `plan_standard`, every
    #: arm layer-sorted and the merging arms' in-patch CZ / XCX transversal, as Fig 6 runs now.
    standard: bool = False
    smoke: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp3")


def _tvd(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 1.0
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _config(settings: Settings, noise, **kwargs):
    """`sim_config`, with the standard translator settings when `settings.standard`."""
    config = sim_config(noise, **kwargs)
    if settings.standard:
        from packing.standard import STANDARD

        config.translator_config.asap_order = STANDARD.asap_order
        config.translator_config.transversal_intra_two_qubit = (
            STANDARD.transversal_intra_two_qubit and kwargs["merging"])
    return config


def _calibrate(original, naive, aer_method, settings: Settings, seed: int) -> Dict[str, float]:
    """Fig 6's two-pilot rule, on Aer, on the naive original with both passes off."""
    from experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated import (  # noqa: E501
        BenchmarkSettings,
        calibrated_alpha_from_gate_count,
        corrected_alpha,
    )
    from experiments.revision.common import translate_metrics

    base = translate_metrics(original, naive, 4, commute=False, merging=False)
    gates = int(base["one_qubit_gates"] + base["two_qubit_gates"])
    fig6 = BenchmarkSettings(target_selection_rate=settings.target_selection_rate)
    alpha0 = calibrated_alpha_from_gate_count(gates, fig6)
    pilot = lambda alpha, s: simulate_arm(original, naive, _config(  # noqa: E731
        settings, NoiseSetup(alpha=alpha, stim=False, distance_noise=False, idle_noise=False,
                   aer_method=aer_method), n_code=4, commute=False, merging=False,
        repetitions=settings.pilot_shots, seed=s))["lsr"]
    lsr0 = pilot(alpha0, seed + 1)
    alpha1 = corrected_alpha(alpha0, lsr0, settings.target_selection_rate)
    lsr1 = pilot(alpha1, seed + 2)
    return {"alpha_initial": alpha0, "pilot_initial_lsr": lsr0, "alpha_final": alpha1,
            "pilot_final_lsr": lsr1, "baseline_gate_count": gates}


def _run_circuit(job) -> List[Dict]:
    from experiments.revision.common import fig6_noise_profile
    from experiments.shared.simulation import ensure_measurements
    from packing.plan import naive_packing, plan_packing

    settings_dict, label, source, original_raw, index, tier = job
    settings = Settings(**settings_dict)
    original = ensure_measurements(original_raw)
    proxy = ensure_measurements(to_proxy(original_raw))
    q = original.num_qubits
    naive = [tuple(p) for p in naive_packing(q, 2)]
    if settings.standard:
        from packing.standard import plan_standard

        plan = plan_standard(original)
    else:
        plan = plan_packing(original, 4, h_commutation="auto", verify=True,
                            noise=fig6_noise_profile())
    commute, aqorn = bool(plan.h_commutation), [tuple(p) for p in plan.placement]
    exact = {"original": exact_ideal_distribution(original_raw),
             "proxy": exact_ideal_distribution(to_proxy(original_raw))}
    aer_method = "density_matrix" if tier == "exact" else None
    shots = settings.dm_shots if tier == "exact" else settings.trajectory_shots
    if settings.smoke:
        shots = min(shots, 2000)
        settings.pilot_shots = 2000
    seed0 = SEED_BASE + 10_000 * index

    alphas = {}
    calibration = {}
    if tier == "exact":
        calibration = _calibrate(original, naive, aer_method, settings, seed0)
        alphas["calibrated"] = calibration["alpha_final"]
    alphas["alpha1"] = 1.0

    arms = [("baseline", naive, False, False), ("h_merge", naive, commute, True),
            ("aqorn", aqorn, commute, True)]
    rows: List[Dict] = []
    for a_index, (alpha_label, alpha) in enumerate(alphas.items()):
        for v_index, (version, program) in enumerate((("original", original), ("proxy", proxy))):
            backends = [("aer", NoiseSetup(alpha=alpha, stim=False, distance_noise=False,
                                           idle_noise=False, aer_method=aer_method))]
            if version == "proxy" and settings.stim_crosscheck:
                backends.append(("stim", NoiseSetup(alpha=alpha, stim=True,
                                                    distance_noise=False, idle_noise=False)))
            for b_index, (backend, noise) in enumerate(backends):
                for arm_index, (arm, placement, h, merge) in enumerate(arms):
                    seed = seed0 + 1000 * a_index + 100 * v_index + 10 * b_index + arm_index
                    began = time.time()
                    result = simulate_arm(program, placement, _config(
                        settings, noise, n_code=4, commute=h, merging=merge, repetitions=shots,
                        seed=seed))
                    rows.append({
                        "name": label, "source": source, "tier": tier, "num_qubits": q,
                        "non_clifford_rotations": non_clifford_rotations(original_raw),
                        "alpha_label": alpha_label, "alpha": alpha, "version": version,
                        "backend": backend, "arm": arm, "h_commutation": h, "merging": merge,
                        "commute": commute, "raw_shots": shots, "seed": seed,
                        "lsr": result["lsr"], "tvd": result["tvd"],
                        "tvd_exact": _tvd(result["noisy"], exact[version]),
                        "accepted_shots": int(round(result["lsr"] * shots)),
                        "one_qubit_gates": result["one_qubit_gates"],
                        "two_qubit_gates": result["two_qubit_gates"], "depth": result["depth"],
                        "physical_qubits": result["physical_qubits"],
                        "sim_seconds": time.time() - began,
                        "placement": json.dumps([list(p) for p in placement]),
                        **{f"cal_{k}": v for k, v in calibration.items()},
                    })
    return rows


def to_proxy(circuit):
    """rz/rx -> s, exactly as `create_proxy_from_qasm` does it; everything else kept."""
    import qiskit

    out = qiskit.QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        if inst.operation.name in {"rz", "rx"}:
            out.s(inst.qubits[0])
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def circuits(settings: Settings) -> List[tuple]:
    from experiments.shared.random_circuits import random_program_circuit

    jobs = []
    originals = load_small_qasmbench(proxy=False)
    index = 0
    for name, circuit in sorted(originals.items()):
        if "exact" in settings.tiers and circuit.num_qubits == 4 \
                and non_clifford_rotations(circuit) > 0:
            jobs.append((name, "qasmbench", circuit, index, "exact"))
            index += 1
    count = 0 if "exact" not in settings.tiers else (2 if settings.smoke else settings.random_circuits)
    for r in range(count):
        seed = settings.random_seed + r
        circuit = random_program_circuit(4, settings.random_gates_per_qubit, seed,
                                         immutable_ratio=settings.random_immutable_ratio,
                                         clifford_only=False)
        jobs.append((f"random_rz_{r}", "random", circuit, index, "exact"))
        index += 1
    if "trajectory" in settings.tiers:
        for name in TRAJECTORY_NAMES:
            if name in originals:
                jobs.append((name, "qasmbench", originals[name], index, "trajectory"))
                index += 1
    if settings.smoke:
        jobs = [j for j in jobs if j[0] in {"qft_n4", "random_rz_0", "qec_en_n5"}]
    return jobs


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / f"{settings.table}.csv", key_columns=["name", "alpha_label",
                                                                 "version", "backend", "arm"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    work = circuits(settings)
    jobs = [(asdict(settings), *w) for w in work
            if not table.is_done((w[0], "alpha1", "original", "aer", "aqorn"))]
    jobs.sort(key=lambda j: 0 if j[5] == "trajectory" else 1)
    log(NAME, f"{len(work)} circuits ({sum(1 for w in work if w[4] == 'exact')} exact, "
              f"{sum(1 for w in work if w[4] == 'trajectory')} trajectory); {len(jobs)} to run")
    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    report(table.frame())
    return table.path


def improvements(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (circuit, alpha, version, backend): AQORN against the baseline, both metrics."""
    rows = []
    for key, block in frame.groupby(["name", "alpha_label", "version", "backend"]):
        arms = block.set_index("arm")
        if not {"baseline", "aqorn"} <= set(arms.index):
            continue
        rows.append({
            "name": key[0], "alpha_label": key[1], "version": key[2], "backend": key[3],
            "lsr_impr_pct": percent_change(arms.loc["baseline", "lsr"], arms.loc["aqorn", "lsr"]),
            "tvd_red_pct": -percent_change(arms.loc["baseline", "tvd"], arms.loc["aqorn", "tvd"]),
            "tvd_exact_red_pct": -percent_change(arms.loc["baseline", "tvd_exact"],
                                                 arms.loc["aqorn", "tvd_exact"]),
            "lsr_impr_vs_hmerge_pct": percent_change(arms.loc["h_merge", "lsr"],
                                                     arms.loc["aqorn", "lsr"]),
            "tvd_exact_red_vs_hmerge_pct": -percent_change(arms.loc["h_merge", "tvd_exact"],
                                                           arms.loc["aqorn", "tvd_exact"]),
        })
    return pd.DataFrame(rows)


def report(frame: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    if frame.empty:
        return
    imp = improvements(frame)
    aer = imp[imp["backend"] == "aer"]
    for alpha_label, block in aer.groupby("alpha_label"):
        wide = block.pivot_table(index="name", columns="version",
                                 values=["lsr_impr_pct", "tvd_exact_red_pct"])
        if "original" not in wide["lsr_impr_pct"] or "proxy" not in wide["lsr_impr_pct"]:
            continue
        for metric in ("lsr_impr_pct", "tvd_exact_red_pct"):
            o, p = wide[metric]["original"], wide[metric]["proxy"]
            agree = (np.sign(o) == np.sign(p)).mean() * 100
            rho = spearmanr(o, p).statistic if len(o) > 2 else math.nan
            print(f"[{NAME}] {alpha_label} {metric}: original mean {o.mean():.2f}%, proxy "
                  f"mean {p.mean():.2f}%, sign agreement {agree:.0f}%, Spearman {rho:.2f} "
                  f"({len(o)} circuits)")
    stim = frame[(frame["backend"] == "stim")].set_index(["name", "alpha_label", "arm"])["tvd"]
    aer_proxy = frame[(frame["backend"] == "aer") & (frame["version"] == "proxy")].set_index(
        ["name", "alpha_label", "arm"])["tvd"]
    both = pd.concat([stim.rename("stim"), aer_proxy.rename("aer")], axis=1).dropna()
    if not both.empty:
        print(f"[{NAME}] Stim vs Aer proxy TVD: mean |diff| "
              f"{(both['stim'] - both['aer']).abs().mean():.4f}, max "
              f"{(both['stim'] - both['aer']).abs().max():.4f} over {len(both)} points")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp3")
    parser.add_argument("--tiers", nargs="+", default=["exact"], choices=["exact", "trajectory"])
    parser.add_argument("--table", default="proxy_simulated")
    parser.add_argument("--trajectory-shots", type=int, default=10_000)
    parser.add_argument("--standard", action="store_true",
                        help="the standard compiler (Settings.standard); default table "
                             "proxy_simulated_standard")
    args = parser.parse_args()
    table = args.table if args.table != "proxy_simulated" or not args.standard \
        else "proxy_simulated_standard"
    return Settings(tiers=tuple(args.tiers), table=table, standard=args.standard,
                    trajectory_shots=args.trajectory_shots, smoke=args.smoke, jobs=args.jobs,
                    out=str(out_dir(args, "exp3"))), args.resume


if __name__ == "__main__":
    run(*parse_args())
