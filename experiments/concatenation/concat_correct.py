"""
concat_correct.py -- error CORRECTION on the concatenated code: mid-circuit syndrome rounds on the
inner patches and a correcting decoder, against post-selection (2026-10-10).

`concat_standard.py` checks every patch once, at the end, and post-selects: concatenated error
detection. This adds the two things correction needs:

  * mid-circuit rounds (`TranslatorConfig.syndrome_rounds`): R checks of every bottom-level
    patch's two stabilizers, evenly spaced over the program's layers, fault-tolerant
    (`IcebergModule.syndrome_round`). Two levels: on the inner patches; one level: on the
    patches themselves (the iceberg scheme with periodic checks).
  * a correcting decoder (`lookup.Table`): exact first-order lookup over every single-fault
    mechanism; a shot is corrected when its syndrome is explained by single faults that agree
    on the correction (confidence >= threshold), kept as is when nothing fired, discarded
    otherwise. Corrections are Pauli-frame XORs on the decoded output (the simulated programs
    are Clifford, so no feedback is needed).

    arm          concat_standard's arms of the same name (outer plan_standard, transversal on
                 AQORN arms, logical optimisation on the program only)
    rounds       0 (checks at the end only), 1, 2, 4
    decoders     post-select; lookup at confidence 0.9 and 0.99; erasure (two levels, the old
                 decoder, for reference)

Noise, alpha calibration (concat_BB at 0 rounds keeps half its shots under post-selection: the
same alpha as concat_standard, so Figure A's operating point) and measurement (200k frames,
100k samples per point) as concat_standard.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_correct --jobs 12
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.concatenation.concat_eval import _counts, _decode, _flips, _tvd
from experiments.concatenation.concat_standard import FIG6_PLANS, _calibrate, _fig6_plans, \
    compile_arm, outer_plan

NAME = "concat_correct"
ARMS = ("single_base", "single_std", "concat_BB", "concat_AA")
ROUNDS = (0, 1, 2, 4)
FIXED_ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
THRESHOLDS = (0.9, 0.99)


@dataclass
class Settings:
    max_qubits: int = 8
    names: Optional[Sequence[str]] = None
    arms: Sequence[str] = ARMS
    rounds: Sequence[int] = ROUNDS
    alphas: Sequence[float] = FIXED_ALPHAS
    thresholds: Sequence[float] = THRESHOLDS
    target_lsr: float = 0.5
    frame_shots: int = 200_000
    sample_shots: int = 100_000
    calibration_shots: int = 20_000
    silent_weight: float = 10.0
    jobs: int = 12
    out: str = "results/concatenation"
    tag: str = ""


def _ler(keep: np.ndarray, outputs: np.ndarray) -> float:
    kept = int(keep.sum())
    return float((outputs.any(axis=1) & keep).sum() / max(1, kept))


def _measure(entry, noisy, *, frame_shots, sample_shots, seed, ideal, thresholds) -> Dict:
    kind, maps, table = entry["kind"], entry["maps"], entry["table"]
    bits = _flips(noisy, frame_shots, seed)
    samples = noisy.compile_sampler(seed=seed + 1).sample(sample_shots)
    out: Dict[str, float] = {}
    decoded = _decode(kind, maps, bits)
    decoded_samples = _decode(kind, maps, samples)
    modes = ("postselect", "erasure") if kind == "two" else ("postselect",)
    for mode in modes:
        out[f"{mode}_lsr"] = float(decoded[mode].mean())
        out[f"{mode}_silent"] = _ler(decoded[mode], decoded[f"{mode}_out"])
        out[f"{mode}_tvd"] = _tvd(decoded_samples[mode], decoded_samples[f"{mode}_out"], ideal)[0]
    for threshold in thresholds:
        label = f"lookup{int(round(threshold * 100))}"
        keep, corrected = table.apply(bits, maps, kind, threshold)
        out[f"{label}_lsr"] = float(keep.mean())
        out[f"{label}_silent"] = _ler(keep, corrected)
        keep_s, corrected_s = table.apply(samples, maps, kind, threshold)
        out[f"{label}_tvd"] = _tvd(keep_s, corrected_s, ideal)[0]
    return out


def _run_circuit(job) -> List[Dict]:
    import stim

    from experiments.concatenation import decode as D
    from experiments.concatenation.lookup import Table, mechanisms, single_fault_flips
    from experiments.concatenation.noise import rescale_text
    from experiments.revision.common import exact_ideal_distribution, fig6_noise_profile
    from experiments.tvd_explore.fastframes import at_alpha_one
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit
    from translator.logical_optimization import state

    settings_dict, circuit, fig6_plan, arms, alpha_star = job
    settings = Settings(**settings_dict)
    program = circuit.proxy
    noise = fig6_noise_profile()
    ideal = exact_ideal_distribution(program)
    plan = outer_plan(circuit)
    matches_fig6 = (None if fig6_plan is None else
                    ([list(p) for p in plan[0]], plan[1]) == (fig6_plan[0], fig6_plan[1]))
    configs = [(arm, rounds) for arm in arms for rounds in settings.rounds]
    if alpha_star is None and ("concat_BB", 0) not in configs:
        configs.insert(0, ("concat_BB", 0))          # needed to calibrate
    compiled = {}
    for arm, rounds in configs:
        began = time.time()
        kind, result, info = compile_arm(arm, program, plan, noise, settings.silent_weight,
                                         rounds=rounds)
        circuit_ = result.circuit
        maps = D.maps_for(result) if kind == "two" else D.single_maps_for(result)
        compile_seconds = time.time() - began
        began = time.time()
        base = str(at_alpha_one(circuit_))
        convert_seconds = time.time() - began
        began = time.time()
        mech = mechanisms(stim.Circuit(base))
        table = Table.build(mech, single_fault_flips(mech), maps, kind)
        del mech
        table_seconds = time.time() - began
        clean = sb.qiskit_to_stim(circuit_, layout=positions_for_circuit(
            circuit_, geometry=sb._geometry()), include_noise=False)
        noiseless = _decode(kind, maps, clean.compile_sampler(seed=3).sample(20_000))
        emitted = sum(1 for r in circuit_.cregs if r.name.startswith("m_parity_round"))
        patches = sum(1 for r in circuit_.cregs if r.name.startswith("m_data_"))
        compiled[(arm, rounds)] = dict(
            kind=kind, maps=maps, base=base, table=table,
            static=dict(_counts(circuit_), compile_seconds=compile_seconds,
                        convert_seconds=convert_seconds, table_seconds=table_seconds,
                        rounds_emitted=emitted // max(1, patches),
                        noiseless_lsr=float(noiseless["postselect"].mean()),
                        noiseless_tvd=_tvd(noiseless["postselect"], noiseless["postselect_out"],
                                           ideal)[0],
                        **{f"first_order_{k}": v for k, v in table.summary.items()},
                        **{k: v for k, v in info.items() if k in ("spread", "inner_plan")}))
    if alpha_star is None:
        reference = compiled[("concat_BB", 0)]
        alpha_star = _calibrate(reference["kind"], reference["maps"], reference["base"],
                                settings.target_lsr, settings.calibration_shots,
                                seed=circuit.circuit_idx * 101)
    configs = [(arm, rounds) for arm, rounds in configs if arm in arms]
    shared = {"circuit_name": circuit.name, "circuit_idx": circuit.circuit_idx,
              "num_qubits": program.num_qubits, "logical_gates": len(program.data),
              "logical_optimization": state(), "outer_plan_matches_fig6": matches_fig6,
              "alpha_star": alpha_star}
    rows: List[Dict] = []
    alphas = [("calibrated", alpha_star)] + [(f"{a:g}", a) for a in settings.alphas]
    for arm, rounds in configs:
        entry = compiled[(arm, rounds)]
        config_index = ARMS.index(arm) * 16 + rounds
        for alpha_index, (label, alpha) in enumerate(alphas):
            seed = 1_000_003 * (circuit.circuit_idx + 1) + 1009 * config_index + 17 * alpha_index
            measured = _measure(entry, rescale_text(entry["base"], alpha),
                                frame_shots=settings.frame_shots,
                                sample_shots=settings.sample_shots, seed=seed, ideal=ideal,
                                thresholds=settings.thresholds)
            rows.append({**shared, "arm": arm, "rounds": rounds, "alpha_label": label,
                         "alpha": alpha, **entry["static"], **measured})
    return rows


def _calibrated_alphas(path: str) -> Dict[str, float]:
    """alpha* per circuit from concat_standard's CSV: the same calibration (concat_BB at 0
    rounds, same seed and shots), so this run sits at Figure A's operating point."""
    import pandas as pd

    source = Path(path)
    if not source.exists():
        return {}
    frame = pd.read_csv(source)
    return frame.drop_duplicates("circuit_name").set_index("circuit_name")["alpha_star"].to_dict()


def run(settings: Settings, fig6_plans: str = FIG6_PLANS,
        alphas_csv: str = "results/concatenation/concat_standard.csv") -> Path:
    from experiments.revision.common import ResultTable, load_fig6_circuits, stream_map

    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    recorded = _fig6_plans(fig6_plans)
    circuits = [c for c in load_fig6_circuits()
                if c.size_group == "small" and c.proxy.num_qubits <= settings.max_qubits
                and (not settings.names or c.name in settings.names)]
    stem = NAME + (f"_{settings.tag}" if settings.tag else "")
    table = ResultTable(out / f"{stem}.csv",
                        key_columns=["circuit_name", "arm", "rounds", "alpha_label"],
                        experiment=NAME, settings=asdict(settings), resume=True)
    calibrated = _calibrated_alphas(alphas_csv) if settings.calibration_shots == 20_000 else {}
    last_round, last_alpha = settings.rounds[-1], f"{settings.alphas[-1]:g}"
    # one job per (circuit, arm): the large two-level circuits dominate, so this balances
    jobs = [(asdict(settings), c, recorded.get(c.name), (arm,), calibrated.get(c.name))
            for c in circuits for arm in settings.arms
            if not table.is_done((c.name, arm, last_round, last_alpha))]
    weight = {"concat_BB": 3.0, "concat_AA": 2.0}
    jobs.sort(key=lambda j: -len(j[1].proxy.data) * weight.get(j[3][0], 0.3))
    print(f"[{NAME}] alpha* from {alphas_csv} for {sum(1 for j in jobs if j[4] is not None)} "
          f"of {len(jobs)} jobs")
    print(f"[{NAME}] {len(circuits)} circuits, {len(jobs)} to run, arms {list(settings.arms)}, "
          f"rounds {list(settings.rounds)}, alphas {list(settings.alphas)}")

    def on_result(_index, rows) -> None:
        table.append(rows)

    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME, on_result=on_result)
    table.write_sidecar("complete" if counts["error"] == 0 else
                        f"complete_with_{counts['error']}_failures")
    print(f"[{NAME}] wrote {table.path}: {counts}")
    return table.path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--rounds", nargs="+", type=int, default=list(ROUNDS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(FIXED_ALPHAS))
    parser.add_argument("--frame-shots", type=int, default=200_000)
    parser.add_argument("--sample-shots", type=int, default=100_000)
    parser.add_argument("--calibration-shots", type=int, default=20_000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--out", default="results/concatenation")
    parser.add_argument("--tag", default="")
    parser.add_argument("--fig6-plans", default=FIG6_PLANS)
    args = parser.parse_args()
    settings = Settings(max_qubits=args.max_qubits, names=args.names, arms=tuple(args.arms),
                        rounds=tuple(args.rounds), alphas=tuple(args.alphas),
                        frame_shots=args.frame_shots, sample_shots=args.sample_shots,
                        calibration_shots=args.calibration_shots, jobs=args.jobs, out=args.out,
                        tag=args.tag)
    return settings, args.fig6_plans


if __name__ == "__main__":
    settings, plans = parse_args()
    run(settings, plans)
