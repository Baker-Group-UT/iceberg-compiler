"""
concat_eval.py -- AQORN at the outer and inner levels of a concatenated [[4,2,2]] x [[4,2,2]] code.

Question: does AQORN carry over to correcting or concatenated codes? Every Fig 6 circuit with at most 8 qubits (the small group, 24
circuits), as its Clifford proxy -- what Fig 6 simulates -- compiled eleven ways:

    single_base     one level, Fig 6's baseline (naive packing, both passes off)
    single_w10      one level, the weight-10 planner's plan (placement + H-commutation), merging on
    single_w10T     one level, w10T (+ transversal in-patch CZ/XCX, its switch on)
    concat_BB       outer baseline, inner baseline: stride layout, passes off
    concat_Bp       outer baseline, inner stride layout with AQORN's passes (H commutation + merging)
    concat_BA       outer baseline, inner AQORN: passes + placement search over spread layouts
    concat_AB       outer w10, inner baseline
    concat_Ap       outer w10, inner passes only
    concat_AA       outer w10, inner AQORN
    concat_AA_T     outer w10T, inner AQORN, switch on at both levels
    concat_AA_index outer w10, inner index layout (NOT spread) with passes -- the cautionary arm

Every translation layer-sorts its program first (`asap_order`, the fair baseline).

Noise: Fig 6's machine (ScaledNoiseModel, distance + idle noise at scalar 1), converted to Stim
once per arm at alpha = 1 and rescaled exactly (`tvd_explore.fastframes`). Per circuit, alpha is
calibrated so concat_BB keeps half its shots under post-selection (Fig 6 calibrates its baseline
the same way, to 0.06); every arm is also run at fixed alpha 0.003, 0.01, 0.03, 0.1.

Measured per (circuit, arm, alpha), for both decoders of `decode.py` (post-selection at both
levels; inner flags as erasures corrected by the outer code):
    lsr        kept fraction                       (Pauli frames, `frame_shots`)
    silent     P(logical error | kept)             (frames: kept and some output flipped)
    tvd        against the exact ideal distribution (real samples, `sample_shots`)
Checked per arm: noiseless, every check passes on every shot and the decoded distribution is the
ideal one up to sampling (`noiseless_lsr`, `noiseless_tvd`).

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_eval --jobs 3
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
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

NAME = "concat_eval"
W10_CSV = "experiments/figures/ablation_silent_aware/ablation_silent_aware_w10.csv"
W10T_CSV = "results/tvd_explore/ablation_variant_w10T.csv"
RATES = "results/revision/exp8/gadget_rates.csv"
RATES_T = "results/tvd_explore/gadget_rates_transversal.csv"
SWITCH = "TRANSVERSAL_INTRA_TWO_QUBIT"
ARMS = ("single_base", "single_w10", "single_w10T", "concat_BB", "concat_Bp", "concat_BA",
        "concat_AB", "concat_Ap", "concat_AA", "concat_AA_T", "concat_AA_index")
FIXED_ALPHAS = (0.003, 0.01, 0.03, 0.1)


@dataclass
class Settings:
    max_qubits: int = 8
    names: Optional[Sequence[str]] = None
    arms: Sequence[str] = ARMS
    target_lsr: float = 0.5
    frame_shots: int = 200_000
    sample_shots: int = 100_000
    calibration_shots: int = 20_000
    silent_weight: float = 10.0
    jobs: int = 3
    out: str = "results/concatenation"
    tag: str = ""


@contextmanager
def switched(on: bool):
    from packing import passes
    from translator import explore_switches

    explore_switches.configure(**{SWITCH: on})
    passes.clear_cache()
    try:
        yield
    finally:
        explore_switches.configure(**{SWITCH: False})
        passes.clear_cache()


def _plans() -> Dict[str, Dict[str, Tuple[list, bool]]]:
    out: Dict[str, Dict[str, Tuple[list, bool]]] = {}
    for label, path in (("w10", W10_CSV), ("w10T", W10T_CSV)):
        frame = pd.read_csv(path).drop_duplicates("circuit_name")
        for row in frame.itertuples():
            out.setdefault(row.circuit_name, {})[label] = (
                [tuple(p) for p in json.loads(row.placement)], bool(row.commute))
    return out


def _counts(circuit) -> Dict[str, float]:
    from analysis.gate_count_stats import one_two_qubit_gate_totals

    one, two = one_two_qubit_gate_totals(circuit)
    swaps = sum(1 for i in circuit.data if i.operation.name == "swap")
    return {"one_qubit_gates": float(one), "two_qubit_gates": float(two), "swaps": float(swaps),
            "depth": float(circuit.depth()), "physical_qubits": float(circuit.num_qubits)}


def compile_arm(arm: str, program, plans, rates, noise, weight: float):
    """(kind, compiled, info) where kind is 'single' (a TranslationResult) or 'two' (TwoLevel)."""
    from experiments.concatenation import construct as C
    from experiments.concatenation.inner_plan import plan_inner

    n = program.num_qubits
    base = ([(i // 2, i % 2) for i in range(n)], False, False)
    outer_of = {"B": base, "A": (*plans["w10"], True), "T": (*plans["w10T"], True)}
    transversal = arm in ("single_w10T", "concat_AA_T")
    info: Dict[str, object] = {}
    with switched(transversal):
        if arm.startswith("single_"):
            key = {"single_base": "B", "single_w10": "A", "single_w10T": "T"}[arm]
            placement, commute, merging = outer_of[key]
            return "single", C.outer_translate(program, placement, commute=commute,
                                               merging=merging), info
        code = arm.split("_", 1)[1]
        outer_key = "T" if arm == "concat_AA_T" else code[0]
        placement, commute, merging = outer_of[outer_key]
        outer = C.outer_translate(program, placement, commute=commute, merging=merging)
        flat = C.flatten(outer.circuit)
        layouts = C.inner_layouts(flat)
        if arm == "concat_AA_index":
            inner_placement, inner_commute, inner_merging = layouts["index"], True, True
        elif code[1] == "B":
            inner_placement, inner_commute, inner_merging = layouts["stride"], False, False
        elif code[1] == "p":
            inner_placement, inner_commute, inner_merging = layouts["stride"], True, True
        else:
            plan = plan_inner(flat, rates=rates["transversal" if transversal else "shipped"],
                              noise=noise, weight=weight)
            inner_placement, inner_commute, inner_merging = plan["placement"], plan["commute"], True
            info.update(inner_plan=plan["label"], inner_plan_score=plan["score"])
        inner = C.inner_translate(flat, inner_placement, commute=inner_commute,
                                  merging=inner_merging)
        info.update(inner_commute=inner_commute, inner_merging=inner_merging,
                    spread=C.is_spread(inner_placement, flat.roles),
                    inner_placement=json.dumps([list(p) for p in inner_placement]),
                    outer_physical_qubits=outer.circuit.num_qubits,
                    outer_two_qubit_gates=_counts(outer.circuit)["two_qubit_gates"])
        two = C.TwoLevel(outer=outer, flat=flat, inner_placement=inner_placement, inner=inner)
        return "two", two, info


def _flips(noisy, shots: int, seed: int) -> np.ndarray:
    import stim

    simulator = stim.FlipSimulator(batch_size=shots, disable_stabilizer_randomization=True,
                                   num_qubits=noisy.num_qubits, seed=seed)
    simulator.do(noisy)
    return simulator.get_measurement_flips(bit_packed=False).T


def _decode(kind, maps, bits):
    from experiments.concatenation import decode as D

    return D.decode(bits, maps) if kind == "two" else D.decode_single(bits, maps)


def _tvd(keep: np.ndarray, outputs: np.ndarray, ideal: Dict[str, float]) -> Tuple[float, int]:
    if keep.sum() == 0:
        return math.nan, 0
    weights = 1 << np.arange(outputs.shape[1])          # character i = program qubit i
    codes = outputs[keep].astype(np.int64) @ weights
    counts = Counter(codes.tolist())
    total = int(keep.sum())
    width = outputs.shape[1]
    ideal_codes = {int(k[::-1], 2): p for k, p in ideal.items()}  # key char i -> bit i
    keys = set(counts) | set(ideal_codes)
    return 0.5 * sum(abs(counts.get(c, 0) / total - ideal_codes.get(c, 0.0)) for c in keys), total


def _measure(kind, maps, noisy, *, frame_shots, sample_shots, seed, ideal, decoders):
    flips = _decode(kind, maps, _flips(noisy, frame_shots, seed))
    samples = _decode(kind, maps, noisy.compile_sampler(seed=seed + 1).sample(sample_shots))
    out = {}
    for mode in decoders:
        keep = flips[mode]
        kept = int(keep.sum())
        out[f"{mode}_lsr"] = float(keep.mean())
        out[f"{mode}_silent"] = float((flips[f"{mode}_out"].any(axis=1) & keep).sum() / max(1, kept))
        tvd, accepted = _tvd(samples[mode], samples[f"{mode}_out"], ideal)
        out[f"{mode}_tvd"] = tvd
        out[f"{mode}_tvd_accepted"] = accepted
        out[f"{mode}_sample_lsr"] = float(samples[mode].mean())
    return out


def _calibrate(kind, maps, base, target: float, shots: int, seed: int) -> float:
    from experiments.concatenation.noise import rescale_text

    low, high = math.log(1e-5), math.log(1.0)
    for _ in range(14):
        mid = 0.5 * (low + high)
        lsr = float(_decode(kind, maps, _flips(rescale_text(base, math.exp(mid)), shots, seed))
                    ["postselect"].mean())
        if lsr > target:
            low = mid
        else:
            high = mid
    return math.exp(0.5 * (low + high))


def _run_circuit(job) -> List[Dict]:
    from experiments.concatenation import decode as D
    from experiments.revision.common import exact_ideal_distribution, fig6_noise_profile
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from experiments.concatenation.noise import rescale_text
    from experiments.tvd_explore.fastframes import at_alpha_one
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    settings_dict, circuit, plans = job
    settings = Settings(**settings_dict)
    program = circuit.proxy
    noise = fig6_noise_profile()
    rates = {"shipped": UndetectedRates.from_table(pd.read_csv(RATES), 4, 3e-3),
             "transversal": UndetectedRates.from_table(pd.read_csv(RATES_T), 4, 3e-3)}
    ideal = exact_ideal_distribution(program)
    compiled = {}
    converted: Dict[Tuple, Tuple[str, float]] = {}   # identical circuits convert once
    for arm in settings.arms:
        began = time.time()
        kind, result, info = compile_arm(arm, program, plans, rates, noise, settings.silent_weight)
        circuit_ = result.circuit
        maps = D.maps_for(result) if kind == "two" else D.single_maps_for(result)
        compile_seconds = time.time() - began
        began = time.time()
        fingerprint = tuple((i.operation.name, tuple(circuit_.find_bit(q).index for q in i.qubits))
                            for i in circuit_.data)
        if fingerprint not in converted:
            converted[fingerprint] = (str(at_alpha_one(circuit_)), time.time() - began)
        base, convert_seconds = converted[fingerprint]
        clean = sb.qiskit_to_stim(circuit_, layout=positions_for_circuit(
            circuit_, geometry=sb._geometry()), include_noise=False)
        noiseless = _decode(kind, maps, clean.compile_sampler(seed=3).sample(20_000))
        tvd0, _ = _tvd(noiseless["postselect"], noiseless["postselect_out"], ideal)
        compiled[arm] = dict(kind=kind, maps=maps, base=base, info=info,
                             static=dict(_counts(circuit_), compile_seconds=compile_seconds,
                                         convert_seconds=convert_seconds,
                                         noiseless_lsr=float(noiseless["postselect"].mean()),
                                         noiseless_tvd=tvd0))
    reference = compiled["concat_BB"]
    alpha_star = _calibrate(reference["kind"], reference["maps"], reference["base"],
                            settings.target_lsr, settings.calibration_shots,
                            seed=circuit.circuit_idx * 101)
    rows: List[Dict] = []
    alphas = [("calibrated", alpha_star)] + [(f"{a:g}", a) for a in FIXED_ALPHAS]
    for arm_index, arm in enumerate(settings.arms):
        entry = compiled[arm]
        decoders = ("postselect", "erasure") if entry["kind"] == "two" else ("postselect",)
        for alpha_index, (label, alpha) in enumerate(alphas):
            seed = 1_000_003 * (circuit.circuit_idx + 1) + 1009 * arm_index + 17 * alpha_index
            measured = _measure(entry["kind"], entry["maps"], rescale_text(entry["base"], alpha),
                                frame_shots=settings.frame_shots,
                                sample_shots=settings.sample_shots, seed=seed, ideal=ideal,
                                decoders=decoders)
            rows.append({"circuit_name": circuit.name, "circuit_idx": circuit.circuit_idx,
                         "num_qubits": program.num_qubits, "logical_gates": len(program.data),
                         "arm": arm, "alpha_label": label, "alpha": alpha,
                         "alpha_star": alpha_star, **entry["static"], **entry["info"],
                         **measured})
    return rows


def run(settings: Settings) -> Path:
    from experiments.revision.common import ResultTable, load_fig6_circuits, stream_map

    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = _plans()
    circuits = [c for c in load_fig6_circuits(names=settings.names)
                if c.size_group == "small" and c.proxy.num_qubits <= settings.max_qubits]
    stem = NAME + (f"_{settings.tag}" if settings.tag else "")
    table = ResultTable(out / f"{stem}.csv", key_columns=["circuit_name", "arm", "alpha_label"],
                        experiment=NAME, settings=asdict(settings), resume=True)
    jobs = [(asdict(settings), c, plans[c.name]) for c in circuits
            if not table.is_done((c.name, settings.arms[-1], "0.1"))]
    jobs.sort(key=lambda j: len(j[1].proxy.data))   # smallest first: results accumulate
    print(f"[{NAME}] {len(circuits)} circuits, {len(jobs)} to run, arms {list(settings.arms)}")

    def on_result(_index, rows) -> None:
        table.append(rows)

    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME, on_result=on_result)
    table.write_sidecar("complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures")
    print(f"[{NAME}] wrote {table.path}: {counts}")
    return table.path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--frame-shots", type=int, default=200_000)
    parser.add_argument("--sample-shots", type=int, default=100_000)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out", default="results/concatenation")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    return Settings(max_qubits=args.max_qubits, names=args.names, arms=tuple(args.arms),
                    frame_shots=args.frame_shots, sample_shots=args.sample_shots,
                    jobs=args.jobs, out=args.out, tag=args.tag)


if __name__ == "__main__":
    run(parse_args())
