"""
concat_standard.py -- the concatenation experiment ([[4,2,2]] inside [[4,2,2]]) under the current
standard: logical optimisation first, `plan_standard` (solo-HH aware), transversal in-patch CZ /
XCX on AQORN's arms, every program layer-sorted (2026-10-10).

`concat_eval.py` (2026-10-07) ran eleven arms on Fig 6's recorded weight-10 / w10T plans, before
the logical optimisation pass and the solo-HH planner fix. This reruns the arms the simplified
figure needs (`concat_figures.py`: AQORN's gain at one level and at two; the shots / accuracy
plane) plus the ones its text cites (outer only, inner only, index order):

    arm              outer level                        inner level
    single_base      tight trivial, passes off          --
    single_std       plan_standard, passes on, T        --
    concat_BB        tight trivial, passes off          stride layout, passes off
    concat_AB        plan_standard, passes on, T        stride layout, passes off
    concat_BA        tight trivial, passes off          inner AQORN (spread search + passes), T
    concat_AA        plan_standard, passes on, T        inner AQORN, T
    concat_AA_index  plan_standard, passes on, T        index layout (NOT spread), passes, T

    T = in-patch CZ / XCX built transversally (S, SX on all four data qubits), as Fig 6 under the
    standard credits it to AQORN's arms only. Inner AQORN is `inner_plan.plan_inner` with the
    transversal rate table.

Logical optimisation. The loader optimises each QASMBench program before its Clifford proxy is
made (the pass is on by default), so every arm, baselines included, compiles the optimised
program, and the outer plan is `plan_standard` on the optimised original -- what Fig 6 under the
pass plans (`ablation_standard_logical.csv`; each row records whether the plan equals Fig 6's).
The INNER level never runs the pass: its program C1 is compiled output -- the outer code's
encoder, gadgets and syndrome extraction -- and a unitary-preserving rewrite of it would change
which faults the outer checks catch. `plan_inner`'s cost model and verify translations sit in
the same scope.

Noise and measurement as in `concat_eval.py` (Fig 6's machine, Stim conversion once at alpha 1,
exact rescaling; Pauli frames for LSR and logical error per kept shot; real samples for TVD), at
each circuit's calibrated alpha (concat_BB keeps half its shots) and at fixed alpha 0.003 ... 1
(p2 = 3e-3 alpha). The grid now reaches the calibrated regime (median alpha 0.3 before).

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_standard --jobs 12
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from experiments.concatenation.concat_eval import _counts, _decode, _flips, _measure, _tvd

NAME = "concat_standard"
ARMS = ("single_base", "single_std", "concat_BB", "concat_AB", "concat_BA", "concat_AA",
        "concat_AA_index")
FIXED_ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
FIG6_PLANS = "../../../../results/data/ablation_standard_logical.csv"


@dataclass
class Settings:
    max_qubits: int = 8
    names: Optional[Sequence[str]] = None
    arms: Sequence[str] = ARMS
    alphas: Sequence[float] = FIXED_ALPHAS
    target_lsr: float = 0.5
    frame_shots: int = 200_000
    sample_shots: int = 100_000
    calibration_shots: int = 20_000
    silent_weight: float = 10.0
    jobs: int = 12
    out: str = "results/concatenation"
    tag: str = ""


def outer_plan(circuit) -> Tuple[List[Tuple[int, int]], bool]:
    """`plan_standard` on the (optimised) original, as Fig 6 under the standard plans it."""
    from experiments.shared.simulation import ensure_measurements
    from packing.standard import plan_standard

    plan = plan_standard(ensure_measurements(circuit.original))
    return [tuple(int(v) for v in p) for p in plan.placement], bool(plan.h_commutation)


def compile_arm(arm: str, program, plan, noise, weight: float, rounds: int = 0):
    """
    (kind, compiled, info): kind 'single' (a TranslationResult) or 'two' (a TwoLevel).
    `rounds`: mid-circuit syndrome rounds on the bottom level's patches -- the one level of a
    single arm, the inner level of a two-level arm (`concat_correct.py`); 0 = checks at the end.
    """
    from experiments.concatenation import construct as C
    from experiments.concatenation.inner_plan import plan_inner
    from iceberg.gadget_options import transversal_intra
    from packing.silent import load_silent_rates
    from translator.logical_optimization import disabled

    n = program.num_qubits
    tight = [(i // 2, i % 2) for i in range(n)]
    outer_key = "B" if arm in ("single_base", "concat_BB", "concat_BA") else "A"
    placement, commute, merging, transversal = (
        (tight, False, False, False) if outer_key == "B" else (plan[0], plan[1], True, True))
    single = arm.startswith("single_")
    outer = C.outer_translate(program, placement, commute=commute, merging=merging,
                              transversal=transversal, rounds=rounds if single else 0)
    info: Dict[str, object] = {"outer": outer_key, "outer_commute": commute}
    if single:
        return "single", outer, info

    flat = C.flatten(outer.circuit)
    layouts = C.inner_layouts(flat)
    inner_aqorn = arm in ("concat_BA", "concat_AA", "concat_AA_index")
    with disabled(), transversal_intra(inner_aqorn):
        if arm == "concat_AA_index":
            inner_placement, inner_commute, inner_merging = layouts["index"], True, True
        elif not inner_aqorn:
            inner_placement, inner_commute, inner_merging = layouts["stride"], False, False
        else:
            rates = load_silent_rates(4, transversal_intra_two_qubit=True)
            chosen = plan_inner(flat, rates=rates, noise=noise, weight=weight)
            inner_placement, inner_commute, inner_merging = (
                chosen["placement"], chosen["commute"], True)
            info.update(inner_plan=chosen["label"], inner_plan_score=chosen["score"])
        inner = C.inner_translate(flat, inner_placement, commute=inner_commute,
                                  merging=inner_merging, transversal=inner_aqorn, rounds=rounds)
    info.update(inner_commute=inner_commute, inner_merging=inner_merging,
                inner_transversal=inner_aqorn,
                spread=C.is_spread(inner_placement, flat.roles),
                inner_placement=json.dumps([list(p) for p in inner_placement]),
                outer_physical_qubits=outer.circuit.num_qubits,
                outer_two_qubit_gates=_counts(outer.circuit)["two_qubit_gates"])
    return "two", C.TwoLevel(outer=outer, flat=flat, inner_placement=inner_placement,
                             inner=inner), info


def _calibrate(kind, maps, base, target: float, shots: int, seed: int) -> float:
    """Alpha at which this arm keeps `target` of its shots (bisection on log alpha)."""
    from experiments.concatenation.noise import rescale_text

    low, high = math.log(1e-5), math.log(3.0)
    for _ in range(16):
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
    from experiments.concatenation.noise import rescale_text
    from experiments.revision.common import exact_ideal_distribution, fig6_noise_profile
    from experiments.tvd_explore.fastframes import at_alpha_one
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit
    from translator.logical_optimization import state

    settings_dict, circuit, fig6_plan = job
    settings = Settings(**settings_dict)
    program = circuit.proxy
    noise = fig6_noise_profile()
    ideal = exact_ideal_distribution(program)
    began = time.time()
    plan = outer_plan(circuit)
    plan_seconds = time.time() - began
    matches_fig6 = (None if fig6_plan is None else
                    ([list(p) for p in plan[0]], plan[1]) == (fig6_plan[0], fig6_plan[1]))
    compiled = {}
    converted: Dict[Tuple, Tuple[str, float]] = {}   # identical circuits convert once
    for arm in settings.arms:
        began = time.time()
        kind, result, info = compile_arm(arm, program, plan, noise, settings.silent_weight)
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
    shared = {"circuit_name": circuit.name, "circuit_idx": circuit.circuit_idx,
              "num_qubits": program.num_qubits, "logical_gates": len(program.data),
              "logical_optimization": state(), "outer_placement": json.dumps(
                  [list(p) for p in plan[0]]), "outer_plan_commute": plan[1],
              "outer_plan_matches_fig6": matches_fig6, "plan_seconds": plan_seconds,
              "alpha_star": alpha_star}
    rows: List[Dict] = []
    alphas = [("calibrated", alpha_star)] + [(f"{a:g}", a) for a in settings.alphas]
    for arm_index, arm in enumerate(settings.arms):
        entry = compiled[arm]
        decoders = ("postselect", "erasure") if entry["kind"] == "two" else ("postselect",)
        for alpha_index, (label, alpha) in enumerate(alphas):
            seed = 1_000_003 * (circuit.circuit_idx + 1) + 1009 * arm_index + 17 * alpha_index
            measured = _measure(entry["kind"], entry["maps"], rescale_text(entry["base"], alpha),
                                frame_shots=settings.frame_shots,
                                sample_shots=settings.sample_shots, seed=seed, ideal=ideal,
                                decoders=decoders)
            rows.append({**shared, "arm": arm, "alpha_label": label, "alpha": alpha,
                         **entry["static"], **entry["info"], **measured})
    return rows


def _fig6_plans(path: str) -> Dict[str, Tuple[list, bool]]:
    source = Path(path)
    if not source.exists():
        print(f"[{NAME}] no Fig 6 plans at {source}: outer plans are not cross-checked")
        return {}
    frame = pd.read_csv(source).drop_duplicates("circuit_name")
    return {r.circuit_name: (json.loads(r.placement), bool(r.commute)) for r in frame.itertuples()}


def run(settings: Settings, fig6_plans: str = FIG6_PLANS) -> Path:
    from experiments.revision.common import ResultTable, load_fig6_circuits, stream_map

    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    recorded = _fig6_plans(fig6_plans)
    circuits = [c for c in load_fig6_circuits()
                if c.size_group == "small" and c.proxy.num_qubits <= settings.max_qubits
                and (not settings.names or c.name in settings.names)]
    stem = NAME + (f"_{settings.tag}" if settings.tag else "")
    last = f"{settings.alphas[-1]:g}"
    table = ResultTable(out / f"{stem}.csv", key_columns=["circuit_name", "arm", "alpha_label"],
                        experiment=NAME, settings=asdict(settings), resume=True)
    jobs = [(asdict(settings), c, recorded.get(c.name)) for c in circuits
            if not table.is_done((c.name, settings.arms[-1], last))]
    jobs.sort(key=lambda j: -len(j[1].proxy.data))   # largest first: shortest makespan
    print(f"[{NAME}] {len(circuits)} circuits, {len(jobs)} to run, arms {list(settings.arms)}, "
          f"alphas {list(settings.alphas)}")

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
                        alphas=tuple(args.alphas), frame_shots=args.frame_shots,
                        sample_shots=args.sample_shots, calibration_shots=args.calibration_shots,
                        jobs=args.jobs, out=args.out, tag=args.tag)
    return settings, args.fig6_plans


if __name__ == "__main__":
    settings, plans = parse_args()
    run(settings, plans)
