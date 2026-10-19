"""
code_size_undo_sim.py -- does macro + undo merging buy shots and accuracy at k >= 4?

`code_size_undo` counts gates. This measures, for the Fig 6 circuits at each k, the
naive and the AQORN-planned placement translated with merging on, with the undo switches
off and on: fraction of shots kept (LSR) and silent logical error per kept shot (the TVD
stand-in, `experiments/revision/exp8_undetected/frames.py`), by frame simulation at the
circuit's Fig 6 alpha and at alpha = 1, from one conversion (`fastframes`). The placement
is planned once, with the switches off, the way Fig 6 plans (noise profile on, commutation
decided per circuit), so the two arms differ in translation only.

Up to `--noiseless-max` qubits each changed arm is also simulated noiselessly through the
pipeline, which catches a miscompile frame simulation cannot see.

    PYTHONPATH=. python -m experiments.tvd_explore.code_size_undo_sim --k 4 6 8 10
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")
UNDO = ("HH_MAJORITY_UNDO", "TRANSVERSAL_MAJORITY_UNDO")


#: Set per worker from the job: layer-sort every arm (`LAYER_ORDER_ALL_ARMS`), so the
#: comparison isolates the undo macros from the re-sort a newly firing merge brings.
_LAYER = {"on": False}


def _set(on: bool):
    from translator import explore_switches

    explore_switches.configure(**{name: on for name in UNDO}, LAYER_ORDER_ALL_ARMS=_LAYER["on"])


def plan(job):
    """AQORN's plan at code size k, switches off, as Fig 6 plans."""
    from experiments.figures.merging_improvement.common import pack_program
    from experiments.revision.common import fig6_noise_profile

    circuit, k = job
    _set(False)
    began = time.time()
    placement, commute = pack_program(circuit.original, k + 2, noise=fig6_noise_profile())
    return {"name": circuit.name, "k": k, "placement": json.dumps([list(map(int, p)) for p in placement]),
            "commute": bool(commute), "plan_seconds": round(time.time() - began, 1)}


def _noiseless(program, placement, n_code, commute, seed=5):
    from experiments.revision.common import NoiseSetup, sim_config, simulate_arm

    config = sim_config(NoiseSetup(alpha=0.0), n_code=n_code, commute=commute, merging=True,
                        repetitions=4000, seed=seed)
    return float(simulate_arm(program, placement, config)["tvd"])


def point(job):
    from experiments.shared.simulation import ensure_measurements
    from experiments.tvd_explore.fastframes import splits
    from experiments.tvd_explore.screen import _translate

    circuit, k, placement, commute, alpha6, shots, noiseless_max, layer = job
    _LAYER["on"] = bool(layer)
    proxy = ensure_measurements(circuit.proxy)
    naive = [(i // k, i % k) for i in range(proxy.num_qubits)]
    rows = []
    for label, where, use_commute in (("naive", naive, True), ("aqorn", placement, commute)):
        encoded = {}
        for arm, on in (("shipped", False), ("undo", True)):
            _set(on)
            encoded[arm] = _translate(proxy, where, commute=use_commute, merging=True,
                                      overrides={"n_code": k + 2})
        same = encoded["shipped"] == encoded["undo"]
        for arm, on in (("shipped", False), ("undo", True)):
            _set(on)
            circ = encoded[arm]
            began = time.time()
            # An unchanged translation is the same circuit: nothing to measure twice, and
            # nothing to measure at all for the comparison, so only gate counts are kept.
            measured = (splits(circ, (alpha6, 1.0), shots=shots, seed=17) if not same
                        else {a: {"p_detected": float("nan"), "p_silent_given_kept": float("nan")}
                              for a in (alpha6, 1.0)})
            row = {
                "name": circuit.name, "num_qubits": proxy.num_qubits, "k": k, "placement_kind": label,
                "arm": arm, "unchanged": bool(same),
                "two_qubit_gates": sum(1 for i in circ.data if len(i.qubits) == 2
                                       and i.operation.name not in ("measure", "barrier")),
                "one_qubit_gates": sum(1 for i in circ.data if len(i.qubits) == 1
                                       and i.operation.name not in ("measure", "barrier", "reset")),
                "depth": circ.depth(),
                "alpha6": alpha6,
                "kept_fig6": 1 - measured[alpha6]["p_detected"],
                "silent_fig6": measured[alpha6]["p_silent_given_kept"],
                "kept_a1": 1 - measured[1.0]["p_detected"],
                "silent_a1": measured[1.0]["p_silent_given_kept"],
                "frame_seconds": round(time.time() - began, 1),
            }
            if arm == "undo" and not same and proxy.num_qubits <= noiseless_max:
                row["noiseless_tvd"] = _noiseless(proxy, where, k + 2, use_commute)
                _set(False)
                row["noiseless_tvd_shipped"] = _noiseless(proxy, where, k + 2, use_commute)
            rows.append(row)
    return rows


def main():
    from experiments.revision.common import fig6_reference, load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--shots", type=int, default=40_000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--noiseless-max", type=int, default=20)
    parser.add_argument("--tag", default="code_size_undo_sim")
    parser.add_argument("--layer-order", action="store_true",
                        help="Layer-sort every arm (LAYER_ORDER_ALL_ARMS) in both arms.")
    parser.add_argument("--plans-from", default=None,
                        help="Reuse another run's plan table (results/tvd_explore/<tag>_plans.csv).")
    args = parser.parse_args()

    ref = fig6_reference().set_index("name")
    circuits = load_fig6_circuits(names=args.names)
    plans_path = OUT / f"{args.plans_from or args.tag}_plans.csv"
    if plans_path.exists():
        plans = pd.read_csv(plans_path)
        print(f"[sim] reusing {plans_path}")
    else:
        jobs = sorted([(c, k) for c in circuits for k in args.k], key=lambda j: -j[0].num_qubits)
        plans = pd.DataFrame([p for p in parallel_map(plan, jobs, jobs=args.jobs, label="plan") if p])
        OUT.mkdir(parents=True, exist_ok=True)
        plans.to_csv(plans_path, index=False)
    by_key = plans.set_index(["name", "k"])
    jobs = []
    for c in circuits:
        for k in args.k:
            if (c.name, k) not in by_key.index:
                continue
            entry = by_key.loc[(c.name, k)]
            jobs.append((c, k, [tuple(p) for p in json.loads(entry["placement"])], bool(entry["commute"]),
                         float(ref.loc[c.name, "alpha_final"]), args.shots, args.noiseless_max,
                         args.layer_order))
    jobs.sort(key=lambda j: -j[0].num_qubits)
    results = parallel_map(point, jobs, jobs=args.jobs, label="frames")
    frame = pd.DataFrame([r for res in results if res for r in res])
    path = OUT / f"{args.tag}.csv"
    frame.to_csv(path, index=False)
    print(f"[sim] wrote {path}: {len(frame)} rows, {sum(r is None for r in results)} points failed")


if __name__ == "__main__":
    main()
