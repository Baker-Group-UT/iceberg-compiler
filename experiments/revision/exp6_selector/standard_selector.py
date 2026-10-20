"""
standard_selector.py -- Experiment6e: the standard resource selector against the submitted one.

Under the standard configuration `compile_circuit` plans every code size with the standard
planner and keeps the placement with the lowest standard objective (`packing.standard`).
Its planner weight above k = 2 (`AqornStandard.wide_silent_weight`) is the open choice; this
measures the whole compiler for each candidate weight against:

    v4          `compile_circuit(planner="v4")`: the submitted compiler
    w10T_k2     the standard planner at k = 2 alone (`plan_standard`, n_code 4): Fig 6's arm
    naive_k2    program-order packing at k = 2, every pass on

Fig 8's 13 circuits (proxies) at noise ratios 1/16, 1 and 16, each at Fig 8's recorded alpha
for that (circuit, ratio). Every arm is translated under the standard settings (layer-sorted;
transversal in-patch CZ / XCX at k = 2) except v4, which is translated as submitted, then
converted to noisy Stim once and frame-simulated: selection rate (LSR) and silent logical
error per kept shot, the TVD stand-in that works at these widths.

    PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.standard_selector --jobs 12
    ... --from-csv results/revision/exp6/standard_selector.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import FIG8_CSV, RESULTS, write_sidecar

NAME = "standard_selector"
RATIOS = (0.0625, 1.0, 16.0)
WIDE_WEIGHTS = (1.0, 5.0, 10.0)
MOVEMENT_ERROR = 3.5e-3            # Fig 8's selector knob (`knobs_for(sim, 3.5e-3)`)
SHOTS = 100_000


def _frames(encoded, sim, alpha: float, seed: int) -> Dict[str, float]:
    from experiments.concatenation.noise import rescale_text
    from experiments.revision.exp8_undetected.frames import split_from_stim, stim_params
    from experiments.revision.exp8_undetected.gadgets import noisy_stim

    noisy = noisy_stim(encoded, stim_params(1.0), idle_time_scalar=sim.idle_time_scalar,
                       distance_scalar=sim.distance_scalar)
    split = split_from_stim(encoded, rescale_text(str(noisy), alpha), shots=SHOTS, seed=seed)
    return {"lsr": 1.0 - split["p_detected"], "p_silent": split["p_silent"],
            "silent_per_kept": split["p_silent_given_kept"]}


def _translate(program, placement, n_code: int, commute: bool, *, standard: bool):
    from experiments.shared.simulation import ensure_measurements
    from translator.config import TranslatorConfig
    from translator.core import Translator

    config = TranslatorConfig(n_code=n_code, enable_h_commutation=commute, enable_merging=True,
                              asap_order=standard,
                              transversal_intra_two_qubit=standard and n_code == 4)
    return Translator(config=config).translate(
        ensure_measurements(program), [tuple(p) for p in placement]).circuit


def _run(job) -> List[Dict]:
    from experiments.shared.simulation import SimSettings, knobs_for, noise_profile_for
    from packing import compile_circuit
    from packing.plan import naive_packing
    from packing.standard import STANDARD, plan_standard

    name, circuit, ratio, alpha, seed = job
    sim = SimSettings().with_noise_ratio(ratio)
    knobs, profile = knobs_for(sim, MOVEMENT_ERROR), noise_profile_for(sim)
    arms = []
    began = time.time()
    v4 = compile_circuit(circuit, noise=knobs, planner="v4")
    arms.append(("v4", v4.placement, v4.n_code, bool(v4.h_commutation), False, v4.source, None))
    for weight in WIDE_WEIGHTS:
        standard = STANDARD.with_(wide_silent_weight=weight)
        built = compile_circuit(circuit, noise=knobs, packing_noise=profile, planner="standard",
                                standard=standard)
        arms.append((f"standard_w{weight:g}", built.placement, built.n_code,
                     bool(built.h_commutation), True, built.source, built.objective))
    k2 = plan_standard(circuit, 4, noise=profile)
    arms.append(("w10T_k2", k2.placement, 4, bool(k2.h_commutation), True, "packer", None))
    arms.append(("naive_k2", naive_packing(circuit.num_qubits, 2), 4, True, True, "naive", None))
    compile_seconds = time.time() - began

    rows, measured = [], {}
    for arm, placement, n_code, commute, standard, source, objective in arms:
        key = json.dumps([[list(p) for p in placement], n_code, commute, standard])
        if key not in measured:
            encoded = _translate(circuit, placement, n_code, commute, standard=standard)
            measured[key] = _frames(encoded, sim, alpha, seed)
        rows.append({"circuit": name, "num_qubits": circuit.num_qubits, "ratio": ratio,
                     "alpha": alpha, "arm": arm, "n_code": n_code, "k": n_code - 2,
                     "patches": len({p for p, _ in placement}), "commute": commute,
                     "source": source,
                     "objective": None if objective is None else objective.total,
                     **measured[key], "compile_seconds": compile_seconds})
    return rows


def compute(args) -> pd.DataFrame:
    from experiments.revision.common import load_named_proxies
    from experiments.shared.parallel import parallel_map

    fig8 = pd.read_csv(FIG8_CSV)
    fig8 = fig8[fig8["series"] == "naive_k2"]
    circuits = sorted(fig8["circuit"].unique())
    loaded = load_named_proxies(circuits)
    jobs = []
    for i, name in enumerate(circuits):
        for j, ratio in enumerate(args.ratios):
            match = fig8[(fig8["circuit"] == name) & np.isclose(fig8["ratio"], ratio)]
            if match.empty:
                continue
            jobs.append((name, loaded[name], ratio, float(match["alpha"].iloc[0]),
                         20261012 + 1000 * i + j))
    jobs.sort(key=lambda j: -j[1].num_qubits)
    print(f"[{NAME}] {len(jobs)} (circuit, ratio) columns")
    results = parallel_map(_run, jobs, jobs=args.jobs, label=NAME)
    failed = [(j[0], j[2]) for j, r in zip(jobs, results) if r is None]
    if failed:
        print(f"[{NAME}] FAILED: {failed}")
    return pd.DataFrame([row for r in results if r for row in r])


def summarize(frame: pd.DataFrame) -> None:
    """Geometric-mean ratios against v4, per noise ratio and overall, and the k chosen."""
    frame = frame.copy()
    frame["jk"] = -np.log(frame["lsr"].clip(lower=1e-6)) + 10 * frame["silent_per_kept"]
    ref = frame[frame["arm"] == "v4"].set_index(["circuit", "ratio"])
    rows = []
    for arm, sub in frame.groupby("arm"):
        sub = sub.set_index(["circuit", "ratio"]).loc[ref.index]
        for ratio in [None] + sorted(frame["ratio"].unique()):
            part = sub if ratio is None else sub.xs(ratio, level="ratio", drop_level=False)
            base = ref.loc[part.index]
            rows.append({
                "arm": arm, "ratio": "all" if ratio is None else ratio,
                "lsr_x": float(np.exp(np.log(part["lsr"] / base["lsr"]).mean())),
                "silent_per_kept_x": float(np.exp(np.log(part["silent_per_kept"].clip(lower=1e-6)
                                                         / base["silent_per_kept"].clip(lower=1e-6)).mean())),
                "d_objective_per_kept": float((part["jk"] - base["jk"]).mean()),
                "k_chosen": json.dumps(part["k"].value_counts().sort_index().to_dict()),
            })
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--from-csv", default=None)
    args = parser.parse_args()
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        began = time.time()
        frame = compute(args)
        out = RESULTS / "exp6"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{NAME}.csv"
        version = 2
        while path.exists():
            path = out / f"{NAME}_v{version}.csv"
            version += 1
        frame.to_csv(path, index=False)
        write_sidecar(path, experiment=NAME, settings=vars(args), started=began, status="complete")
        print(f"[{NAME}] wrote {path}")
    summarize(frame)


if __name__ == "__main__":
    main()
