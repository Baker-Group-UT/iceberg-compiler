"""
noise_mismatch.py -- Experiment6c: the resource selector told the wrong machine.

Tests robustness to parameter mismatch. The selector ranks
code sizes and patch counts with an analytic noise model of a machine; in practice that
model is never exactly the machine. Here it is deliberately wrong: the selector chooses
for idle/movement ratio `r_model`, and the choice is simulated on the machine at `r_true`.

Ratios {1/16, 1, 16} (the middle three of Fig 8's axis), all nine (r_model, r_true) pairs
per circuit -- the diagonal is the matched selector, re-run, so every mismatched point
has a paired reference from the same run. Fig 8's 13 circuits, Fig 8's recorded alpha for
(circuit, r_true), 40k raw shots, the same commutation rule and miscompile guard.

Reported per (circuit, r_model, r_true): LSR, and the chosen code size and patch count.
The summary is the mismatch penalty, LSR(r_model -> r_true) / LSR(matched at r_true), as a
geometric mean over circuits, and against naive k = 2 from Fig 8's CSV.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
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
    load_named_proxies,
    log,
    out_dir,
    sim_config,
    simulate_arm,
    stream_map,
)
from experiments.revision.exp6_selector.aqorn_static_k import Settings as StaticSettings
from experiments.revision.exp6_selector.aqorn_static_k import fig8_columns

NAME = "exp6c"
SEED_BASE = 20261006


@dataclass
class Settings:
    ratios: Sequence[float] = (1 / 16, 1.0, 16.0)
    repetitions: int = 40_000
    movement_error: float = 3.5e-3
    smoke: bool = False
    jobs: int = 12
    out: str = str(RESULTS / "exp6")


def _run_circuit(job) -> List[Dict]:
    from experiments.shared.simulation import (
        SimSettings,
        knobs_for,
        miscompiled,
        naive_packing,
        noiseless_tvd,
    )
    from packing.passes import commutation_pays
    from packing.resource.select import select_resources

    settings_dict, name, circuit, alphas, circuit_index = job
    settings = Settings(**settings_dict)
    repetitions = 2000 if settings.smoke else settings.repetitions
    choices = {}
    for r_model in settings.ratios:
        sim_model = SimSettings().with_noise_ratio(r_model)
        chosen = select_resources(circuit, noise=knobs_for(sim_model, settings.movement_error))
        choices[r_model] = (chosen.n_code, [tuple(p) for p in chosen.placement])

    rows: List[Dict] = []
    clean_reference: Dict[int, float] = {}
    for true_index, r_true in enumerate(settings.ratios):
        if r_true not in alphas:
            continue
        sim_true = SimSettings().with_noise_ratio(r_true)
        noise = NoiseSetup(alpha=alphas[r_true], distance_scalar=sim_true.distance_scalar,
                           idle_scalar=sim_true.idle_time_scalar)
        for model_index, r_model in enumerate(settings.ratios):
            n_code, placement = choices[r_model]
            commute = commutation_pays(circuit, n_code)
            if n_code not in clean_reference:
                clean_reference[n_code] = noiseless_tvd(
                    circuit, naive_packing(circuit.num_qubits, n_code - 2), sim_true,
                    n_code=n_code, h_commutation=commute)
            clean = noiseless_tvd(circuit, placement, sim_true, n_code=n_code,
                                  h_commutation=commute)
            if miscompiled(clean_reference[n_code], clean):
                log(NAME, f"skipping {name} r_model={r_model} n_code={n_code}: miscompiled")
                continue
            seed = SEED_BASE + 100_000 * circuit_index + 100 * true_index + model_index
            config = sim_config(noise, n_code=n_code, commute=commute, merging=True,
                                repetitions=repetitions, seed=seed)
            result = simulate_arm(circuit, placement, config)
            rows.append({
                "circuit": name, "program_qubits": circuit.num_qubits, "r_model": r_model,
                "r_true": r_true, "matched": bool(abs(r_model - r_true) < 1e-12),
                "n_code": n_code, "k": n_code - 2,
                "patches": len({p for p, _ in placement}), "alpha": alphas[r_true],
                "repetitions": repetitions, "h_commutation": commute, "lsr": result["lsr"],
                "tvd": result["tvd"], "seed": seed,
            })
    return rows


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / "noise_mismatch.csv",
                        key_columns=["circuit", "r_true", "r_model"], experiment=NAME,
                        settings=asdict(settings), resume=resume)
    columns = fig8_columns(StaticSettings())
    columns = columns[columns["ratio"].isin(settings.ratios)]
    names = sorted(columns["circuit"].unique())
    if settings.smoke:
        names = names[:2]
    loaded = load_named_proxies(names)
    jobs = []
    for index, name in enumerate(names):
        block = columns[columns["circuit"] == name]
        alphas = {float(r): float(a) for r, a in zip(block["ratio"], block["alpha"])}
        last = settings.ratios[-1]
        if table.is_done((name, last, last)):
            continue
        jobs.append((asdict(settings), name, loaded[name], alphas, index))
    log(NAME, f"{len(names)} circuits x {len(settings.ratios)}^2 (r_model, r_true); "
              f"{len(jobs)} to run")
    counts = stream_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    summarize(table.frame())
    return table.path


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    matched = frame[frame["matched"]].set_index(["circuit", "r_true"])["lsr"]
    frame = frame.join(matched.rename("matched_lsr"), on=["circuit", "r_true"])
    frame["penalty"] = frame["lsr"] / frame["matched_lsr"]
    table = frame.groupby(["r_true", "r_model"])["penalty"].apply(
        lambda v: float(np.exp(np.log(v.clip(lower=1e-9)).mean()))).unstack()
    print(f"[{NAME}] geometric-mean LSR(r_model -> r_true) / LSR(matched), rows r_true:")
    print(table.round(3).to_string())
    return table


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp6")
    args = parser.parse_args()
    return Settings(smoke=args.smoke, jobs=args.jobs, out=str(out_dir(args, "exp6"))), args.resume


if __name__ == "__main__":
    run(*parse_args())
