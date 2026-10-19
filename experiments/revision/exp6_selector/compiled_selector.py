"""
compiled_selector.py -- Experiment6d: Fig 8's selector arm, but the whole compiler.

Fig 8's "AQORN selector" is `select_resources` alone: it picks the code size and patch count
and returns *its own* placement, from a thinner search (an affinity-greedy seed and one
descent). Experiment6b found that this arm loses to AQORN's planner at a single static k
chosen with hindsight -- and that the selector chooses that same k (k = 4) in 55 of 65
columns. The loss is placement, not code-size choice.

The shipped compiler is `compile_circuit`: `select_resources` for the shape, then
`plan_packing` at that code size, then the selector's placement, the planner's and naive
ranked on the resource objective. This simulates *that* arm on Fig 8's sweep -- same 13
circuits, 5 ratios, recorded alpha, 40k raw shots, `commutation_pays` (compile_circuit's
own default), the same miscompile guard -- so 6a's comparison can be redone with the
compiler the paper describes rather than its first stage.

Output uses Fig 8's schema with series `compile_circuit`.
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
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

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

NAME = "exp6d"
SEED_BASE = 20261011


@dataclass
class Settings:
    repetitions: int = 40_000
    movement_error: float = 3.5e-3
    smoke: bool = False
    jobs: int = 4
    out: str = str(RESULTS / "exp6")


def _run_column(job) -> List[Dict]:
    from experiments.shared.simulation import (
        SimSettings,
        knobs_for,
        miscompiled,
        naive_packing,
        noiseless_tvd,
    )
    from packing import compile_circuit

    settings_dict, column, circuit_index, ratio_index, circuit = job
    settings = Settings(**settings_dict)
    name, ratio, alpha = column["circuit"], float(column["ratio"]), float(column["alpha"])
    sim = SimSettings().with_noise_ratio(ratio)
    began = time.time()
    built = compile_circuit(circuit, noise=knobs_for(sim, settings.movement_error))
    compile_seconds = time.time() - began
    placement = [tuple(p) for p in built.placement]
    n_code, commute = built.n_code, bool(built.h_commutation)
    clean_ref = noiseless_tvd(circuit, naive_packing(circuit.num_qubits, n_code - 2), sim,
                              n_code=n_code, h_commutation=commute)
    clean = noiseless_tvd(circuit, placement, sim, n_code=n_code, h_commutation=commute)
    if miscompiled(clean_ref, clean):
        log(NAME, f"skipping {name} ratio={ratio}: miscompiled ({clean:.3f} vs {clean_ref:.3f})")
        return []
    repetitions = 2000 if settings.smoke else settings.repetitions
    seed = SEED_BASE + 100_000 * circuit_index + 1_000 * ratio_index
    noise = NoiseSetup(alpha=alpha, distance_scalar=sim.distance_scalar,
                       idle_scalar=sim.idle_time_scalar)
    result = simulate_arm(circuit, placement, sim_config(
        noise, n_code=n_code, commute=commute, merging=True, repetitions=repetitions, seed=seed))
    return [{
        "circuit": name, "program_qubits": circuit.num_qubits, "program_gates": circuit.size(),
        "ratio": ratio, "measured_ratio_ref": column["measured_ratio_ref"],
        "movement_share_ref": column["movement_share_ref"],
        "idle_scalar": sim.idle_time_scalar, "distance_scalar": sim.distance_scalar,
        "series": "compile_circuit", "n_code": n_code, "k": n_code - 2,
        "patches": len({p for p, _ in placement}), "alpha": alpha,
        "repetitions": repetitions, "h_commutation": commute,
        "movement_error": settings.movement_error, "lsr": result["lsr"], "tvd": result["tvd"],
        "one_qubit_gates": result["one_qubit_gates"],
        "two_qubit_gates": result["two_qubit_gates"], "circuit_depth": result["depth"],
        "seed": seed, "source": built.source, "compile_seconds": compile_seconds,
        "noiseless_tvd": clean,
    }]


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / "compiled_selector.csv", key_columns=["circuit", "ratio", "series"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    columns = fig8_columns(StaticSettings())
    if settings.smoke:
        columns = columns[(columns["circuit"] == "knn_n25") & (columns["ratio"] == 1.0)]
    circuits = sorted(columns["circuit"].unique())
    ratios = sorted(columns["ratio"].unique())
    loaded = load_named_proxies(circuits)
    jobs = [(asdict(settings), row, circuits.index(row["circuit"]), ratios.index(row["ratio"]),
             loaded[row["circuit"]]) for row in columns.to_dict("records")
            if not table.is_done((row["circuit"], row["ratio"], "compile_circuit"))]
    jobs.sort(key=lambda j: -j[1]["program_qubits"])
    log(NAME, f"{len(jobs)} (circuit, ratio) columns to run")
    counts = stream_map(_run_column, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    frame = table.frame()
    if not frame.empty:
        print(frame.groupby("ratio")[["lsr"]].mean().round(4).to_string())
        print("source:", frame["source"].value_counts().to_dict(),
              "k:", frame["k"].value_counts().to_dict())
    return table.path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp6")
    args = parser.parse_args()
    return Settings(smoke=args.smoke, jobs=args.jobs, out=str(out_dir(args, "exp6"))), args.resume


if __name__ == "__main__":
    run(*parse_args())
