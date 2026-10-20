"""
aqorn_static_k.py -- Experiment6b: AQORN's planner at every static code size.

Adds the best-static comparator with the same commutation and merging passes as the
selector, so that only code selection differs. Fig 8 has AQORN's planner only at k = 2 (`aqorn_fixed`); the static ladder is
naive. This adds `plan_packing` at k = 4, 6, 8, 10 so that "best static k, AQORN" can be
compared against the selector, which then differs from it in the choice of code size and
patch count only.

Fig 8's sweep exactly: the same 13 circuits and 5 nominal ratios, `SimSettings
.with_noise_ratio(ratio)`, the planner given `commutation_pays(circuit, n_code)` and the
noise profile for that ratio with `verify=True`, Stim with distance and idle noise, 40k
raw shots, and Fig 8's miscompile guard (`noiseless_tvd` / `miscompiled`). The noise scale
is **not re-calibrated**: each (circuit, ratio) reuses the alpha Fig 8 recorded, so every
new arm sits on the same machine as the arms already in Fig 8's CSV. (Fig 8's own pilots
were unseeded, so re-calibrating would have moved alpha.)

Three re-runs ride along as consistency checks, each a fresh sample of an arm Fig 8
already has: `aqorn_k2` (= `aqorn_fixed`), `selector_rerun` (= `selector`) and
`naive_k2_rerun` (= `naive_k2`). Seeds are explicit here (Fig 8's were not).

Output series use Fig 8's column schema, so `best_static_replot.py --aqorn-csv` merges them.
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

import pandas as pd

from experiments.revision.common import (
    FIG8_CSV,
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

NAME = "exp6b"
SEED_BASE = 20261003


@dataclass
class Settings:
    k_values: Sequence[int] = (2, 4, 6, 8, 10)
    repetitions: int = 40_000
    movement_error: float = 3.5e-3  # Fig 8's selector default
    include_reruns: bool = True
    smoke: bool = False
    circuits: Sequence[str] = ()
    ratios: Sequence[float] = ()
    jobs: int = 12
    out: str = str(RESULTS / "exp6")


def fig8_columns(settings: Settings) -> pd.DataFrame:
    """(circuit, ratio, alpha) for every column Fig 8 measured completely."""
    from experiments.figures.resource_selection_headroom.resource_selection_headroom import (
        complete_circuits,
    )

    frame = complete_circuits(pd.read_csv(FIG8_CSV), (2, 4, 6, 8, 10))
    columns = frame.groupby(["circuit", "ratio"]).agg(
        alpha=("alpha", "first"), measured_ratio_ref=("measured_ratio_ref", "first"),
        movement_share_ref=("movement_share_ref", "first"),
        program_qubits=("program_qubits", "first")).reset_index()
    if settings.circuits:
        columns = columns[columns["circuit"].isin(settings.circuits)]
    if settings.ratios:
        columns = columns[columns["ratio"].isin(settings.ratios)]
    return columns


def _run_column(job) -> List[Dict]:
    from experiments.shared.simulation import (
        SimSettings,
        knobs_for,
        miscompiled,
        naive_packing,
        noise_profile_for,
        noiseless_tvd,
    )
    from packing.passes import commutation_pays
    from packing.plan import plan_packing
    from packing.resource.select import select_resources

    settings_dict, column, circuit_index, ratio_index, circuit = job
    settings = Settings(**settings_dict)
    name, ratio, alpha = column["circuit"], float(column["ratio"]), float(column["alpha"])
    sim = SimSettings().with_noise_ratio(ratio)
    profile = noise_profile_for(sim)
    repetitions = 2000 if settings.smoke else settings.repetitions
    noise = NoiseSetup(alpha=alpha, stim=True, distance_noise=True,
                       distance_scalar=sim.distance_scalar, idle_noise=True,
                       idle_scalar=sim.idle_time_scalar)

    commutes: Dict[int, bool] = {}

    def commute_for(n_code: int) -> bool:
        if n_code not in commutes:
            commutes[n_code] = commutation_pays(circuit, n_code)
        return commutes[n_code]

    arms = []
    for k in settings.k_values:
        began = time.time()
        try:
            plan = plan_packing(circuit, k + 2, h_commutation=commute_for(k + 2), verify=True,
                                noise=profile)
            arms.append((f"aqorn_k{k}", k + 2, [tuple(p) for p in plan.placement],
                         time.time() - began, bool(plan.fell_back)))
        except Exception:  # noqa: BLE001 - a refused plan costs one arm
            pass
    if settings.include_reruns:
        try:
            chosen = select_resources(circuit, noise=knobs_for(sim, settings.movement_error))
            arms.append(("selector_rerun", chosen.n_code, [tuple(p) for p in chosen.placement],
                         math.nan, False))
        except Exception:  # noqa: BLE001
            pass
        arms.append(("naive_k2_rerun", 4, naive_packing(circuit.num_qubits, 2), 0.0, False))

    clean_reference: Dict[int, float] = {}
    rows: List[Dict] = []
    for series_index, (series, n_code, placement, plan_seconds, fell_back) in enumerate(arms):
        commute = commute_for(n_code)
        clean_arm = math.nan
        if not series.startswith("naive"):
            if n_code not in clean_reference:
                clean_reference[n_code] = noiseless_tvd(
                    circuit, naive_packing(circuit.num_qubits, n_code - 2), sim,
                    n_code=n_code, h_commutation=commute)
            clean_arm = noiseless_tvd(circuit, placement, sim, n_code=n_code,
                                      h_commutation=commute)
            if miscompiled(clean_reference[n_code], clean_arm):
                log(NAME, f"skipping {name} {series} n_code={n_code}: miscompiled "
                          f"({clean_arm:.3f} vs naive {clean_reference[n_code]:.3f})")
                continue
        seed = SEED_BASE + 100_000 * circuit_index + 1_000 * ratio_index + series_index
        config = sim_config(noise, n_code=n_code, commute=commute, merging=True,
                            repetitions=repetitions, seed=seed)
        began = time.time()
        result = simulate_arm(circuit, placement, config)
        rows.append({
            "circuit": name, "program_qubits": circuit.num_qubits,
            "program_gates": circuit.size(), "ratio": ratio,
            "measured_ratio_ref": column["measured_ratio_ref"],
            "movement_share_ref": column["movement_share_ref"],
            "idle_scalar": sim.idle_time_scalar, "distance_scalar": sim.distance_scalar,
            "series": series, "n_code": n_code, "k": n_code - 2,
            "patches": len({p for p, _ in placement}), "alpha": alpha,
            "repetitions": repetitions, "h_commutation": commute,
            "movement_error": settings.movement_error, "lsr": result["lsr"],
            "tvd": result["tvd"], "one_qubit_gates": result["one_qubit_gates"],
            "two_qubit_gates": result["two_qubit_gates"], "circuit_depth": result["depth"],
            "seed": seed, "plan_seconds": plan_seconds, "fell_back": fell_back,
            "noiseless_tvd": clean_arm, "sim_seconds": time.time() - began,
        })
    return rows


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / "aqorn_static_k.csv", key_columns=["circuit", "ratio", "series"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    columns = fig8_columns(settings)
    if settings.smoke:
        columns = columns[(columns["circuit"].isin(["knn_n25", "ising_n26"]))
                          & (columns["ratio"].isin([1.0, 256.0]))]
    circuits = sorted(columns["circuit"].unique())
    ratios = sorted(columns["ratio"].unique())
    loaded = load_named_proxies(circuits)
    jobs = []
    for row in columns.to_dict("records"):
        if table.is_done((row["circuit"], row["ratio"], f"aqorn_k{settings.k_values[-1]}")):
            continue
        jobs.append((asdict(settings), row, circuits.index(row["circuit"]),
                     ratios.index(row["ratio"]), loaded[row["circuit"]]))
    jobs.sort(key=lambda j: -j[1]["program_qubits"])
    log(NAME, f"{len(columns)} (circuit, ratio) columns, {len(jobs)} to run; k {list(settings.k_values)}")
    counts = stream_map(_run_column, jobs, jobs=settings.jobs, label=NAME,
                        on_result=lambda _i, rows: table.append(rows))
    status = "complete" if counts["error"] == 0 else f"complete_with_{counts['error']}_failures"
    table.write_sidecar(status, extra={"failed": counts["error"]})
    frame = table.frame()
    if not frame.empty:
        print(frame.pivot_table(index="ratio", columns="series", values="lsr").round(4).to_string())
    return table.path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp6")
    parser.add_argument("--circuits", nargs="+", default=[])
    parser.add_argument("--ratios", nargs="+", type=float, default=[])
    parser.add_argument("--repetitions", type=int, default=40_000)
    args = parser.parse_args()
    settings = Settings(repetitions=args.repetitions, smoke=args.smoke,
                        circuits=tuple(args.circuits), ratios=tuple(args.ratios),
                        jobs=args.jobs, out=str(out_dir(args, "exp6")))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
