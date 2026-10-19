"""
calibration_table.py -- Experiment2a: the per-circuit noise scale Fig 6 actually used.

Reports the noise scale factors Fig 6 chose, per circuit. No
simulation: every number comes from Fig 6's CSV (copied to `results/revision/_inputs/`).

Fig 6 solves a noise scale `alpha` per circuit so that the *baseline* (naive packing, both
passes off) keeps about `target_selection_rate` of its shots, then runs every arm of that
circuit at that alpha. `alpha` multiplies every error rate of the NEUTRAL_ATOM machine
(`simulator.noise_models`): p1, p2, readout and the per-carriage move error scale with
it, and T1/T2 shrink by it. This table spells out the physical machine each circuit was
simulated on.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.revision.common import FIG6_CSV, RESULTS, fig6_reference, log, write_sidecar

NAME = "exp2a"


def build() -> pd.DataFrame:
    from simulator.noise_models import BASE_MOVE_ERROR, NEUTRAL_ATOM

    ref = fig6_reference()
    alpha = ref["alpha_final"].astype(float)
    table = pd.DataFrame({
        "circuit": ref["name"],
        "size_group": ref["size_group"],
        "num_qubits": ref["num_qubits"],
        "baseline_gate_count": ref["baseline_gate_count"],
        "alpha_final": alpha,
        "p1": np.minimum(NEUTRAL_ATOM["one_qubit_error"] * alpha, 1.0),
        "p2": np.minimum(NEUTRAL_ATOM["two_qubit_error"] * alpha, 1.0),
        "p_meas": np.minimum(NEUTRAL_ATOM["readout_error"] * alpha, 1.0),
        "move_error": np.minimum(BASE_MOVE_ERROR * alpha, 1.0),
        "T1_s": NEUTRAL_ATOM["T1"] / alpha,
        "T2_s": NEUTRAL_ATOM["T2"] / alpha,
        "final_repetitions": ref["final_repetitions"],
        "pilot_final_lsr": ref["pilot_final_lsr"],
        "baseline_lsr": ref["baseline_lsr"],
        "baseline_accepted_shots": (ref["baseline_lsr"] * ref["final_repetitions"]).round(),
        "baseline_tvd": ref["baseline_tvd"],
    })
    return table.sort_values(["size_group", "num_qubits", "circuit"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "exp2")
    args = parser.parse_args()
    started = time.time()
    table = build()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "fig6_calibration_table.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite (pass a new --out)")
    from experiments.shared.figures import stamp_provenance

    stamp_provenance(table).to_csv(path, index=False)
    alpha = table["alpha_final"]
    summary = {
        "circuits": int(len(table)),
        "alpha_min": float(alpha.min()), "alpha_median": float(alpha.median()),
        "alpha_mean": float(alpha.mean()), "alpha_max": float(alpha.max()),
        "alpha_spread_factor": float(alpha.max() / alpha.min()),
        "p2_min": float(table["p2"].min()), "p2_max": float(table["p2"].max()),
        "baseline_lsr_mean": float(table["baseline_lsr"].mean()),
        "baseline_lsr_min": float(table["baseline_lsr"].min()),
        "baseline_lsr_max": float(table["baseline_lsr"].max()),
        "target_selection_rate": 0.06,
        "paper_text_says": "~0.2",
        "circuits_alpha_above_1": int((alpha > 1).sum()),
        "circuits_alpha_below_1": int((alpha < 1).sum()),
    }
    write_sidecar(path, experiment=NAME, settings={"source": str(FIG6_CSV)},
                  started=started, status="complete", extra=summary)
    log(NAME, f"wrote {path}")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
