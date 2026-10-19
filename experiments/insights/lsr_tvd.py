"""
lsr_tvd.py -- are selection rate and TVD always anticorrelated? Paired comparisons, everywhere.

Selection rate (LSR) counts shots with no *detected* error; TVD measures how wrong the *kept*
shots are, which is driven by errors the code did not detect (plus a sampling floor that
falls with the number of kept shots). They are different error channels, so a change to a
circuit can move them together or apart. This collects every paired comparison the revision's
data offer -- the same circuit, the same noise and the same shot budget, two configurations --
and classifies each pair by how the two moved:

    both better    LSR up,   error down   (the "anticorrelated" case: fewer errors of every kind)
    shots for accuracy  LSR down, error down   (a trade: more detection, fewer silent errors)
    accuracy for shots  LSR up,   error up     (the opposite trade)
    both worse     LSR down, error up

"error" is TVD where it resolves, else the silent logical error per kept shot from Pauli
frames (TVD ~ 0.95 x it where both resolve). A pair counts only when both moved by more than
`--band` (relative); the rest are "flat".

Sources (all already on disk):
    fig6_v4, fig6_standard   Fig 6's four arms per circuit (submitted; standard)
    planner_k2               the packed arm under v4 vs w10T, and adjacent silent weights 3..30
    planner_k4               the k = 4 weight sweep (frames)
    headroom                 Fig 8 under the standard: every arm against naive [[4,2,2]] (frames)
    phase_diagram            adjacent code sizes per (circuit, noise ratio) (frames)
    density                  the density ablation: stages per density, and density steps (TVD
                             and frames)
Plus the noise axis (`--alpha-sweep`): one placement, alpha varied, frames.

    PYTHONPATH=. venv/bin/python -m experiments.insights.lsr_tvd [--alpha-sweep]
Writes results/insights/lsr_tvd_pairs[_vN].csv and prints the quadrant table.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

OUT = Path("results/insights")
FIG6_V4 = Path("results/revision/_inputs/merging_improvement_benchmark_stylized_simulated.csv")
FIG6_STD = Path("experiments/figures/ablation_standard/ablation_standard.csv")
VARIANTS = Path("results/tvd_explore")
SILENT_AWARE = Path("experiments/figures/ablation_silent_aware")
K4 = Path("results/tvd_explore/weight_sweep_k4.csv")
HEADROOM = Path("experiments/figures/resource_selection_headroom/resource_selection_headroom_standard.csv")
PHASE = Path("experiments/figures/1_best_k_phase_diagram/1_best_k_phase_diagram_standard.csv")
DENSITY = Path("experiments/figures/density_ablation_sweep/density_ablation_sweep_standard.csv")
TVD_RESOLVED = 0.95          # Fig 6's `TVD_SATURATION`
COMPONENTS = ("h_commutation", "merging", "packing")
LSR = "pct_improvement_logical_selection_rate"
TVD = "pct_reduction_logical_total_variation_distance"
STAGE_PAIRS = (("baseline", "h_only", "+ H commutation"), ("h_only", "h_merge", "+ merging"),
               ("h_merge", "packed", "+ packing"), ("baseline", "packed", "all three"))


def _pair(source, comparison, circuit, context, lsr_a, lsr_b, err_a, err_b, measure,
          shots=math.nan) -> Dict:
    return {"source": source, "comparison": comparison, "circuit": circuit, "context": context,
            "lsr_a": lsr_a, "lsr_b": lsr_b, "err_a": err_a, "err_b": err_b, "measure": measure,
            "shots": shots}


def _fig6_arms(rows: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-arm LSR / TVD of one circuit from Fig 6's schema (cumulative percentages)."""
    first = rows.iloc[0]
    by = rows.set_index("component")
    base_lsr, base_tvd = float(first["baseline_lsr"]), float(first["baseline_tvd"])
    arms = {"baseline": {"lsr": base_lsr, "tvd": base_tvd}}
    lsr = tvd = 0.0
    for component, arm in zip(COMPONENTS, ("h_only", "h_merge", "packed")):
        lsr += float(by.loc[component, LSR])
        tvd += float(by.loc[component, TVD])
        arms[arm] = {"lsr": base_lsr * (1 + lsr / 100.0), "tvd": base_tvd * (1 - tvd / 100.0)}
    return arms


def fig6_pairs() -> List[Dict]:
    out = []
    v4 = pd.read_csv(FIG6_V4)
    for name, rows in v4.groupby("circuit_name"):
        if float(rows["baseline_tvd"].iloc[0]) > TVD_RESOLVED:
            continue
        arms, shots = _fig6_arms(rows), float(rows["final_repetitions"].iloc[0])
        for a, b, label in STAGE_PAIRS:
            out.append(_pair("fig6_v4", label, name, "", arms[a]["lsr"], arms[b]["lsr"],
                             arms[a]["tvd"], arms[b]["tvd"], "tvd", shots))
    std = pd.read_csv(FIG6_STD).drop_duplicates("circuit_name")
    for record in std.itertuples():
        if record.baseline_tvd > TVD_RESOLVED:
            continue
        for a, b, label in STAGE_PAIRS:
            out.append(_pair("fig6_standard", label, record.circuit_name, "",
                             getattr(record, f"{a}_lsr"), getattr(record, f"{b}_lsr"),
                             getattr(record, f"{a}_tvd"), getattr(record, f"{b}_tvd"), "tvd",
                             record.final_repetitions))
    return out


def planner_k2_pairs() -> List[Dict]:
    """The packed arm under different planners on Fig 6's machine, alpha, shots and seed."""
    packed: Dict[str, Dict[str, Dict[str, float]]] = {}

    def add(label, frame):
        for name, rows in frame.groupby("circuit_name"):
            if float(rows["baseline_tvd"].iloc[0]) > TVD_RESOLVED:
                continue
            packed.setdefault(name, {})[label] = {
                **_fig6_arms(rows)["packed"], "shots": float(rows["final_repetitions"].iloc[0])}

    add("v4", pd.read_csv(FIG6_V4))
    for weight in (3, 5, 7, 8, 9, 10, 12, 15, 20, 30):
        path = VARIANTS / f"ablation_variant_w{weight}T.csv"
        if path.exists():
            add(f"w{weight}T", pd.read_csv(path))
    out = []
    order = [f"w{w}T" for w in (3, 5, 7, 8, 9, 10, 12, 15, 20, 30)]
    for name, arms in packed.items():
        if "v4" in arms and "w10T" in arms:
            a, b = arms["v4"], arms["w10T"]
            out.append(_pair("planner_k2", "v4 -> w10T", name, "", a["lsr"], b["lsr"],
                             a["tvd"], b["tvd"], "tvd", a["shots"]))
        for lo, hi in zip(order, order[1:]):
            if lo in arms and hi in arms:
                a, b = arms[lo], arms[hi]
                out.append(_pair("planner_k2", "silent weight up (k=2)", name, f"{lo}->{hi}",
                                 a["lsr"], b["lsr"], a["tvd"], b["tvd"], "tvd", a["shots"]))
    return out


def planner_k4_pairs() -> List[Dict]:
    frame = pd.read_csv(K4)
    out = []
    for (name, regime), block in frame.groupby(["circuit", "regime"]):
        block = block.sort_values("weight").drop_duplicates("weight")
        rows = list(block.itertuples())
        for a, b in zip(rows, rows[1:]):
            out.append(_pair("planner_k4", "silent weight up (k=4)", name,
                             f"{regime}:{a.weight:g}->{b.weight:g}", a.lsr, b.lsr,
                             a.silent_per_kept, b.silent_per_kept, "silent_per_kept"))
        # And each plan against the naive k = 4 baseline it was calibrated on.
        for r in rows:
            out.append(_pair("planner_k4", "planner vs naive (k=4)", name, f"{regime}:{r.weight:g}",
                             r.baseline_lsr, r.lsr, r.baseline_silent_per_kept,
                             r.silent_per_kept, "silent_per_kept"))
    return out


def headroom_pairs() -> List[Dict]:
    frame = pd.read_csv(HEADROOM)
    frame = frame[frame["series"] != "oracle_cand"]
    out = []
    for (name, ratio), block in frame.groupby(["circuit", "ratio"]):
        ref = block[block["series"] == "naive_k2"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for r in block.itertuples():
            if r.series == "naive_k2":
                continue
            label = ("larger code, naive" if r.series.startswith("naive_k") else
                     "AQORN static k=2" if r.series == "aqorn_fixed" else "AQORN selector")
            out.append(_pair("headroom", label, name, f"ratio {ratio:g}", ref.lsr, r.lsr,
                             ref.silent_per_kept, r.silent_per_kept, "silent_per_kept",
                             r.repetitions))
    return out


def phase_pairs() -> List[Dict]:
    frame = pd.read_csv(PHASE)
    out = []
    for (name, ratio), block in frame.groupby(["circuit", "ratio"]):
        rows = list(block.sort_values("k").itertuples())
        for a, b in zip(rows, rows[1:]):
            out.append(_pair("phase_diagram", "larger code (AQORN at each k)", name,
                             f"ratio {ratio:g}: k{a.k}->k{b.k}", a.lsr, b.lsr,
                             a.silent_per_kept, b.silent_per_kept, "silent_per_kept",
                             a.repetitions))
    return out


def density_pairs() -> List[Dict]:
    frame = pd.read_csv(DENSITY).drop_duplicates(["circuit_name", "n_codes"])
    out = []
    for r in frame.itertuples():
        resolved = r.baseline_tvd <= TVD_RESOLVED
        for a, b, label in STAGE_PAIRS:
            lsr_a, lsr_b = getattr(r, f"{a}_lsr"), getattr(r, f"{b}_lsr")
            if resolved:
                out.append(_pair("density", label, r.circuit_name, f"{r.n_codes} patches",
                                 lsr_a, lsr_b, getattr(r, f"{a}_tvd"), getattr(r, f"{b}_tvd"),
                                 "tvd", r.final_repetitions))
            out.append(_pair("density_frames", label, r.circuit_name, f"{r.n_codes} patches",
                             lsr_a, lsr_b, getattr(r, f"{a}_silent_per_kept"),
                             getattr(r, f"{b}_silent_per_kept"), "silent_per_kept",
                             r.final_repetitions))
    for name, block in frame.groupby("circuit_name"):
        rows = list(block.sort_values("n_codes").itertuples())
        for a, b in zip(rows, rows[1:]):
            if a.baseline_tvd <= TVD_RESOLVED:
                out.append(_pair("density", "spread over more patches (baseline)", name,
                                 f"{a.n_codes}->{b.n_codes}", a.baseline_lsr, b.baseline_lsr,
                                 a.baseline_tvd, b.baseline_tvd, "tvd", a.final_repetitions))
            out.append(_pair("density_frames", "spread over more patches (baseline)", name,
                             f"{a.n_codes}->{b.n_codes}", a.baseline_lsr, b.baseline_lsr,
                             a.baseline_silent_per_kept, b.baseline_silent_per_kept,
                             "silent_per_kept", a.final_repetitions))
    return out


QUADRANTS = ("both better", "shots for accuracy", "accuracy for shots", "both worse")


def classify(frame: pd.DataFrame, band: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["d_log_lsr"] = np.log(frame["lsr_b"].clip(lower=1e-9) / frame["lsr_a"].clip(lower=1e-9))
    frame["d_log_err"] = np.log(frame["err_b"].clip(lower=1e-9) / frame["err_a"].clip(lower=1e-9))
    up, down = frame["d_log_lsr"] > band, frame["d_log_lsr"] < -band
    better, worse = frame["d_log_err"] < -band, frame["d_log_err"] > band
    frame["quadrant"] = np.select(
        [up & better, down & better, up & worse, down & worse], list(QUADRANTS), "flat")
    return frame


def table(frame: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for (source, comparison), block in frame.groupby(["source", "comparison"], sort=False):
        moved = block[block["quadrant"] != "flat"]
        counts = moved["quadrant"].value_counts()
        rho = (spearmanr(block["d_log_lsr"], block["d_log_err"]).correlation
               if len(block) > 3 else math.nan)
        rows.append({"source": source, "comparison": comparison, "pairs": len(block),
                     "moved": len(moved),
                     **{q: round(100 * counts.get(q, 0) / max(1, len(moved)), 1) for q in QUADRANTS},
                     "spearman(dlogLSR, dlogErr)": round(rho, 2),
                     "error measure": block["measure"].iloc[0]})
    return pd.DataFrame(rows)


def alpha_sweep(names, factors=(0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0)) -> pd.DataFrame:
    """The noise axis: one placement per arm, alpha scaled around Fig 6's, frames."""
    import math as _math

    from experiments.concatenation.noise import rescale_text
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.revision.common import load_fig6_circuits
    from experiments.revision.exp8_undetected.frames import split_from_stim, stim_params
    from experiments.revision.exp8_undetected.gadgets import noisy_stim
    from packing.standard import standard_translator_config
    from translator.core import Translator

    std = pd.read_csv(FIG6_STD).drop_duplicates("circuit_name").set_index("circuit_name")
    rows = []
    for circuit in load_fig6_circuits(names=list(names)):
        if circuit.name not in std.index:
            continue
        record = std.loc[circuit.name]
        proxy = fig6.ensure_measurements(circuit.proxy)
        n = proxy.num_qubits
        tight = fig6.tight_trivial_packing(n, _math.ceil(n / 2))
        placement = [tuple(p) for p in json.loads(record["placement"])]
        for arm, perm, commute, merge in (("baseline", tight, False, False),
                                          ("packed", placement, bool(record["commute"]), True)):
            config = standard_translator_config(4, commute=commute, merging=merge)
            config.transversal_intra_two_qubit = merge
            encoded = Translator(config=config).translate(proxy, list(perm)).circuit
            text = str(noisy_stim(encoded, stim_params(1.0), idle_time_scalar=1.0,
                                  distance_scalar=1.0))
            for factor in factors:
                alpha = float(record["alpha_final"]) * factor
                split = split_from_stim(encoded, rescale_text(text, alpha), shots=100_000,
                                        seed=5)
                rows.append({"circuit": circuit.name, "arm": arm, "alpha_factor": factor,
                             "alpha": alpha, "lsr": 1 - split["p_detected"],
                             "silent_per_kept": split["p_silent_given_kept"]})
    return pd.DataFrame(rows)


def _free(path: Path) -> Path:
    version = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        version += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--band", type=float, default=0.03,
                        help="relative change (log) below which a metric counts as flat")
    parser.add_argument("--alpha-sweep", nargs="*", default=None,
                        help="circuits for the noise-axis sweep (frames)")
    args = parser.parse_args()
    pairs = []
    for loader in (fig6_pairs, planner_k2_pairs, planner_k4_pairs, headroom_pairs, phase_pairs,
                   density_pairs):
        pairs.extend(loader())
    frame = classify(pd.DataFrame(pairs), args.band)
    OUT.mkdir(parents=True, exist_ok=True)
    path = _free(OUT / "lsr_tvd_pairs.csv")
    frame.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(f"[lsr_tvd] {len(frame)} paired comparisons -> {path}")
    print(table(frame).to_string(index=False))
    if args.alpha_sweep is not None:
        sweep = alpha_sweep(args.alpha_sweep or ("adder_n10", "qft_n18", "bv_n19", "ising_n26",
                                                 "knn_n25", "multiplier_n15"))
        sweep_path = _free(OUT / "lsr_tvd_alpha_sweep.csv")
        sweep.to_csv(sweep_path, index=False)
        print(f"[lsr_tvd] alpha sweep -> {sweep_path}")
        print(sweep.pivot_table(index=["circuit", "alpha_factor"], columns="arm",
                                values=["lsr", "silent_per_kept"]).round(4).to_string())


if __name__ == "__main__":
    main()
