"""
circuit_level.py -- Fig 6's circuits with a solo patch, decoded as [[4,2,2]] and as [[4,1,2]].

The same noisy samples, decoded two ways (paired):

    [[4,2,2]]   Fig 6's rule: drop a shot if any parity bit fires or any patch's data
                readout has odd weight; read every tracked lane (`iceberg_postselect` +
                `reduce_distribution`).
    [[4,1,2]]   the same, and also drop it if a valid gauge check of a solo patch fires
                (`gauge.analyse`: the spare lane's final Zbar must be fixed by its |0bar>
                preparation for every program input).

Arms: Fig 6's baseline (tight trivial packing, H commutation and merging off, seed offset
3) and the packed arms of v4 (shipped AQORN), w10 (silent-aware, weight 10) and w10T (w10
with transversal in-patch CZ/XCX), each at the circuit's Fig 6 alpha, shot count and seed
(`10 * 1_000_003 + circuit_idx * 17 + offset`, offset 6 for packed arms). The [[4,2,2]]
numbers must reproduce the source CSVs exactly; the run refuses to plot otherwise.

Optional extra arms (`--variants`), opt-in translator switches, never defaults:
    *_tH   SOLO_H_TARGETED: a solo H is a targeted single-lane H, not a whole-patch HH,
           so the spare is never rotated (costs gates; the targeted H leaks silent error)
    *_fix  SOLO_GAUGE_PARITY_FIX: keep HH, but if a solo patch received an odd number of
           HH, one targeted H on its spare lane before readout restores the Zbar frame

Per arm: LSR and TVD under both rules (TVD only meaningful for n <= 18, see the memory
note on TVD estimability), a paired bootstrap of the TVD difference, the exact count of
shots only the gauge check drops, and a Pauli-frame estimate of silent error per kept shot
under both rules (valid at any width), split into shots the gauge check catches with an
output error and shots it drops harmlessly.

Run (from iceberg-compiler/):
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.circuit_level --jobs 3
    PYTHONPATH=. venv/bin/python -m experiments.solo_gauge.circuit_level \\
        --from-csv results/solo_gauge/circuit_level.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.revision.common import RESULTS, ResultTable, log, stream_map

NAME = "solo_gauge"
OUT = RESULTS.parent / "solo_gauge"
REPO = RESULTS.parents[1]
FIG6_CSV = RESULTS / "_inputs" / "merging_improvement_benchmark_stylized_simulated.csv"
V4_PLANS = RESULTS / "exp8" / "planner_static.csv"
W10_CSV = REPO / "experiments" / "figures" / "ablation_silent_aware" / "ablation_silent_aware_w10.csv"
W10T_CSV = RESULTS.parent / "tvd_explore" / "ablation_variant_w10T.csv"
LSR = "pct_improvement_logical_selection_rate"
TVD = "pct_reduction_logical_total_variation_distance"
OFFSET = {"baseline": 3, "packed": 6, "sparse": 7}
#: Memory note `tvd-estimability-limit`: TVD carries signal only up to ~18 program qubits.
TVD_MAX_QUBITS = 18
BASE_ARMS = ("baseline", "v4", "w10", "w10T")
VARIANT_SWITCH = {"tH": "SOLO_H_TARGETED", "fix": "SOLO_GAUGE_PARITY_FIX"}


# ----------------------------------------------------------------------------------------
# arm specifications
# ----------------------------------------------------------------------------------------


def _reference_values(rows: pd.DataFrame) -> Tuple[float, float]:
    """(LSR, TVD) of the packed arm from a Fig 6-schema block of three rows."""
    first = rows.iloc[0]
    lsr = float(first["baseline_lsr"]) * (1 + rows[LSR].sum() / 100.0)
    tvd = float(first["baseline_tvd"]) * (1 - rows[TVD].sum() / 100.0)
    return lsr, tvd


def arm_specs(names: Sequence[str], variants: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    """{circuit: {arm: spec}} with placement, commute, merging, switches, seed offset, reference."""
    import math as _math

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    fig6_rows = pd.read_csv(FIG6_CSV)
    v4 = pd.read_csv(V4_PLANS)
    v4 = v4[v4["silent_weight"] == 0].set_index("name")
    w10 = pd.read_csv(W10_CSV)
    w10t = pd.read_csv(W10T_CSV)
    out: Dict[str, Dict[str, dict]] = {}
    for name in names:
        block = fig6_rows[fig6_rows["circuit_name"] == name]
        first = block.iloc[0]
        n = int(first["num_qubits"])
        tight = [list(p) for p in fig6.tight_trivial_packing(n, _math.ceil(n / 2))]
        specs = {
            "baseline": dict(placement=tight, commute=False, merging=False, switches=(),
                             offset=OFFSET["baseline"],
                             ref=(float(first["baseline_lsr"]), float(first["baseline_tvd"]))),
            "v4": dict(placement=json.loads(v4.loc[name, "placement"]),
                       commute=bool(v4.loc[name, "commute"]), merging=True, switches=(),
                       offset=OFFSET["packed"], ref=_reference_values(block)),
        }
        wb = w10[w10["circuit_name"] == name]
        specs["w10"] = dict(placement=json.loads(wb.iloc[0]["placement"]),
                            commute=bool(wb.iloc[0]["commute"]), merging=True, switches=(),
                            offset=OFFSET["packed"], ref=_reference_values(wb))
        tb = w10t[w10t["circuit_name"] == name]
        specs["w10T"] = dict(placement=json.loads(tb.iloc[0]["placement"]),
                             commute=bool(tb.iloc[0]["commute"]), merging=True,
                             switches=("TRANSVERSAL_INTRA_TWO_QUBIT",),
                             offset=OFFSET["packed"], ref=_reference_values(tb))
        # Packing-density arms: every program qubit alone in its patch, so every patch is
        # solo and every spare a candidate check. Fig 6's baseline translation (passes off),
        # a seed offset Fig 6 never used.
        specs["sparse"] = dict(placement=[[i, 0] for i in range(n)], commute=False,
                               merging=False, switches=(), offset=OFFSET["sparse"], ref=None)
        for variant in variants:
            base, tag = variant.split("_")
            spec = dict(specs[base])
            spec["base_switches"] = tuple(spec["switches"])
            spec["switches"] = tuple(spec["switches"]) + (VARIANT_SWITCH[tag],)
            spec["ref"] = None
            specs[variant] = spec
        out[name] = specs
    return out


# ----------------------------------------------------------------------------------------
# one arm
# ----------------------------------------------------------------------------------------


def _paired_bootstrap(decoded, ideal: Dict[str, float], raw_shots: int, ideal_shots: int,
                      rng: np.random.Generator, draws: int) -> Dict[str, float]:
    """Resample the noisy shots and the sampled ideal; TVD under both rules per draw."""
    keys = list(decoded.keys)
    index = {key: i for i, key in enumerate(keys)}
    for key in ideal:
        if key not in index:
            index[key] = len(keys)
            keys.append(key)
    q = np.zeros(len(keys))
    for key, p in ideal.items():
        q[index[key]] = p
    q = q / q.sum()
    probs = decoded.probs / decoded.probs.sum()
    k422, k412 = decoded.kept_422, decoded.kept_412
    diffs = np.empty(draws)
    base = np.empty(draws)
    for b in range(draws):
        counts = rng.multinomial(raw_shots, probs).astype(float)
        qb = rng.multinomial(ideal_shots, q) / ideal_shots
        values = []
        for kept in (k422, k412):
            mass = np.bincount(decoded.key_index[kept], weights=counts[kept],
                               minlength=len(keys))
            p = mass / max(mass.sum(), 1.0)
            values.append(0.5 * np.abs(p - qb).sum())
        base[b] = values[0]
        diffs[b] = values[1] - values[0]
    return {"boot_draws": draws, "d_tvd_boot_sd": float(diffs.std(ddof=1)),
            "d_tvd_ci_lo": float(np.percentile(diffs, 2.5)),
            "d_tvd_ci_hi": float(np.percentile(diffs, 97.5)),
            "tvd_422_boot_sd": float(base.std(ddof=1))}


def run_arm(job) -> Dict[str, object]:
    """Translate, analyse the gauge, simulate with Fig 6's seed, decode both ways."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from analysis.metrics import total_variation_distance
    from analysis.postselection import postselect_against_reference
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.revision.exp8_undetected.frames import capture_noisy_stim
    from experiments.solo_gauge import gauge
    from simulator.simulator import simulate_batch
    from translator import explore_switches
    from translator.core import Translator

    began = time.time()
    circuit, arm, spec, settings = job
    s = fig6.BenchmarkSettings()
    reference = settings["fig6"][circuit.name]
    alpha, reps, idx = reference["alpha"], reference["reps"], reference["idx"]
    seed = s.seed * 1_000_003 + idx * 17 + spec["offset"]
    cfg = fig6.build_pipeline_config(s, alpha=alpha, repetitions=reps, seed=seed,
                                     enable_h_commutation=spec["commute"],
                                     enable_merging=spec["merging"])
    program = fig6.ensure_measurements(circuit.proxy)
    placement = [tuple(p) for p in spec["placement"]]
    explore_switches.configure(**{name: True for name in spec["switches"]})
    try:
        translated = Translator(config=cfg.translator_config).translate_batch(program, [placement])
    finally:
        explore_switches.configure(**{name: False for name in spec["switches"]})
    if not translated:
        raise RuntimeError(f"{circuit.name}/{arm}: placement rejected")
    tr = translated[0]
    physical = tr.circuit
    one, two = one_two_qubit_gate_totals(physical)
    identical = False
    if "base_switches" in spec:
        # A variant whose switch changes nothing (no solo H; or an even HH count for the
        # parity fix) is the base arm's circuit: same seed, same samples. Not re-simulated;
        # the analysis takes the base arm's row.
        explore_switches.configure(**{name: True for name in spec["base_switches"]})
        try:
            base_tr = Translator(config=cfg.translator_config).translate_batch(
                program, [placement])[0]
        finally:
            explore_switches.configure(**{name: False for name in spec["base_switches"]})
        identical = base_tr.circuit == physical

    analysis = gauge.analyse_translation(tr)
    row: Dict[str, object] = {
        "name": circuit.name, "size_group": circuit.size_group, "circuit_idx": idx,
        "num_qubits": program.num_qubits, "arm": arm, "alpha": alpha, "raw_shots": reps,
        "seed": seed, "commute": spec["commute"], "merging": spec["merging"],
        "switches": "+".join(spec["switches"]), "placement": json.dumps(spec["placement"]),
        "one_qubit_gates": float(one), "two_qubit_gates": float(two),
        "depth": float(physical.depth()), **analysis.summary(),
        "status": json.dumps({str(k): v for k, v in analysis.status.items()}),
        "contains_faults": bool(tr.contains_faults), "identical_to_base": identical,
    }
    if identical:
        row["seconds"] = time.time() - began
        return row
    row.update(gauge.noiseless_gate(physical, analysis, shots=settings["noiseless_shots"]))
    if program.num_qubits <= settings["random_input_max_qubits"]:
        row.update(gauge.randomized_input_gate(physical, analysis, tr.permutation,
                                               inputs=settings["random_inputs"]))
    translate_seconds = time.time() - began

    with capture_noisy_stim() as captured:
        ideal, noisy = simulate_batch(program, translated, cfg.sim_config)
    result = noisy[0]
    sim_seconds = time.time() - began - translate_seconds

    decoded = gauge.decode_samples(result.noisy_dist, result.key_order, result.key_widths,
                                   tr.context.state_positions, analysis.checks,
                                   cfg.translator_config.n_code)
    dist_422, lsr_422 = decoded.distribution("422")
    dist_412, lsr_412 = decoded.distribution("412")
    ideal_dist = ideal.ideal_dist
    tvd_422 = total_variation_distance(dist_422, ideal_dist)
    tvd_412 = total_variation_distance(dist_412, ideal_dist)
    shots_422 = int(round(lsr_422 * reps))
    shots_412 = int(round(lsr_412 * reps))
    row.update({
        "lsr_422": lsr_422, "lsr_412": lsr_412, "tvd_422": tvd_422, "tvd_412": tvd_412,
        "kept_422": shots_422, "kept_412": shots_412, "gauge_dropped": shots_422 - shots_412,
        "ideal_support": len(ideal_dist),
    })
    if program.num_qubits <= settings["library_check_max_qubits"]:
        lib = postselect_against_reference(translated, ideal, noisy)[0]
        row["lib_lsr"] = float(lib.logical_selection_rate)
        row["lib_tvd"] = float(lib.logical_total_variation_distance)
    if spec["ref"] is not None:
        row["ref_lsr"], row["ref_tvd"] = spec["ref"]
        row["reproduced"] = bool(abs(lsr_422 - spec["ref"][0]) < 1e-9
                                 and abs(tvd_422 - spec["ref"][1]) < 1e-9)
    if program.num_qubits <= settings["tvd_max_qubits"] and analysis.checks:
        rng = np.random.default_rng(settings["boot_seed"] + idx)
        row.update(_paired_bootstrap(decoded, ideal_dist, reps, reps, rng,
                                     settings["boot_draws"]))
    if captured:
        noisy_stim = captured[-1][1]
        row.update(gauge.frame_split(physical, noisy_stim, tr.context.state_positions,
                                     analysis.checks, shots=settings["frame_shots"],
                                     seed=settings["frame_seed"]))
    row.update({"translate_seconds": translate_seconds, "sim_seconds": sim_seconds,
                "seconds": time.time() - began})
    return row


# ----------------------------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------------------------


def settings_from(args) -> Dict[str, object]:
    return {
        "arms": list(args.arms), "variants": list(args.variants), "names": list(args.names),
        "noiseless_shots": 2000, "random_inputs": 6, "random_input_max_qubits": 30,
        "library_check_max_qubits": args.library_check_max_qubits,
        "tvd_max_qubits": TVD_MAX_QUBITS, "boot_draws": args.boot_draws, "boot_seed": 2027,
        "frame_shots": args.frame_shots, "frame_seed": 5,
    }


def compute(args) -> Path:
    from experiments.revision.common import fig6_reference, load_fig6_circuits

    settings = settings_from(args)
    reference = fig6_reference()
    odd = reference[reference["num_qubits"] % 2 == 1]
    if args.max_qubits:
        names = list(args.names) or sorted(
            reference.loc[reference["num_qubits"] <= args.max_qubits, "name"])
    else:
        names = list(args.names) or sorted(odd["name"])
    specs = arm_specs(names, args.variants)
    # Every circuit with a solo patch in any arm (in this corpus: exactly the odd widths).
    from experiments.solo_gauge.gauge import solo_patches_of

    names = [n for n in names
             if any(solo_patches_of([tuple(p) for p in sp["placement"]]) for sp in specs[n].values())]
    circuits = {c.name: c for c in load_fig6_circuits(names=names)}
    settings["fig6"] = {
        row.name: {"alpha": float(row.alpha_final), "reps": int(row.final_repetitions),
                   "idx": int(row.circuit_idx)}
        for row in reference.itertuples(index=False) if row.name in names
    }
    out = Path(args.out)
    table = ResultTable(out / f"{args.stem}.csv", key_columns=["name", "arm"],
                        experiment=NAME, settings=settings)
    arms = list(args.arms) + list(args.variants)
    jobs = [(circuits[n], arm, specs[n][arm], settings) for n in names for arm in arms
            if not table.is_done((n, arm))]
    jobs.sort(key=lambda j: -j[0].num_qubits)
    log(NAME, f"{len(names)} circuits x {len(arms)} arms; {len(jobs)} to run")

    def on_result(_index, row):
        table.append([row])
        if row.get("identical_to_base"):
            log(NAME, f"{row['name']}/{row['arm']}: identical to its base arm, not simulated")
            return
        log(NAME, f"{row['name']}/{row['arm']}: checks {row['checks']} "
                  f"LSR {row['lsr_422']:.4f}->{row['lsr_412']:.4f} "
                  f"TVD {row['tvd_422']:.4f}->{row['tvd_412']:.4f} "
                  f"repro {row.get('reproduced')} ({row['seconds']:.0f}s)")

    counts = stream_map(run_arm, jobs, jobs=args.jobs, label=NAME, on_result=on_result)
    table.write_sidecar(status="complete" if counts["error"] == 0 else "partial",
                        extra={"counts": counts})
    return table.path


# ----------------------------------------------------------------------------------------
# summary and figure
# ----------------------------------------------------------------------------------------


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-arm aggregates printed for the report."""
    rows = []
    for arm, block in frame.groupby("arm", sort=False):
        with_check = block[block["checks"] > 0]
        narrow = with_check[with_check["num_qubits"] <= TVD_MAX_QUBITS]
        rel_tvd = 100 * (narrow["tvd_412"] - narrow["tvd_422"]) / narrow["tvd_422"]
        rel_lsr = 100 * (block["lsr_412"] - block["lsr_422"]) / block["lsr_422"]
        rel_silent = 100 * (with_check["silent_per_kept_412"] - with_check["silent_per_kept_422"]) \
            / with_check["silent_per_kept_422"]
        rows.append({
            "arm": arm, "circuits": len(block),
            "check_available": int((block["checks"] > 0).sum()),
            "x_frame": int(block["cand_x_frame"].sum()), "entangled": int(block["cand_entangled"].sum()),
            "reproduced": (f"{int(block['reproduced'].sum())}/{len(block)}"
                           if "reproduced" in block and block["reproduced"].notna().any() else "n/a"),
            "tvd_circuits": len(narrow),
            "mean_rel_tvd_pct": float(rel_tvd.mean()) if len(narrow) else math.nan,
            "median_rel_tvd_pct": float(rel_tvd.median()) if len(narrow) else math.nan,
            "tvd_better_sig": int(((narrow.get("d_tvd_ci_hi", pd.Series(dtype=float)) < 0)).sum())
            if len(narrow) else 0,
            "mean_rel_lsr_pct": float(rel_lsr.mean()),
            "mean_rel_lsr_pct_with_check": float(rel_lsr[block["checks"] > 0].mean())
            if (block["checks"] > 0).any() else math.nan,
            "mean_rel_silent_pct": float(rel_silent.mean()) if len(with_check) else math.nan,
            "noiseless_check_fired": int(block["noiseless_check_fired"].sum()),
            "random_input_check_fired": int(block.get("random_input_check_fired",
                                                      pd.Series([0])).fillna(0).sum()),
        })
    return pd.DataFrame(rows)


ARM_LABEL = {"baseline": "Baseline", "v4": "AQORN (v4)", "w10": "AQORN w10",
             "w10T": "AQORN w10T", "baseline_tH": "Baseline, targeted solo H",
             "w10T_tH": "w10T, targeted solo H", "baseline_fix": "Baseline, spare fix-up H",
             "w10T_fix": "w10T, spare fix-up H", "v4_tH": "v4, targeted solo H",
             "v4_fix": "v4, spare fix-up H"}


def plot(frame: pd.DataFrame, output_dir: Optional[Path], arms: Sequence[str],
         stem: str = "solo_gauge_paired") -> List[Path]:
    """
    Per circuit, the paired change from [[4,2,2]] to [[4,1,2]]-on-solo-patches decoding:
    (a) TVD (n <= 18), (b) silent error per kept shot (frames, every width), (c) LSR.
    Bars for circuits without a valid check are exactly zero and drawn as a hollow tick.
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    frame = frame[frame["arm"].isin(arms)].copy()
    order = (frame.drop_duplicates("name").sort_values(["num_qubits", "name"])["name"].tolist())
    narrow = [n for n in order if int(frame.loc[frame["name"] == n, "num_qubits"].iloc[0])
              <= TVD_MAX_QUBITS]
    colors = ["#0097fd", "#3b4c56", "#cdc2c0", "#636363", "#bdbdbd", "#0097fd"]
    hatches = [None, None, None, None, None, "////"]
    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH, 6.6),
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    width = 0.8 / len(arms)

    def panel(ax, names, value, err=None, label=""):
        x = np.arange(len(names))
        for i, arm in enumerate(arms):
            block = frame[frame["arm"] == arm].set_index("name").reindex(names)
            v = value(block)
            e = err(block) if err is not None else None
            has = (block["checks"] > 0).to_numpy()
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos[has], v[has], width=width * 0.92, color=colors[i], hatch=hatches[i],
                   edgecolor="#3b4c56" if hatches[i] else colors[i], linewidth=0.4,
                   label=ARM_LABEL.get(arm, arm),
                   yerr=None if e is None else e[has],
                   error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#3b4c56"})
            ax.plot(pos[~has], np.zeros((~has).sum()), linestyle="none", marker="x",
                    markersize=3.5, color=colors[i] if colors[i] != "#bdbdbd" else "#636363",
                    markeredgewidth=0.8)
        ax.axhline(0, color="#636363", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel(label)

    def rel(a, b):
        return lambda block: (100 * (block[a] - block[b]) / block[b]).to_numpy(dtype=float)

    panel(axes[0], narrow, rel("tvd_412", "tvd_422"),
          err=lambda block: (100 * block.get("d_tvd_boot_sd", np.nan) / block["tvd_422"]).to_numpy(dtype=float),
          label="TVD change (%)")
    panel(axes[1], order, rel("silent_per_kept_412", "silent_per_kept_422"),
          label="silent / kept\nchange (%)")
    panel(axes[2], order, rel("lsr_412", "lsr_422"), label="LSR change (%)")
    for ax, tag in zip(axes, "abc"):
        ax.text(-0.075, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold",
                fontsize=10, va="bottom")
    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="x", linestyle="none", color="#636363",
                          markersize=3.5, markeredgewidth=0.8))
    labels.append("no valid check (exactly 0)")
    axes[0].legend(handles, labels, ncol=len(arms) + 1, loc="lower left", fontsize=8,
                   bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.0)
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def availability_plot(frame: pd.DataFrame, output_dir: Optional[Path], arms: Sequence[str],
                      stem: str = "solo_gauge_availability") -> List[Path]:
    """How often a solo patch's gauge check survives translation, per arm."""
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    labels, valid, xframe, ent = [], [], [], []
    for arm in arms:
        block = frame[frame["arm"] == arm]
        labels.append(ARM_LABEL.get(arm, arm).replace(", ", ",\n"))
        valid.append(int(block["cand_valid"].sum()))
        xframe.append(int(block["cand_x_frame"].sum()))
        ent.append(int(block["cand_entangled"].sum()))
    x = np.arange(len(arms))
    ax.bar(x, valid, color="#0097fd", label="check valid")
    ax.bar(x, xframe, bottom=valid, color="#cdc2c0", label="spare in X frame (odd HH)")
    ax.bar(x, ent, bottom=np.add(valid, xframe), color="#3b4c56", label="spare entangled")
    for xi, v, t in zip(x, valid, np.add(np.add(valid, xframe), ent)):
        ax.text(xi, t + 0.5, f"{v}/{t}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("solo patches")
    ax.set_ylim(0, max(np.add(np.add(valid, xframe), ent)) * 1.25)
    ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.28))
    paths = save_figure(fig, stem, Path(__file__), override=output_dir, versioned=True)
    plt.close(fig)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--arms", nargs="*", default=list(BASE_ARMS),
                        help="baseline, v4, w10, w10T, sparse (one qubit per patch)")
    parser.add_argument("--variants", nargs="*", default=[],
                        help="extra arms like baseline_tH, w10T_fix (opt-in switches)")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--stem", default="circuit_level")
    parser.add_argument("--max-qubits", type=int, default=0,
                        help="every Fig 6 circuit up to this width (density panel), not "
                             "only the odd widths")
    parser.add_argument("--frame-shots", type=int, default=100_000)
    parser.add_argument("--boot-draws", type=int, default=200)
    parser.add_argument("--library-check-max-qubits", type=int, default=40)
    parser.add_argument("--from-csv", default=None, help="Redraw from a cached CSV.")
    parser.add_argument("--output-dir", default=None,
                        help="Figure directory (default: the paper's results/figures/).")
    parser.add_argument("--plot-arms", nargs="+", default=None)
    parser.add_argument("--skip-availability", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    csv = Path(args.from_csv) if args.from_csv else compute(args)
    frame = pd.read_csv(csv)
    if "reproduced" in frame and (frame["reproduced"] == False).any():  # noqa: E712
        bad = frame.loc[frame["reproduced"] == False, ["name", "arm"]].values.tolist()  # noqa: E712
        raise SystemExit(f"[{NAME}] [[4,2,2]] arm did not reproduce its source CSV: {bad}")
    if (frame["noiseless_check_fired"] > 0).any():
        raise SystemExit(f"[{NAME}] a valid check fired on a noiseless run")
    summary = summarize(frame)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    output_dir = Path(args.output_dir) if args.output_dir else None
    arms = args.plot_arms or [a for a in BASE_ARMS if a in set(frame["arm"])]
    paths = plot(frame, output_dir, arms) if arms else []
    if not args.skip_availability:  # `options.py` draws the version that includes variants
        paths += availability_plot(frame, output_dir, [a for a in frame["arm"].unique()])
    for path in paths:
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
