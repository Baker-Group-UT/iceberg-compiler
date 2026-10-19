"""
ablation_silent_aware.py -- Fig 6 (v4) regenerated with the silent-error-aware packing stage.

Fig 6 of the submitted paper (v4, `merging_improvement_benchmark_stylized_simulated`) stacks
three cumulative stages on the naive baseline -- H commutation, + merging, + packing -- over
71 QASMBench circuits. This reruns that figure's own per-circuit pipeline with one change:
the packing stage is planned by `experiments.revision.exp8_undetected.cost_model.
plan_silent_aware` (AQORN's planner plus `silent_weight` x expected silent logical errors)
instead of `plan_packing`. The planner also decides H commutation, so the +H and +merge
stages follow its decision, as they followed AQORN's in Fig 6.

Everything else is Fig 6's, reused rather than re-derived:

  * the 71 circuits with their per-size-group indices (`load_fig6_circuits`), so seeds match;
  * each circuit's calibrated alpha and shot count and the baseline arm's LSR/TVD, read from
    the submitted figure's CSV -- they depend only on the baseline arm and the two pilots,
    which no planner touches;
  * Fig 6's translate configs, `evaluate_variant` (`run_specific_scan`) and seed derivation
    `seed * 1_000_003 + circuit_idx * 17 + offset` (4: +H, 5: +H+merge, 6: packed).

Where the updated planner returns Fig 6's own plan, an arm is identical by construction and
Fig 6's value is reused; where it changes the placement or the commutation decision, the
affected arms are re-simulated with Fig 6's seeds -- what rerunning the Fig 6 script with the
new planner would draw. Checked on every run: the recomputed gate counts and depth of AQORN's
arms must reproduce Fig 6's percentages (`fig6_static_reproduced`).

Outputs, cached next to this script:
    ablation_silent_aware_w<weight>.csv               Fig 6's schema, three rows per circuit
    ablation_silent_aware_w<weight>_versus_aqorn.csv  raw AQORN vs silent-aware values per
                                                      circuit, read by `silent_aware_vs_aqorn`
and the figure as SVG + PDF in results/figures/.

Deviation from house style, inherited on purpose: Fig 6 draws on a 22 in canvas with its own
size ladder (see `AXIS_PT` in the Fig 6 script for why). This keeps Fig 6's plotting code so
the regenerated figure can replace Fig 6 one for one.

Run:
    PYTHONPATH=. venv/bin/python -m experiments.figures.ablation_silent_aware.ablation_silent_aware
    ... --from-csv experiments/figures/ablation_silent_aware/ablation_silent_aware_w3.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import math
import shutil
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "ablation_silent_aware"

#: Fig 6's per-row metric columns, in its order.
STATIC = ("pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates",
          "pct_reduction_circuit_depth")
LSR = "pct_improvement_logical_selection_rate"
TVD = "pct_reduction_logical_total_variation_distance"
#: Stage seed offsets in Fig 6's derivation.
OFFSET = {"h_only": 4, "h_merge": 5, "packed": 6}


@dataclass
class Settings:
    silent_weight: float = 3.0
    #: Experiment8a's plans (weight 0 == shipped AQORN, checked there on all 71 circuits).
    #: A weight missing from it is planned here instead.
    plans_csv: str = "results/revision/exp8/planner_static.csv"
    names: Tuple[str, ...] = ()
    jobs: Optional[int] = None
    output_dir: Optional[str] = None
    persist: bool = False
    from_csv: Optional[str] = None
    skip_gate: bool = False
    stem: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.stem = f"{NAME}_w{self.silent_weight:g}"


def _plans(settings: Settings, names) -> Dict[str, Dict[str, Tuple[list, bool]]]:
    """{circuit: {"aqorn": (placement, commute), "silent": ...}} from Experiment8a, if there."""
    path = Path(settings.plans_csv)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out: Dict[str, Dict[str, Tuple[list, bool]]] = {}
    for name, block in frame.groupby("name"):
        if name not in names:
            continue
        by_weight = {float(r.silent_weight): (json.loads(r.placement), bool(r.commute))
                     for r in block.itertuples()}
        if 0.0 in by_weight and settings.silent_weight in by_weight:
            out[name] = {"aqorn": by_weight[0.0], "silent": by_weight[settings.silent_weight]}
    return out


def _plan_here(program, weight: float) -> Tuple[list, bool]:
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware
    from experiments.revision.exp8_undetected.gadgets import load_rates

    plan = plan_silent_aware(program, 4, rates=load_rates(4), silent_weight=weight,
                             noise=fig6_noise_profile())
    return [list(p) for p in plan.placement], bool(plan.h_commutation)


def _run_circuit(job) -> Dict[str, List[Dict]]:
    """One circuit: Fig 6's stages under the updated planner, plus AQORN's arm for figure 2."""
    from pipeline.config import PipelineConfig

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    settings_dict, circuit, reference_rows, plans = job
    weight = float(settings_dict["silent_weight"])
    s = fig6.BenchmarkSettings()
    resource = fig6.ensure_measurements(circuit.original)
    simulation = fig6.ensure_measurements(circuit.proxy)
    n = resource.num_qubits
    tight = fig6.tight_trivial_packing(n, math.ceil(n / 2))
    ref = {row["component"]: row for row in reference_rows}
    first = reference_rows[0]
    alpha, reps, idx = float(first["alpha_final"]), int(first["final_repetitions"]), int(first["circuit_idx"])

    if plans is None:
        from packing.plan import plan_packing

        shipped = plan_packing(resource, 4, noise=_fig6_profile(), verify=True)
        plans = {"aqorn": ([list(p) for p in shipped.placement], bool(shipped.h_commutation)),
                 "silent": _plan_here(resource, weight)}
    aqorn_perm, aqorn_commute = [tuple(p) for p in plans["aqorn"][0]], plans["aqorn"][1]
    new_perm, new_commute = [tuple(p) for p in plans["silent"][0]], plans["silent"][1]
    plan_changed = (new_perm, new_commute) != (aqorn_perm, aqorn_commute)

    def config(commute: bool, merge: bool) -> PipelineConfig:
        cfg = PipelineConfig()
        cfg.translator_config.enable_h_commutation = commute
        cfg.translator_config.enable_merging = merge
        return cfg

    def counts(perm, commute, merge):
        return fig6.compute_translation_metrics(resource, perm, config(commute, merge))

    # --- static: Fig 6's own translations, under each planner's commutation decision ---
    base = counts(tight, False, False)
    arms = {"aqorn": {}, "silent": {}}
    for key, (perm, commute) in (("aqorn", (aqorn_perm, aqorn_commute)),
                                 ("silent", (new_perm, new_commute))):
        if key == "silent" and not plan_changed:
            arms["silent"] = arms["aqorn"]
            continue
        arms[key] = {"h_only": counts(tight, commute, False),
                     "h_merge": counts(tight, commute, True),
                     "packed": counts(perm, commute, True)}
        if key == "silent" and new_commute == aqorn_commute:
            arms["silent"]["h_only"], arms["silent"]["h_merge"] = (
                arms["aqorn"]["h_only"], arms["aqorn"]["h_merge"])

    def static_pct(values) -> List[float]:
        return [fig6.pct_reduction(b, v) for b, v in zip(base, values)]

    # The reproduction check: AQORN's arms, recomputed, against the submitted CSV.
    expected = {
        "h_only": [ref["h_commutation"][c] for c in STATIC],
        "h_merge": [ref["h_commutation"][c] + ref["merging"][c] for c in STATIC],
        "packed": [ref["h_commutation"][c] + ref["merging"][c] + ref["packing"][c] for c in STATIC],
    }
    reproduced = all(abs(a - b) < 1e-6 for stage in expected
                     for a, b in zip(static_pct(arms["aqorn"][stage]), expected[stage]))

    # --- simulated: Fig 6's draws where the arm is Fig 6's, Fig 6's seeds where it is not ---
    base_lsr, base_tvd = float(first["baseline_lsr"]), float(first["baseline_tvd"])

    def from_fig6(components) -> Dict[str, float]:
        lsr = sum(ref[c][LSR] for c in components)
        tvd = sum(ref[c][TVD] for c in components)
        return {"lsr": base_lsr * (1 + lsr / 100.0), "tvd": base_tvd * (1 - tvd / 100.0)}

    def simulate(perm, commute, merge, stage) -> Dict[str, float]:
        cfg = fig6.build_pipeline_config(
            s, alpha=alpha, repetitions=reps, seed=s.seed * 1_000_003 + idx * 17 + OFFSET[stage],
            enable_h_commutation=commute, enable_merging=merge)
        out = fig6.evaluate_variant(simulation, list(perm), cfg)
        return {"lsr": out["logical_selection_rate"], "tvd": out["logical_total_variation_distance"]}

    aqorn_sim = {"h_only": from_fig6(["h_commutation"]),
                 "h_merge": from_fig6(["h_commutation", "merging"]),
                 "packed": from_fig6(["h_commutation", "merging", "packing"])}
    began = time.time()
    if new_commute == aqorn_commute:
        h_source = "fig6"
        silent_sim = {"h_only": aqorn_sim["h_only"], "h_merge": aqorn_sim["h_merge"]}
    else:
        h_source = "resimulated"
        silent_sim = {"h_only": simulate(tight, new_commute, False, "h_only"),
                      "h_merge": simulate(tight, new_commute, True, "h_merge")}
    if plan_changed:
        packed_source = "resimulated"
        silent_sim["packed"] = simulate(new_perm, new_commute, True, "packed")
    else:
        packed_source = "fig6"
        silent_sim["packed"] = aqorn_sim["packed"]
    sim_seconds = time.time() - began

    # --- Fig 6's rows: cumulative percentages, differenced into stages ---
    def cumulative(stage) -> Dict[str, float]:
        pct = dict(zip(STATIC, static_pct(arms["silent"][stage])))
        pct[LSR] = fig6.pct_improvement(base_lsr, silent_sim[stage]["lsr"])
        pct[TVD] = fig6.pct_reduction(base_tvd, silent_sim[stage]["tvd"])
        return pct

    stages = {"h_commutation": cumulative("h_only"), "merging": cumulative("h_merge"),
              "packing": cumulative("packed")}
    previous = {key: 0.0 for key in stages["h_commutation"]}
    shared = {key: first[key] for key in (
        "size_group", "circuit_idx", "circuit_name", "num_qubits", "target_codes",
        "baseline_gate_count", "alpha_initial", "alpha_final", "pilot_initial_lsr",
        "pilot_final_lsr", "final_repetitions", "baseline_lsr", "baseline_tvd")}
    extra = {"silent_weight": weight, "commute": new_commute, "commute_aqorn": aqorn_commute,
             "plan_changed": plan_changed, "h_source": h_source,
             "packed_source": packed_source, "fig6_static_reproduced": reproduced,
             "sim_seconds": sim_seconds, "placement": json.dumps([list(p) for p in new_perm])}
    rows = []
    for component, values in stages.items():
        rows.append({**shared, "component": component,
                     **{key: values[key] - previous[key] for key in values}, **extra})
        previous = values

    labels = ("one_qubit_gates", "two_qubit_gates", "circuit_depth")
    versus = {**{k: shared[k] for k in ("size_group", "circuit_idx", "circuit_name",
                                         "num_qubits", "baseline_tvd", "final_repetitions",
                                         "alpha_final")},
              "silent_weight": weight, "plan_changed": plan_changed,
              "commute": new_commute, "commute_aqorn": aqorn_commute,
              "fig6_static_reproduced": reproduced}
    for key, arm, sim in (("aqorn", arms["aqorn"]["packed"], aqorn_sim["packed"]),
                          ("silent", arms["silent"]["packed"], silent_sim["packed"])):
        versus.update({f"{key}_{label}": float(v) for label, v in zip(labels, arm)})
        versus[f"{key}_lsr"] = sim["lsr"]
        versus[f"{key}_tvd"] = sim["tvd"]
    return {"rows": rows, "versus": [versus]}


def _fig6_profile():
    from experiments.revision.common import fig6_noise_profile

    return fig6_noise_profile()


#: Small circuits whose AQORN packed arm is re-simulated before the sweep and must match
#: Fig 6's sample exactly: the check that this environment still draws Fig 6's numbers.
GATE_CIRCUITS = ("adder_n10", "bell_n4")


def _gate_point(job) -> Tuple[str, float, float]:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    circuit, reference_rows, plans = job
    first = reference_rows[0]
    s = fig6.BenchmarkSettings()
    cfg = fig6.build_pipeline_config(
        s, alpha=float(first["alpha_final"]), repetitions=int(first["final_repetitions"]),
        seed=s.seed * 1_000_003 + int(first["circuit_idx"]) * 17 + OFFSET["packed"],
        enable_h_commutation=plans["aqorn"][1], enable_merging=True)
    got = fig6.evaluate_variant(fig6.ensure_measurements(circuit.proxy),
                                [tuple(p) for p in plans["aqorn"][0]], cfg)
    expected = float(first["baseline_lsr"]) * (1 + sum(
        row[LSR] for row in reference_rows) / 100.0)
    return circuit.name, float(got["logical_selection_rate"]), expected


def gate(settings: Settings) -> None:
    from experiments.revision.common import FIG6_CSV, load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    fig6_rows = pd.read_csv(FIG6_CSV)
    circuits = load_fig6_circuits(names=GATE_CIRCUITS)
    plans = _plans(settings, set(GATE_CIRCUITS))
    jobs = [(c, fig6_rows[fig6_rows["circuit_name"] == c.name].to_dict("records"),
             plans[c.name]) for c in circuits if c.name in plans]
    results = parallel_map(_gate_point, jobs, jobs=len(jobs), label=f"{NAME}-gate", quiet=True)
    for result in results:
        if result is None or abs(result[1] - result[2]) > 1e-9:
            raise SystemExit(f"[{NAME}] GATE FAILED: AQORN's packed arm did not reproduce "
                             f"Fig 6 ({result}); refusing to draw a mixed figure")
        print(f"[{NAME}] gate: {result[0]} AQORN packed LSR {result[1]:.6f} == Fig 6 {result[2]:.6f}")


def compute(settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from experiments.revision.common import FIG6_CSV, load_fig6_circuits
    from experiments.shared.figures import stamp_provenance
    from experiments.shared.parallel import parallel_map

    fig6_rows = pd.read_csv(FIG6_CSV)
    names = list(settings.names) or sorted(fig6_rows["circuit_name"].unique())
    circuits = load_fig6_circuits(names=names)
    plans = _plans(settings, set(names))
    jobs = [(asdict(settings), c,
             fig6_rows[fig6_rows["circuit_name"] == c.name].to_dict("records"),
             plans.get(c.name)) for c in circuits]
    # Widest first: their translations and shots dominate, so start them early.
    jobs.sort(key=lambda j: -j[1].num_qubits)
    print(f"[{NAME}] {len(jobs)} circuits, weight {settings.silent_weight:g}; plans from "
          f"Experiment8a for {sum(j[3] is not None for j in jobs)}")
    results = parallel_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME)
    failed = [j[1].name for j, r in zip(jobs, results) if r is None]
    if failed:
        raise SystemExit(f"[{NAME}] {len(failed)} circuits failed: {failed}")
    rows = stamp_provenance(pd.DataFrame([row for r in results for row in r["rows"]]))
    versus = pd.DataFrame([row for r in results for row in r["versus"]])
    order = {"small": 0, "medium": 1, "large": 2}
    rows = rows.sort_values(["size_group", "circuit_idx", "component"],
                            key=lambda col: col.map(order) if col.name == "size_group" else col)
    return rows, versus


def plot(rows: pd.DataFrame, settings: Settings) -> List[Path]:
    """Fig 6's own `plot_results`, then the packing label and the TVD-withheld markers."""
    import matplotlib.pyplot as plt

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.shared.figures import save_figure

    with tempfile.TemporaryDirectory() as scratch:
        fig6_settings = fig6.BenchmarkSettings(output_path=Path(scratch) / "draft.svg",
                                               show_plot=True)
        fig6.plot_results(rows, fig6_settings)  # Agg: show() is a no-op, the figure stays open
    fig = plt.gcf()
    axes = fig.axes
    legend = axes[0].get_legend()
    for text in legend.get_texts():
        if text.get_text() == "+ Packing benefit":
            text.set_text(f"+ Packing benefit (silent-aware, weight {settings.silent_weight:g})")
    # Fig 6's caption marks the circuits withheld from TVD with a star; mark them here too.
    withheld = set(rows.loc[~fig6._tvd_is_resolved(rows), "circuit_name"])
    tvd_axis = axes[-1]
    top = tvd_axis.get_ylim()[1]
    for x, label in zip(tvd_axis.get_xticks(minor=True), tvd_axis.get_xticklabels(minor=True)):
        if label.get_text() in withheld:
            tvd_axis.plot([x], [0.06 * top], marker="*", markersize=14, color="#3b4c56",
                          linestyle="none", clip_on=False)
    paths = save_figure(fig, settings.stem, Path(__file__),
                        override=Path(settings.output_dir) if settings.output_dir else None,
                        versioned=True)
    plt.close(fig)
    return paths


def summarize(rows: pd.DataFrame) -> None:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    fig6.print_metric_averages(rows)
    per = rows.drop_duplicates("circuit_name")
    print(f"[{NAME}] plan changed on {int(per['plan_changed'].sum())}/{len(per)} circuits "
          f"(commutation on {int((per['commute'] != per['commute_aqorn']).sum())}); AQORN's "
          f"static arms reproduce Fig 6 on {int(per['fig6_static_reproduced'].sum())}/{len(per)}")


def run(settings: Settings) -> None:
    from experiments.revision.common import write_sidecar
    from experiments.shared.figures import data_dir

    if settings.from_csv:
        rows = pd.read_csv(settings.from_csv)
    else:
        began = time.time()
        if not settings.skip_gate:
            gate(settings)
        rows, versus = compute(settings)
        csv = HERE / f"{settings.stem}.csv"
        rows.to_csv(csv, index=False)
        versus.to_csv(HERE / f"{settings.stem}_versus_aqorn.csv", index=False)
        write_sidecar(csv, experiment=NAME, settings=asdict(settings), started=began,
                      status="complete")
        if settings.persist:
            target = data_dir(Path(__file__))
            for path in (csv, HERE / f"{settings.stem}_versus_aqorn.csv"):
                shutil.copy2(path, target / path.name)
        print(f"[{NAME}] wrote {csv}")
    summarize(rows)
    for path in plot(rows, settings):
        print(f"[{NAME}] saved {path}")


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--silent-weight", type=float, default=3.0)
    parser.add_argument("--plans-csv", default="results/revision/exp8/planner_static.csv")
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the cached CSVs to results/data/ (they take ~15 min "
                             "of simulation to rebuild).")
    parser.add_argument("--from-csv", default=None,
                        help="Redraw from a cached CSV without recomputing.")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()
    return Settings(silent_weight=args.silent_weight, plans_csv=args.plans_csv,
                    names=tuple(args.names), jobs=args.jobs, output_dir=args.output_dir,
                    persist=args.persist, from_csv=args.from_csv, skip_gate=args.skip_gate)


if __name__ == "__main__":
    run(parse_args())
