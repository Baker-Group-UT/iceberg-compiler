"""
ablation_standard.py -- Fig 6 under the standard configuration (w10T), every panel re-simulated.

Fig 6 of the submitted paper (`merging_improvement_benchmark_stylized_simulated`) stacks three
cumulative stages on a naive baseline -- H commutation, + merging, + packing -- over 71
QASMBench circuits. Since 2026-10-07 AQORN's standard configuration is
`packing.standard.STANDARD` ("w10T"). This reruns Fig 6's own per-circuit procedure end to
end -- four translations, the two-pilot alpha calibration, four simulated arms -- under it:

    arm         placement       H commutation   merging   in-patch CZ / XCX      order
    baseline    tight trivial   off             off       3-gate gadget          layer-sorted
    + H         tight trivial   plan's          off       3-gate gadget          layer-sorted
    + merging   tight trivial   plan's          on        transversal (S, SX)    layer-sorted
    + packing   the plan        plan's          on        transversal (S, SX)    layer-sorted

  * The plan is `packing.standard.plan_standard` (weight-10 silent-aware planner, Fig 6's
    noise profile). The recorded w10T plans (`results/tvd_explore/plans_w10T.csv`) are used
    when present -- they equal `plan_standard` on all 71 circuits, and the gate re-checks
    that on its circuits every run -- and any missing plan is made here.
  * The transversal in-patch gadget is credited to the merging stage, as it was measured: it
    is a construction AQORN's compiler chooses, not one the baseline compiler had.
  * Every arm, baseline included, is layer-sorted before translation
    (`TranslatorConfig.asap_order`), so a stage is credited with what its pass does, not
    with the order it emits gates in (the confound `ablation_fair_depth` measured).
  * The baseline changed, so alpha and the shot count are recalibrated on it with Fig 6's
    pilots and pilot seeds, and "alpha calibrated to a baseline LSR of 6%" stays true.

Everything else is Fig 6's, reused rather than re-derived: the circuits with their per-group
indices (`load_fig6_circuits`), `BenchmarkSettings` (target LSR, pilot and kept-shot counts,
Stim with distance and idle noise at scalar 1), `evaluate_variant`, and the seed derivation
`seed * 1_000_003 + circuit_idx * 17 + offset` (1, 2 pilots; 3 baseline; 4 +H; 5 +merge;
6 packed).

Gate, before the sweep: the same code under the submitted configuration (AQORN's shipped
plans from Experiment8a, program order, the 3-gate gadget) must reproduce the submitted
Fig 6 rows -- calibration, baseline and all five panels -- exactly on two small circuits:
the check that this procedure is Fig 6's and that this environment still draws Fig 6's
samples. `--configuration submitted` runs that configuration on every circuit. The gate
always runs with the logical optimisation pass off (the submitted figure predates it).

Logical optimisation (`translator.logical_optimization`, on by default since 2026-10-09):
with the pass on, every circuit is the optimised program (the loader optimises it before
making its proxy; Fig 6's 71 are the logical-optimisation study's verified circuits,
`experiments/logical_opt/seed_records.py`), every arm -- baseline included -- translates
that program, and every circuit is planned here with `plan_standard` on it: the recorded
w10T plans were made on the unoptimised programs, so they are not used and the recorded-plan
gate is skipped (pass `--plans-csv` to force a plans file made under the pass). Outputs then
go under the stem `ablation_standard_logical`. AQORN_LOGICAL_OPT=0 is the run as before.

Outputs: `ablation_standard[_vN].csv` next to this script (Fig 6's schema, three rows per
circuit, plus every arm's raw counts and simulated values), never overwritten; `--persist`
also copies it to results/data/ (about 35 min of simulation to rebuild). The figure goes to
results/figures/ under the same stem, drawn by Fig 6's own `plot_results`: its 22 in canvas
and size ladder are inherited on purpose, so this replaces Fig 6 one for one.

    PYTHONPATH=. venv/bin/python -m experiments.figures.ablation_standard.ablation_standard --persist
    ... --from-csv experiments/figures/ablation_standard/ablation_standard.csv
    ... --from-csv <csv> --compare results/tvd_explore/ablation_variant_w10T.csv
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "ablation_standard"
COMPONENTS = ("h_commutation", "merging", "packing")
STATIC = ("pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates",
          "pct_reduction_circuit_depth")
LSR = "pct_improvement_logical_selection_rate"
TVD = "pct_reduction_logical_total_variation_distance"
METRIC_COLUMNS = STATIC + (LSR, TVD)
CALIBRATION = ("baseline_gate_count", "alpha_initial", "alpha_final", "pilot_initial_lsr",
               "pilot_final_lsr", "final_repetitions", "baseline_lsr", "baseline_tvd")
#: Arm -> Fig 6's seed offset. The pilots use offsets 1 and 2 on the baseline arm.
ARMS = {"baseline": 3, "h_only": 4, "h_merge": 5, "packed": 6}
#: The arm whose cumulative values each component row ends at.
ARM_OF = {"h_commutation": "h_only", "merging": "h_merge", "packing": "packed"}

STANDARD_PLANS = "results/tvd_explore/plans_w10T.csv"
SUBMITTED_PLANS = "results/revision/exp8/planner_static.csv"     # weight 0 = shipped AQORN
GATE_CIRCUITS = ("adder_n10", "bell_n4")


@dataclass(frozen=True)
class Configuration:
    name: str
    #: Every arm layer-sorted before translation (`TranslatorConfig.asap_order`).
    asap_order: bool
    #: Transversal in-patch CZ / XCX on the merging-enabled arms.
    transversal_merging: bool


def configurations() -> Dict[str, Configuration]:
    from packing.standard import STANDARD

    return {
        "standard": Configuration("standard", asap_order=STANDARD.asap_order,
                                  transversal_merging=STANDARD.transversal_intra_two_qubit),
        "submitted": Configuration("submitted", asap_order=False, transversal_merging=False),
    }


def load_plans(configuration: str, path: Optional[str] = None) -> Dict[str, Tuple[list, bool]]:
    """
    {circuit: (placement, commute)}: recorded plans for this configuration, if any. None for
    the standard under the logical optimisation pass unless `path` names a file: the recorded
    plans were made on the unoptimised programs.
    """
    from translator.logical_optimization import enabled

    if configuration == "standard" and path is None and enabled():
        return {}
    if configuration == "submitted":
        frame = pd.read_csv(path or SUBMITTED_PLANS)
        frame = frame[frame["silent_weight"] == 0.0]
    else:
        source = Path(path or STANDARD_PLANS)
        if not source.exists():
            return {}
        frame = pd.read_csv(source)
    return {r.name: (json.loads(r.placement), bool(r.commute)) for r in frame.itertuples()}


def plan_here(program, configuration: Configuration) -> Tuple[list, bool]:
    """The configuration's planner on Fig 6's resource circuit."""
    if configuration.name == "submitted":
        from experiments.figures.merging_improvement.common import pack_program
        from experiments.revision.common import fig6_noise_profile

        placement, commute = pack_program(program, noise=fig6_noise_profile())
        return [list(p) for p in placement], bool(commute)
    from packing.standard import plan_standard

    plan = plan_standard(program)
    return [list(p) for p in plan.placement], bool(plan.h_commutation)


def _run_circuit(job) -> List[Dict]:
    """One circuit: Fig 6's four translations, two pilots and four simulations."""
    from pipeline.config import PipelineConfig

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    configuration, circuit, plan = job
    s = fig6.BenchmarkSettings()
    resource = fig6.ensure_measurements(circuit.original)
    simulation = fig6.ensure_measurements(circuit.proxy)
    n = resource.num_qubits
    target_codes = math.ceil(n / 2)
    tight = fig6.tight_trivial_packing(n, target_codes)
    plan_source = "recorded"
    if plan is None:
        plan, plan_source = plan_here(resource, configuration), "planned"
    placement, commute = [tuple(p) for p in plan[0]], bool(plan[1])
    arms = {"baseline": (tight, False, False), "h_only": (tight, commute, False),
            "h_merge": (tight, commute, True), "packed": (placement, commute, True)}

    def configure(config, commute_: bool, merge: bool):
        translator = config.translator_config
        translator.enable_h_commutation = commute_
        translator.enable_merging = merge
        translator.asap_order = configuration.asap_order
        translator.transversal_intra_two_qubit = configuration.transversal_merging and merge
        return config

    static = {arm: fig6.compute_translation_metrics(resource, perm, configure(PipelineConfig(), c, m))
              for arm, (perm, c, m) in arms.items()}

    def simulate(arm: str, alpha: float, repetitions: int, offset: int) -> Dict[str, float]:
        perm, c, m = arms[arm]
        config = fig6.build_pipeline_config(
            s, alpha=alpha, repetitions=repetitions,
            seed=s.seed * 1_000_003 + circuit.circuit_idx * 17 + offset,
            enable_h_commutation=c, enable_merging=m)
        return fig6.evaluate_variant(simulation, list(perm), configure(config, c, m))

    began = time.time()
    baseline_gate_count = max(1, static["baseline"][0] + static["baseline"][1])
    alpha_initial = fig6.calibrated_alpha_from_gate_count(baseline_gate_count, s)
    pilot_initial = simulate("baseline", alpha_initial, s.pilot_repetitions, 1)
    alpha_final = fig6.corrected_alpha(alpha_initial, pilot_initial["logical_selection_rate"],
                                       s.target_selection_rate)
    pilot_final = simulate("baseline", alpha_final, s.pilot_repetitions, 2)
    repetitions = fig6.repetitions_from_selection_rate(pilot_final["logical_selection_rate"], s)
    sims = {arm: simulate(arm, alpha_final, repetitions, offset) for arm, offset in ARMS.items()}
    sim_seconds = time.time() - began

    base_lsr = sims["baseline"]["logical_selection_rate"]
    base_tvd = sims["baseline"]["logical_total_variation_distance"]

    def cumulative(arm: str) -> Dict[str, float]:
        out = {column: fig6.pct_reduction(b, v)
               for column, b, v in zip(STATIC, static["baseline"], static[arm])}
        out[LSR] = fig6.pct_improvement(base_lsr, sims[arm]["logical_selection_rate"])
        out[TVD] = fig6.pct_reduction(base_tvd, sims[arm]["logical_total_variation_distance"])
        return out

    shared = {"size_group": circuit.size_group, "circuit_idx": circuit.circuit_idx,
              "circuit_name": circuit.name, "num_qubits": n, "target_codes": target_codes,
              "baseline_gate_count": baseline_gate_count, "alpha_initial": alpha_initial,
              "alpha_final": alpha_final,
              "pilot_initial_lsr": pilot_initial["logical_selection_rate"],
              "pilot_final_lsr": pilot_final["logical_selection_rate"],
              "final_repetitions": repetitions, "baseline_lsr": base_lsr,
              "baseline_tvd": base_tvd}
    extra = {"configuration": configuration.name, "asap_order": configuration.asap_order,
             "transversal_merging": configuration.transversal_merging, "commute": commute,
             "placement": json.dumps([list(p) for p in placement]), "plan_source": plan_source,
             "sim_seconds": sim_seconds}
    for arm in ARMS:
        one, two, depth = static[arm]
        extra.update({f"{arm}_one_qubit_gates": one, f"{arm}_two_qubit_gates": two,
                      f"{arm}_depth": depth, f"{arm}_lsr": sims[arm]["logical_selection_rate"],
                      f"{arm}_tvd": sims[arm]["logical_total_variation_distance"]})
    rows, previous = [], {column: 0.0 for column in METRIC_COLUMNS}
    for component in COMPONENTS:
        now = cumulative(ARM_OF[component])
        rows.append({**shared, "component": component,
                     **{column: now[column] - previous[column] for column in METRIC_COLUMNS},
                     **extra})
        previous = now
    return rows


def gate(plans_path: Optional[str]) -> None:
    """
    Stop unless (1) the submitted configuration reproduces Fig 6's rows exactly on the gate
    circuits and (2) the recorded standard plans, where used, equal `plan_standard` on them.
    """
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.revision.common import FIG6_CSV, load_fig6_circuits
    from experiments.shared.parallel import parallel_map
    from translator.logical_optimization import disabled, enabled

    configs = configurations()
    reference = pd.read_csv(FIG6_CSV)
    pass_on = enabled()
    # The submitted figure predates the logical optimisation pass: reproduce it without it,
    # loader and workers included (the switch is an environment variable, so they inherit it).
    with disabled():
        circuits = load_fig6_circuits(names=GATE_CIRCUITS)
        submitted = load_plans("submitted")
        jobs = [(configs["submitted"], c, submitted[c.name]) for c in circuits]
        results = parallel_map(_run_circuit, jobs, jobs=len(jobs), label=f"{NAME}-gate",
                               quiet=True)
    for circuit, rows in zip(circuits, results):
        if rows is None:
            raise SystemExit(f"[{NAME}] GATE FAILED: {circuit.name} raised")
        expected = reference[reference["circuit_name"] == circuit.name].set_index("component")
        for row in rows:
            for column in CALIBRATION + METRIC_COLUMNS:
                got, want = float(row[column]), float(expected.loc[row["component"], column])
                if abs(got - want) > 1e-9:
                    raise SystemExit(f"[{NAME}] GATE FAILED: {circuit.name} {row['component']} "
                                     f"{column} = {got!r}, Fig 6 has {want!r}")
        print(f"[{NAME}] gate: {circuit.name} reproduces Fig 6 exactly (calibration, "
              f"baseline, all five panels)")

    if pass_on and plans_path is None:
        print(f"[{NAME}] gate: logical optimisation on -- recorded plans not used, every "
              f"circuit is planned on its optimised program")
        return
    recorded = load_plans("standard", plans_path)
    for circuit in circuits:
        if circuit.name not in recorded:
            continue
        here = plan_here(fig6.ensure_measurements(circuit.original), configs["standard"])
        want = recorded[circuit.name]
        if [list(p) for p in want[0]] != here[0] or bool(want[1]) != here[1]:
            raise SystemExit(f"[{NAME}] GATE FAILED: recorded w10T plan for {circuit.name} "
                             f"is not plan_standard's ({want} vs {here})")
        print(f"[{NAME}] gate: recorded plan for {circuit.name} == plan_standard")


def compute(settings: "Settings") -> pd.DataFrame:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.figures import stamp_provenance
    from experiments.shared.parallel import parallel_map

    configuration = configurations()[settings.configuration]
    circuits = load_fig6_circuits(names=list(settings.names) or None)
    plans = load_plans(settings.configuration, settings.plans_csv)
    jobs = [(configuration, c, plans.get(c.name)) for c in circuits]
    jobs.sort(key=lambda j: -j[1].original.num_qubits)   # widest first: they dominate
    print(f"[{NAME}] {configuration}: {len(jobs)} circuits, recorded plans for "
          f"{sum(j[2] is not None for j in jobs)}")
    results = parallel_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME)
    failed = [j[1].name for j, r in zip(jobs, results) if r is None]
    if failed:
        raise SystemExit(f"[{NAME}] {len(failed)} circuits failed: {failed}")
    rows = pd.DataFrame([row for r in results for row in r])
    order = {"small": 0, "medium": 1, "large": 2}
    rows = rows.sort_values(["size_group", "circuit_idx", "component"],
                            key=lambda col: col.map(order) if col.name == "size_group" else col)
    return stamp_provenance(rows.reset_index(drop=True))


def caption_numbers(rows: pd.DataFrame, names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """
    The caption convention (`values.tex`): the mean over circuits of each circuit's
    cumulative value after every stage, unclipped; TVD over the circuits whose baseline TVD
    is resolved (Fig 6's `_tvd_is_resolved`). `names` restricts the circuits.
    """
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    if names is not None:
        rows = rows[rows["circuit_name"].isin(set(names))]
    resolved = fig6._tvd_is_resolved(rows)
    out: Dict[str, float] = {}
    for label, column in zip(("1q", "2q", "depth", "LSR", "TVD"), METRIC_COLUMNS):
        frame = rows.loc[resolved] if column == TVD else rows
        wide = frame.pivot_table(index="circuit_name", columns="component", values=column)
        totals = wide[list(COMPONENTS)].cumsum(axis=1).mean()
        for component in COMPONENTS:
            out[f"{label} +{component}"] = float(totals[component])
        out[label] = float(totals[COMPONENTS[-1]])
    out["circuits"] = int(rows["circuit_name"].nunique())
    out["TVD circuits"] = int(rows.loc[resolved, "circuit_name"].nunique())
    return out


def summarize(rows: pd.DataFrame) -> None:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    numbers = caption_numbers(rows)
    print(f"[{NAME}] caption convention, cumulative after each stage "
          f"({numbers['circuits']} circuits, TVD on {numbers['TVD circuits']}):")
    for label in ("1q", "2q", "depth", "LSR", "TVD"):
        stages = ", ".join(f"+{c} {numbers[f'{label} +{c}']:.1f}" for c in COMPONENTS)
        print(f"  {label:>5}: {stages}")
    for group in ("small", "medium", "large"):
        if not (rows["size_group"] == group).any():
            continue  # a --names subset need not reach every group
        sub = caption_numbers(rows[rows["size_group"] == group])
        print(f"  {group:>6} ({sub['circuits']}): " + ", ".join(
            f"{label} {sub[label]:.1f}" for label in ("1q", "2q", "depth", "LSR", "TVD")))
    per = rows.drop_duplicates("circuit_name")
    print(f"[{NAME}] alpha mean {per['alpha_final'].mean():.4f}, baseline LSR mean "
          f"{per['baseline_lsr'].mean():.4f}, shots mean {per['final_repetitions'].mean():.0f}; "
          f"plans {per['plan_source'].value_counts().to_dict()}")
    fig6.print_metric_averages(rows)


def _totals(rows: pd.DataFrame, column: str) -> pd.Series:
    """Each circuit's cumulative value after the last stage; TVD only where resolved."""
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    frame = rows.loc[fig6._tvd_is_resolved(rows)] if column == TVD else rows
    wide = frame.pivot_table(index="circuit_name", columns="component", values=column)
    return wide[list(COMPONENTS)].sum(axis=1)


def compare(rows: pd.DataFrame, others: Sequence[str], *, seed: int = 20261007) -> None:
    """
    Caption numbers of this run against other Fig 6-schema CSVs, on the circuits both have,
    with the per-circuit paired difference: its bootstrap 95% interval over circuits and the
    number of circuits on which it is not exactly zero (same arms, same seeds -> same draw).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    for path in others:
        other = pd.read_csv(path)
        common = sorted(set(rows["circuit_name"]) & set(other["circuit_name"]))
        mine, theirs = caption_numbers(rows, common), caption_numbers(other, common)
        print(f"[{NAME}] vs {path} ({len(common)} common circuits; TVD on "
              f"{mine['TVD circuits']} here, {theirs['TVD circuits']} there):")
        a_rows = rows[rows["circuit_name"].isin(common)]
        b_rows = other[other["circuit_name"].isin(common)]
        for label, column in zip(("1q", "2q", "depth", "LSR", "TVD"), METRIC_COLUMNS):
            a, b = _totals(a_rows, column), _totals(b_rows, column)
            both = a.index.intersection(b.index)
            diff = (a[both] - b[both]).to_numpy()
            means = diff[rng.integers(0, len(diff), size=(4000, len(diff)))].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            print(f"  {label:>5}: {theirs[label]:7.2f} -> {mine[label]:7.2f} "
                  f"({mine[label] - theirs[label]:+.2f}; paired {diff.mean():+.2f} "
                  f"[{low:+.2f}, {high:+.2f}], nonzero on {int((abs(diff) > 1e-9).sum())}/"
                  f"{len(diff)})")
        if "alpha_final" in other.columns:
            a = rows.drop_duplicates("circuit_name").set_index("circuit_name").loc[common]
            b = other.drop_duplicates("circuit_name").set_index("circuit_name").loc[common]
            moved = (abs(a["alpha_final"] / b["alpha_final"] - 1) > 1e-12)
            print(f"         alpha recalibrated to a different value on {int(moved.sum())}/"
                  f"{len(common)} circuits (median ratio there "
                  f"{(a['alpha_final'] / b['alpha_final'])[moved].median():.4f}); baseline LSR "
                  f"mean {a['baseline_lsr'].mean():.4f} here, {b['baseline_lsr'].mean():.4f} there")


#: Fig 6's panel labels, in drawing order, as `--panels` names them.
PANELS = ("1q", "2q", "depth", "LSR", "TVD")


def plot(rows: pd.DataFrame, stem: str, output_dir: Optional[str],
         panels: Tuple[str, ...] = PANELS, average_bar: str = "bars") -> List[Path]:
    """Fig 6's own `plot_results`, then the TVD-withheld stars its caption describes.

    `panels` keeps a subset of the rows (e.g. without LSR); the canvas height scales with it so
    every remaining row keeps its size."""
    import matplotlib.pyplot as plt

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.shared.figures import save_figure

    every = dict(zip(PANELS, fig6.METRICS.items()))
    kept = dict(every[name] for name in PANELS if name in panels)
    original = fig6.METRICS
    original_average = fig6.AVERAGE_BAR
    fig6.METRICS = kept           # `plot_results` reads the module's METRICS for its rows
    fig6.AVERAGE_BAR = average_bar
    try:
        with tempfile.TemporaryDirectory() as scratch:
            fig6_settings = fig6.BenchmarkSettings(output_path=Path(scratch) / "draft.svg",
                                                   show_plot=True)
            fig6.plot_results(rows, fig6_settings)  # Agg: show() is a no-op, figure stays open
    finally:
        fig6.METRICS = original
        fig6.AVERAGE_BAR = original_average
    fig = plt.gcf()
    if len(kept) != len(PANELS):
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height * len(kept) / len(PANELS))
        fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))   # as `plot_results` lays it out
    withheld = set(rows.loc[~fig6._tvd_is_resolved(rows), "circuit_name"])
    tvd_axis = fig.axes[-1]
    top = tvd_axis.get_ylim()[1]
    for x, label in zip(tvd_axis.get_xticks(minor=True), tvd_axis.get_xticklabels(minor=True)):
        if label.get_text() in withheld:
            tvd_axis.plot([x], [0.06 * top], marker="*", markersize=14, color="#3b4c56",
                          linestyle="none", clip_on=False)
    paths = save_figure(fig, stem, Path(__file__),
                        override=Path(output_dir) if output_dir else None, versioned=True)
    plt.close(fig)
    return paths


def free_stem(base: str, output_dir: Optional[str]) -> str:
    """`base`, `base_v2`, ... -- the first stem with no CSV here and no figure anywhere."""
    from experiments.shared.figures import output_dir as figure_dir

    figures = figure_dir(Path(__file__), Path(output_dir) if output_dir else None)

    def taken(stem: str) -> bool:
        return (HERE / f"{stem}.csv").exists() or any(
            (directory / f"{stem}.{suffix}").exists()
            for directory in (figures, HERE) for suffix in ("svg", "pdf"))

    stem, version = base, 2
    while taken(stem):
        stem, version = f"{base}_v{version}", version + 1
    return stem


@dataclass
class Settings:
    configuration: str = "standard"
    plans_csv: Optional[str] = None
    names: Tuple[str, ...] = ()
    jobs: Optional[int] = None
    output_dir: Optional[str] = None
    persist: bool = False
    from_csv: Optional[str] = None
    compare: Tuple[str, ...] = ()
    skip_gate: bool = False
    no_plot: bool = False
    panels: Tuple[str, ...] = PANELS
    average_bar: str = "bars"


def run(settings: Settings) -> None:
    from experiments.revision.common import write_sidecar
    from experiments.shared.figures import data_dir

    from translator.logical_optimization import enabled, state

    base = NAME if settings.configuration == "standard" else f"{NAME}_{settings.configuration}"
    if enabled() and not settings.from_csv:
        base = f"{base}_logical"
    print(f"[{NAME}] logical optimisation: {state()}")
    if settings.from_csv:
        rows = pd.read_csv(settings.from_csv)
        stem = Path(settings.from_csv).stem
    else:
        began = time.time()
        if not settings.skip_gate:
            gate(settings.plans_csv)
        rows = compute(settings)
        stem = free_stem(base, settings.output_dir)
        csv = HERE / f"{stem}.csv"
        rows.to_csv(csv, index=False)
        write_sidecar(csv, experiment=NAME, settings=asdict(settings), started=began,
                      status="complete")
        if settings.persist:
            target = data_dir(Path(__file__))
            for path in (csv, csv.with_name(csv.name + ".json")):
                if not (target / path.name).exists():
                    shutil.copy2(path, target / path.name)
        print(f"[{NAME}] wrote {csv}")
    summarize(rows)
    if settings.compare:
        compare(rows, settings.compare)
    if settings.no_plot:
        return
    if tuple(settings.panels) != PANELS:
        stem = f"{stem}_{'_'.join(p.lower() for p in settings.panels)}"
    if settings.average_bar != "bars":
        stem = f"{stem}_{settings.average_bar}"
    for path in plot(rows, stem, settings.output_dir, tuple(settings.panels),
                     settings.average_bar):
        print(f"[{NAME}] saved {path}")


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--configuration", choices=("standard", "submitted"), default="standard")
    parser.add_argument("--plans-csv", default=None,
                        help=f"recorded plans (default: {STANDARD_PLANS} for the standard, "
                             f"{SUBMITTED_PLANS} weight 0 for the submitted configuration)")
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--persist", action="store_true",
                        help="Also copy the CSV to results/data/ (about 35 min of simulation "
                             "to rebuild).")
    parser.add_argument("--from-csv", default=None,
                        help="Redraw from a cached CSV without recomputing.")
    parser.add_argument("--compare", nargs="+", default=[],
                        help="Fig 6-schema CSVs to compare the caption numbers against.")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--no-plot", action="store_true",
                        help="With --from-csv: print the numbers only, write no figure.")
    parser.add_argument("--panels", nargs="+", choices=PANELS, default=list(PANELS),
                        help="rows to draw, e.g. --panels 1q 2q depth TVD (no LSR row)")
    parser.add_argument("--average-bar", choices=("bars", "mean"), default="bars",
                        help="summary bar: mean of the clipped bars (default) or the unclipped "
                             "per-stage means the caption quotes")
    args = parser.parse_args()
    return Settings(configuration=args.configuration, plans_csv=args.plans_csv,
                    panels=tuple(args.panels), average_bar=args.average_bar,
                    names=tuple(args.names), jobs=args.jobs, output_dir=args.output_dir,
                    persist=args.persist, from_csv=args.from_csv, compare=tuple(args.compare),
                    skip_gate=args.skip_gate, no_plot=args.no_plot)


if __name__ == "__main__":
    run(parse_args())
