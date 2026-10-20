"""
ablation_fair_depth.py -- Fig 6 with the depth panel measured against an equally scheduled baseline.

Fig 6 (`merging_improvement_benchmark_stylized_simulated`) credits each stage with the
reduction of the translated circuit's depth relative to the naive baseline. Depth depends on
the order the translator meets the logical gates in -- gadgets on one patch serialize on its
anchors -- and the stages did not get the same order: the merging pass re-emits the whole
program in ASAP-layer order whenever any macro fires, while the baseline and +H stages reach
the translator in the order Qiskit rebuilds them from a DAG. So the +merging and +packing
segments carried a scheduling gain the baseline could have had for free.

This recomputes the depth panel with `TranslatorConfig.asap_order` on for every stage,
baseline included, so a segment is what its pass does. Everything else is the source figure's:
the same circuits, plans (placement and commutation decision per circuit), translate configs,
and the 1q / 2q / LSR / TVD panels, which are copied unchanged -- emission order moves gate
counts by at most 0.1 points (measured: `experiments/tvd_explore/order_effect.py`), and the
simulated panels are not re-drawn here.

Sources (`--source`):
    v4    the submitted Fig 6 (FIG6_CSV) with AQORN's plans (Experiment8a, weight 0, which
          reproduce its static arms on all 71 circuits)
    w10   Fig 6 regenerated with the weight-10 silent-aware planner (ablation_silent_aware)
    w10T  weight 10 + transversal in-patch CZ/XCX (branch v2-tvd-explore,
          experiments/tvd_explore/ablation_variant); its merging stages run under the switch

Checked on every run: the source's own depth numbers, recomputed here with the shipped
ordering, must be reproduced exactly (`static_reproduced`), or the plans or configs are not
the ones the source figure used and the run stops.

Deviation from house style, inherited on purpose: Fig 6's own `plot_results` draws the
figure (22 in canvas, its own size ladder) so this can replace Fig 6 one for one.

    PYTHONPATH=. venv/bin/python -m experiments.figures.ablation_fair_depth.ablation_fair_depth --source v4
    ... --source w10
    ... --from-csv experiments/figures/ablation_fair_depth/ablation_fair_depth_v4.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import logging
import math
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "ablation_fair_depth"
DEPTH = "pct_reduction_circuit_depth"
STATIC = ("pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates", DEPTH)
COMPONENTS = ("h_commutation", "merging", "packing")
STAGE_OF = {"h_commutation": "h_only", "merging": "h_merge", "packing": "packed"}

#: label -> (source CSV, switches for the merging stages, packing legend label)
SOURCES = {
    "v4": ("results/revision/_inputs/merging_improvement_benchmark_stylized_simulated.csv",
           (), None),
    "w10": ("experiments/figures/ablation_silent_aware/ablation_silent_aware_w10.csv",
            (), "+ Packing benefit (silent-aware, weight 10)"),
    "w10T": ("results/tvd_explore/ablation_variant_w10T.csv",
             ("TRANSVERSAL_INTRA_TWO_QUBIT",),
             "+ Packing benefit (silent-aware, weight 10, transversal CZ/XCX)"),
}
AQORN_PLANS = "results/revision/exp8/planner_static.csv"


@dataclass
class Settings:
    source: str = "v4"
    jobs: Optional[int] = None
    output_dir: Optional[str] = None
    from_csv: Optional[str] = None

    @property
    def stem(self) -> str:
        return f"{NAME}_{self.source}"


def _plans(source: str, rows: pd.DataFrame) -> Dict[str, Tuple[list, bool]]:
    """{circuit: (placement, commute)} the source figure was drawn with."""
    if "placement" in rows.columns:
        first = rows.drop_duplicates("circuit_name").set_index("circuit_name")
        return {name: (json.loads(r["placement"]), bool(r["commute"])) for name, r in first.iterrows()}
    frame = pd.read_csv(AQORN_PLANS)
    frame = frame[frame["silent_weight"] == 0.0]
    return {r.name: (json.loads(r.placement), bool(r.commute)) for r in frame.itertuples()}


def _run_circuit(job) -> List[Dict]:
    from pipeline.config import PipelineConfig

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from translator import explore_switches

    circuit, source_rows, plan, switches = job
    resource = fig6.ensure_measurements(circuit.original)
    n = resource.num_qubits
    tight = fig6.tight_trivial_packing(n, math.ceil(n / 2))
    placement, commute = [tuple(p) for p in plan[0]], bool(plan[1])

    def counts(perm, commute_, merge, asap):
        # The source's switches belong to its merging stages only, as its own run set them.
        explore_switches.configure(**{name: merge for name in switches})
        cfg = PipelineConfig()
        cfg.translator_config.enable_h_commutation = commute_
        cfg.translator_config.enable_merging = merge
        cfg.translator_config.asap_order = asap
        try:
            return fig6.compute_translation_metrics(resource, perm, cfg)
        finally:
            explore_switches.configure(**{name: False for name in switches})

    stages = {"base": (tight, False, False), "h_only": (tight, commute, False),
              "h_merge": (tight, commute, True), "packed": (placement, commute, True)}
    measured = {asap: {stage: counts(*args, asap) for stage, args in stages.items()}
                for asap in (False, True)}

    def cumulative(asap, stage, metric_index):
        base = measured[asap]["base"][metric_index]
        return fig6.pct_reduction(base, measured[asap][stage][metric_index])

    by_component = {row["component"]: row for row in source_rows}
    running = 0.0
    reproduced = True
    for component in COMPONENTS:
        running += float(by_component[component][DEPTH])
        reproduced &= abs(cumulative(False, STAGE_OF[component], 2) - running) < 1e-6

    rows = []
    previous = {"depth": 0.0, "one": 0.0, "two": 0.0}
    for component in COMPONENTS:
        stage = STAGE_OF[component]
        now = {"depth": cumulative(True, stage, 2), "one": cumulative(True, stage, 0),
               "two": cumulative(True, stage, 1)}
        row = dict(by_component[component])
        row["depth_shipped_segment"] = row[DEPTH]
        row[DEPTH] = now["depth"] - previous["depth"]
        row["fair_one_qubit_segment"] = now["one"] - previous["one"]
        row["fair_two_qubit_segment"] = now["two"] - previous["two"]
        row["static_reproduced"] = bool(reproduced)
        row["depth_ordering"] = "asap_order, every stage"
        for asap, tag in ((False, "shipped"), (True, "fair")):
            row[f"{tag}_baseline_depth"] = measured[asap]["base"][2]
            row[f"{tag}_stage_depth"] = measured[asap][stage][2]
        rows.append(row)
        previous = now
    return rows


def compute(settings: Settings) -> pd.DataFrame:
    from experiments.revision.common import load_fig6_circuits
    from experiments.shared.figures import stamp_provenance
    from experiments.shared.parallel import parallel_map

    path, switches, _ = SOURCES[settings.source]
    source = pd.read_csv(path)
    plans = _plans(settings.source, source)
    names = sorted(source["circuit_name"].unique())
    missing = [n for n in names if n not in plans]
    if missing:
        raise SystemExit(f"[{NAME}] no plan for {missing}")
    circuits = load_fig6_circuits(names=names)
    jobs = [(c, source[source["circuit_name"] == c.name].to_dict("records"), plans[c.name],
             tuple(switches)) for c in circuits]
    jobs.sort(key=lambda j: -j[0].num_qubits)
    print(f"[{NAME}] {settings.source}: {len(jobs)} circuits from {path}")
    results = parallel_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME)
    failed = [j[0].name for j, r in zip(jobs, results) if r is None]
    if failed:
        raise SystemExit(f"[{NAME}] {len(failed)} circuits failed: {failed}")
    rows = pd.DataFrame([row for r in results for row in r])
    bad = sorted(rows.loc[~rows["static_reproduced"], "circuit_name"].unique())
    if bad:
        raise SystemExit(f"[{NAME}] shipped depth not reproduced for {bad}: wrong plans or configs")
    order = {"small": 0, "medium": 1, "large": 2}
    rows = rows.sort_values(["size_group", "circuit_idx", "component"],
                            key=lambda col: col.map(order) if col.name == "size_group" else col)
    return stamp_provenance(rows)


def plot(rows: pd.DataFrame, settings: Settings) -> List[Path]:
    """Fig 6's own `plot_results`, then the source's packing label and the TVD-withheld stars."""
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
    packing_label = SOURCES[settings.source][2]
    if packing_label:
        for text in axes[0].get_legend().get_texts():
            if text.get_text() == "+ Packing benefit":
                text.set_text(packing_label)
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
    """
    Depth, shipped against equally scheduled, in the caption's convention (unclipped mean of
    each circuit's cumulative reduction, as in `values.tex`), then Fig 6's own summary of
    what the bars draw.
    """
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    def caption(column: str) -> pd.DataFrame:
        out = []
        for group in ("small", "medium", "large", "all"):
            sub = rows if group == "all" else rows[rows["size_group"] == group]
            wide = sub.pivot_table(index="circuit_name", columns="component", values=column)
            totals = wide[list(COMPONENTS)].cumsum(axis=1).mean()
            out.append({"group": group, "n": len(wide),
                        **{f"+{c}": totals[c] for c in COMPONENTS}})
        return pd.DataFrame(out).round(1)

    print(f"[{NAME}] depth reduction vs baseline, cumulative per stage, caption convention")
    print("  shipped ordering (the source figure):")
    print(caption("depth_shipped_segment").to_string(index=False))
    print("  every stage layer-sorted (this figure):")
    print(caption(DEPTH).to_string(index=False))
    per = rows.drop_duplicates("circuit_name")
    print(f"[{NAME}] baseline depth, mean: shipped {per['shipped_baseline_depth'].mean():.1f}, "
          f"layer-sorted {per['fair_baseline_depth'].mean():.1f}; source depth reproduced on "
          f"{int(per['static_reproduced'].sum())}/{len(per)}")
    fig6.print_metric_averages(rows)


def run(settings: Settings) -> None:
    from experiments.revision.common import write_sidecar

    if settings.from_csv:
        rows = pd.read_csv(settings.from_csv)
    else:
        began = time.time()
        rows = compute(settings)
        csv = HERE / f"{settings.stem}.csv"
        rows.to_csv(csv, index=False)
        write_sidecar(csv, experiment=NAME, settings=asdict(settings), started=began,
                      status="complete")
        print(f"[{NAME}] wrote {csv}")
    summarize(rows)
    for path in plot(rows, settings):
        print(f"[{NAME}] saved {path}")


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--source", choices=sorted(SOURCES), default="v4")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--from-csv", default=None,
                        help="Redraw from a cached CSV without recomputing (seconds to rebuild "
                             "anyway: translations only, no simulation).")
    args = parser.parse_args()
    return Settings(source=args.source, jobs=args.jobs, output_dir=args.output_dir,
                    from_csv=args.from_csv)


if __name__ == "__main__":
    run(parse_args())
