"""
logical_opt.py -- are Fig 6's inputs already near-optimal logical circuits, and does AQORN's
gain survive a strong logical pre-optimisation of them?

One can suspect that AQORN's reductions are the clean-up of sloppy input circuits. Two
measurements answer that:

Stage A  every one of Fig 6's 71 circuits through `optimizer.optimize_verified` (Qiskit O3
         to a fixed point plus structurally different Clifford / CNOT-block / commutation
         routes, best per circuit, never worse than the input) and an equivalence check of
         the result against the original. Logical 1q, 2q, total gates and depth, before and
         after, the route that won, and the O3-only result. The optimised circuits are cached
         as QPY (`<out>/circuits/`), because O3 is not deterministic run to run.
Stage B  Fig 6's static pipeline on the original and on the optimised circuit: baseline
         (tight trivial packing, both passes off), +H, +H+merge, packed, for AQORN v4
         (`plan_packing`, the submitted planner), w10 (weight-10 silent-aware planner, what
         the paper now uses) and w10T (w10 plus TRANSVERSAL_INTRA_TWO_QUBIT on its merging
         stages). Each input is planned on itself; the originals reuse the plans their
         figures were drawn with, and the shipped-order counts must reproduce those figures'
         CSVs exactly (`reproduced`), after a gate that re-plans four originals with each
         planner and must get the cached plans back. Every stage is translated twice: shipped
         emission order and `TranslatorConfig.asap_order = True` (the fair baseline: the
         merging rewriter re-sorts the program whenever a macro fires, so without it the
         baseline lacks a free scheduling gain; see `ablation_fair_depth`). Depth is reported
         fair. Every packed arm gets the noiseless miscompile check
         (`experiments.shared.simulation.miscompiled`), whose naive-arm floors also feed the
         TVD guard in `combine` (an optimised proxy whose output support exploded).
Stage C  LSR and TVD of the optimised inputs' baseline and packed arms (w10, w10T; v4 on
         request) at each circuit's Fig 6 alpha and shot count -- the original's physical
         noise, not recalibrated -- with Fig 6's seeds (`10 * 1_000_003 + idx * 17 + offset`,
         offsets 3 baseline / 6 packed) and Fig 6's shipped emission order, so they compare
         one for one with the originals' Fig 6 / ablation numbers. Circuits the optimiser
         left unchanged reuse those numbers instead of being resimulated. Before the sweep,
         two original arms are re-drawn and must reproduce their figures' samples exactly.

Outputs (`--output-dir`, default results/logical_opt/, gitignored): stage_a.csv,
stage_b.csv, stage_c.csv (each with a JSON sidecar), circuits/*.qpy, and logical_opt.csv (or a
timestamped name when that exists), one row per circuit with everything the summary and the
figure read, plus its .summary.txt with the report's tables. No CSV or figure is overwritten:
Stage A/B refuse an existing CSV, Stage C resumes its table by key only under identical
settings (`ResultTable`), the figure is versioned. Figure:
results/figures/logical_preoptimization_<plot variant>.{svg,pdf}, full width; panel (a) what
the optimiser does to each circuit, panel (b) AQORN's gain on original vs optimised input.

`--routes o3` restricts the optimiser to plain O3 (the sensitivity run).

    PYTHONPATH=. venv/bin/python -m experiments.logical_opt.logical_opt --jobs 3
    ... --stages B C           # reuse the cached earlier stages
    ... --stages               # recombine the cached stage CSVs, summary + figure
    ... --from-csv results/logical_opt/logical_opt.csv [--plot-variant w10T]   # figure only
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
# This study *is* the optimisation, done by hand on originals it compares against: the
# compiler's own pass (translator/logical_optimization.py) would hand it pre-optimised
# "originals" and re-optimise what it measures. Off for this process and its workers.
os.environ["AQORN_LOGICAL_OPT"] = "0"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import io
import json
import logging
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAME = "logical_opt"
DEFAULT_OUT = REPO / "results" / "logical_opt"

FIG6_CSV = REPO / "results/revision/_inputs/merging_improvement_benchmark_stylized_simulated.csv"
V4_PLANS = REPO / "results/revision/exp8/planner_static.csv"
W10_CSV = REPO / "experiments/figures/ablation_silent_aware/ablation_silent_aware_w10.csv"
W10T_CSV = REPO / "results/tvd_explore/ablation_variant_w10T.csv"

#: label -> planner weight, translator switches for its merging stages, silent-rate table,
#: and the CSV its original-input figure was drawn from.
VARIANTS: Dict[str, Dict] = {
    "v4": {"weight": 0.0, "switches": (), "rates": None, "source": FIG6_CSV},
    "w10": {"weight": 10.0, "switches": (),
            "rates": REPO / "results/revision/exp8/gadget_rates.csv", "source": W10_CSV},
    "w10T": {"weight": 10.0, "switches": ("TRANSVERSAL_INTRA_TWO_QUBIT",),
             "rates": REPO / "results/tvd_explore/gadget_rates_transversal.csv",
             "source": W10T_CSV},
}
STAGES = ("base", "h_only", "h_merge", "packed")
STAGE_OF = {"h_commutation": "h_only", "merging": "h_merge", "packing": "packed"}
COMPONENTS = tuple(STAGE_OF)
STATIC = ("pct_reduction_one_qubit_gates", "pct_reduction_two_qubit_gates",
          "pct_reduction_circuit_depth")
LSR = "pct_improvement_logical_selection_rate"
TVD = "pct_reduction_logical_total_variation_distance"
METRIC_KEYS = ("one", "two", "depth")
#: Fig 6's seed offsets.
OFFSET = {"baseline": 3, "packed": 6}
FIG6_SEED = 10
GROUPS = ("small", "medium", "large")
#: Fig 6 withholds TVD where the baseline cannot resolve it (support beyond any shot count).
TVD_SATURATION = 0.95


@dataclass
class Settings:
    stages: Tuple[str, ...] = ("A", "B", "C")
    jobs: int = 3
    output_dir: str = str(DEFAULT_OUT)
    from_csv: Optional[str] = None
    names: Tuple[str, ...] = ()
    sim_arms: Tuple[str, ...] = ("baseline", "w10", "w10T")
    plot_variant: str = "w10"
    #: None: the canonical results/figures/ (versioned). Smoke runs pass a scratch directory.
    figure_dir: Optional[str] = None
    #: Figure name is `<stem>_<plot_variant>`; `save_figure` also drops an SVG beside this
    #: script and versions against it, so draft renders should use a different stem.
    figure_stem: str = "logical_preoptimization"
    skip_gates: bool = False
    noiseless_check: bool = True
    seed: int = 0
    #: Optimiser routes (`optimizer.ROUTES` keys); empty means all of them. `--routes o3` is
    #: the sensitivity run: plain Qiskit O3 to a fixed point, what a skeptic would try first.
    routes: Tuple[str, ...] = ()

    @property
    def out(self) -> Path:
        return Path(self.output_dir)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{NAME}] {message}", flush=True)


def _pct_reduction(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference if reference > 0 else 0.0


def _pct_improvement(reference: float, value: float) -> float:
    return 100.0 * (value - reference) / reference if reference > 0 else 0.0


# ----------------------------------------------------------------------------------------
# circuits
# ----------------------------------------------------------------------------------------


def _load_circuits(settings: Settings):
    from experiments.revision.common import load_fig6_circuits

    names = list(settings.names) or None
    return load_fig6_circuits(names=names)


def _qpy_path(out: Path, name: str) -> Path:
    return out / "circuits" / f"{name}.qpy"


def _qpy_bytes(circuit) -> bytes:
    from qiskit import qpy

    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


def load_optimized(out: Path, name: str):
    """The cached optimised circuit, or None if the optimiser left `name` unchanged."""
    from qiskit import qpy

    path = _qpy_path(out, name)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return qpy.load(handle)[0]


# ----------------------------------------------------------------------------------------
# Stage A: optimise and verify
# ----------------------------------------------------------------------------------------


def _stage_a_point(job) -> Tuple[Dict, Optional[bytes]]:
    from experiments.logical_opt.optimizer import (
        is_clifford,
        logical_counts,
        non_clifford_rotations,
        optimize_verified,
    )

    from experiments.logical_opt.optimizer import ROUTES

    meta, original, seed, routes = job
    chosen = {name: ROUTES[name] for name in routes} if routes else None
    result, check = optimize_verified(original, seed=seed, routes=chosen)
    row = dict(meta)
    row.update({"num_qubits": original.num_qubits, "clifford": is_clifford(original),
                "non_clifford_rotations": non_clifford_rotations(original),
                "barriers": sum(1 for i in original.data if i.operation.name == "barrier")})
    for tag, circuit in (("orig", original), ("opt", result.circuit), ("o3", result.o3_circuit)):
        row.update({f"{tag}_{key}": value for key, value in logical_counts(circuit).items()})
    row.update({"changed": bool(result.changed), "route": result.route,
                "o3_iterations": result.o3_iterations, "rounds": result.rounds,
                "opt_seconds": result.seconds,
                "first_round": json.dumps({k: list(v) if v else None
                                           for k, v in result.first_round.items()}),
                "route_failures": json.dumps(result.failures)})
    row.update(check)
    return row, (_qpy_bytes(result.circuit) if result.changed else None)


def stage_a(settings: Settings, circuits) -> pd.DataFrame:
    from experiments.revision.common import write_sidecar
    from experiments.shared.parallel import parallel_map

    began = time.time()
    path = settings.out / "stage_a.csv"
    _refuse_overwrite(path)
    if any((settings.out / "circuits").glob("*.qpy")):
        raise SystemExit(f"[{NAME}] {settings.out / 'circuits'} already holds cached circuits; "
                         f"refusing to overwrite them (pass a new --output-dir)")
    jobs = [({"name": c.name, "size_group": c.size_group, "circuit_idx": c.circuit_idx},
             c.original, settings.seed, tuple(settings.routes)) for c in circuits]
    jobs.sort(key=lambda j: -j[1].num_qubits)
    log(f"Stage A: optimising {len(jobs)} circuits")
    results = parallel_map(_stage_a_point, jobs, jobs=settings.jobs, label="stage-A")
    failed = [j[0]["name"] for j, r in zip(jobs, results) if r is None]
    if failed:
        raise SystemExit(f"[{NAME}] Stage A failed on {failed}")
    (settings.out / "circuits").mkdir(parents=True, exist_ok=True)
    rows = []
    for (row, payload) in results:
        rows.append(row)
        if payload is not None:
            _qpy_path(settings.out, row["name"]).write_bytes(payload)
    frame = _canonical(pd.DataFrame(rows))
    bad = frame.loc[~frame["equivalent"].astype(bool), "name"].tolist()
    if bad:
        raise SystemExit(f"[{NAME}] equivalence failed after every fallback: {bad}")
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment=f"{NAME}-A", settings=asdict(settings), started=began,
                  status="complete")
    log(f"Stage A: {int(frame['changed'].sum())}/{len(frame)} circuits changed; wrote {path}")
    return frame


def _canonical(frame: pd.DataFrame, name_column: str = "name") -> pd.DataFrame:
    order = {g: i for i, g in enumerate(GROUPS)}
    return frame.sort_values(["size_group", "circuit_idx"] + (
        [c for c in ("input", "variant", "arm") if c in frame.columns]),
        key=lambda col: col.map(order) if col.name == "size_group" else col
    ).reset_index(drop=True)


def _refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"[{NAME}] {path} exists; refusing to overwrite it (pass a new "
                         f"--output-dir, or run only the later stages)")


# ----------------------------------------------------------------------------------------
# Stage B: Fig 6's static pipeline on both inputs, three planners
# ----------------------------------------------------------------------------------------


def _original_plans(names: Sequence[str]) -> Dict[str, Dict[str, Tuple[list, bool]]]:
    """{circuit: {variant: (placement, commute)}}: the plans each original figure drew."""
    plans: Dict[str, Dict[str, Tuple[list, bool]]] = {name: {} for name in names}
    v4 = pd.read_csv(V4_PLANS)
    v4 = v4[v4["silent_weight"] == 0.0].set_index("name")
    for variant, path in (("w10", W10_CSV), ("w10T", W10T_CSV)):
        frame = pd.read_csv(path).drop_duplicates("circuit_name").set_index("circuit_name")
        for name in names:
            plans[name][variant] = (json.loads(frame.loc[name, "placement"]),
                                    bool(frame.loc[name, "commute"]))
    for name in names:
        plans[name]["v4"] = (json.loads(v4.loc[name, "placement"]), bool(v4.loc[name, "commute"]))
    return plans


def _source_rows(names: Sequence[str]) -> Dict[str, Dict[str, List[Dict]]]:
    out: Dict[str, Dict[str, List[Dict]]] = {name: {} for name in names}
    for variant, spec in VARIANTS.items():
        frame = pd.read_csv(spec["source"])
        for name in names:
            out[name][variant] = frame[frame["circuit_name"] == name].to_dict("records")
    return out


def _rates(variant: str):
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from simulator.noise_models import StimNoiseParameters

    return UndetectedRates.from_table(pd.read_csv(VARIANTS[variant]["rates"]), 4,
                                      StimNoiseParameters.neutral_atom().two_qubit_error)


def plan(program, variant: str) -> Dict:
    """Plan `program` (measurements included) the way `variant`'s figure planned its circuits."""
    from experiments.revision.common import fig6_noise_profile
    from translator import explore_switches

    spec = VARIANTS[variant]
    began = time.time()
    if variant == "v4":
        from packing.plan import plan_packing

        result = plan_packing(program, 4, noise=fig6_noise_profile(), verify=True)
        currency = None
    else:
        from experiments.revision.exp8_undetected.cost_model import plan_silent_aware

        # As `tvd_explore.plan_variant`: switches on for the whole planning call, so the
        # planner's tables, its verify translations and the rates describe one translator.
        explore_switches.configure(**{name: True for name in spec["switches"]})
        try:
            result = plan_silent_aware(program, 4, rates=_rates(variant),
                                       silent_weight=spec["weight"], noise=fig6_noise_profile())
        finally:
            explore_switches.configure(**{name: False for name in spec["switches"]})
        currency = result.currency
    return {"placement": [[int(a), int(b)] for a, b in result.placement],
            "commute": bool(result.h_commutation), "fell_back": bool(getattr(result, "fell_back", False)),
            "currency": currency, "plan_seconds": time.time() - began}


class _Translations:
    """Fig 6's `compute_translation_metrics`, memoised per (placement, flags, switches)."""

    def __init__(self, program) -> None:
        self.program = program
        self.memo: Dict[Tuple, Tuple[int, int, int]] = {}

    def __call__(self, perm, commute: bool, merge: bool, asap: bool,
                 switches: Sequence[str] = ()) -> Tuple[int, int, int]:
        from pipeline.config import PipelineConfig
        from translator import explore_switches

        from experiments.figures.merging_improvement import (
            merging_improvement_benchmark_stylized_simulated as fig6,
        )

        active = tuple(switches) if merge else ()  # switches belong to the merging stages
        key = (tuple(tuple(p) for p in perm), bool(commute), bool(merge), bool(asap), active)
        if key not in self.memo:
            explore_switches.configure(**{name: True for name in active})
            try:
                config = PipelineConfig()
                config.translator_config.enable_h_commutation = bool(commute)
                config.translator_config.enable_merging = bool(merge)
                config.translator_config.asap_order = bool(asap)
                self.memo[key] = tuple(int(v) for v in fig6.compute_translation_metrics(
                    self.program, [tuple(p) for p in perm], config))
            finally:
                explore_switches.configure(**{name: False for name in active})
        return self.memo[key]


def _noiseless(proxy, placement, commute: bool, switches: Sequence[str]) -> float:
    from experiments.tvd_explore.screen import _noiseless_tvd
    from translator import explore_switches

    explore_switches.configure(**{name: True for name in switches})
    try:
        return float(_noiseless_tvd(proxy, [tuple(p) for p in placement], commute=commute,
                                    merging=True, overrides={}, seed=5))
    finally:
        explore_switches.configure(**{name: False for name in switches})


def _stage_b_point(job) -> List[Dict]:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from experiments.shared.simulation import miscompiled

    meta, inputs, original_plans, source_rows, noiseless_check = job
    rows: List[Dict] = []
    for input_tag, (program, proxy) in inputs.items():
        resource = fig6.ensure_measurements(program)
        n = resource.num_qubits
        tight = fig6.tight_trivial_packing(n, math.ceil(n / 2))
        translate = _Translations(resource)
        naive_clean: Dict[Tuple, float] = {}
        for variant, spec in VARIANTS.items():
            switches = spec["switches"]
            if input_tag == "original":
                placement, commute = original_plans[variant]
                planned = {"placement": placement, "commute": commute, "fell_back": None,
                           "currency": None, "plan_seconds": float("nan")}
            else:
                planned = plan(resource, variant)
                placement, commute = planned["placement"], planned["commute"]
            arms = {"base": (tight, False, False), "h_only": (tight, commute, False),
                    "h_merge": (tight, commute, True), "packed": (placement, commute, True)}
            row = {**meta, "input": input_tag, "variant": variant, "num_qubits": n,
                   "placement": json.dumps(placement), "commute": bool(commute),
                   "fell_back": planned["fell_back"], "currency": planned["currency"],
                   "plan_seconds": planned["plan_seconds"],
                   "patches": len({int(p) for p, _ in placement})}
            for asap, tag in ((False, "shipped"), (True, "fair")):
                for stage, (perm, com, merge) in arms.items():
                    one, two, depth = translate(perm, com, merge, asap, switches)
                    row[f"{tag}_{stage}_one"] = one
                    row[f"{tag}_{stage}_two"] = two
                    row[f"{tag}_{stage}_depth"] = depth
            if input_tag == "original":
                # The source figure's own static percentages, recomputed with its plans and
                # shipped ordering: anything else means the wrong plans or configs.
                by_component = {r["component"]: r for r in source_rows[variant]}
                running = dict.fromkeys(STATIC, 0.0)
                reproduced = True
                for component in COMPONENTS:
                    stage = STAGE_OF[component]
                    for column, key in zip(STATIC, METRIC_KEYS):
                        running[column] += float(by_component[component][column])
                        mine = _pct_reduction(row[f"shipped_base_{key}"],
                                              row[f"shipped_{stage}_{key}"])
                        reproduced &= abs(mine - running[column]) < 1e-6
                row["reproduced"] = bool(reproduced)
            else:
                row["reproduced"] = None
            if noiseless_check:
                key = (bool(commute), tuple(switches))
                if key not in naive_clean:
                    naive_clean[key] = _noiseless(proxy, tight, commute, switches)
                clean = _noiseless(proxy, placement, commute, switches)
                row["noiseless_naive_tvd"] = naive_clean[key]
                row["noiseless_packed_tvd"] = clean
                row["miscompiled"] = bool(miscompiled(naive_clean[key], clean))
            rows.append(row)
    return rows


def stage_b(settings: Settings, circuits, stage_a_frame: pd.DataFrame) -> pd.DataFrame:
    from experiments.logical_opt.optimizer import proxy_of
    from experiments.revision.common import write_sidecar
    from experiments.shared.parallel import parallel_map

    began = time.time()
    path = settings.out / "stage_b.csv"
    _refuse_overwrite(path)
    names = [c.name for c in circuits]
    plans = _original_plans(names)
    sources = _source_rows(names)
    if not settings.skip_gates:
        planner_gate(settings, circuits, plans)
    changed = set(stage_a_frame.loc[stage_a_frame["changed"].astype(bool), "name"])
    jobs = []
    for c in circuits:
        inputs = {"original": (c.original, c.proxy)}
        if c.name in changed:
            optimized = load_optimized(settings.out, c.name)
            if optimized is None:
                raise SystemExit(f"[{NAME}] {c.name} changed in Stage A but has no cached circuit")
            inputs["optimized"] = (optimized, proxy_of(optimized))
        meta = {"name": c.name, "size_group": c.size_group, "circuit_idx": c.circuit_idx}
        jobs.append((meta, inputs, plans[c.name], sources[c.name], settings.noiseless_check))
    # Widest first: their plans dominate (bv_n140 ~6 min per planner).
    jobs.sort(key=lambda j: -j[1]["original"][0].num_qubits)
    log(f"Stage B: {len(jobs)} circuits, {len(changed & set(names))} with an optimised input")
    results = parallel_map(_stage_b_point, jobs, jobs=settings.jobs, label="stage-B")
    failed = [j[0]["name"] for j, r in zip(jobs, results) if r is None]
    if failed:
        raise SystemExit(f"[{NAME}] Stage B failed on {failed}")
    rows = [row for result in results for row in result]
    # Unchanged circuits: the optimised input *is* the original, so its rows are reused.
    for result in results:
        if any(r["input"] == "optimized" for r in result):
            continue
        for r in result:
            rows.append({**r, "input": "optimized", "reused": True})
    frame = _canonical(pd.DataFrame(rows))
    frame["reused"] = frame.get("reused", False)
    frame["reused"] = frame["reused"].fillna(False).astype(bool)
    bad = frame.loc[(frame["input"] == "original") & ~frame["reproduced"].astype(bool),
                    ["name", "variant"]]
    if not bad.empty:
        raise SystemExit(f"[{NAME}] source figures not reproduced:\n{bad}")
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment=f"{NAME}-B", settings=asdict(settings), started=began,
                  status="complete")
    if "miscompiled" in frame.columns:
        flagged = frame[frame["miscompiled"].fillna(False).astype(bool)]
        log(f"Stage B: miscompiled arms: {flagged[['name', 'input', 'variant']].values.tolist()}")
    log(f"Stage B: wrote {path}")
    return frame


#: Small circuits whose original is re-planned before the sweep: each planner must return
#: the plan its figure was drawn with, or new plans would not be comparable to old ones.
PLANNER_GATE = ("adder_n10", "qaoa_n6", "bell_n4", "qft_n4")


def _planner_gate_point(job) -> Tuple[str, str, bool]:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    name, variant, original, expected = job
    got = plan(fig6.ensure_measurements(original), variant)
    same = ([tuple(p) for p in got["placement"]] == [tuple(p) for p in expected[0]]
            and got["commute"] == bool(expected[1]))
    return name, variant, same


def planner_gate(settings: Settings, circuits, plans) -> None:
    from experiments.shared.parallel import parallel_map

    jobs = [(c.name, variant, c.original, plans[c.name][variant])
            for c in circuits if c.name in PLANNER_GATE for variant in VARIANTS]
    if not jobs:
        return
    results = parallel_map(_planner_gate_point, jobs, jobs=settings.jobs, label="planner-gate",
                           quiet=True)
    bad = [r for r in results if r is None or not r[2]]
    if bad:
        raise SystemExit(f"[{NAME}] PLANNER GATE FAILED: {bad}; the planners do not reproduce "
                         f"their figures' plans here")
    log(f"planner gate: {len(results)} (circuit, planner) plans reproduced exactly")


# ----------------------------------------------------------------------------------------
# Stage C: simulate the optimised inputs at Fig 6's noise
# ----------------------------------------------------------------------------------------


def _simulate(job) -> Dict:
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from translator import explore_switches

    meta, proxy, arm, placement, commute, merge, switches, alpha, reps, seed = job
    settings = fig6.BenchmarkSettings()
    config = fig6.build_pipeline_config(settings, alpha=alpha, repetitions=reps, seed=seed,
                                        enable_h_commutation=commute, enable_merging=merge)
    began = time.time()
    explore_switches.configure(**{name: True for name in switches})
    try:
        out = fig6.evaluate_variant(fig6.ensure_measurements(proxy),
                                    [tuple(p) for p in placement], config)
    finally:
        explore_switches.configure(**{name: False for name in switches})
    return {**meta, "arm": arm, "lsr": float(out["logical_selection_rate"]),
            "tvd": float(out["logical_total_variation_distance"]), "alpha": alpha,
            "repetitions": reps, "seed": seed, "commute": bool(commute), "merging": bool(merge),
            "switches": ",".join(switches), "sim_seconds": time.time() - began}


def _fig6_rows() -> pd.DataFrame:
    return pd.read_csv(FIG6_CSV).drop_duplicates("circuit_name").set_index("circuit_name")


def original_simulated(names: Sequence[str]) -> pd.DataFrame:
    """The originals' simulated arms, from the figures that drew them (no new samples)."""
    rows = []
    for variant, spec in VARIANTS.items():
        frame = pd.read_csv(spec["source"])
        for name in names:
            block = frame[frame["circuit_name"] == name]
            first = block.iloc[0]
            base_lsr, base_tvd = float(first["baseline_lsr"]), float(first["baseline_tvd"])
            rows.append({"name": name, "arm": variant,
                         "lsr": base_lsr * (1 + block[LSR].sum() / 100.0),
                         "tvd": base_tvd * (1 - block[TVD].sum() / 100.0)})
            if variant == "v4":
                rows.append({"name": name, "arm": "baseline", "lsr": base_lsr, "tvd": base_tvd})
    return pd.DataFrame(rows)


def simulation_gate(settings: Settings, circuits, plans) -> None:
    """Re-draw two original arms with Fig 6's seeds; they must equal the figures' samples."""
    from experiments.shared.parallel import parallel_map

    fig6_rows = _fig6_rows()
    reference = original_simulated([c.name for c in circuits]).set_index(["name", "arm"])
    by_name = {c.name: c for c in circuits}
    jobs = []
    for name, arm in (("bell_n4", "baseline"), ("adder_n10", "w10")):
        if name not in by_name:
            continue
        c = by_name[name]
        row = fig6_rows.loc[name]
        n = c.proxy.num_qubits
        if arm == "baseline":
            perm, commute, merge = [(i // 2, i % 2) for i in range(n)], False, False
        else:
            perm, commute = plans[name][arm]
            merge = True
        seed = FIG6_SEED * 1_000_003 + int(row["circuit_idx"]) * 17 + OFFSET[
            "baseline" if arm == "baseline" else "packed"]
        jobs.append(({"name": name}, c.proxy, arm, perm, commute, merge,
                     VARIANTS.get(arm, {}).get("switches", ()), float(row["alpha_final"]),
                     int(row["final_repetitions"]), seed))
    results = parallel_map(_simulate, jobs, jobs=settings.jobs, label="sim-gate", quiet=True)
    for result in results:
        if result is None:
            raise SystemExit(f"[{NAME}] SIMULATION GATE FAILED: a gate arm did not run")
        expected = reference.loc[(result["name"], result["arm"])]
        if abs(result["lsr"] - expected["lsr"]) > 1e-9 or abs(result["tvd"] - expected["tvd"]) > 1e-9:
            raise SystemExit(f"[{NAME}] SIMULATION GATE FAILED: {result['name']} {result['arm']} "
                             f"LSR {result['lsr']:.6f} / TVD {result['tvd']:.6f} against "
                             f"{expected['lsr']:.6f} / {expected['tvd']:.6f}")
        log(f"simulation gate: {result['name']} {result['arm']} LSR {result['lsr']:.6f} "
            f"TVD {result['tvd']:.6f} == source figure")


def stage_c(settings: Settings, circuits, stage_a_frame: pd.DataFrame,
            stage_b_frame: pd.DataFrame) -> pd.DataFrame:
    from experiments.logical_opt.optimizer import proxy_of
    from experiments.revision.common import ResultTable, stream_map

    fig6_rows = _fig6_rows()
    names = [c.name for c in circuits]
    if not settings.skip_gates:
        simulation_gate(settings, circuits, _original_plans(names))
    changed = set(stage_a_frame.loc[stage_a_frame["changed"].astype(bool), "name"])
    b = stage_b_frame[stage_b_frame["input"] == "optimized"].set_index(["name", "variant"])
    path = settings.out / "stage_c.csv"
    table = ResultTable(path, key_columns=("name", "arm"), experiment=f"{NAME}-C",
                        settings={"sim_arms": list(settings.sim_arms), "seed": FIG6_SEED,
                                  "source": str(FIG6_CSV)}, resume=True)
    jobs = []
    for c in circuits:
        if c.name not in changed:
            continue
        optimized = load_optimized(settings.out, c.name)
        proxy = proxy_of(optimized)
        row = fig6_rows.loc[c.name]
        alpha, reps, idx = float(row["alpha_final"]), int(row["final_repetitions"]), int(row["circuit_idx"])
        meta = {"name": c.name, "size_group": c.size_group, "circuit_idx": idx,
                "num_qubits": c.num_qubits}
        for arm in settings.sim_arms:
            if table.is_done((c.name, arm)):
                continue
            n = proxy.num_qubits
            if arm == "baseline":
                perm, commute, merge, switches = [(i // 2, i % 2) for i in range(n)], False, False, ()
            else:
                planned = b.loc[(c.name, arm)]
                perm, commute = json.loads(planned["placement"]), bool(planned["commute"])
                merge, switches = True, VARIANTS[arm]["switches"]
            seed = FIG6_SEED * 1_000_003 + idx * 17 + OFFSET[
                "baseline" if arm == "baseline" else "packed"]
            jobs.append((meta, proxy, arm, perm, commute, merge, tuple(switches), alpha, reps, seed))
    # Arms in priority order (baseline and w10 before w10T), widest first within each.
    priority = {arm: i for i, arm in enumerate(settings.sim_arms)}
    jobs.sort(key=lambda j: (priority[j[2]] > priority.get("w10", 1), -j[0]["num_qubits"]))
    log(f"Stage C: {len(jobs)} arms to simulate ({len(changed)} changed circuits; unchanged "
        f"ones reuse their figures' samples)")
    def on_result(_index, row):
        table.append([row])

    counts = stream_map(_simulate, jobs, jobs=settings.jobs, label="stage-C", on_result=on_result)
    table.write_sidecar(status="complete" if counts["error"] == 0 else "partial",
                        extra={"errors": counts["error"]})
    log(f"Stage C: {counts}; wrote {path}")
    return table.frame()


# ----------------------------------------------------------------------------------------
# combine, summarise
# ----------------------------------------------------------------------------------------


def combine(settings: Settings, a: pd.DataFrame, b: pd.DataFrame,
            c: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per circuit: Stage A counts, Stage B reductions, Stage C fidelity."""
    out = a.set_index("name").copy()
    for _, row in b.iterrows():
        prefix = f"{row['input'][:3]}_{row['variant']}"
        for tag in ("fair", "shipped"):
            for key in METRIC_KEYS:
                base = row[f"{tag}_base_{key}"]
                out.loc[row["name"], f"{prefix}_{tag}_base_{key}"] = base
                for stage in STAGES[1:]:
                    value = row[f"{tag}_{stage}_{key}"]
                    out.loc[row["name"], f"{prefix}_{tag}_{stage}_{key}"] = value
                    out.loc[row["name"], f"{prefix}_{tag}_{stage}_{key}_red"] = _pct_reduction(base, value)
        out.loc[row["name"], f"{prefix}_commute"] = bool(row["commute"])
        out.loc[row["name"], f"{prefix}_patches"] = row["patches"]
        out.loc[row["name"], f"{prefix}_plan_seconds"] = row["plan_seconds"]
        if "miscompiled" in row and row["miscompiled"] == row["miscompiled"]:
            out.loc[row["name"], f"{prefix}_miscompiled"] = bool(row["miscompiled"])
    # Simulated arms: originals from their figures, optimised from Stage C or reused.
    original = original_simulated(list(out.index)).set_index(["name", "arm"])
    simulated = c.set_index(["name", "arm"]) if c is not None and not c.empty else None
    for name in out.index:
        for arm in ("baseline",) + tuple(VARIANTS):
            o = original.loc[(name, arm)]
            out.loc[name, f"ori_{arm}_lsr"] = o["lsr"]
            out.loc[name, f"ori_{arm}_tvd"] = o["tvd"]
            if not bool(out.loc[name, "changed"]):
                out.loc[name, f"opt_{arm}_lsr"], out.loc[name, f"opt_{arm}_tvd"] = o["lsr"], o["tvd"]
                out.loc[name, f"opt_{arm}_sim"] = "reused"
            elif simulated is not None and (name, arm) in simulated.index:
                s = simulated.loc[(name, arm)]
                out.loc[name, f"opt_{arm}_lsr"], out.loc[name, f"opt_{arm}_tvd"] = s["lsr"], s["tvd"]
                out.loc[name, f"opt_{arm}_sim"] = "simulated"
        for tag in ("ori", "opt"):
            base_lsr = out.loc[name].get(f"{tag}_baseline_lsr", np.nan)
            base_tvd = out.loc[name].get(f"{tag}_baseline_tvd", np.nan)
            for arm in VARIANTS:
                lsr = out.loc[name].get(f"{tag}_{arm}_lsr", np.nan)
                tvd = out.loc[name].get(f"{tag}_{arm}_tvd", np.nan)
                if pd.notna(lsr) and pd.notna(base_lsr):
                    out.loc[name, f"{tag}_{arm}_lsr_impr"] = _pct_improvement(base_lsr, lsr)
                if pd.notna(tvd) and pd.notna(base_tvd):
                    out.loc[name, f"{tag}_{arm}_tvd_red"] = _pct_reduction(base_tvd, tvd)
        resolved = out.loc[name, "ori_baseline_tvd"] <= TVD_SATURATION
        if pd.notna(out.loc[name].get("opt_baseline_tvd", np.nan)):
            resolved = resolved and out.loc[name, "opt_baseline_tvd"] <= TVD_SATURATION
        out.loc[name, "tvd_resolved_fig6_rule"] = bool(resolved)
    # A second TVD guard, specific to this experiment. The proxy of a resynthesised circuit is
    # a different Clifford circuit (every rz/rx -> s), and where block resynthesis scattered
    # rotations through the circuit its output support can explode: dnn_n33's optimised proxy
    # has a noiseless floor of 0.94 at 4,000 shots against the original's < 0.15. TVD on such
    # an input measures the sampling floor, not the compiler, so the pair is withheld from TVD
    # (LSR, a binomial, is unaffected) by the same differential rule `miscompiled` uses for
    # placements: optimised floor > 3x the original's and > 0.10 above it (naive arm, w10's
    # commutation setting, merging on).
    from experiments.shared.simulation import miscompiled

    if "noiseless_naive_tvd" in b.columns:
        floors = b[(b["variant"] == "w10")].pivot_table(
            index="name", columns="input", values="noiseless_naive_tvd", aggfunc="first")
    else:  # --no-noiseless-check: no floors measured, so this guard cannot run
        floors = pd.DataFrame()
    for name in out.index:
        if name in floors.index and {"original", "optimized"} <= set(floors.columns):
            orig_floor, opt_floor = floors.loc[name, "original"], floors.loc[name, "optimized"]
            out.loc[name, "ori_proxy_floor"], out.loc[name, "opt_proxy_floor"] = orig_floor, opt_floor
            comparable = not miscompiled(float(orig_floor), float(opt_floor))
        else:
            comparable = True
        out.loc[name, "proxy_floor_comparable"] = bool(comparable)
        out.loc[name, "tvd_resolved"] = bool(out.loc[name, "tvd_resolved_fig6_rule"] and comparable)
    return out.reset_index()


def _group_means(frame: pd.DataFrame, column: str, mask: Optional[pd.Series] = None) -> Dict[str, float]:
    scoped = frame if mask is None else frame[mask]
    means = {g: scoped.loc[scoped["size_group"] == g, column].mean() for g in GROUPS}
    means["all"] = scoped[column].mean()
    means["n"] = int(scoped[column].notna().sum())
    return means


def summarize(frame: pd.DataFrame) -> str:
    """Console + report tables: everything in the caption convention (unclipped means)."""
    lines: List[str] = []
    changed = frame["changed"].astype(bool)
    lines.append(f"circuits: {len(frame)}, changed by the optimiser: {int(changed.sum())}, "
                 f"unchanged: {int((~changed).sum())}")
    lines.append(f"equivalence methods: {frame['equiv_method'].value_counts().to_dict()}, "
                 f"all equivalent: {bool(frame['equivalent'].astype(bool).all())}")

    def logical(metric: str) -> pd.Series:
        return 100.0 * (frame[f"orig_{metric}"] - frame[f"opt_{metric}"]) / frame[f"orig_{metric}"].replace(0, np.nan)

    lines.append("\nStage A -- logical reduction from optimisation (% of the original; per-circuit "
                 "mean / median / corpus total):")
    for metric in ("one_qubit", "two_qubit", "total", "depth"):
        per = logical(metric).fillna(0.0)
        total = 100.0 * (1 - frame[f"opt_{metric}"].sum() / frame[f"orig_{metric}"].sum())
        o3 = 100.0 * (frame[f"orig_{metric}"] - frame[f"o3_{metric}"]) / frame[f"orig_{metric}"].replace(0, np.nan)
        groups = {g: per[frame["size_group"] == g].mean() for g in GROUPS}
        lines.append(f"  {metric:9s} mean {per.mean():6.2f}  median {per.median():6.2f}  total "
                     f"{total:6.2f}  | small {groups['small']:6.2f} medium {groups['medium']:6.2f} "
                     f"large {groups['large']:6.2f}  | O3 alone mean {o3.fillna(0).mean():6.2f}")
    two = logical("two_qubit").fillna(0.0)
    lines.append(f"  2q unchanged on {int((two == 0).sum())}/{len(frame)} circuits; "
                 f">=10% on {int((two >= 10).sum())}; >=30% on {int((two >= 30).sum())}")

    lines.append("\nStage B -- AQORN cumulative reduction vs its own input's baseline "
                 "(fair = every stage layer-sorted), caption convention:")
    for variant in VARIANTS:
        for stage in ("h_only", "h_merge", "packed"):
            for key, label in (("one", "1q"), ("two", "2q"), ("depth", "depth")):
                parts = []
                for tag in ("ori", "opt"):
                    means = _group_means(frame, f"{tag}_{variant}_fair_{stage}_{key}_red")
                    parts.append(f"{tag} " + " ".join(f"{g[:1]}={means[g]:6.2f}" for g in GROUPS)
                                 + f" all={means['all']:6.2f}")
                if stage == "packed" or key == "depth":
                    lines.append(f"  {variant:5s} {stage:8s} {label:6s} " + " | ".join(parts))
    lines.append("\nStage B -- absolute physical counts (corpus totals, fair depth):")
    for variant in VARIANTS:
        for key, label in (("one", "1q"), ("two", "2q"), ("depth", "depth")):
            ob, op = frame[f"ori_{variant}_fair_base_{key}"].sum(), frame[f"ori_{variant}_fair_packed_{key}"].sum()
            pb, pp = frame[f"opt_{variant}_fair_base_{key}"].sum(), frame[f"opt_{variant}_fair_packed_{key}"].sum()
            ratio_opt_alone = 100 * (1 - (frame[f"opt_{variant}_fair_base_{key}"] /
                                          frame[f"ori_{variant}_fair_base_{key}"])).mean()
            ratio_aqorn_alone = 100 * (1 - (frame[f"ori_{variant}_fair_packed_{key}"] /
                                            frame[f"ori_{variant}_fair_base_{key}"])).mean()
            ratio_both = 100 * (1 - (frame[f"opt_{variant}_fair_packed_{key}"] /
                                     frame[f"ori_{variant}_fair_base_{key}"])).mean()
            lines.append(f"  {variant:5s} {label:6s} baseline {ob:9.0f} -> {pb:9.0f} (optimiser)  "
                         f"packed {op:9.0f} -> {pp:9.0f}  | per-circuit mean vs original "
                         f"baseline: optimiser alone {ratio_opt_alone:6.2f}%, AQORN alone "
                         f"{ratio_aqorn_alone:6.2f}%, both {ratio_both:6.2f}%")
    if any(c.endswith("_miscompiled") for c in frame.columns):
        flagged = [(n, c) for c in frame.columns if c.endswith("_miscompiled")
                   for n in frame.loc[frame[c].fillna(False).astype(bool), "name"]]
        lines.append(f"  noiseless miscompile check: {len(flagged)} flagged {flagged}")

    lines.append("\nStage C -- simulated, packed vs its own input's baseline, Fig 6 alpha and "
                 "seeds (TVD on resolved circuits only):")
    resolved = frame["tvd_resolved"].astype(bool)
    for variant in VARIANTS:
        if f"opt_{variant}_lsr_impr" not in frame.columns:
            continue
        have = frame[f"opt_{variant}_lsr_impr"].notna()
        for metric, mask in (("lsr_impr", have), ("tvd_red", have & resolved)):
            parts = []
            for tag in ("ori", "opt"):
                means = _group_means(frame, f"{tag}_{variant}_{metric}", mask)
                parts.append(f"{tag} " + " ".join(f"{g[:1]}={means[g]:6.2f}" for g in GROUPS)
                             + f" all={means['all']:6.2f} (n={means['n']})")
            delta = (frame.loc[mask, f"opt_{variant}_{metric}"] - frame.loc[mask, f"ori_{variant}_{metric}"])
            sim = mask & frame["changed"].astype(bool)
            lines.append(f"  {variant:5s} {metric:8s} " + " | ".join(parts)
                         + f" | paired delta mean {delta.mean():+.2f} (changed only "
                           f"{(frame.loc[sim, f'opt_{variant}_{metric}'] - frame.loc[sim, f'ori_{variant}_{metric}']).mean():+.2f}, n={int(sim.sum())})")
    lines.append("\nStage C -- baseline fidelity, optimised vs original input (same alpha, seeds):")
    if "opt_baseline_lsr" in frame.columns:
        sim = frame["changed"].astype(bool) & frame["opt_baseline_lsr"].notna()
        lsr = 100 * (frame.loc[sim, "opt_baseline_lsr"] / frame.loc[sim, "ori_baseline_lsr"] - 1)
        tv = sim & resolved
        tvd = 100 * (1 - frame.loc[tv, "opt_baseline_tvd"] / frame.loc[tv, "ori_baseline_tvd"])
        lines.append(f"  optimiser alone (baseline arm): LSR {lsr.mean():+.2f}% (n={int(sim.sum())}), "
                     f"TVD reduction {tvd.mean():+.2f}% (n={int(tv.sum())})")
    lines.append("\n" + tables(frame))
    text = "\n".join(lines)
    print(text)
    return text


def _boot_ci(values: pd.Series, draws: int = 10_000, seed: int = 0) -> Tuple[float, float]:
    """95% percentile bootstrap interval of the mean, resampling circuits."""
    v = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(draws, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _cell(value: float, digits: int = 1) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def tables(frame: pd.DataFrame) -> str:
    """The report's tables, as markdown, all in the caption convention (unclipped means)."""
    out: List[str] = []
    resolved = frame["tvd_resolved"].astype(bool)
    changed = frame["changed"].astype(bool)

    def logical(metric: str, tag: str = "opt") -> pd.Series:
        return (100.0 * (frame[f"orig_{metric}"] - frame[f"{tag}_{metric}"])
                / frame[f"orig_{metric}"].replace(0, np.nan))

    out.append("Table A. Logical reduction from optimisation, per-circuit mean % (two circuits "
               "with no 2q gates count as 0 for 2q).")
    out.append("| group | circuits | changed | 1q | 2q | all gates | depth | 2q, O3 only | "
               "all gates, O3 only |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for group in GROUPS + ("all",):
        mask = frame["size_group"] == group if group != "all" else pd.Series(True, index=frame.index)
        cells = [logical(m)[mask].fillna(0).mean() for m in ("one_qubit", "two_qubit", "total", "depth")]
        cells += [logical(m, "o3")[mask].fillna(0).mean() for m in ("two_qubit", "total")]
        out.append(f"| {group} | {int(mask.sum())} | {int((mask & changed).sum())} | "
                   + " | ".join(_cell(c) for c in cells) + " |")

    for variant in VARIANTS:
        if f"ori_{variant}_fair_packed_two_red" not in frame.columns:
            continue
        out.append(f"\nTable B-{variant}. Full pipeline vs its own input's baseline, cumulative % "
                   f"reduction, fair depth (every stage layer-sorted).")
        out.append("| metric | input | small | medium | large | all | delta all [95% CI] |")
        out.append("|---|---|---|---|---|---|---|")
        for key, label in (("one", "1q gates"), ("two", "2q gates"), ("depth", "depth")):
            for tag, name in (("ori", "original"), ("opt", "optimized")):
                means = _group_means(frame, f"{tag}_{variant}_fair_packed_{key}_red")
                delta = ""
                if tag == "opt":
                    d = (frame[f"opt_{variant}_fair_packed_{key}_red"]
                         - frame[f"ori_{variant}_fair_packed_{key}_red"])
                    lo, hi = _boot_ci(d)
                    delta = f"{d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]"
                out.append(f"| {label} | {name} | " + " | ".join(
                    _cell(means[g]) for g in GROUPS + ("all",)) + f" | {delta} |")

    if "ori_w10_fair_h_only_two_red" in frame.columns:
        out.append("\nTable B-stages (w10). Cumulative % reduction after each stage, all circuits, "
                   "fair depth.")
        out.append("| stage | 1q orig | 1q opt | 2q orig | 2q opt | depth orig | depth opt |")
        out.append("|---|---|---|---|---|---|---|")
        for stage, label in (("h_only", "+H"), ("h_merge", "+H+merge"), ("packed", "packed")):
            cells = []
            for key in METRIC_KEYS:
                for tag in ("ori", "opt"):
                    cells.append(frame[f"{tag}_w10_fair_{stage}_{key}_red"].mean())
            out.append(f"| {label} | " + " | ".join(_cell(c) for c in cells) + " |")

    out.append("\nTable C. Simulated, full pipeline vs its own input's baseline (Fig 6 alpha, "
               "shots and seeds); TVD on circuits both inputs resolve (baseline TVD <= 0.95).")
    out.append("| variant | metric | circuits | original | optimized | delta [95% CI] | "
               "changed circuits: original | optimized | delta |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for variant in VARIANTS:
        column = f"opt_{variant}_lsr_impr"
        if column not in frame.columns:
            continue
        have = frame[column].notna() & frame[f"ori_{variant}_lsr_impr"].notna()
        if not (have & changed).any():
            continue
        for metric, label, mask in (("lsr_impr", "LSR improvement %", have),
                                    ("tvd_red", "TVD reduction %", have & resolved)):
            o, p = frame.loc[mask, f"ori_{variant}_{metric}"], frame.loc[mask, f"opt_{variant}_{metric}"]
            d = p - o
            lo, hi = _boot_ci(d)
            sub = mask & changed
            so, sp = frame.loc[sub, f"ori_{variant}_{metric}"], frame.loc[sub, f"opt_{variant}_{metric}"]
            out.append(f"| {variant} | {label} | {int(mask.sum())} | {_cell(o.mean())} | "
                       f"{_cell(p.mean())} | {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | "
                       f"{_cell(so.mean())} (n={int(sub.sum())}) | {_cell(sp.mean())} | "
                       f"{(sp - so).mean():+.1f} |")

    out.append("\nTable D. Who does the work: per-circuit mean % against the ORIGINAL input's "
               "baseline (fair depth; LSR as improvement, TVD as reduction).")
    out.append("| variant | metric | optimizer alone | AQORN alone | optimizer + AQORN |")
    out.append("|---|---|---|---|---|")
    for variant in VARIANTS:
        if f"opt_{variant}_fair_base_two" not in frame.columns:
            continue
        for key, label in (("one", "1q gates"), ("two", "2q gates"), ("depth", "depth")):
            base = frame[f"ori_{variant}_fair_base_{key}"]
            alone = 100 * (1 - frame[f"opt_{variant}_fair_base_{key}"] / base)
            aqorn = 100 * (1 - frame[f"ori_{variant}_fair_packed_{key}"] / base)
            both = 100 * (1 - frame[f"opt_{variant}_fair_packed_{key}"] / base)
            out.append(f"| {variant} | {label} | {_cell(alone.mean())} | {_cell(aqorn.mean())} | "
                       f"{_cell(both.mean())} |")
        if f"opt_{variant}_lsr" in frame.columns and frame[f"opt_{variant}_lsr"].notna().all():
            base = frame["ori_baseline_lsr"]
            alone = 100 * (frame["opt_baseline_lsr"] / base - 1)
            aqorn = 100 * (frame[f"ori_{variant}_lsr"] / base - 1)
            both = 100 * (frame[f"opt_{variant}_lsr"] / base - 1)
            out.append(f"| {variant} | LSR | {_cell(alone.mean())} | {_cell(aqorn.mean())} | "
                       f"{_cell(both.mean())} |")
            base = frame.loc[resolved, "ori_baseline_tvd"]
            alone = 100 * (1 - frame.loc[resolved, "opt_baseline_tvd"] / base)
            aqorn = 100 * (1 - frame.loc[resolved, f"ori_{variant}_tvd"] / base)
            both = 100 * (1 - frame.loc[resolved, f"opt_{variant}_tvd"] / base)
            out.append(f"| {variant} | TVD | {_cell(alone.mean())} | {_cell(aqorn.mean())} | "
                       f"{_cell(both.mean())} |")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------


def _outlier_cut(values: np.ndarray) -> float:
    """Lower axis limit that leaves out a lone extreme low point (ising_n98's -250% depth)."""
    ordered = np.sort(values[np.isfinite(values)])
    if len(ordered) < 4:
        return float(ordered[0]) if len(ordered) else 0.0
    spread = ordered[-1] - ordered[1]
    if ordered[1] - ordered[0] > spread:  # the lowest point is further from the rest than their span
        return float(ordered[1])
    return float(ordered[0])


def plot(frame: pd.DataFrame, settings: Settings) -> List[Path]:
    """
    (a) what the optimizer does to each circuit; (b) AQORN's gain on each input, per circuit.

    Full width (figure*), house style. Panel (b) is one planner (`--plot-variant`): x is the
    full pipeline's cumulative reduction (improvement for LSR) against the original input's own
    baseline, y the same on the optimized input; the dashed diagonal is "unchanged". Gate
    counts and depth are static (fair, every stage layer-sorted); LSR and TVD are Fig 6-style
    single samples at each circuit's Fig 6 alpha. TVD shows only circuits whose TVD both
    inputs resolve (`tvd_resolved`). A lone extreme point is cut off and labelled at the edge.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from experiments.shared.figures import FULL_WIDTH, save_figure, use_style

    use_style()
    variant = settings.plot_variant
    blue, dark, mid, light = "#0097fd", "#3b4c56", "#636363", "#bdbdbd"
    fig = plt.figure(figsize=(FULL_WIDTH, 4.1))
    grid = GridSpec(2, 4, figure=fig, height_ratios=(1.0, 1.2))
    ax = fig.add_subplot(grid[0, :])

    def reduction(metric: str) -> pd.Series:
        return (100.0 * (frame[f"orig_{metric}"] - frame[f"opt_{metric}"])
                / frame[f"orig_{metric}"].replace(0, np.nan)).fillna(0.0)

    two, total, depth = reduction("two_qubit"), reduction("total"), reduction("depth")
    order = np.lexsort((-depth.values, -total.values, -two.values))
    x = np.arange(len(frame))
    ax.bar(x, two.values[order], width=0.8, color=blue, label="2q gates", zorder=2)
    ax.plot(x, total.values[order], linestyle="none", marker="s", markersize=2.6, color=dark,
            label="all gates", zorder=3)
    ax.plot(x, depth.values[order], linestyle="none", marker="^", markersize=2.6, color=mid,
            label="depth", zorder=3)
    ax.axhline(0.0, color=mid, linestyle="--", linewidth=1.0, zorder=1)
    unchanged = int((~frame["changed"].astype(bool)).sum())
    ax.set_xlim(-0.8, len(frame) - 0.2)
    top_data = float(max(two.max(), total.max(), depth.max()))
    # Headroom above the data for the legend and the outlier names, so neither covers a bar.
    ax.set_ylim(min(0.0, float(min(two.min(), total.min(), depth.min()))) - 4.0, top_data * 1.42)
    ax.set_xticks([])
    ax.set_xlabel(f"Fig. 6 benchmark circuits, sorted by 2q reduction ({unchanged} of "
                  f"{len(frame)} unchanged by the optimizer)")
    ax.set_ylabel("logical reduction (%)")
    ax.legend(loc="upper right", ncol=3, handlelength=1.2, columnspacing=1.0)
    names = frame["name"].values[order]
    top = [f"{names[r]} {two.values[order][r]:.0f}%" for r in range(len(frame))
           if two.values[order][r] >= 45.0]
    if top:
        half = (len(top) + 1) // 2
        lines = ",  ".join(top[:half]) + ("\n" + ",  ".join(top[half:]) if top[half:] else "")
        ax.text(0.012, 0.975, lines, transform=ax.transAxes, fontsize=8, va="top", ha="left",
                color=dark, linespacing=1.15)
    bv = [i for i, name in enumerate(names) if name.startswith("bv_")]
    if bv:
        ax.text(float(np.mean(bv)), float(total.values[order][bv].max()) + 4.0,
                f"Bernstein-Vazirani ({len(bv)})", fontsize=8, ha="center", va="bottom",
                color=dark)
    ax.text(-0.075, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="left")

    panels = (
        (f"{variant}_fair_packed_two_red", "2q gates"),
        (f"{variant}_fair_packed_depth_red", "depth"),
        (f"{variant}_lsr_impr", "LSR"),
        (f"{variant}_tvd_red", "TVD"),
    )
    changed = frame["changed"].astype(bool)
    for index, (column, title) in enumerate(panels):
        axis = fig.add_subplot(grid[1, index])
        if f"ori_{column}" not in frame.columns or f"opt_{column}" not in frame.columns:
            axis.set_axis_off()
            continue
        mask = frame[f"ori_{column}"].notna() & frame[f"opt_{column}"].notna()
        if title == "TVD":
            mask &= frame["tvd_resolved"].astype(bool)
        xs, ys = frame.loc[mask, f"ori_{column}"], frame.loc[mask, f"opt_{column}"]
        both = np.concatenate([xs.values, ys.values])
        cut = _outlier_cut(np.minimum(xs.values, ys.values))
        lo = min(cut, 0.0)
        hi = float(np.nanmax(both))
        pad = 0.07 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        axis.plot([lo, hi], [lo, hi], color=mid, linestyle="--", linewidth=1.0, marker="none",
                  zorder=1)
        inside = mask & (frame[f"ori_{column}"] >= lo) & (frame[f"opt_{column}"] >= lo)
        same, diff = inside & ~changed, inside & changed
        axis.plot(frame.loc[same, f"ori_{column}"], frame.loc[same, f"opt_{column}"],
                  linestyle="none", marker="o", markersize=3.2, markerfacecolor="none",
                  markeredgecolor=light, markeredgewidth=0.7, label="unchanged", zorder=2)
        axis.plot(frame.loc[diff, f"ori_{column}"], frame.loc[diff, f"opt_{column}"],
                  linestyle="none", marker="o", markersize=3.2, color=blue, alpha=0.85,
                  label="optimized", zorder=3)
        clipped = frame.loc[mask & ~inside]
        for number, (_, row) in enumerate(clipped.iterrows()):
            # Under the mean, where no panel has points (nobody gains 30+ points from it).
            axis.text(0.04, 0.80 - 0.16 * number,
                      f"off scale:\n{row['name']}\n({row[f'ori_{column}']:.0f}, "
                      f"{row[f'opt_{column}']:.0f})", transform=axis.transAxes, fontsize=8,
                      color=dark, ha="left", va="top")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, axis="both")
        axis.set_title(title, fontsize=10)
        axis.text(0.04, 0.97, f"mean {xs.mean():.1f}\n\u2192 {ys.mean():.1f}",
                  transform=axis.transAxes, fontsize=8, va="top", ha="left")
        axis.locator_params(nbins=4)
        if index == 0:
            axis.set_ylabel("optimized input (%)")
            axis.text(-0.42, 1.06, "(b)", transform=axis.transAxes, fontweight="bold",
                      fontsize=10, va="bottom", ha="left")
            axis.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), handletextpad=0.1,
                        borderaxespad=0.2, fontsize=8, handlelength=1.0)
        axis.set_xlabel("original input (%)")
    paths = save_figure(fig, f"{settings.figure_stem}_{variant}", Path(__file__),
                        override=Path(settings.figure_dir) if settings.figure_dir else None,
                        versioned=True)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[{NAME}] {path} missing; run its stage first")
    return pd.read_csv(path)


def run(settings: Settings) -> None:
    from experiments.revision.common import write_sidecar

    if settings.from_csv:
        frame = pd.read_csv(settings.from_csv)
        summarize(frame)
        for path in plot(frame, settings):
            log(f"saved {path}")
        return
    settings.out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    circuits = _load_circuits(settings)
    a = stage_a(settings, circuits) if "A" in settings.stages else _read(settings.out / "stage_a.csv")
    b = (stage_b(settings, circuits, a) if "B" in settings.stages
         else _read(settings.out / "stage_b.csv"))
    if "C" in settings.stages:
        c = stage_c(settings, circuits, a, b)
    else:
        path = settings.out / "stage_c.csv"
        c = pd.read_csv(path) if path.exists() else None
    frame = combine(settings, a, b, c)
    path = settings.out / f"{NAME}.csv"
    if path.exists():
        path = settings.out / f"{NAME}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    frame.to_csv(path, index=False)
    write_sidecar(path, experiment=NAME, settings=asdict(settings), started=began,
                  status="complete")
    log(f"wrote {path}")
    summary = summarize(frame)
    (path.with_suffix(".summary.txt")).write_text(summary + "\n")
    for figure in plot(frame, settings):
        log(f"saved {figure}")


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1])
    parser.add_argument("--stages", nargs="*", default=["A", "B", "C"], choices=["A", "B", "C"],
                        help="Stages to compute; the others are read from --output-dir. "
                             "Give the flag with no stages to recombine, summarise and plot.")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--from-csv", default=None,
                        help="Summary and figure from a cached logical_opt.csv; no computation.")
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--sim-arms", nargs="+", default=["baseline", "w10", "w10T"],
                        choices=["baseline", "v4", "w10", "w10T"])
    parser.add_argument("--plot-variant", default="w10", choices=sorted(VARIANTS))
    parser.add_argument("--figure-dir", default=None,
                        help="Write the figure here instead of results/figures/ (smoke runs).")
    parser.add_argument("--figure-stem", default="logical_preoptimization",
                        help="Figure file name stem; the plotted planner is appended.")
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--no-noiseless-check", dest="noiseless_check", action="store_false")
    parser.add_argument("--seed", type=int, default=0, help="Equivalence-check random states.")
    parser.add_argument("--routes", nargs="+", default=[],
                        choices=["o3", "o3_sx", "clifford_ag", "clifford_greedy",
                                 "clifford_blocks", "linear_blocks", "commute"],
                        help="Restrict the optimiser to these routes (default: all).")
    args = parser.parse_args()
    return Settings(stages=tuple(args.stages), jobs=args.jobs, output_dir=args.output_dir,
                    from_csv=args.from_csv, names=tuple(args.names),
                    sim_arms=tuple(args.sim_arms), plot_variant=args.plot_variant,
                    figure_dir=args.figure_dir, figure_stem=args.figure_stem,
                    skip_gates=args.skip_gates, noiseless_check=args.noiseless_check,
                    seed=args.seed, routes=tuple(args.routes))


if __name__ == "__main__":
    run(parse_args())
