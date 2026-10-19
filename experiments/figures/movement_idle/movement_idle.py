"""
movement_idle.py -- Fig 6's ablation measured in atom movement and qubit idle time.

Fig 6 (`merging_improvement_benchmark_stylized_simulated`) stacks what each compiler stage
removes -- H commutation, + merging, + packing, cumulative against the naive baseline (tight
trivial packing, both passes off) -- in gate counts, depth and two simulated fidelity
metrics. This keeps its 71 circuits, plans, stages and plotting and measures the scheduled
physical circuit instead, in three amounts that depth does not see:

    movement  atoms carried: one count per atom per carriage (out and back), over every
              cross-patch transport group, intra-patch beyond-blockade move and
              swap-by-exchange. It is what the simulator charges `move_error` on, and that
              charge is 97-98% of the movement error on this corpus; flight time is kept as
              `flight_atom_seconds` (the rest, flight dephasing; at 1 m/s, 1 us = 1 um).
    idle      qubit-seconds of idling over the whole array: every physical qubit, every
              layer, minus the seconds it works. Split by what made each layer long (gate,
              transport, readout) as `simulator.scheduler.idle_attribution` does. Stages of
              one circuit share the physical qubit count, so per qubit is the same percent.
    makespan  the schedule's wall-clock length, readout included (one 3 ms image at the
              end, the same in every stage; `compute_seconds` is the time before it).

The schedule is the simulator's own -- `stim_backend._scan`, `transport.derive_transport_
groups`, `stim_backend._build_units`, `scheduler.build_layers` / `idle_attribution`, run as
`simulator.attribution.measured_noise_split` runs them -- and `schedule_amounts` returns the
amounts that function prices. The amounts do not depend on alpha (it scales rates, not
times), so the error terms at Fig 6's per-circuit alpha are `alpha_final` times the `E_*`
columns (alpha = 1) and every percent here holds at any alpha.

Checked on every run, before anything is written: the shipped-ordering 1q / 2q / depth of
the originals reproduce the source CSV exactly (`static_reproduced`); the layer-sorted depth
reproduces `ablation_fair_depth`; on every proxy schedule amount x rate reproduces
`measured_noise_split`'s error terms exactly (`split_max_abs_dev`); every original schedules
identically under a second set of stand-ins (`standin_invariant`); and for v4 / w10 the
packed proxies' E_* match Experiment8a's `planner_static.csv`, an independent code path.

Sensitivity (`MOVE_OVERHEADS`, columns `*_h20us` / `*_h50us`): the transport model charges
a carriage its flight time only, so pick-up / put-down time is free; `with_move_overhead`
adds a fixed 20 or 50 us per carriage. Not the simulator's model -- a check on how much the
idle conclusions lean on it.

Ordering. Every stage, baseline included, is translated with `TranslatorConfig.asap_order`
on (layer-sorted before translation): the merging pass re-emits the program in ASAP-layer
order whenever a macro fires, so without it the merging and packing stages are credited
with a scheduling gain the baseline could have had for free (Fig 6 depth: 28.9% shipped,
15.5% layer-sorted; see `ablation_fair_depth`). Those are the primary numbers; the shipped
ordering (`asap_order` False, what Fig 6 measured) is kept as `ordering == "shipped"`.

Originals and proxies. Fig 6 counts gates and depth on the original circuits and simulates
Clifford proxies (rz/rx -> s before translation). Both are scheduled here. Stim's scanner
rejects rz/rx, so for an original each physical rz/rx is replaced *after* translation by a
same-duration Clifford stand-in (`SCHEDULE_STANDINS`): the scheduler reads only an
instruction's kind and qubits -- every one-qubit record lasts `t_1q`, none is virtual, the
name matters only for `swap` -- so timing, transport groups and idling are unchanged. The
primary figure is on originals, consistent with Fig 6's gate and depth panels.

Plans (`--source`) are the template's (`ablation_fair_depth`): v4 = the submitted Fig 6 with
AQORN's plans; w10 = the weight-10 silent-aware planner; w10T = weight 10 + transversal
in-patch CZ/XCX (switch on in the merging stages only).

Deviation from house style, inherited on purpose: Fig 6's own `plot_results` draws the
figure (22 in canvas, its own size ladder), so it reads beside Fig 6. What an amount that
can grow needs on top (see `draw`): a circuit whose amount grew over the three stages gets
a hatched bar below zero, cut at `FLOOR` and labelled; the average bar is a waterfall of
the per-stage means at the caption convention (unclipped mean of the cumulative per-circuit
reductions), where Fig 6 averages clipped stacks.

Run from the repository root (each source takes about a minute at --jobs 4):

    PYTHONPATH=. venv/bin/python experiments/figures/movement_idle/movement_idle.py --source v4 --jobs 4
    ... --source w10 --jobs 4
    ... --source w10T --jobs 4
    ... --from-csv experiments/figures/movement_idle/movement_idle_v4.csv          # redraw
    ... --from-csv <csv> --ordering shipped    # or --program proxy: the other views
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
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "movement_idle"
STAGES = ("base", "h_only", "h_merge", "packed")
COMPONENTS = ("h_commutation", "merging", "packing")
STAGE_OF = {"h_commutation": "h_only", "merging": "h_merge", "packing": "packed"}
#: ordering label -> `TranslatorConfig.asap_order`
ORDERINGS = {"fair": True, "shipped": False}
PROGRAMS = ("original", "proxy")
SOURCE_STATIC = {
    "one_qubit_gates": "pct_reduction_one_qubit_gates",
    "two_qubit_gates": "pct_reduction_two_qubit_gates",
    "depth": "pct_reduction_circuit_depth",
}
FAIR_DEPTH_CSV = "experiments/figures/ablation_fair_depth/ablation_fair_depth_{source}.csv"
#: Experiment8a's static table: E_* of each plan's proxy (packed stage, shipped ordering,
#: alpha = 1) -- an independent path to the same numbers for v4 (weight 0) and w10.
PLANNER_STATIC_WEIGHT = {"v4": 0.0, "w10": 10.0}

#: Same-duration Clifford stand-ins for the rotations Stim's scanner rejects (see the module
#: docstring for why the schedule cannot tell the difference). Every original is also
#: scheduled with the alternates and must come out identical (`standin_invariant`).
SCHEDULE_STANDINS = {"rz": "s", "rx": "sx"}
ALTERNATE_STANDINS = {"rz": "z", "rx": "x"}

#: `with_move_overhead` sensitivity: seconds of pick-up / put-down per carriage. Not the
#: simulator's model (which charges flight time only); kept as `<amount>_h<us>us` columns.
MOVE_OVERHEADS = (20e-6, 50e-6)
OVERHEAD_AMOUNTS = ("idle_qubit_seconds", "makespan_seconds", "compute_seconds")

#: Amount -> (figure column, panel label). The panels, top to bottom.
PANELS = {
    "atoms_carried": ("pct_reduction_atoms_carried", "#atom moves"),
    "idle_qubit_seconds": ("pct_reduction_idle_time", "qubit idle time"),
    "makespan_seconds": ("pct_reduction_makespan", "makespan"),
}
#: Every amount that gets a reduction column: the panels, and the context the report quotes.
REDUCED = (
    "atoms_carried", "flight_atom_seconds", "move_units", "transport_groups",
    "transport_layer_seconds", "idle_qubit_seconds", "idle_per_qubit",
    "idle_transport_seconds", "idle_gate_seconds", "makespan_seconds", "compute_seconds",
    "E_movement", "E_idle", "E_total", "one_qubit_gates", "two_qubit_gates", "depth",
) + tuple(f"{key}_h{round(seconds * 1e6)}us" for seconds in MOVE_OVERHEADS
          for key in OVERHEAD_AMOUNTS)
#: Carried into every row from the source figure, so the cache stands alone.
META = ("size_group", "circuit_idx", "circuit_name", "num_qubits", "target_codes",
        "baseline_gate_count", "alpha_final", "baseline_lsr", "baseline_tvd")


@dataclass
class Settings:
    source: Optional[str] = "v4"
    jobs: Optional[int] = 4
    output_dir: Optional[str] = None
    from_csv: Optional[str] = None
    names: Optional[Tuple[str, ...]] = None
    program: str = "original"
    ordering: str = "fair"

    @property
    def stem(self) -> str:
        stem = f"{NAME}_{self.source}"
        if self.names:
            stem += "_smoke"
        return stem

    @property
    def figure_stem(self) -> str:
        stem = self.stem
        if self.program != "original":
            stem += f"_{self.program}"
        if self.ordering != "fair":
            stem += f"_{self.ordering}"
        return stem


# ----------------------------------------------------------------------------------------
# the schedule, as amounts
# ----------------------------------------------------------------------------------------


def schedulable(circuit, standins: Optional[Dict[str, str]] = None) -> Tuple[object, int]:
    """
    `circuit` with every rz/rx replaced by its same-duration Clifford stand-in, and the count.

    Registers are kept (`copy_empty_like`), so the patch map and atom layout -- both read
    off register names -- are the original's. Clifford circuits come back unchanged.
    """
    from qiskit.circuit.library import SGate, SXGate, XGate, ZGate

    standins = SCHEDULE_STANDINS if standins is None else standins
    gates = {"s": SGate, "sx": SXGate, "x": XGate, "z": ZGate}
    if not any(inst.operation.name.lower() in standins for inst in circuit.data):
        return circuit, 0
    out = circuit.copy_empty_like()
    replaced = 0
    for inst in circuit.data:
        name = inst.operation.name.lower()
        if name in standins:
            out.append(gates[standins[name]](), inst.qubits, inst.clbits)
            replaced += 1
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out, replaced


def _noise_overrides(params, idle_time_scalar: float, distance_scalar: float) -> Dict:
    """`measured_noise_split`'s overrides, spelled the same way."""
    return {
        "T1": params.T1, "T2": params.T2,
        "t_1q": params.one_qubit_gate_time, "t_2q": params.two_qubit_gate_time,
        "t_meas": params.meas_time, "p1": params.one_qubit_error,
        "p2": params.two_qubit_error, "p_meas": params.readout_error,
        "atom_pitch": params.atom_pitch, "blockade_radius": params.blockade_radius,
        "movement_speed": params.movement_speed, "patch_gutter": params.patch_gutter,
        "move_error": params.move_error,
        "idle_time_scalar": idle_time_scalar, "distance_scalar": distance_scalar,
    }


def _units(circuit, params, idle_time_scalar: float, distance_scalar: float):
    """`measured_noise_split`'s schedule up to the units: (records, units)."""
    from iceberg.layout import patch_of_qubit
    from simulator import stim_backend as sb
    from simulator.grid_constructor import centres_for_circuit, positions_for_circuit
    from simulator.transport import derive_transport_groups

    with sb._active_params(_noise_overrides(params, idle_time_scalar, distance_scalar)):
        geometry = sb._geometry()
        layout = positions_for_circuit(circuit, geometry=geometry)
        patch_map = patch_of_qubit(circuit)
        records = sb._scan(circuit)
        groups = derive_transport_groups(
            [frozenset(patch_map[q] for q in r.qubits if q in patch_map) for r in records],
            [r.qubits for r in records],
            patch_map,
            centres_for_circuit(circuit, geometry=geometry),
            geometry,
        )
        units = sb._build_units(records, groups, layout, patch_map, geometry)
    return records, units


def with_move_overhead(units, seconds: float):
    """
    Sensitivity, not the simulator's model: every carriage also takes `seconds` of handling.

    `simulator.transport` charges a carriage its flight time only -- picking an atom up and
    putting it down (trap handoff, ramps, settling) costs infidelity (`move_error`) but no
    time, and its docstring notes that demonstrated systems spend tens of microseconds on
    it, comparable to the 30-60 us flights. This adds `seconds` to every unit that carries
    atoms -- a transport group, an intra-patch mover, a swap-by-exchange -- as more flight:
    the unit is longer, the carried atoms are busy for it, and everyone else idles. Default
    0 returns `units` untouched.
    """
    import dataclasses

    if seconds <= 0:
        return units
    out = []
    for unit in units:
        if not unit.moved:
            out.append(unit)
            continue
        busy = unit.busy
        if busy is not None:
            busy = dict(busy)
            for qubit in unit.moved:
                if qubit in busy:
                    busy[qubit] += seconds
        out.append(dataclasses.replace(unit, duration=unit.duration + seconds,
                                       move_time=unit.move_time + seconds, busy=busy))
    return out


def schedule_amounts(
    circuit, params=None, *, idle_time_scalar: float = 1.0, distance_scalar: float = 1.0,
    move_overhead: float = 0.0, units=None,
) -> Dict[str, float]:
    """
    The schedule `simulator.attribution.measured_noise_split` prices, as physical amounts.

    Same steps, same order, same parameters (times and geometry are all that matter here;
    rates enter only in `expected_errors`). `circuit` must be Clifford -- pass an original
    through `schedulable` first. `units` may be passed in, as `(records, units)` from
    `_units`, to reuse one build; `move_overhead` is the `with_move_overhead` sensitivity.

    Movement: `atoms_carried` counts atoms per carriage over every unit that carries any
    (`ScheduleUnit.moved`: a transport group's control side, the mover of an intra-patch
    beyond-blockade gate, both ends of a swap-by-exchange); `flight_atom_seconds` weights
    each by its flight time. `move_units` / `move_seconds` count the carriages themselves
    (an AOD move of several atoms is one), `transport_groups` the cross-patch ones.
    Transport-paced time: `transport_layers` / `transport_layer_seconds` are the layers whose
    slowest unit carries atoms -- the time the array spends waiting on movement.

    Idling: qubit-seconds by cause, from `idle_attribution` (marginal: a transport layer is
    charged only for the idling it adds over what the layer would have cost anyway).
    `idle_qubit_seconds` is their sum, which is the array's total idling.
    """
    from simulator.noise_models import StimNoiseParameters
    from simulator.scheduler import build_layers, idle_attribution

    params = params or StimNoiseParameters()
    if units is None:
        units = _units(circuit, params, idle_time_scalar, distance_scalar)
    records, units = units
    units = with_move_overhead(units, move_overhead)
    layers = build_layers(units, circuit.num_qubits)
    attributed = idle_attribution(layers, circuit.num_qubits)

    # The two sums `measured_noise_split` prices, in its order, so the check is exact.
    flight = sum(
        unit.move_time * len(unit.moved)
        for layer in layers for unit in layer.units if unit.moved
    )
    carried = sum(
        len(unit.moved)
        for layer in layers for unit in layer.units if unit.moved
    )
    carriers = [unit for layer in layers for unit in layer.units if unit.moved]
    paced = [layer for layer in layers
             if layer.units and max(layer.units, key=lambda u: u.duration).moved]
    readout = [layer for layer in layers
               if layer.units and all(unit.cause == "readout" for unit in layer.units)]
    makespan = sum(layer.duration for layer in layers)
    readout_seconds = sum(layer.duration for layer in readout)
    idle = {cause: float(attributed.get(cause, 0.0)) for cause in ("gate", "transport", "readout")}
    unknown = set(attributed) - set(idle)
    if unknown:
        raise RuntimeError(f"idle_attribution returned unexpected causes {sorted(unknown)}")
    counts = {kind: sum(1 for r in records if r.kind == kind and r.noise)
              for kind in ("one_q", "two_q", "measure")}
    return {
        "atoms_carried": float(carried),
        "flight_atom_seconds": float(flight),
        "move_units": float(len(carriers)),
        "move_seconds": float(sum(unit.move_time for unit in carriers)),
        "transport_groups": float(sum(1 for unit in carriers if unit.transport is not None)),
        "group_atoms_carried": float(sum(len(unit.moved) for unit in carriers
                                         if unit.transport is not None)),
        "transport_layers": float(len(paced)),
        "transport_layer_seconds": float(sum(layer.duration for layer in paced)),
        "layers": float(len(layers)),
        "makespan_seconds": float(makespan),
        "readout_seconds": float(readout_seconds),
        "compute_seconds": float(makespan - readout_seconds),
        "idle_gate_seconds": idle["gate"],
        "idle_transport_seconds": idle["transport"],
        "idle_readout_seconds": idle["readout"],
        "idle_qubit_seconds": idle["gate"] + idle["transport"] + idle["readout"],
        "physical_qubits": float(circuit.num_qubits),
        "idle_per_qubit": (idle["gate"] + idle["transport"] + idle["readout"])
                          / max(1, circuit.num_qubits),
        "one_q_records": float(counts["one_q"]),
        "two_q_records": float(counts["two_q"]),
        "measure_records": float(counts["measure"]),
        "swap_records": float(sum(1 for r in records if r.name == "swap" and r.noise)),
    }


def expected_errors(
    amounts: Dict[str, float], params, *, idle_time_scalar: float = 1.0,
    distance_scalar: float = 1.0,
) -> Dict[str, float]:
    """Amounts x rates, term for term as `measured_noise_split` prices them."""
    from simulator.stim_backend import thermal_error_rate

    idle_rate = thermal_error_rate(params.T1, params.T2)
    move_error = min(1.0, params.move_error * distance_scalar)
    split = {
        "gates": (amounts["one_q_records"] * params.one_qubit_error
                  + (amounts["two_q_records"] - amounts["swap_records"])
                  * params.two_qubit_error),
        "measurement": amounts["measure_records"] * params.readout_error,
        "idle_gates": amounts["idle_gate_seconds"] * idle_time_scalar * idle_rate,
        "idle_transport": amounts["idle_transport_seconds"] * idle_time_scalar * idle_rate,
        "idle_readout": amounts["idle_readout_seconds"] * idle_time_scalar * idle_rate,
        "move_handling": amounts["atoms_carried"] * move_error,
        "flight_dephasing": amounts["flight_atom_seconds"] * idle_rate,
    }
    split["movement"] = split["move_handling"] + split["flight_dephasing"]
    split["idle"] = split["idle_gates"] + split["idle_transport"] + split["idle_readout"]
    split["total"] = sum(
        split[key] for key in
        ("gates", "measurement", "idle_gates", "idle_transport", "idle_readout",
         "move_handling", "flight_dephasing")
    )
    return split


# ----------------------------------------------------------------------------------------
# compute
# ----------------------------------------------------------------------------------------


def _translate(program, permutation, *, commute: bool, merge: bool, asap: bool,
               switches: Sequence[str]):
    """One stage through Fig 6's own `compute_translated_circuit`, switches on merging only."""
    from pipeline.config import PipelineConfig

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from translator import explore_switches

    explore_switches.configure(**{name: merge for name in switches})
    try:
        config = PipelineConfig()
        config.translator_config.enable_h_commutation = commute
        config.translator_config.enable_merging = merge
        config.translator_config.asap_order = asap
        return fig6.compute_translated_circuit(program, permutation, config)
    finally:
        explore_switches.configure(**{name: False for name in switches})


def _run_circuit(job) -> List[Dict]:
    """Sixteen schedules for one circuit: {original, proxy} x {fair, shipped} x 4 stages."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from simulator.attribution import measured_noise_split
    from simulator.noise_models import StimNoiseParameters

    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    circuit, plan, switches, meta = job
    programs = {"original": fig6.ensure_measurements(circuit.original),
                "proxy": fig6.ensure_measurements(circuit.proxy)}
    n = programs["original"].num_qubits
    tight = fig6.tight_trivial_packing(n, math.ceil(n / 2))
    placement, commute = [tuple(p) for p in plan[0]], bool(plan[1])
    stages = {"base": (tight, False, False), "h_only": (tight, commute, False),
              "h_merge": (tight, commute, True), "packed": (placement, commute, True)}
    # Fig 6's simulator: NEUTRAL_ATOM at alpha = 1, distance and idle noise on, scalars 1.
    # alpha scales every rate and no time, so the amounts are alpha-free and the error
    # terms at Fig 6's per-circuit alpha are exactly alpha_final times these.
    params = StimNoiseParameters.neutral_atom()

    rows: List[Dict] = []
    for program_label, program in programs.items():
        for ordering, asap in ORDERINGS.items():
            for stage, (perm, commute_, merge) in stages.items():
                began = time.perf_counter()
                physical = _translate(program, perm, commute=commute_, merge=merge,
                                      asap=asap, switches=switches)
                one, two = one_two_qubit_gate_totals(physical)
                clifford, standins = schedulable(physical)
                if program_label == "proxy" and standins:
                    raise RuntimeError(f"{circuit.name}: proxy carries {standins} rz/rx")
                built = _units(clifford, params, 1.0, 1.0)
                amounts = schedule_amounts(clifford, params, units=built)
                errors = expected_errors(amounts, params)
                row = {
                    **meta, "program": program_label, "ordering": ordering,
                    "asap_order": asap, "stage": stage, "commute": commute_,
                    "merging": merge, "switches": ",".join(switches) if merge else "",
                    "one_qubit_gates": float(one), "two_qubit_gates": float(two),
                    "depth": float(physical.depth()), "rotation_standins": standins,
                    **amounts, **{f"E_{key}": value for key, value in errors.items()},
                }
                for overhead in MOVE_OVERHEADS:
                    tag = f"h{round(overhead * 1e6)}us"
                    extra = schedule_amounts(clifford, params, units=built,
                                             move_overhead=overhead)
                    for key in OVERHEAD_AMOUNTS:
                        row[f"{key}_{tag}"] = extra[key]
                if program_label == "proxy":
                    reference = measured_noise_split(physical, params)
                    row["split_max_abs_dev"] = max(
                        abs(errors[key] - reference[key]) for key in reference
                    )
                elif standins:
                    # The stand-in is arbitrary: another one-qubit Clifford must give the
                    # identical schedule, or the scheduler is reading gate names after all.
                    other, _ = schedulable(physical, ALTERNATE_STANDINS)
                    again = schedule_amounts(other, params)
                    row["standin_invariant"] = all(
                        again[key] == amounts[key] for key in amounts)
                row["seconds"] = time.perf_counter() - began
                rows.append(row)
    return rows


def _plans(source: str, frame: pd.DataFrame):
    from experiments.figures.ablation_fair_depth.ablation_fair_depth import _plans as plans

    return plans(source, frame)


def _cumulative(long: pd.DataFrame, column: str) -> pd.DataFrame:
    """circuit x stage table of cumulative % reduction against that circuit's baseline."""
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    wide = long.pivot_table(index="circuit_name", columns="stage", values=column,
                            aggfunc="first")
    out = pd.DataFrame(index=wide.index)
    for stage in STAGES[1:]:
        out[stage] = [fig6.pct_reduction(b, s) for b, s in zip(wide["base"], wide[stage])]
    return out


def _check_reproduction(long: pd.DataFrame, source: pd.DataFrame, settings: Settings) -> Dict:
    """The gates every run must pass before its numbers are trusted. Raises on a failure."""
    checks: Dict[str, object] = {}
    shipped = long[(long["program"] == "original") & (long["ordering"] == "shipped")]
    by = source.pivot_table(index="circuit_name", columns="component", aggfunc="first",
                            values=list(SOURCE_STATIC.values()))
    bad: Dict[str, List[str]] = {}
    for metric, column in SOURCE_STATIC.items():
        mine = _cumulative(shipped, metric)
        theirs = by[column][list(COMPONENTS)].cumsum(axis=1)
        theirs.columns = [STAGE_OF[c] for c in COMPONENTS]
        theirs = theirs.loc[mine.index]
        off = (mine[list(theirs.columns)] - theirs).abs().max(axis=1) > 1e-6
        if off.any():
            bad[metric] = sorted(off[off].index)
    circuits = shipped["circuit_name"].nunique()
    checks["static_reproduced"] = {metric: circuits - len(bad.get(metric, []))
                                   for metric in SOURCE_STATIC}
    if bad:
        raise SystemExit(f"[{NAME}] shipped 1q/2q/depth not reproduced: {bad}")

    fair_path = Path(FAIR_DEPTH_CSV.format(source=settings.source))
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        fair_rows = long[(long["program"] == "original") & (long["ordering"] == "fair")]
        mine = fair_rows.pivot_table(index="circuit_name", columns="stage", values="depth",
                                     aggfunc="first")
        mismatched = []
        for row in fair.itertuples():
            if row.circuit_name not in mine.index:
                continue
            stage = STAGE_OF[row.component]
            if (mine.loc[row.circuit_name, "base"] != row.fair_baseline_depth
                    or mine.loc[row.circuit_name, stage] != row.fair_stage_depth):
                mismatched.append(row.circuit_name)
        checks["fair_depth_matches_ablation_fair_depth"] = (
            int(len(set(fair["circuit_name"]) & set(mine.index)) - len(set(mismatched))))
        if mismatched:
            raise SystemExit(f"[{NAME}] layer-sorted depth differs from {fair_path} on "
                             f"{sorted(set(mismatched))}")
    else:
        checks["fair_depth_matches_ablation_fair_depth"] = "skipped (no CSV)"

    originals = long[long["program"] == "original"]
    flagged = originals["standin_invariant"].dropna() if "standin_invariant" in originals \
        else pd.Series(dtype=bool)
    checks["standin_invariant"] = f"{int(flagged.astype(bool).sum())}/{len(flagged)}"
    if not flagged.astype(bool).all():
        raise SystemExit(f"[{NAME}] an original's schedule depends on which stand-in "
                         f"replaced its rotations")

    proxies = long[long["program"] == "proxy"]
    worst = float(proxies["split_max_abs_dev"].max())
    checks["split_max_abs_dev"] = worst
    if not worst <= 1e-12:
        raise SystemExit(f"[{NAME}] amounts x rates differ from measured_noise_split by {worst}")

    weight = PLANNER_STATIC_WEIGHT.get(settings.source)
    if weight is not None:
        from experiments.figures.ablation_fair_depth.ablation_fair_depth import AQORN_PLANS

        static = pd.read_csv(AQORN_PLANS)
        static = static[static["silent_weight"] == weight].set_index("name")
        packed = proxies[(proxies["ordering"] == "shipped") & (proxies["stage"] == "packed")]
        packed = packed.set_index("circuit_name")
        keys = [c for c in static.columns if c.startswith("E_")]
        worst_rel = 0.0
        for name, row in packed.iterrows():
            for key in keys:
                ref = float(static.loc[name, key])
                worst_rel = max(worst_rel, abs(float(row[key]) - ref) / max(1e-12, abs(ref)))
        checks["planner_static_E_max_rel_dev"] = worst_rel
        if worst_rel > 1e-6:
            raise SystemExit(f"[{NAME}] packed proxy E_* differ from {AQORN_PLANS} "
                             f"(weight {weight}) by {worst_rel:.3g} relative")
    return checks


def compute(settings: Settings) -> Tuple[pd.DataFrame, Dict]:
    from experiments.figures.ablation_fair_depth.ablation_fair_depth import SOURCES
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
    if settings.names:
        circuits = [c for c in circuits if c.name in settings.names]
        source = source[source["circuit_name"].isin(settings.names)]
    meta = source.drop_duplicates("circuit_name").set_index("circuit_name")
    jobs = [(c, plans[c.name], tuple(switches),
             {key: (c.name if key == "circuit_name" else meta.loc[c.name, key])
              for key in META})
            for c in circuits]
    jobs.sort(key=lambda j: -j[0].num_qubits)
    print(f"[{NAME}] {settings.source}: {len(jobs)} circuits from {path}", flush=True)
    results = parallel_map(_run_circuit, jobs, jobs=settings.jobs, label=NAME)
    failed = [j[0].name for j, r in zip(jobs, results) if r is None]
    if failed:
        raise SystemExit(f"[{NAME}] {len(failed)} circuits failed: {failed}")
    long = pd.DataFrame([row for r in results for row in r])
    checks = _check_reproduction(long, source, settings)
    order = {"small": 0, "medium": 1, "large": 2}
    long["_g"] = long["size_group"].map(order)
    long["_s"] = long["stage"].map({s: i for i, s in enumerate(STAGES)})
    long = long.sort_values(["_g", "circuit_idx", "program", "ordering", "_s"]).drop(
        columns=["_g", "_s"])
    long.insert(0, "source", settings.source)
    return stamp_provenance(long), checks


# ----------------------------------------------------------------------------------------
# derived tables
# ----------------------------------------------------------------------------------------


def segments(long: pd.DataFrame, program: str, ordering: str) -> pd.DataFrame:
    """
    Fig 6's frame: one row per (circuit, component), each reduction column holding that
    component's segment (difference of cumulative % reductions against the baseline).
    """
    scoped = long[(long["program"] == program) & (long["ordering"] == ordering)]
    if scoped.empty:
        raise SystemExit(f"[{NAME}] no rows for program={program}, ordering={ordering}")
    meta = scoped.drop_duplicates("circuit_name").set_index("circuit_name")
    cumulative = {amount: _cumulative(scoped, amount) for amount in REDUCED}
    rows = []
    for name in meta.index:
        previous = {amount: 0.0 for amount in REDUCED}
        for component in COMPONENTS:
            stage = STAGE_OF[component]
            row = {key: meta.loc[name, key] for key in META if key != "circuit_name"}
            row.update({"circuit_name": name, "component": component,
                        "program": program, "ordering": ordering})
            for amount in REDUCED:
                now = float(cumulative[amount].loc[name, stage])
                column = PANELS[amount][0] if amount in PANELS else f"pct_reduction_{amount}"
                row[column] = now - previous[amount]
                previous[amount] = now
            rows.append(row)
    return pd.DataFrame(rows)


def _column(amount: str) -> str:
    return PANELS[amount][0] if amount in PANELS else f"pct_reduction_{amount}"


def _scoped(long: pd.DataFrame, program: str, ordering: str) -> pd.DataFrame:
    return long[(long["program"] == program) & (long["ordering"] == ordering)]


def _raw(scoped: pd.DataFrame, amount: str) -> pd.DataFrame:
    """circuit x stage table of the amount itself."""
    return scoped.pivot_table(index="circuit_name", columns="stage", values=amount,
                              aggfunc="first")[list(STAGES)]


def _direction(before: pd.Series, after: pd.Series) -> pd.Series:
    """+1 where `after` is larger, -1 where smaller, 0 where equal to rounding."""
    tolerance = 1e-9 * np.maximum(before.abs(), after.abs())
    return np.sign((after - before).where((after - before).abs() > tolerance, 0.0))


def caption_table(long: pd.DataFrame, program: str, ordering: str,
                  amount: str) -> pd.DataFrame:
    """
    Caption convention, per size group and overall: the unclipped mean over circuits of
    each circuit's cumulative % reduction after each stage (Fig 6's `pct_reduction`, which
    counts a zero baseline as 0), its median after packing, and how many circuits end with
    more (`worse`), less (`better`) or the same amount as the baseline -- counted on the
    amounts, so a circuit that starts from zero is counted where it belongs. `nonzero`
    repeats the +packing mean over circuits whose baseline is not zero (Experiment1a's
    convention); it differs only for movement, where three circuits move no atoms at all.
    """
    scoped = _scoped(long, program, ordering)
    raw = _raw(scoped, amount)
    cumulative = _cumulative(scoped, amount)
    groups = scoped.drop_duplicates("circuit_name").set_index("circuit_name")["size_group"]
    out = []
    for group in ("small", "medium", "large", "all"):
        names = groups.index if group == "all" else groups.index[groups == group]
        if len(names) == 0:
            continue
        cum, amounts = cumulative.loc[names], raw.loc[names]
        moved = _direction(amounts["base"], amounts["packed"])
        nonzero = amounts["base"] > 0
        out.append({
            "group": group, "n": len(names),
            "+h_commutation": cum["h_only"].mean(), "+merging": cum["h_merge"].mean(),
            "+packing": cum["packed"].mean(), "median": cum["packed"].median(),
            "worse": int((moved > 0).sum()), "better": int((moved < 0).sum()),
            "same": int((moved == 0).sum()),
            "nonzero": cum.loc[nonzero, "packed"].mean() if nonzero.any() else math.nan,
        })
    return pd.DataFrame(out)


def stage_table(long: pd.DataFrame, program: str, ordering: str, amount: str) -> pd.DataFrame:
    """
    What each stage does to the amount on its own (against the stage before it): on how many
    circuits it raises or lowers it, its segment of the caption-convention mean, and its
    share of the corpus-total change (points of the baseline total).
    """
    scoped = _scoped(long, program, ordering)
    raw = _raw(scoped, amount)
    cumulative = _cumulative(scoped, amount)
    cumulative.insert(0, "base", 0.0)
    rows = []
    for previous, stage, component in zip(STAGES[:-1], STAGES[1:], COMPONENTS):
        moved = _direction(raw[previous], raw[stage])
        rows.append({
            "stage": f"+{component}", "raises": int((moved > 0).sum()),
            "lowers": int((moved < 0).sum()), "same": int((moved == 0).sum()),
            "mean segment": float((cumulative[stage] - cumulative[previous]).mean()),
            "aggregate points": float(100.0 * (raw[previous].sum() - raw[stage].sum())
                                      / raw["base"].sum()),
        })
    return pd.DataFrame(rows)


def aggregate_table(long: pd.DataFrame, program: str, ordering: str,
                    amounts: Sequence[str]) -> pd.DataFrame:
    """% reduction of the corpus total (sum over circuits) after each stage, vs baseline."""
    totals = _scoped(long, program, ordering).groupby("stage")[list(amounts)].sum()
    totals = totals.loc[list(STAGES)]
    return 100.0 * (totals.loc["base"] - totals) / totals.loc["base"]


#: Printed for every view; the first three are the panels.
SUMMARY_AMOUNTS = (
    "atoms_carried", "idle_qubit_seconds", "makespan_seconds", "flight_atom_seconds",
    "transport_layer_seconds", "idle_transport_seconds", "compute_seconds", "E_movement",
    "E_idle", "E_total", "depth", "idle_qubit_seconds_h20us", "idle_qubit_seconds_h50us",
    "makespan_seconds_h50us",
)


def summarize(long: pd.DataFrame, settings: Settings, checks: Optional[Dict] = None) -> str:
    """Console summary: caption-convention numbers for every view, then stage and corpus tables."""
    lines: List[str] = []
    say = lines.append

    def block(frame: pd.DataFrame) -> None:
        say("    " + frame.round(1).to_string(index=False).replace("\n", "\n    "))

    if checks:
        say(f"[{NAME}] checks: {json.dumps(checks, default=str)}")
    for program in PROGRAMS:
        for ordering in ("fair", "shipped"):
            label = "proxies" if program == "proxy" else "originals"
            say(f"\n[{NAME}] {settings.source}, {label}, {ordering} ordering -- cumulative % "
                f"reduction vs baseline (caption convention: unclipped per-circuit mean; "
                f"negative = more than the baseline)")
            for amount in SUMMARY_AMOUNTS:
                say(f"  {amount}:")
                block(caption_table(long, program, ordering, amount))
            say(f"  per stage, against the stage before it ({label}, {ordering}):")
            for amount in PANELS:
                say(f"  {amount}:")
                block(stage_table(long, program, ordering, amount))
            say(f"  corpus totals, % reduction vs baseline ({label}, {ordering}):")
            change = aggregate_table(long, program, ordering, SUMMARY_AMOUNTS)
            say("    " + change.round(1).to_string().replace("\n", "\n    "))
    text = "\n".join(lines)
    print(text, flush=True)
    return text


# ----------------------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------------------


#: Fig 6's stage colours (`plot_results`), bottom of the stack upwards.
STAGE_COLORS = ("#3b4c56", "#cdc2c0", "#0097fd")
WORSE_COLOR = "#b2182b"
#: Increases are unbounded on a reduction axis; below this a bar is cut and labelled.
FLOOR = -200.0


def _slots(frame: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """Fig 6's x layout (`plot_results`): circuit -> bar centre, and the average bar's."""
    positions: Dict[str, float] = {}
    current = 0.0
    for group in ("small", "medium", "large"):
        names = sorted(frame.loc[frame["size_group"] == group, "circuit_name"].unique())
        if not names:
            continue
        xs = np.arange(len(names)) * 1.1 + current
        positions.update(zip(names, xs.tolist()))
        current = float(xs[-1]) + 4.0
    return positions, current + 1.0


def _from_zero(long: pd.DataFrame, settings: Settings, amount: str) -> List[str]:
    """Circuits whose baseline has none of `amount` and whose full pipeline has some."""
    scoped = long[(long["program"] == settings.program)
                  & (long["ordering"] == settings.ordering)]
    wide = scoped.pivot_table(index="circuit_name", columns="stage", values=amount,
                              aggfunc="first")
    return sorted(wide.index[(wide["base"] <= 0) & (wide["packed"] > 0)])


def draw(long: pd.DataFrame, settings: Settings):
    """
    Fig 6's own `plot_results` over this figure's panels, then what an amount that can grow
    needs and a gate count never did:

    * a circuit whose amount grew over all three stages gets a hatched bar below zero
      (Fig 6's `stacked_percent` clips a negative total to nothing), cut at `FLOOR` and
      labelled there; a circuit whose baseline moves no atoms at all and whose pipeline
      does (a reduction that is undefined, counted 0 by Fig 6's `pct_reduction`) gets a
      marker instead;
    * the average bar is redrawn as a three-step waterfall of the per-stage mean segments
      -- the caption convention, unclipped -- so a stage that adds movement is drawn going
      down instead of disappearing, and the total is printed.

    Circuits whose total is a reduction keep Fig 6's stack exactly (its proportional split
    when an intermediate stage dipped).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    from experiments.figures.ablation_fair_depth.ablation_fair_depth import SOURCES
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )

    frame = segments(long, settings.program, settings.ordering)
    panels = dict(PANELS)
    saved = dict(fig6.METRICS)
    fig6.METRICS.clear()
    fig6.METRICS.update(panels)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            fig6_settings = fig6.BenchmarkSettings(output_path=Path(scratch) / "draft.svg",
                                                   show_plot=True)
            fig6.plot_results(frame, fig6_settings)  # Agg: show() is a no-op, figure stays
    finally:
        fig6.METRICS.clear()
        fig6.METRICS.update(saved)
    fig = plt.gcf()
    axes = fig.axes
    # Fig 6's panels are 4 in tall (20 in over five); keep that height for three.
    fig.set_size_inches(22, 4.0 * len(panels) + 2.5)

    positions, average_x = _slots(frame)
    drawn = np.asarray(axes[-1].get_xticks(minor=True), dtype=float)
    expected = np.asarray(sorted(positions.values()) + [average_x], dtype=float)
    if drawn.shape != expected.shape or not np.allclose(drawn, expected):
        raise RuntimeError("Fig 6's x layout changed; `_slots` no longer matches it")

    width, slot = 0.825, 5 * 0.825
    marked = False
    for ax, (amount, (column, _label)) in zip(axes, panels.items()):
        wide = frame.pivot_table(index="circuit_name", columns="component",
                                 values=column)[list(COMPONENTS)]
        totals = wide.sum(axis=1)
        means = wide.mean()
        cumulative = means.cumsum()
        lowest = min(0.0, float(totals.min()), float(cumulative.min()))
        floor = max(FLOOR, lowest)
        highest = max(5.0, float(totals.max()), float(cumulative.max()))
        span = highest - floor

        for name, total in totals.items():
            if total >= -1e-9:
                continue
            shown = max(total, floor)
            ax.bar(positions[name], shown, width=width, bottom=0.0, color="white",
                   edgecolor=WORSE_COLOR, hatch="///", linewidth=0.8)
            if total < floor - 1e-9:
                ax.text(positions[name], floor + 0.015 * span, f"{total:.0f}",
                        rotation=90, ha="center", va="bottom", fontsize=13,
                        color="black", bbox=dict(facecolor="white", edgecolor="none",
                                                 pad=0.6, alpha=0.9))
        for name in _from_zero(long, settings, amount):
            ax.plot([positions[name]], [-0.04 * span], marker="v", markersize=11,
                    color=WORSE_COLOR, linestyle="none", clip_on=False)
            marked = True

        # Waterfall average: Fig 6's average bar is the mean of *clipped* stacks.
        for patch in list(ax.patches):
            centre = patch.get_x() + patch.get_width() / 2.0
            if abs(centre - average_x) < 1e-6 and patch.get_width() > 1.0:
                patch.remove()
        start = 0.0
        sub = slot / 3.0
        for index, (component, color) in enumerate(zip(COMPONENTS, STAGE_COLORS)):
            end = float(cumulative[component])
            x = average_x - slot / 2.0 + sub * (index + 0.5)
            ax.bar(x, end - start, width=sub * 0.92, bottom=start, color=color,
                   edgecolor=WORSE_COLOR if end < start else "white",
                   hatch="///" if end < start else None, linewidth=0.8, alpha=0.95)
            if index < len(COMPONENTS) - 1:
                ax.plot([x - sub * 0.46, x + sub * 1.46], [end, end], color="#636363",
                        linewidth=0.8, linestyle=":")
            start = end
        total = float(cumulative.iloc[-1])
        ax.plot([average_x - slot / 2.0, average_x + slot / 2.0], [total, total],
                color="black", linewidth=1.6)
        ax.text(average_x, total + (0.03 if total >= 0 else -0.03) * span, f"{total:+.1f}%",
                ha="center", va="bottom" if total >= 0 else "top", fontsize=16,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))

        ax.set_ylim(bottom=floor - 0.09 * span, top=highest + 0.12 * span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))

    packing_label = SOURCES[settings.source][2] or "+ Packing benefit"
    handles = [Patch(facecolor=color, edgecolor="white", alpha=0.85, label=label)
               for color, label in zip(STAGE_COLORS, ("H commutation benefit",
                                                      "+ Merging benefit", packing_label))]
    handles.append(Patch(facecolor="white", edgecolor=WORSE_COLOR, hatch="///",
                         label="net increase over baseline (all stages)"))
    if marked:
        handles.append(Line2D([], [], marker="v", markersize=11, color=WORSE_COLOR,
                              linestyle="none", label="baseline moves no atoms (counted 0)"))
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
                   ncol=len(handles), frameon=False, borderaxespad=0.0)
    ordering = {"fair": "every stage layer-sorted",
                "shipped": "shipped emission order"}[settings.ordering]
    program = {"original": "original circuits",
               "proxy": "Clifford proxies"}[settings.program]
    axes[-1].set_xlabel(f"Benchmark circuit (grouped by size); {program}, {ordering}",
                        fontsize=fig6.AXIS_PT)
    fig.supylabel("Percent reduction vs. baseline", fontsize=fig6.AXIS_PT)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    return fig


def plot(long: pd.DataFrame, settings: Settings) -> List[Path]:
    """`draw`, then SVG + PDF through `save_figure` (versioned: never replaces a file)."""
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure

    fig = draw(long, settings)
    paths = save_figure(fig, settings.figure_stem, Path(__file__),
                        override=Path(settings.output_dir) if settings.output_dir else None,
                        versioned=True)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------


def _free(path: Path) -> Path:
    """`path`, or `<stem>_vN<suffix>` -- the first that does not exist. Never overwrites."""
    if not path.exists():
        return path
    version = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def run(settings: Settings) -> None:
    from experiments.revision.common import write_sidecar

    checks = None
    if settings.from_csv:
        long = pd.read_csv(settings.from_csv)
        cached = sorted(long["source"].unique()) if "source" in long else []
        if len(cached) == 1:
            if settings.source not in (None, cached[0]):
                raise SystemExit(f"[{NAME}] --source {settings.source} but "
                                 f"{settings.from_csv} holds {cached[0]}")
            settings.source = cached[0]
        elif settings.source is None:
            raise SystemExit(f"[{NAME}] {settings.from_csv} names no source; pass --source")
        sidecar = Path(str(settings.from_csv) + ".json")
        if sidecar.exists():
            checks = json.loads(sidecar.read_text()).get("extra", {}).get("checks")
    else:
        settings.source = settings.source or "v4"
        began = time.time()
        long, checks = compute(settings)
        # The cache sits beside the script unless --output-dir redirects the whole run.
        data = Path(settings.output_dir) if settings.output_dir else HERE
        data.mkdir(parents=True, exist_ok=True)
        csv = _free(data / f"{settings.stem}.csv")
        long.to_csv(csv, index=False)
        write_sidecar(csv, experiment=NAME, settings=asdict(settings), started=began,
                      status="complete", extra={"checks": checks})
        print(f"[{NAME}] wrote {csv}", flush=True)
        # Fig 6's per-(circuit, component) frame for every view, for replotting elsewhere.
        seg = _free(data / f"{settings.stem}_segments.csv")
        pd.concat([segments(long, program, ordering) for program in PROGRAMS
                   for ordering in ORDERINGS]).to_csv(seg, index=False)
        print(f"[{NAME}] wrote {seg}", flush=True)
    summarize(long, settings, checks)
    for path in plot(long, settings):
        print(f"[{NAME}] saved {path}", flush=True)


def parse_args() -> Settings:
    from experiments.figures.ablation_fair_depth.ablation_fair_depth import SOURCES

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--source", choices=sorted(SOURCES), default=None,
                        help="Plans to schedule (default v4; with --from-csv, the CSV's).")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Worker processes (default 4: the machine is shared).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--from-csv", default=None,
                        help="Redraw and re-summarize from a cached CSV without recomputing.")
    parser.add_argument("--names", default=None,
                        help="Comma-separated circuit names (smoke run; writes *_smoke).")
    parser.add_argument("--program", choices=PROGRAMS, default="original",
                        help="Which schedules the figure draws (default: originals).")
    parser.add_argument("--ordering", choices=sorted(ORDERINGS), default="fair",
                        help="fair = every stage layer-sorted (default); shipped = Fig 6's.")
    args = parser.parse_args()
    return Settings(source=args.source, jobs=args.jobs, output_dir=args.output_dir,
                    from_csv=args.from_csv,
                    names=tuple(args.names.split(",")) if args.names else None,
                    program=args.program, ordering=args.ordering)


if __name__ == "__main__":
    run(parse_args())
