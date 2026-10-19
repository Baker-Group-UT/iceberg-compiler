"""
compile_scaling.py -- how AQORN's compile cost grows with program width n and code size k.

Question: does the method scale as claimed? Experiment7c timed Fig 6's call (k = 2, the v4 planner) on
real circuits up to 150 qubits. This sweeps width n = 8, 16, 32, ... (doubling) at every code
size k = n_code - 2 in {2, 4, 6, 8, 10}, for four planners:

    w10    the planner the paper now uses: weight-10 silent-aware planning,
           `exp8_undetected.cost_model.plan_silent_aware(..., silent_weight=10,
           noise=fig6_noise_profile())`, with the exact gadget silent rates at that n_code
           (n_code 4-8 from Experiment8's table, 10 and 12 computed for this experiment)
    w10i   w10 with the descent's movement and depth terms evaluated incrementally
           (`incremental.py`, installed in the child only). Bit-identical plans by
           construction -- every summand is an integer-valued double -- and checked: every
           w10i cell is compared with the w10 cell for placement, evaluations and model cost
    v4     the submitted planner, `packing.plan.plan_packing(..., h_commutation="auto",
           verify=True, noise=fig6_noise_profile())` -- a reference arm
    w10T   w10 under `explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT`, planned with the
           transversal rate table. k = 2 only: the switch is gated on n_code == 4 in
           `iceberg/costs.py` and `iceberg/iceberg_module.py`, so at k > 2 it is w10.

Laptop timing is contaminated by whatever else runs, so every run records three things:

    time      CPU (`process_time`, the primary number) and wall (`perf_counter`) seconds of the
              planner call and of translating its plan; `os.getloadavg()` before and after
              every repeat; a fixed pure-Python speed probe beside every repeat; how many other
              measurement children were running (`concurrent_cells_*`)
    work      machine-independent counts: descent evaluations (candidate moves priced, summed
              over every descent and both commutation arms), sweeps, descents, accepted moves,
              cost-term computations (`PackingCost` unary/binary cache misses) and lookups,
              exact translator calls, Stim schedule pricings
    quality   1q / 2q / depth of the planned circuit against the naive packing with both passes
              off, every arm layer-sorted (`TranslatorConfig.asap_order`, the fair baseline);
              plus the naive packing with both passes on, so the packing segment is visible

plus an exclusive-time phase breakdown (profile, cost tables, seed, descent, lanes, exact
translation, schedule pricing, silent-error pricing, verify glue, other) and peak RSS.

Protocol. One fresh subprocess per (planner, family, k, n) cell, timing cells strictly one at a
time: switches set first, a warm-up plan on a small program, module caches cleared before every
repeat (Experiment7c's list). Three repeats and the median when a run takes under
`long_run_seconds` (60 s), otherwise one. A planner run is stopped at `cap_seconds` (30 min)
and its series ends there; a size whose planner time, extrapolated from the series' last two
sizes, exceeds `predict_factor` x the cap is not started (recorded as skipped, with the
prediction), and neither is one that would overrun the invocation's budget. Children are
single-threaded (OMP/BLAS threads 1), run with PYTHONHASHSEED=0, and are watched for memory
(`mem_cap_gb`). Instrumentation wraps library functions inside the child only; no library
default changes.

`--workers N` (N > 1) is a separate, non-timing mode: whole series run concurrently, one repeat,
optionally under `taskpolicy -b` (efficiency cores), into their own files; every such row has
`timing_valid = False`. Counts and quality do not depend on load; their seconds are not timings.

Workloads. Generated, gates per qubit fixed so growth in n is clean:

    random   Experiment7c's generator: planted 2-blocks on shuffled labels, locality 0.85,
             10 logical gates per qubit, 40% CX/CZ, Clifford only; seed 20261001 + 1000 n
    ghz      H + CX chain                                          1 CX per qubit
    bv       Bernstein-Vazirani, all-ones secret, one ancilla      star, 1 CX per qubit
    qft      banded QFT (bandwidth 4), each CP as rz/cx/rz/cx/rz   8 CX per qubit, rotations
    ising    1D transverse-field Ising, 3 Trotter steps, rz/rx     6 CX per qubit, rotations

and QASMBench families at their corpus widths (`QB_FAMILIES`: ghz, bv, ising, knn, swap_test,
adder, wstate; up to 255-433 qubits), loaded as Fig 6 loads its originals. Rotation-carrying
programs are priced by verify in the gate-count currency, because Stim cannot scan rz/rx --
exactly what happens to Fig 6's originals.

Outputs (the JSONL files are append-only; CSVs are regenerated from them):

    results/scaling/compile_scaling_runs.jsonl          timing cells, one raw record each
    results/scaling/compile_scaling.csv                 the same, flattened, as it runs
    results/scaling/compile_scaling_pool_runs.jsonl     non-timing family cells (--workers)
    results/scaling/compile_scaling_retime_runs.jsonl   representative cells re-timed at the end
    results/scaling/compile_scaling_final.csv           --finalize: timing cells + concurrency
                                                        flags + re-times; what the figures read
    results/scaling/compile_scaling_final_fits.csv      log-log slopes over n >= 32
    results/figures/compile_scaling_*.pdf|svg           versioned, via save_figure

    PYTHONPATH=. python experiments/scaling/compile_scaling.py --planners w10 --families random
    PYTHONPATH=. python experiments/scaling/compile_scaling.py --finalize
    PYTHONPATH=. python experiments/scaling/compile_scaling.py \
        --from-csv results/scaling/compile_scaling_final.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import functools
import gc
import json
import logging
import math
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
#: Overridable so a frozen copy of this script (run from elsewhere, so later edits cannot
#: change the code a queued run's children execute) still finds the repository.
REPO = Path(os.environ.get("AQORN_REPO", HERE.parents[1])).resolve()
NAME = "compile_scaling"
OUT = REPO / "results" / "scaling"
RUNS = OUT / "compile_scaling_runs.jsonl"
CSV = OUT / "compile_scaling.csv"
FITS = OUT / "compile_scaling_fits.csv"

#: Silent-rate tables. Experiment8's covers n_code 4-8; 10 and 12 were computed for this
#: experiment with `exp8_undetected/gadgets.py --n-codes 10 12 --out results/scaling/rates`.
RATES_EXP8 = REPO / "results" / "revision" / "exp8" / "gadget_rates.csv"
RATES_WIDE = OUT / "rates" / "gadget_rates.csv"
RATES_TRANSVERSAL = REPO / "results" / "tvd_explore" / "gadget_rates_transversal.csv"

PLANNERS: Dict[str, Dict[str, Any]] = {
    "w10": {"weight": 10.0, "switches": (), "ks": None},
    "v4": {"weight": None, "switches": (), "ks": None},
    "w10T": {"weight": 10.0, "switches": ("TRANSVERSAL_INTRA_TWO_QUBIT",), "ks": (2,)},
    #: w10 with the descent's movement and depth terms evaluated incrementally
    #: (`experiments/scaling/incremental.py`, opt-in, this experiment only). Exact by
    #: construction; every cell is checked against w10's placement and counts.
    "w10i": {"weight": 10.0, "switches": (), "ks": None, "incremental": True},
}
GENERATED = ("random", "ghz", "bv", "qft", "ising")
QASMBENCH = REPO / "benchmarks" / "qasmbench"
#: Real circuits: QASMBench families with at least five widths and gate counts linear in
#: width (README "Gate Count" between 1 and ~32 per qubit), loaded exactly as Fig 6 loads its
#: originals (level-1 transpile to h/x/z/cx/cz/s/rz/rx, rotations kept, then measure_all),
#: but with no gate limit and no load timeout, so the widths Fig 6 left out are included.
QB_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "qb_ghz": ("ghz_state_n23", "ghz_n40", "ghz_n78", "ghz_n127", "ghz_n255"),
    "qb_bv": ("bv_n14", "bv_n19", "bv_n30", "bv_n70", "bv_n140", "bv_n280"),
    "qb_ising": ("ising_n26", "ising_n34", "ising_n42", "ising_n66", "ising_n98",
                 "ising_n420"),
    "qb_knn": ("knn_n25", "knn_n31", "knn_n41", "knn_n67", "knn_n129", "knn_n341"),
    "qb_swap_test": ("swap_test_n25", "swap_test_n41", "swap_test_n83", "swap_test_n115",
                     "swap_test_n361"),
    "qb_adder": ("adder_n10", "adder_n28", "adder_n64", "adder_n118", "adder_n433"),
    "qb_wstate": ("wstate_n27", "wstate_n36", "wstate_n76", "wstate_n118", "wstate_n380"),
}
FAMILIES = GENERATED + tuple(QB_FAMILIES)


def qb_width(name: str) -> int:
    return int(name.rsplit("_n", 1)[1])


def qb_sizes(family: str) -> Tuple[int, ...]:
    return tuple(qb_width(name) for name in QB_FAMILIES[family])


def qb_program(family: str, n: int):
    from experiments.shared.benchmark_circuits import create_proxy_from_qasm

    name = next(name for name in QB_FAMILIES[family] if qb_width(name) == n)
    for group in ("small", "medium", "large"):
        folder = QASMBENCH / group / name
        if folder.is_dir():
            files = sorted(f for f in folder.glob("*.qasm") if "_transpiled" not in f.name)
            if not files:
                break
            circuit = create_proxy_from_qasm(files[0], timeout_seconds=None,
                                             replace_rz_rx_with_s=False)
            if circuit is None:
                raise RuntimeError(f"{files[0]} does not load into Fig 6's basis")
            if circuit.num_qubits != n:
                raise RuntimeError(f"{name}: {circuit.num_qubits} qubits, expected {n}")
            return circuit, name
    raise FileNotFoundError(f"no QASM file for {name} under {QASMBENCH}")
SEED_BASE = 20261001
GATES_PER_QUBIT = 10
QFT_BANDWIDTH = 4
ISING_STEPS = 3

#: Exclusive-time phases, in the order the report lists them.
PHASES = ("profile", "cost_tables", "seed", "descent", "lanes", "translate",
          "schedule_pricing", "silent_pricing", "verify_glue")


@dataclass
class Settings:
    planners: Sequence[str] = ("w10",)
    families: Sequence[str] = ("random",)
    ks: Sequence[int] = (2, 4, 6, 8, 10)
    sizes: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    order: str = "size"
    cap_seconds: float = 1800.0
    predict_factor: float = 1.5
    long_run_seconds: float = 60.0
    repeats: int = 3
    budget_hours: float = 3.0
    mem_cap_gb: float = 12.0
    max_n: Optional[int] = None
    #: >1 runs whole series concurrently. Such rows are flagged `timing_valid = False`: their
    #: counts and quality are exact, their seconds are not timing measurements.
    workers: int = 1
    #: Launch children under `taskpolicy -b` (background QoS: efficiency cores only), so a
    #: concurrent non-timing pool cannot take a performance core from the timing queue.
    background: bool = False
    tag: str = ""
    runs: str = str(RUNS)
    csv: str = str(CSV)


# ----------------------------------------------------------------------------------------
# workloads
# ----------------------------------------------------------------------------------------


def family_seed(family: str, n: int) -> Optional[int]:
    """Experiment7c's seed formula (index 0) for the random family; the others are fixed."""
    return SEED_BASE + 1000 * n if family == "random" else None


def build_program(family: str, n: int):
    """The logical program, with measurements, as the planner and translator receive it."""
    from qiskit import QuantumCircuit

    from experiments.shared.simulation import ensure_measurements

    if family in QB_FAMILIES:
        circuit, _ = qb_program(family, n)
    elif family == "random":
        from experiments.shared.random_circuits import plant_blocks, structured_program_circuit

        seed = family_seed(family, n)
        circuit = structured_program_circuit(
            n, GATES_PER_QUBIT, seed=seed, blocks=plant_blocks(n, 2, seed=seed),
            locality=0.85, immutable_ratio=0.0, clifford_only=True,
        )
    elif family == "ghz":
        circuit = QuantumCircuit(n)
        circuit.h(0)
        for qubit in range(1, n):
            circuit.cx(qubit - 1, qubit)
    elif family == "bv":
        circuit = QuantumCircuit(n)
        ancilla = n - 1
        circuit.x(ancilla)
        for qubit in range(n):
            circuit.h(qubit)
        for qubit in range(n - 1):
            circuit.cx(qubit, ancilla)
        for qubit in range(n - 1):
            circuit.h(qubit)
    elif family == "qft":
        circuit = QuantumCircuit(n)
        for target in range(n):
            circuit.h(target)
            for distance in range(1, QFT_BANDWIDTH + 1):
                control = target + distance
                if control >= n:
                    break
                theta = math.pi / 2 ** distance
                # CP(theta) = RZ_c(theta/2) CX RZ_t(-theta/2) CX RZ_t(theta/2), up to phase
                circuit.rz(theta / 2, control)
                circuit.cx(control, target)
                circuit.rz(-theta / 2, target)
                circuit.cx(control, target)
                circuit.rz(theta / 2, target)
    elif family == "ising":
        circuit = QuantumCircuit(n)
        for qubit in range(n):
            circuit.h(qubit)
        for _ in range(ISING_STEPS):
            for parity in (0, 1):
                for left in range(parity, n - 1, 2):
                    circuit.cx(left, left + 1)
                    circuit.rz(0.3, left + 1)
                    circuit.cx(left, left + 1)
            for qubit in range(n):
                circuit.rx(0.7, qubit)
    else:
        raise ValueError(f"unknown family {family!r}")
    return ensure_measurements(circuit)


def logical_counts(program) -> Dict[str, int]:
    one = two = 0
    for instruction in program.data:
        name = instruction.operation.name
        if name in ("measure", "barrier"):
            continue
        if len(instruction.qubits) == 1:
            one += 1
        elif len(instruction.qubits) == 2:
            two += 1
    return {"logical_1q": one, "logical_2q": two, "logical_gates": one + two,
            "logical_depth": int(program.depth())}


# ----------------------------------------------------------------------------------------
# planners
# ----------------------------------------------------------------------------------------


def load_rates(planner: str, n_code: int):
    import pandas as pd

    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from simulator.noise_models import StimNoiseParameters

    if planner == "w10T":
        path = RATES_TRANSVERSAL
    else:
        path = RATES_EXP8 if n_code in (4, 6, 8) else RATES_WIDE
    table = pd.read_csv(path)
    if n_code not in set(table["n_code"]):
        raise KeyError(f"no silent rates for n_code={n_code} in {path}")
    p2 = StimNoiseParameters.neutral_atom().two_qubit_error
    return UndetectedRates.from_table(table, n_code, p2), str(path)


def make_planner(planner: str, n_code: int) -> Tuple[Callable, Dict[str, Any]]:
    from experiments.revision.common import fig6_noise_profile

    noise = fig6_noise_profile()
    if planner == "v4":
        from packing.plan import plan_packing

        def run(program):
            return plan_packing(program, n_code, h_commutation="auto", verify=True, noise=noise)

        return run, {"rates_table": ""}

    from experiments.revision.exp8_undetected.cost_model import plan_silent_aware

    rates, path = load_rates(planner, n_code)
    weight = PLANNERS[planner]["weight"]

    def run(program):
        return plan_silent_aware(program, n_code, rates=rates, silent_weight=weight, noise=noise)

    return run, {"rates_table": path}


def clear_caches() -> None:
    """Experiment7c's list: every module cache that could make a repeat cheaper."""
    import packing.passes as passes
    import packing.resource.noise as rnoise
    import packing.tables as tables

    passes.clear_cache()
    tables._MEAN_HOPS.clear()
    tables._HOP_MATRIX.clear()
    rnoise._INTRA.clear()


def translate_config(n_code: int, *, commute: bool, merging: bool):
    """`plan.measured_cost`'s translate config, with the fair-baseline layer sort on."""
    from pipeline.config import PipelineConfig

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    config.translator_config.fault_tolerant = False
    config.translator_config.asap_order = True
    return config


# ----------------------------------------------------------------------------------------
# instrumentation (child process only)
# ----------------------------------------------------------------------------------------


class CapExceeded(BaseException):
    """Raised by SIGALRM. BaseException, so the library's `except Exception` cannot swallow it."""


class Probe:
    """
    Times and counts the planner's phases by wrapping library functions at every call site.

    Exclusive time: a wrapped call nested in another is charged to its own phase and taken
    off its parent's, so the phases add up to the wrapped total and `other` is the rest. A
    few dozen wrapped calls per plan, two `perf_counter` reads each: nothing measurable.
    The descent's inner loop (`Packing.value_after`) is not wrapped; its count comes off
    `DescentResult.evaluations`, which the library already keeps.
    """

    def __init__(self) -> None:
        self._undo: List[Tuple[Any, str, Any]] = []
        self.stack: List[List[Any]] = []
        self.reset()

    def reset(self) -> None:
        self.exclusive: Dict[str, float] = defaultdict(float)
        self.calls: Counter = Counter()
        self.failed: Counter = Counter()
        self.counts: Counter = Counter()
        self.costs: List[Any] = []

    def _wrap(self, label: str, original: Callable, after: Optional[Callable] = None,
              before: Optional[Callable] = None) -> Callable:
        probe = self

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            state = before(args, kwargs) if before else None
            frame = [label, time.perf_counter(), 0.0]
            probe.stack.append(frame)
            try:
                result = original(*args, **kwargs)
            except Exception:
                probe.failed[label] += 1
                raise
            finally:
                elapsed = time.perf_counter() - frame[1]
                probe.stack.pop()
                probe.exclusive[label] += elapsed - frame[2]
                probe.calls[label] += 1
                if probe.stack:
                    probe.stack[-1][2] += elapsed
            if after:
                after(args, kwargs, result, state)
            return result

        return wrapper

    def patch(self, owner: Any, name: str, label: str, **hooks) -> None:
        original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
        setattr(owner, name, self._wrap(label, original, **hooks))
        self._undo.append((owner, name, original))

    def install(self) -> None:
        import packing.cost as pcost
        import packing.lanes as planes
        import packing.plan as pplan
        import packing.profiles as pprofiles
        import packing.search as psearch
        import packing.tables as ptables
        import simulator.attribution as attribution
        from experiments.revision.exp8_undetected import cost_model
        from translator.core import Translator

        def descend_before(args, kwargs):
            given = kwargs.get("result")
            return (given.evaluations, given.rounds, given.moves) if given else (0, 0, 0)

        def descend_after(args, kwargs, result, state):
            # `self.counts`, not a captured reference: `reset` replaces the Counter.
            self.counts["descents"] += 1
            self.counts["evaluations"] += result.evaluations - state[0]
            self.counts["sweeps"] += result.rounds - state[1]
            self.counts["moves"] += result.moves - state[2]

        def register_cost(args, kwargs, result, state):
            self.costs.append(args[0])

        for owner in (psearch, pplan):
            self.patch(owner, "descend", "descent", before=descend_before, after=descend_after)
            self.patch(owner, "naive_seed", "seed")
            self.patch(owner, "shuffled_seed", "seed")
        self.patch(pplan, "graph_seed", "seed")
        self.patch(cost_model, "_seed", "seed")
        for owner in (pplan, ptables):
            self.patch(owner, "translated_view", "profile")
        for owner in (pplan, pprofiles):
            self.patch(owner, "extract_profiles", "profile")
        self.patch(pcost.PackingCost, "__init__", "cost_tables", after=register_cost)
        self.patch(cost_model.SilentAwareCost, "__init__", "cost_tables")
        self.patch(cost_model, "build_costs", "cost_tables")
        for owner in (pplan, planes):
            self.patch(owner, "assign_lanes", "lanes")
        self.patch(Translator, "translate", "translate")
        self.patch(attribution, "measured_noise_split", "schedule_pricing")
        self.patch(cost_model, "silent_errors", "silent_pricing")
        self.patch(pplan, "measured_cost", "verify_glue")
        self.patch(cost_model, "measured_with_currency", "verify_glue")

    def uninstall(self) -> None:
        while self._undo:
            owner, name, original = self._undo.pop()
            setattr(owner, name, original)

    def snapshot(self, total: float) -> Dict[str, Any]:
        phases = {f"phase_{label}_s": round(self.exclusive.get(label, 0.0), 6)
                  for label in PHASES}
        phases["phase_other_s"] = round(total - sum(self.exclusive.values()), 6)
        hits = sum(getattr(c, "hits", 0) for c in self.costs)
        misses = sum(getattr(c, "misses", 0) for c in self.costs)
        return {
            **phases,
            "evaluations": int(self.counts["evaluations"]),
            "descents": int(self.counts["descents"]),
            "sweeps": int(self.counts["sweeps"]),
            "moves": int(self.counts["moves"]),
            "cost_term_computations": int(misses),
            "cost_term_lookups": int(hits),
            "cost_objects": len(self.costs),
            "translator_calls": int(self.calls["translate"]),
            "schedule_pricings": int(self.calls["schedule_pricing"]),
            "schedule_pricings_failed": int(self.failed["schedule_pricing"]),
            "silent_pricings": int(self.calls["silent_pricing"]),
        }


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


#: Speed-probe CPU seconds, one per probe, read by `run_cell` (child process only).
PROBES: List[Tuple[float, float]] = []


def speed_probe() -> Tuple[float, float]:
    """
    (CPU, wall) seconds of a fixed pure-Python workload shaped like the descent's inner loop
    -- frozenset-keyed dict lookups and float sums -- run beside every timed repeat. On a
    quiet performance core it takes a fixed time; contention, a slower core or thermal
    limits stretch it, so its ratio to the quietest probe on record says how much a
    measurement taken next to it was slowed. A diagnostic beside the measurement, not a
    correction applied to it.
    """
    table = {frozenset((i, i + 1, i + 2)): float(i) for i in range(4000)}
    keys = list(table)
    began_cpu, began_wall = time.process_time(), time.perf_counter()
    total = 0.0
    for _ in range(800):
        for key in keys:
            total += table[key] * 1.0000001
    return time.process_time() - began_cpu, time.perf_counter() - began_wall


def _timed_repeats(fn: Callable, *, repeats: int, long_run: float,
                   cap: Optional[float] = None, between: Optional[Callable] = None):
    """Run `fn` up to `repeats` times (once if a run exceeds `long_run`). Raises CapExceeded."""
    walls, cpus, loads, results = [], [], [], []
    for _ in range(repeats):
        if between:
            between()
        gc.collect()
        PROBES.append(speed_probe())
        load_before = os.getloadavg()
        if cap:
            signal.setitimer(signal.ITIMER_REAL, cap)
        began_wall, began_cpu = time.perf_counter(), time.process_time()
        try:
            result = fn()
        finally:
            wall = time.perf_counter() - began_wall
            cpu = time.process_time() - began_cpu
            if cap:
                signal.setitimer(signal.ITIMER_REAL, 0)
        walls.append(wall)
        cpus.append(cpu)
        loads.append([round(x, 2) for x in load_before] + [round(os.getloadavg()[0], 2)])
        results.append(result)
        PROBES.append(speed_probe())
        if wall > long_run:
            break
    return walls, cpus, loads, results


def run_cell(spec: Dict[str, Any]) -> Dict[str, Any]:
    """One (planner, family, k, n) measurement, in a fresh process."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from experiments.figures.merging_improvement import (
        merging_improvement_benchmark_stylized_simulated as fig6,
    )
    from packing.plan import naive_packing
    from pipeline.tasks.translate_permutation import translate_permutation
    from translator import explore_switches

    planner, family, k, n = spec["planner"], spec["family"], spec["k"], spec["n"]
    n_code = k + 2
    switches = PLANNERS[planner]["switches"]
    explore_switches.configure(**{name: True for name in switches})
    if PLANNERS[planner].get("incremental"):
        from experiments.scaling import incremental

        incremental.install()
    row: Dict[str, Any] = {"planner": planner, "family": family, "k": k, "n_code": n_code,
                           "n": n, "seed": family_seed(family, n), "switches": "+".join(switches)}

    def cap_handler(signum, frame):
        raise CapExceeded()

    signal.signal(signal.SIGALRM, cap_handler)

    program = build_program(family, n)
    row.update(logical_counts(program))
    if family in QB_FAMILIES:
        row["benchmark"] = next(name for name in QB_FAMILIES[family] if qb_width(name) == n)
    plan_fn, meta = make_planner(planner, n_code)
    row.update(meta)

    # Warm-up: imports, lazy tables and Stim's first scan, on a small program at this k
    # with at least two patches, so verify and schedule pricing run too.
    warm = build_program("random", max(8, 2 * k + 1))
    warm_plan = plan_fn(warm)
    translate_permutation(warm, warm_plan.placement,
                          config=translate_config(n_code, commute=warm_plan.h_commutation,
                                                  merging=True))

    probe = Probe()
    probe.install()
    snapshots: List[Dict[str, Any]] = []

    def plan_once():
        probe.reset()
        began = time.perf_counter()
        plan = plan_fn(program)
        snapshots.append(probe.snapshot(time.perf_counter() - began))
        return plan

    status = "ok"
    try:
        walls, cpus, loads, plans = _timed_repeats(
            plan_once, repeats=spec["repeats"], long_run=spec["long_run_seconds"],
            cap=spec["cap_seconds"], between=clear_caches)
    except CapExceeded:
        status = "cap"
        partial = probe.snapshot(spec["cap_seconds"])
        probe.uninstall()
        row.update(status=status, plan_wall_s=float(spec["cap_seconds"]),
                   plan_cpu_s=math.nan, peak_rss_mb=_peak_rss_mb(),
                   **{f"partial_{key}": value for key, value in partial.items()
                      if not key.startswith("phase_")})
        return row
    probe.uninstall()

    plan_probes = list(PROBES)
    row.update(probe_cpu_s_min=min(c for c, _ in plan_probes),
               probe_cpu_s_max=max(c for c, _ in plan_probes),
               probe_cpu_s_mean=statistics.mean(c for c, _ in plan_probes),
               probe_wall_over_cpu=statistics.mean(w / c for c, w in plan_probes if c > 0),
               probes=json.dumps([[round(c, 4), round(w, 4)] for c, w in plan_probes]))
    plan = plans[0]
    first = snapshots[0]
    stable = all({key: s[key] for key in s if not key.startswith("phase_")}
                 == {key: first[key] for key in first if not key.startswith("phase_")}
                 for s in snapshots)
    placement = [(int(p), int(s)) for p, s in plan.placement]
    if any([(int(p), int(s)) for p, s in other.placement] != placement for other in plans[1:]):
        stable = False
    median_index = sorted(range(len(walls)), key=lambda i: walls[i])[len(walls) // 2]
    row.update(
        status=status,
        plan_wall_s=statistics.median(walls), plan_cpu_s=statistics.median(cpus),
        plan_wall_min_s=min(walls), plan_wall_max_s=max(walls),
        plan_cpu_min_s=min(cpus), plan_cpu_max_s=max(cpus),
        plan_repeats=len(walls), plan_walls=json.dumps([round(w, 4) for w in walls]),
        plan_cpus=json.dumps([round(c, 4) for c in cpus]),
        plan_loadavg=json.dumps(loads),
        load1_mean=statistics.mean(l[0] for l in loads),
        repeat_counts_identical=bool(stable),
        **first,
        **{f"median_run_{key}": value for key, value in snapshots[median_index].items()
           if key.startswith("phase_")},
        patches=int(plan.patches), physical_qubits=int(plan.physical_qubits),
        tightest_patches=max(1, -(-n // k)),
        h_commutation=bool(plan.h_commutation), fell_back=bool(plan.fell_back),
        verified=bool(plan.verified), currency=getattr(plan, "currency", None) or "",
        model_cost=float(plan.cost),
        considered=json.dumps([(label, float(value)) for label, value in plan.considered]),
        chosen_descent_evaluations=int(plan.descent.evaluations) if plan.descent else 0,
        placement=json.dumps(placement),
        peak_rss_mb_plan=_peak_rss_mb(),
    )

    # Translating the chosen plan: the rest of the compile, timed the same way. Layer-sorted,
    # because that is the circuit whose quality is reported.
    planned_config = translate_config(n_code, commute=plan.h_commutation, merging=True)
    t_walls, t_cpus, t_loads, translated = _timed_repeats(
        lambda: translate_permutation(program, placement, config=planned_config),
        repeats=spec["repeats"], long_run=spec["long_run_seconds"], between=clear_caches)
    circuit = translated[0].circuit
    planned_1q, planned_2q = one_two_qubit_gate_totals(circuit)
    planned_depth = int(circuit.depth())
    row.update(
        translate_wall_s=statistics.median(t_walls), translate_cpu_s=statistics.median(t_cpus),
        translate_repeats=len(t_walls), translate_loadavg=json.dumps(t_loads),
        planned_1q=int(planned_1q), planned_2q=int(planned_2q), planned_depth=planned_depth,
        physical_qubits_translated=int(circuit.num_qubits),
    )

    # Quality references, untimed: naive packing with both passes off (the baseline) and with
    # both passes on under the plan's commutation decision. The switches belong to the merging
    # stages only, as `ablation_fair_depth` sets them.
    naive = naive_packing(n, k)
    began = time.perf_counter()
    h_merge = fig6.compute_translation_metrics(
        program, naive, translate_config(n_code, commute=plan.h_commutation, merging=True))
    explore_switches.configure(**{name: False for name in switches})
    base = fig6.compute_translation_metrics(
        program, naive, translate_config(n_code, commute=False, merging=False))
    explore_switches.configure(**{name: True for name in switches})
    row["reference_translations_s"] = time.perf_counter() - began
    row.update(base_1q=int(base[0]), base_2q=int(base[1]), base_depth=int(base[2]),
               naive_passes_1q=int(h_merge[0]), naive_passes_2q=int(h_merge[1]),
               naive_passes_depth=int(h_merge[2]))
    for metric, planned_value, index in (("1q", planned_1q, 0), ("2q", planned_2q, 1),
                                         ("depth", planned_depth, 2)):
        row[f"red_{metric}_pct"] = fig6.pct_reduction(base[index], planned_value)
        row[f"red_{metric}_passes_only_pct"] = fig6.pct_reduction(base[index], h_merge[index])
    row["peak_rss_mb"] = _peak_rss_mb()
    return row


# ----------------------------------------------------------------------------------------
# parent: schedule, watchdog, records
# ----------------------------------------------------------------------------------------


def cell_key(row: Dict[str, Any]) -> Tuple:
    return (row["planner"], row["family"], int(row["k"]), int(row["n"]))


def read_runs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import pandas as pd

    if not rows:
        return
    frame = pd.DataFrame(rows)
    # Latest record per cell wins (a resumed run never repeats a finished cell, but a cell
    # that errored may be retried).
    frame = frame.drop_duplicates(subset=["planner", "family", "k", "n"], keep="last")
    frame = frame.sort_values(["planner", "family", "k", "n"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def predict_seconds(history: List[Tuple[int, float]], n: int) -> Optional[float]:
    """Planner seconds at `n`, extrapolated on the last two sizes' log-log slope (1 to 5)."""
    if not history:
        return None
    history = sorted(history)
    last_n, last_t = history[-1]
    if len(history) >= 2:
        prev_n, prev_t = history[-2]
        slope = math.log(max(last_t, 1e-6) / max(prev_t, 1e-6)) / math.log(last_n / prev_n)
        slope = min(5.0, max(1.0, slope))
    else:
        slope = 3.0
    return last_t * (n / last_n) ** slope


def _rss_mb(pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return float(out) / 1024 if out else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _other_cells(own_pid: int) -> int:
    """Other measurement children of this script running right now (any invocation)."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:  # noqa: BLE001
        return -1
    count = 0
    for line in out.splitlines():
        pid, _, command = line.strip().partition(" ")
        if "compile_scaling" in command and "--cell" in command and pid != str(own_pid):
            count += 1
    return count


def launch_cell(spec: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0",
               TQDM_DISABLE="1", MPLBACKEND="Agg")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "cell.json"
        err = Path(scratch) / "cell.err"
        command = [sys.executable, str(Path(__file__).resolve()), "--cell", json.dumps(spec),
                   "--cell-out", str(out)]
        if settings.background:
            command = ["taskpolicy", "-b"] + command
        began = time.time()
        # Hard limit: the cap covers one planner run; repeats only happen under 60 s, and
        # translation plus references add minutes at most.
        hard_limit = spec["cap_seconds"] * 1.25 + 900
        with open(err, "w") as stderr:
            child = subprocess.Popen(command, env=env, cwd=str(REPO), stdout=subprocess.DEVNULL,
                                     stderr=stderr)
            peak_rss = 0.0
            killed = ""
            polls = 0
            others: List[int] = [_other_cells(child.pid)]
            loads: List[float] = [os.getloadavg()[0]]
            while child.poll() is None:
                time.sleep(3.0)
                polls += 1
                rss = _rss_mb(child.pid)
                peak_rss = max(peak_rss, rss)
                if polls % 5 == 0:
                    others.append(_other_cells(child.pid))
                    loads.append(os.getloadavg()[0])
                if rss > settings.mem_cap_gb * 1024:
                    killed = "mem_cap"
                elif time.time() - began > hard_limit:
                    killed = "hard_timeout"
                if killed:
                    child.kill()
                    child.wait()
                    break
        elapsed = time.time() - began
        concurrency = {"concurrent_cells_max": max(others),
                       "concurrent_cells_mean": round(statistics.mean(others), 2),
                       "load1_during_mean": round(statistics.mean(loads), 2)}
        base = {**{key: spec[key] for key in ("planner", "family", "k", "n")},
                "n_code": spec["k"] + 2, "seed": family_seed(spec["family"], spec["n"]),
                "cell_seconds": round(elapsed, 2), "watchdog_peak_rss_mb": round(peak_rss, 1),
                **concurrency}
        if killed:
            return {**base, "status": killed, "plan_wall_s": math.nan}
        if not out.exists():
            tail = err.read_text()[-1500:] if err.exists() else ""
            return {**base, "status": "error", "error": tail, "plan_wall_s": math.nan}
        row = json.loads(out.read_text())
        row.update(cell_seconds=round(elapsed, 2), watchdog_peak_rss_mb=round(peak_rss, 1),
                   **concurrency)
        return row


def sizes_for(family: str, settings: Settings) -> Tuple[int, ...]:
    """Doubling sizes for generated families, the corpus widths for QASMBench ones."""
    sizes = qb_sizes(family) if family in QB_FAMILIES else tuple(settings.sizes)
    return tuple(n for n in sizes if settings.max_n is None or n <= settings.max_n)


def schedule(settings: Settings) -> List[Tuple[Tuple[str, str, int], int]]:
    series = []
    for planner in settings.planners:
        allowed = PLANNERS[planner]["ks"]
        for family in settings.families:
            for k in settings.ks:
                if allowed is None or k in allowed:
                    series.append((planner, family, k))
    cells = [(s, n) for s in series for n in sizes_for(s[1], settings)]
    if settings.order == "series":
        return cells
    # Breadth first: every series' small sizes before anyone's large one.
    return sorted(cells, key=lambda cell: cell[1])


def run_parallel(settings: Settings) -> None:
    """
    Non-timing mode: whole series run concurrently (`settings.workers` at a time), each in
    increasing n, stopping at its first non-ok cell. One repeat per cell. Every row carries
    `timing_valid = False`; work counts and quality do not depend on load, seconds do.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from experiments.revision.common import log

    runs_path, csv_path = Path(settings.runs), Path(settings.csv)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + settings.budget_hours * 3600
    done = {cell_key(r): r for r in read_runs(runs_path)}
    lock = threading.Lock()
    series = []
    for planner in settings.planners:
        allowed = PLANNERS[planner]["ks"]
        for family in settings.families:
            for k in settings.ks:
                if allowed is None or k in allowed:
                    series.append((planner, family, k))
    # Longest first: wide families and small k carry the most work.
    series.sort(key=lambda s: (-max(sizes_for(s[1], settings) or (0,)), s[2]))
    log(NAME, f"parallel: {len(series)} series on {settings.workers} workers"
              f"{' (background QoS)' if settings.background else ''}, budget "
              f"{settings.budget_hours:.2f} h, {len(done)} cells already in {runs_path}")

    def record(row: Dict[str, Any]) -> None:
        row.setdefault("tag", settings.tag)
        row["timing_valid"] = False
        row.setdefault("recorded", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with lock:
            with open(runs_path, "a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            done[cell_key(row)] = row
            write_csv(list(done.values()), csv_path)

    def one_series(item):
        planner, family, k = item
        for n in sizes_for(family, settings):
            key = (planner, family, k, n)
            previous = done.get(key)
            if previous is not None:
                if previous.get("status") != "ok":
                    return
                continue
            if time.time() > deadline:
                return
            spec = {"planner": planner, "family": family, "k": k, "n": n,
                    "cap_seconds": settings.cap_seconds, "repeats": 1,
                    "long_run_seconds": settings.long_run_seconds}
            row = launch_cell(spec, settings)
            record(row)
            log(NAME, f"[parallel] {planner:>4} {family:>12} k={k:<2} n={n:<5} "
                      f"{row.get('status'):>8} {row.get('plan_wall_s', math.nan):9.2f} s  evals "
                      f"{row.get('evaluations', row.get('partial_evaluations', 0))}  2q "
                      f"{row.get('red_2q_pct', math.nan):5.1f}%  depth "
                      f"{row.get('red_depth_pct', math.nan):5.1f}%")
            if row.get("status") != "ok":
                return

    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        list(pool.map(one_series, series))
    log(NAME, f"parallel finished; {len(done)} cells in {csv_path}")


def run(settings: Settings) -> None:
    from experiments.revision.common import log

    if settings.workers > 1:
        return run_parallel(settings)

    runs_path, csv_path = Path(settings.runs), Path(settings.csv)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + settings.budget_hours * 3600
    done = {cell_key(r): r for r in read_runs(runs_path)}
    # A budget skip says nothing about the cell, so a later invocation may retry it.
    done = {key: r for key, r in done.items() if r.get("status") != "skipped_budget"}
    stopped: Dict[Tuple[str, str, int], str] = {}
    for key, row in done.items():
        if row.get("status") not in ("ok",):
            stopped[key[:3]] = row.get("status", "?")
    cells = schedule(settings)
    log(NAME, f"{len(cells)} cells scheduled ({settings.order} order), cap "
              f"{settings.cap_seconds:.0f} s, budget {settings.budget_hours:.2f} h, "
              f"{len(done)} cells already recorded in {runs_path}")
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = "?"

    def record(row: Dict[str, Any]) -> None:
        row.setdefault("tag", settings.tag)
        row["timing_valid"] = True
        row.setdefault("cpu", cpu)
        row.setdefault("recorded", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(runs_path, "a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        done[cell_key(row)] = row
        write_csv(list(done.values()), csv_path)

    for series, n in cells:
        planner, family, k = series
        key = (planner, family, k, n)
        if key in done or series in stopped:
            continue
        history = [(int(r["n"]), float(r["plan_wall_s"])) for kk, r in done.items()
                   if kk[:3] == series and r.get("status") == "ok" and int(r["n"]) < n]
        predicted = predict_seconds(history, n)
        base = {"planner": planner, "family": family, "k": k, "n": n, "n_code": k + 2,
                "seed": family_seed(family, n),
                "predicted_plan_s": round(predicted, 1) if predicted else None}
        if predicted is not None and predicted > settings.predict_factor * settings.cap_seconds:
            record({**base, "status": "skipped_predicted", "plan_wall_s": math.nan})
            stopped[series] = "skipped_predicted"
            log(NAME, f"{key}: predicted {predicted:.0f} s > {settings.predict_factor} x cap, "
                      f"series stopped")
            continue
        remaining = deadline - time.time()
        expected = 0.0 if predicted is None else min(predicted, settings.cap_seconds) * (
            3 if predicted < settings.long_run_seconds else 1) * 1.3 + 15
        if expected > remaining:
            record({**base, "status": "skipped_budget", "plan_wall_s": math.nan})
            stopped[series] = "skipped_budget"
            log(NAME, f"{key}: expected {expected:.0f} s > {remaining:.0f} s left, series stopped")
            continue
        spec = {"planner": planner, "family": family, "k": k, "n": n,
                "cap_seconds": settings.cap_seconds, "repeats": settings.repeats,
                "long_run_seconds": settings.long_run_seconds}
        row = launch_cell(spec, settings)
        row["predicted_plan_s"] = base["predicted_plan_s"]
        record(row)
        status = row.get("status")
        if status != "ok":
            stopped[series] = status
        quality = (f"2q {row.get('red_2q_pct', math.nan):5.1f}% depth "
                   f"{row.get('red_depth_pct', math.nan):5.1f}%") if status == "ok" else ""
        log(NAME, f"{planner:>4} {family:>6} k={k:<2} n={n:<5} {status:>8} plan "
                  f"{row.get('plan_wall_s', math.nan):9.2f} s wall "
                  f"{row.get('plan_cpu_s', math.nan):9.2f} s cpu x{row.get('plan_repeats', 0)}  "
                  f"evals {row.get('evaluations', row.get('partial_evaluations', 0))}  "
                  f"P={row.get('patches', '-')}  {quality}  load "
                  f"{row.get('load1_mean', math.nan):.1f}  rss "
                  f"{row.get('watchdog_peak_rss_mb', 0):.0f} MB")
    log(NAME, f"finished; {len(done)} cells in {csv_path}")


# ----------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------

FIT_METRICS = ("compile_wall_s", "compile_cpu_s", "plan_wall_s", "plan_cpu_s", "evaluations",
               "cost_term_computations", "phase_descent_s", "us_per_evaluation",
               "translate_wall_s", "translate_cpu_s", "peak_rss_mb")


def fit_exponents(frame, min_n: int = 32):
    """Least-squares slope of log(metric) on log(n), per series, over sizes >= `min_n`."""
    import numpy as np
    import pandas as pd

    rows = []
    frame = with_derived(frame)
    ok = frame[frame["status"] == "ok"]
    for (planner, family, k), block in ok.groupby(["planner", "family", "k"]):
        block = block[block["n"] >= min_n].sort_values("n")
        for metric in FIT_METRICS:
            if metric not in block:
                continue
            sub = block[block[metric].notna() & (block[metric] > 0)]
            if len(sub) < 2:
                continue
            x = np.log(sub["n"].to_numpy(float))
            y = np.log(sub[metric].to_numpy(float))
            slope, intercept = np.polyfit(x, y, 1)
            fitted = slope * x + intercept
            residual = y - fitted
            r2 = 1 - (residual ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
            stderr = (math.sqrt((residual ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
                      if len(x) > 2 else math.nan)
            rows.append({"planner": planner, "family": family, "k": int(k), "metric": metric,
                         "exponent": slope, "stderr": stderr, "r2": r2, "points": len(sub),
                         "n_min": int(sub["n"].min()), "n_max": int(sub["n"].max())})
    return pd.DataFrame(rows)


def markdown(frame, index: bool = False) -> str:
    """A pipe table without `tabulate` (not installed in this venv)."""
    frame = frame.reset_index() if index else frame
    header = [str(c) if not isinstance(c, tuple) else "/".join(map(str, c)) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if math.isnan(value) else f"{value:g}" if abs(value) >= 1e4
                             else f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def finalize(runs: Path, pool_runs: Optional[Path] = None,
             retime_runs: Optional[Path] = None):
    """
    The timing table, flagged: one row per timing cell (latest record per cell) with
    `pool_cells_overlapping` -- how many non-timing pool cells ran during it -- and, where
    the cell was re-timed later, the re-timed CPU and wall seconds beside the original.
    """
    import pandas as pd

    def frame_of(path: Optional[Path]):
        rows = read_runs(path) if path and Path(path).exists() else []
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).drop_duplicates(
            subset=["planner", "family", "k", "n"], keep="last")
        frame["recorded_t"] = pd.to_datetime(frame["recorded"])
        frame["started_t"] = frame["recorded_t"] - pd.to_timedelta(
            frame["cell_seconds"].fillna(0), unit="s")
        return frame

    main = frame_of(runs)
    pool = frame_of(pool_runs)
    if main.empty:
        return main
    main["timing_valid"] = main.get("timing_valid", pd.Series(True, index=main.index)).fillna(True)
    if not pool.empty:
        def overlapping(row) -> int:
            return int(((pool["started_t"] < row["recorded_t"])
                        & (pool["recorded_t"] > row["started_t"])).sum())

        main["pool_cells_overlapping"] = main.apply(overlapping, axis=1)
    else:
        main["pool_cells_overlapping"] = 0
    retime = frame_of(retime_runs)
    if not retime.empty:
        keep = retime[retime["status"] == "ok"][
            ["planner", "family", "k", "n", "plan_cpu_s", "plan_wall_s", "load1_mean",
             "probe_cpu_s_mean"]].rename(columns={
                 "plan_cpu_s": "retime_plan_cpu_s", "plan_wall_s": "retime_plan_wall_s",
                 "load1_mean": "retime_load1_mean", "probe_cpu_s_mean": "retime_probe_cpu_s"})
        main = main.merge(keep, on=["planner", "family", "k", "n"], how="left")
    return main.drop(columns=["recorded_t", "started_t"]).sort_values(
        ["planner", "family", "k", "n"])


def _fmt_s(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.3g}" if value < 100 else f"{value:.0f}"


def summarize(frame, pool=None) -> str:
    """Markdown tables for the report."""
    import pandas as pd

    lines: List[str] = []
    frame = with_derived(frame)
    ok = frame[frame["status"] == "ok"].copy()
    fits = fit_exponents(frame)
    if not fits.empty:
        lines.append("### Exponents (log-log least squares over n >= 32)\n")
        wanted = ["compile_cpu_s", "compile_wall_s", "evaluations", "us_per_evaluation",
                  "peak_rss_mb"]
        sub = fits[fits["metric"].isin(wanted)]
        table = sub.pivot_table(index=["planner", "family", "k"], columns="metric",
                                values="exponent")[[m for m in wanted if m in set(sub["metric"])]]
        span = sub[sub["metric"] == "compile_cpu_s"].set_index(["planner", "family", "k"])
        table["n range"] = [f"{int(span.loc[i, 'n_min'])}-{int(span.loc[i, 'n_max'])} "
                            f"({int(span.loc[i, 'points'])} pts)" if i in span.index else ""
                            for i in table.index]
        lines.append(markdown(table.round(2), index=True))
        lines.append("")

    local = []
    for (planner, family, k), block in ok.groupby(["planner", "family", "k"]):
        if family != "random":
            continue
        block = block.sort_values("n")
        entry = {"planner": planner, "k": int(k)}
        for (n0, t0, e0), (n1, t1, e1) in zip(
                block[["n", "compile_cpu_s", "evaluations"]].values[:-1],
                block[["n", "compile_cpu_s", "evaluations"]].values[1:]):
            if n0 >= 16 and t0 > 0 and t1 > 0:
                entry[f"{int(n0)}->{int(n1)}"] = (
                    f"{math.log(t1 / t0) / math.log(n1 / n0):.2f}"
                    + (f" ({math.log(e1 / e0) / math.log(n1 / n0):.2f})" if e0 > 0 else ""))
        local.append(entry)
    if local:
        lines.append("### Local slopes between successive sizes, random family: CPU time "
                     "(evaluations)\n")
        lines.append(markdown(pd.DataFrame(local).fillna("")))
        lines.append("")

    reach = []
    for (planner, family, k), block in frame.groupby(["planner", "family", "k"]):
        good = block[block["status"] == "ok"].sort_values("n")
        bad = block[block["status"] != "ok"].sort_values("n")
        entry = {"planner": planner, "family": family, "k": int(k),
                 "max n": int(good["n"].max()) if not good.empty else 0}
        if not good.empty:
            last = good.iloc[-1]
            entry["CPU s at max n"] = _fmt_s(last.get("compile_cpu_s"))
            entry["wall s at max n"] = _fmt_s(last.get("compile_wall_s"))
            entry["peak RSS (GB)"] = f"{last.get('peak_rss_mb', math.nan) / 1024:.1f}"
        if not bad.empty:
            first = bad.iloc[0]
            note = f"n={int(first['n'])}: {first['status']}"
            if pd.notna(first.get("predicted_plan_s", math.nan)):
                note += f" (predicted {first['predicted_plan_s']:.0f} s)"
            entry["next size"] = note
        reach.append(entry)
    lines.append("### Reach per series (cap 30 min per planner run)\n")
    lines.append(markdown(pd.DataFrame(reach).fillna("")))
    lines.append("")

    random = ok[ok["family"] == "random"]
    if not random.empty:
        lines.append("### Compile CPU seconds (wall) at fixed n, random family\n")
        sizes = [n for n in TICKS if n >= 32 and n in set(random["n"])]
        table = []
        for (planner, k), block in random.groupby(["planner", "k"]):
            entry = {"planner": planner, "k": int(k)}
            for n in sizes:
                cell = block[block["n"] == n]
                if not cell.empty:
                    entry[f"n={n}"] = (f"{_fmt_s(cell['compile_cpu_s'].iloc[0])} "
                                       f"({_fmt_s(cell['compile_wall_s'].iloc[0])})")
            table.append(entry)
        lines.append(markdown(pd.DataFrame(table).fillna("")))
        lines.append("")
        lines.append("### Quality against the fair naive baseline (random family; %)\n")
        table = []
        for (planner, k), block in random.groupby(["planner", "k"]):
            for metric, label in (("red_2q_pct", "2q"), ("red_depth_pct", "depth"),
                                  ("red_1q_pct", "1q"),
                                  ("red_2q_passes_only_pct", "2q passes only")):
                entry = {"planner": planner, "k": int(k), "metric": label}
                for n in [n for n in TICKS if n in set(block["n"])]:
                    entry[f"n={n}"] = round(float(block[block["n"] == n][metric].iloc[0]), 1)
                table.append(entry)
        lines.append(markdown(pd.DataFrame(table).fillna("")))
        lines.append("")
        lines.append("### Where planner time goes (median run, random family)\n")
        table = []
        for (planner, k), block in random.groupby(["planner", "k"]):
            for _, row in block.sort_values("n").iterrows():
                if row["n"] < 32:
                    continue
                table.append({"planner": planner, "k": int(k), "n": int(row["n"]),
                              "descent %": round(100 * row["descent_share"], 1),
                              "exact translation %": round(100 * row["translate_share"], 1),
                              "translator calls": int(row["translator_calls"]),
                              "evaluations": int(row["evaluations"]),
                              "us/evaluation": round(row["us_per_evaluation"], 1),
                              "cost terms computed": int(row["cost_term_computations"])})
        lines.append(markdown(pd.DataFrame(table)))
        lines.append("")

    exact = ok[ok["planner"] == "w10i"].merge(
        ok[ok["planner"] == "w10"], on=["family", "k", "n"], suffixes=("", "_w10"))
    if pool is not None and not pool.empty:
        pool_ok = pool[pool["status"] == "ok"]
        exact = pd.concat([exact, pool_ok[pool_ok["planner"] == "w10i"].merge(
            pool_ok[pool_ok["planner"] == "w10"], on=["family", "k", "n"],
            suffixes=("", "_w10"))], ignore_index=True)
    if not exact.empty:
        same = ((exact["placement"] == exact["placement_w10"])
                & (exact["evaluations"] == exact["evaluations_w10"])
                & (exact["model_cost"] == exact["model_cost_w10"]))
        lines.append(f"### Incremental evaluation (w10i) against the library (w10)\n\n"
                     f"{int(same.sum())}/{len(exact)} cells identical in placement, descent "
                     f"evaluations and model cost (bit for bit).")
        if (~same).any():
            lines.append("Differences: " + ", ".join(
                f"{r.family} k={r.k} n={r.n}" for r in exact[~same].itertuples()))
        both = exact.dropna(subset=["plan_cpu_s", "plan_cpu_s_w10"])
        if not both.empty:
            both = both.assign(speedup=both["plan_cpu_s_w10"] / both["plan_cpu_s"])
            table = both[both["family"] == "random"].pivot_table(index="k", columns="n",
                                                                 values="speedup")
            if not table.empty:
                lines.append("\nPlanner CPU-time speed-up of w10i over w10 (random family):\n")
                lines.append(markdown(table.round(1), index=True))
        lines.append("")

    if "pool_cells_overlapping" in frame:
        lines.append("### Timing conditions\n")
        table = []
        for (planner, family), block in ok.groupby(["planner", "family"]):
            table.append({"planner": planner, "family": family, "cells": len(block),
                          "load1 min": round(block["load1_mean"].min(), 1),
                          "load1 max": round(block["load1_mean"].max(), 1),
                          "wall/CPU max": round((block["plan_wall_s"] / block["plan_cpu_s"]).max(), 2),
                          "cells with pool running": int((block["pool_cells_overlapping"] > 0).sum())})
        lines.append(markdown(pd.DataFrame(table)))
        lines.append("")
    if "retime_plan_cpu_s" in frame:
        again = ok.dropna(subset=["retime_plan_cpu_s"])
        if not again.empty:
            lines.append("### Re-timed sequentially, nothing else of ours running\n")
            table = again[["planner", "family", "k", "n", "plan_cpu_s", "retime_plan_cpu_s",
                           "plan_wall_s", "retime_plan_wall_s", "load1_mean",
                           "retime_load1_mean"]].copy()
            table["CPU ratio (first/retime)"] = table["plan_cpu_s"] / table["retime_plan_cpu_s"]
            lines.append(markdown(table.round(2)))
            lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------

K_LABEL = {k: f"$k={k}$" for k in (2, 4, 6, 8, 10)}
PLANNER_LABEL = {"w10": "silent-aware $w=10$ (as implemented)",
                 "w10i": "same, incremental evaluation (identical plans)",
                 "v4": "v4 (submitted)", "w10T": "$w=10$ + transversal CZ/XCX"}
FAMILY_LABEL = {"random": "random", "ghz": "GHZ", "bv": "BV", "qft": "banded QFT",
                "ising": "Ising", "qb_ghz": "ghz", "qb_bv": "bv", "qb_ising": "ising",
                "qb_knn": "knn", "qb_swap_test": "swap_test", "qb_adder": "adder",
                "qb_wstate": "wstate"}
TICKS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def with_derived(frame):
    """Compile time = planner + translating its plan; descent share of planner time."""
    frame = frame.copy()
    if "translate_wall_s" in frame:
        frame["compile_wall_s"] = frame["plan_wall_s"] + frame["translate_wall_s"]
        frame["compile_cpu_s"] = frame["plan_cpu_s"] + frame["translate_cpu_s"]
    if "median_run_phase_descent_s" in frame:
        phases = [c for c in frame.columns if c.startswith("median_run_phase_")]
        total = frame[phases].sum(axis=1)
        frame["descent_share"] = frame["median_run_phase_descent_s"] / total
        frame["translate_share"] = frame["median_run_phase_translate_s"] / total
    if "evaluations" in frame and "phase_descent_s" in frame:
        frame["us_per_evaluation"] = 1e6 * frame["phase_descent_s"] / frame["evaluations"]
    return frame


def _styles() -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt

    return [dict(style) for style in plt.rcParams["axes.prop_cycle"]]


def _log2_x(axis, lo: float, hi: float, label: bool = True, max_labels: int = 6) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    axis.set_xscale("log", base=2)
    ticks = [t for t in TICKS if lo <= t <= hi]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    # Every other label once there are more than `max_labels` (5 fit a column-width panel, 6
    # a full-width one) -- counted from the widest size, so the largest n is always labelled.
    labels = [str(t) if len(ticks) <= max_labels or (len(ticks) - 1 - index) % 2 == 0 else ""
              for index, t in enumerate(ticks)]
    axis.xaxis.set_major_formatter(FixedFormatter(labels))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlim(lo / 1.3, hi * 1.3)
    if label:
        axis.set_xlabel("program qubits $n$")


def _plain_log_y(axis, powers: bool = False) -> None:
    """
    Decades labelled plainly -- 0.1, 1, 10, 100 s, no offsets, no 1e+02 -- or, for counts
    spanning many decades (`powers`), uniformly as 10^p.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: rf"$10^{{{int(round(math.log10(value)))}}}$"
        if powers or not 1e-3 <= value <= 1e4 else f"{value:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _percent_axis(axis, floor: Optional[float] = None) -> None:
    """
    y from zero, as house style asks -- unless a reduction is negative, which must show. With
    `floor`, a lone extreme below it is drawn as a marker on the axis edge with its value
    written beside it, so it does not flatten every other series (say so in the caption).
    """
    lines = [line for line in axis.get_lines() if len(line.get_ydata())]
    lows = [min(line.get_ydata()) for line in lines]
    highs = [max(line.get_ydata()) for line in lines]
    low = min(lows) if lows else 0.0
    high = max(highs) if highs else 5.0
    if floor is not None and low < floor:
        for line in lines:
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                if y < floor:
                    axis.plot([x], [floor], marker="v", markersize=5, color=line.get_color(),
                              linestyle="none", clip_on=False)
                    axis.annotate(f"{y:.0f}%", (x, floor), xytext=(-4, 4),
                                  textcoords="offset points", ha="right", fontsize=8,
                                  color="#3b4c56")
        low = floor
    if low < 0:
        axis.axhline(0, color="#636363", linestyle="--", linewidth=1.0, zorder=0.5)
        axis.set_ylim(floor if floor is not None and low <= floor else low * 1.15 - 1,
                      max(5.0, high * 1.08))
    else:
        axis.set_ylim(0, max(5.0, high * 1.08))


def _save(fig, stem: str, output: Optional[str]) -> List[Path]:
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure

    paths = save_figure(fig, stem, Path(__file__), override=Path(output) if output else None,
                        versioned=True)
    plt.close(fig)
    return paths


def plot_main(frame, output: Optional[str], *, planner: str = "w10",
              family: str = "random") -> List[Path]:
    """Full width: (a) compile time, (b) descent evaluations, (c) 2q and (d) depth reduction."""
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, use_style

    use_style()
    frame = with_derived(frame)
    ok = frame[(frame["planner"] == planner) & (frame["family"] == family)
               & (frame["status"] == "ok")]
    if ok.empty:
        return []
    fits = fit_exponents(frame)
    fits = fits[(fits["planner"] == planner) & (fits["family"] == family)]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.3))
    styles = _styles()
    handles = []
    for index, k in enumerate(sorted(int(k) for k in ok["k"].unique())):
        style = styles[index % len(styles)]
        block = ok[ok["k"] == k].sort_values("n")
        label = K_LABEL.get(k, f"$k={k}$")
        fit = fits[(fits["k"] == k) & (fits["metric"] == "compile_cpu_s")]
        if not fit.empty:
            label += rf": $\propto n^{{{fit['exponent'].iloc[0]:.1f}}}$"
        (line,) = axes[0].plot(block["n"], block["compile_cpu_s"], label=label, **style)
        handles.append(line)
        multi = block[block["n"] > k]  # one patch at n <= k: nothing to search
        axes[1].plot(multi["n"], multi["evaluations"], **style)
        axes[2].plot(block["n"], block["red_2q_pct"], **style)
        axes[3].plot(block["n"], block["red_depth_pct"], **style)
    lo, hi = ok["n"].min(), ok["n"].max()
    axes[0].axhline(1800, color="#636363", linestyle=":", linewidth=1.0)
    axes[0].text(lo, 1800 * 1.15, "30-min cap", fontsize=8, color="#636363", va="bottom")
    # Reference slope n^2 through the median k's n = 32 evaluations.
    anchor = ok[(ok["n"] == 32)]["evaluations"].median()
    if anchor and not math.isnan(anchor):
        xs = [16, hi]
        axes[1].plot(xs, [anchor * (x / 32) ** 2 for x in xs], color="#636363",
                     linestyle=":", linewidth=1.0, marker="")
        axes[1].text(hi, anchor * (hi / 32) ** 2 * 0.25, r"$\propto n^2$", fontsize=8,
                     color="#636363", ha="right", va="top")
    for axis in axes:
        _log2_x(axis, lo, hi)
    _plain_log_y(axes[0])
    _plain_log_y(axes[1], powers=True)
    axes[0].set_ylabel("compile CPU time (s)")
    axes[1].set_ylabel("descent evaluations")
    axes[2].set_ylabel("2q-gate reduction (%)")
    axes[3].set_ylabel("depth reduction (%)")
    for axis in axes[2:]:
        _percent_axis(axis)
    for letter, axis in zip("abcd", axes):
        axis.set_title(f"({letter})", loc="left", fontweight="bold")
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles),
               handlelength=2.4, columnspacing=1.2)
    return _save(fig, f"{NAME}_{planner}_{family}", output)


def plot_compact(frame, output: Optional[str], *, planner: str = "w10",
                 family: str = "random", reference: Optional[str] = None) -> List[Path]:
    """
    Single column, the paper candidate: (a) compile CPU time and (b) 2q-gate reduction
    against n, one series per k. `reference` overlays another planner's times as thin dotted
    lines in the same colours (e.g. the as-implemented planner under the incremental one).
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, use_style

    use_style()
    frame = with_derived(frame)
    ok = frame[(frame["family"] == family) & (frame["status"] == "ok")]
    main = ok[ok["planner"] == planner]
    if main.empty:
        return []
    fits = fit_exponents(frame)
    fits = fits[(fits["planner"] == planner) & (fits["family"] == family)
                & (fits["metric"] == "compile_cpu_s")]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.55))
    styles = _styles()
    handles = []
    for index, k in enumerate(sorted(int(k) for k in main["k"].unique())):
        style = styles[index % len(styles)]
        block = main[main["k"] == k].sort_values("n")
        label = K_LABEL.get(k, f"$k={k}$")
        fit = fits[fits["k"] == k]
        if not fit.empty:
            label += rf": $n^{{{fit['exponent'].iloc[0]:.1f}}}$"
        (line,) = axes[0].plot(block["n"], block["compile_cpu_s"], label=label, **style)
        handles.append(line)
        axes[1].plot(block["n"], block["red_2q_pct"], **style)
        if reference:
            other = ok[(ok["planner"] == reference) & (ok["k"] == k)].sort_values("n")
            axes[0].plot(other["n"], other["compile_cpu_s"], color=style["color"],
                         linestyle=":", linewidth=1.1, marker="")
    if reference:
        from matplotlib.lines import Line2D

        handles.append(Line2D([], [], color="#636363", linestyle=":", linewidth=1.1,
                              label="dotted: as implemented"))
    lo, hi = main["n"].min(), main["n"].max()
    for axis in axes:
        _log2_x(axis, lo, hi, max_labels=5)
    _plain_log_y(axes[0])
    axes[0].set_ylabel("compile CPU time (s)")
    axes[1].set_ylabel("2q-gate reduction (%)")
    _percent_axis(axes[1])
    for letter, axis in zip("ab", axes):
        axis.set_title(f"({letter})", loc="left", fontweight="bold")
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8,
               handlelength=2.2, columnspacing=0.9)
    suffix = f"_vs_{reference}" if reference else ""
    return _save(fig, f"{NAME}_compact_{planner}{suffix}_{family}", output)


def plot_breakdown(frame, output: Optional[str], *, planner: str = "w10",
                   family: str = "random") -> List[Path]:
    """Single column: (a) compile time against k at fixed n, (b) descent share of planner time."""
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, use_style

    use_style()
    frame = with_derived(frame)
    ok = frame[(frame["planner"] == planner) & (frame["family"] == family)
               & (frame["status"] == "ok")]
    if ok.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.3))
    styles = _styles()
    sizes = [n for n in TICKS if n >= 32 and n in set(ok["n"])][-5:]
    for index, n in enumerate(sizes):
        block = ok[ok["n"] == n].sort_values("k")
        axes[0].plot(block["k"], block["compile_cpu_s"], **styles[index % len(styles)])
        # Labelled at the line's end rather than in a legend, which had nowhere empty to go.
        last = block.iloc[-1]
        axes[0].annotate(f"$n={n}$", (last["k"], last["compile_cpu_s"]), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8)
    _plain_log_y(axes[0])
    ks = sorted(int(k) for k in ok["k"].unique())
    axes[0].set_xticks(ks)
    axes[0].set_xlim(min(ks) - 0.5, max(ks) + 3.2)
    axes[0].set_xlabel("code size $k$")
    axes[0].set_ylabel("compile CPU time (s)")
    for index, k in enumerate(sorted(int(k) for k in ok["k"].unique())):
        block = ok[(ok["k"] == k) & (ok["n"] > k)].sort_values("n")
        axes[1].plot(block["n"], 100 * block["descent_share"], label=K_LABEL.get(k),
                     **styles[index % len(styles)])
    _log2_x(axes[1], ok["n"].min(), ok["n"].max(), max_labels=5)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("descent share of planner (%)")
    axes[1].legend(loc="lower right", fontsize=8, handlelength=2.0)
    for letter, axis in zip("ab", axes):
        axis.set_title(f"({letter})", loc="left", fontweight="bold")
    return _save(fig, f"{NAME}_breakdown_{planner}_{family}", output)


def plot_planners(frame, output: Optional[str], *, k: int = 2,
                  family: str = "random") -> List[Path]:
    """Single column: (a) compile time and (b) 2q reduction against n for each planner at k."""
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, use_style

    use_style()
    frame = with_derived(frame)
    ok = frame[(frame["k"] == k) & (frame["family"] == family) & (frame["status"] == "ok")]
    planners = [p for p in ("w10", "w10i", "v4", "w10T") if p in set(ok["planner"])]
    if len(planners) < 2:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.4))
    styles = _styles()
    for index, planner in enumerate(planners):
        block = ok[ok["planner"] == planner].sort_values("n")
        style = styles[index % len(styles)]
        axes[0].plot(block["n"], block["compile_cpu_s"], label=PLANNER_LABEL[planner], **style)
        axes[1].plot(block["n"], block["red_2q_pct"], **style)
    for axis in axes:
        _log2_x(axis, ok["n"].min(), ok["n"].max(), max_labels=5)
    _plain_log_y(axes[0])
    axes[0].set_ylabel("compile CPU time (s)")
    axes[1].set_ylabel("2q-gate reduction (%)")
    _percent_axis(axes[1])
    for letter, axis in zip("ab", axes):
        axis.set_title(f"({letter})", loc="left", fontweight="bold")
    fig.legend(loc="outside upper center", ncol=1, fontsize=8, handlelength=2.4)
    return _save(fig, f"{NAME}_planners_k{k}_{family}", output)


def plot_families(frame, output: Optional[str], *, k: int = 2, planner: str = "w10",
                  rows=(("generated", ("random", "ghz", "bv", "qft", "ising")),
                        ("QASMBench", ("qb_ghz", "qb_bv", "qb_ising", "qb_knn", "qb_adder")))
                  ) -> List[Path]:
    """
    Full width, one row per corpus: compile CPU time (timing cells only), 2q and depth
    reduction against n. w10 and w10i are the same planner -- bit-identical plans and counts,
    checked per cell -- so either supplies a family's quality; the time panel shows the
    incremental implementation where it was timed (sequential cells), else the library's.
    """
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, use_style

    use_style()
    frame = with_derived(frame)
    planners = (planner, "w10i") if planner == "w10" else (planner,)
    ok = frame[(frame["k"] == k) & (frame["planner"].isin(planners)) & (frame["status"] == "ok")]
    valid = ok["timing_valid"].fillna(True).astype(bool) if "timing_valid" in ok else True
    ok = ok.assign(_timed=valid, _incremental=ok["planner"] == "w10i")
    timed = ok[ok["_timed"]].sort_values("_incremental", ascending=False).drop_duplicates(
        subset=["family", "k", "n"])
    ok = ok.sort_values(["_timed", "_incremental"], ascending=False).drop_duplicates(
        subset=["family", "k", "n"])
    rows = [(name, [f for f in families if f in set(ok["family"])]) for name, families in rows]
    rows = [(name, families) for name, families in rows if families]
    if not rows:
        return []
    fig, axes = plt.subplots(len(rows), 3, figsize=(FULL_WIDTH, 2.0 * len(rows) + 0.3),
                             squeeze=False, sharex=True)
    styles = _styles()
    lo, hi = ok["n"].min(), ok["n"].max()
    for row_index, (name, families) in enumerate(rows):
        for index, family in enumerate(families):
            block = ok[(ok["family"] == family) & (ok["n"] > k)].sort_values("n")
            clock = timed[(timed["family"] == family) & (timed["n"] > k)].sort_values("n")
            style = styles[index % len(styles)]
            axes[row_index][0].plot(clock["n"], clock["compile_cpu_s"],
                                    label=FAMILY_LABEL[family], **style)
            axes[row_index][1].plot(block["n"], block["red_2q_pct"], **style)
            axes[row_index][2].plot(block["n"], block["red_depth_pct"], **style)
        _plain_log_y(axes[row_index][0])
        axes[row_index][0].set_ylabel("compile CPU time (s)")
        axes[row_index][1].set_ylabel("2q-gate reduction (%)")
        axes[row_index][2].set_ylabel("depth reduction (%)")
        axes[row_index][0].legend(loc="upper left", fontsize=8, title=name, title_fontsize=8,
                                  handlelength=2.0)
        _percent_axis(axes[row_index][1])
        _percent_axis(axes[row_index][2], floor=-60.0)
        for column, axis in enumerate(axes[row_index]):
            _log2_x(axis, lo, hi, label=row_index == len(rows) - 1)
            axis.set_title(f"({'abcdef'[row_index * 3 + column]})", loc="left",
                           fontweight="bold")
    return _save(fig, f"{NAME}_families_k{k}", output)


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--planners", nargs="+", default=["w10"], choices=sorted(PLANNERS))
    parser.add_argument("--families", nargs="+", default=["random"], choices=FAMILIES)
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--order", choices=["size", "series"], default="size")
    parser.add_argument("--cap-seconds", type=float, default=1800.0)
    parser.add_argument("--predict-factor", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-hours", type=float, default=3.0)
    parser.add_argument("--mem-cap-gb", type=float, default=12.0)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help=">1: non-timing mode, series run concurrently (rows flagged)")
    parser.add_argument("--background", action="store_true",
                        help="launch children under `taskpolicy -b` (efficiency cores)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--runs", default=str(RUNS), help="append-only JSONL of raw records")
    parser.add_argument("--csv", default=str(CSV), help="flattened CSV, regenerated from --runs")
    parser.add_argument("--from-csv", default=None, help="redraw and summarize, no compute")
    parser.add_argument("--output-dir", default=None, help="figure directory override")
    parser.add_argument("--pool-csv", default=str(OUT / "compile_scaling_pool.csv"),
                        help="non-timing family runs, for the families figure")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--finalize", action="store_true",
                        help="flag concurrency, merge re-times, write compile_scaling_final.csv")
    parser.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cell-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell:
        try:
            row = run_cell(json.loads(args.cell))
        except BaseException as error:  # noqa: BLE001 - the parent records it
            spec = json.loads(args.cell)
            row = {**{key: spec[key] for key in ("planner", "family", "k", "n")},
                   "status": "error", "error": traceback.format_exc()[-1500:],
                   "plan_wall_s": math.nan}
            if isinstance(error, KeyboardInterrupt):
                raise
        Path(args.cell_out).write_text(json.dumps(row, default=str))
        return

    import pandas as pd

    pool = pd.read_csv(args.pool_csv) if args.pool_csv and Path(args.pool_csv).exists() else None
    if args.finalize:
        frame = finalize(Path(args.runs), OUT / "compile_scaling_pool_runs.jsonl",
                         OUT / "compile_scaling_retime_runs.jsonl")
        target = OUT / "compile_scaling_final.csv"
        frame.to_csv(target, index=False)
        fits = fit_exponents(frame)
        fits.to_csv(OUT / "compile_scaling_final_fits.csv", index=False)
        print(f"[{NAME}] wrote {target} ({len(frame)} cells) and its fits")
    elif args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        settings = Settings(
            planners=tuple(args.planners), families=tuple(args.families), ks=tuple(args.ks),
            sizes=tuple(args.sizes), order=args.order, cap_seconds=args.cap_seconds,
            predict_factor=args.predict_factor, repeats=args.repeats,
            budget_hours=args.budget_hours, mem_cap_gb=args.mem_cap_gb, max_n=args.max_n,
            workers=args.workers, background=args.background, tag=args.tag,
            runs=args.runs, csv=args.csv)
        run(settings)
        frame = pd.read_csv(args.csv)
    fits = fit_exponents(frame)
    if not args.from_csv and not args.finalize and not fits.empty:
        fits.to_csv(Path(args.csv).with_name(Path(args.csv).stem + "_fits.csv"), index=False)
    print(summarize(frame, pool))
    if args.no_plot:
        return
    out = args.output_dir
    paths = plot_main(frame, out, planner="w10") + plot_main(frame, out, planner="w10i")
    paths += plot_compact(frame, out, planner="w10")
    paths += plot_compact(frame, out, planner="w10i", reference="w10")
    paths += plot_breakdown(frame, out, planner="w10") + plot_breakdown(frame, out, planner="w10i")
    paths += plot_planners(frame, out, k=2)
    family_rows = frame[frame["planner"].isin(["w10", "w10i"])]
    if pool is not None:
        family_rows = pd.concat([pool, family_rows], ignore_index=True)
    paths += plot_families(family_rows, out)
    for path in paths:
        print(f"[{NAME}] saved {path}")


if __name__ == "__main__":
    main()
