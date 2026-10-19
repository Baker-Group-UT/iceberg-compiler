"""
common.py -- plumbing shared by every revision experiment.

Five jobs, each of which an earlier figure in this repo got wrong at least once:

*provenance*  every CSV gets a JSON sidecar (git SHA, `COST_MODEL_VERSION`, settings, seeds,
              wall time) and a `cost_model_version` column via `stamp_provenance`.

*resumable tables*  `ResultTable` appends rows as work items finish and, on restart, skips
              keys already written -- but only after checking that the sidecar's settings
              fingerprint matches the current run. A mismatch is refused, never merged: a
              CSV mixing two settings is the silent-reuse failure (gotcha 6) by another name.

*streaming parallelism*  `stream_map` is `experiments.shared.parallel.parallel_map` with a
              callback per finished item, so a crash loses one circuit rather than the run.
              Spawn, not fork, so work functions must be module-level (gotcha 1).

*circuits*    `load_fig6_circuits` returns exactly the 71 circuits behind Fig 6 -- same
              loader, same names, same `circuit_idx` (which feeds the seed derivation) -- and
              refuses to continue if it cannot.

*simulation with distributions*  `simulate_arm` mirrors `pipeline.tasks.scan_specific
              .run_specific_scan` call for call, so a seeded arm reproduces Fig 6 bit for bit,
              but returns the post-selected distributions instead of discarding them. The
              Experiment4 statistics (`tvd_statistics`, `equal_shot_tvd`) need those counts.
"""

from __future__ import annotations

import os

# Before anything imports tqdm: its env overrides are read when the module is imported, and
# per-shot bars under a process pool cost 2.6x throughput (gotcha 2).
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import contextlib
import io
import json
import math
import multiprocessing as mp
import platform
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "revision"
INPUTS = RESULTS / "_inputs"
SMOKE = RESULTS / "_smoke"
FIG6_CSV = INPUTS / "merging_improvement_benchmark_stylized_simulated.csv"
FIG8_CSV = INPUTS / "resource_selection_headroom.csv"
OLD_EXHAUSTIVE_CSV = INPUTS / "8_optimality_vs_exhaustive.csv"
QASM_ROOT = REPO / "benchmarks" / "qasmbench"

Perm = List[Tuple[int, int]]

#: The 4-6 qubit QASMBench circuits the plan names for Experiment7 and Experiment3b.
SMALL_QASMBENCH: Tuple[str, ...] = (
    "adder_n4", "basis_trotter_n4", "bell_n4", "cat_state_n4", "error_correctiond3_n5",
    "hs4_n4", "inverseqft_n4", "lpn_n5", "pea_n5", "qaoa_n6", "qec_en_n5", "qec_sm_n5",
    "qft_n4", "qrng_n4", "shor_n5", "simon_n6", "variational_n4", "vqe_n4",
    "vqe_uccsd_n4", "vqe_uccsd_n6",
)


def log(tag: str, message: str) -> None:
    """Timestamped, flushed -- these run unattended and the log is the only witness."""
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {message}", flush=True)


# ----------------------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------------------


def git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked_files": len([line for line in status.splitlines() if line.strip()]),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("qiskit", "qiskit_aer", "stim", "numpy", "pandas", "networkx", "scipy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:  # noqa: BLE001
            versions[name] = "missing"
    return versions


def sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".json")


def _logical_state() -> str:
    from translator.logical_optimization import state

    return state()


def write_sidecar(
    csv_path: Path,
    *,
    experiment: str,
    settings: Dict[str, Any],
    started: float,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """`<csv>.json`: everything needed to say what produced the CSV beside it."""
    from packing.version import COST_MODEL_VERSION

    payload = {
        "experiment": experiment,
        "csv": str(csv_path),
        "status": status,
        "git": git_state(),
        "cost_model_version": COST_MODEL_VERSION,
        "logical_optimization": _logical_state(),
        "settings": _jsonable(settings),
        "fingerprint": fingerprint(settings),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(time.time() - started, 1),
        "argv": sys.argv,
        "host": socket.gethostname(),
        "packages": package_versions(),
    }
    if extra:
        payload["extra"] = _jsonable(extra)
    path = sidecar_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)
    return path


#: Settings that change how fast a run goes but not what it computes. Excluded from the
#: fingerprint so a resume at a different `--jobs` is not refused.
VOLATILE_SETTINGS = {"jobs", "out", "resume", "limit"}


def fingerprint(settings: Dict[str, Any]) -> str:
    from packing.version import COST_MODEL_VERSION

    from translator.logical_optimization import enabled, state

    stable = {k: v for k, v in _jsonable(settings).items() if k not in VOLATILE_SETTINGS}
    stable["cost_model_version"] = COST_MODEL_VERSION
    # Part of what produced the rows, so a table started without the logical optimisation
    # pass is never resumed with it. Absent when off, so older tables still resume then.
    if enabled():
        stable["logical_optimization"] = state()
    return json.dumps(stable, sort_keys=True)


# ----------------------------------------------------------------------------------------
# resumable result tables
# ----------------------------------------------------------------------------------------


class ResultTable:
    """
    One CSV, appended to as work finishes, resumable by key, refusing mixed settings.

    On construction against an existing CSV the sidecar must exist and carry the same
    fingerprint; otherwise this raises rather than mixing two runs in one file. Pass a new
    `--out` to start over -- nothing here deletes or overwrites a finished table.
    """

    def __init__(
        self,
        path: Path,
        *,
        key_columns: Sequence[str],
        experiment: str,
        settings: Dict[str, Any],
        resume: bool = True,
    ) -> None:
        self.path = Path(path)
        self.key_columns = list(key_columns)
        self.experiment = experiment
        self.settings = settings
        self.started = time.time()
        self.done: set = set()
        self.rows_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            side = sidecar_path(self.path)
            if not resume:
                raise SystemExit(
                    f"{self.path} exists and --no-resume was given. Refusing to overwrite; "
                    f"pass a new --out."
                )
            if not side.exists():
                raise SystemExit(
                    f"{self.path} exists without a sidecar, so its settings are unknown. "
                    f"Refusing to append to it; pass a new --out."
                )
            recorded = json.loads(side.read_text()).get("fingerprint")
            if recorded != fingerprint(settings):
                raise SystemExit(
                    f"{self.path} was produced with different settings (fingerprint "
                    f"mismatch). Refusing to mix runs; pass a new --out.\n"
                    f"  recorded: {recorded}\n  current:  {fingerprint(settings)}"
                )
            existing = pd.read_csv(self.path)
            if not existing.empty:
                self.done = {
                    tuple(_key_value(v) for v in row)
                    for row in existing[self.key_columns].itertuples(index=False)
                }
            log(experiment, f"resuming {self.path.name}: {len(self.done)} keys already "
                            f"written, they will be skipped")
        self.write_sidecar(status="running")

    def is_done(self, key: Sequence[Any]) -> bool:
        return tuple(_key_value(v) for v in key) in self.done

    def append(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        from experiments.shared.figures import stamp_provenance

        frame = stamp_provenance(pd.DataFrame(list(rows)))
        if self.path.exists():
            header = list(pd.read_csv(self.path, nrows=0).columns)
            extra = [c for c in frame.columns if c not in header]
            if extra:
                # Union the schema with an atomic rewrite. Rare: only when an arm reports a
                # column the first rows did not.
                old = pd.read_csv(self.path)
                merged = pd.concat([old, frame], ignore_index=True, sort=False)
                tmp = self.path.with_suffix(".tmp.csv")
                merged.to_csv(tmp, index=False)
                tmp.replace(self.path)
            else:
                frame.reindex(columns=header).to_csv(
                    self.path, mode="a", header=False, index=False
                )
        else:
            frame.to_csv(self.path, index=False)
        for row in rows:
            self.done.add(tuple(_key_value(row.get(c)) for c in self.key_columns))
        self.rows_written += len(rows)

    def frame(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def write_sidecar(self, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        write_sidecar(
            self.path, experiment=self.experiment, settings=self.settings,
            started=self.started, status=status, extra=extra,
        )


def _key_value(value: Any) -> Any:
    """Normalise a key cell so CSV round trips (ints read back as floats, etc.) match."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return repr(round(float(value), 12))
    return str(value)


# ----------------------------------------------------------------------------------------
# parallelism
# ----------------------------------------------------------------------------------------


def _guarded(payload: Tuple[Callable, Any]) -> Tuple[str, Any, float]:
    function, item = payload
    started = time.time()
    try:
        return "ok", function(item), time.time() - started
    except Exception:  # noqa: BLE001 - one bad item must not end the sweep
        return "error", traceback.format_exc(), time.time() - started


def stream_map(
    function: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    jobs: int,
    label: str,
    on_result: Callable[[int, Any], None],
) -> Dict[str, int]:
    """
    Run `function` over `items` in a spawn pool, handing each result to `on_result` in the
    parent as it completes. Failures are logged with their traceback and counted, never
    raised. `jobs=1` runs in-process for debugging.
    """
    total = len(items)
    counts = {"ok": 0, "error": 0}
    if total == 0:
        return counts
    started = time.time()

    def progress(done: int) -> None:
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        left = (total - done) / rate if rate > 0 else float("nan")
        log(label, f"{done}/{total} done, {elapsed / 60:.1f} min elapsed, "
                   f"~{left / 60:.1f} min left ({counts['error']} failed)")

    def handle(index: int, outcome: Tuple[str, Any, float]) -> None:
        status, value, seconds = outcome
        counts[status] += 1
        if status == "ok":
            try:
                on_result(index, value)
            except Exception:  # noqa: BLE001
                counts["error"] += 1
                log(label, f"item {index}: writing its result failed\n{traceback.format_exc()}")
        else:
            log(label, f"item {index} FAILED after {seconds:.0f}s:\n{value}")

    step = max(1, total // 25)
    if jobs <= 1:
        for index, item in enumerate(items):
            handle(index, _guarded((function, item)))
            if (index + 1) % step == 0 or index + 1 == total:
                progress(index + 1)
        return counts

    workers = max(1, min(jobs, total))
    log(label, f"{total} items across {workers} workers")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(_guarded, (function, item)): i for i, item in enumerate(items)}
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcome = future.result()
            except Exception:  # noqa: BLE001 - a dead worker, not a raising item
                outcome = ("error", traceback.format_exc(), 0.0)
            handle(index, outcome)
            done += 1
            if done % step == 0 or done == total:
                progress(done)
    return counts


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


@contextlib.contextmanager
def quiet_stdout():
    """`generate_permutations` prints a progress line per hundredth of a second."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ----------------------------------------------------------------------------------------
# circuits
# ----------------------------------------------------------------------------------------


@dataclass
class BenchCircuit:
    """One Fig 6 benchmark: the original (gate counts) and its Clifford proxy (simulation)."""

    size_group: str
    circuit_idx: int
    name: str
    original: Any
    proxy: Any

    @property
    def num_qubits(self) -> int:
        return self.original.num_qubits

    @property
    def family(self) -> str:
        return circuit_family(self.name)


def circuit_family(name: str) -> str:
    """`ising_n26` -> `ising`, `vqe_uccsd_n6` -> `vqe_uccsd`, `qft_n4` -> `qft`."""
    head = name.lower()
    if "_n" in head:
        head = head[: head.rfind("_n")]
    return head


def _qasm_by_name(group_dir: Path) -> Dict[str, Path]:
    from experiments.shared.benchmark_circuits import _read_circuit_name, find_qasm_files

    out: Dict[str, Path] = {}
    for path in find_qasm_files(group_dir):
        name = _read_circuit_name(path.parent / "README.md") or path.stem
        out.setdefault(name, path)
    return out


def fig6_reference() -> pd.DataFrame:
    """One row per Fig 6 circuit: name, size group, index, calibration, baseline, packed."""
    frame = pd.read_csv(FIG6_CSV)
    per = frame.drop_duplicates("circuit_name").set_index("circuit_name")
    totals = frame.groupby("circuit_name")[[
        "pct_improvement_logical_selection_rate",
        "pct_reduction_logical_total_variation_distance",
        "pct_reduction_one_qubit_gates",
        "pct_reduction_two_qubit_gates",
        "pct_reduction_circuit_depth",
    ]].sum()
    out = per[[
        "size_group", "circuit_idx", "num_qubits", "baseline_gate_count", "alpha_initial",
        "alpha_final", "pilot_initial_lsr", "pilot_final_lsr", "final_repetitions",
        "baseline_lsr", "baseline_tvd",
    ]].copy()
    out["packed_lsr"] = out["baseline_lsr"] * (
        1 + totals["pct_improvement_logical_selection_rate"] / 100.0
    )
    out["packed_tvd"] = out["baseline_tvd"] * (
        1 - totals["pct_reduction_logical_total_variation_distance"] / 100.0
    )
    hmerge = frame[frame["component"].isin(["h_commutation", "merging"])].groupby(
        "circuit_name"
    )[["pct_improvement_logical_selection_rate",
       "pct_reduction_logical_total_variation_distance"]].sum()
    out["hmerge_lsr"] = out["baseline_lsr"] * (
        1 + hmerge["pct_improvement_logical_selection_rate"] / 100.0
    )
    out["hmerge_tvd"] = out["baseline_tvd"] * (
        1 - hmerge["pct_reduction_logical_total_variation_distance"] / 100.0
    )
    for column in totals.columns:
        out[f"fig6_total_{column}"] = totals[column]
    return out.reset_index().rename(columns={"circuit_name": "name"})


def load_fig6_circuits(
    names: Optional[Iterable[str]] = None, *, optimization_level: int = 1
) -> List[BenchCircuit]:
    """
    Exactly the circuits Fig 6 used, with Fig 6's `circuit_idx`, in canonical order.

    Loads through the Fig 6 script's own `_load_group_circuits` (level-1 transpile, 1 s
    timeout, the same gate filters), then retries anything the Fig 6 CSV lists but the
    timeout dropped -- under load a 1 s SIGALRM can lose a large circuit that loaded fine
    the night the figure was made. Refuses to continue if a name is still missing.

    `optimization_level=3` is Experiment1b's pre-optimised input: same files, transpiled at
    level 3 instead of 1 (with a seeded, generous timeout). Never used for Fig 6 arms.
    """
    reference = fig6_reference()
    wanted = set(reference["name"]) if names is None else set(names)
    reference = reference[reference["name"].isin(wanted)]
    if reference.empty:
        raise SystemExit(f"none of {sorted(wanted)[:5]}... are Fig 6 circuits")

    loaded: Dict[str, Tuple[Any, Any]] = {}
    if optimization_level == 1:
        from experiments.figures.merging_improvement.merging_improvement_benchmark_stylized_simulated import (  # noqa: E501
            BenchmarkSettings,
            _load_group_circuits,
        )

        settings = BenchmarkSettings()
        for group in sorted(set(reference["size_group"])):
            group_names = set(reference.loc[reference["size_group"] == group, "name"])
            if names is not None and len(group_names) <= 4:
                continue  # a few named circuits: load by name below, skip the full group
            for name, original, proxy in _load_group_circuits(QASM_ROOT, group, settings):
                if name in group_names:
                    loaded[name] = (original, proxy)

    missing = [n for n in reference["name"] if n not in loaded]
    for group in sorted(set(reference["size_group"])):
        todo = [n for n in missing
                if reference.loc[reference["name"] == n, "size_group"].item() == group]
        if not todo:
            continue
        files = _qasm_by_name(QASM_ROOT / group)
        for name in todo:
            if name not in files:
                raise SystemExit(f"Fig 6 circuit {name} has no QASM file under {group}")
            original = build_circuit(files[name], proxy=False,
                                     optimization_level=optimization_level)
            proxy = build_circuit(files[name], proxy=True,
                                  optimization_level=optimization_level)
            if original is None or proxy is None:
                raise SystemExit(f"could not rebuild Fig 6 circuit {name} "
                                 f"(optimization_level={optimization_level})")
            loaded[name] = (original, proxy)

    out = [
        BenchCircuit(
            size_group=str(row.size_group),
            circuit_idx=int(row.circuit_idx),
            name=str(row.name),
            original=loaded[row.name][0],
            proxy=loaded[row.name][1],
        )
        for row in reference.itertuples(index=False)
    ]
    order = {"small": 0, "medium": 1, "large": 2}
    out.sort(key=lambda c: (order.get(c.size_group, 9), c.circuit_idx))
    return out


def build_circuit(
    path: Path, *, proxy: bool, optimization_level: int = 1, timeout: Optional[float] = None
):
    """
    `create_proxy_from_qasm` with the transpile level exposed.

    Level 1 delegates to the shared loader untouched (no timeout here, so a loaded machine
    cannot drop a circuit). Other levels re-implement the same three steps -- transpile to
    the same basis, drop measurements, optionally map rz/rx to s -- because the shared
    function hardcodes level 1 and its default must not move.
    """
    import qiskit

    from experiments.shared.benchmark_circuits import create_proxy_from_qasm

    if optimization_level == 1:
        return create_proxy_from_qasm(path, timeout_seconds=timeout,
                                      replace_rz_rx_with_s=proxy)
    from experiments.shared.benchmark_circuits import finish_loaded

    try:
        qc = qiskit.QuantumCircuit.from_qasm_file(str(path))
        qc = qiskit.transpile(
            qc, basis_gates=["h", "x", "z", "cx", "cz", "s", "rz", "rx"],
            optimization_level=optimization_level, seed_transpiler=0,
        )
    except Exception:  # noqa: BLE001
        return None
    # The same last steps as `create_proxy_from_qasm`: drop measurements, the logical
    # optimisation pass, then (for a proxy) rotations as S.
    return finish_loaded(qc, proxy)


def load_named_proxies(names: Iterable[str], *, proxy: bool = True) -> Dict[str, Any]:
    """
    QASMBench circuits by name from any size group, built like `load_proxy_circuits`
    (level-1 transpile) but without its 1 s SIGALRM timeout, which can drop a large
    circuit on a loaded machine. Raises if a name is missing or will not build.
    """
    files: Dict[str, Path] = {}
    for group in ("small", "medium", "large"):
        for name, path in _qasm_by_name(QASM_ROOT / group).items():
            files.setdefault(name, path)
    out: Dict[str, Any] = {}
    for name in names:
        if name not in files:
            raise SystemExit(f"no QASMBench file for {name}")
        circuit = build_circuit(files[name], proxy=proxy)
        if circuit is None:
            raise SystemExit(f"{name} could not be built (proxy={proxy})")
        out[name] = circuit
    return out


def load_small_qasmbench(
    names: Sequence[str] = SMALL_QASMBENCH, *, proxy: bool
) -> Dict[str, Any]:
    """The named 4-6 qubit circuits, as originals or as Clifford proxies."""
    files = _qasm_by_name(QASM_ROOT / "small")
    out: Dict[str, Any] = {}
    for name in names:
        if name not in files:
            log("load", f"no QASM file for {name}; skipped")
            continue
        circuit = build_circuit(files[name], proxy=proxy)
        if circuit is None:
            log("load", f"{name} could not be built (proxy={proxy}); skipped")
            continue
        out[name] = circuit
    return out


def count_rotations(circuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.name in {"rz", "rx"})


def non_clifford_rotations(circuit, tol: float = 1e-9) -> int:
    """Rz/Rx whose angle is not a multiple of pi/2 -- the ones a proxy actually changes."""
    count = 0
    for inst in circuit.data:
        if inst.operation.name not in {"rz", "rx"}:
            continue
        angle = float(inst.operation.params[0])
        quarter = angle / (math.pi / 2)
        if abs(quarter - round(quarter)) > tol:
            count += 1
    return count


# ----------------------------------------------------------------------------------------
# translation
# ----------------------------------------------------------------------------------------


def translator_config(n_code: int, *, commute: bool, merging: bool = True):
    from pipeline.config import PipelineConfig

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    return config


def translate_metrics(
    program, placement: Perm, n_code: int, *, commute: bool, merging: bool = True,
    noise_split: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Translate and count. `None` for a rejected placement.

    `work` is weighted gates (the archived exhaustive script's objective); `work_depth` adds
    `DEFAULT_DEPTH_WEIGHT` per layer, which is exactly what `plan.measured_cost` charges when
    no noise profile is given -- the gate-counting planner's own verify objective.

    `noise_split=True` adds `simulator.attribution.measured_noise_split` at the NEUTRAL_ATOM
    rates (alpha = 1), prefixed `E_`: expected detectable errors by source. `E_total` is
    exactly what `plan.measured_cost` returns under a noise profile -- the criterion
    `verify` uses in every simulated figure.
    """
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from iceberg.costs import DEFAULT_WORK_WEIGHTS
    from packing.cost import DEFAULT_DEPTH_WEIGHT
    from pipeline.tasks.translate_permutation import translate_permutation

    config = translator_config(n_code, commute=commute, merging=merging)
    try:
        translated = translate_permutation(program, [tuple(p) for p in placement],
                                           config=config)
    except Exception:  # noqa: BLE001 - a rejected placement is a normal outcome
        return None
    circuit = translated.circuit
    one, two = one_two_qubit_gate_totals(circuit)
    depth = circuit.depth()
    w1, w2 = DEFAULT_WORK_WEIGHTS
    out = {
        "one_qubit_gates": float(one),
        "two_qubit_gates": float(two),
        "depth": float(depth),
        "work": float(w1 * one + w2 * two),
        "work_depth": float(w1 * one + w2 * two + DEFAULT_DEPTH_WEIGHT * depth),
        "merge_savings_1q": float(translated.merge_savings_1q),
        "merge_savings_2q": float(translated.merge_savings_2q),
        "merge_savings_weighted": float(translated.merge_savings_weighted),
        "patches": float(len({p for p, _ in placement})),
        "physical_qubits": float(circuit.num_qubits),
    }
    if noise_split:
        from simulator.attribution import measured_noise_split
        from simulator.noise_models import StimNoiseParameters

        try:
            split = measured_noise_split(circuit, StimNoiseParameters.neutral_atom(),
                                         idle_time_scalar=1.0, distance_scalar=1.0)
            for key, value in split.items():
                out[f"E_{key}"] = float(value)
        except Exception:  # noqa: BLE001
            pass
    return out


def fig6_noise_profile():
    """The `NoiseProfile` Fig 6 hands the planner (Bug B6): distance + idle on, scalars 1."""
    from experiments.shared.simulation import SimSettings, noise_profile_for

    return noise_profile_for(SimSettings(
        distance_noise=True, distance_scalar=1.0, idle_noise=True, idle_time_scalar=1.0,
        stim=True, shot_progress=False,
    ))


def placement_key(placement: Perm) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(p), int(s)) for p, s in placement)


def partition_of(placement: Perm) -> Tuple[Tuple[int, ...], ...]:
    """The partition a placement induces, canonically ordered. Lanes dropped."""
    groups: Dict[int, List[int]] = {}
    for qubit, (patch, _) in enumerate(placement):
        groups.setdefault(int(patch), []).append(qubit)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


# ----------------------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseSetup:
    """What the simulator charges. Defaults are Fig 6's: Stim, distance + idle at 1."""

    alpha: float
    stim: bool = True
    distance_noise: bool = True
    distance_scalar: float = 1.0
    idle_noise: bool = True
    idle_scalar: float = 1.0
    #: Aer only: passed to `SimulatorConfig.aer_method` (opt-in, see simulator/config.py).
    aer_method: Optional[str] = None


def sim_config(noise: NoiseSetup, *, n_code: int, commute: bool, merging: bool,
               repetitions: int, seed: Optional[int]):
    """
    The same `PipelineConfig` Fig 6's `build_pipeline_config` builds, plus `n_code`.

    Shot progress is off -- it changes no sample, only how fast workers write to the pipe.
    """
    from pipeline.config import PipelineConfig
    from simulator.config_presets import ScaledSimulator

    sim = ScaledSimulator(alpha=noise.alpha, repetitions=int(repetitions), seed=seed)
    sim.show_shot_progress = False
    sim = sim.with_stim(noise.stim)
    if noise.distance_noise:
        sim = sim.with_distance_noise(distance_scalar=noise.distance_scalar)
    if noise.idle_noise:
        sim = sim.with_idle_noise(idle_time_scalar=noise.idle_scalar)
    if noise.aer_method is not None:
        sim.aer_method = noise.aer_method
    config = PipelineConfig(verbose=False, save=False, sim_config=sim, seed=seed)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    return config


def simulate_arm(program, placement: Perm, config) -> Dict[str, Any]:
    """
    `run_specific_scan` for one placement, keeping the distributions.

    Same calls in the same order -- `Translator.translate_batch`, `simulate_batch`,
    `postselect_against_reference` -- so with the same seed this returns the same samples
    Fig 6 drew. Skipped: `result_to_row` and `ScanResult.to_dict`, which serialise the
    physical circuit to QASM 3 and contribute nothing measured.

    The ideal reference is sampled with the same raw shot count as the noisy run, which
    is the pipeline's convention and what every figure's TVD was measured against.
    """
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from analysis.postselection import postselect_against_reference
    from experiments.shared.simulation import ensure_measurements
    from simulator.simulator import simulate_batch
    from translator.core import Translator

    program = ensure_measurements(program)
    translated = Translator(config=config.translator_config).translate_batch(
        program, [[tuple(p) for p in placement]]
    )
    ideal, noisy = simulate_batch(program, translated, config.sim_config)
    result = postselect_against_reference(translated, ideal, noisy)[0]
    circuit = translated[0].circuit
    one, two = one_two_qubit_gate_totals(circuit)
    return {
        "lsr": float(result.logical_selection_rate),
        "tvd": float(result.logical_total_variation_distance),
        "noisy": dict(result.postselected_noisy_dist),
        "ideal": dict(result.postselected_ideal_dist),
        "raw_shots": int(config.sim_config.repetitions),
        "one_qubit_gates": float(one),
        "two_qubit_gates": float(two),
        "depth": float(circuit.depth()),
        "physical_qubits": int(circuit.num_qubits),
    }


def exact_ideal_distribution(program) -> Dict[str, float]:
    """
    The noiseless logical distribution, exactly, keyed the way post-selection keys it
    (character i = program qubit i). Statevector, so keep it under ~20 qubits.
    """
    from qiskit.quantum_info import Statevector

    unitary_part = program.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_instruction(unitary_part).probabilities_dict()
    return {key[::-1]: float(p) for key, p in probabilities.items() if p > 1e-14}


# ----------------------------------------------------------------------------------------
# Experiment4 statistics
# ----------------------------------------------------------------------------------------


def _aligned(noisy: Dict[str, float], ideal: Dict[str, float]):
    keys = sorted(set(noisy) | set(ideal))
    p = np.array([noisy.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([ideal.get(k, 0.0) for k in keys], dtype=float)
    return keys, p, q


def _normalised(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    return values / total if total > 0 else values


def counts_from(dist: np.ndarray, shots: int) -> np.ndarray:
    """Probabilities back to integer counts. Exact for sampled distributions."""
    return np.rint(dist * shots).astype(np.int64)


def tvd_statistics(
    noisy: Dict[str, float],
    ideal: Dict[str, float],
    *,
    accepted: int,
    ideal_shots: Optional[int],
    rng: np.random.Generator,
    bootstrap: int = 200,
    floor_draws: int = 20,
) -> Dict[str, float]:
    """
    Uncertainty on one TVD measurement, and the floor it cannot go below.

    `ideal_shots=None` means `ideal` is exact (no sampling noise on that side).

    * `tvd_ci_lo/hi`: percentile bootstrap, resampling the accepted noisy counts *and* the
      sampled ideal reference -- both sides of the TVD are finite samples.
    * `tvd_floor`: the noiseless finite-sample floor at this arm's accepted count -- two
      samples of the ideal distribution (plug-in: the measured ideal stands in for the
      truth) at `accepted` and `ideal_shots` draws. A perfect compiler scores this.
    """
    out = {"accepted_shots": int(accepted)}
    if accepted <= 0 or not noisy or not ideal:
        out.update(tvd_ci_lo=math.nan, tvd_ci_hi=math.nan, tvd_boot_sd=math.nan,
                   tvd_floor=math.nan, tvd_floor_sd=math.nan, support_ideal=len(ideal))
        return out
    _, p, q = _aligned(noisy, ideal)
    p, q = _normalised(p), _normalised(q)
    boots = np.empty(bootstrap)
    for b in range(bootstrap):
        pn = rng.multinomial(accepted, p) / accepted
        qn = q if ideal_shots is None else rng.multinomial(ideal_shots, q) / ideal_shots
        boots[b] = 0.5 * np.abs(pn - qn).sum()
    floors = np.empty(floor_draws)
    for d in range(floor_draws):
        a = rng.multinomial(accepted, q) / accepted
        b_ = q if ideal_shots is None else rng.multinomial(ideal_shots, q) / ideal_shots
        floors[d] = 0.5 * np.abs(a - b_).sum()
    out.update(
        tvd_ci_lo=float(np.percentile(boots, 2.5)),
        tvd_ci_hi=float(np.percentile(boots, 97.5)),
        tvd_boot_sd=float(boots.std(ddof=1)),
        tvd_floor=float(floors.mean()),
        tvd_floor_sd=float(floors.std(ddof=1)) if floor_draws > 1 else 0.0,
        support_ideal=int((q > 0).sum()),
        support_noisy=int((p > 0).sum()),
    )
    return out


def equal_shot_tvd(
    noisy: Dict[str, float],
    ideal: Dict[str, float],
    *,
    accepted: int,
    target: int,
    rng: np.random.Generator,
    draws: int = 20,
) -> Tuple[float, float]:
    """
    TVD after downsampling this arm's accepted shots to `target`, without replacement
    (multivariate hypergeometric on the counts), averaged over `draws`. Returns
    (mean, sd). With `target >= accepted` it is just the TVD.
    """
    if accepted <= 0 or target <= 0 or not noisy or not ideal:
        return math.nan, math.nan
    _, p, q = _aligned(noisy, ideal)
    q = _normalised(q)
    if target >= accepted:
        return float(0.5 * np.abs(_normalised(p) - q).sum()), 0.0
    counts = counts_from(_normalised(p), accepted)
    values = np.empty(draws)
    for d in range(draws):
        sub = rng.multivariate_hypergeometric(counts, target, method="marginals")
        values[d] = 0.5 * np.abs(sub / target - q).sum()
    return float(values.mean()), float(values.std(ddof=1)) if draws > 1 else 0.0


def histogram_json(dist: Dict[str, float], shots: int) -> str:
    """Counts, not probabilities, so a reader can re-derive any statistic."""
    return json.dumps({k: int(round(v * shots)) for k, v in sorted(dist.items()) if v > 0})


# ----------------------------------------------------------------------------------------
# small CLI helpers
# ----------------------------------------------------------------------------------------


def add_common_arguments(parser, *, default_out: Path) -> None:
    parser.add_argument("--jobs", type=int, default=default_jobs())
    parser.add_argument("--out", type=Path, default=default_out,
                        help="Output directory. A finished table there is resumed only "
                             "if its settings fingerprint matches.")
    parser.add_argument("--smoke", action="store_true",
                        help="A few circuits, tiny shot counts, every code path. Writes "
                             "under results/revision/_smoke/ unless --out is given.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")


def out_dir(args, name: str) -> Path:
    if args.smoke and str(args.out).startswith(str(RESULTS)) and "_smoke" not in str(args.out):
        return SMOKE / name
    return Path(args.out)


def percent_change(reference: float, value: float) -> float:
    """Signed % change of `value` against `reference`; NaN if the reference is ~0."""
    if reference is None or value is None:
        return math.nan
    if not (math.isfinite(reference) and math.isfinite(value)) or abs(reference) < 1e-12:
        return math.nan
    return 100.0 * (value - reference) / reference
