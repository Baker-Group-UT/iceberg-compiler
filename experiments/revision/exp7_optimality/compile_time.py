"""
compile_time.py -- Experiment7c: how long AQORN takes to compile, against circuit size.

Reports compilation time across circuit sizes, alongside fidelity and resource
metrics. Section V-C says every benchmark compiles in seconds; this measures it.

Two stages are timed, separately:

    plan_packing      `plan_packing(n_code=4, h_commutation="auto", verify=True)` with Fig 6's
                      noise profile -- exactly the call behind Fig 6's packing row
    compile_circuit   the full compiler: resource selection over code sizes 4-12 and patch
                      counts, then packing, then the three-arm comparison

Single process, nothing else running (the queue runs this before any simulation), wall
clock via `perf_counter`, `repeats` runs, median reported -- except that a run longer than
`long_run_seconds` is not repeated (recorded in `repeats`): the Fig 6 call takes ~3 minutes
on `bv_n140`, and three of those per large circuit would cost an hour to tighten a number
whose run-to-run spread is a few percent. Every module-level cache that
could make a repeat cheaper than the first run is cleared before each run: the
commutation decision cache (`packing.passes`), the geometry caches (`packing.tables`) and
the selector's intra-patch cache (`packing.resource.noise`). Exact translations are
counted on the first run (by wrapping `Translator.translate`, ~1 us per call), and the
descent's evaluation count is read off its result.

Circuits: the 71 Fig 6 originals (with measurements, as Fig 6 compiles them) and random
structured programs at 10, 20, 40, 80 and 150 qubits, three seeds each.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import logging
import math
import statistics
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

from experiments.revision.common import (
    RESULTS,
    ResultTable,
    add_common_arguments,
    fig6_noise_profile,
    load_fig6_circuits,
    log,
    out_dir,
)

NAME = "exp7c"


@dataclass
class Settings:
    repeats: int = 3
    long_run_seconds: float = 30.0
    random_sizes: Sequence[int] = (10, 20, 40, 80, 150)
    random_seeds: int = 3
    random_seed_base: int = 20261001
    gates_per_qubit: int = 10
    fig6: bool = True
    stages: Sequence[str] = ("plan_packing", "compile_circuit")
    smoke: bool = False
    out: str = str(RESULTS / "exp7")


def clear_caches() -> None:
    import packing.passes as passes
    import packing.resource.noise as rnoise
    import packing.tables as tables

    passes.clear_cache()
    tables._MEAN_HOPS.clear()
    tables._HOP_MATRIX.clear()
    rnoise._INTRA.clear()


class TranslationCounter:
    """Counts `Translator.translate` calls inside the block. Restores the original."""

    def __enter__(self):
        from translator.core import Translator

        self.calls = 0
        self._original = Translator.translate
        counter = self

        def counted(this, *args, **kwargs):
            counter.calls += 1
            return counter._original(this, *args, **kwargs)

        Translator.translate = counted
        return self

    def __exit__(self, *exc):
        from translator.core import Translator

        Translator.translate = self._original
        return False


def run_stage(stage: str, program, profile):
    from packing import compile_circuit
    from packing.plan import plan_packing

    if stage == "plan_packing":
        return plan_packing(program, 4, h_commutation="auto", verify=True, noise=profile)
    if stage == "compile_circuit":
        return compile_circuit(program)
    raise ValueError(stage)


def circuits(settings: Settings) -> List[Tuple[str, str, object]]:
    from experiments.shared.random_circuits import plant_blocks, structured_program_circuit
    from experiments.shared.simulation import ensure_measurements

    out: List[Tuple[str, str, object]] = []
    if settings.fig6:
        loaded = load_fig6_circuits()
        if settings.smoke:
            loaded = [c for c in loaded if c.name in {"adder_n10", "knn_n25", "bv_n140"}]
        for c in loaded:
            out.append((c.name, "fig6", ensure_measurements(c.original)))
    sizes = settings.random_sizes if not settings.smoke else (10, 40)
    seeds = settings.random_seeds if not settings.smoke else 1
    for size in sizes:
        for index in range(seeds):
            seed = settings.random_seed_base + 1000 * size + index
            blocks = plant_blocks(size, 2, seed=seed)
            circuit = structured_program_circuit(
                size, settings.gates_per_qubit, seed=seed, blocks=blocks, locality=0.85,
                immutable_ratio=0.0, clifford_only=True,
            )
            out.append((f"random_n{size}_s{index}", "random", ensure_measurements(circuit)))
    return out


def run(settings: Settings, resume: bool = True) -> Path:
    out = Path(settings.out)
    table = ResultTable(out / "compile_time.csv", key_columns=["name", "stage"],
                        experiment=NAME, settings=asdict(settings), resume=resume)
    profile = fig6_noise_profile()
    work = circuits(settings)
    log(NAME, f"{len(work)} circuits x {list(settings.stages)}, {settings.repeats} repeats, "
              f"single process")
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = "?"
    for position, (name, source, program) in enumerate(work, start=1):
        for stage in settings.stages:
            if table.is_done((name, stage)):
                continue
            times: List[float] = []
            failure = ""
            result = None
            translations = math.nan
            evaluations = math.nan
            for repeat in range(settings.repeats):
                clear_caches()
                try:
                    # Counting wraps every run: one extra Python call per translation, ~1 us
                    # against milliseconds each, so it costs nothing measurable and saves a
                    # separate counting run on circuits that take minutes.
                    with TranslationCounter() as counter:
                        began = time.perf_counter()
                        result = run_stage(stage, program, profile)
                        times.append(time.perf_counter() - began)
                except Exception as error:  # noqa: BLE001
                    failure = repr(error)[:300]
                    break
                if repeat == 0:
                    translations = counter.calls
                    plan = result if stage == "plan_packing" else result.plan
                    evaluations = plan.descent.evaluations if plan.descent else 0
                if times[-1] > settings.long_run_seconds:
                    break
            row: Dict = {
                "name": name, "source": source, "num_qubits": program.num_qubits,
                "program_gates": program.size(), "stage": stage,
                "seconds_median": statistics.median(times) if times else math.nan,
                "seconds_min": min(times) if times else math.nan,
                "seconds_max": max(times) if times else math.nan,
                "repeats": len(times), "translations": translations,
                "descent_evaluations": evaluations, "failure": failure, "cpu": cpu,
            }
            if result is not None and not failure:
                if stage == "compile_circuit":
                    row.update(chosen_n_code=result.n_code, chosen_patches=result.patches,
                               chosen_source=result.source)
                else:
                    row.update(chosen_n_code=4, chosen_patches=result.patches,
                               fell_back=bool(result.fell_back))
            table.append([row])
            log(NAME, f"[{position}/{len(work)}] {name:>22} {stage:<15} "
                      f"{row['seconds_median']:8.2f} s  ({translations} translations)")
    table.write_sidecar("complete", extra={"cpu": cpu})
    frame = table.frame()
    plot(frame, out)
    return table.path


def plot(frame: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    if frame.empty:
        return
    plt.rcdefaults()
    use_style()
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    labels = {"plan_packing": "packing ($k=2$)", "compile_circuit": "full compiler"}
    for stage, label in labels.items():
        block = frame[(frame["stage"] == stage) & frame["seconds_median"].notna()]
        if block.empty:
            continue
        axis.plot(block["num_qubits"], block["seconds_median"], linestyle="none",
                  label=label, markersize=3.5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("program qubits")
    axis.set_ylabel("compile time (s)")
    axis.legend(frameon=False, loc="upper left")
    save_figure(fig, "compile_time", Path(__file__), override=out, versioned=True)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_common_arguments(parser, default_out=RESULTS / "exp7")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stages", nargs="+", default=["plan_packing", "compile_circuit"])
    args = parser.parse_args()
    settings = Settings(repeats=args.repeats if not args.smoke else 1,
                        stages=tuple(args.stages), smoke=args.smoke,
                        out=str(out_dir(args, "exp7")))
    return settings, args.resume


if __name__ == "__main__":
    run(*parse_args())
