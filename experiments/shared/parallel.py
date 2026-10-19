"""
parallel.py -- fan a figure's work items across cores.

Every simulated figure in this directory was single-threaded, and simulation is the
schedule risk: per circuit the harness runs two 4,000-shot calibration pilots and then
20,000-400,000 shots per evaluation. A sweep over (circuit, k, ratio) is embarrassingly
parallel -- the points share nothing -- so the only reason they ran in sequence is that
nobody had written this file.

**Why `spawn` rather than `fork`.** macOS defaults to spawn for good reason: Qiskit and
Stim both hold state that does not survive a fork cleanly, and a forked child that
deadlocks inside a C extension produces a run that hangs with no traceback. Spawn costs
a fresh interpreter and a Qiskit import per worker (~2-3 s), paid once per worker rather
than once per item, which is negligible against a sweep measured in hours.

**Work functions must be importable at module scope.** Spawn re-imports the module in the
child, so a closure or a lambda cannot cross the boundary. That is also why every caller
needs its `if __name__ == "__main__":` guard: without one, the child re-runs the sweep on
import and the pool forks bombs. `python -m package.module` is handled correctly by
multiprocessing's `__mp_main__` shim, which is how these scripts are invoked.

**Failures are values, not exceptions.** A rejected placement is a normal outcome in this
codebase -- `evaluate` already returns None for one -- and a sweep that dies on item 400
of 600 loses the 399 results that worked. `parallel_map` catches per item, logs, and
returns None in that slot, so a caller filters rather than restarts.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

__all__ = [
    "DEFAULT_JOBS",
    "add_jobs_argument",
    "resolve_jobs",
    "parallel_map",
]

T = TypeVar("T")
R = TypeVar("R")

#: Leave two cores for the machine. These runs go overnight on a workstation that is also
#: running other sweeps, and saturating every core makes the whole box unresponsive
#: without making any single sweep finish meaningfully sooner.
_RESERVED_CORES = 2


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


DEFAULT_JOBS = max(1, _cpu_count() - _RESERVED_CORES)


def add_jobs_argument(parser, default: Optional[int] = None) -> None:
    """The `--jobs` flag, so every parallel figure spells it the same way."""
    parser.add_argument(
        "--jobs",
        type=int,
        default=default if default is not None else DEFAULT_JOBS,
        help=(
            "Worker processes for the sweep. 1 runs in-process, which is what to use "
            "when debugging since a traceback then points at the real frame. Default "
            f"is cores minus {_RESERVED_CORES} (here {DEFAULT_JOBS})."
        ),
    )


def resolve_jobs(requested: Optional[int], item_count: int) -> int:
    """Never start more workers than there is work, or than the machine has cores."""
    jobs = DEFAULT_JOBS if requested is None else requested
    return max(1, min(jobs, item_count, _cpu_count()))


def parallel_map(
    function: Callable[[T], R],
    items: Sequence[T],
    *,
    jobs: Optional[int] = None,
    label: str = "work",
    quiet: bool = False,
) -> List[Optional[R]]:
    """
    Apply `function` to every item, in parallel, preserving input order.

    Returns a list the same length as `items`, with None wherever the call raised. The
    caller filters; see the module docstring on why a failure is a value here.

    `jobs=1` runs in-process rather than starting a pool of one, so a debugging run gets
    a real traceback instead of a re-raised copy from a worker.
    """
    total = len(items)
    if total == 0:
        return []

    worker_count = resolve_jobs(jobs, total)
    results: List[Optional[R]] = [None] * total
    started = time.monotonic()
    done = 0

    def progress() -> None:
        if quiet:
            return
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        print(
            f"  [{label}] {done}/{total}  "
            f"{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left",
            file=sys.stderr,
            flush=True,
        )

    if worker_count == 1:
        for index, item in enumerate(items):
            try:
                results[index] = function(item)
            except Exception as error:  # noqa: BLE001 - one bad item must not end the sweep
                print(f"  [{label}] item {index} failed: {error!r}", file=sys.stderr)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                progress()
        return results

    if not quiet:
        print(
            f"  [{label}] {total} items across {worker_count} workers",
            file=sys.stderr,
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(function, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"  [{label}] item {index} failed: {error!r}", file=sys.stderr)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                progress()

    return results


def chunked(items: Iterable[T], size: int) -> List[List[T]]:
    """Group items so a worker amortises its interpreter startup over several of them."""
    batch: List[T] = []
    out: List[List[T]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            out.append(batch)
            batch = []
    if batch:
        out.append(batch)
    return out
