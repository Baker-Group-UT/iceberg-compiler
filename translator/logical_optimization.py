"""
logical_optimization.py -- the compiler's first pass: one verified round of logical optimisation
of the program, before any other pass, packing decision or translation sees it (2026-10-09).

    optimized = logically_optimized(program)

What a round is. `translator.logical_optimizer.optimize_verified` on the program's unitary body:
Qiskit O3 run to a fixed point plus structurally different routes (whole-circuit and block
Clifford resynthesis, CNOT-block resynthesis, commutation), the best by (two-qubit gates, total
gates, depth), never worse than the input, and kept only if it is the same unitary up to global
phase (exact operator up to 12 qubits, tableau for Clifford circuits, a three-state miter
otherwise). It is the optimiser of the logical-optimisation study (reports/logical_opt.md),
unchanged. Final measurements are split off first and put back on the same classical bits; the
qubit order is never permuted; the output uses the translator's gates (h x z s sx rz rx cx cz).

Then every rz / rx at a multiple of pi/2 is written as the translator's named Clifford gates
(`translator_cliffords`: s, z, s+z, sx, x, sx+x). The optimiser works in the corpus basis, so it
emits Cliffords as rotations -- 1841 of the 8684 rotations in the study's optimised Fig 6 circuits
sat at Clifford angles -- and the translator builds a rotation as the RZ gadget, which costs more
two-qubit gates and leaks about twice the silent error of the S gadget (`iceberg.costs`,
`packing/data/gadget_rates.csv`). An exact rewrite (up to global phase), applied to the input too
when the optimiser finds nothing else.

No new rotations (2026-10-10). No program comes back with more non-Clifford rotations than it
had (`_adds_rotations`); for a Clifford program, none at all. The optimiser's objective puts
two-qubit gates first, so it would trade them for rotations: O3's two-qubit block resynthesis
writes a Clifford unitary with arbitrary-angle rotations (3 of 60 random Clifford programs),
and 7 of the study's 56 optimised Fig 6 circuits carry more rotations than their originals
(qugan_n71 519 -> 635). Every rotation is a rotation gadget to the translator, which AQORN
cannot merge or commute away, and the Clifford proxy maps each to S, which spread four of those
proxies' outputs over almost every bitstring. So the search itself never steps to a circuit
with more (`logical_optimizer.optimize(max_rotations=)`): a chain that would trade two-qubit
gates for rotations stops there, and the routes that add none -- commutation, CNOT-block
resynthesis -- keep going. An answer that still adds some (a record from before the rule) gives
way to the best other candidate that adds none, verified, or to the program as it is.

Where it runs. On by default and process-wide (`enabled()`), in the places every program passes
through, so no caller can skip it and the planner and the translator never see two programs:

    translator.core.Translator.translate   first, before `cancel_inverse_gates`, the first of
                                           the translator's own program-level passes
    packing.tables.translated_view         the planner's view of the program, which by contract
                                           mirrors those passes; every cost model, interaction
                                           graph and partitioner reads the program through it
    the QASMBench loaders                  `experiments.shared.benchmark_circuits.
                                           create_proxy_from_qasm` and `experiments.revision.
                                           common.build_circuit`: the circuit is optimised
                                           *before* its Clifford proxy is made, so the proxy has
                                           the optimised program's gates (rotations as S), and
                                           the proxy is registered as already optimised
                                           (`register_optimized`), so the translator never
                                           re-optimises it into a structure the program lacks

Turning it off. `AQORN_LOGICAL_OPT=0` in the environment, or `set_enabled(False)` / `with
disabled():` in code. The switch lives in `os.environ` so that worker processes started while
it is set inherit it (spawn workers re-import every module and would never see a module
global). Off reproduces everything made before 2026-10-09, the submitted figures included. The
test suite runs with it off (conftest.py), since it pins that behaviour; the pass's own tests
turn it on.

Why a cache. O3 is not deterministic in Qiskit 2.2.1, even at a fixed seed (see
`logical_optimizer.O3_TRIALS`), so two calls on one program can return different circuits.
Two callers optimising the same program -- the planner and the translator, or two workers --
would then plan one circuit and run another. So the first result for each program is recorded,
content-addressed, and every later call in any process or worktree gets exactly that circuit:

    <cache>/<key>.qpy     the record for the program whose body fingerprint is <key>: its
                          optimised body, or empty when the optimiser could not improve it
    <cache>/<key>.fixed   this body is already optimised (an output of the pass, or a proxy made
                          from one): use it as it is. Takes precedence over a `.qpy` record

Records are first-writer-wins (a hard link of a finished temporary file), so concurrent workers
agree. <cache> is `results/cache/logical_optimization/` in the main checkout, shared by every
worktree, or `$AQORN_LOGICAL_OPT_CACHE`. Deleting it re-optimises everything and O3 may then
choose differently, so keep it with the results it produced. `OPTIMIZER_VERSION` is part of
every key: change it whenever the optimiser or this module changes what a round returns.

Time budget. A program not yet on record is optimised in a child process with a wall-clock
budget (`AQORN_LOGICAL_OPT_TIMEOUT`, default 120 s; 0 runs it in-process with no limit). The
equivalence check simulates wide non-Clifford circuits as matrix-product states, which can blow
up and which no Python-level timeout can interrupt, so the child is killed instead and the
program is recorded as unchanged. The cost of a cache miss is that child's start-up (~2 s).

Passed through unchanged: programs with a measurement that is not final, a reset, classical
control or an instruction the optimiser cannot take (`_body_and_tail` returns None), programs
the optimiser cannot improve (the caller's object comes back), and programs whose round ran out
of budget.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

__all__ = [
    "ENV",
    "CACHE_ENV",
    "TIMEOUT_ENV",
    "OPTIMIZER_VERSION",
    "enabled",
    "set_enabled",
    "switched",
    "disabled",
    "state",
    "logically_optimized",
    "register_optimized",
    "seed_record",
    "drop_record",
    "fingerprint",
    "translator_cliffords",
    "cache_dir",
    "clear_memory",
]

#: The switch. Anything but 0 / false / off / no (case-insensitive) means on; unset means on.
ENV = "AQORN_LOGICAL_OPT"
#: Overrides where the records live (see the module docstring).
CACHE_ENV = "AQORN_LOGICAL_OPT_CACHE"
#: Wall-clock budget in seconds for optimising one program that is not on record. The round
#: runs in a child process, killed when the budget runs out -- the equivalence check simulates
#: wide circuits as matrix-product states, which no Python-level timeout can interrupt -- and a
#: program that ran out is recorded as unchanged, so the answer stays fixed afterwards. 0 runs
#: the round in this process with no limit.
TIMEOUT_ENV = "AQORN_LOGICAL_OPT_TIMEOUT"
DEFAULT_TIMEOUT = 120.0
#: Part of every cache key and of every provenance stamp. Change it whenever the optimiser
#: (`logical_optimizer`) or this module changes what a round returns.
OPTIMIZER_VERSION = "verified-routes-2"
#: Key under which an optimised circuit's `metadata` records what produced it.
METADATA_KEY = "logical_optimization"

_OFF = {"0", "false", "off", "no"}
#: Instructions the body cannot carry: the optimiser and its equivalence check are unitary.
_UNSUPPORTED = {"reset", "initialize", "delay", "store", "set_state"}


# ---------------------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------------------


def enabled() -> bool:
    """Whether the pass runs in this process (and in workers it starts from now on)."""
    return os.environ.get(ENV, "1").strip().lower() not in _OFF


def set_enabled(flag: bool) -> None:
    """Turn the pass on or off for this process and every worker it starts afterwards."""
    os.environ[ENV] = "1" if flag else "0"


@contextlib.contextmanager
def switched(flag: bool) -> Iterator[None]:
    """The pass on (or off) for the duration, restoring the previous setting after."""
    previous = os.environ.get(ENV)
    set_enabled(flag)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous


def disabled() -> contextlib.AbstractContextManager:
    """`with disabled():` -- the program exactly as given, as before 2026-10-09."""
    return switched(False)


def state() -> str:
    """What to record next to a result: the optimiser version, or "off"."""
    return OPTIMIZER_VERSION if enabled() else "off"


# ---------------------------------------------------------------------------------------
# fingerprints and the body / measurement split
# ---------------------------------------------------------------------------------------


def _param(value) -> str:
    """A parameter, exactly: floats by their hex form so no digit is lost to formatting."""
    try:
        return float(value).hex()
    except (TypeError, ValueError):
        return repr(value)


def fingerprint(body: QuantumCircuit) -> str:
    """
    The record key of a measurement-free body: sha256 over the optimiser version, the width
    and every instruction's name, qubit indices and exact parameters, in order.
    """
    ops = tuple(
        (inst.operation.name,
         tuple(body.find_bit(q).index for q in inst.qubits),
         tuple(_param(p) for p in inst.operation.params))
        for inst in body.data
    )
    payload = repr((OPTIMIZER_VERSION, body.num_qubits, ops)).encode()
    return hashlib.sha256(payload).hexdigest()[:40]


def _body_and_tail(circuit: QuantumCircuit) -> Optional[Tuple[QuantumCircuit, List[Tuple[int, int]]]]:
    """
    (unitary body on a plain register, final measurements as (qubit, clbit) indices), or None
    when the circuit is not a unitary followed by measurements.

    A barrier is kept in the body unless it touches an already measured qubit or ends the body
    (then it is part of the measurement block -- `measure_all` puts one there -- and dropped, so
    a program and its measured copy share one record); any other instruction after a
    measurement on one of its qubits makes the measurement mid-circuit, and the circuit is
    left alone.
    """
    from qiskit.circuit import ControlFlowOp

    measured: set = set()
    ops: List[Tuple[object, Tuple[int, ...]]] = []
    tail: List[Tuple[int, int]] = []
    for inst in circuit.data:
        op = inst.operation
        name = op.name
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if isinstance(op, ControlFlowOp) or name in _UNSUPPORTED \
                or getattr(op, "condition", None) is not None:
            return None
        if name == "measure":
            tail.append((qubits[0], circuit.find_bit(inst.clbits[0]).index))
            measured.add(qubits[0])
            continue
        if measured.intersection(qubits):
            if name == "barrier":
                continue
            return None
        if inst.clbits:
            return None
        ops.append((op, qubits))
    while ops and ops[-1][0].name == "barrier":
        ops.pop()
    body = QuantumCircuit(circuit.num_qubits)
    for op, qubits in ops:
        body.append(op, qubits)
    return body, tail


# ---------------------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------------------

#: In-process view of the records: key -> optimised body, or _UNCHANGED / _FIXED.
_MEMORY: Dict[str, Union[QuantumCircuit, str]] = {}
_UNCHANGED = "unchanged"
_FIXED = "fixed"
_GIT_ROOT: List[Optional[Path]] = []
_WARNED: set = set()


def clear_memory() -> None:
    """Forget the in-process view (the records on disk stay). For tests."""
    _MEMORY.clear()


def _main_checkout() -> Path:
    """The main checkout of this repository (worktrees share its records), or this one."""
    if not _GIT_ROOT:
        here = Path(__file__).resolve().parents[1]
        try:
            common = subprocess.run(
                ["git", "-C", str(here), "rev-parse", "--path-format=absolute",
                 "--git-common-dir"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            _GIT_ROOT.append(Path(common).parent if common else here)
        except Exception:  # noqa: BLE001 - not a git checkout: keep the records beside the code
            _GIT_ROOT.append(here)
    return _GIT_ROOT[0]


def cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    return _main_checkout() / "results" / "cache" / "logical_optimization"


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        logger.warning(message)


def _create_exclusive(path: Path, data: bytes) -> bool:
    """Write `path` only if nobody has; True if this call wrote it. Never a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        os.unlink(temporary)


def _dump(circuit: QuantumCircuit) -> bytes:
    from qiskit import qpy

    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


def _load(path: Path) -> QuantumCircuit:
    from qiskit import qpy

    with path.open("rb") as stream:
        return qpy.load(stream)[0]


def _read_record(key: str) -> Optional[Union[QuantumCircuit, str]]:
    """What is on record for `key` (memory first, then disk), or None."""
    if key in _MEMORY:
        return _MEMORY[key]
    try:
        folder = cache_dir()
        if (folder / f"{key}.fixed").exists():
            _MEMORY[key] = _FIXED
        else:
            record = folder / f"{key}.qpy"
            if record.exists():
                _MEMORY[key] = _UNCHANGED if record.stat().st_size == 0 else _load(record)
    except OSError as error:
        _warn_once(f"logical optimisation records unreadable ({error}); in-memory only")
    return _MEMORY.get(key)


def _write_record(key: str, body: Optional[QuantumCircuit]) -> Union[QuantumCircuit, str]:
    """
    Record `body` (None: unchanged) for `key` unless a record exists, and return whatever is
    on record afterwards -- another process's, if it got there first.
    """
    data = b"" if body is None else _dump(body)
    try:
        path = cache_dir() / f"{key}.qpy"
        if not _create_exclusive(path, data):
            _MEMORY.pop(key, None)
            on_record = _read_record(key)
            if on_record is not None:
                return on_record
    except OSError as error:
        _warn_once(f"logical optimisation records unwritable ({error}); in-memory only")
    _MEMORY[key] = _UNCHANGED if body is None else body
    return _MEMORY[key]


def _mark_fixed(key: str) -> None:
    _MEMORY[key] = _FIXED
    try:
        path = cache_dir() / f"{key}.fixed"
        if not path.exists():
            _create_exclusive(path, b"")
    except OSError as error:
        _warn_once(f"logical optimisation records unwritable ({error}); in-memory only")


# ---------------------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------------------


#: rz / rx at quarter turns 0..3 as the translator's named gates (no sdg / sxdg handler).
_NAMED_CLIFFORDS = {"rz": ((), ("s",), ("z",), ("s", "z")),
                    "rx": ((), ("sx",), ("x",), ("sx", "x"))}


def translator_cliffords(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Every rz / rx at a multiple of pi/2 (to 1e-9) as named Clifford gates the translator has
    cheap gadgets for; everything else as it is. Exact up to global phase.
    """
    from translator.logical_optimizer import _angle_is_clifford

    out = circuit.copy_empty_like()
    for inst in circuit.data:
        name = inst.operation.name
        if name in _NAMED_CLIFFORDS and _angle_is_clifford(inst.operation.params[0]):
            quarter = int(round(float(inst.operation.params[0]) / (math.pi / 2))) % 4
            for gate in _NAMED_CLIFFORDS[name][quarter]:
                getattr(out, gate)(inst.qubits[0])
            continue
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def _adds_rotations(body: QuantumCircuit, answer: QuantumCircuit) -> bool:
    """
    An answer with more non-Clifford rotations than the program (any, for a Clifford one). O3
    trades two-qubit gates for rotations -- its block resynthesis writes even a Clifford
    unitary with rz(pi/4), rx(-3pi/4) ... -- and every rotation is a rotation gadget to the
    translator, which AQORN cannot merge or commute away.
    """
    from translator.logical_optimizer import non_clifford_rotations

    return non_clifford_rotations(answer) > non_clifford_rotations(body)


def _optimize_here(body: QuantumCircuit) -> Tuple[Optional[QuantumCircuit], Dict[str, object]]:
    """The round's answer (None: the body as it is), and how it got there."""
    import time

    from translator.logical_optimizer import (check_equivalence, non_clifford_rotations,
                                              optimize_verified, score)

    began = time.time()
    # No new rotations, inside the search: neither chain steps to a circuit with more.
    result, check = optimize_verified(body, seed=0, max_rotations=non_clifford_rotations(body))
    report = {"route": result.route, "changed": bool(result.changed),
              "optimize_seconds": round(result.seconds, 3),
              "equiv_method": check.get("equiv_method"), "rejected": check.get("rejected", "")}
    chosen = result.circuit if result.changed and result.circuit is not body else body
    if chosen.num_qubits != body.num_qubits:
        raise RuntimeError("logical optimisation changed the program's width")
    answer = translator_cliffords(chosen)
    if _adds_rotations(body, answer):
        # No new rotations: the best of the optimiser's other candidates that adds none and is
        # shown to be the same unitary, else the program as it is.
        report["rotation_rule"] = f"rejected {result.route}"
        answer = translator_cliffords(body)
        for option_score, _, option, route in sorted(result.alternatives, key=lambda o: o[:2]):
            if option is chosen or option is body or option_score >= score(body):
                continue
            candidate = translator_cliffords(option)
            if not _adds_rotations(body, candidate) and \
                    check_equivalence(body, option, seed=0)["equivalent"]:
                answer, report["rotation_rule"] = candidate, f"rejected {result.route}, took {route}"
                break
    report["total_seconds"] = round(time.time() - began, 3)
    if fingerprint(answer) == fingerprint(body):
        return None, report
    return answer, report


def _timeout() -> float:
    try:
        return float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def _optimize_body(body: QuantumCircuit) -> Optional[QuantumCircuit]:
    """
    `_optimize_here` within the time budget (`TIMEOUT_ENV`): in a child process, so a round
    that runs out can be killed. None when there is nothing to gain or the budget ran out.
    """
    import sys

    budget = _timeout()
    if budget <= 0:
        return _optimize_here(body)[0]
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo), env.get("PYTHONPATH")]))
    try:
        done = subprocess.run(
            [sys.executable, "-m", "translator.logical_optimization", "--worker"],
            input=_dump(body), capture_output=True, timeout=budget, cwd=str(repo), env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("logical optimisation ran out of its %.0f s budget on a %d-qubit, "
                       "%d-instruction program; recorded unchanged", budget, body.num_qubits,
                       len(body.data))
        return None
    if done.returncode != 0:
        raise RuntimeError("logical optimisation worker failed: "
                           + done.stderr.decode(errors="replace")[-2000:])
    if not done.stdout:
        return None
    from qiskit import qpy

    return qpy.load(io.BytesIO(done.stdout))[0]


def _rebuild(circuit: QuantumCircuit, body: QuantumCircuit, tail: List[Tuple[int, int]],
             key: str) -> QuantumCircuit:
    """`body` on `circuit`'s registers, then `circuit`'s final measurements, with provenance."""
    out = circuit.copy_empty_like()
    out.global_phase = circuit.global_phase + body.global_phase
    for inst in body.data:
        out.append(inst.operation, [out.qubits[body.find_bit(q).index] for q in inst.qubits])
    for qubit, clbit in tail:
        out.measure(out.qubits[qubit], out.clbits[clbit])
    metadata = dict(circuit.metadata or {})
    metadata[METADATA_KEY] = {"version": OPTIMIZER_VERSION, "program": key}
    out.metadata = metadata
    return out


def logically_optimized(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    `circuit` after one verified round of logical optimisation, the same object when the pass
    is off, cannot take the circuit, or finds nothing to improve. Never mutates `circuit`.
    The same program always gets the same answer (see "Why a cache" in the module docstring).
    """
    if not enabled():
        return circuit
    split = _body_and_tail(circuit)
    if split is None:
        _warn_once(f"logical optimisation skipped a circuit with mid-circuit measurement, reset "
                   f"or classical control ({circuit.name})")
        return circuit
    body, tail = split
    key = fingerprint(body)
    record = _read_record(key)
    if record is None:
        record = _write_record(key, _optimize_body(body))
        if isinstance(record, QuantumCircuit):
            _mark_fixed(fingerprint(record))
    if not isinstance(record, QuantumCircuit):
        return circuit
    if _adds_rotations(body, record):
        # Recorded before the rule (Clifford programs 2026-10-09 22:00, every program
        # 2026-10-10): not used. `drop_record` lets a round with the rule decide again.
        _warn_once(f"logical optimisation: the recorded answer for {circuit.name} adds "
                   f"non-Clifford rotations; the program is used as it is")
        return circuit
    return _rebuild(circuit, record, tail, key)


def seed_record(program: QuantumCircuit, optimized: Optional[QuantumCircuit]) -> str:
    """
    Record `optimized` (None: unchanged) as the pass's answer for `program`, after checking it
    is the same unitary -- to make the pass return circuits already chosen and measured
    elsewhere (the logical-optimisation study's). First writer wins, as for every record:
    returns "seeded", "agrees" (the same answer was on record) or "kept" (a different one was,
    and stays). Measurements on either side are ignored. "refused": `optimized` adds
    non-Clifford rotations, so the pass would never use it (`_adds_rotations`); nothing is
    written, and a round under the rule can decide instead (`drop_record` an older record first).
    """
    from translator.logical_optimizer import check_equivalence

    split = _body_and_tail(program)
    if split is None:
        raise ValueError("seed_record needs a unitary program followed by measurements")
    body = split[0]
    key = fingerprint(body)
    candidate = None
    if optimized is not None:
        candidate_split = _body_and_tail(optimized)
        if candidate_split is None:
            raise ValueError("the optimised circuit is not unitary + final measurements")
        candidate = candidate_split[0]
        check = check_equivalence(body, candidate)
        if not check["equivalent"]:
            raise ValueError(f"not the same unitary ({check['equiv_method']}: "
                             f"{check['equiv_detail']})")
    # The pass's own last step, so a seeded answer is what a fresh round would return.
    answer = translator_cliffords(candidate if candidate is not None else body)
    if _adds_rotations(body, answer):
        return "refused"
    candidate = None if fingerprint(answer) == key else answer
    before = _read_record(key)
    on_record = _write_record(key, candidate)
    if isinstance(on_record, QuantumCircuit):
        _mark_fixed(fingerprint(on_record))
    if before is None and (on_record is _UNCHANGED if candidate is None
                           else isinstance(on_record, QuantumCircuit)
                           and fingerprint(on_record) == fingerprint(candidate)):
        return "seeded"
    same = (candidate is None and on_record in (_UNCHANGED, _FIXED)) or (
        candidate is not None and isinstance(on_record, QuantumCircuit)
        and fingerprint(on_record) == fingerprint(candidate))
    return "agrees" if same else "kept"


def drop_record(program: QuantumCircuit) -> bool:
    """
    Forget the recorded answer for `program` (maintenance: so `seed_record` can replace an
    answer recorded by mistake). Never while other processes may be using the records -- they
    would keep the old answer in memory. True if there was a record.
    """
    split = _body_and_tail(program)
    if split is None:
        return False
    key = fingerprint(split[0])
    _MEMORY.pop(key, None)
    path = cache_dir() / f"{key}.qpy"
    if not path.exists():
        return False
    path.unlink()
    return True


def register_optimized(circuit: QuantumCircuit) -> None:
    """
    Declare `circuit` already optimised: `logically_optimized` will return it unchanged. For
    circuits derived gate for gate from an optimised program -- the Clifford proxy -- which the
    optimiser would otherwise simplify into a structure the program does not have.
    """
    if not enabled():
        return
    split = _body_and_tail(circuit)
    if split is not None:
        _mark_fixed(fingerprint(split[0]))


def _worker() -> None:
    """`python -m translator.logical_optimization --worker`: QPY body on stdin, the optimised
    body (or nothing) on stdout, a JSON report on stderr's last line."""
    import json
    import sys
    import warnings

    from qiskit import qpy

    warnings.filterwarnings("ignore")
    body = qpy.load(io.BytesIO(sys.stdin.buffer.read()))[0]
    optimized, report = _optimize_here(body)
    if optimized is not None:
        sys.stdout.buffer.write(_dump(optimized))
    sys.stderr.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--worker"]:
        _worker()
    else:
        raise SystemExit("usage: python -m translator.logical_optimization --worker")
