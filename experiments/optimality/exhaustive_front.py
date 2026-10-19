"""
exhaustive_front.py -- every tightest placement of every small program, scored on AQORN's own
objective and measured: the data of the optimality-front figure
(`experiments/figures/optimality_front/`).

Two claims rest on it. (1) Search: AQORN finds the placement its own objective ranks best --
the gap between its pick and the exhaustive optimum of that objective. (2) Trade-off: that pick
sits close to the measured front of kept shots against accuracy -- its gap to the best
placement on LSR, on logical error rate and on TVD, and how much of the space beats it on both.

Programs (75, as the earlier exhaustive suites): the 4-6 qubit QASMBench circuits that build, as
Clifford proxies (frames need a stabilizer circuit; `basis_trotter_n4` is the real 1888-gate file,
not the test file the by-name loader picks), and 20 random structured Clifford programs per width
(Experiment7a's generator and seeds). Everything is the current standard
(`packing.standard.STANDARD`, w10T with solo-HH pricing) **with the logical optimisation pass**
(`translator.logical_optimization`): each program is optimised once at the start and every
placement, plan and score sees the optimised program.

The space: the complete tightest [[4,2,2]] packing (partition x patch order x lane map;
24 / 720 / 720 placements at 4 / 5 / 6 qubits), under both H-commutation settings.

Per placement:
    cost             AQORN's own objective, exactly what `plan_standard` minimises when it
                     chooses: `measured_with_currency` + weight x model silent error (solo-HH
                     pricing, transversal gadget set). Checked: the planner's recorded value for
                     its own pick equals this (`cost_matches_planner`)
    two_qubit_gates, one_qubit_gates, depth
                     the physical circuit under the standard translator (layer-sorted,
                     transversal in-patch CZ / XCX, merging on)
    lsr, ler, tvd    Pauli frames on the noisy Stim circuit the simulator runs (Fig 6's
                     machine, alpha = 1): kept-shot fraction; logical error per kept shot on
                     the occupied lanes only; and TVD of the kept distribution, exactly:
                     for a Clifford program the noisy output is the ideal one XOR an
                     independent flip, the ideal output is uniform on an affine subspace
                     a + V, so TVD = P(flip not in V | kept). Equal to LER when the ideal output
                     is one bitstring, 0 when it is uniform. Validated against sampled TVD
                     (`--validate`)
    *_sel            the same three on independent shots: "best" on a measured metric is
                     decided on these, the gap read on the others (the winner's curse)
Placements that translate to the same physical circuit are converted and simulated once.

Arms: AQORN (`plan_standard`), naive packing (program order) and the interaction-count
partition, both at AQORN's commutation setting.

Outputs (`--out`, default results/optimality_front/): plans.csv, programs/*.qpy, parts/*.csv (one
per program and commutation setting; a rerun skips finished parts), space.csv (the figure's
input, one row per placement with `pick` naming the arms), each CSV with a JSON sidecar.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.exhaustive_front --validate
    PYTHONPATH=. venv/bin/python -m experiments.optimality.exhaustive_front --smoke
    PYTHONPATH=. venv/bin/python -m experiments.optimality.exhaustive_front --jobs 12
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import io
import json
import logging
import math
import re
import time
import warnings
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

NAME = "exhaustive_front"
N_CODE = 4
K = 2
_DATA_REGISTER = re.compile(r"data_(\d+)$")


@dataclass
class Settings:
    widths: Sequence[int] = (4, 5, 6)
    random_circuits: int = 20
    circuit_seed: int = 20260730  # Experiment7a's base seed, as exhaustive_silent / _frames
    gates_per_qubit: int = 10
    locality: float = 0.85
    qasmbench: bool = True
    alpha: float = 1.0
    shots: int = 500_000          # per half: one half picks "best", the other is reported
    smoke: bool = False
    max_placements: Optional[int] = None
    jobs: int = 12
    out: str = "results/optimality_front"


# ----------------------------------------------------------------------------------------
# programs
# ----------------------------------------------------------------------------------------


def _qasm_path(name: str) -> Optional[Path]:
    """The file a small QASMBench circuit lives in, preferring the one named after it."""
    from experiments.revision.common import QASM_ROOT

    folder = QASM_ROOT / "small" / name
    exact = folder / f"{name}.qasm"
    if exact.exists():
        return exact
    found = sorted(p for p in folder.glob("*.qasm") if "_transpiled" not in p.name)
    return found[0] if found else None


def programs(settings: Settings) -> List[Tuple[str, str, int, object]]:
    """(program id, source, width, measured program after the logical optimisation pass)."""
    from experiments.optimality.exhaustive_silent import Settings as RandomSettings, _random_circuit
    from experiments.revision.common import SMALL_QASMBENCH
    from experiments.shared.benchmark_circuits import create_proxy_from_qasm
    from experiments.shared.simulation import ensure_measurements
    from translator.logical_optimization import logically_optimized

    out = []
    if settings.qasmbench:
        for name in SMALL_QASMBENCH:
            path = _qasm_path(name)
            proxy = create_proxy_from_qasm(path, timeout_seconds=None) if path else None
            if proxy is None or proxy.num_qubits not in settings.widths:
                continue
            out.append((f"qb:{name}", "qasmbench", proxy.num_qubits,
                        logically_optimized(ensure_measurements(proxy))))
    random_settings = RandomSettings(gates_per_qubit=settings.gates_per_qubit,
                                     locality=settings.locality)
    count = 1 if settings.smoke else settings.random_circuits
    for q in settings.widths:
        for index in range(count):
            seed = settings.circuit_seed + index
            circuit = _random_circuit(q, seed, random_settings)
            out.append((f"rand:{q}q:{index:02d}", "random", q,
                        logically_optimized(ensure_measurements(circuit))))
    if settings.smoke:
        keep, seen = [], set()
        for item in out:
            if (item[1], item[2]) not in seen:
                seen.add((item[1], item[2]))
                keep.append(item)
        out = keep
    return out


def support_basis(program) -> List[int]:
    """
    A reduced basis of V, the linear part of the affine subspace the ideal output is uniform
    on (bit q = program qubit q). Exact, from the statevector (these programs are <= 6 qubits).
    """
    from qiskit.quantum_info import Statevector

    probabilities = Statevector(program.remove_final_measurements(inplace=False)).probabilities()
    support = [int(i) for i in np.nonzero(probabilities > 1e-9)[0]]
    basis: Dict[int, int] = {}  # pivot (highest bit) -> vector
    for x in support[1:]:
        v = x ^ support[0]
        for pivot in sorted(basis, reverse=True):
            if (v >> pivot) & 1:
                v ^= basis[pivot]
        if v:
            basis[v.bit_length() - 1] = v
    # Fully reduce so every pivot appears in exactly one vector.
    for pivot in sorted(basis):
        for other in basis:
            if other != pivot and (basis[other] >> pivot) & 1:
                basis[other] ^= basis[pivot]
    return [basis[p] for p in sorted(basis, reverse=True)]


def outside(flips: np.ndarray, basis: Sequence[int]) -> np.ndarray:
    """Which flip vectors are not in span(basis) (basis from `support_basis`)."""
    rest = flips.astype(np.int64, copy=True)
    for vector in basis:
        pivot = int(vector).bit_length() - 1
        rest ^= ((rest >> pivot) & 1) * np.int64(vector)
    return rest != 0


# ----------------------------------------------------------------------------------------
# AQORN's own objective
# ----------------------------------------------------------------------------------------


class StandardScorer:
    """`plan_standard`'s choice criterion for any placement (`packing.silent._plan_silent_aware`
    `score`), with the per-commutation cost models built once."""

    def __init__(self, program):
        from iceberg.costs import DEFAULT_WORK_WEIGHTS
        from iceberg.gadget_options import transversal_intra
        from packing.silent import build_costs, load_silent_rates
        from packing.standard import STANDARD, standard_noise_profile

        self.program = program
        self.noise = standard_noise_profile()
        self.transversal = STANDARD.transversal_intra_two_qubit
        self.rates = load_silent_rates(N_CODE, transversal_intra_two_qubit=self.transversal)
        self.weight = STANDARD.silent_weight_for(N_CODE)
        self.units = DEFAULT_WORK_WEIGHTS[1] / self.rates.p2
        self.costs = {}
        with transversal_intra(self.transversal):
            for commute in (False, True):
                profiles, priced = build_costs(program, N_CODE, commute=commute, noise=self.noise,
                                               rates=self.rates, silent_weight=1.0,
                                               solo_hh=STANDARD.solo_hh_aware)
                _, plain = build_costs(program, N_CODE, commute=commute, noise=self.noise,
                                       rates=self.rates, silent_weight=0.0,
                                       solo_hh=STANDARD.solo_hh_aware)
                self.costs[commute] = (profiles, priced, plain)

    def __call__(self, placement, commute: bool) -> Dict[str, object]:
        from iceberg.gadget_options import transversal_intra
        from packing.cost import DEFAULT_DEPTH_WEIGHT
        from packing.silent import groups_of, measured_with_currency, silent_constant

        with transversal_intra(self.transversal):
            measured, currency = measured_with_currency(
                self.program, placement, N_CODE, depth_weight=DEFAULT_DEPTH_WEIGHT,
                h_commutation=commute, noise=self.noise)
        if measured is None:
            return {"cost": math.inf, "cost_detected": math.inf, "model_silent": math.nan,
                    "currency": currency}
        profiles, priced, plain = self.costs[commute]
        groups = groups_of(placement)
        silent = silent_constant(profiles, self.rates) + (
            priced.of(groups) - plain.of(groups)) / priced.silent_scale
        scale = 1.0 if currency == "expected_error" else self.units
        return {"cost": measured + self.weight * silent * scale, "cost_detected": measured,
                "model_silent": silent, "currency": currency}


# ----------------------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------------------


def frame_metrics(encoded, noisy, positions, basis, *, shots: int, seed: int) -> Dict[str, float]:
    """LSR, occupied-lane LER, all-lane LER and exact TVD from one Pauli-frame sample."""
    import stim

    from experiments.revision.exp8_undetected.frames import _records
    from iceberg.layout import CodeLayout

    simulator = stim.FlipSimulator(batch_size=shots, disable_stabilizer_randomization=True,
                                   num_qubits=noisy.num_qubits, seed=seed)
    simulator.do(noisy)
    flips = simulator.get_measurement_flips(bit_packed=False)
    rows: Dict[str, Dict[int, int]] = {}
    for row, (name, index) in enumerate(_records(encoded)):
        rows.setdefault(name, {})[index] = row
    if flips.shape[0] != sum(len(v) for v in rows.values()):
        raise ValueError("measurement records and Stim flips disagree in count")

    detected = np.zeros(shots, dtype=bool)
    every = np.zeros(shots, dtype=bool)
    slot: Dict[Tuple[int, int], np.ndarray] = {}
    for name, by_index in rows.items():
        block = flips[[by_index[i] for i in sorted(by_index)]]
        if "parity" in name or "flag" in name:
            detected |= block.any(axis=0)
        elif "data" in name:
            detected |= (block.sum(axis=0) % 2).astype(bool)
            match = _DATA_REGISTER.search(name)
            if match is None:
                raise ValueError(f"cannot tell which patch register {name!r} reads")
            code = CodeLayout(block.shape[0])
            for lane in range(code.k):
                anchor, qubit = code.z_support(lane)
                flip = block[anchor] ^ block[qubit]
                slot[(int(match.group(1)), lane)] = flip
                every |= flip
    from experiments.optimality.placement_measure import outside_rows

    # One row of logical flips per shot, one column per program qubit -- a bool matrix, not a
    # packed integer, so programs of 64 qubits and more are counted correctly too.
    flips = np.stack([slot[tuple(position)] for position in positions], axis=1)
    kept = ~detected
    count = max(1, int(kept.sum()))
    kept_flips = flips[kept]
    return {"lsr": float(kept.mean()),
            "ler": float(kept_flips.any(axis=1).sum() / count),
            "tvd": float(outside_rows(kept_flips, basis).sum() / count),
            "ler_all_lanes": float((every & kept).sum() / count)}


def _circuit_key(circuit, positions) -> str:
    ops = [(inst.operation.name, tuple(circuit.find_bit(q).index for q in inst.qubits),
            tuple(circuit.find_bit(c).index for c in inst.clbits),
            tuple(float(p).hex() if isinstance(p, (int, float)) else repr(p)
                  for p in inst.operation.params))
           for inst in circuit.data]
    names = [(r.name, r.size) for r in circuit.qregs + circuit.cregs]
    return f"{zlib.crc32(repr((names, ops, list(positions))).encode()):08x}-{len(ops)}"


def _load_program(path: Path):
    from qiskit import qpy

    with path.open("rb") as stream:
        return qpy.load(stream)[0]


def _measure_part(job) -> Optional[str]:
    """Every placement of one program under one commutation setting -> one part CSV."""
    from analysis.gate_count_stats import one_two_qubit_gate_totals
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.tvd_explore.fastframes import at_alpha_one, rescale
    from packing.standard import standard_translator_config
    from translator.core import Translator

    settings_dict, pid, q, commute, program_path, basis, part_path = job
    settings = Settings(**settings_dict)
    part = Path(part_path)
    if part.exists():
        return str(part)
    program = _load_program(Path(program_path))
    scorer = StandardScorer(program)
    translator = Translator(config=standard_translator_config(N_CODE, commute=commute))
    cache: Dict[str, Dict[str, float]] = {}
    rows: List[Dict] = []
    began = time.time()
    for index, (placement, _) in enumerate(exhaustive_space(q, K, "tightest")):
        if settings.max_placements is not None and index >= settings.max_placements:
            break
        placement = [tuple(int(v) for v in p) for p in placement]
        row: Dict[str, object] = {"program": pid, "num_qubits": q, "commute": commute,
                                  "placement_index": index, "placement": json.dumps(placement)}
        row.update(scorer(placement, commute))
        try:
            result = translator.translate(program, placement)
        except RuntimeWarning:
            row["rejected"] = True
            rows.append(row)
            continue
        circuit = result.circuit
        positions = [tuple(p) for p in result.context.state_positions]
        one, two = one_two_qubit_gate_totals(circuit)
        row.update({"rejected": False, "one_qubit_gates": int(one), "two_qubit_gates": int(two),
                    "depth": int(circuit.depth())})
        key = _circuit_key(circuit, positions)
        if key not in cache:
            noisy = rescale(at_alpha_one(circuit), settings.alpha)
            seed = zlib.crc32(f"{pid}:{key}".encode()) % 1_000_000_007
            report = frame_metrics(circuit, noisy, positions, basis, shots=settings.shots,
                                   seed=seed)
            select = frame_metrics(circuit, noisy, positions, basis, shots=settings.shots,
                                   seed=seed + 1)
            cache[key] = dict(report, **{f"{m}_sel": select[m] for m in ("lsr", "ler", "tvd")})
        row.update(cache[key])
        row["circuit_key"] = key
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["part_seconds"] = round(time.time() - began, 1)
    frame["distinct_circuits"] = len(cache)
    temporary = part.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.rename(part)
    return str(part)


# ----------------------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------------------


def _plan_one(job) -> Dict[str, object]:
    """AQORN's plan, the two conventional placements, the support basis, for one program."""
    from iceberg.gadget_options import transversal_intra
    from experiments.revision.packers import interaction_partition, make_context
    from packing.plan import naive_packing
    from packing.standard import STANDARD, plan_standard, standard_noise_profile
    from translator.logical_optimizer import logical_counts

    pid, source, q, program = job
    began = time.time()
    plan = plan_standard(program)
    commute = bool(plan.h_commutation)
    considered = dict(plan.considered)
    recorded = considered.get("naive" if plan.fell_back else "packed")
    with transversal_intra(STANDARD.transversal_intra_two_qubit):
        interaction = [tuple(int(v) for v in p) for p in interaction_partition(
            make_context(program, N_CODE, commute, standard_noise_profile()))]
    counts = logical_counts(program)
    return {"program": pid, "source": source, "num_qubits": q, "commute": commute,
            "aqorn": json.dumps([tuple(int(v) for v in p) for p in plan.placement]),
            "naive": json.dumps([tuple(int(v) for v in p) for p in naive_packing(q, K)]),
            "interaction": json.dumps(interaction),
            "fell_back": bool(plan.fell_back), "planner_recorded": recorded,
            "basis": json.dumps(support_basis(program)),
            "support_bits": len(support_basis(program)),
            "logical_two_qubit": counts["two_qubit"], "logical_total": counts["total"],
            "plan_seconds": round(time.time() - began, 2)}


def run(settings: Settings, redo: Sequence[str] = (), only: Sequence[str] = ()) -> Optional[Path]:
    from qiskit import qpy

    from experiments.revision.common import write_sidecar
    from experiments.shared.parallel import parallel_map
    from translator.logical_optimization import state

    began = time.time()
    out = Path(settings.out)
    (out / "programs").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    print(f"[{NAME}] logical optimisation: {state()}; out {out}", flush=True)

    loaded = programs(settings)
    print(f"[{NAME}] {len(loaded)} programs: "
          + ", ".join(f"{q}q {sum(1 for p in loaded if p[2] == q)}" for q in settings.widths),
          flush=True)
    plans_path = out / "plans.csv"
    plans = pd.read_csv(plans_path) if plans_path.exists() else pd.DataFrame(columns=["program"])
    for pid in redo:  # after a fix to the pass: plan and measure these programs again
        stem = pid.replace(":", "_")
        for stale in [out / "programs" / f"{stem}.qpy", *(out / "parts").glob(f"{stem}__c*.csv")]:
            stale.unlink(missing_ok=True)
        plans = plans[plans["program"] != pid]
    missing = [p for p in loaded if p[0] not in set(plans["program"])]
    if missing:
        planned = parallel_map(_plan_one, missing, jobs=settings.jobs, label="plan")
        failed = [p[0] for p, r in zip(missing, planned) if r is None]
        if failed:
            raise SystemExit(f"[{NAME}] planning failed for {failed}")
        plans = pd.concat([plans, pd.DataFrame(planned)], ignore_index=True)
        order = {p[0]: i for i, p in enumerate(loaded)}
        plans = plans.sort_values("program", key=lambda c: c.map(order)).reset_index(drop=True)
        plans.to_csv(plans_path, index=False)
        write_sidecar(plans_path, experiment=NAME, settings=asdict(settings), started=began,
                      status="complete", extra={"replanned": [p[0] for p in missing]})
    print(f"[{NAME}] plans: {len(plans)}; AQORN fell back to naive on "
          f"{int(plans['fell_back'].sum())}", flush=True)

    jobs = []
    by_id = {pid: program for pid, _, _, program in loaded}
    for row in plans.itertuples(index=False):
        program_path = out / "programs" / f"{row.program.replace(':', '_')}.qpy"
        if not program_path.exists():
            buffer = io.BytesIO()
            qpy.dump(by_id[row.program], buffer)
            program_path.write_bytes(buffer.getvalue())
        for commute in (True, False):
            part = out / "parts" / f"{row.program.replace(':', '_')}__c{int(commute)}.csv"
            jobs.append((asdict(settings), row.program, int(row.num_qubits), commute,
                         str(program_path), json.loads(row.basis), str(part)))
    restrict = set(only) | set(redo)
    if restrict:  # measure these programs only, e.g. beside a running sweep; no combine
        jobs = [job for job in jobs if job[1] in restrict]
    jobs.sort(key=lambda job: -job[2])  # widest first: their parts take longest
    todo = sum(1 for job in jobs if not Path(job[-1]).exists())
    print(f"[{NAME}] {len(jobs)} parts, {todo} to do", flush=True)
    done = parallel_map(_measure_part, jobs, jobs=settings.jobs, label="measure")
    failed = [job[1] for job, result in zip(jobs, done) if result is None]
    if failed:
        print(f"[{NAME}] {len(failed)} parts failed: {failed}", flush=True)
    if restrict:
        print(f"[{NAME}] measured {sorted(restrict)} only; run without --redo / --programs to "
              f"combine", flush=True)
        return None
    return combine(settings, plans, began)


def combine(settings: Settings, plans: pd.DataFrame, began: float) -> Path:
    from experiments.revision.common import write_sidecar
    from experiments.shared.figures import stamp_provenance

    out = Path(settings.out)
    parts = [pd.read_csv(p) for p in sorted((out / "parts").glob("*.csv"))]
    space = pd.concat(parts, ignore_index=True)
    space = space[~space["rejected"].astype(bool)].copy()
    space["pick"] = ""
    space["source"] = "measured"
    checks = []
    for plan in plans.itertuples(index=False):
        mine = space["program"] == plan.program
        for arm in ("aqorn", "naive", "interaction"):
            hit = mine & (space["placement"] == getattr(plan, arm)) & (
                space["commute"].astype(bool) == bool(plan.commute))
            if hit.sum() == 0 and settings.max_placements is not None:
                continue  # a truncated (smoke) space need not contain the arm's placement
            if hit.sum() != 1:
                raise SystemExit(f"[{NAME}] {plan.program}: {arm} matches {int(hit.sum())} rows")
            index = space.index[hit][0]
            space.loc[index, "pick"] = ",".join(filter(None, [space.loc[index, "pick"], arm]))
        aqorn_rows = space[mine & space["pick"].str.contains("aqorn")]
        if len(aqorn_rows) and plan.planner_recorded == plan.planner_recorded:  # not NaN
            aqorn = aqorn_rows.iloc[0]
            checks.append(abs(float(aqorn["cost"]) - float(plan.planner_recorded))
                          <= 1e-6 * max(1.0, abs(float(plan.planner_recorded))))
    space = stamp_provenance(space)
    path = out / "space.csv"
    space.to_csv(path, index=False)
    matched = f"{sum(checks)}/{len(checks)}"
    write_sidecar(path, experiment=NAME, settings=asdict(settings), started=began,
                  status="complete", extra={"cost_matches_planner": matched})
    print(f"[{NAME}] space.csv: {len(space)} placements of {space['program'].nunique()} "
          f"programs, {space['circuit_key'].nunique()} distinct circuits; AQORN's cost equals "
          f"the planner's recorded value on {matched}", flush=True)
    return path


# ----------------------------------------------------------------------------------------
# validation: exact TVD from frames against the simulator's sampled TVD
# ----------------------------------------------------------------------------------------


def validate(settings: Settings, cases: int = 4) -> None:
    """A few placements: frames' exact TVD and LSR next to `evaluate`'s sampled ones."""
    from experiments.revision.exp7_optimality.space import exhaustive_space
    from experiments.shared.simulation import SimSettings, build_config, evaluate
    from experiments.tvd_explore.fastframes import at_alpha_one
    from packing.standard import standard_translator_config
    from translator.core import Translator

    picked = [p for p in programs(replace(settings, smoke=True)) if p[2] in (4, 5)]
    sim = SimSettings().with_standard()
    for pid, _, q, program in picked[:cases]:
        basis = support_basis(program)
        placement = [tuple(int(v) for v in p) for p in next(iter(exhaustive_space(q, K, "tightest")))[0]]
        result = Translator(config=standard_translator_config(N_CODE, commute=True)).translate(
            program, placement)
        positions = [tuple(p) for p in result.context.state_positions]
        exact = frame_metrics(result.circuit, at_alpha_one(result.circuit), positions, basis,
                              shots=1_000_000, seed=11)
        config = build_config(sim, n_code=N_CODE, alpha=settings.alpha, repetitions=400_000,
                              h_commutation=True)
        sampled = evaluate(program, placement, config)
        print(f"[validate] {pid:24s} support 2^{len(basis)}: LSR frames {exact['lsr']:.4f} "
              f"sampled {sampled['lsr']:.4f} | TVD exact {exact['tvd']:.4f} sampled "
              f"{sampled['tvd']:.4f} | LER {exact['ler']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=Settings.jobs)
    parser.add_argument("--shots", type=int, default=Settings.shots)
    parser.add_argument("--out", default=Settings.out)
    parser.add_argument("--smoke", action="store_true",
                        help="one program per source and width, 30 placements, 20k shots")
    parser.add_argument("--validate", action="store_true",
                        help="compare exact TVD from frames with the simulator's sampled TVD")
    parser.add_argument("--programs", nargs="*", default=[],
                        help="measure only these program ids' parts (no combine)")
    parser.add_argument("--redo", nargs="*", default=[],
                        help="program ids to plan and measure again (their plans, programs and "
                             "parts are dropped first)")
    args = parser.parse_args()
    settings = Settings(jobs=args.jobs, shots=args.shots, out=args.out)
    if args.validate:
        validate(settings)
        return
    if args.smoke:
        settings = replace(settings, smoke=True, max_placements=30, shots=20_000,
                           out=str(Path(args.out) / "smoke"))
    run(settings, redo=args.redo, only=args.programs)


if __name__ == "__main__":
    main()
