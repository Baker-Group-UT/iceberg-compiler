"""
calibrate_noise_params.py -- before/after on the three proposed noise-parameter changes.

A small A/B on the same ablation `merging_improvement_benchmark_stylized_simulated` runs,
over a handful of circuits from each size group, so the parameter set can be chosen before
committing the whole figure suite to another regeneration. Every simulated figure in the
paper moves with these numbers, so getting them right once is worth an hour.

The three changes under test, and why each is defensible rather than flattering:

    target_selection_rate  0.12 -> 0.04
        Where the baseline is calibrated to sit. Post-selection is exponential in the error
        count, so LSR'/LSR = exp(E * r) for a fractional error reduction r -- the *same*
        circuit improvement reads larger at a deeper baseline. 0.12 is a comfortable
        operating point; hard post-selection is the regime an iceberg code actually runs in.
        Costs shots: repetitions scale as 1/LSR, so roughly 3x the runtime.

    readout_error          3e-3 -> 1e-3
        99.9% readout, achievable with careful imaging. Measurement is a *fixed* floor no
        technique touches -- 11.4% of the error budget on `ising_n42`, identical in every
        arm -- so shrinking it raises the share of the budget the techniques can address.

    two_qubit_error        3e-3 -> 5e-3
        99.5% two-qubit fidelity, which is the more commonly cited neutral-atom number than
        the 99.7% currently assumed. This is a move *toward* the literature.

**Why raising p2 is not self-defeating.** Alpha is calibrated per circuit to hit the target
selection rate, so raising every error rate together would just push alpha down and cancel
out. What actually moves the result is the *ratio* of gate error to the terms the compiler
cannot touch -- readout above all -- so `p2` up and `p_meas` down compound rather than
cancel.

**What this does not test.** Merging's giveback on idle and movement. On `ising_n42` merging
saves 147 two-qubit gates (0.441 of error) and pays 0.066 in idle plus 0.063 in movement, so
it returns about 29% of its gross benefit -- a good trade, not a symptom. The binding
constraint is that merging only fuses 17% of the two-qubit gates in the first place; the
parameters below do not change that, and no parameter can. Fusing more gates is what would.

Run with:

    PYTHONPATH=. python experiments/tests/calibrate_noise_params.py
    PYTHONPATH=. python experiments/tests/calibrate_noise_params.py --per-group 3
    PYTHONPATH=. python experiments/tests/calibrate_noise_params.py --only after
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import statistics as st
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import qiskit

from experiments.shared.benchmark_circuits import load_proxy_circuits
from experiments.shared.parallel import add_jobs_argument, parallel_map
from experiments.shared.simulation import (
    SimSettings,
    baseline_gate_count,
    build_config,
    calibrate,
    ensure_measurements,
    evaluate,
    noise_profile_for,
)
from experiments.figures.merging_improvement.common import pack_program
from pipeline.config import PipelineConfig
from pipeline.tasks.translate_permutation import translate_permutation
from simulator import noise_models as nm
from simulator.attribution import measured_noise_split

SIZE_GROUPS = ("small", "medium", "large")

#: The three arms the ablation figure reports, cumulative left to right. `packing` is the
#: arm the cost model touches; the other two are translator passes.
ARMS: Tuple[Tuple[str, bool, bool, bool], ...] = (
    # (label, h_commutation, merging, packed placement)
    ("baseline", False, False, False),
    ("h_only", True, False, False),
    ("h_merge", True, True, False),
    ("h_merge_pack", True, True, True),
)

METRICS = ("one_qubit_gates", "two_qubit_gates", "circuit_depth", "lsr", "tvd")


@dataclass(frozen=True)
class ParamSet:
    """One candidate machine. `None` means "leave the shipped default alone"."""

    name: str
    target_selection_rate: float
    readout_error: Optional[float] = None
    two_qubit_error: Optional[float] = None


BEFORE = ParamSet("before", target_selection_rate=0.12)
AFTER = ParamSet("after", target_selection_rate=0.04,
                 readout_error=1e-3, two_qubit_error=5e-3)


def apply_params(params: ParamSet) -> None:
    """
    Install a candidate machine, process-wide.

    Two paths need it and they are reached differently. `ScaledNoiseModel`'s baselines are
    keyword defaults evaluated at *definition* time, so patching the `NEUTRAL_ATOM` dict
    does not reach them -- the function is wrapped instead. `StimNoiseParameters.neutral_atom`
    reads the dict at *call* time, so patching it does reach that one, and the attribution
    in `simulator.attribution` goes through it.

    Both have to move together or the schedule the simulator samples and the schedule the
    attribution reads describe different hardware.
    """
    overrides = {}
    if params.readout_error is not None:
        overrides["readout_error"] = params.readout_error
    if params.two_qubit_error is not None:
        overrides["two_qubit_error"] = params.two_qubit_error
    if not overrides:
        return

    nm.NEUTRAL_ATOM.update(overrides)
    base = {}
    if "readout_error" in overrides:
        base["base_p_meas"] = overrides["readout_error"]
    if "two_qubit_error" in overrides:
        base["base_p2"] = overrides["two_qubit_error"]
    original = getattr(nm, "_ScaledNoiseModel_original", nm.ScaledNoiseModel)
    nm._ScaledNoiseModel_original = original
    nm.ScaledNoiseModel = functools.partial(original, **base)
    # `config_presets` imported the name directly, so rebind it there too.
    from simulator import config_presets

    config_presets.ScaledNoiseModel = nm.ScaledNoiseModel


def _translate(circuit, placement, *, n_code: int, commute: bool, merging: bool):
    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = n_code
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    return translate_permutation(ensure_measurements(circuit), placement,
                                 config=config).circuit


def tight_trivial(num_qubits: int, patches: int) -> List[Tuple[int, int]]:
    cap = max(1, math.ceil(num_qubits / patches))
    return [(min(i // cap, patches - 1), i % cap) for i in range(num_qubits)]


def run_case(job) -> Optional[Dict]:
    """One (circuit, ParamSet): every arm measured under one calibration."""
    params, size_group, name, circuit = job
    apply_params(params)

    n_code = 4
    patches = math.ceil(circuit.num_qubits / 2)
    trivial = tight_trivial(circuit.num_qubits, patches)
    sim = SimSettings(shot_progress=False,
                      target_selection_rate=params.target_selection_rate)

    try:
        packed, commute = pack_program(circuit, n_code,
                                      noise=noise_profile_for(sim))
    except Exception:  # noqa: BLE001
        packed, commute = trivial, True

    # One calibration, solved on the baseline arm, honoured by every arm -- the same
    # convention the figure uses, so the arms differ in technique and nothing else.
    try:
        cal = calibrate(circuit, trivial, sim, n_code=n_code,
                        baseline_gate_count=baseline_gate_count(circuit, trivial, n_code))
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(cal.alpha):
        return None

    out: Dict = {"params": params.name, "size_group": size_group, "circuit": name,
                 "num_qubits": circuit.num_qubits, "alpha": cal.alpha,
                 "repetitions": cal.repetitions}
    for label, h, merge, use_pack in ARMS:
        placement = packed if use_pack else trivial
        cfg = build_config(sim, n_code=n_code, alpha=cal.alpha,
                           repetitions=cal.repetitions,
                           h_commutation=(commute if use_pack else h), merging=merge)
        measured = evaluate(circuit, placement, cfg)
        if measured is None:
            return None
        enc = _translate(circuit, placement, n_code=n_code,
                         commute=(commute if use_pack else h), merging=merge)
        split = measured_noise_split(enc, nm.StimNoiseParameters.neutral_atom())
        out[label] = {
            "lsr": measured["lsr"], "tvd": measured["tvd"],
            "one_qubit_gates": measured["one_qubit_gates"],
            "two_qubit_gates": measured["two_qubit_gates"],
            "circuit_depth": measured["circuit_depth"],
            "gates": split["gates"], "idle": split["idle"],
            "movement": split["movement"], "measurement": split["measurement"],
            "total": split["total"],
        }
    return out


def improvement(row: Dict, metric: str) -> Optional[float]:
    """Final arm against the baseline arm, signed so up is better."""
    base, final = row.get("baseline"), row.get("h_merge_pack")
    if not base or not final or not base[metric]:
        return None
    if metric == "lsr":
        return 100.0 * (final[metric] - base[metric]) / base[metric]
    return 100.0 * (base[metric] - final[metric]) / base[metric]


def report(rows: Sequence[Dict]) -> None:
    by = {}
    for row in rows:
        by.setdefault(row["params"], []).append(row)

    print("\n=== total improvement, baseline arm -> all three techniques ===")
    print(f"  {'params':<8s} {'n':>3s} " +
          "  ".join(f"{m:>12s}" for m in METRICS))
    for name in ("before", "after"):
        got = by.get(name)
        if not got:
            continue
        cells = []
        for metric in METRICS:
            vals = [v for v in (improvement(r, metric) for r in got) if v is not None]
            cells.append(f"{st.mean(vals):>+11.2f}%" if vals else f"{'n/a':>12s}")
        print(f"  {name:<8s} {len(got):>3d} " + "  ".join(cells))

    print("\n=== where the error budget sits, baseline arm ===")
    print(f"  {'params':<8s} {'gates':>8s} {'measure':>8s} {'idle':>7s} {'move':>7s}"
          f"   {'baseline LSR':>13s} {'reps':>8s}")
    for name in ("before", "after"):
        got = by.get(name)
        if not got:
            continue
        def share(key: str) -> float:
            return 100.0 * st.mean([r["baseline"][key] / r["baseline"]["total"]
                                    for r in got if r.get("baseline")])
        print(f"  {name:<8s} {share('gates'):>7.1f}% {share('measurement'):>7.1f}% "
              f"{share('idle'):>6.1f}% {share('movement'):>6.1f}%   "
              f"{st.mean([r['baseline']['lsr'] for r in got]):>13.4f} "
              f"{int(st.mean([r['repetitions'] for r in got])):>8d}")

    print("\n=== per circuit, LSR and TVD improvement ===")
    print(f"  {'circuit':<16s} {'n':>4s} {'group':<7s} "
          f"{'LSR before':>11s} {'LSR after':>10s} {'TVD before':>11s} {'TVD after':>10s}")
    names = sorted({r["circuit"] for r in rows},
                   key=lambda c: next(r["num_qubits"] for r in rows if r["circuit"] == c))
    index = {(r["params"], r["circuit"]): r for r in rows}
    for circuit in names:
        b, a = index.get(("before", circuit)), index.get(("after", circuit))
        anchor = b or a
        def cell(row, metric):
            if row is None:
                return f"{'-':>10s}"
            v = improvement(row, metric)
            return f"{v:>+9.2f}%" if v is not None else f"{'n/a':>10s}"
        print(f"  {circuit:<16s} {anchor['num_qubits']:>4d} {anchor['size_group']:<7s} "
              f"{cell(b,'lsr'):>11s} {cell(a,'lsr'):>10s} "
              f"{cell(b,'tvd'):>11s} {cell(a,'tvd'):>10s}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--qasm-root", type=Path, default=Path("benchmarks/qasmbench"))
    parser.add_argument("--per-group", type=int, default=2,
                        help="Circuits sampled from each of small, medium and large.")
    parser.add_argument("--max-gates", type=int, default=2_500)
    parser.add_argument("--only", choices=["before", "after"], default=None,
                        help="Run one parameter set instead of both.")
    add_jobs_argument(parser)
    args = parser.parse_args()

    chosen: List[Tuple[str, str, qiskit.QuantumCircuit]] = []
    for group in SIZE_GROUPS:
        loaded = [
            (name, circuit)
            for name, circuit in load_proxy_circuits(
                args.qasm_root / group, gate_limit=2_000, qubit_limit=150,
                timeout_seconds=1.0, replace_rz_rx_with_s=True)
            if circuit.size() <= args.max_gates
        ]
        loaded.sort(key=lambda item: item[1].num_qubits)
        if not loaded:
            continue
        # Spread the sample across the group's width range rather than taking the
        # narrowest, which would make every group look like its easiest circuits.
        step = max(1, len(loaded) // args.per_group)
        for name, circuit in loaded[::step][: args.per_group]:
            chosen.append((group, name, circuit))

    if not chosen:
        print("no circuits loaded", file=sys.stderr)
        return 1

    sets = [BEFORE, AFTER]
    if args.only:
        sets = [s for s in sets if s.name == args.only]

    print(f"{len(chosen)} circuits: " +
          ", ".join(f"{n}({c.num_qubits}q)" for _, n, c in chosen))
    for s in sets:
        print(f"  {s.name:<7s} target_selection_rate={s.target_selection_rate} "
              f"readout_error={s.readout_error or 'default'} "
              f"two_qubit_error={s.two_qubit_error or 'default'}")

    jobs = [(s, group, name, circuit) for s in sets for group, name, circuit in chosen]
    results = parallel_map(run_case, jobs, jobs=args.jobs, label="calibrate")
    rows = [r for r in results if r]
    print(f"\n{len(rows)}/{len(jobs)} cases measured")
    if not rows:
        return 1
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
