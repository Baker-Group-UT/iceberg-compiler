"""
density_references.py
---------------------

Figures from the matched density sweep (`density_ablation_sweep.py --standard
--matched-patches`), with two measurement fixes applied first. No new noisy simulation.

    recount   logical error per kept shot, counting only the lanes a program qubit occupies
    floors    each arm's TVD sampling floor, from the program's exact output distribution
    plot      three figures (below), from the matched CSV and the two tables

"Logical error per kept shot" is the post-selected logical error rate: the fraction of shots
that pass every check yet carry a wrong logical outcome. The CSVs and the planner call it the
silent error (`silent_per_kept`, `silent_weight`); the figures use the standard name.

Each step writes `<matched CSV stem>_{recount,floors,summary}.csv` beside the matched CSV
(`_v2`, `_v3`, ... rather than replacing one), and `plot` reads the newest of each.

Why no new simulation: within a circuit every density point shares alpha and the shot count,
so the four arms at every point are measurements on one machine, and both fixed references --
the naive tight packing (ceil(n/2) patches) and the naive sparse one (n patches, one qubit
apiece) -- are arms the sweep already ran at its two end points.

The two fixes
-------------
Neither matters much *within* one density; both matter across densities.

  * Logical error per kept shot (`frames.split_from_stim`) counts a logical flip on every lane
    of every patch, including lanes no program qubit occupies. A layout at P patches has
    2P - n of them, up to n when every qubit sits alone, so the metric reads high on sparse
    layouts (TVD / logical error per kept shot is 0.86 on tight arms, 0.66 on all-solo ones).
    `recount` re-runs the frames counting only the lanes a program qubit ends on
    (`TranslationContext.state_positions`, the bits TVD keeps). It reruns each arm's own
    pipeline with one shot to rebuild the noisy Stim circuit -- the conversion is the
    expensive step, shots are not -- and, from the same frame sample with the sweep's own
    frame seed, also counts every lane: that number must equal the CSV's, which is the check
    that the circuit, its noise and the seed are the sweep's. `plot` refuses to run if any
    arm fails it.
  * TVD is measured between K kept noisy shots and R noiseless shots of the program, so it has
    a floor that grows as K falls -- sparse arms keep about half the shots of tight ones, and
    AQORN's arm keeps about 1.4x the baseline's at the tightest point. `floors` computes each
    arm's zero-noise floor, E[TVD] between K and R draws from the program's exact output
    distribution, and the TVD panels show TVD above it, wherever the reference arm's TVD is at
    least twice its own floor. Subtracting the zero-noise floor slightly over-corrects where
    noise moves the output a lot; at these noise levels the floor is close to additive.

    The floor is set by how many outcomes the ideal output has, not by width, so this test
    replaces the blanket 18-qubit cutoff (`tvd-estimability`, measured on broad outputs). The
    simulated programs are Clifford proxies, whose output is uniform over an affine subspace,
    so its exact size comes from the GF(2) rank of a noiseless sample at any width. In this
    corpus 30 of the 35 circuits above 18 qubits have one or two outcomes and a floor near
    zero; the Ising family (ising_n26..n98, all 2^n outcomes, TVD = 1.0 on every arm) is what
    the test excludes, the same five circuits Fig 6 withholds.

The figures
-----------
  density_ablation_matched   the ablation regenerated: H commutation / + Merging / + Packing
                             stacked against the naive packing at each density (same patch
                             count); panels gates, depth, LSR, TVD (above floor), logical
                             error per kept shot (occupied lanes); bars with >= 4 circuits
  density_vs_naive_tight     AQORN's packed arm at each density against the naive tight
                             packing with the same passes (+H +merge)
  density_vs_naive_sparse    the same against the naive sparse packing
  density_ablation_split     from a `--split-merging` CSV: the ablation on a baseline with no
                             merging at all, with merging in two stages -- + 1q merging (HH,
                             XX, ZZ, a solo qubit's H as HH) and + 2q merging (CXT / CZT / XCXT,
                             transversal in-patch CZ / XCX). The two reference figures are
                             drawn from it too (they use its +merge and packed arms).

In the two reference figures a bar splits into a patch-count segment (the naive packing at that
density against the reference) and a placement segment (AQORN against the naive packing at that
density); either can be negative, so the stack diverges from zero and a black tick marks the net.

    python -m experiments.figures.density_ablation_sweep.density_references recount
    python -m experiments.figures.density_ablation_sweep.density_references floors
    python -m experiments.figures.density_ablation_sweep.density_references plot \\
        --output-dir <project>/results/figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from experiments.figures.density_ablation_sweep.density_ablation_sweep import (
    DEFAULT_SETTINGS, MATCHED_CSV, SIZE_GROUPS, _density_bins, _load_group_circuits,
    build_pipeline_config, ensure_measurements, tight_trivial_packing,
)
from experiments.shared.parallel import add_jobs_argument, parallel_map

N_CODE = 4
ARMS = ("baseline", "h_only", "h_merge", "packed")
#: A `--split-merging` CSV's arms: merging in two steps on a merging-free baseline.
SPLIT_ARMS = ("baseline", "h_only", "h_merge_1q", "h_merge", "packed")
#: arm -> (H commutation on?, merging on?, seed offset), as `_run_one_circuit_standard` has
#: them; None means the circuit's own commutation decision.
ARM_SETTINGS = {"baseline": (False, False, 3), "h_only": (None, False, 4),
                "h_merge_1q": (None, True, 7), "h_merge": (None, True, 5),
                "packed": (None, True, 6)}


def is_split(frame: pd.DataFrame) -> bool:
    return "h_merge_1q_lsr" in frame.columns


def arms_of(frame: pd.DataFrame) -> Tuple[str, ...]:
    return SPLIT_ARMS if is_split(frame) else ARMS
#: A reference TVD within this factor of its own floor is not a measurement (`floors`).
SIGNAL_OVER_FLOOR = 2.0
#: Above this many support bits two samples of the ideal output barely share an outcome, so
#: the floor is 1 to the precision that matters and is not sampled.
MAX_SAMPLED_SUPPORT_BITS = 20
#: panel -> (label, the per-arm columns it averages, lower is better)
PANELS = {
    "gates": ("#gates (1q, 2q avg)", ("one_qubit_gates", "two_qubit_gates"), True),
    "depth": ("depth", ("depth",), True),
    "lsr": ("LSR", ("lsr",), False),
    "tvd": ("TVD above floor", ("tvd",), True),
    "logical_error": ("logical error rate", ("logical_error",), True),
}
GROUP_INITIAL = {"small": "S", "medium": "M", "large": "L"}
_DATA_REGISTER = re.compile(r"data_(\d+)$")


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def companion(matched_path: Path, kind: str) -> Path:
    return matched_path.with_name(f"{matched_path.stem}_{kind}.csv")


def _free(base: Path) -> Path:
    """`base`, else `<stem>_v2`, `_v3`, ...: never replace a table."""
    path, version = base, 2
    while path.exists():
        path, version = base.with_name(f"{base.stem}_v{version}{base.suffix}"), version + 1
    return path


def _latest(base: Path) -> Path:
    """The newest of `base`, `<stem>_v2`, ... that exists."""
    found = [base] if base.exists() else []
    version = 2
    while (candidate := base.with_name(f"{base.stem}_v{version}{base.suffix}")).exists():
        found.append(candidate)
        version += 1
    if not found:
        raise FileNotFoundError(base)
    return found[-1]


def _density_rows(matched: pd.DataFrame) -> pd.DataFrame:
    """One row per (circuit, density point): every arm's columns are on each component row."""
    return matched[matched["component"] == "packing"].copy()


def _circuits(matched: pd.DataFrame, max_qubits: Optional[int] = None):
    """(group, index, name, simulation circuit, density rows) for every circuit in the CSV."""
    rows = _density_rows(matched)
    for group in SIZE_GROUPS:
        names = sorted(rows.loc[rows["size_group"] == group, "circuit_name"].unique())
        if not names:
            continue
        loaded = _load_group_circuits(DEFAULT_SETTINGS.qasm_root, group, DEFAULT_SETTINGS)
        for name in names:
            block = rows[rows["circuit_name"] == name]
            index = int(block["circuit_idx"].iloc[0])
            loaded_name, _, simulation = loaded[index]
            if loaded_name != name:
                raise RuntimeError(f"{group}[{index}] is {loaded_name}, the CSV says {name}")
            if max_qubits is not None and simulation.num_qubits > max_qubits:
                continue
            yield group, index, name, simulation, block.to_dict("records")


def shared_arms(row: Dict) -> Dict[str, str]:
    """{arm: the arm whose simulation it took} at one density point (`--share-identical`)."""
    value = row.get("shared_arms")
    if not isinstance(value, str) or not value:
        return {}
    return dict(pair.split("=", 1) for pair in value.split(";"))


def _permutation(arm: str, row: Dict, num_qubits: int):
    if arm == "packed":
        return [tuple(p) for p in json.loads(row["placement"])]
    return tight_trivial_packing(num_qubits, int(row["n_codes"]))


# ---------------------------------------------------------------------------
# recount: logical error per kept shot on occupied lanes (the CSVs' silent_per_kept)
# ---------------------------------------------------------------------------

def _arm_flags(arm: str, commute: bool, split: bool) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Every translator setting the sweep varies by arm (`configured` in
    `_run_one_circuit_standard`): H commutation, merging, the transversal in-patch CZ / XCX,
    the solo-H promotion, and two-qubit macros.
    """
    from packing.standard import STANDARD

    h_commutation, merging, _ = ARM_SETTINGS[arm]
    h_commutation = commute if h_commutation is None else h_commutation
    single_qubit_only = split and arm == "h_merge_1q"
    transversal = STANDARD.transversal_intra_two_qubit and merging and not single_qubit_only
    solo_promotion = merging if split else True
    return h_commutation, merging, transversal, solo_promotion, not single_qubit_only


def _arm_config(arm: str, *, alpha: float, seed: int, commute: bool, split: bool = False):
    """The arm's simulation config as the sweep builds it, with one shot."""
    from packing.standard import STANDARD

    h_commutation, merging, transversal, solo_promotion, two_qubit = _arm_flags(
        arm, commute, split)
    config = build_pipeline_config(
        DEFAULT_SETTINGS, alpha=alpha, repetitions=1, seed=seed,
        enable_h_commutation=h_commutation, enable_merging=merging,
    )
    translator = config.translator_config
    translator.n_code = N_CODE
    translator.enable_h_commutation = h_commutation
    translator.enable_merging = merging
    translator.asap_order = STANDARD.asap_order
    translator.transversal_intra_two_qubit = transversal
    translator.solo_hh_promotion = solo_promotion
    translator.merge_two_qubit = two_qubit
    return config


def split_by_lanes(encoded, noisy, occupied, *, shots: int, seed: int) -> Dict[str, float]:
    """
    `frames.split_from_stim`, counting a logical flip two ways from one frame sample: on every
    lane (what the sweep recorded) and on the `occupied` (patch, lane)s only.
    """
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
    used = np.zeros(shots, dtype=bool)
    for name, by_index in rows.items():
        block = flips[[by_index[i] for i in sorted(by_index)]]
        if "parity" in name or "flag" in name:
            detected |= block.any(axis=0)
        elif "data" in name:
            detected |= (block.sum(axis=0) % 2).astype(bool)
            match = _DATA_REGISTER.search(name)
            if match is None:
                raise ValueError(f"cannot tell which patch register {name!r} reads")
            patch = int(match.group(1))
            code = CodeLayout(block.shape[0])
            for lane in range(code.k):
                anchor, qubit = code.z_support(lane)
                flip = block[anchor] ^ block[qubit]
                every |= flip
                if (patch, lane) in occupied:
                    used |= flip
    kept = max(1, int((~detected).sum()))
    return {
        "silent_all": float((every & ~detected).sum() / kept),
        "silent_used": float((used & ~detected).sum() / kept),
        "frame_lsr": 1.0 - float(detected.mean()),
    }


@contextmanager
def _capture_translations():
    """Record what `Translator.translate_batch` returns, so the context comes from the very
    translation the simulator converted rather than from a second one."""
    from translator.core import Translator

    original = Translator.translate_batch
    captured: List = []

    def recording(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        captured.extend(out)
        return out

    Translator.translate_batch = recording
    try:
        yield captured
    finally:
        Translator.translate_batch = original


def _recount_work(matched: pd.DataFrame):
    """
    (copies, items): arm-points that need no recount -- every lane occupied, at 2P = n -- and
    one work item per distinct circuit to convert, listing the arms that share it (the packed
    arm at one qubit per patch is the +merge arm's circuit). Largest circuits first.
    """
    copies: List[Dict] = []
    items: List[Tuple] = []
    split = is_split(matched)
    for group, index, name, simulation, rows in _circuits(matched):
        n = simulation.num_qubits
        code_counts = _density_bins(n, DEFAULT_SETTINGS.n_codes_samples)
        for row in rows:
            n_codes = int(row["n_codes"])
            commute = bool(row["commute"])
            shared: Dict[Tuple, Tuple[str, list, list]] = {}
            took = shared_arms(row)
            for arm in arms_of(matched):
                recorded = float(row[f"{arm}_silent_per_kept"])
                entry = {"size_group": group, "circuit_name": name, "num_qubits": n,
                         "n_codes": n_codes, "arm": arm, "recorded": recorded}
                if 2 * n_codes == n:
                    copies.append({**entry, "silent_all": recorded, "silent_used": recorded,
                                   "frame_lsr": float("nan"), "recounted": False,
                                   "matches": True})
                    continue
                # An arm that took another's simulation is recounted as that arm: its circuit,
                # converted once, and its seed, which drew the frames the sweep recorded.
                source = took.get(arm, arm)
                offset = ARM_SETTINGS[source][2]
                permutation = _permutation(source, row, n)
                seed = (DEFAULT_SETTINGS.seed * 1_000_003 + index * 100_000
                        + code_counts.index(n_codes) * 10 + offset)
                # Every setting that changes the circuit, not just H commutation and merging:
                # the split run's 1q-only arm shares those two with +merge.
                key = (_arm_flags(source, commute, split), tuple(tuple(p) for p in permutation))
                if key not in shared:
                    shared[key] = (source, permutation, [])
                shared[key][2].append((entry, seed))
            for first, permutation, arms in shared.values():
                items.append((first, float(row["alpha_final"]), commute, split, simulation,
                              permutation, arms, DEFAULT_SETTINGS.frame_shots))
    items.sort(key=lambda item: -item[4].num_qubits)
    return copies, items


def _recount_item(job) -> List[Dict]:
    """One conversion, and the frames of every arm that shares it. Module scope, for `spawn`."""
    from experiments.revision.exp8_undetected.frames import capture_noisy_stim
    from pipeline.tasks.scan_specific import run_specific_scan

    first, alpha, commute, split, simulation, permutation, arms, frames = job
    program = ensure_measurements(simulation)
    config = _arm_config(first, alpha=alpha, seed=arms[0][1], commute=commute, split=split)
    with capture_noisy_stim() as converted, _capture_translations() as translations:
        run_specific_scan(program_circuit=program, permutations=[permutation], config=config)
    encoded, noisy = converted[-1]
    result = translations[-1]
    if (result.circuit.num_qubits != encoded.num_qubits
            or dict(result.circuit.count_ops()) != dict(encoded.count_ops())):
        raise RuntimeError(f"{arms[0][0]['circuit_name']} n_codes={arms[0][0]['n_codes']}: the "
                           f"translation does not match the circuit the simulator converted")
    occupied = {tuple(p) for p in result.context.state_positions}
    out = []
    for entry, seed in arms:
        split = split_by_lanes(encoded, noisy, occupied, shots=frames, seed=seed)
        out.append({**entry, **split, "recounted": True,
                    "matches": bool(np.isclose(split["silent_all"], entry["recorded"],
                                               rtol=1e-12, atol=0.0))})
    return out


def run_recount(matched_path: Path, jobs: Optional[int]) -> Path:
    matched = pd.read_csv(matched_path)
    copies, items = _recount_work(matched)
    print(f"[density_references] {len(items)} conversions for "
          f"{sum(len(item[6]) for item in items)} arm-points; {len(copies)} need none")
    produced = parallel_map(_recount_item, items, jobs=jobs, label="recount")
    failed = [(item[6][0][0]["circuit_name"], item[6][0][0]["n_codes"], item[0])
              for item, result in zip(items, produced) if result is None]
    table = pd.DataFrame(copies + [row for result in produced if result for row in result])
    path = _free(companion(matched_path, "recount"))
    table.to_csv(path, index=False)
    recounted = table[table["recounted"]]
    print(f"[density_references] {len(table)} arm-points, {len(recounted)} recounted; the "
          f"all-lane count equals the sweep's on {int(recounted['matches'].sum())}/"
          f"{len(recounted)}; failed: {failed or 'none'}")
    print(f"[density_references] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# floors: TVD's zero-noise sampling floor per arm
# ---------------------------------------------------------------------------

def support_bits(program, *, shots: int = 4096) -> int:
    """
    log2 of how many outcomes the program's ideal output has.

    The simulated programs are Clifford, and a stabilizer circuit's measurement outcomes are
    uniform over an affine subspace of GF(2)^m; its dimension is the rank of a noiseless
    sample's differences from its first row. Exact with overwhelming probability once `shots`
    exceeds the dimension by a margin, and at any width -- no state vector.
    """
    from simulator.stim_backend import qiskit_to_stim

    sample = qiskit_to_stim(program, layout=None, include_noise=False).compile_sampler(
        seed=0).sample(shots).astype(np.uint8)
    rows = (sample[1:] ^ sample[0]).copy()
    rank = 0
    for column in range(rows.shape[1]):
        pivots = np.nonzero(rows[rank:, column])[0]
        if pivots.size == 0:
            continue
        pivot = rank + pivots[0]
        rows[[rank, pivot]] = rows[[pivot, rank]]
        hits = np.nonzero(rows[:, column])[0]
        rows[hits[hits != rank]] ^= rows[rank]
        rank += 1
        if rank == rows.shape[0]:
            break
    return rank


def tvd_floor(bits: int, kept: int, reps: int, *, draws: int, seed: int) -> Tuple[float, float]:
    """
    E[TVD] between `kept` and `reps` independent draws from the uniform distribution over 2^bits
    outcomes: (mean, standard error). Above `MAX_SAMPLED_SUPPORT_BITS` the two samples barely
    overlap and the floor is 1.
    """
    if bits > MAX_SAMPLED_SUPPORT_BITS:
        return 1.0, 0.0
    p = np.full(2 ** bits, 1.0 / 2 ** bits)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for index in range(draws):
        a = rng.multinomial(kept, p) / kept
        b = rng.multinomial(reps, p) / reps
        values[index] = 0.5 * np.abs(a - b).sum()
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(draws))


def _floors_circuit(job) -> List[Dict]:
    """Every arm at every density point of one circuit. Module scope, for `spawn`."""
    group, _, name, simulation, rows, draws, arms = job
    program = ensure_measurements(simulation)
    bits = support_bits(program)
    out = []
    for row in rows:
        reps = int(row["final_repetitions"])
        for arm in arms:
            kept = int(round(float(row[f"{arm}_lsr"]) * reps))
            if kept > 0:
                floor, error = tvd_floor(bits, kept, reps, draws=draws, seed=int(row["n_codes"]))
            else:
                floor, error = float("nan"), float("nan")
            out.append({"size_group": group, "circuit_name": name,
                        "num_qubits": program.num_qubits, "n_codes": int(row["n_codes"]),
                        "arm": arm, "kept": kept, "reps": reps, "support_bits": bits,
                        "tvd": float(row[f"{arm}_tvd"]), "floor": floor, "floor_se": error})
    return out


def run_floors(matched_path: Path, jobs: Optional[int], draws: int) -> Path:
    matched = pd.read_csv(matched_path)
    work = [(group, index, name, simulation, rows, draws, arms_of(matched))
            for group, index, name, simulation, rows in _circuits(matched)]
    produced = parallel_map(_floors_circuit, work, jobs=jobs, label="floors")
    failed = [job[2] for job, result in zip(work, produced) if result is None]
    table = pd.DataFrame([row for result in produced if result for row in result])
    path = _free(companion(matched_path, "floors"))
    table.to_csv(path, index=False)
    per_circuit = table.groupby("circuit_name")[["support_bits", "num_qubits"]].first()
    measurable = table[(table["arm"] == "h_merge")].groupby("circuit_name").apply(
        lambda b: bool((b["tvd"] >= SIGNAL_OVER_FLOOR * b["floor"]).all()), include_groups=False)
    print(f"[density_references] floors for {len(per_circuit)} circuits (failed: "
          f"{failed or 'none'}); support of at most 1 bit on {int((per_circuit.support_bits <= 1).sum())}; "
          f"TVD at least {SIGNAL_OVER_FLOOR:g}x its floor at every point on "
          f"{int(measurable.sum())}/{len(measurable)}")
    print(f"[density_references] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# assembling the metrics
# ---------------------------------------------------------------------------

def arm_values(matched: pd.DataFrame, recount: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (circuit, n_codes, arm) with every corrected metric."""
    rows = _density_rows(matched)
    long = []
    for arm in arms_of(matched):
        frame = rows[["size_group", "circuit_name", "num_qubits", "n_codes"]].copy()
        frame["arm"] = arm
        for quantity in ("one_qubit_gates", "two_qubit_gates", "depth", "lsr"):
            frame[quantity] = rows[f"{arm}_{quantity}"].astype(float)
        frame["tvd_raw"] = rows[f"{arm}_tvd"].astype(float)
        long.append(frame)
    table = pd.concat(long, ignore_index=True)
    key = ["circuit_name", "n_codes", "arm"]
    table = table.merge(recount[key + ["silent_used"]], on=key, how="left")
    table = table.merge(floors[key + ["floor"]], on=key, how="left")
    table["logical_error"] = table["silent_used"]
    table["tvd"] = (table["tvd_raw"] - table["floor"]).clip(lower=0.0)
    table["fraction"] = table["n_codes"] / table["num_qubits"]
    return table


def _wide(table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(circuit, n_codes) x (quantity, arm), and the per-point info, on one index."""
    quantities = ["one_qubit_gates", "two_qubit_gates", "depth", "lsr", "tvd", "tvd_raw",
                  "floor", "logical_error"]
    wide = table.pivot(index=["circuit_name", "n_codes"], columns="arm", values=quantities)
    info = (table.drop_duplicates(["circuit_name", "n_codes"])
            .set_index(["circuit_name", "n_codes"])[["size_group", "num_qubits", "fraction"]]
            .reindex(wide.index))
    return wide, info


def _improvement(value: pd.Series, reference: pd.Series, lower_is_better: bool) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        if lower_is_better:
            out = 100.0 * (reference - value) / reference
        else:
            out = 100.0 * (value - reference) / reference
    return out.where(reference > 0)


#: The ablation's stages, bottom up: (name, label, colour, arm before, arm after).
STAGES = [
    ("h_commutation", "H commutation", "#3b4c56", "baseline", "h_only"),
    ("merging", "+ Merging", "#cdc2c0", "h_only", "h_merge"),
    ("packing", "+ Packing", "#0097fd", "h_merge", "packed"),
]
#: The same with merging split: single-qubit macros (HH, XX, ZZ, a solo qubit's H as HH),
#: then two-qubit ones (CXT / CZT / XCXT and the transversal in-patch CZ / XCX).
SPLIT_STAGES = [
    ("h_commutation", "H commutation", "#3b4c56", "baseline", "h_only"),
    ("merging_1q", "+ 1q merging", "#cdc2c0", "h_only", "h_merge_1q"),
    ("merging_2q", "+ 2q merging", "#8c7f7d", "h_merge_1q", "h_merge"),
    ("packing", "+ Packing", "#0097fd", "h_merge", "packed"),
]


def stages_of(table: pd.DataFrame):
    return SPLIT_STAGES if (table["arm"] == "h_merge_1q").any() else STAGES


def ablation_segments(table: pd.DataFrame) -> pd.DataFrame:
    """Per (circuit, n_codes, quantity): stage deltas against the naive packing at that point."""
    wide, info = _wide(table)
    stages = stages_of(table)
    frames = []
    for panel, (_, quantities, lower) in PANELS.items():
        for quantity in quantities:
            base = wide[(quantity, "baseline")]
            cumulative = {arm: _improvement(wide[(quantity, arm)], base, lower)
                          for _, _, _, _, arm in stages}
            cumulative["baseline"] = base * 0.0
            frame = pd.DataFrame({name: cumulative[after] - cumulative[before]
                                  for name, _, _, before, after in stages})
            if quantity == "tvd":
                ok = (wide[("tvd_raw", "baseline")]
                      >= SIGNAL_OVER_FLOOR * wide[("floor", "baseline")])
                frame.loc[~ok.fillna(False).astype(bool)] = np.nan
            frame["panel"], frame["quantity"] = panel, quantity
            frames.append(frame.join(info).reset_index())
    return pd.concat(frames, ignore_index=True)


def reference_segments(table: pd.DataFrame, which: str) -> pd.DataFrame:
    """
    AQORN's packed arm at each density against a fixed reference -- the naive packing with
    H commutation and merging at the tightest (`which="tight"`) or sparsest (`"sparse"`)
    point -- split into patch count (naive at P vs reference) and placement (packed vs naive
    at P). Per (circuit, n_codes, quantity).
    """
    wide, info = _wide(table)
    n = info["num_qubits"]
    target = np.ceil(n / 2) if which == "tight" else n
    is_reference = wide.index.get_level_values("n_codes").to_numpy() == target.to_numpy()
    circuits = wide.index.get_level_values("circuit_name")

    def at_reference(column) -> pd.Series:
        per_circuit = wide.loc[is_reference, column].droplevel("n_codes")
        return pd.Series(circuits.map(per_circuit).to_numpy(dtype=float), index=wide.index)

    frames = []
    for panel, (_, quantities, lower) in PANELS.items():
        for quantity in quantities:
            reference = at_reference((quantity, "h_merge"))
            patch = _improvement(wide[(quantity, "h_merge")], reference, lower)
            total = _improvement(wide[(quantity, "packed")], reference, lower)
            frame = pd.DataFrame({"patch_count": patch, "placement": total - patch,
                                  "net": total})
            if quantity == "tvd":
                ok = (at_reference(("tvd_raw", "h_merge"))
                      >= SIGNAL_OVER_FLOOR * at_reference(("floor", "h_merge")))
                frame.loc[~ok.fillna(False).astype(bool)] = np.nan
            frame["panel"], frame["quantity"] = panel, quantity
            frames.append(frame.join(info).reset_index())
    return pd.concat(frames, ignore_index=True)


def _with_bins(frame: pd.DataFrame, n_bins: int = 6) -> pd.DataFrame:
    """The sweep's own density bins (`_assign_plot_bins`): six over n_codes / n in [0.5, 1]."""
    frame = frame.copy()
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    frame["plot_bin"] = pd.cut(frame["fraction"], bins=edges, labels=centers,
                               include_lowest=True).astype(float)
    return frame


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

BIN_STEP, OFFSETS, WIDTH = 2.6, np.array([-0.84, 0.0, 0.84]), 0.82


def _figure():
    import matplotlib.pyplot as plt

    from experiments.shared.figures import FULL_WIDTH, use_style

    use_style()
    fig, axes = plt.subplots(3, 2, figsize=(FULL_WIDTH, 1.55 * 3 + 1.25), sharex=True)
    return fig, axes.ravel()


def _style_axis(ax, label: str, low: float, high: float) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.set_ylabel(label)
    span = max(high - low, 1.0)
    ax.set_ylim(min(0.0, low) - 0.06 * span, max(0.0, high) + 0.10 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _finish(fig, axes, centers, handles, ylabel: str, stem: str, output_dir) -> List[Path]:
    import matplotlib.pyplot as plt

    from experiments.shared.figures import save_figure

    ticks = [i * BIN_STEP + OFFSETS[1] for i in range(len(centers))]
    for ax in axes[3:5]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{1.0 / c:.2f}" for c in centers])
        ax.xaxis.set_tick_params(labelbottom=True)
    axes[-1].set_visible(False)
    axes[-2].set_xlim(-1.32, (len(centers) - 1) * BIN_STEP + 1.32)
    # The ticks are n/P (logical qubits per patch), not the paper's packing density n/(kP),
    # which is at most 1.
    fig.supxlabel(r"logical qubits per patch, $n/\mathcal{P}$")
    fig.supylabel(ylabel)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=plt.rcParams["font.size"] * 0.9,
               columnspacing=1.2, handlelength=1.4, handletextpad=0.4, borderpad=0.0)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))
    if output_dir is None:
        paths = save_figure(fig, stem, Path(__file__), versioned=True)
    else:
        override = Path(output_dir).resolve()
        paths = save_figure(fig, stem, override / Path(__file__).name, override, versioned=True)
    plt.close(fig)
    return paths


def _cells(frame: pd.DataFrame, centers, min_circuits: int):
    """(x, group, cell) for every drawable bar."""
    for b, center in enumerate(centers):
        for g, group in enumerate(SIZE_GROUPS):
            cell = frame[(frame["plot_bin"] == center) & (frame["size_group"] == group)]
            if cell["circuit_name"].nunique() >= min_circuits:
                yield b * BIN_STEP + OFFSETS[g], group, cell


def plot_ablation(table: pd.DataFrame, output_dir, min_circuits: int,
                  suffix: str = "") -> List[Path]:
    """The matched ablation, stacked as the sweep stacks it (`stacked_percent` per quantity)."""
    import matplotlib.patches as mpatches

    from experiments.shared.figures import stacked_percent

    stages = [(name, label, colour) for name, label, colour, _, _ in stages_of(table)]
    columns = [s for s, _, _ in stages]
    segments = _with_bins(ablation_segments(table))
    centers = sorted(segments["plot_bin"].dropna().unique())
    fig, axes = _figure()
    for ax, (panel, (label, quantities, _)) in zip(axes, PANELS.items()):
        block = segments[segments["panel"] == panel].dropna(subset=columns)
        high = 1.0
        for x, group, cell in _cells(block, centers, min_circuits):
            bounds = np.mean([
                np.mean([stacked_percent(r) for r in
                         cell.loc[cell["quantity"] == q, columns].to_numpy()], axis=0)
                for q in quantities], axis=0)
            for i, (_, _, color) in enumerate(stages):
                ax.bar(x, bounds[i + 1] - bounds[i], bottom=bounds[i], width=WIDTH,
                       color=color, edgecolor="white", alpha=0.85)
            ax.text(x, bounds[-1], GROUP_INITIAL[group], ha="center", va="bottom", fontsize=5,
                    color="#444444")
            high = max(high, float(bounds[-1]))
        _style_axis(ax, label, 0.0, high)
    handles = [mpatches.Patch(color=c, alpha=0.85, label=l) for _, l, c in stages]
    stem = ("density_ablation_split" if len(stages) == 4 else "density_ablation_matched") + suffix
    return _finish(fig, axes, centers, handles, "percent improvement / reduction", stem,
                   output_dir)


#: The single-column paper version (10-11): four panels, without the logical-error panel,
#: which reads the same as TVD here (user decision).
COLUMN_PANELS = (("gates", "gates"), ("depth", "depth"), ("lsr", "LSR"), ("tvd", "TVD"))
SIZE_LETTER = {"small": "S", "medium": "M", "large": "L"}


def plot_ablation_column(table: pd.DataFrame, output_dir, min_circuits: int,
                         suffix: str = "") -> List[Path]:
    """`plot_ablation` at one column: gates, depth, LSR, TVD above floor, stacked 4 x 1.

    Stacked rather than 2 x 2 so each bar gets its own S / M / L label at 8pt: at a quarter
    column the bars sit ~4pt apart, at a full column ~9pt."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, stacked_percent, use_style

    stages = [(name, label, colour) for name, label, colour, _, _ in stages_of(table)]
    columns = [s for s, _, _ in stages]
    segments = _with_bins(ablation_segments(table))
    centers = sorted(segments["plot_bin"].dropna().unique())
    use_style()
    fig, axes = plt.subplots(4, 1, figsize=(COLUMN_WIDTH, 4.4), sharex=True)
    shrink = 0.80          # pull each bin's three bars together so the bins read as groups
    for ax, (panel, short) in zip(axes, COLUMN_PANELS):
        _, quantities, _ = PANELS[panel]
        block = segments[segments["panel"] == panel].dropna(subset=columns)
        high = 1.0
        for x, group, cell in _cells(block, centers, min_circuits):
            bounds = np.mean([
                np.mean([stacked_percent(r) for r in
                         cell.loc[cell["quantity"] == q, columns].to_numpy()], axis=0)
                for q in quantities], axis=0)
            g = SIZE_GROUPS.index(group)
            x = x - OFFSETS[g] + shrink * OFFSETS[g]
            for i, (_, _, color) in enumerate(stages):
                ax.bar(x, bounds[i + 1] - bounds[i], bottom=bounds[i], width=0.62,
                       color=color, edgecolor="white", linewidth=0.3, alpha=0.85)
            high = max(high, float(bounds[-1]))
        _style_axis(ax, short, 0.0, high)
        ax.yaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(3))
    bottom = axes[-1]
    # Two rows of x labels: S / M / L under every bar (minor ticks), n/P under each bin.
    bars = [b * BIN_STEP + shrink * OFFSETS[g] for b in range(len(centers))
            for g in range(len(SIZE_GROUPS))]
    # The middle bar of each bin sits on the bin's major tick; keep its minor tick (and its M).
    bottom.xaxis.remove_overlapping_locs = False
    bottom.set_xticks(bars, minor=True)
    bottom.set_xticklabels([SIZE_LETTER[SIZE_GROUPS[g]] for _ in centers
                            for g in range(len(SIZE_GROUPS))], minor=True, fontsize=8)
    bottom.set_xticks([b * BIN_STEP for b in range(len(centers))])
    bottom.set_xticklabels([f"{1.0 / c:.2f}" for c in centers])
    bottom.tick_params(axis="x", which="minor", length=2, pad=1)
    bottom.tick_params(axis="x", which="major", length=0, pad=12)
    bottom.set_xlim(-1.32, (len(centers) - 1) * BIN_STEP + 1.32)
    fig.supxlabel(r"logical qubits per patch, $n/\mathcal{P}$")
    fig.supylabel("improvement over naive (%)")
    handles = [mpatches.Patch(color=c, alpha=0.85, label=l) for _, l, c in stages]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, fontsize=8, columnspacing=0.8, handlelength=1.0,
               handletextpad=0.3, borderpad=0.0)
    fig.text(0.5, 0.953, "bars in each bin: S = small, M = medium, L = large circuits",
             ha="center", va="top", fontsize=8, color="#444444")
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.925))
    stem = "density_ablation_split_column" + suffix
    if output_dir is None:
        paths = save_figure(fig, stem, Path(__file__), versioned=True)
    else:
        override = Path(output_dir).resolve()
        paths = save_figure(fig, stem, override / Path(__file__).name, override, versioned=True)
    plt.close(fig)
    return paths


def plot_reference(table: pd.DataFrame, which: str, output_dir, min_circuits: int,
                   suffix: str = "") -> List[Path]:
    """AQORN at each density against a fixed naive reference, as diverging two-segment stacks."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    parts = [("patch_count", "patch count (naive at P vs reference)", "#9bd3f5"),
             ("placement", "placement (AQORN vs naive at P)", "#0097fd")]
    segments = _with_bins(reference_segments(table, which))
    centers = sorted(segments["plot_bin"].dropna().unique())
    fig, axes = _figure()
    for ax, (panel, (label, _, _)) in zip(axes, PANELS.items()):
        block = segments[segments["panel"] == panel].dropna(subset=["net"])
        low, high = 0.0, 1.0
        for x, group, cell in _cells(block, centers, min_circuits):
            # Quantities of one panel average per circuit first (1q and 2q gates).
            means = cell.groupby("circuit_name")[["patch_count", "placement", "net"]].mean().mean()
            up = down = 0.0
            for column, _, color in parts:
                value = float(means[column])
                bottom = up if value >= 0 else down
                ax.bar(x, value, bottom=bottom, width=WIDTH, color=color, edgecolor="white",
                       alpha=0.9)
                if value >= 0:
                    up += value
                else:
                    down += value
            net = float(means["net"])
            ax.plot([x - WIDTH / 2, x + WIDTH / 2], [net, net], color="black", linewidth=1.2,
                    linestyle="-", marker="", solid_capstyle="butt", zorder=5)
            ax.text(x, up, GROUP_INITIAL[group], ha="center", va="bottom", fontsize=5,
                    color="#444444")
            low, high = min(low, down, net), max(high, up, net)
        _style_axis(ax, label, low, high)
    handles = [mpatches.Patch(color=c, alpha=0.9, label=l) for _, l, c in parts]
    handles.append(mlines.Line2D([], [], color="black", linewidth=1.2, linestyle="-", marker="",
                                 label="net"))
    reference = "naive tight packing" if which == "tight" else "naive sparse packing"
    return _finish(fig, axes, centers, handles, f"percent improvement over the {reference}",
                   f"density_vs_naive_{which}{suffix}", output_dir)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per (figure, panel, density bin, segment): circuits, mean, median -- for the report."""
    out = []
    stages = tuple(name for name, *_ in stages_of(table))
    figures = [("ablation", ablation_segments(table), stages)]
    figures += [(f"vs_{which}", reference_segments(table, which),
                 ("patch_count", "placement", "net")) for which in ("tight", "sparse")]
    for name, frame, columns in figures:
        frame = _with_bins(frame)
        per_circuit = (frame.groupby(["panel", "plot_bin", "size_group", "circuit_name"])[list(columns)]
                       .mean().reset_index())
        for (panel, center), cell in per_circuit.groupby(["panel", "plot_bin"]):
            for column in columns:
                values = cell[column].dropna()
                out.append({"figure": name, "panel": panel, "density": round(1.0 / center, 2),
                            "segment": column, "circuits": int(values.size),
                            "mean": values.mean(), "median": values.median()})
    return pd.DataFrame(out)


def stem_suffix(matched: pd.DataFrame) -> str:
    """"_logical" for a sweep made under the logical optimisation pass, else "" (older CSVs)."""
    if "logical_optimization" not in matched:
        return ""
    states = set(matched["logical_optimization"].fillna("off").astype(str))
    if len(states) > 1:
        raise RuntimeError(f"the sweep mixes logical optimisation settings: {sorted(states)}")
    return "" if states == {"off"} else "_logical"


def run_plot(matched_path: Path, recount_path: Path, floors_path: Path, output_dir,
             min_circuits: int, column: bool = False) -> None:
    matched = pd.read_csv(matched_path)
    recount = pd.read_csv(recount_path)
    floors = pd.read_csv(floors_path)
    bad = recount[recount["recounted"].astype(bool) & ~recount["matches"].astype(bool)]
    if len(bad):
        raise RuntimeError(f"{len(bad)} recounted arm-points do not reproduce the sweep's own "
                           f"logical error, so the recount is not the sweep's measurement:\n"
                           f"{bad.head()}")
    table = arm_values(matched, recount, floors)
    if column:
        for item in plot_ablation_column(table, output_dir, min_circuits, stem_suffix(matched)):
            print(f"[density_references] wrote {item}")
        return
    written = []
    suffix = stem_suffix(matched)
    written += plot_ablation(table, output_dir, min_circuits, suffix)
    # The references use the +merge and packed arms alone, which every CSV here has.
    written += plot_reference(table, "tight", output_dir, min_circuits, suffix)
    written += plot_reference(table, "sparse", output_dir, min_circuits, suffix)
    path = _free(companion(matched_path, "summary"))
    summary(table).to_csv(path, index=False)
    for item in written:
        print(f"[density_references] wrote {item}")
    print(f"[density_references] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("recount", "floors", "plot"))
    parser.add_argument("--matched", type=Path, default=None,
                        help=f"the matched sweep's CSV (default: the newest {MATCHED_CSV.name})")
    parser.add_argument("--recount", type=Path, default=None,
                        help="plot: the recount table (default: the newest for --matched)")
    parser.add_argument("--floors", type=Path, default=None,
                        help="plot: the floors table (default: the newest for --matched)")
    parser.add_argument("--draws", type=int, default=100,
                        help="floors: Monte Carlo draws per arm and density point")
    parser.add_argument("--min-circuits", type=int, default=4,
                        help="plot: draw a bar only with at least this many circuits")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="plot: where the figures go (default results/figures)")
    parser.add_argument("--column", action="store_true",
                        help="plot: only the single-column 2 x 2 ablation (no logical-error panel)")
    add_jobs_argument(parser)
    args = parser.parse_args()
    matched = args.matched or _latest(MATCHED_CSV)
    if args.step == "recount":
        run_recount(matched, args.jobs)
    elif args.step == "floors":
        run_floors(matched, args.jobs, args.draws)
    else:
        run_plot(matched, args.recount or _latest(companion(matched, "recount")),
                 args.floors or _latest(companion(matched, "floors")),
                 args.output_dir, args.min_circuits, column=args.column)


if __name__ == "__main__":
    main()
