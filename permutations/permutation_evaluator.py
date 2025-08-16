# permutations/permutation_evaluator.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, Iterable
import cirq
import pandas as pd
import numpy as np

# experiments-superstaq imports
import experiments_superstaq as ess
from experiments_superstaq.simulation.sqale_leakage_model import (
    SqaleLeakageModel,
    DEFAULT_PARAMS,
)

# Simulator utilities
from sqale_sim.postselection import (
    post_select_no_leak,
    iceberg_post_select,
    ordered_measurement_keys,
    iceberg_kept_mass
)
from sqale_sim.fidelity import bhattacharyya

# Iceberg utility
from iceberg.IcebergCode import BarrierGate

from permutations.efficiency_finder import extremal_per_codes

# ──────────────────────────────────────────────────────────────────────
# Basic utilities
# ──────────────────────────────────────────────────────────────────────

def _strip_barriers(circuit: cirq.Circuit) -> cirq.Circuit:
    """Remove BarrierGate operations (for readability only)."""
    return cirq.Circuit(
        (op for op in circuit.all_operations() if not isinstance(op.gate, BarrierGate)),
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

def _ensure_measured(c: cirq.Circuit, *, key: str = "m") -> cirq.Circuit:
    if any(isinstance(op.gate, cirq.MeasurementGate) for op in c.all_operations()):
        return c
    qs = sorted(c.all_qubits())
    return c if not qs else c + cirq.Circuit(cirq.measure(*qs, key=key))

def _estimate_distribution(
    circuit: cirq.Circuit,
    *,
    repetitions: int,
    oversample: int,
    ch_form: bool,
) -> Dict[str, float]:
    return ess.simulation.leakage_sim.estimate_distribution(
        circuit,
        repetitions=repetitions,
        oversample=oversample,
        ch_form=ch_form,
    )

def _params_to_kwargs(params) -> dict:
    """Support Pydantic v2 (.model_dump) and v1 (.dict) for DEFAULT_PARAMS."""
    if hasattr(params, "model_dump"):
        kw = params.model_dump()
    elif hasattr(params, "dict"):
        kw = params.dict()
    else:
        kw = dict(params)  # last resort

    # Ensure list-like, not numpy scalars—SqaleLeakageModel accepts array-like.
    for k in ("cz_transition_matrix", "rz_transition_matrix", "initial_state_probs"):
        if k in kw and kw[k] is not None:
            kw[k] = np.asarray(kw[k]).tolist()
    return kw

def _default_noise_model() -> SqaleLeakageModel:
    kw = _params_to_kwargs(DEFAULT_PARAMS)
    # Optional guard: set dimension from initial_state_probs if not provided
    if "dimension" not in kw and "initial_state_probs" in kw and kw["initial_state_probs"] is not None:
        kw["dimension"] = len(kw["initial_state_probs"])
    return SqaleLeakageModel(**kw)

def _perm_key(perm):
    """Normalize list/tuple permutation to a hashable tuple-of-tuples key."""
    if isinstance(perm, (list, tuple)):
        return tuple(tuple(t) for t in perm)
    return perm  # already a key


# ──────────────────────────────────────────────────────────────────────
# Post-selection & distribution helpers
# ──────────────────────────────────────────────────────────────────────

def _get_key_order_and_codes(circ: cirq.Circuit) -> tuple[list[str], int]:
    """
    Return (key_order, codes_used) where key_order is the simulator's terminal
    measurement key order, and codes_used is the number of data_* registers
    (Iceberg patches used).
    """
    ko = ordered_measurement_keys(circ)
    return ko, sum(1 for k in ko if str(k).startswith("data_"))

def _simulate_raw(
    circ: cirq.Circuit,
    noise_model: Optional[cirq.NoiseModel],
    *,
    repetitions: int,
    oversample: int,
    ch_form: bool,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Return (ideal_raw, noisy_raw) distributions for the *same* circuit."""
    ideal_raw = _estimate_distribution(
        circ, repetitions=repetitions, oversample=oversample, ch_form=ch_form
    )
    noisy_raw = _estimate_distribution(
        circ if noise_model is None else circ.with_noise(noise_model),
        repetitions=repetitions, oversample=oversample, ch_form=ch_form,
    )
    return ideal_raw, noisy_raw

def _apply_ps(
    dist: Dict[str, float],
    *,
    post_select: bool,
    key_order: List[str],
) -> Dict[str, float]:
    """Apply (leakage→drop, then Iceberg projection) post-selection if requested."""
    if not post_select:
        return dist
    no_leak = post_select_no_leak(dist)
    return iceberg_post_select(no_leak, key_order=key_order)

def simulate_with_ps(
    circ: cirq.Circuit,
    noise_model: Optional[cirq.NoiseModel],
    *,
    repetitions: int,
    oversample: int,
    ch_form: bool,
    post_select: bool,
    logical_post_select: bool = True,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[str], int]:
    """
    Simulate *circ* both ideal and noisy, optionally apply post-selection to both.
    Returns (ideal_used, noisy_raw, noisy_used, key_order, codes_used).
    """
    stripped = _strip_barriers(_ensure_measured(circ))
    key_order, codes_used = _get_key_order_and_codes(stripped)
    ideal_raw, noisy_raw = _simulate_raw(
        stripped, noise_model,
        repetitions=repetitions, oversample=oversample, ch_form=ch_form,
    )
    if logical_post_select:
        ideal_used = _apply_ps(ideal_raw, post_select=post_select, key_order=key_order)
        noisy_used = _apply_ps(noisy_raw, post_select=post_select, key_order=key_order)
    else: 
        ideal_used = ideal_raw
        noisy_used = noisy_raw
    return ideal_used, noisy_raw, noisy_used, key_order, codes_used

def _project_iceberg_to_program_dist(
    postselected_dist: Dict[str, float],
    permutation: List[Tuple[int, int]],
    *,
    codes_used: int,
) -> Dict[str, float]:
    """
    Map Iceberg post-selected bitstrings (2 bits per patch) to program-qubit order
    using the permutation [(patch, slot), ...] where slot∈{0,1} for n_code=4.
    The current Iceberg PS returns per-patch strings in patch-index order and then
    reverses the final logical string; we invert that here.
    """
    if not postselected_dist:
        return {}

    prog_len = len(permutation)
    out: Dict[str, float] = {}

    for s, p in postselected_dist.items():
        s_rev = s[::-1]  # undo final reverse
        chunks = [s_rev[2*i:2*(i+1)] for i in range(codes_used)]  # 2 bits per patch
        if any(len(ch) < 2 for ch in chunks):
            return {}
        bits = ["0"] * prog_len
        for i, (patch, slot) in enumerate(permutation):
            if patch < 0 or patch >= len(chunks) or slot not in (0, 1):
                return {}
            bits[i] = chunks[patch][slot]
        prog_key = "".join(bits)
        out[prog_key] = out.get(prog_key, 0.0) + float(p)

    Z = sum(out.values()) or 1.0
    return {k: v / Z for k, v in out.items()}

def _safe_bhattacharyya(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Robust wrapper in case either distribution is empty."""
    if not a or not b:
        return 0.0
    return float(bhattacharyya(a, b))


# ──────────────────────────────────────────────────────────────────────
# Public: simulation for a single permutation
# ──────────────────────────────────────────────────────────────────────

def simulate_effectiveness_for_perm(
    permutation: List[Tuple[int, int]],
    logical_circ: cirq.Circuit,
    *,
    build_fn: Callable[[List[Tuple[int,int]], cirq.Circuit], Dict],
    n_code: int = 4,
    faulty: bool = True,
    noise_model: Optional[cirq.NoiseModel] = None,
    repetitions: int = 10_000,
    oversample: int = 2,
    ch_form: bool = False,
    post_select: bool = True,
    compare_against: str = "physical",  # "physical" | "program"
    verbose: bool = False,
) -> Dict:
    """
    Evaluate one permutation.

    compare_against:
      - "physical" (default): Ideal *physical* vs Noisy *physical*, under identical PS.
      - "program": Ideal *program* vs post-selected *physical* projected back to program order.
    """
    if noise_model is None:
        noise_model = _default_noise_model()
    if verbose:
        print(f"\n<======== perm {permutation} ========>")

    # 1) Build the physical circuit (translator should add decode()+named measures)
    variant = build_fn(permutation, logical_circ, n_code=n_code, faulty=faulty, verbose=False)
    phys_circ = variant["circuit"]

    # 2) Simulate (ideal & noisy) with optional PS
    ideal_used_phys, noisy_raw_phys, noisy_used_phys, key_order, codes_used = simulate_with_ps(
        phys_circ, noise_model,
        repetitions=repetitions, oversample=oversample, ch_form=ch_form,
        post_select=post_select,
    )

    # 2a) Selection rate diagnostics (computed from the *raw* noisy distribution)
    #     - leakage pass fraction: mass with no '2'
    #     - iceberg pass (given no-leak): additional mass that survives ancilla/data checks
    #     - overall selection fraction: product of the two (equivalently: kept mass directly)
    sel_no_leak = sum(v for k, v in noisy_raw_phys.items() if "2" not in k)
    # build an unnormalized, leakage-filtered dict for Iceberg acceptance calc
    leak_filtered_raw = {k: v for k, v in noisy_raw_phys.items() if "2" not in k}
    sel_iceberg_given_no_leak = (
        iceberg_kept_mass(leak_filtered_raw, key_order) / sel_no_leak
        if sel_no_leak > 0 else 0.0
    )
    selection_rate_overall = sel_no_leak * sel_iceberg_given_no_leak
    kept_shots = int(round(selection_rate_overall * repetitions))
    kept_shots_after_leak = int(round(sel_no_leak * repetitions))  # just for reference

    # 3) Choose comparison baseline & compute fidelity
    if compare_against == "program":
        ideal_prog = _estimate_distribution(
            _ensure_measured(logical_circ),
            repetitions=min(repetitions, 1_000),
            oversample=oversample, ch_form=ch_form,
        )
        proj_noisy_prog = _project_iceberg_to_program_dist(
            noisy_used_phys, permutation, codes_used=codes_used
        )
        fid = _safe_bhattacharyya(ideal_prog, proj_noisy_prog)
        ideal_used_for_return = ideal_prog
        post_selected_for_return = proj_noisy_prog
    else:
        fid = _safe_bhattacharyya(ideal_used_phys, noisy_used_phys)
        ideal_used_for_return = ideal_used_phys
        post_selected_for_return = noisy_used_phys

    # 4) Bookkeeping
    permkey     = _perm_key(permutation)
    gate_count  = variant.get("gate_count", {})
    total_gates = int(sum(gate_count.values())) if gate_count else None

    if verbose:
        print(_strip_barriers(_ensure_measured(phys_circ)))
        print("ideal_used:", ideal_used_for_return)
        print("noisy_raw :", noisy_raw_phys)
        print("noisy_used:", post_selected_for_return)
        print("selection_rate_no_leak:", sel_no_leak)
        print("selection_rate_iceberg_given_no_leak:", sel_iceberg_given_no_leak)
        print("selection_rate_overall:", selection_rate_overall)
        print("Bhattacharyya Fidelity:", fid, "\n")

    return {
        "permutation"        : permutation,
        "perm_key"           : permkey,
        "codes_used"         : codes_used,
        "fidelity"           : float(fid),
        "logical_error_rate" : float(1.0 - fid),
        "post_selected"      : bool(post_select),

        # Distributions (for transparency / downstream analysis)
        "ideal_dist"         : ideal_used_for_return,     # baseline used for fidelity
        "noisy_dist"         : noisy_raw_phys,            # raw noisy physical (pre-PS)
        "post_selected_dist" : post_selected_for_return,  # noisy after PS (or projected)

        # NEW: selection/acceptance metrics
        "selection_rate_no_leak"              : float(sel_no_leak),
        "selection_rate_iceberg_given_no_leak": float(sel_iceberg_given_no_leak),
        "selection_rate_overall"              : float(selection_rate_overall),
        "kept_shots_overall"                  : int(kept_shots),
        "kept_shots_after_leak"               : int(kept_shots_after_leak),
        "total_shots"                         : int(repetitions),

        # Gate stats / flags
        "gate_count"         : gate_count,
        "total_gates"        : total_gates,
        "fault_tolerant"     : variant.get("fault_tolerant", True),
    }


# ──────────────────────────────────────────────────────────────────────
# Public: evaluate a list of permutations (general) + extremals wrapper
# ──────────────────────────────────────────────────────────────────────

def evaluate_permutations(
    permutations: Iterable[List[Tuple[int, int]]],
    logical_circ: cirq.Circuit,
    *,
    build_fn: Callable[[List[Tuple[int,int]], cirq.Circuit], Dict],
    n_code: int = 4,
    faulty: bool = True,
    noise_model: Optional[cirq.NoiseModel] = None,
    repetitions: int = 10_000,
    oversample: int = 2,
    ch_form: bool = False,
    post_select: bool = True,
    compare_against: str = "physical",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Simulate effectiveness for *every* permutation in `permutations` and
    return a tidy DataFrame of results.
    """
    seen: set[Tuple[Tuple[int,int], ...]] = set()
    uniq: list[List[Tuple[int,int]]] = []
    for p in permutations:
        t = _perm_key(p)
        if t not in seen:
            seen.add(t)
            uniq.append(p)

    rows = [
        simulate_effectiveness_for_perm(
            perm, logical_circ,
            build_fn=build_fn, n_code=n_code, faulty=faulty,
            noise_model=noise_model, repetitions=repetitions,
            oversample=oversample, ch_form=ch_form, post_select=post_select,
            compare_against=compare_against, verbose=verbose,
        )
        for perm in uniq
    ]
    return pd.DataFrame(rows)

def evaluate_extremals_effectiveness(
    df_summary: pd.DataFrame,
    logical_circ: cirq.Circuit,
    *,
    build_fn: Callable[[List[Tuple[int,int]], cirq.Circuit], Dict],
    n_code: int = 4,
    faulty: bool = True,
    metric: str = "total_gates",
    only_ft: bool = True,
    include_ties: bool = True,
    noise_model: Optional[cirq.NoiseModel] = None,
    repetitions: int = 10_000,
    oversample: int = 2,
    ch_form: bool = False,
    post_select: bool = True,
    compare_against: str = "physical",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Simulate best & worst permutations per codes-used bucket (extremals).
    Thin wrapper over `evaluate_permutations`.
    """
    if "permutation" not in df_summary.columns:
        raise ValueError("df_summary must contain a 'permutation' column")

    ex = extremal_per_codes(df_summary, metric=metric, only_ft=only_ft, include_ties=include_ties)

    perms: List[List[Tuple[int,int]]] = []
    for key in ("best_by_codes", "worst_by_codes"):
        if key in ex and not ex[key].empty:
            perms.extend(ex[key]["permutation"].tolist())

    return evaluate_permutations(
        perms, logical_circ,
        build_fn=build_fn, n_code=n_code, faulty=faulty,
        noise_model=noise_model, repetitions=repetitions,
        oversample=oversample, ch_form=ch_form, post_select=post_select,
        compare_against=compare_against, verbose=verbose,
    )


# ──────────────────────────────────────────────────────────────────────
# Join helper
# ──────────────────────────────────────────────────────────────────────

def attach_effectiveness_columns(df_summary: pd.DataFrame, eff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join effectiveness results (fidelity, logical_error_rate, codes_used, …)
    onto the summary by a normalized permutation key.
    """
    left  = df_summary.copy()
    right = eff_df.copy()

    # Ensure join-safe keys on both sides
    if "perm_key" not in left.columns:
        left["perm_key"]  = left["permutation"].map(_perm_key)
    else:
        left["perm_key"] = left["perm_key"].map(_perm_key)

    if "perm_key" not in right.columns:
        right["perm_key"] = right["permutation"].map(_perm_key)
    else:
        right["perm_key"] = right["perm_key"].map(_perm_key)

    cols = [c for c in [
        "perm_key", "codes_used", "fidelity", "logical_error_rate", "post_selected"
    ] if c in right.columns]

    merged = left.merge(right[cols], on="perm_key", how="left", suffixes=("", "_eff"))
    return merged.drop(columns=["perm_key"])