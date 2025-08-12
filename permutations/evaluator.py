# permutations/effectiveness_estimator.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import cirq
import pandas as pd
import numpy as np

# sqale_sim imports – adjust if your layout differs
import sqale_sim as ess
from sqale_sim.postselection import post_select_no_leak, iceberg_post_select, ordered_measurement_keys
from sqale_sim.fidelity import bhattacharyya
import experiments_superstaq as ess
from experiments_superstaq.simulation.sqale_leakage_model import SqaleLeakageModel, DEFAULT_PARAMS

# --- cleaning helpers -------------------------------------------------
from iceberg.IcebergCode import BarrierGate
import cirq

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

def _codes_used_from_perm(perm: List[Tuple[int,int]]) -> int:
    return len({p for p, _ in perm})

def _estimate_distribution(
    circuit: cirq.Circuit,
    *,
    repetitions: int,
    oversample: int,
    ch_form: bool,
) -> dict:
    return ess.simulation.leakage_sim.estimate_distribution(
        circuit,
        repetitions=repetitions,
        oversample=oversample,
        ch_form=ch_form,
    )

def _params_to_kwargs(params) -> dict:
    """Support Pydantic v2 (.model_dump) and v1 (.dict)."""
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
) -> Dict:
    """Build with build_fn, simulate ideal/noisy, compute fidelity."""
    # Build the default model lazily if none was provided
    if noise_model is None:
        noise_model = _default_noise_model()

    variant   = build_fn(permutation, logical_circ, n_code=n_code, faulty=faulty, verbose=False)
    phys_circ = _ensure_measured(variant["circuit"])
    print(phys_circ)

    sim_circ = _strip_barriers(phys_circ)
    print(sim_circ)
    ideal_dist = _estimate_distribution(sim_circ,
        repetitions=repetitions, oversample=oversample, ch_form=ch_form)

    noisy_circuit = sim_circ if noise_model is None else sim_circ.with_noise(noise_model)
    noisy_dist = _estimate_distribution(noisy_circuit,
        repetitions=repetitions, oversample=oversample, ch_form=ch_form)

    used_dist = post_select_no_leak(noisy_dist) if post_select else noisy_dist
    # add iceberg post selection here
    iceberg_post_select(dist=used_dist, key_order=ordered_measurement_keys(circuit=sim_circ))

    fid = bhattacharyya(ideal_dist, used_dist)
    print(fid)
    perm_key = _perm_key(permutation)

    return {
        "permutation"        : permutation,
        "perm_key"           : perm_key,
        "codes_used"         : _codes_used_from_perm(permutation),
        "fidelity"           : float(fid),
        "logical_error_rate" : float(1.0 - fid),
        "post_selected"      : bool(post_select),
        "ideal_dist"         : ideal_dist,
        "noisy_dist"         : noisy_dist,
        "post_selected_dist" : used_dist,
        "gate_count"         : variant.get("gate_count", {}),
        "fault_tolerant"     : variant.get("fault_tolerant", True),
    }

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
) -> pd.DataFrame:
    """Simulate best & worst per codes-used bucket and return tidy DF."""
    if "permutation" not in df_summary.columns:
        raise ValueError("df_summary must contain a 'permutation' column")

    # Late import to avoid cycles
    from permutations.efficiency_finder import extremal_per_codes

    ex = extremal_per_codes(df_summary, metric=metric, only_ft=only_ft, include_ties=include_ties)

    perms: List[List[Tuple[int,int]]] = []
    for key in ("best_by_codes", "worst_by_codes"):
        if key in ex and not ex[key].empty:
            perms.extend(ex[key]["permutation"].tolist())

    seen, uniq = set(), []
    for p in perms:
        t = tuple(p)
        if t not in seen:
            seen.add(t); uniq.append(p)

    rows = []
    for perm in uniq:
        rows.append(simulate_effectiveness_for_perm(
            perm, logical_circ,
            build_fn=build_fn, n_code=n_code, faulty=faulty,
            noise_model=noise_model, repetitions=repetitions,
            oversample=oversample, ch_form=ch_form, post_select=post_select
        ))
    return pd.DataFrame(rows)

def _perm_key(perm):
    # Accept list/tuple and normalize:
    if isinstance(perm, (list, tuple)):
        return tuple(tuple(t) for t in perm)
    return perm  # already a key


def attach_effectiveness_columns(df_summary: pd.DataFrame,
                                 eff_df: pd.DataFrame) -> pd.DataFrame:
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

    # Choose columns that might exist on eff_df
    cols = [c for c in ["perm_key", "codes_used",
                        "fidelity", "post_selected_fidelity",
                        "logical_error_rate"] if c in right.columns]

    merged = left.merge(right[cols], on="perm_key", how="left", suffixes=("", "_eff"))
    return merged.drop(columns=["perm_key"])