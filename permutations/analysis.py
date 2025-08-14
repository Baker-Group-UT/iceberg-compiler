# analysis.py
from __future__ import annotations
from typing import Iterable, Optional, List, Dict, Any
import cirq, pandas as pd
import sys, os

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from permutations.pipeline import (
    enumerate_permutations,
    summarise_permutations,
    build_physical_circuit,
    print_in_windows,
)
from permutations.efficiency_finder import extremal_per_codes, add_efficiency_score
from permutations.permutation_evaluator import (
    evaluate_extremals_effectiveness,
    attach_effectiveness_columns,
    evaluate_permutations,
)
from permutations.results_manager import save_permutation_run

__all__ = ["run_permutation_scan", "run_batch_scans", "_expand_gate_counts"]

pd.set_option('display.max_rows', None)

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compose_logical(
    body: cirq.Circuit,
    state_prep: Optional[cirq.Circuit],
    post_circuit: Optional[cirq.Circuit],
) -> cirq.Circuit:
    c = cirq.Circuit()
    if state_prep is not None:
        c += state_prep
    c += body
    if post_circuit is not None:
        c += post_circuit
    return c

def _default_codes(full_logical: cirq.Circuit, codes: Optional[List[int]], n_code: int) -> List[int]:
    if codes is not None:
        return codes
    n = len(sorted(full_logical.all_qubits()))
    return [n_code] * n

def _perm_key_local(perm):
    """Normalize permutation to a hashable tuple-of-tuples (local helper)."""
    if isinstance(perm, (list, tuple)):
        return tuple(tuple(t) for t in perm)
    return perm

def _vprint(flag: bool, *args, **kwargs) -> None:
    if flag:
        print(*args, **kwargs)

def _print_effectiveness_table(eff_df: pd.DataFrame) -> None:
    if eff_df.empty:
        return
    # Friendly alias for display
    tmp = eff_df.rename(columns={
        "selection_rate_iceberg_given_no_leak": "logical_selection_rate"
    })
    cols = [c for c in [
        "permutation", "codes_used",
        "fidelity", "logical_error_rate",
        "logical_selection_rate",           # NEW: show logical selection rate
        "kept_shots_after_leak",            # optional counts
        "kept_shots_overall", "total_shots",
        "post_selected", "gate_count", "total_gates",
    ] if c in tmp.columns]
    print("\n=== Effectiveness (fidelity) on extremal permutations ===")
    print(tmp.sort_values(["codes_used", "logical_error_rate"])[cols])

def _build_top_k(
    df_sorted: pd.DataFrame,
    full_logical: cirq.Circuit,
    n_code: int,
    faulty: bool,
    top_k: int,
    verbose: bool,
    show_best_windows: Optional[int],
) -> List[Dict[str, Any]]:
    best_rows = df_sorted.iloc[:max(1, top_k)].copy()
    builds: List[Dict[str, Any]] = []
    for i, row in best_rows.iterrows():
        perm = row["permutation"]
        _vprint(verbose, f"\n[build] top#{i} permutation = {perm}")
        variant = build_physical_circuit(perm, full_logical, n_code=n_code, verbose=verbose, faulty=faulty)
        builds.append(variant)
        if verbose and show_best_windows:
            print("[build] gate counts =", variant.get("gate_count"))
            print_in_windows(variant["circuit"], window=show_best_windows)
    return builds

def _merge_selection_metrics_into_summary(df_with_eff: pd.DataFrame, eff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the logical selection rate (and related counts) are present in the
    merged summary, even if attach_effectiveness_columns doesn't include them yet.
    """
    if eff_df.empty:
        return df_with_eff

    right = eff_df.copy()
    # Build join keys on both sides
    if "perm_key" not in right.columns:
        right["perm_key"] = right["permutation"].map(_perm_key_local)
    left = df_with_eff.copy()
    if "perm_key" not in left.columns:
        left["perm_key"] = left["permutation"].map(_perm_key_local)

    take = [c for c in [
        "perm_key",
        "selection_rate_iceberg_given_no_leak",
        "selection_rate_no_leak",
        "selection_rate_overall",
        "kept_shots_after_leak",
        "kept_shots_overall",
        "total_shots",
    ] if c in right.columns]

    merged = left.merge(right[take], on="perm_key", how="left")
    # Friendly alias for downstream use/printing
    merged = merged.rename(columns={
        "selection_rate_iceberg_given_no_leak": "logical_selection_rate"
    })
    return merged.drop(columns=["perm_key"], errors="ignore")

def _print_summary_with_effectiveness(df_with_eff: pd.DataFrame, *, label: str, head_n: int = 100) -> None:
    """Pretty-print the 'Summary DF with effectiveness columns' for one circuit."""
    if df_with_eff.empty:
        print(f"\n=== {label}: Summary DF with effectiveness columns (no rows) ===")
        return

    df = df_with_eff.copy()
    df["circuit"] = label  # tag rows with the circuit name

    keep_cols = [
        "circuit",
        "permutation", "fault_tolerant", "total_gates",
        "H", "CNOT", "RZ", "SWAP",
        "fidelity", "logical_error_rate", "post_selected",
        "logical_selection_rate",            # NEW: show in summary table
        "kept_shots_after_leak", "kept_shots_overall", "total_shots",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    print(f"\n=== {label}: Summary DF with effectiveness columns (first {head_n}) ===")
    print(
        df.sort_values(["total_gates", "logical_error_rate"], ascending=[True, True])[
            keep_cols
        ].head(head_n)
    )
    print("\n\n")

def _expand_gate_counts(df: pd.DataFrame, src: str = "gate_count") -> pd.DataFrame:
    """Expand {'H':..., 'CNOT':..., 'RZ':..., 'SWAP':...} dict column into flat cols."""
    if src not in df.columns:
        return df.copy()
    out = df.copy()
    gate_df = out[src].apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series())
    for c in ["H", "CNOT", "RZ", "SWAP"]:
        if c not in gate_df.columns:
            gate_df[c] = 0
    out = pd.concat([out.drop(columns=[src]), gate_df], axis=1)
    if "total_gates" not in out.columns:
        out["total_gates"] = out[["H", "CNOT", "RZ", "SWAP"]].sum(axis=1)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_permutation_scan(
    logical: cirq.Circuit,
    *,
    codes: Optional[List[int]] = None,
    n_code: int = 4,
    faulty: bool = True,
    metric: str = "total_gates",
    only_ft: bool = False,
    include_ties: bool = True,
    lambda_codes: float = 2.0,
    top_k: int = 1,
    verbose: bool = True,
    show_best_windows: Optional[int] = 32,
    state_prep: Optional[cirq.Circuit] = None,
    post_circuit: Optional[cirq.Circuit] = None,
    repetitions: int = 2_000,
    # results manager parameters
    save_results: bool = False,
    circuit_name: Optional[str] = None,
    save_dir: str = "results/permutations",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> dict:
    # Compose the full logical circuit and codes vector
    full_logical = _compose_logical(logical, state_prep, post_circuit)
    codes = _default_codes(full_logical, codes, n_code)

    _vprint(verbose, f"[scan] n_code={n_code}, codes={codes}, faulty={faulty}")
    _vprint(verbose, f"[scan] logical qubits: {sorted(full_logical.all_qubits())}")
    _vprint(verbose, "[scan] building permutations…")

    # Fast summary over all permutations
    perms = enumerate_permutations(codes, full_logical)
    _vprint(verbose, f"[scan] summarising {len(perms)} permutations…\n")
    df = summarise_permutations(perms, full_logical, n_code=n_code, faulty=faulty)
    if df.empty:
        raise RuntimeError("summarise_permutations returned an empty DataFrame.")

    # Effectiveness only on extremals (keeps runtime reasonable)
    eff_df = evaluate_extremals_effectiveness(
        df, logical,
        build_fn=build_physical_circuit,
        n_code=n_code,
        faulty=faulty,
        metric="total_gates",
        only_ft=only_ft,
        include_ties=include_ties,
        repetitions=repetitions,
        oversample=2,
        ch_form=False,
        post_select=True,
        verbose=verbose
    )
    # Merge fidelity/etc.
    df_with_eff = attach_effectiveness_columns(df, eff_df)
    # Ensure logical_selection_rate and counts are present in the merged summary
    df_with_eff = _merge_selection_metrics_into_summary(df_with_eff, eff_df)

    # Sort/choose top-k by the requested metric
    df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
    _vprint(verbose, "[scan] head of summary:\n", df_sorted.head(min(10, len(df_sorted))))
    _vprint(verbose, f"[scan] selected top_k={top_k} rows by '{metric}'.")
    _print_effectiveness_table(eff_df)

    # Build top-k physical circuits for inspection
    best_builds = _build_top_k(
        df_sorted, full_logical, n_code, faulty, top_k, verbose, show_best_windows
    )
    best_rows = df_sorted.iloc[:max(1, top_k)].copy()

    # Extremals on chosen and scored metrics
    _vprint(verbose, f"\n[extremals] by metric='{metric}' (only_ft={only_ft}, include_ties={include_ties})")
    extremals = extremal_per_codes(df, metric=metric, only_ft=only_ft, include_ties=include_ties)

    df_scored = add_efficiency_score(df, lambda_codes=lambda_codes)
    extremals_scored = extremal_per_codes(
        df_scored, metric="efficiency_score", only_ft=only_ft, include_ties=include_ties
    )

    if save_results:
        save_permutation_run(
            circuit=_compose_logical(logical, state_prep, post_circuit),  # program/full logical
            circuit_name=circuit_name or "circuit",
            df=df_with_eff,
            repetitions=repetitions,
            base_dir=save_dir,
            extra_meta=extra_meta or {},
            options={
                "n_code": n_code,
                "faulty": faulty,
                "metric": metric,
                "only_ft": only_ft,
                "include_ties": include_ties,
                "lambda_codes": lambda_codes,
                "top_k": top_k,
            },
        )

    return {
        "df": df,
        "best_rows": best_rows,
        "best_builds": best_builds,
        "extremals": extremals,
        "df_scored": df_scored,
        "extremals_scored": extremals_scored,
        "eff_df": eff_df,
        "df_with_eff": df_with_eff,   # <- carries logical_selection_rate
    }

def run_batch_scans(
    items: Iterable[dict],
    *,
    n_code: int = 4,
    faulty: bool = True,
    metric: str = "total_gates",
    only_ft: bool = True,
    include_ties: bool = True,
    lambda_codes: float = 2.0,
    top_k: int = 1,
    verbose: bool = True,
    repetitions: int = 2_000,
    summary_head: int = 100,
    # results manager params
    save_results: bool = False,
    save_dir: str = "results/permutations",
) -> List[dict]:
    """
    For each circuit item (with keys {'label','logical', 'codes'? ...}),
    run the scan and print a 'Summary DF with effectiveness columns' that
    includes the circuit label. Returns the list of per-circuit result dicts.
    """
    results: List[dict] = []
    for idx, it in enumerate(items):
        label = it.get("label", f"item#{idx}")
        _vprint(verbose, f"\n========== Batch: {label} ==========")

        res = run_permutation_scan(
            it["logical"],
            codes=it.get("codes"),
            n_code=n_code,
            faulty=faulty,
            metric=metric,
            only_ft=only_ft,
            include_ties=include_ties,
            lambda_codes=lambda_codes,
            top_k=top_k,
            verbose=verbose,
            state_prep=it.get("state_prep"),
            post_circuit=it.get("post_circuit"),
            repetitions=repetitions,
            # saving
            save_results=save_results,
            circuit_name=label,
            save_dir=save_dir,
            extra_meta={"item_index": idx},
        )
        res["label"] = label
        results.append(res)

        # Print the per-circuit effectiveness summary with the circuit name
        _print_summary_with_effectiveness(res["df_with_eff"], label=label, head_n=summary_head)

    return results

def run_all_permutation_effectiveness(
    logical: cirq.Circuit,
    *,
    codes: Optional[List[int]] = None,
    n_code: int = 4,
    faulty: bool = True,
    repetitions: int = 2_000,
    oversample: int = 2,
    ch_form: bool = False,
    post_select: bool = True,
    compare_against: str = "physical",   # or "program"
    verbose: bool = False,
    # optional program wrappers (so we can save the exact program circuit)
    state_prep: Optional[cirq.Circuit] = None,
    post_circuit: Optional[cirq.Circuit] = None,
    # saving options
    save_results: bool = False,
    circuit_name: Optional[str] = None,
    save_dir: str = "results/permutations",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Enumerate *all* permutations for `logical` and simulate the logical error rate
    (and selection rates) for each permutation. Optionally saves the run.
    Returns a tidy DataFrame from evaluate_permutations.
    """
    # Compose full program circuit (what actually defines the program)
    full_logical = _compose_logical(logical, state_prep, post_circuit)
    codes = _default_codes(full_logical, codes, n_code)

    # 1) enumerate all perms for this program
    perms = enumerate_permutations(codes, full_logical)

    # 2) evaluate every permutation
    eff_all = evaluate_permutations(
        permutations=perms,
        logical_circ=logical,                 # baseline program for "program" compare mode
        build_fn=build_physical_circuit,
        n_code=n_code,
        faulty=faulty,
        noise_model=None,                     # default evaluator noise model
        repetitions=repetitions,
        oversample=oversample,
        ch_form=ch_form,
        post_select=post_select,
        compare_against=compare_against,
        verbose=verbose,
    )

    # Optional: nice print for quick review
    if not eff_all.empty:
        pretty = eff_all.rename(columns={
            "selection_rate_iceberg_given_no_leak": "logical_selection_rate"
        })
        cols = [c for c in [
            "permutation", "codes_used",
            "fidelity", "logical_error_rate",
            "logical_selection_rate",
            "kept_shots_after_leak", "kept_shots_overall", "total_shots",
            "total_gates"
        ] if c in pretty.columns]
        print("\n=== All permutations – effectiveness summary ===")
        print(pretty.sort_values(["logical_error_rate", "codes_used", "total_gates"])[cols])

    # 3) save (program circuit + full results dataframe + metadata)
    if save_results:
        save_permutation_run(
            circuit=full_logical,                             # exact program circuit used
            circuit_name=circuit_name or "circuit",
            df=eff_all,                                       # per-permutation effectiveness (incl. selection rates)
            repetitions=repetitions,
            base_dir=save_dir,
            extra_meta=(extra_meta or {}),
            options={
                "mode": "all_permutations",
                "n_code": n_code,
                "faulty": faulty,
                "oversample": oversample,
                "ch_form": ch_form,
                "post_select": post_select,
                "compare_against": compare_against,
                "num_permutations": len(perms),
            },
        )

    return eff_all


from permutations.example_circuits import square_cluster_4, grover_3, ghz_3, half_adder, ripple_inc_3, toffoli_like_snippet




# def build_logical_circuit():
#     qA, qB, qC = cirq.LineQubit.range(3)
#     return cirq.Circuit(
#         cirq.H(qA),
#         cirq.CNOT(qA, qB),
#         cirq.CNOT(qA, qC),
#         cirq.H(qA),
#     )

# if __name__ == "__main__":
#     logical = build_logical_circuit()
#     # n=4 codes for 3 logical qubits:
#     df_all = run_all_permutation_effectiveness(
#         logical,
#         codes=[4, 4, 4],
#         n_code=4,
#         faulty=True,
#         repetitions=2_000,
#         post_select=True,
#         compare_against="physical",  # or "program"
#         verbose=False,
#         save_results=True,
#         circuit_name="unscalable_shors_3q"
#     )


# "unscalable_shor_3": shor_toy_3,
#     "half_adder": half_adder,
#     "ripple_incrementer_3": ripple_inc_3,
#     "": toffoli_like_snippet,

# if __name__ == "__main__":
#     items = [
#         {"label": "ghz_3",                      "logical": ghz_3()["logical"],                      "codes": [4,4,4]},
#         {"label": "square_cluster_4",           "logical": square_cluster_4()["logical"],           "codes": [4,4,4,4]},
#         {"label": "grover_3",                   "logical": grover_3()["logical"],                   "codes": [4,4,4]},
#         {"label": "half_adder",                 "logical": half_adder()["logical"],                 "codes": [4,4,4,4]},
#         {"label": "ripple_incrementer_3",       "logical": ripple_inc_3()["logical"],               "codes": [4,4,4]},
#         {"label": "toffoli",                    "logical": toffoli_like_snippet()["logical"],       "codes": [4,4,4]},
        
#     ]

#     # Will print the named “Summary DF with effectiveness columns …” for each item:
#     batch = run_batch_scans(items, repetitions=1_000, verbose=False, summary_head=100, save_results=True)


# ------------------------------- TESTING ------------------------------- #

def build_logical_circuit():
    qA, qB, qC = cirq.LineQubit.range(3)
    # Clifford-only example: H_A → CX_A→B → CX_A→C → H_A
    return cirq.Circuit(
        cirq.H(qA),
        cirq.CNOT(qA, qB),
        cirq.CNOT(qA, qC),
        cirq.H(qA),
    )


if __name__ == "__main__":
    print("\n========== Smoke test: run_permutation_scan() ==========")
    logical = build_logical_circuit()

    # Use 3 codes of size n=4 (capacity 2 each) for 3 logical qubits
    res = run_permutation_scan(
        logical=logical,
        codes=[4, 4, 4],
        n_code=4,
        faulty=False,                # allow swaps / transversals as needed
        metric="total_gates",
        only_ft=False,
        include_ties=True,
        lambda_codes=2.0,
        top_k=1,
        verbose=False,
        show_best_windows=48,
        repetitions=1_000
    )

    # Show the single best (by total_gates) from the summary DF
    best_rows = res["best_rows"]
    print("\n=== Best permutation by 'total_gates' ===")
    cols = ["permutation", "fault_tolerant", "total_gates", "H", "CNOT", "RZ", "SWAP"]
    print(best_rows[cols] if all(c in best_rows.columns for c in cols) else best_rows)

    # Use results computed inside run_permutation_scan
    eff_df = res["eff_df"]
    df_with_eff = res["df_with_eff"]

    print("\n=== Effectiveness (fidelity) on extremal permutations ===")
    print(eff_df.sort_values(["codes_used", "logical_error_rate"]))

    print("\n=== Summary DF with effectiveness columns (first 10) ===")
    keep_cols = [c for c in ["permutation","fault_tolerant","total_gates","H","CNOT","RZ","SWAP",
                            "fidelity","logical_error_rate","post_selected"] if c in df_with_eff.columns]
    print(df_with_eff.sort_values(["total_gates","logical_error_rate"], ascending=[True, True])
                    [keep_cols].head(100))

    # Print the best physical circuit nicely
    best_perm = best_rows.iloc[0]["permutation"]
    variant = build_physical_circuit(best_perm, logical, n_code=4, verbose=False, faulty=True)
    print("\n=== Best physical circuit ===")
    print_in_windows(variant["circuit"], window=48)


# # ------------------------------------------------------------------ #
# #  D.  Tiny demo / smoke-test                                        #
# # ------------------------------------------------------------------ #
# if __name__ == "__main__":

#     # ---------- logical circuit (a Toffoli-like snippet) ------------
#     qA, qB, qC = cirq.LineQubit.range(3)

#     logical = cirq.Circuit(
#         cirq.H(qC),                        # 0)  H_C
#         cirq.CNOT(qB, qC),                 # 1)  CNOT_B→C
#         cirq.rz(-math.pi / 4)(qC),         # 2)  RZ(-π/4)_C
#         cirq.CNOT(qA, qC),                 # 3)  CNOT_A→C
#         cirq.rz(math.pi / 4)(qC),          # 4)  RZ(π/4)_C
#         cirq.CNOT(qB, qC),                 # 5)  CNOT_B→C
#         cirq.rz(math.pi / 4)(qB),          # 6)  RZ(π/4)_B
#         cirq.rz(-math.pi / 4)(qC),         # 7)  RZ(-π/4)_C
#         cirq.CNOT(qA, qC),                 # 8)  CNOT_A→C
#         cirq.CNOT(qA, qB),                 # 9)  CNOT_A→B
#         cirq.rz(math.pi / 4)(qC),          # 10) RZ(π/4)_C
#         cirq.H(qC),                        # 11) H_C
#         cirq.rz(math.pi / 4)(qA),          # 12) RZ(π/4)_A
#         cirq.rz(-math.pi / 4)(qB),         # 13) RZ(-π/4)_B
#         cirq.CNOT(qA, qB),                 # 14) CNOT_A→B
#         # optional final measurement (comment out if not wanted)
#         # cirq.measure(qA, qB, qC, key="m")
#     )

#     # ---------- enumerate permutations for 3 logical qubits ---------
#     codes = [4,4,4] # three n=4 Iceberg codes for 3 logical states in circuit
#     perms = enumerate_permutations(codes, logical)
#     faulty = True
#     # ---------- fast scan -------------------------------------------
#     df = summarise_permutations(perms, logical, n_code=4, faulty=faulty)
#     print(df.head(100))

#     # ---------- build the very best permutation ---------------------
#     best = df.iloc[0]["permutation"]
#     print(best)
#     variant = build_physical_circuit(best, logical, n_code=4, verbose=True, faulty=faulty)
#     print("\nBest permutation =", best)
#     print("Gate counts      =", variant["gate_count"])
#     print_in_windows(variant["circuit"], window=24)


#     # After df = summarise_permutations(...)
#     ext = extremal_per_codes(df, metric="total_gates", only_ft=True, include_ties=True)
#     print("Best per codes:\n", ext["best_by_codes"])
#     print("Worst per codes:\n", ext["worst_by_codes"])

#     # If you want to try a composite metric:
#     df_scored = add_efficiency_score(df, lambda_codes=2.0)
#     ext2 = extremal_per_codes(df_scored, metric="efficiency_score")
#     print("Score-best per codes:\n", ext2["best_by_codes"])
