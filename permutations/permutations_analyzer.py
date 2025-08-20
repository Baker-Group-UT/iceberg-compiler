# ---------------- circuit_permutations_analyzer.py --------------------
"""
High–level pipeline:

    logical_circ          (Cirq – logical gate set)
          │
    enumerate_permutations()
          │   →  list[perm]           e.g. [[(0,0),(0,1),(1,0)], …]
          ▼
    summarise_permutations()          (FAST – counts only)
          │   →  pd.DataFrame         one row per perm
          ▼
    build_physical_circuit()          (SLOW – only when you ask for it)
          →  full IcebergGrid circuit
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import math, cirq, pandas as pd
from collections import Counter

from cirq.ops.common_gates import HPowGate, CXPowGate, Rz
from cirq.ops.swap_gates   import SwapPowGate
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from permutations.circuit_translator import replay_logical_circuit
from permutations.permutation_generator import (
    unique_logical_permutations,
    logical_capacity,
)
from iceberg.IcebergCode import IcebergGrid, BarrierGate
from permutations.efficiency_finder import extremal_per_codes, add_efficiency_score
from permutations.evaluator import (
    evaluate_extremals_effectiveness,
    attach_effectiveness_columns,
)   
from experiments_superstaq.simulation.sqale_leakage_model import SqaleLeakageModel


def print_in_windows(circ: cirq.Circuit, window: int = 8):
    for t in range(0, len(circ), window):
        sub = cirq.Circuit(circ[t : t + window])   # slice by moment index
        print(f"\n# Moments {t}–{t+window-1}")
        print(sub)

# ------------------------------------------------------------------ #
#  A.  PERMUTATION ENUMERATION                                       #
# ------------------------------------------------------------------ #
def enumerate_permutations(code_sizes: List[int],
                           logical_circuit: cirq.Circuit,
                           *,
                           status_interval: float = 0.05
                           ) -> List[List[Tuple[int,int]]]:
    """Generate the **unique** permutations required for *logical_circuit*.

    `q` is inferred as `logical_circuit.num_qubits()`.
    """
    q = len(logical_circuit.all_qubits())
    return unique_logical_permutations(code_sizes, q,
                                       status_interval=status_interval)


# ────────────────────────────────────────────────────────────────────
#  helper – count physical gates   (encoding-overhead subtracted)
# ────────────────────────────────────────────────────────────────────

ENCODING_H  = 1                # one H per code during encoding
ENCODING_CNOT = 3              # three CX per code during encoding

_CANON = {                # physical-type → readable label
    HPowGate      : "H",
    CXPowGate     : "CNOT",
    Rz            : "RZ",
    SwapPowGate   : "SWAP",
    cirq.ops.measure     : "Measure",
}

_SKIP = (BarrierGate, cirq.MeasurementGate)

# ------------------------------------------------------------------
#  cheap counter minus encoding layer (unchanged)
# ------------------------------------------------------------------
def _gate_counts_minus_encoding(circ: cirq.Circuit, n_codes: int) -> Counter:
    """Return non-encoding gate counts for *circ* (Barrier/Meas skipped)."""
    enc_H  = 1 * n_codes          # 1  H per patch during encode()
    enc_CX = 3 * n_codes          # 3 CX per patch during encode()

    # ── raw physical-gate histogram, but ignore barriers & measurements
    cnt = Counter(
        type(op.gate)
        for op in circ.all_operations()
        if not isinstance(op.gate, _SKIP)          # <-- filter here
    )

    # subtract encoding overhead
    cnt[cirq.HPowGate]  -= enc_H
    cnt[cirq.CXPowGate] -= enc_CX

    # rename to canonical labels and drop zeros / unknowns
    canon = {
        cirq.HPowGate        : "H",
        cirq.CXPowGate       : "CNOT",
        cirq.ops.Rz          : "RZ",
        cirq.ops.SwapPowGate : "SWAP",
        cirq.ops.measure     : "Measure",
    }
    return Counter(
        {canon[k]: v for k, v in cnt.items() if v > 0 and k in canon}
    )
# ------------------------------------------------------------------
#  helper – build a qubit-remapping for ONE permutation
# ------------------------------------------------------------------
def _remap_for_perm(perm: List[Tuple[int,int]], k_log: int) -> Dict[cirq.LineQubit, cirq.LineQubit]:
    """
    LineQubit(i)  →  LineQubit( patch*k_log + slot )
    with *patch* and *slot* taken from *perm[i]*.
    """
    return {cirq.LineQubit(i): cirq.LineQubit(p* k_log + s)
            for i, (p, s) in enumerate(perm)}

def _perm_key(perm):
    # list[(p,s), ...]  ->  tuple of tuples for hashing/joins
    return tuple(tuple(t) for t in perm)

# ------------------------------------------------------------------
#  MAIN:  summarise_permutations
# ------------------------------------------------------------------
def summarise_permutations(perms       : List[List[Tuple[int,int]]],
                           logical_circ: cirq.Circuit,
                           *,
                           n_code : int = 4,
                           faulty : bool = False,
                           verbose: bool = False
                           ) -> pd.DataFrame:

    rows: list[dict] = []
    k_log = logical_capacity(n_code)                 # = n_code – 2

    for perm in perms:

        # ----------- build remapped logical circuit -----------------
        remap     = _remap_for_perm(perm, k_log)
        phys_circ = logical_circ.transform_qubits(lambda q: remap[q])

        phys_idxs = [q.x for q in remap.values()]
        n_prog    = max(phys_idxs) + 1
        n_codes   = (n_prog + k_log - 1) // k_log

        # ---------------- meta-tables for translator ----------------
        logical_to_patch = {i: i // k_log for i in range(n_prog)}
        idx_in_patch     = {i: i %  k_log for i in range(n_prog)}

        active_by_patch: Dict[int, List[int]] = {p: [] for p in range(n_codes)}
        for li in phys_idxs:
            active_by_patch[li // k_log].append(li)

        state_map = {i: "0" for i in range(n_prog)}
        fault_log: List[str] = []

        # -------------------- build grid & replay -------------------
        grid = IcebergGrid(n_program=n_prog, n=n_code, circuit=cirq.Circuit())
        grid.encode(); grid.barrier()

        replay_logical_circuit(grid, phys_circ,
                               logical_to_patch=logical_to_patch,
                               idx_in_patch    =idx_in_patch,
                               active_by_patch =active_by_patch,
                               state_map       =state_map,
                               fault_log       =fault_log,
                               faulty=faulty, verbose=verbose)

        # print_in_windows(grid.circuit, window=24)

        counts = _gate_counts_minus_encoding(grid.circuit, n_codes)
        
        perm_key = _perm_key(perm)

        rows.append({
            "permutation"   : perm,        # keep human-friendly
            "perm_key"      : perm_key,    # add a join-safe key
            "fault_tolerant": len(fault_log) == 0,
            "total_gates"   : sum(counts.values()),
            **counts
        })

        del grid        # drop the heavy circuit

    return (pd.DataFrame(rows)
              .fillna(0)
              .sort_values("total_gates")
              .reset_index(drop=True))

# ---------------------------------------------------------------------- #
#  C.  ON-DEMAND full circuit builder                                    #
# ---------------------------------------------------------------------- #
def build_physical_circuit(permutation : List[Tuple[int,int]],
                           logical_circ: cirq.Circuit,
                           *,
                           n_code : int = 4,
                           faulty : bool = False,
                           verbose: bool = False
                           ) -> Dict:

    k_log = logical_capacity(n_code)                # logical per code

    # -------- 1) remap the logical circuit to physical indices ---------
    remap     = _remap_for_perm(permutation, k_log)
    phys_circ = logical_circ.transform_qubits(lambda q: remap[q])

    phys_idxs = [q.x for q in remap.values()]
    n_prog    = max(phys_idxs) + 1
    n_codes   = (n_prog + k_log - 1) // k_log     # ceil

    # -------- 2) translator meta-tables --------------------------------
    l2p  = {i: i // k_log for i in range(n_prog)}
    idxp = {i: i %  k_log for i in range(n_prog)}

    active = {p: [] for p in range(n_codes)}
    for li in phys_idxs:
        active[li // k_log].append(li)

    state_map: Dict[int, str] = {i: "0" for i in range(n_prog)}
    fault_log: List[str]      = []

    # -------- 3) build grid & replay -----------------------------------
    grid = IcebergGrid(n_program=n_prog, n=n_code, circuit=cirq.Circuit())
    grid.encode(); grid.barrier()

    replay_logical_circuit(grid, phys_circ,
                           logical_to_patch = l2p,
                           idx_in_patch     = idxp,
                           active_by_patch  = active,
                           state_map        = state_map,
                           fault_log        = fault_log,
                           faulty           = faulty,
                           verbose          = verbose)

    counts = _gate_counts_minus_encoding(grid.circuit, n_codes)

    return {
        "permutation"   : permutation,
        "circuit"       : grid.circuit,
        "fault_tolerant": (len(fault_log) == 0),
        "gate_count"    : counts,
        "log"           : fault_log,
    }

import math
import cirq
from typing import Iterable, Optional, Callable, Union

# Types you already have in your codebase:
# - enumerate_permutations(codes, logical)
# - summarise_permutations(perms, logical, n_code, faulty)
# - build_physical_circuit(best_perm, logical, n_code, verbose, faulty)
# - print_in_windows(circuit, window)
# - extremal_per_codes(df, metric, only_ft, include_ties)
# - add_efficiency_score(df, lambda_codes)

def run_permutation_scan(
    logical: cirq.Circuit,
    *,
    codes: list[int] = None,
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
) -> dict:
    """
    One‑stop scan for a single logical circuit.

    Args:
        logical: The logical circuit body to map.
        codes:  List of code sizes for each logical qubit (default: [n_code]*num_qubits).
        n_code: Size of each Iceberg code (e.g., 4 for [[4,2,2]]).
        faulty: If True, allow faulty gates in physical synthesis.
        metric: Column name in the summary DF to pick extremals on (e.g., 'total_gates').
        only_ft: Filter to only fault‑tolerant rows when extracting extremals.
        include_ties: Keep ties when extracting extremals.
        lambda_codes: Weight for composite efficiency score.
        top_k: How many top permutations to build (sorted by `metric` ascending).
        verbose: Print brief status and build logs.
        show_best_windows: If set, pretty‑print best circuit in windows of this width.
        state_prep: Optional circuit to prepend (e.g., to prepare a specific input state).
        post_circuit: Optional circuit to append (e.g., measurements).

    Returns:
        {
          "df": summary DataFrame,
          "best_rows": DataFrame of the top_k rows,
          "best_builds": list of dicts from build_physical_circuit for each best row,
          "extremals": dict from extremal_per_codes(df, metric, ...),
          "df_scored": df with 'efficiency_score',
          "extremals_scored": dict from extremal_per_codes(df_scored, 'efficiency_score', ...),
        }
    """
    # Compose the logical body with optional pre/post
    full_logical = cirq.Circuit()
    if state_prep is not None:
        full_logical += state_prep
    full_logical += logical
    if post_circuit is not None:
        full_logical += post_circuit

    # Default codes vector if not supplied
    if codes is None:
        # Infer number of logical qubits from the circuit's qubits
        num_logical = len(sorted(full_logical.all_qubits()))
        codes = [n_code] * num_logical

    if verbose:
        print(f"[scan] n_code={n_code}, codes={codes}, faulty={faulty}")
        print(f"[scan] logical qubits: {sorted(full_logical.all_qubits())}")
        print(f"[scan] building permutations…")

    perms = enumerate_permutations(codes, full_logical)

    if verbose:
        print(f"[scan] summarising {len(perms)} permutations…")

    df = summarise_permutations(perms, full_logical, n_code=n_code, faulty=faulty)

    if df.empty:
        raise RuntimeError("summarise_permutations returned an empty DataFrame.")


    # Evaluate only the best & worst (per code-count bucket) on total_gates
    eff_df = evaluate_extremals_effectiveness(
        df, logical,
        build_fn=build_physical_circuit,   # <— pass the builder to avoid cycles
        n_code=4,
        faulty=True,
        metric="total_gates",
        only_ft=only_ft,
        include_ties=True,
        repetitions=10_000,
        oversample=2,
        ch_form=False,
        post_select=True,
    )

    # Merge fidelity into your summary (others remain NaN)
    df_with_eff = attach_effectiveness_columns(df, eff_df)

    print(eff_df.sort_values(["codes_used", "logical_error_rate"]))
    print(df_with_eff.head(20))

    # Sort by the chosen metric (assumed 'lower is better')
    df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
    best_rows = df_sorted.iloc[:max(1, top_k)].copy()

    if verbose:
        print("[scan] head of summary:\n", df_sorted.head(min(10, len(df_sorted))))
        print(f"[scan] selected top_k={top_k} rows by '{metric}'.")

    # Build the best few
    best_builds = []
    for i, row in best_rows.iterrows():
        perm = row["permutation"]
        if verbose:
            print(f"\n[build] top#{i} permutation = {perm}")
        variant = build_physical_circuit(
            perm, full_logical, n_code=n_code, verbose=verbose, faulty=faulty
        )
        best_builds.append(variant)
        if verbose:
            print("[build] gate counts =", variant.get("gate_count"))
            if show_best_windows:
                print_in_windows(variant["circuit"], window=show_best_windows)

    # Extremals on base metric
    if verbose:
        print(f"\n[extremals] by metric='{metric}' (only_ft={only_ft}, include_ties={include_ties})")
    extremals = extremal_per_codes(
        df, metric=metric, only_ft=only_ft, include_ties=include_ties
    )
    if verbose:
        print("Best per codes:\n", extremals["best_by_codes"])
        print("Worst per codes:\n", extremals["worst_by_codes"])

    # Scored version
    df_scored = add_efficiency_score(df, lambda_codes=lambda_codes)
    if verbose:
        print(f"\n[score] added efficiency_score (lambda_codes={lambda_codes})")
    extremals_scored = extremal_per_codes(
        df_scored, metric="efficiency_score", only_ft=only_ft, include_ties=include_ties
    )
    if verbose:
        print("Score-best per codes:\n", extremals_scored["best_by_codes"])

    return {
        "df": df,
        "best_rows": best_rows,
        "best_builds": best_builds,
        "extremals": extremals,
        "df_scored": df_scored,
        "extremals_scored": extremals_scored,
        "eff_df": eff_df,          
        "df_with_eff": df_with_eff,
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
) -> list[dict]:
    """
    Batch version for multiple state/circuit combos.

    Each item is a dict with:
      {
        "logical": cirq.Circuit,             # required
        "state_prep": Optional[cirq.Circuit],
        "post_circuit": Optional[cirq.Circuit],
        "codes": Optional[list[int]],
        "label": Optional[str],              # for display
      }

    Returns: list of results (same structure as run_permutation_scan + 'label').
    """
    results = []
    for idx, it in enumerate(items):
        label = it.get("label", f"item#{idx}")
        if verbose:
            print(f"\n========== Batch: {label} ==========")
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
        )
        res["label"] = label
        results.append(res)
    return results

def _expand_gate_counts(df: pd.DataFrame, src: str = "gate_count") -> pd.DataFrame:
    """Expand {'H':..., 'CNOT':..., 'RZ':..., 'SWAP':...} dict column into flat cols."""
    if src not in df.columns:
        return df.copy()
    out = df.copy()
    # Safely expand dict-like rows
    gate_df = out[src].apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series())
    for c in ["H","CNOT","RZ","SWAP"]:
        if c not in gate_df.columns:
            gate_df[c] = 0
    out = pd.concat([out.drop(columns=[src]), gate_df], axis=1)
    # Add/refresh total_gates if needed
    if "total_gates" not in out.columns:
        out["total_gates"] = out[["H","CNOT","RZ","SWAP"]].sum(axis=1)
    return out

# ---------------- Smoke test ----------------
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
        faulty=True,                # allow swaps / transversals as needed
        metric="total_gates",
        only_ft=False,
        include_ties=True,
        lambda_codes=2.0,
        top_k=1,
        verbose=True,
        show_best_windows=48,
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
