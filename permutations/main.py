# main.py
import math
import argparse
import cirq

# --- adjust these imports to your package structure --------------------------
# If you put run_permutation_scan in driver.py as in the previous message:
# If you need to step through each permutation, we also call these directly:
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from permutations.permutations_analyzer import (
    enumerate_permutations,
    summarise_permutations,
    build_physical_circuit,
    print_in_windows,
    run_permutation_scan,
    extremal_per_codes,
)
import pandas as pd
# -----------------------------------------------------------------------------

# ------------------------- circuit factory fns -------------------------------

def square_cluster_4():
    qA, qB, qC, qD = cirq.LineQubit.range(4)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.H(qB),                        # 1)  H_B
        cirq.H(qC),                        # 2)  H_C
        cirq.H(qD),                        # 3)  H_D
        cirq.CZ(qA, qB),                   # 4)  CZ_A,B
        cirq.CZ(qB, qD),                   # 5)  CZ_B,D
        cirq.CZ(qD, qC),                   # 6)  CZ_D,C
        cirq.CZ(qC, qA),                   # 7)  CZ_C,A
    )
    return {"label": "square_cluster_4", "logical": logical, "codes": [4,4,4,4]}

def grover_3():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.H(qB),                        # 1)  H_B
        cirq.H(qC),                        # 2)  H_C
        cirq.H(qC),                        # 3)  H_C
        cirq.CCX(qA, qB, qC),              # 4)  CCX_{A,B->C}
        cirq.H(qC),                        # 5)  H_C
        cirq.H(qA),                        # 6)  H_A
        cirq.H(qB),                        # 7)  H_B
        cirq.H(qC),                        # 8)  H_C
        cirq.X(qA),                        # 9)  X_A
        cirq.X(qB),                        # 10) X_B
        cirq.X(qC),                        # 11) X_C
        cirq.H(qC),                        # 12) H_C
        cirq.CCX(qA, qB, qC),              # 13) CCX_{A,B->C}
        cirq.H(qC),                        # 14) H_C
        cirq.X(qA),                        # 15) X_A
        cirq.X(qB),                        # 16) X_B
        cirq.X(qC),                        # 17) X_C
        cirq.H(qA),                        # 18) H_A
        cirq.H(qB),                        # 19) H_B
        cirq.H(qC),                        # 20) H_C
    )
    return {"label": "grover_3", "logical": logical, "codes": [4,4,4]}

# def random_bad_circuit_7_qubits():
#     qA, qB, qC, qD, qE, qF = cirq.LineQubit.range(6)
#     logical = cirq.Circuit(
#         cirq.H(qA),
#         cirq.CNOT(qA, qB),
#         cirq.CZ(qB, qC),
#         # cirq.CCX(qA, qB, qC),
#         cirq.rz(math.pi/4)(qD),
#         cirq.H(qE),
#         cirq.CNOT(qE, qF),
#         cirq.CZ(qF, qA),
#         cirq.CCX(qD, qE, qF),
#         cirq.rz(-math.pi/4)(qA),
#         cirq.H(qB),
#         cirq.CNOT(qC, qD),
#         cirq.CZ(qD, qE),
#         # cirq.CCX(qE, qF, qB),
#         cirq.rz(math.pi/4)(qC),
#         cirq.H(qC),
#         cirq.CNOT(qA, qD),
#         cirq.CZ(qB, qF),
#         # cirq.CCX(qA, qE, qF),
#         cirq.rz(-math.pi/4)(qF),
#     )
#     return {"label": "random_bad_circuit_7_qubits", "logical": logical, "codes": [4,4,4,4,4,4]}

def ghz_3():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.CNOT(qA, qB),                 # 1)  CNOT_A→B
        cirq.CNOT(qA, qC),                 # 2)  CNOT_A→C
    )
    return {"label": "ghz_3", "logical": logical, "codes": [4,4,4]}

def shor_toy_3():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qA),                        # 0)  H_A
        cirq.CNOT(qA, qB),                 # 1)  CNOT_A→B
        cirq.CNOT(qA, qC),                 # 2)  CNOT_A→C
        cirq.H(qA),                        # 3)  H_A
    )
    return {"label": "unscalable_shor_3", "logical": logical, "codes": [4,4,4]}

def half_adder():
    qA, qB, qS, qC = cirq.LineQubit.range(4)  # A,B,S(|0>),C(|0>)
    logical = cirq.Circuit(
        cirq.CNOT(qA, qS),                 # 0)  CNOT_A→S
        cirq.CNOT(qB, qS),                 # 1)  CNOT_B→S
        cirq.CCX(qA, qB, qC),              # 2)  CCX_{A,B->C}
    )
    return {"label": "half_adder", "logical": logical, "codes": [4,4,4,4]}

def ripple_inc_3():
    q0, q1, q2, c1, c2 = cirq.LineQubit.range(5)
    logical = cirq.Circuit(
        cirq.CNOT(q0, c1),                 # 0
        cirq.CNOT(c1, q0),                 # 1
        cirq.CCX(q0, c1, q1),              # 2

        cirq.CNOT(q1, c2),                 # 3
        cirq.CNOT(c2, q1),                 # 4
        cirq.CCX(q1, c2, q2),              # 5

        cirq.CNOT(c2, q1),                 # 6
        cirq.CNOT(q1, c2),                 # 7
        cirq.CCX(q0, c1, q1),              # 8
        cirq.CNOT(c1, q0),                 # 9
        cirq.CNOT(q0, c1),                 # 10
    )
    return {"label": "ripple_incrementer_3", "logical": logical, "codes": [4,4,4,4,4]}

def toffoli_like_snippet():
    qA, qB, qC = cirq.LineQubit.range(3)
    logical = cirq.Circuit(
        cirq.H(qC),                        # 0)  H_C
        cirq.CNOT(qB, qC),                 # 1)  CNOT_B→C
        cirq.rz(-math.pi / 4)(qC),         # 2)  RZ(-π/4)_C
        cirq.CNOT(qA, qC),                 # 3)  CNOT_A→C
        cirq.rz(math.pi / 4)(qC),          # 4)  RZ(π/4)_C
        cirq.CNOT(qB, qC),                 # 5)  CNOT_B→C
        cirq.rz(math.pi / 4)(qB),          # 6)  RZ(π/4)_B
        cirq.rz(-math.pi / 4)(qC),         # 7)  RZ(-π/4)_C
        cirq.CNOT(qA, qC),                 # 8)  CNOT_A→C
        cirq.CNOT(qA, qB),                 # 9)  CNOT_A→B
        cirq.rz(math.pi / 4)(qC),          # 10) RZ(π/4)_C
        cirq.H(qC),                        # 11) H_C
        cirq.rz(math.pi / 4)(qA),          # 12) RZ(π/4)_A
        cirq.rz(-math.pi / 4)(qB),         # 13) RZ(-π/4)_B
        cirq.CNOT(qA, qB),                 # 14) CNOT_A→B
    )
    return {"label": "toffoli_like_snippet", "logical": logical, "codes": [4,4,4]}

ALL_FACTORIES = {
    "square_cluster_4": square_cluster_4,
    "grover_3": grover_3,
    # "random_bad_circuit_7_qubits": random_bad_circuit_7_qubits,
    "ghz_3": ghz_3,
    "unscalable_shor_3": shor_toy_3,
    "half_adder": half_adder,
    "ripple_incrementer_3": ripple_inc_3,
    "toffoli_like_snippet": toffoli_like_snippet,
}

# ----------------------------- runner ----------------------------------------

def step_through_permutations(item, *, n_code: int, faulty: bool, metric: str,
                              pause: bool, windows: int, lambda_codes: float):
    """
    Run search, then pause only on the extremal (best & worst) permutation
    for each *codes shape* using extremal_per_codes.
    """
    label  = item["label"]
    logical= item["logical"]
    codes  = item.get("codes", [n_code] * len(sorted(logical.all_qubits())))

    print(f"\n========== {label} ==========")
    print(f"[cfg] n_code={n_code}  faulty={faulty}  metric='{metric}'  codes={codes}")

    # Enumerate + summarize, so metrics/columns match the pipeline
    perms = enumerate_permutations(codes, logical)
    print(f"[info] enumerated {len(perms)} permutations.")

    df = summarise_permutations(perms, logical, n_code=n_code, faulty=faulty)
    if df.empty:
        print("[warn] summary is empty; skipping.")
        return

    # Ensure metric exists
    if metric not in df.columns:
        print(f"[warn] metric '{metric}' not found in df.columns = {list(df.columns)}; defaulting to 'total_gates'")
        metric = "total_gates"

    # Compute extremals per codes-shape (your helper returns dict of DFs)
    ext = extremal_per_codes(df, metric=metric, only_ft=True, include_ties=True)
    best_df  = ext.get("best_by_codes",  pd.DataFrame())
    worst_df = ext.get("worst_by_codes", pd.DataFrame())

    if best_df.empty and worst_df.empty:
        print("[warn] no extremals returned; skipping.")
        return

    # Determine all shapes we have results for
    key_col = "codes_shape" if "codes_shape" in df.columns else "codes"
    shapes = sorted(set(best_df.get(key_col, pd.Series(dtype=str))).union(
                    set(worst_df.get(key_col, pd.Series(dtype=str)))))

    print("\n[extremals] stepping through best & worst per codes-shape:")
    print("  shapes:", ", ".join(map(str, shapes)))

    # Helper for printing & pausing on one permutation row
    def _show_perm(row, tag):
        perm = row["permutation"]
        show_cols = [c for c in [
            "total_gates","two_qubit_gates","single_qubit_gates",
            "ft_only","codes_shape","depth", metric
        ] if c in row.index]

        print("\n" + "-"*80)
        print(f"[{tag}] {row.get(key_col, 'shape=?')}  →  permutation: {perm}")
        print("[metrics]", {c: row[c] for c in show_cols})

        variant = build_physical_circuit(perm, logical, n_code=n_code, verbose=False, faulty=faulty)
        print("[gate_count]", variant.get("gate_count"))
        print_in_windows(variant["circuit"], window=windows)

        if pause:
            try:
                input("Press Enter to continue (Ctrl+C to quit)… ")
            except KeyboardInterrupt:
                print("\n[exit] interrupted by user.")
                raise SystemExit

    # For each codes-shape, show BEST then WORST (skip duplicates)
    for shape in shapes:
        # pick first row per shape (include_ties=True ⇒ may have several; we just take the first)
        best_row  = best_df[best_df[key_col] == shape].head(1)
        worst_row = worst_df[worst_df[key_col] == shape].head(1)

        if not best_row.empty:
            _show_perm(best_row.iloc[0], tag="BEST")

        # Avoid repeating if the same perm is both best & worst (degenerate case)
        if not worst_row.empty:
            if best_row.empty or (worst_row.iloc[0]["permutation"] != best_row.iloc[0]["permutation"]):
                _show_perm(worst_row.iloc[0], tag="WORST")

    # Nice-to-have: print the tables once at the end for logging
    print("\n[best_by_codes]")
    print(best_df)
    print("\n[worst_by_codes]")
    print(worst_df)

def parse_args():
    parser = argparse.ArgumentParser(description="Run permutation searches and step through each permutation.")
    parser.add_argument("--circuits", nargs="*", default=list(ALL_FACTORIES.keys()),
                        help=f"Subset to run (default: all). Options: {', '.join(ALL_FACTORIES.keys())}")
    parser.add_argument("--n-code", type=int, default=4, help="Iceberg code size (default: 4).")
    parser.add_argument("--faulty", action="store_true", help="Allow faulty gates in physical synthesis.")
    parser.add_argument("--metric", type=str, default="total_gates", help="Sort metric for stepping (default: total_gates).")
    parser.add_argument("--no-pause", action="store_true", help="Disable Enter-to-continue pauses.")
    parser.add_argument("--windows", type=int, default=32, help="Print window width for circuits (default: 32).")
    parser.add_argument("--lambda-codes", type=float, default=2.0, help="Weight for efficiency_score (default: 2.0).")
    return parser.parse_args()

def main():
    args = parse_args()

    # Build the selected circuits
    items = []
    for name in args.circuits:
        if name not in ALL_FACTORIES:
            print(f"[warn] unknown circuit '{name}' – skipping.")
            continue
        items.append(ALL_FACTORIES[name]())

    if not items:
        print("[error] no valid circuits selected.")
        return

    for item in items:
        step_through_permutations(
            item,
            n_code=args.n_code,
            faulty=False,
            metric=args.metric,
            pause=(not args.no_pause),
            windows=args.windows,
            lambda_codes=args.lambda_codes,
        )

if __name__ == "__main__":
    main()