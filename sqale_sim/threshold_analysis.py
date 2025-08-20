# ------------------------------------------------------------------------ #
# -                             Simulator data                             #
# ------------------------------------------------------------------------ #
# DEFAULT_PARAMS = HilbertNoiseParams(
#     gr_static_overrotation_rads=x,
#     rz_relative_overrotation=x,
#     rz_transition_matrix=[ ... 5x5 ... ],
#     cz_transition_matrix=[ ... 5x5 ... ],
#     classifier_errors=(x, x),
#     initial_state_probs=[x, x, x, x, x],
# )

import math
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd
from dataclasses import dataclass

import cirq
import numpy as np
import experiments_superstaq as ess
from experiments_superstaq.simulation.sqale_leakage_model import HilbertNoiseParams, SqaleLeakageModel
from experiments_superstaq.simulation.leakage_sim import simulate_true_distribution, estimate_distribution

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sqale_sim.fidelity import bhattacharyya
from sqale_sim.postselection import post_select_no_leak
from iceberg.IcebergCode import IcebergGrid, BarrierGate
from permutations.circuit_translator import replay_logical_circuit
from permutations.permutation_evaluator import evaluate_permutations
from permutations.pipeline import build_physical_circuit

# ------------------------- Example circuit builder ------------------------- #

def build_logical_circuit():
    qA, qB, qC = cirq.LineQubit.range(3)
    # Clifford-only example:
    return cirq.Circuit(cirq.H(qA), cirq.CNOT(qA, qB), cirq.CNOT(qA, qC), cirq.H(qA))

    # Toffoli-like snippet (contains π/4 RZ → non-Clifford)
    return cirq.Circuit(
        cirq.H(qC),
        cirq.CNOT(qB, qC),
        cirq.rz(-math.pi / 4)(qC),
        cirq.CNOT(qA, qC),
        cirq.rz(math.pi / 4)(qC),
        cirq.CNOT(qB, qC),
        cirq.rz(math.pi / 4)(qB),
        cirq.rz(-math.pi / 4)(qC),
        cirq.CNOT(qA, qC),
        cirq.CNOT(qA, qB),
        cirq.rz(math.pi / 4)(qC),
        cirq.H(qC),
        cirq.rz(math.pi / 4)(qA),
        cirq.rz(-math.pi / 4)(qB),
        cirq.CNOT(qA, qB),
    )


# ------------------------------ Noise scaling ----------------------------- #

def scale_noise_params(
    base: HilbertNoiseParams,
    rz_scalar: float,
    cz_scalar: float,
) -> HilbertNoiseParams:
    if base.rz_transition_matrix is None or base.cz_transition_matrix is None:
        raise ValueError("Base transition matrices must be initialized.")
    """Scale only the transition matrices. Keep overrotation terms zero."""
    scaled_rz = [[x * rz_scalar for x in row] for row in base.rz_transition_matrix]
    scaled_cz = [[x * cz_scalar for x in row] for row in base.cz_transition_matrix]
    return HilbertNoiseParams(
        gr_static_overrotation_rads=0.0,
        rz_relative_overrotation=0.0,
        rz_transition_matrix=scaled_rz,
        cz_transition_matrix=scaled_cz,
        classifier_errors=base.classifier_errors,
        initial_state_probs=base.initial_state_probs,
    )

def alpha_scale_noise_params(
    base: HilbertNoiseParams,
    alpha: float,
) -> HilbertNoiseParams:
    if base.rz_transition_matrix is None or base.cz_transition_matrix is None:
        raise ValueError("Base transition matrices must be initialized.")

    base_dict = base.model_dump()

    # Scale transition matrices
    base_dict["rz_transition_matrix"] = [
        [x * alpha for x in row] for row in base.rz_transition_matrix
    ]
    base_dict["cz_transition_matrix"] = [
        [x * alpha for x in row] for row in base.cz_transition_matrix
    ]

    base_dict["gr_static_overrotation_rads"] *= alpha
    base_dict["rz_relative_overrotation"] *= alpha
    base_dict["classifier_errors"] = (base_dict["classifier_errors"][0] * alpha, base_dict["classifier_errors"][1] * alpha)

    return HilbertNoiseParams(**base_dict)

# --------------------------- Circuit utilities ---------------------------- #

def _remap_for_perm(perm: List[Tuple[int, int]], k_log: int) -> Dict[cirq.LineQubit, cirq.LineQubit]:
    return {cirq.LineQubit(i): cirq.LineQubit(p * k_log + s) for i, (p, s) in enumerate(perm)}

def _strip_barriers(circuit: cirq.Circuit) -> cirq.Circuit:
    """Remove BarrierGate operations (for readability only)."""
    return cirq.Circuit(
        (op for op in circuit.all_operations() if not isinstance(op.gate, BarrierGate)),
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

def _is_clifford_op(op: cirq.Operation) -> bool:
    """Heuristic Clifford check (good enough for warnings)."""
    g = op.gate
    if isinstance(g, cirq.MeasurementGate):
        return True

    def is_int(x: float) -> bool:
        return abs(x - round(x)) < 1e-9

    if isinstance(g, cirq.HPowGate):
        return is_int(getattr(g, "exponent", 1.0))
    if isinstance(g, cirq.CXPowGate):
        return is_int(getattr(g, "exponent", 1.0))
    if isinstance(g, cirq.CZPowGate):
        return is_int(getattr(g, "exponent", 1.0))
    if isinstance(g, cirq.SwapPowGate):
        return is_int(getattr(g, "exponent", 1.0))
    if isinstance(g, cirq.ZPowGate):
        # Clifford iff exponent is multiple of 1/2
        e = float(getattr(g, "exponent", 0.0))
        return abs(2 * e - round(2 * e)) < 1e-9
    if isinstance(g, (cirq.XPowGate, cirq.YPowGate)):
        return is_int(getattr(g, "exponent", 1.0))
    return False

def _warn_if_non_clifford(c: cirq.Circuit, *, label: str) -> None:
    non_cliff = [op for op in c.all_operations() if not _is_clifford_op(op)]
    if non_cliff:
        print(
            f"[warning] {label} contains {len(non_cliff)} non‑Clifford operation(s). "
            "Simulation may be significantly slower and memory‑intensive."
        )

def _validate_permutation(perm: List[Tuple[int,int]], n_logical: int, k_log: int) -> None:
    if len(perm) != n_logical:
        raise ValueError(f"Permutation length {len(perm)} != number of logical qubits {n_logical}")
    for i, (patch, slot) in enumerate(perm):
        if not (isinstance(patch, int) and isinstance(slot, int)):
            raise ValueError(f"Permutation entries must be ints. Got {perm[i]}")
        if slot < 0 or slot >= k_log:
            raise ValueError(f"Permutation slot out of range for k_log={k_log}: {perm[i]}")


# ------------------------------ Main runner ------------------------------- #

def run_param_sweep(
    circuit_name: str,
    logical_circuit: cirq.Circuit,
    rz_scalars: List[float],
    cz_scalars: List[float],
    *,
    faulty: bool,
    out_path: Path,
    shots: int = 1000,
    permutation: Optional[List[Tuple[int, int]]] = None,
    n_code: int = 4,   # we’re using n=4 Icebergs by default
):
    """Run a scalar sweep on a single logical circuit and one physical permutation.

    Args:
        permutation: list of (patch, slot) for each logical qubit, e.g. [(0,0),(0,1),(1,0)].
                     If None, a minimal default is chosen for the given number of logical qubits.
        n_code: Iceberg code size (default 4 → k_log = n_code - 2 = 2).
    """
    manifest = {
        "description": f"{'Faulty' if faulty else 'Fault-tolerant'} run for {circuit_name}",
        "rz_range": (min(rz_scalars), max(rz_scalars)),
        "cz_range": (min(cz_scalars), max(cz_scalars)),
        "timestamp": datetime.now().isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_params = ess.simulation.sqale_leakage_model.DEFAULT_PARAMS.model_copy()

    results = []

    # --- logical → physical mapping via permutation -----------------------
    n_logical = len(sorted(logical_circuit.all_qubits()))
    k_log = n_code - 2  # logical capacity per code for Iceberg [[n, n-2, 2]]

    if permutation is None:
        # minimal default: pack into earliest patches & slots
        permutation = [(i // k_log, i % k_log) for i in range(n_logical)]
    _validate_permutation(permutation, n_logical, k_log)

    remap = _remap_for_perm(permutation, k_log)
    phys_circ = logical_circuit.transform_qubits(lambda q: remap[q])

    phys_idxs = [q.x for q in remap.values()]
    n_prog = max(phys_idxs) + 1
    n_codes = (n_prog + k_log - 1) // k_log

    logical_to_patch = {i: i // k_log for i in range(n_prog)}
    idx_in_patch =     {i: i %  k_log for i in range(n_prog)}
    active_by_patch: Dict[int, List[int]] = {p: [] for p in range(n_codes)}
    for li in phys_idxs:
        active_by_patch[li // k_log].append(li)
    state_map = {i: "0" for i in range(n_prog)}
    fault_log: List[str] = []

    for rz in rz_scalars:
        for cz in cz_scalars:
            scaled_params = scale_noise_params(base_params, rz, cz)
            noise_model = SqaleLeakageModel(**scaled_params.model_dump())

            grid = IcebergGrid(n_program=n_prog, n=n_code, circuit=cirq.Circuit())
            grid.encode(); grid.barrier()

            replay_logical_circuit(
                grid,
                phys_circ,
                logical_to_patch=logical_to_patch,
                idx_in_patch=idx_in_patch,
                active_by_patch=active_by_patch,
                state_map=state_map,
                fault_log=fault_log,
                faulty=faulty,
                verbose=False,
            )
            grid.measure()

            clean_circuit = _strip_barriers(grid.circuit)
            _warn_if_non_clifford(clean_circuit, label="Circuit after mapping")

            clean_dist = estimate_distribution(
                clean_circuit, repetitions=shots, oversample=1, ch_form=False)

            noisy_circuit = clean_circuit.with_noise(noise_model)
            noisy_dist = estimate_distribution(
                noisy_circuit, repetitions=shots, oversample=1, ch_form=False
            )

            post_selected = post_select_no_leak(noisy_dist)
            fidelity = bhattacharyya(clean_dist, post_selected)

            results.append({
                "rz_scalar": rz,
                "cz_scalar": cz,
                "fidelity": fidelity,
                "faulty": faulty,
            })

            with open(out_path, "w") as f:
                json.dump({"_manifest": manifest, "data": results}, f, indent=2)

    return results


# def avg_fidelity_diff_scan_across_program_circuit(
#         program_circuit: cirq.Circuit, 
#         codes: List[int],
#         rz_scalars: List[float],
#         cz_scalars: List[float],
#         repetitions: int = 1_000,
#         save: bool = True,

#     ):
#     # pseudo code:
#     # dataframe here if you can do 3d dataframes, otherwise 2d dataframe update after each perm
#     # path = /results/avg_diff_across_fidelities
#     # for perm in permutations:
#         # compute physical circuit by calling replay_logical_circuit()
#         # path += /perm_to_string(perm)
#         # for for rz in rz_scalars:
#             # for cz in cz_scalars:
#                 # res_FT = simulate_effectiveness_for_perm(faulty = False)
#                 # save(res_FT) to filename: {datetime}_{perm}_R{numrepetitions}_FT.json.gz to path
#                 # res_Faulty = simulate_effectiveness_for_perm(faulty = True)
#                 # save(res_Faulty) to filename: {datetime}_{perm}_R{numrepetitions}_Faulty.json.gz to path
#                 # compute difference in fidelity
#                 # compute difference in logical selection rate
#                 # update dataframe for specific combination with differences and any other important data that isn't too much to store
#     # end result is a 3D dataframe which we can compute the average difference across all permutations and graph against the different scalar values
#     pass






# ---------- Noise model scaling ----------

def build_scaled_noise_model(
    base_params: HilbertNoiseParams, rz_scalar: float, cz_scalar: float
) -> SqaleLeakageModel:
    """Create a SQALE leakage model for a given (rz, cz) scale."""
    scaled = scale_noise_params(base_params, rz_scalar, cz_scalar)
    return SqaleLeakageModel(**scaled.model_dump())


# ---------- Permutation key & alignment ----------

def perm_key(perm: List[Tuple[int, int]]) -> str:
    """Stable, filesystem‑safe key for a permutation."""
    return "-".join(f"p{p}s{s}" for p, s in perm)

def _with_perm_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'perm_key' column if missing (simulate_effectiveness_for_perm already returns it)."""
    if "perm_key" in df.columns:
        return df
    if "permutation" in df.columns:
        return df.assign(perm_key=df["permutation"].apply(lambda p: perm_key(list(p))))
    raise ValueError("DataFrame missing both 'perm_key' and 'permutation' columns.")


# ---------- Storage helpers (optional) ----------

def _ensure_perm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee 'perm_key' and 'permutation' exist."""
    out = df.copy()
    # perm_key
    if "perm_key" not in out.columns:
        if "permutation" in out.columns:
            out["perm_key"] = out["permutation"].apply(lambda p: perm_key(list(p)))
        else:
            raise ValueError("Need either 'perm_key' or 'permutation' in evaluator results.")
    return out

def _enrich_for_save(df: pd.DataFrame, *, fidelity_col: str) -> pd.DataFrame:
    """
    Build a compact, rich view for disk:
      permutation (list[[patch,slot]]), perm_key (str),
      fidelity (float), logical_selection_rate (float), total_gates (int|None)
    """
    df = _ensure_perm_columns(df)

    # Logical selection rate: prefer overall if available, else try components, else NaN
    if "selection_rate_overall" in df.columns:
        lsr = df["selection_rate_overall"]
    elif {"selection_rate_no_leak", "selection_rate_iceberg_given_no_leak"} <= set(df.columns):
        lsr = df["selection_rate_no_leak"] * df["selection_rate_iceberg_given_no_leak"]
    else:
        lsr = np.nan

    # fidelity column normalize to 'fidelity'
    if fidelity_col not in df.columns:
        raise KeyError(f"Expected '{fidelity_col}' in DataFrame.")
    fid = df[fidelity_col]

    return pd.DataFrame({
        "perm_key": df["perm_key"],                  # stable id
        "fidelity": fid.astype(float),               # numeric
        "logical_selection_rate": lsr.astype(float), # numeric (may be NaN)
    })

@dataclass(frozen=True)
class AvgScanMeta:
    circuit_name: str
    timestamp_utc: str
    repetitions: int
    n_code: int
    codes: List[int]
    rz_range: Tuple[float, float]
    cz_range: Tuple[float, float]
    n_permutations: int

def _make_meta(
    circuit_name: str,
    repetitions: int,
    n_code: int,
    codes: List[int],
    rz_vals: List[float],
    cz_vals: List[float],
    n_perms: int,
) -> AvgScanMeta:
    return AvgScanMeta(
        circuit_name=circuit_name,
        timestamp_utc=datetime.now().isoformat(timespec="seconds") + "Z",
        repetitions=int(repetitions),
        n_code=int(n_code),
        codes=list(codes),
        rz_range=(min(rz_vals), max(rz_vals)),
        cz_range=(min(cz_vals), max(cz_vals)),
        n_permutations=int(n_perms),
    )

def _scalar_tag(x: float) -> str:
    s = f"{x}".replace(".", "p").replace("-", "m")
    return s

def _scalar_tag(val: float) -> str:
    """Convert a float to a filesystem-safe tag, e.g. 0.5 -> '0p5', -1.25 -> 'm1p25'."""
    s = f"{val}".replace("-", "m").replace(".", "p")
    return s

def avg_fidelity_diff_across_permutations(
    *,
    program_circuit: cirq.Circuit,
    codes: List[int],
    rz_scalars: List[float],
    cz_scalars: List[float],
    repetitions: int = 1_000,
    n_code: int = 4,
    base_params: Optional[HilbertNoiseParams] = None,
    save_runs: bool = True,
    circuit_name: str = "program",
    compare_against: str = "physical",
    oversample: int = 1,
    ch_form: bool = False,
) -> Dict:
    """Average Δfidelity (FT − Faulty) across permutations for each (rz, cz) point."""
    # ---- sanity checks
    q = len(sorted(program_circuit.all_qubits()))
    if len(codes) != q:
        raise ValueError(f"'codes' length {len(codes)} must equal #logical qubits {q}")
    if not all(c == n_code for c in codes):
        raise ValueError("This helper assumes uniform codes equal to n_code for all logical qubits.")

    base_params = base_params or ess.simulation.sqale_leakage_model.DEFAULT_PARAMS.model_copy()

    from permutations.permutation_generator import unique_logical_permutations, logical_capacity
    from permutations.results_manager import save_permutation_run

    perms = unique_logical_permutations(codes, q)
    _ = logical_capacity(n_code)  # not directly used

    meta = _make_meta(
        circuit_name=circuit_name,
        repetitions=repetitions,
        n_code=n_code,
        codes=codes,
        rz_vals=rz_scalars,
        cz_vals=cz_scalars,
        n_perms=len(perms),
    )

    def make_build_fn_adapter(default_n_code: int):
        def _build_fn(permutation, logical_circ, *, n_code=None, faulty=False, verbose=False):
            return build_physical_circuit(
                permutation,
                logical_circ,
                n_code=(n_code if n_code is not None else default_n_code),
                faulty=faulty,
                verbose=verbose,
            )
        return _build_fn

    grid_rows: List[Dict] = []
    save_opts = {
        "n_code": n_code,
        "compare_against": compare_against,
        "oversample": oversample,
        "ch_form": ch_form,
    }

    base_dir_root = os.path.join("results", "avg_diff_across_fidelities", circuit_name)

    for rz in sorted(rz_scalars):
        for cz in sorted(cz_scalars):
            noise_model = build_scaled_noise_model(base_params, rz, cz)
            build_fn = make_build_fn_adapter(n_code)

            # Faulty run
            df_faulty_raw: pd.DataFrame = evaluate_permutations(
                permutations=perms,
                logical_circ=program_circuit,
                build_fn=build_fn,
                n_code=n_code,
                faulty=True,
                noise_model=noise_model,
                repetitions=repetitions,
                oversample=oversample,
                ch_form=ch_form,
                post_select=True,
                compare_against=compare_against,
                verbose=False,
            )
            df_faulty_raw = _ensure_perm_columns(df_faulty_raw)

            # For merge math (keep small)
            df_faulty = df_faulty_raw[["perm_key", "fidelity"]].rename(columns={"fidelity": "fid_faulty"})

            # For saving (rich view)
            df_faulty_save = _enrich_for_save(df_faulty_raw, fidelity_col="fidelity")

            # FT run
            df_ft_raw = evaluate_permutations(
                permutations=perms,
                logical_circ=program_circuit,
                build_fn=build_fn,
                n_code=n_code,
                faulty=False,
                noise_model=noise_model,
                repetitions=repetitions,
                oversample=oversample,
                ch_form=ch_form,
                post_select=True,
                compare_against=compare_against,
                verbose=False,
            )
            df_ft_raw = _ensure_perm_columns(df_ft_raw)

            # For merge math (keep small)
            df_ft = df_ft_raw[["perm_key", "fidelity"]].rename(columns={"fidelity": "fid_ft"})

            # For saving (rich view)
            df_ft_save = _enrich_for_save(df_ft_raw, fidelity_col="fidelity")

            # Merge + delta
            df_merged = pd.merge(df_ft, df_faulty, on="perm_key", how="inner")
            if df_merged.empty:
                raise RuntimeError("No overlapping permutations between FT and Faulty runs.")

            df_merged["delta_fidelity"] = df_merged["fid_ft"] - df_merged["fid_faulty"]
            avg_delta = float(df_merged["delta_fidelity"].mean())
            n_perms   = int(df_merged.shape[0])
            print(avg_delta)
            grid_rows.append({
                "rz_scalar": float(rz),
                "cz_scalar": float(cz),
                "avg_delta_fidelity": avg_delta,
                "n_perms": n_perms,
            })

            if save_runs:
                rz_tag = _scalar_tag(rz)
                cz_tag = _scalar_tag(cz)

                scalar_folder = os.path.join(base_dir_root, f"rz{_scalar_tag(rz)}_cz{_scalar_tag(cz)}")                
                os.makedirs(scalar_folder, exist_ok=True)

                save_permutation_run(
                    circuit=program_circuit,
                    circuit_name=f"faulty_rz{rz_tag}_cz{cz_tag}",
                    df=df_faulty_save,  # <— includes permutation, perm_key, fidelity, logical_selection_rate, total_gates
                    repetitions=repetitions,
                    base_dir=scalar_folder,
                    extra_meta={"mode": "Faulty", "rz_scalar": rz, "cz_scalar": cz},
                    options=save_opts,
                    compress=False  # (optional) leave default or toggle per your debugging needs
                )
                save_permutation_run(
                    circuit=program_circuit,
                    circuit_name=f"ft_rz{rz_tag}_cz{cz_tag}",
                    df=df_ft_save,  # <— same enriched columns
                    repetitions=repetitions,
                    base_dir=scalar_folder,
                    extra_meta={"mode": "FT", "rz_scalar": rz, "cz_scalar": cz},
                    options=save_opts,
                    compress=False
                )

    return {
        "_manifest": vars(meta),
        "grid": grid_rows,
    }


if __name__ == "__main__":
    prog = build_logical_circuit()
    codes = [4, 4, 4]  # three logical qubits → three n=4 codes
    rz_vals = [0.5, 1.0, 2.0, 4.0]
    cz_vals = [0.5, 1.0, 2.0, 4.0]

    avg_result = avg_fidelity_diff_across_permutations(
        program_circuit=prog,
        codes=codes,
        rz_scalars=rz_vals,
        cz_scalars=cz_vals,
        repetitions=25,     # adjust for runtime/precision
        oversample=40,
        n_code=4,
        save_runs=True,      # stores per‑param FT/Faulty frames
        circuit_name="3QubitShorsOvernight",
        compare_against="physical",
    )

    # avg_result is a dict with "_manifest" and "grid"
    print(json.dumps(avg_result["_manifest"], indent=2))
    print("Rows:", len(avg_result["grid"]))
    print(avg_result["grid"][:3])  # peek first few rows





# # ------------------------------ Example run ------------------------------- #

# if __name__ == "__main__":
#     lc = build_logical_circuit()
#     # Example permutation for 3 logical qubits with n_code=4 (k_log=2):
#     # A,B,C → [(0,0),(0,1),(1,0)]
#     perm = [(0, 0), (1, 0), (2, 0)]

#     run_param_sweep(
#         circuit_name="test",
#         logical_circuit=lc,
#         rz_scalars=[0.5, 1.0, 2.0, 4.0],
#         cz_scalars=[0.5, 1.0, 2.0, 4.0],
#         faulty=False,
#         out_path=Path("results/test/ft_run.json"),
#         shots=1000,
#         permutation=perm,
#         n_code=4,
#     )

"""
I'd like to recreate this graph with some new code functionality we have. Please reference the following file. I believe all you will need is simulate_effective_for_perm or evaluate_permutations (if we want to average the fidelity result across every permutation). Don't use avg_fidelity_diff_across_permutations. 
"""
