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

import cirq
import numpy as np
import experiments_superstaq as ess
from experiments_superstaq.simulation.sqale_leakage_model import HilbertNoiseParams, SqaleLeakageModel
from experiments_superstaq.simulation.leakage_sim import simulate_true_distribution, estimate_distribution
from postselection import post_select_no_leak
from fidelity import bhattacharyya

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from iceberg.IcebergCode import IcebergGrid, BarrierGate
from permutations.circuit_translator import replay_logical_circuit


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


# ------------------------------ Example run ------------------------------- #

if __name__ == "__main__":
    lc = build_logical_circuit()
    # Example permutation for 3 logical qubits with n_code=4 (k_log=2):
    # A,B,C → [(0,0),(0,1),(1,0)]
    perm = [(0, 0), (1, 0), (2, 0)]

    run_param_sweep(
        circuit_name="test",
        logical_circuit=lc,
        rz_scalars=[0.5, 1.0, 2.0, 4.0],
        cz_scalars=[0.5, 1.0, 2.0, 4.0],
        faulty=False,
        out_path=Path("results/test/ft_run.json"),
        shots=1000,
        permutation=perm,
        n_code=4,
    )