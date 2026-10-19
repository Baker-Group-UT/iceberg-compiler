"""
logical_optimizer.py -- the strongest logical optimiser Qiskit 2.2.1 offers on its own, and the
checks that its output computes what its input did.

Moved here from `experiments/logical_opt/optimizer.py` (which re-exports it) when a round of
logical optimisation became the compiler's first pass (`translator.logical_optimization`,
2026-10-09). The code below is the experiment's, verbatim.

The question it serves: are the Fig 6 benchmark circuits already near-optimal *logical*
circuits, so that AQORN's gains cannot be the clean-up of sloppy inputs? PyZX, BQSKit and
pytket are not installed and nothing may be installed, so the optimiser is Qiskit only:

  o3              `transpile(optimization_level=3)` into the corpus basis, best of
                  `O3_TRIALS` calls (O3 is not deterministic), re-run on its own output until
                  the score stops improving (O3 already loops internally to a minimum; the
                  outer loop also re-runs its init stage)
  o3_sx           the same into the translator's full single-qubit set (corpus basis + sx)
  clifford_ag     Clifford circuits only: the whole circuit as a `Clifford` tableau,
  clifford_greedy resynthesised (Aaronson-Gottesman / greedy), then O3 into the basis
  clifford_blocks every maximal Clifford block (by matrix, so rz(pi/2) counts) resynthesised
                  greedily, then O3 -- the structurally different route for mixed circuits
  linear_blocks   every CX block resynthesised by Patel-Markov-Hayes, then O3
  commute         1q runs commuted through 2q gates and resynthesised, commutative inverse
                  cancellation, then O3

`optimize` alternates: each round runs every route on the current circuit and keeps the best
candidate if it beats the current one, until a round improves nothing. The O3-only chain is
kept as well (it is what a skeptic would run), and the result is the best of the original,
the O3 chain and the alternating chain. Nothing worse than the input is ever returned: when
no route improves the score, the original circuit object itself comes back (`changed` False),
barriers and all, so downstream stages can reuse its results instead of resimulating.

Score: lexicographic (two-qubit gates, total gates, depth), logical counts, barriers and
measurements excluded. Two-qubit count first is O3's own priority (`ConsolidateBlocks` /
`UnitarySynthesis` trade single-qubit gates for fewer two-qubit ones), so the route choice
agrees with the optimiser rather than overruling it.

Target gate set. `translator.gate_handlers` dispatches h, s, sx, x, z, rz, rx, cx, cz and a
program swap. The corpus (`create_proxy_from_qasm`) is transpiled into
h, x, z, cx, cz, s, rz, rx. Swap is excluded on purpose: the translator implements a program
swap as a relabel, but only inside a patch -- a cross-patch swap is rejected and the whole
placement with it -- so an optimiser that emitted swaps would make placements infeasible.

Equivalence (`check_equivalence`), up to global phase, against the original with its
barriers removed:
    n <= 12                 `Operator` equality (exact unitary)
    Clifford in and out     `Clifford` tableau equality (exact, any width)
    otherwise               miter amplitude |<psi| U_opt^dag U_orig |psi>| = 1 for |0...0> and
                            two Haar-random product states, Aer statevector (n <= 20) or
                            matrix-product-state (wider), single-threaded
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: What Fig 6's loader transpiles every QASM file into (`create_proxy_from_qasm`).
CORPUS_BASIS: Tuple[str, ...] = ("h", "x", "z", "s", "rz", "rx", "cx", "cz")
#: Every single- and two-qubit gate `translator.gate_handlers` dispatches, minus the program
#: swap (see the module docstring).
TRANSLATOR_BASIS: Tuple[str, ...] = CORPUS_BASIS + ("sx",)
_DIRECTIVES = {"barrier", "measure"}
_CLIFFORD_NAMES = {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "id"}
#: Fig 6's proxy keeps these and maps rz/rx to s (`create_proxy_from_qasm`); sx is added
#: because the o3_sx route may emit it and the translator accepts it.
_PROXY_KEEP = {"h", "x", "z", "s", "sx", "cx", "cz", "barrier"}

Score = Tuple[int, int, int]


# ----------------------------------------------------------------------------------------
# counting
# ----------------------------------------------------------------------------------------


def logical_counts(circuit) -> Dict[str, int]:
    """Logical gate counts and depth, barriers and measurements excluded."""
    one = two = rotations = hadamards = 0
    for inst in circuit.data:
        name = inst.operation.name
        if name in _DIRECTIVES:
            continue
        if inst.operation.num_qubits == 1:
            one += 1
            rotations += name in ("rz", "rx")
            hadamards += name == "h"
        else:
            two += 1
    return {"one_qubit": one, "two_qubit": two, "total": one + two,
            "depth": int(circuit.depth()), "rotations": rotations, "hadamards": hadamards}


def score(circuit) -> Score:
    counts = logical_counts(circuit)
    return counts["two_qubit"], counts["total"], counts["depth"]


def _angle_is_clifford(angle: float, tol: float = 1e-9) -> bool:
    quarter = float(angle) / (math.pi / 2)
    return abs(quarter - round(quarter)) <= tol


def is_clifford(circuit) -> bool:
    """Every gate a Clifford (rotations at multiples of pi/2 included)."""
    for inst in circuit.data:
        name = inst.operation.name
        if name in _DIRECTIVES or name in _CLIFFORD_NAMES:
            continue
        if name in ("rz", "rx", "ry") and _angle_is_clifford(inst.operation.params[0]):
            continue
        return False
    return True


def non_clifford_rotations(circuit) -> int:
    return sum(1 for inst in circuit.data if inst.operation.name in ("rz", "rx")
               and not _angle_is_clifford(inst.operation.params[0]))


def gates_ok(circuit, basis: Sequence[str] = TRANSLATOR_BASIS) -> bool:
    allowed = set(basis) | _DIRECTIVES
    return all(inst.operation.name in allowed for inst in circuit.data)


def proxy_of(circuit):
    """
    Fig 6's Clifford proxy rule (`experiments.shared.benchmark_circuits.create_proxy_from_qasm`
    with `replace_rz_rx_with_s=True`): every rz/rx becomes s, everything else is kept.
    """
    from qiskit import QuantumCircuit

    out = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        name = inst.operation.name
        if name == "measure":
            continue
        qargs = [out.qubits[circuit.find_bit(q).index] for q in inst.qubits]
        if name in ("rz", "rx"):
            out.s(qargs[0])
        elif name in _PROXY_KEEP:
            out.append(inst.operation, qargs)
        else:
            raise ValueError(f"no proxy rule for gate {name!r}")
    return out


# ----------------------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------------------


def _no_barriers(circuit):
    from qiskit.transpiler.passes import RemoveBarriers

    return RemoveBarriers()(circuit)


#: O3 is not deterministic in Qiskit 2.2.1, even at a fixed `seed_transpiler`, a fixed
#: PYTHONHASHSEED and one Rust thread: repeated calls on the same circuit in one process
#: return different single-qubit counts and depths (two-qubit counts were stable in every
#: probe; bell_n4 33/36/30 gates, basis_trotter_n4 1022-1096). So every O3 call here is the
#: best of this many, which makes the optimiser stronger and less dependent on luck, and the
#: chosen circuits are cached (QPY) so the later stages use exactly what was verified.
O3_TRIALS = 3


def _transpile_o3_once(circuit, basis: Sequence[str], seed: int):
    from qiskit import transpile

    out = transpile(circuit, basis_gates=list(basis), optimization_level=3,
                    seed_transpiler=seed)
    # No coupling map, so there is no layout stage; ElidePermutations could still record a
    # final permutation if it removed a swap. A permuted output is not the same circuit to
    # the translator, so refuse it rather than silently relabel.
    layout = getattr(out, "layout", None)
    if layout is not None:
        final = getattr(layout, "final_layout", None)
        if final is not None and any(
            final[q] != i for i, q in enumerate(out.qubits)
        ):
            return None
        out = out.copy()
        out._layout = None  # noqa: SLF001 -- trivial layout, drop it so circuits compare plainly
    return out


def _transpile_o3(circuit, basis: Sequence[str] = CORPUS_BASIS, trials: Optional[int] = None):
    """Best of `trials` O3 calls (see `O3_TRIALS`)."""
    best, best_score = None, None
    for seed in range(O3_TRIALS if trials is None else trials):
        out = _transpile_o3_once(circuit, basis, seed)
        if out is None:
            continue
        out_score = score(out)
        if best is None or out_score < best_score:
            best, best_score = out, out_score
    return best


def route_o3(circuit):
    return _transpile_o3(circuit, CORPUS_BASIS)


def route_o3_sx(circuit):
    return _transpile_o3(circuit, TRANSLATOR_BASIS)


def snap_cliffords(circuit):
    """
    rz/rx at a multiple of pi/2 (to 1e-9) as the named Clifford gate, everything else as is.

    Equal up to global phase. Needed because Qiskit's own matrix-based Clifford test is
    loose for rx: `Clifford(RXGate(1.570068))` is accepted although the angle is 7e-4 off
    pi/2, so `CollectCliffords(matrix_based=True)` silently rounds near-Clifford rotations
    (measured: 3e-6 process infidelity on basis_trotter_n4). Snapping with a strict
    tolerance and collecting by name instead is exact.
    """
    from qiskit import QuantumCircuit

    named = {"rz": (None, "s", "z", "sdg"), "rx": (None, "sx", "x", "sxdg")}
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs, global_phase=circuit.global_phase)
    for inst in circuit.data:
        name = inst.operation.name
        if name in named and _angle_is_clifford(inst.operation.params[0]):
            quarter = int(round(float(inst.operation.params[0]) / (math.pi / 2))) % 4
            gate = named[name][quarter]
            if gate is not None:
                getattr(out, gate)(inst.qubits[0])
            continue
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def _clifford_global(circuit, method: str):
    if not is_clifford(circuit):
        return None
    from qiskit.quantum_info import Clifford
    from qiskit.synthesis import synth_clifford_full, synth_clifford_greedy

    tableau = Clifford(snap_cliffords(circuit))
    synth = {"ag": synth_clifford_full, "greedy": synth_clifford_greedy}[method]
    return _transpile_o3(synth(tableau), CORPUS_BASIS)


def route_clifford_ag(circuit):
    return _clifford_global(circuit, "ag")


def route_clifford_greedy(circuit):
    return _clifford_global(circuit, "greedy")


def route_clifford_blocks(circuit):
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import CollectCliffords, HighLevelSynthesis
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

    # By name, after a strict snap: see `snap_cliffords` for why not `matrix_based=True`.
    manager = PassManager([
        CollectCliffords(matrix_based=False, min_block_size=2, split_blocks=True),
        HighLevelSynthesis(hls_config=HLSConfig(clifford=[("greedy", {})])),
    ])
    return _transpile_o3(manager.run(snap_cliffords(circuit)), CORPUS_BASIS)


def route_linear_blocks(circuit):
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import CollectLinearFunctions, HighLevelSynthesis
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

    manager = PassManager([
        CollectLinearFunctions(min_block_size=2, split_blocks=True),
        HighLevelSynthesis(hls_config=HLSConfig(linear_function=[("pmh", {})])),
    ])
    return _transpile_o3(manager.run(circuit), CORPUS_BASIS)


def route_commute(circuit):
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        CommutativeCancellation,
        CommutativeInverseCancellation,
        Optimize1qGatesDecomposition,
        Optimize1qGatesSimpleCommutation,
    )

    basis = list(CORPUS_BASIS)
    manager = PassManager([
        Optimize1qGatesSimpleCommutation(basis=basis, run_to_completion=True),
        CommutativeInverseCancellation(),
        CommutativeCancellation(basis_gates=basis),
        Optimize1qGatesDecomposition(basis=basis),
    ])
    return _transpile_o3(manager.run(circuit), CORPUS_BASIS)


ROUTES: Dict[str, Callable] = {
    "o3": route_o3,
    "o3_sx": route_o3_sx,
    "clifford_ag": route_clifford_ag,
    "clifford_greedy": route_clifford_greedy,
    "clifford_blocks": route_clifford_blocks,
    "linear_blocks": route_linear_blocks,
    "commute": route_commute,
}


# ----------------------------------------------------------------------------------------
# the optimiser
# ----------------------------------------------------------------------------------------


@dataclass
class Optimized:
    """The chosen circuit and how it was reached."""

    circuit: object
    changed: bool
    route: str
    o3_circuit: object
    o3_iterations: int
    rounds: int
    seconds: float
    first_round: Dict[str, Optional[Score]] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    #: (score, tie-break, circuit, route) for the original and both chains, best first.
    alternatives: List[Tuple] = field(default_factory=list)


def _too_many_rotations(candidate, max_rotations: Optional[int]) -> bool:
    return max_rotations is not None and non_clifford_rotations(candidate) > max_rotations


def _iterate_o3(circuit, max_rounds: int,
                max_rotations: Optional[int] = None) -> Tuple[object, int]:
    """O3 on its own output until the score stops improving. Returns (circuit, improving passes)."""
    best, best_score, improved = circuit, score(circuit), 0
    for _ in range(max_rounds):
        candidate = route_o3(best)
        if candidate is None or not gates_ok(candidate, CORPUS_BASIS) or \
                _too_many_rotations(candidate, max_rotations):
            break
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best, best_score, improved = candidate, candidate_score, improved + 1
        else:
            break
    return best, improved


def optimize(original, *, max_rounds: int = 8,
             routes: Optional[Dict[str, Callable]] = None,
             max_rotations: Optional[int] = None) -> Optimized:
    """
    Best of: the original, the O3 fixed point, and the alternating all-routes fixed point.
    `max_rotations`: neither chain takes a step to a circuit with more non-Clifford rotations
    than that (the compiler's pass sets it to the program's own count); None, no limit.
    """
    routes = ROUTES if routes is None else routes
    began = time.time()
    start = _no_barriers(original)
    o3_circuit, o3_iterations = _iterate_o3(start, max_rounds, max_rotations)

    current, current_score = start, score(start)
    steps: List[str] = []
    first_round: Dict[str, Optional[Score]] = {}
    failures: Dict[str, str] = {}
    rounds = 0
    for round_index in range(max_rounds):
        rounds += 1
        best_name, best_candidate, best_score = None, None, current_score
        for name, route in routes.items():
            try:
                candidate = route(current)
            except Exception as error:  # noqa: BLE001 - a route that cannot run is skipped
                failures.setdefault(name, f"{type(error).__name__}: {error}"[:200])
                candidate = None
            if candidate is None or not gates_ok(candidate):
                if round_index == 0:
                    first_round[name] = None
                continue
            candidate_score = score(candidate)
            if round_index == 0:
                first_round[name] = candidate_score
            if _too_many_rotations(candidate, max_rotations):
                continue
            if candidate_score < best_score:
                best_name, best_candidate, best_score = name, candidate, candidate_score
        if best_candidate is None:
            break
        current, current_score = best_candidate, best_score
        steps.append(best_name)

    original_score = score(original)
    # Ties go to the earlier entry: the original, then the plain O3 chain.
    options = [(original_score, 0, original, "none"),
               (score(o3_circuit), 1, o3_circuit, "+".join(["o3"] * o3_iterations) or "none"),
               (current_score, 2, current, "+".join(steps) or "none")]
    options.sort(key=lambda option: option[:2])
    chosen_score, _, chosen, route_name = options[0]
    changed = chosen_score < original_score
    if not changed:
        chosen, route_name = original, "none"
    return Optimized(circuit=chosen, changed=changed, route=route_name,
                     o3_circuit=o3_circuit if o3_iterations else original,
                     o3_iterations=o3_iterations, rounds=rounds,
                     seconds=time.time() - began, first_round=first_round, failures=failures,
                     alternatives=options)


def optimize_verified(original, *, seed: int = 0, **kwargs) -> Tuple[Optimized, Dict[str, object]]:
    """
    `optimize`, then `check_equivalence` on its choice. A choice that fails the check is
    discarded for the next-best alternative (the other chain, then the original), and the
    discarded routes are reported in `rejected` -- a circuit that cannot be shown to compute
    the same thing is never handed to the compiler.
    """
    result = optimize(original, **kwargs)
    check = check_equivalence(original, result.circuit, seed=seed)
    rejected: List[str] = []
    if not check["equivalent"]:
        rejected.append(result.route)
        original_score = score(original)
        for option_score, _, option, route_name in result.alternatives:
            if option is result.circuit or option_score >= original_score:
                continue
            option_check = check_equivalence(original, option, seed=seed)
            if option_check["equivalent"]:
                result.circuit, result.route, check = option, route_name, option_check
                break
            rejected.append(route_name)
        else:
            result.circuit, result.route = original, "none"
            result.changed = False
            check = check_equivalence(original, original)
    check["rejected"] = ";".join(rejected)
    return result, check


# ----------------------------------------------------------------------------------------
# equivalence
# ----------------------------------------------------------------------------------------

#: Operator equality is exact and cheap up to here (a 4096 x 4096 unitary).
OPERATOR_MAX_QUBITS = 12
#: Aer statevector miter up to here; matrix-product-state beyond.
STATEVECTOR_MAX_QUBITS = 20
#: Pass criteria, both far below any real difference and far above float round-off.
#: `Operator.equiv`'s elementwise atol of 1e-8 is too tight for circuits with hundreds of
#: float-angle rotations (hhl_n7 drifts 8e-8 elementwise at 1 - 2e-16 overlap), so the
#: test is the phase-free overlap |tr(A^dag B)| / d. A single wrong gate costs O(1); the
#: one genuine approximation found while building this (Qiskit's loose rx Clifford test,
#: see `snap_cliffords`) cost 3e-6, which both tolerances catch.
OPERATOR_TOLERANCE = 1e-9
MITER_TOLERANCE = 1e-7


def _random_product_prep(num_qubits: int, rng: np.random.Generator):
    """A layer of Haar-random single-qubit unitaries: |psi> = prep |0...0>."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UGate

    prep = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        # Haar on SU(2) in Euler form: theta from arccos of a uniform cosine.
        theta = float(np.arccos(1.0 - 2.0 * rng.random()))
        phi, lam = (float(v) for v in rng.uniform(0.0, 2.0 * np.pi, size=2))
        prep.append(UGate(theta, phi, lam), [qubit])
    return prep


def _miter_amplitude(original, optimized, prep, method: str) -> complex:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    n = original.num_qubits
    miter = QuantumCircuit(n)
    miter.compose(prep, inplace=True)
    miter.compose(original, inplace=True)
    miter.compose(optimized.inverse(), inplace=True)
    miter.compose(prep.inverse(), inplace=True)
    miter.save_amplitudes([0])
    simulator = AerSimulator(method=method, max_parallel_threads=1,
                             max_parallel_experiments=1, max_parallel_shots=1)
    # No transpile against the simulator: its coupling map stops at 63 qubits, and every
    # gate here (the basis, their inverses, u) is native to Aer anyway.
    result = simulator.run(miter, shots=1).result()
    return complex(result.data(0)["amplitudes"][0])


def check_equivalence(original, optimized, *, seed: int = 0) -> Dict[str, object]:
    """Is `optimized` the same unitary as `original`, up to global phase?"""
    began = time.time()
    if optimized is original:
        return {"equiv_method": "identical", "equivalent": True, "equiv_detail": 1.0,
                "equiv_seconds": 0.0}
    reference = _no_barriers(original)
    n = reference.num_qubits
    if optimized.num_qubits != n:
        return {"equiv_method": "width", "equivalent": False, "equiv_detail": float("nan"),
                "equiv_seconds": time.time() - began}
    if n <= OPERATOR_MAX_QUBITS:
        from qiskit.quantum_info import Operator

        a, b = Operator(reference).data, Operator(optimized).data
        # |tr(A^dag B)| / d = 1 exactly when B = e^{i phi} A.
        overlap = abs(np.vdot(a, b)) / a.shape[0]
        return {"equiv_method": "operator", "equivalent": bool(overlap >= 1 - OPERATOR_TOLERANCE),
                "equiv_detail": float(overlap), "equiv_seconds": time.time() - began}
    if is_clifford(reference) and is_clifford(optimized):
        from qiskit.quantum_info import Clifford

        same = Clifford(snap_cliffords(reference)) == Clifford(snap_cliffords(optimized))
        return {"equiv_method": "tableau", "equivalent": bool(same),
                "equiv_detail": 1.0 if same else 0.0, "equiv_seconds": time.time() - began}
    method = "statevector" if n <= STATEVECTOR_MAX_QUBITS else "matrix_product_state"
    rng = np.random.default_rng(seed)
    from qiskit import QuantumCircuit

    preps = [QuantumCircuit(n), _random_product_prep(n, rng), _random_product_prep(n, rng)]
    worst = min(abs(_miter_amplitude(reference, optimized, prep, method)) for prep in preps)
    return {"equiv_method": f"miter_{'sv' if method == 'statevector' else 'mps'}_x3",
            "equivalent": bool(worst >= 1.0 - MITER_TOLERANCE), "equiv_detail": float(worst),
            "equiv_seconds": time.time() - began}
