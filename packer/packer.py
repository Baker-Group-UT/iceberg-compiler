"""
Packing planner for [[4,2,2]] Iceberg codes.

Implementation of a multi-start stochastic greedy packer that groups logical
qubits (from a transpiled Qiskit circuit) into codes of capacity 2 (pairs) or 1 (solo),
aiming to (1) balance expected error across codes, (2) keep each code under a bound,
(3) minimize total expected error (secondary).

Notes
-----
- We do not import Qiskit to keep this module lightweight. The functions assume
  a Qiskit-like circuit interface:
    * circuit.qubits -> list of qubit objects
    * circuit.data   -> iterable of (Instruction, qargs, cargs) or similar
    * instruction.name or instruction.operation.name -> gate name string
- Gate weights are constants at the top of the file (not tunable via config).
- The interaction graph stores counts/weights of gate usages; final scoring applies
  the error weights.
- All randomness is seeded via PackConfig.seed for reproducibility.

This file is intended to be imported by higher-level pipeline/translator code.
"""

# ------------------------------------------------------------
# PSEUDOCODE SUMMARY
# ------------------------------------------------------------
# This module finds a way to group qubits from a transpiled Qiskit circuit
# into small groups (codes) of size 1 or 2. Each group represents a logical
# [[4,2,2]] Iceberg code.
#
# Goal:
#   - Make the expected error of each code as even as possible.
#   - Keep all codes below a set error bound.
#   - If possible, keep the total error low.
#
# Main function: pack(circuit, config)
#   1) Remove any barrier gates that have no effect.
#   2) Check that only allowed gates appear.
#   3) Build a simple graph where each qubit is a node.
#      - Node weight = total single qubit gate error.
#      - Edge weight = total two qubit gate error count.
#   4) Run a randomized greedy search several times.
#      - Each run picks qubit pairs or solos step by step.
#      - At each step it checks which pair or solo looks best.
#      - Sometimes it makes a slightly random choice to explore.
#   5) Compute the total and per-code errors for each result.
#   6) Return the result that gives the best balance of errors.
#
# Helpers:
#   - remove_barriers: drop barrier ops from the circuit.
#   - validate_gateset: check only known gates are used.
#   - build_interaction_graph: count all 1q and 2q uses.
#   - shortlist_topk_pairs: list top few best pair or solo choices.
#   - select_pair: pick one based on a random draw.
#   - commit_pair / commit_solo: record the choice.
#   - build_packing_from_state: produce the list of codes.
#   - final_score: compute code errors and the overall score.
#   - run_multi_start: repeat search many times to find the best.
#   - packing_to_permutation: convert packing into code-slot mapping.
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import math
import random

# ----------------------------
# CONSTANTS (weights are fixed here; not tunable via config)
# ----------------------------

ALLOWED_GATESET = ["x", "z", "h", "rz", "cx", "cz", "measure"]

# Per-gate single-qubit error weights (FILL THESE WITH REAL VALUES)
W1: Dict[str, float] = {
    "x": 0.1,    # placeholder
    "z": 0.1,    # placeholder
    "h": 1.0,    # placeholder
    "rz": 1.0,   # placeholder
}

# Two-qubit error weights (FILL THESE WITH REAL VALUES)
W2_SAME: float = 2.0   # when both qubits are packed in the same code (placeholder)
W2_DIFF: float = 5.0   # when qubits are in different codes (placeholder)

# Optional solo discount (when a qubit is alone in a code)
SOLO_MULTIPLIER: float = 1.0  # set < 1.0 to enable discount

# Scoring weights
LARGE_PENALTY: float = 1e6   # penalty per bound violation
LAMBDA_TOTAL: float = 1e-3   # small weight on total error to break ties

# Heuristic knobs for marginals
BETA_LOOKAHEAD: float = 0.0  # small positive to encourage pairing high-degree nodes
GAMMA_SOLO: float = 0.0      # small positive to bias against making hub nodes solo


# ----------------------------
# DATA STRUCTURES
# ----------------------------

@dataclass
class PackConfig:
    """Runtime knobs (weights are constants above).
    m: number of multi-start iterations
    k: shortlist size per step
    deviation_probs: probability by rank (length >= k), will be renormalized
    bound: per-code error bound (float)
    scoring_mode: 'variance' or 'range'
    solo_enabled: allow 'pair-with-null' candidates
    seed: RNG seed (int or None)
    """
    m: int = 1000
    k: int = 5
    deviation_probs: Optional[List[float]] = None
    bound: float = math.inf
    scoring_mode: str = "variance"
    solo_enabled: bool = True
    seed: Optional[int] = None


@dataclass
class ScoreBundle:
    """Full evaluation of a completed packing."""
    score: float
    per_code_errors: List[float]
    violations: int
    range_or_stdev: float


@dataclass
class BestPackingResult:
    """What pack(...) returns."""
    packing: List[Dict[str, Any]] # [{"code_id": i, "members": [qids...]}]
    per_code_errors: List[float]
    score: float
    bound_violations: int
    permutation: List[Tuple] # Permutation list: [(patch_id, slot)], index = logical qubit order
    params_used: PackConfig
    debug: Optional[Dict[str, Any]] = None


@dataclass
class GraphData:
    """Interaction graph extracted from the circuit."""
    nodes: List[Any]                                  # Qiskit qubits or indices
    n1q: Dict[Any, float]                             # cumulative 1q error per node
    n2q_total: Dict[Tuple[Any, Any], float]           # cumulative 2q weight per edge
    meta: Dict[str, Any]                              # optional extras


@dataclass
class PartialState:
    """State maintained during greedy construction within one iteration."""
    topack: List[Any]                                 # remaining qubits to assign
    pairs: List[Tuple[Any, Any]]                      # committed pairs
    solos: List[Any]                                  # committed solos (if enabled)
    caches: Dict[str, Any]                            # precomputed sums, etc.


# ----------------------------
# TOP-LEVEL API
# ----------------------------

def pack(circuit, config: PackConfig = None) -> BestPackingResult:
    """Plan a packing for the given (transpiled) Qiskit circuit.

    Steps
    -----
    1) Remove barrier gates (no-ops for packing)
    2) Validate gateset
    3) Build interaction graph
    4) Run multi-start stochastic greedy to propose packings
    5) Score and return the best result
    """
    if not config:
        config = PackConfig()

    circuit = remove_barriers(circuit)
    validate_gateset(circuit)
    graph = build_interaction_graph(circuit)
    result = run_multi_start(graph, config)
    return result


# ----------------------------
# HELPERS
# ----------------------------

def remove_barriers(circuit):
    """Return a circuit with all 'barrier' operations removed.

    We avoid importing Qiskit. This function tries to use circuit.copy() to avoid
    mutating the original. If that is unavailable, it will operate in-place on
    circuit.data as a best-effort fallback.
    """
    # Try to make a shallow copy if supported
    try:
        new_circ = circuit.copy()
    except Exception:
        new_circ = circuit

    data = getattr(new_circ, "data", [])
    if data is None:
        return new_circ

    filtered = []
    for entry in list(data):
        name = _get_gate_name_lower(entry)
        if name == "barrier":
            continue
        filtered.append(entry)

    # Try to assign a new data list
    try:
        new_circ.data = filtered
        return new_circ
    except Exception:
        # Fallback: mutate the existing list in place
        try:
            data.clear()
            data.extend(filtered)
        except Exception:
            # If even this fails, just return whatever we have
            pass
        return new_circ

# End remove_barriers

# ----------------------------

def _get_gate_name_lower(instruction: Any) -> str:
    """Try to obtain a lowercased gate name from a Qiskit-like instruction.
    Compatible with both Instruction objects and tuples from circuit.data.
    """
    # Qiskit usually stores circuit.data as list of (Instruction, qargs, cargs)
    if isinstance(instruction, tuple) and len(instruction) >= 1:
        inst = instruction[0]
        name = getattr(inst, "name", None)
        if name is None and hasattr(inst, "operation"):
            name = getattr(inst.operation, "name", None)
    else:
        name = getattr(instruction, "name", None)
        if name is None and hasattr(instruction, "operation"):
            name = getattr(instruction.operation, "name", None)
    if not isinstance(name, str):
        return ""
    return name.lower()


def _extract_qargs(instruction_tuple: Any) -> List[Any]:
    """Extract qargs from a Qiskit (Instruction, qargs, cargs) tuple.
    Returns a list of qubit objects.
    """
    if isinstance(instruction_tuple, tuple) and len(instruction_tuple) >= 2:
        qargs = instruction_tuple[1]
        return list(qargs) if qargs is not None else []
    # Fallback: try to read .qargs
    qargs = getattr(instruction_tuple, "qargs", [])
    return list(qargs) if qargs is not None else []


def validate_gateset(circuit) -> None:
    """Ensure all gates in the circuit belong to the allowed gateset.

    Raises
    ------
    ValueError if an unsupported gate is found.
    """
    for entry in getattr(circuit, "data", []):
        gname = _get_gate_name_lower(entry)
        if gname and gname not in ALLOWED_GATESET:
            raise ValueError(f"Unsupported gate in circuit: {gname}")


def build_interaction_graph(circuit) -> GraphData:
    """Build the interaction graph and aggregate gate usage weights.

    Nodes are the circuit's logical qubits. We store cumulative 1q error weights
    per node and cumulative 2q counts per unordered pair edge.
    """
    qubits = list(getattr(circuit, "qubits", []))
    nodes = list(range(len(qubits)))  # use integer indices for internal maps
    q2i = {q: i for i, q in enumerate(qubits)}

    n1q: Dict[int, float] = {i: 0.0 for i in nodes}
    n2q_total: Dict[Tuple[int, int], float] = {}

    for entry in getattr(circuit, "data", []):
        gname = _get_gate_name_lower(entry)
        qargs = _extract_qargs(entry)
        if not gname:
            continue
        if gname in ("x", "z", "h", "rz"):
            if len(qargs) != 1:
                continue
            qi = q2i.get(qargs[0])
            if qi is None:
                continue
            n1q[qi] = n1q.get(qi, 0.0) + W1[gname]
        elif gname in ("cx", "cz"):
            if len(qargs) != 2:
                continue
            u = q2i.get(qargs[0])
            v = q2i.get(qargs[1])
            if u is None or v is None or u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            n2q_total[(a, b)] = n2q_total.get((a, b), 0.0) + 1.0
        elif gname == "measure":
            # ignore for error aggregation (measurement excluded here)
            pass
        else:
            # validate_gateset should have caught this already, but be safe
            raise ValueError(f"Unsupported gate encountered: {gname}")

    # Precompute simple degree-like sums for heuristics
    deg_sum: Dict[int, float] = {i: 0.0 for i in nodes}
    for (a, b), w in n2q_total.items():
        deg_sum[a] += w
        deg_sum[b] += w

    meta = {
        "q2i": q2i,
        "i2q": {i: q for q, i in q2i.items()},
        "deg_sum": deg_sum,
    }

    return GraphData(nodes=nodes, n1q=n1q, n2q_total=n2q_total, meta=meta)


def initialize_state(nodes: List[Any], graph: GraphData) -> PartialState:
    """Initialize the partial state for one greedy iteration."""
    topack = list(nodes)
    pairs: List[Tuple[Any, Any]] = []
    solos: List[Any] = []
    caches = {
        "deg_sum": dict(graph.meta.get("deg_sum", {})),
    }
    return PartialState(topack=topack, pairs=pairs, solos=solos, caches=caches)


def shortlist_topk_pairs(state: PartialState,
                         graph: GraphData,
                         k: int,
                         solo_enabled: bool = True) -> List[Tuple[Any, Optional[Any], float]]:
    """Produce the top-k candidate pairings (and optional solos) with scores.

    Returns a list of tuples (u, v, score) where v may be None for solo.
    Lower score is better.
    """
    candidates: List[Tuple[Any, Optional[Any], float]] = []
    tp = state.topack
    n = len(tp)
    # Pair candidates
    for i in range(n):
        u = tp[i]
        for j in range(i + 1, n):
            v = tp[j]
            s = estimate_pair_marginal(u, v, state, graph)
            candidates.append((u, v, s))
    # Solo candidates (optional)
    if solo_enabled:
        for q in tp:
            s = estimate_solo_marginal(q, state, graph)
            candidates.append((q, None, s))

    # Sort by score ascending and return top-k
    candidates.sort(key=lambda t: t[2])
    return candidates[:k] if k < len(candidates) else candidates


def estimate_pair_marginal(u: Any, v: Any, state: PartialState, graph: GraphData) -> float:
    """Heuristic local score for pairing (u,v). Lower is better.

    within_pair_gain = (W2_SAME - W2_DIFF) * n2q(u,v)
      (typically negative if W2_SAME < W2_DIFF).
    lookahead = BETA_LOOKAHEAD * (deg(u) + deg(v) - 2*n2q(u,v))
      encourages pairing high-degree neighbors, excluding mutual edge counted twice.
    """
    a, b = (u, v) if u < v else (v, u)
    n2 = graph.n2q_total.get((a, b), 0.0)
    within_pair_gain = (W2_SAME - W2_DIFF) * n2

    deg_sum = state.caches.get("deg_sum", {})
    du = deg_sum.get(u, 0.0)
    dv = deg_sum.get(v, 0.0)
    lookahead = BETA_LOOKAHEAD * max(0.0, (du + dv - 2.0 * n2))

    # Balance bias could be added here if we maintain projected per-code loads
    balance_bias = 0.0

    return within_pair_gain + lookahead + balance_bias


def estimate_solo_marginal(q: Any, state: PartialState, graph: GraphData) -> float:
    """Heuristic local score for placing q as a solo.

    1q_effect = (SOLO_MULTIPLIER - 1) * n1q[q]  (usually <= 0 if multiplier < 1)
    cross_effect approximated with degree: GAMMA_SOLO * deg(q)
    """
    n1 = graph.n1q.get(q, 0.0)
    deg_sum = state.caches.get("deg_sum", {})
    d = deg_sum.get(q, 0.0)

    oneq_effect = (SOLO_MULTIPLIER - 1.0) * n1
    cross_effect = GAMMA_SOLO * d

    return oneq_effect + cross_effect


def select_pair(topk: List[Tuple[Any, Optional[Any], float]],
                rng: random.Random,
                deviation_probs: Optional[List[float]]) -> Tuple[Any, Optional[Any]]:
    """Sample a candidate from the ranked shortlist using deviation probabilities."""
    if not topk:
        raise RuntimeError("select_pair called with empty shortlist")

    n = len(topk)
    if deviation_probs is None or len(deviation_probs) == 0:
        # Default geometric-like distribution: 1/2, 1/4, 1/8, ...
        probs = [0.5 ** (i + 1) for i in range(n)]
    else:
        probs = deviation_probs[:n]
        if len(probs) < n:
            # pad tail with geometric decay
            tail = [probs[-1] * (0.5 ** (i + 1)) for i in range(n - len(probs))]
            probs = probs + tail

    s = sum(probs)
    if s <= 0:
        probs = [1.0 / n] * n
    else:
        probs = [p / s for p in probs]

    # Sample index
    r = rng.random()
    cdf = 0.0
    idx = 0
    for i, p in enumerate(probs):
        cdf += p
        if r <= cdf:
            idx = i
            break

    u, v, _ = topk[idx]
    return u, v


def commit_pair(state: PartialState, pair: Tuple[Any, Any]) -> None:
    """Commit a pair (u,v) and remove them from the pool."""
    u, v = pair
    state.pairs.append((u, v))
    # Remove from topack
    state.topack = [q for q in state.topack if q != u and q != v]
    # Optionally update caches; deg_sum could be reduced but not strictly needed for heuristic


def commit_solo(state: PartialState, q: Any) -> None:
    """Commit a solo qubit q and remove it from the pool."""
    state.solos.append(q)
    state.topack = [x for x in state.topack if x != q]


def build_packing_from_state(state: PartialState) -> List[Dict[str, Any]]:
    """Turn committed pairs and solos into a packing list with code_id labels."""
    packing: List[Dict[str, Any]] = []
    code_id = 0
    for u, v in state.pairs:
        packing.append({"code_id": code_id, "members": [u, v]})
        code_id += 1
    for q in state.solos:
        packing.append({"code_id": code_id, "members": [q]})
        code_id += 1
    return packing


def _code_pairs(members: List[int]) -> List[Tuple[int, int]]:
    if len(members) < 2:
        return []
    # Only capacity 2 is expected; if extended, generalize combinations
    if len(members) == 2:
        u, v = members
        return [(min(u, v), max(u, v))]
    # Fallback for unexpected sizes
    mem = sorted(members)
    pairs = []
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            pairs.append((mem[i], mem[j]))
    return pairs


def final_score(packing: List[Dict[str, Any]],
                graph: GraphData,
                bound: float,
                scoring_mode: str,
                solo_multiplier: float) -> ScoreBundle:
    """Compute per-code expected errors and aggregate score."""
    # Map code_id to members for convenience
    codes: Dict[int, List[int]] = {item["code_id"]: list(item["members"]) for item in packing}

    per_code_errors: List[float] = []

    # Precompute for faster lookup
    n2 = graph.n2q_total

    # Compute per-code errors
    for cid in sorted(codes.keys()):
        members = codes[cid]
        size = len(members)

        # 1q contribution
        e1 = 0.0
        for q in members:
            mult = solo_multiplier if size == 1 else 1.0
            e1 += graph.n1q.get(q, 0.0) * mult

        # 2q within-code
        e2_within = 0.0
        for a, b in _code_pairs(members):
            e2_within += W2_SAME * n2.get((a, b), 0.0)

        # 2q across-code (charge to this code for edges to other codes)
        e2_across = 0.0
        for other_cid in sorted(codes.keys()):
            if other_cid == cid:
                continue
            members2 = codes[other_cid]
            for q in members:
                for r in members2:
                    a, b = (q, r) if q < r else (r, q)
                    base = W2_DIFF * n2.get((a, b), 0.0)
                    # Apply solo multipliers per policy: multiply if either code is solo
                    if size == 1:
                        base *= solo_multiplier
                    if len(members2) == 1:
                        base *= solo_multiplier
                    e2_across += base

        Ei = e1 + e2_within + e2_across
        per_code_errors.append(Ei)

    # Balance metric
    if not per_code_errors:
        balance = 0.0
    elif scoring_mode == "variance":
        balance = stdev(per_code_errors)
    else:  # 'range'
        balance = max(per_code_errors) - min(per_code_errors)

    violations = sum(1 for x in per_code_errors if x > bound)
    total_err = sum(per_code_errors)
    score = balance + LARGE_PENALTY * violations + LAMBDA_TOTAL * total_err

    return ScoreBundle(score=score,
                       per_code_errors=per_code_errors,
                       violations=violations,
                       range_or_stdev=balance)


def run_multi_start(graph: GraphData, config: PackConfig) -> BestPackingResult:
    """Run multi-start stochastic greedy and return the best packing found."""
    rng = seeded_rng(config.seed)
    best: Optional[BestPackingResult] = None

    for _ in range(max(1, config.m)):
        state = initialize_state(graph.nodes, graph)
        while len(state.topack) >= 2:
            topk = shortlist_topk_pairs(state, graph, config.k, solo_enabled=config.solo_enabled)
            u, v = select_pair(topk, rng, config.deviation_probs)
            if v is None:
                commit_solo(state, u)
            else:
                commit_pair(state, (u, v))
        if config.solo_enabled and len(state.topack) == 1:
            commit_solo(state, state.topack[0])

        packing = build_packing_from_state(state)
        sb = final_score(packing, graph, config.bound, config.scoring_mode, SOLO_MULTIPLIER)
        candidate = assemble_best_result(packing, sb, graph, config)
        if best is None or candidate.score < best.score:
            best = candidate

    assert best is not None
    return best


def packing_to_permutation(packing: List[Dict[str, Any]],
                           nodes: List[Any]) -> Any:
    """Create a permutation list indexed by logical qubit index.

    Returns
    -------
    List[Tuple[int, int]]
        At index i (logical qubit index), the tuple is (patch_id, slot_in_patch).
        For pairs, slots are 0 and 1 assigned by ascending member index.
        For solos, slot is 0.
    """
    # Build a temporary map from node -> (code_id, slot)
    tmp: Dict[int, Tuple[int, int]] = {}
    for item in packing:
        cid = item["code_id"]
        members = list(item["members"])
        if len(members) == 1:
            q = members[0]
            tmp[q] = (cid, 0)
        elif len(members) == 2:
            a, b = members
            if a == b:
                tmp[a] = (cid, 0)
            else:
                u, v = (a, b) if a < b else (b, a)
                tmp[u] = (cid, 0)
                tmp[v] = (cid, 1)
        else:
            for slot, q in enumerate(sorted(members)):
                tmp[q] = (cid, slot)

    # Ensure all nodes have an assignment. If any node is missing, assign it
    # to a fresh solo code_id to keep the permutation complete.
    next_code_id = (max([cid for cid, _ in tmp.values()], default=-1) + 1)
    for q in nodes:
        if q not in tmp:
            tmp[q] = (next_code_id, 0)
            next_code_id += 1

    # Build the permutation list in node index order
    perm: List[Tuple[int, int]] = []
    for q in nodes:
        perm.append(tmp[q])

    return perm


def assemble_best_result(packing: List[Dict[str, Any]],
                         sb: ScoreBundle,
                         graph: GraphData,
                         config: PackConfig) -> BestPackingResult:
    perm = packing_to_permutation(packing, graph.nodes)
    return BestPackingResult(
        packing=packing,
        per_code_errors=sb.per_code_errors,
        score=sb.score,
        bound_violations=sb.violations,
        permutation=perm,
        params_used=config,
        debug=None,
    )


# ----------------------------
# UTILS
# ----------------------------

def seeded_rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed)


def stdev(xs: List[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = sum(xs) / n
    var = sum((x - mu) * (x - mu) for x in xs) / n
    return var ** 0.5
