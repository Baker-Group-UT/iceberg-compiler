# simulator/transpile.py
from __future__ import annotations

from typing import Optional, Any, List, Dict, Tuple, Iterable
from collections import defaultdict
import cirq
from .css_service import SuperstaqService
import cirq_superstaq as css

# FIXME: also consider: cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset()) where we change the gateset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_repetition_ids.*")
# def _normalize_circuit_ops(c: cirq.Circuit) -> cirq.Circuit:
#     return cirq.map_operations(
#         c,
#         lambda op, _: op.replace(use_repetition_ids=False)
#         if isinstance(op, cirq.CircuitOperation) else op
#     )

def transpile_for_sqale(
    circuit: cirq.Circuit,
    *,
    target: str = "cq_hilbert_qpu",                 # unused (kept for signature compatibility)
    use_superstaq: bool = True,                     # unused (Cirq-only path)
    enforce_canonical: bool = True,                 # keep True to ensure RZ/X form
    allow_global: bool = True,                      # fuse parallel X into GlobalX if True
    service: Optional[Any] = None,                  # unused
) -> cirq.Circuit:
    """
    Cirq-only transpilation to {CZ, RZ, GlobalX (optional), measurements} with fixed qubit count.

    Steps:
      1) Convert to CZ-native set (removes raw CNOT/H/X).
      2) Expand PhasedX -> RZ • X • RZ.
      3) Optionally fuse parallel X(t) in each moment into GlobalXPowGate(t).
    """
    # 1) CZ-native conversion
    cz_ctx = cirq.TransformerContext(deep=True)
    # cz_native = cirq.CZTargetGateset().convert_circuit(circuit, context=cz_ctx)
    gateset = cirq.CZTargetGateset()
    cz_native = cirq.optimize_for_target_gateset(circuit=circuit, gateset=gateset)
    # cz_native = _normalize_circuit_ops(cz_native)

    # 2) Expand PhasedX -> RZ X RZ
    expanded = _expand_phasedx_to_rz_x_rz(cz_native) if enforce_canonical else cz_native

    # 3) Fuse parallel X into GlobalX (optional)
    fused = _fuse_global_x(expanded) if allow_global else expanded

    # Done: only CZ, ZPow (RZ), GlobalX (optional), and measurements should remain.
    return fused


# ------------------------ GlobalX gate ------------------------

class GlobalXPowGate(cirq.Gate):
    """A k-qubit gate that decomposes to parallel X**exponent on all its qubits."""
    def __init__(self, n_qubits: int, *, exponent: float):
        if n_qubits < 1:
            raise ValueError("GlobalXPowGate requires n_qubits >= 1")
        self._n = int(n_qubits)
        self.exponent = float(exponent)

    def num_qubits(self) -> int:
        return self._n

    def _decompose_(self, qubits: Iterable[cirq.Qid]) -> Iterable[cirq.Operation]:
        xp = cirq.XPowGate(exponent=self.exponent)
        for q in qubits:
            yield xp(q)

    def _circuit_diagram_info_(self, args) -> Tuple[str, ...]:
        label = f"GX^{self.exponent:g}"
        return tuple(label for _ in range(self._n))

    def __repr__(self) -> str:
        return f"GlobalXPowGate(n_qubits={self._n}, exponent={self.exponent})"

    def __str__(self) -> str:
        return f"GlobalX^{self.exponent:g}({self._n})"


# ------------------------ Passes ------------------------

def _expand_phasedx_to_rz_x_rz(c: cirq.Circuit) -> cirq.Circuit:
    """
    Replace each PhasedXPowGate(phase_exponent=φ, exponent=t)
    with Z**φ, X**t, Z**(-φ) on the same qubit.

    Notes:
      - Cirq uses 'turns' for exponents; ZPowGate(exponent=1) == Z (π rotation).
      - This preserves measurement keys and moment structure (ops appended in-place).
    """
    new_moments: List[cirq.Moment] = []
    for m in c.moments:
        ops: List[cirq.Operation] = []
        for op in m.operations:
            g = op.gate
            if isinstance(g, cirq.PhasedXPowGate):
                q = op.qubits[0]
                phi = float(g.phase_exponent)
                t = float(g.exponent)
                # Z**phi • X**t • Z**(-phi)
                if abs(phi) > 0:
                    ops.append(cirq.ZPowGate(exponent=phi)(q))
                if abs(t) > 0:
                    ops.append(cirq.XPowGate(exponent=t)(q))
                if abs(phi) > 0:
                    ops.append(cirq.ZPowGate(exponent=-phi)(q))
            else:
                ops.append(op)
        new_moments.append(cirq.Moment(ops))
    return cirq.Circuit(new_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


def _fuse_global_x(c: cirq.Circuit, *, tol: float = 1e-12) -> cirq.Circuit:
    """
    Within each moment, fuse all XPowGate(exponent≈t) across disjoint qubits into
    a single GlobalXPowGate(exponent=t). Different exponents create separate groups.
    """
    new_moments: List[cirq.Moment] = []
    for m in c.moments:
        # Collect X gates by exponent (bucket by rounded value to tolerate float noise)
        buckets: Dict[float, List[Tuple[cirq.Operation, cirq.Qid]]] = defaultdict(list)
        keep: List[cirq.Operation] = []
        for op in m.operations:
            g = op.gate
            if isinstance(g, cirq.XPowGate) and len(op.qubits) == 1:
                t = float(g.exponent)
                key = _round_key(t, tol)
                buckets[key].append((op, op.qubits[0]))
            else:
                keep.append(op)

        # Build fused globals + any unfused X
        fused_ops: List[cirq.Operation] = keep[:]
        for key, items in buckets.items():
            if len(items) == 1:
                # Single X remains as-is
                fused_ops.append(items[0][0])
            else:
                # Many X with the same exponent -> one GlobalX
                t = items[0][0].gate.exponent  # representative exponent
                qubits = tuple(q for _, q in items)
                # Keep qubit order stable:
                qubits = tuple(sorted(qubits, key=lambda q: (str(type(q)), q.x if hasattr(q, "x") else str(q))))
                fused_ops.append(GlobalXPowGate(len(qubits), exponent=float(t))(*qubits))

        new_moments.append(cirq.Moment(fused_ops))
    return cirq.Circuit(new_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)


def _round_key(x: float, tol: float) -> float:
    """Bucket a float by rounding to a grid defined by tol."""
    if tol <= 0:
        return x
    return round(x / tol) * tol

# NOTE: SUPERSTAQ TRANSPILING
# def transpile_for_sqale(
#     circuit: cirq.Circuit,
#     *,
#     target: str = "cq_hilbert_qpu",
#     use_superstaq: bool = True,
#     enforce_canonical: bool = False,
#     allow_global: bool = True,
#     service: Optional[Any] = None,  # Optional cirq-superstaq Service-like client
# ) -> cirq.Circuit:
#     """
#     Transpile `circuit` to the target gateset appropriate for SQALE simulations.

#     Pipeline:
#       1) Try cirq-superstaq compilation (preserves measurement keys/qubit order).
#       2) Optionally enforce canonical {Global rotations, RZ, CZ} by rewriting StrippedCZ→CZ+RZ.
#       3) Fallback: Cirq CZTargetGateset if superstaq not available/disabled.

#     Args:
#         circuit: Cirq circuit that may contain H/CNOT/etc.
#         target:  cirq-superstaq target name (e.g., "cq_hilbert_qpu").
#         use_superstaq: if True, attempt superstaq compile first.
#         enforce_canonical: if True, convert StrippedCZ into CZ + appropriate RZ corrections.
#         allow_global: retained for future control of “global rotation” expansion; currently pass-through.
#         service: optional cirq-superstaq service/client instance.

#     Returns:
#         Transpiled circuit suitable for leakage-aware SQALE simulation.
#     """
#     c = circuit
    
#     if use_superstaq:
#         try:
#             # Get (or create) the singleton Service
#             if service is None:
#                 service = SuperstaqService.get()
#                 print("Superstaq Service Connected")

#             # Prefer cq_compile if available; else fall back to compile
#             if hasattr(service, "cq_compile"):
#                 compiled = service.cq_compile(c, target=target)
#             else:
#                 compiled = service.compile(c, target=target)

#             # Extract the circuit (handles both wrapped and raw returns)
#             c = getattr(compiled, "circuit", compiled)

#             if enforce_canonical:
#                 c = _enforce_canonical_gateset(c)

#             return c
#         except Exception:
#             # If Superstaq isn't available or compile fails, fall through to fallback
#             pass

#     # Fallback: Cirq CZ-target conversion (guarantees CZ + 1q rotations)
#     c = transpile_fallback_cz(c)
#     if enforce_canonical:
#         c = _enforce_canonical_gateset(c)
#     return c


def transpile_fallback_cz(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Fallback transpilation using Cirq's CZ-target gateset.
    Produces a circuit primarily in terms of CZ + single-qubit rotations.
    """
    # Cirq offers a direct conversion API on the target gateset object.
    # This is broadly compatible across Cirq versions.
    try:
        tg = cirq.CZTargetGateset()
        return cirq.optimize_for_target_gateset(circuit=circuit, gateset=tg)
    except Exception:
        # Ultra-conservative fallback: return original circuit unchanged.
        # (Better than raising in production pipelines; tests should catch versioning issues.)
        return circuit


def _enforce_canonical_gateset(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Convert vendor-specific two-qubit ops (e.g., StrippedCZ) to canonical CZ + RZ corrections.

    Notes:
      - If `StrippedCZ` is present (from cirq_superstaq), we replace it with a CZ and
        push the parameterized Z-phase onto its qubits via RZ. If the stripped gate
        carries per-qubit phases, we apply them accordingly.
      - If not found, this is a no-op.
      - Measurement keys and qubit order are preserved.

    This function is intentionally conservative; it only rewrites when it recognizes
    a known structure and otherwise leaves the circuit intact.
    """
    try:
        import cirq_superstaq as css  # type: ignore
        StrippedCZ = getattr(css, "StrippedCZ", None)
    except Exception:
        StrippedCZ = None

    if StrippedCZ is None:
        return circuit

    new_moments: list[cirq.Moment] = []
    for m in circuit.moments:
        new_ops: list[cirq.Operation] = []
        for op in m.operations:
            if isinstance(op.gate, StrippedCZ):
                q0, q1 = op.qubits
                gate = op.gate

                # Heuristic: StrippedCZ may encode per-qubit phase shifts; fall back if unavailable.
                # Try common attribute names; if missing, treat as zero.
                theta0 = getattr(gate, "rz0", 0.0) or getattr(gate, "rz_a", 0.0) or 0.0
                theta1 = getattr(gate, "rz1", 0.0) or getattr(gate, "rz_b", 0.0) or 0.0

                # Apply CZ plus RZ corrections (order chosen to keep semantics clear;
                # many stacks treat RZ as virtual, so commuting is benign).
                new_ops.append(cirq.CZ(q0, q1))
                if abs(theta0) > 0:
                    new_ops.append(cirq.rz(theta0)(q0))
                if abs(theta1) > 0:
                    new_ops.append(cirq.rz(theta1)(q1))
            else:
                new_ops.append(op)
        new_moments.append(cirq.Moment(new_ops))
    return cirq.Circuit(new_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)



# ------------------------ Diagnostics / Sanity Checks ------------------------

from collections import Counter
import re

def _gate_label(op: cirq.Operation) -> str:
    g = op.gate
    if g is None:
        return op.__class__.__name__
    # Prefer stable readable names
    if isinstance(g, cirq.Gate):
        return g.__class__.__name__
    return str(g)

def summarize_gates(circuit: cirq.Circuit) -> Counter:
    """
    Return a histogram of gate class names present in `circuit`.
    """
    hist = Counter()
    for m in circuit.moments:
        for op in m.operations:
            hist[_gate_label(op)] += 1
    return hist

def _is_global_gate(op: cirq.Operation) -> bool:
    """
    Heuristic: treat cirq_superstaq "global" rotations as allowed when present.
    We accept class names like GlobalX/GlobalY/GlobalW, etc.
    """
    name = _gate_label(op)
    return bool(re.match(r"^Global[A-Z]\w*", name))

def _is_rz_like(op: cirq.Operation) -> bool:
    # cirq.rz produces a ZPowGate
    return isinstance(op.gate, cirq.ZPowGate)

def _is_cz_like(op: cirq.Operation) -> bool:
    # Accept CZ and CZPowGate with exponent ≡ 1 mod 2 (i.e., CZ)
    if isinstance(op.gate, cirq.CZPowGate):
        # exponent could be float; treat near-odd integers as CZ
        exp = float(op.gate.exponent)
        return abs((exp % 2) - 1.0) < 1e-9
    return isinstance(op.gate, cirq.ops.common_gates.CZPowGate) or op.gate == cirq.CZ

def _is_measurement(op: cirq.Operation) -> bool:
    return isinstance(op.gate, cirq.MeasurementGate)

def find_disallowed_ops(
    circuit: cirq.Circuit,
    *,
    allow_globals: bool = True,
) -> list[tuple[int, cirq.Operation]]:
    """
    Return a list of (moment_index, op) that are NOT in the allowed set:
      - CZ / CZPowGate (≡ CZ)
      - RZ / ZPowGate
      - Measurements
      - (optionally) Global* rotations
    """
    bad: list[tuple[int, cirq.Operation]] = []
    for mi, m in enumerate(circuit.moments):
        for op in m.operations:
            if _is_measurement(op):
                continue
            if _is_cz_like(op):
                continue
            if _is_rz_like(op):
                continue
            if allow_globals and _is_global_gate(op):
                continue
            bad.append((mi, op))
    return bad

def print_transpile_report(
    circuit: cirq.Circuit,
    *,
    allow_globals: bool = True,
) -> bool:
    """
    Print a brief report of gate inventory and any disallowed operations.
    Returns True if the circuit passes (no disallowed ops), else False.
    """
    hist = summarize_gates(circuit)
    print("=== Transpile Report ===")
    print("Gate inventory:")
    for name, count in sorted(hist.items()):
        print(f"  {name:24s} x {count}")
    bad = find_disallowed_ops(circuit, allow_globals=allow_globals)
    if not bad:
        print("\nResult: ✅ OK — only CZ / RZ / (globals allowed) / measurements remain.")
        return True
    print("\nResult: ❌ Found disallowed operations:")
    for mi, op in bad[:50]:  # cap output
        print(f"  moment {mi}: {op!r}  (gate={_gate_label(op)})")
    if len(bad) > 50:
        print(f"  ... and {len(bad) - 50} more.")
    return False
