"""
translator.passes.py - Shared Qiskit transpiler passes for circuit cleanup.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    HGate,
    RZGate,
    RXGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    XGate,
    ZGate,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation

from .custom_gates import XCXGate

__all__ = [
    "cancel_inverse_gates",
    "cancel_inverse_gates_through_swaps",
    "commute_hadamards",
    "commutation_gain",
]


_GATE_CANCELLATION_PASS = PassManager(
    InverseCancellation(
        gates_to_cancel=[
            XGate(),
            ZGate(),
            HGate(),
            CXGate(),
            CZGate(),
            (SGate(), SdgGate()),
            (SXGate(), SXdgGate()),
        ],
        run_default=True,
    )
)


def cancel_inverse_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Run a lightweight inverse-gate cancellation pass across the circuit.

    The pass removes back-to-back pairs of gates that undo each other, reducing
    redundant X/Z/S/H/CX/CZ/... instructions before further processing.
    """
    if circuit is None or circuit.num_qubits == 0 or len(circuit) == 0:
        return circuit
    return _GATE_CANCELLATION_PASS.run(circuit)


def cancel_inverse_gates_through_swaps(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Cancel inverse single-qubit gates even if separated by SWAPs.

    This pass tracks logical qubits through SWAPs, allowing cancellations like:
        H0; SWAP 0 1; H1  ->  SWAP 0 1
    but not:
        H0; SWAP 0 1; H0  ->  (no cancellation)
    """
    if circuit is None or circuit.num_qubits == 0 or len(circuit) == 0:
        return circuit

    inv_pairs = {
        "h": "h",
        "x": "x",
        "z": "z",
        "s": "sdg",
        "sdg": "s",
        "sx": "sxdg",
        "sxdg": "sx",
    }

    def _is_inverse(a_name: str, b_name: str) -> bool:
        return inv_pairs.get(a_name) == b_name

    num_qubits = circuit.num_qubits
    log_to_phys = list(range(num_qubits))
    phys_to_log = list(range(num_qubits))
    pending_by_log: dict[int, object] = {}

    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)

    def _emit_pending_for_log(log_idx: int) -> None:
        pending = pending_by_log.pop(log_idx, None)
        if pending is None:
            return
        inst, _, cargs = pending
        phys = log_to_phys[log_idx]
        new_circuit.append(inst, [circuit.qubits[phys]], cargs)

    for op in circuit.data:
        inst, qargs, cargs = _extract_instruction(op)
        gname = getattr(inst, "name", "").lower()

        if gname == "swap":
            if len(qargs) != 2:
                new_circuit.append(inst, qargs, cargs)
                continue
            qa = circuit.find_bit(qargs[0]).index
            qb = circuit.find_bit(qargs[1]).index
            la, lb = phys_to_log[qa], phys_to_log[qb]
            # Swap the logical->physical and physical->logical mappings.
            phys_to_log[qa], phys_to_log[qb] = lb, la
            log_to_phys[la], log_to_phys[lb] = qb, qa
            new_circuit.append(inst, qargs, cargs)
            continue

        if len(qargs) == 1 and gname in inv_pairs:
            phys = circuit.find_bit(qargs[0]).index
            log_idx = phys_to_log[phys]
            pending = pending_by_log.get(log_idx)
            if pending is not None:
                prev_inst, _, _ = pending
                prev_name = getattr(prev_inst, "name", "").lower()
                if _is_inverse(prev_name, gname):
                    pending_by_log.pop(log_idx, None)
                    continue
                _emit_pending_for_log(log_idx)
            pending_by_log[log_idx] = (inst, qargs, cargs)
            continue

        # For any other op, flush pending on touched qubits first.
        for q in qargs:
            phys = circuit.find_bit(q).index
            log_idx = phys_to_log[phys]
            _emit_pending_for_log(log_idx)

        new_circuit.append(inst, qargs, cargs)

    # Flush any remaining pending gates at the end.
    for log_idx in list(pending_by_log.keys()):
        _emit_pending_for_log(log_idx)

    return new_circuit


def _extract_instruction(op):
    if hasattr(op, "operation") and hasattr(op, "qubits"):
        return op.operation, op.qubits, getattr(op, "clbits", ())
    if isinstance(op, tuple) and len(op) >= 2:
        inst, qargs = op[0], op[1]
        cargs = op[2] if len(op) > 2 else []
        return inst, qargs, cargs
    raise TypeError("Unsupported circuit instruction format for H-commutation pass.")


# How each single-qubit gate looks through a pending Hadamard: H G H.
# `sx`/`sxdg` used to be missing here and `sdg` mapped to S rather than SX-dagger,
# so any program using them was rewritten into a different circuit.
_H_CONJUGATE = {
    "x": lambda inst: ZGate(),
    "z": lambda inst: XGate(),
    "s": lambda inst: SXGate(),
    "sdg": lambda inst: SXdgGate(),
    "sx": lambda inst: SGate(),
    "sxdg": lambda inst: SdgGate(),
    "rx": lambda inst: RZGate(inst.params[0]),
    "rz": lambda inst: RXGate(inst.params[0]),
}

# Gates that commute with H and so need no substitution.
_H_INVARIANT = {"h", "id", "barrier"}


def commute_hadamards(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Commute H gates through supported operations by flipping their basis.

    The pass removes explicit H instructions, toggling a per-qubit flag instead.
    When the flag is active, X<->Z substitutions are applied. For two-qubit
    operations, CX/CZ/XCX are rewritten based on which qubit has an active H flag:
      - CX  with active control -> XCX
      - CX  with active target  -> CZ
      - CZ  with active target  -> CX
      - XCX with active control -> CX          (H_a XCX_ab H_a == CX(a->b))
      - XCX with both active    -> CZ
    A SWAP exchanges the two qubits' states, so it also exchanges their flags.
    Any other multi-qubit operation flushes the frame on the qubits it touches by
    re-emitting explicit H gates, so an unrecognized gate can never be silently
    reinterpreted in the wrong basis.
    If the flag is still active at the end of the circuit, a trailing H is inserted.
    """
    if circuit is None or circuit.num_qubits == 0 or len(circuit) == 0:
        return circuit

    ops: list[tuple] = []
    h_active = [False] * circuit.num_qubits
    first_measure_index = None

    for op in circuit.data:
        inst, qargs, cargs = _extract_instruction(op)
        gname = getattr(inst, "name", "").lower()

        if gname == "h":
            q_index = circuit.find_bit(qargs[0]).index
            h_active[q_index] = not h_active[q_index]
            continue
        if gname == "measure":
            if first_measure_index is None:
                first_measure_index = len(ops)
            ops.append((inst, qargs, cargs))
            continue

        if len(qargs) == 1:
            q_index = circuit.find_bit(qargs[0]).index
            if h_active[q_index]:
                # Each entry is H G H, i.e. the gate seen through the pending
                # basis change. Verified against the matrices in
                # tests/test_translator_passes.py.
                replacement = _H_CONJUGATE.get(gname)
                if replacement is not None:
                    inst = replacement(inst)
                elif gname not in _H_INVARIANT:
                    # Unknown single-qubit gate: flush the frame rather than
                    # emit it in the wrong basis.
                    ops.append((HGate(), [qargs[0]], []))
                    h_active[q_index] = False
            ops.append((inst, qargs, cargs))
            continue

        if len(qargs) == 2 and gname == "swap":
            # A SWAP exchanges the two qubits' states, so the pending basis
            # change travels with them. Without this the H lands on the wrong
            # qubit and the circuit is silently wrong.
            a = circuit.find_bit(qargs[0]).index
            b = circuit.find_bit(qargs[1]).index
            h_active[a], h_active[b] = h_active[b], h_active[a]
            ops.append((inst, qargs, cargs))
            continue

        if len(qargs) == 2 and gname in {"cx", "cz", "xcx"}:
            ctrl_index = circuit.find_bit(qargs[0]).index
            tgt_index = circuit.find_bit(qargs[1]).index
            ctrl_active = h_active[ctrl_index]
            tgt_active = h_active[tgt_index]

            if gname == "cx":
                if ctrl_active and tgt_active:
                    inst = CXGate()
                    qargs = [qargs[1], qargs[0]]
                elif ctrl_active:
                    inst = XCXGate()
                elif tgt_active:
                    inst = CZGate()
            elif gname == "cz":
                if ctrl_active and tgt_active:
                    inst = XCXGate()
                elif ctrl_active:
                    inst = CXGate()
                    qargs = [qargs[1], qargs[0]]
                elif tgt_active:
                    inst = CXGate()
            else:  # xcx — symmetric, and the dual of the cz case
                if ctrl_active and tgt_active:
                    inst = CZGate()
                elif ctrl_active:
                    inst = CXGate()
                elif tgt_active:
                    inst = CXGate()
                    qargs = [qargs[1], qargs[0]]
            ops.append((inst, qargs, cargs))
            continue

        # Anything else: flush the frame on the qubits involved rather than
        # guessing how the gate transforms.
        for qubit in qargs:
            index = circuit.find_bit(qubit).index
            if h_active[index]:
                ops.append((HGate(), [qubit], []))
                h_active[index] = False
        ops.append((inst, qargs, cargs))

    trailing_ops = []
    for q in circuit.qubits:
        q_index = circuit.find_bit(q).index
        if h_active[q_index]:
            trailing_ops.append((HGate(), [q], []))

    if trailing_ops:
        insert_at = first_measure_index if first_measure_index is not None else len(ops)
        ops[insert_at:insert_at] = trailing_ops

    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for inst, qargs, cargs in ops:
        new_circuit.append(inst, qargs, cargs)

    return new_circuit


# ---------------------------------------------------------------------------
# is the rewrite worth doing?
# ---------------------------------------------------------------------------

#: Weighted-gate units per layer of depth, for `commutation_gain`.
#:
#: Not optional here. Commuting a Hadamard through a CNOT rewrites it to an XCX, and
#: on *gate counts alone that is an improvement* -- intra-patch CX is `(0, 4)` = 52
#: weighted units against XCX's `(5, 3)` = 44. The cost is entirely in depth, 4 layers
#: against 6. Price depth at zero and this check concludes that commutation always pays,
#: including on the circuits where it is catastrophic.
COMMUTATION_DEPTH_WEIGHT = 12.0


def _program_cost(
    circuit: QuantumCircuit, n_code: int, depth_weight: float
) -> float:
    """
    Weighted cost of a program circuit, priced per gate at its intra-patch construction.

    Placement-independent on purpose: this runs *before* anything is placed, and the
    decision it informs -- whether to rewrite gate kinds at all -- is a program-level
    one. Intra-patch is the right reference because it is where the CX/XCX difference
    this check exists to catch actually lives.
    """
    from iceberg.costs import DEFAULT_WORK_WEIGHTS, depth_table, work_table

    keys = {
        "x": "X", "y": "Y", "z": "Z", "h": "H", "s": "S", "sdg": "S",
        "sx": "SX", "sxdg": "SX", "rz": "RZ", "rx": "RX",
        "cx": "CX", "cz": "CZ", "xcx": "XCX", "swap": "SWAP",
    }
    work, depths = work_table(n_code), depth_table(n_code)
    w1, w2 = DEFAULT_WORK_WEIGHTS
    total = 0.0
    for instruction in circuit.data:
        key = keys.get(instruction.operation.name.lower())
        if key is None:
            continue
        cost = work.get(key, (0, 0))
        total += w1 * cost[0] + w2 * cost[1] + depth_weight * depths.get(key, 0)
    return total


def commutation_gain(
    circuit: QuantumCircuit,
    n_code: int,
    *,
    depth_weight: float = COMMUTATION_DEPTH_WEIGHT,
) -> float:
    """
    Weighted units `commute_hadamards` would save on `circuit`. Negative means it costs.

    Commutation removes a Hadamard by carrying a basis flip forward in a per-qubit
    frame, and pays for it by rewriting the two-qubit gates the frame passes through --
    CX becomes XCX or CZ, and so on. That trade is only good when the Hadamards
    actually *go away*. When the frame is still live at the end of the circuit the pass
    re-emits a trailing H, so the rewrite was paid for and nothing was saved.

    Measured on QASMBench large proxies, depth change from commutation alone:

        adder_n64    112 of 112 Hadamards absorbed    +21.4% at k=2
        knn_n67       68 of  68 absorbed               -2.0%
        bv_n70         0 of 139 absorbed                0.0%
        ising_n66      0 of  66 absorbed             -190.0%

    `ising_n66` is the case this exists for: every Hadamard survives, so the pass buys
    nothing and spends the whole rewrite. The damage reads worst at k=2 because the
    baseline there is shallowest -- more patches, more parallelism -- so the same
    absolute layers are a larger fraction.

    Accuracy, and why this is an estimate rather than the decision
    --------------------------------------------------------------
    Compared against translating both ways on 28 (circuit, code size) pairs, the sign
    of this number agreed with the measured outcome 21 times. Every disagreement was a
    false positive -- it recommended commuting on a circuit that commutation hurt, by
    up to 8,627 weighted units on `dnn_n51`.

    The reason is that the damage is in the *schedule*, not the gate list. Removing
    Hadamards also removes the whole-patch `hh` macros the merging pass would have
    fired, and lengthens the critical path; a per-gate sum sees neither. Use this only
    where a translation is unaffordable. `packing.passes.commutation_pays` decides by
    measurement and is what the pipeline actually calls.
    """
    before = cancel_inverse_gates(circuit)
    after = cancel_inverse_gates(commute_hadamards(before))
    return _program_cost(before, n_code, depth_weight) - _program_cost(
        after, n_code, depth_weight
    )

