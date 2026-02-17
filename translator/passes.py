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

__all__ = ["cancel_inverse_gates", "cancel_inverse_gates_through_swaps", "commute_hadamards"]


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


def commute_hadamards(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Commute H gates through supported operations by flipping their basis.

    The pass removes explicit H instructions, toggling a per-qubit flag instead.
    When the flag is active, X<->Z substitutions are applied. For two-qubit
    operations, CX/CZ are rewritten based on which qubit has an active H flag:
      - CX with active control -> XCX
      - CX with active target  -> CZ
      - CZ with active target  -> CX
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
                if gname == "x":
                    inst = ZGate()
                elif gname == "z":
                    inst = XGate()
                elif gname == "s":
                    inst = SXGate()
                elif gname == "sdg":
                    inst = SGate()
                elif gname == "rx":
                    inst = RZGate(inst.params[0])
                elif gname == "rz":
                    inst = RXGate(inst.params[0])
            ops.append((inst, qargs, cargs))
            continue

        if len(qargs) == 2 and gname in {"cx", "cz"}:
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
            else:
                if ctrl_active and tgt_active:
                    inst = XCXGate()
                elif ctrl_active:
                    inst = CXGate()
                    qargs = [qargs[1], qargs[0]]
                elif tgt_active:
                    inst = CXGate()
            ops.append((inst, qargs, cargs))
            continue

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
