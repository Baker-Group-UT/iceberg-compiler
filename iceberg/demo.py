"""
demo.py — encode a logical circuit onto [[n, n-2, 2]] patches and show the result.

Run it:

    python -m iceberg.demo                      # default program at n = 4, 6, 8
    python -m iceberg.demo --n 6                # one code size
    python -m iceberg.demo --n 6 --per-gate     # each logical op's expansion, alone
    python -m iceberg.demo --n 4 --translator   # the production compiler, for comparison
    python -m iceberg.demo --fault-tolerant     # flagged gadgets
    python -m iceberg.demo --no-draw            # counts only

`encode_program` here is a *reference* encoder, not the compiler: it places logical
qubit i at (patch i // k, slot i % k) and expands each gate in order, with no
merging, no macro gates and no fault-tolerance placement rules. The compiler's job
is to do better than this — choose a placement, merge Hadamards into whole-patch
HH, promote aligned pairs to transversal macros. Use `--translator` to see that,
though `translator.TranslationContext` still hardcodes n=4 (GENERALIZATION_PLAN.md
phase 2), so the production path is n=4-only for now.

Because this encoder is n-general it doubles as the correctness reference for
n > 4: `--verify` simulates the encoded circuit, postselects, and compares the
logical distribution against the unencoded program.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Sequence

from qiskit import QuantumCircuit

from .iceberg_set import IcebergSet
from .layout import CodeLayout

__all__ = ["encode_program", "demo_program", "SUPPORTED_GATES"]


# --------------------------------------------------------------------------
# reference encoder
# --------------------------------------------------------------------------

SUPPORTED_GATES = (
    "x", "y", "z", "h", "s", "sx", "rz", "rx", "rzz", "cx", "cz", "swap", "xcx",
    "barrier", "measure",
)


def encode_program(
    program: QuantumCircuit,
    n: int,
    *,
    fault_tolerant: bool = False,
    decode: bool = True,
) -> tuple[QuantumCircuit, IcebergSet]:
    """
    Encode `program` onto [[n, n-2, 2]] patches, gate by gate, in order.

    Logical qubit i goes to (patch i // k, slot i % k). Returns the physical
    circuit and the `IcebergSet` that owns it. With `decode=True` the circuit ends
    with syndrome extraction and a data measurement, ready to simulate and
    postselect.
    """
    circuit = QuantumCircuit()
    iceberg = IcebergSet(n_program=program.num_qubits, n=n, circuit=circuit)
    iceberg.encode()

    for instruction in program.data:
        name = instruction.operation.name
        qubits = [program.find_bit(q).index for q in instruction.qubits]
        params = list(instruction.operation.params)
        _apply(iceberg, name, qubits, params, fault_tolerant)

    if decode:
        iceberg.decode()
        iceberg.measure()
    return circuit, iceberg


def _apply(
    iceberg: IcebergSet,
    name: str,
    qubits: Sequence[int],
    params: Sequence[float],
    fault_tolerant: bool,
) -> None:
    ft = fault_tolerant
    if name == "barrier":
        iceberg.BARRIER()
    elif name == "measure":
        pass  # the encoder adds its own decode + measure at the end
    elif name == "x":
        iceberg.X(qubits[0])
    elif name == "z":
        iceberg.Z(qubits[0])
    elif name == "y":
        iceberg.X(qubits[0])
        iceberg.Z(qubits[0])
    elif name == "h":
        iceberg.H(qubits[0], fault_tolerant=ft)
    elif name == "s":
        iceberg.S(qubits[0])
    elif name == "sx":
        iceberg.SX(qubits[0], fault_tolerant=ft)
    elif name == "rz":
        iceberg.RZ(qubits[0], float(params[0]), fault_tolerant=ft)
    elif name == "rx":
        iceberg.RX(qubits[0], float(params[0]), fault_tolerant=ft)
    elif name == "rzz":
        iceberg.RZZ(qubits[0], qubits[1], float(params[0]), fault_tolerant=ft)
    elif name == "cx":
        iceberg.CNOT(qubits[0], qubits[1], fault_tolerant=ft)
    elif name == "cz":
        iceberg.CZ(qubits[0], qubits[1], fault_tolerant=ft)
    elif name == "xcx":
        iceberg.XCX(qubits[0], qubits[1], fault_tolerant=ft)
    elif name == "swap":
        _apply_swap(iceberg, qubits[0], qubits[1], ft)
    else:
        raise NotImplementedError(
            f"gate {name!r} is not in the reference encoder's set: "
            f"{', '.join(SUPPORTED_GATES)}"
        )


def _apply_swap(iceberg: IcebergSet, i: int, j: int, fault_tolerant: bool) -> None:
    """Intra-patch SWAP is one instruction; across patches it takes three CNOTs."""
    patch_i, slot_i = iceberg.locate(i)
    patch_j, slot_j = iceberg.locate(j)
    if patch_i == patch_j:
        iceberg.SWAP([[patch_i, slot_i, slot_j]], fault_tolerant=fault_tolerant)
        return
    iceberg.CNOT(i, j, fault_tolerant=fault_tolerant)
    iceberg.CNOT(j, i, fault_tolerant=fault_tolerant)
    iceberg.CNOT(i, j, fault_tolerant=fault_tolerant)


# --------------------------------------------------------------------------
# the example program
# --------------------------------------------------------------------------


def demo_program(num_qubits: int = 4) -> QuantumCircuit:
    """
    A small program that exercises most of the gate set.

    Kept at a fixed logical width so the same program can be encoded at several
    code sizes: at n=4 it needs two patches and several gates land across them,
    while at n=6 everything fits in one patch. That contrast is the point.

    Deliberately stays inside what the production translator accepts, so
    `--translator` has something to show. The reference encoder is less
    restricted — it will also place a cross-patch SWAP, which the translator
    rejects under its fault-tolerance rules.
    """
    if num_qubits < 2:
        raise ValueError("need at least 2 logical qubits")
    qc = QuantumCircuit(num_qubits, name="demo")
    qc.h(0)
    qc.cx(0, 1)
    if num_qubits >= 4:
        qc.h(2)
        qc.cx(2, 3)
        qc.cx(1, 2)   # crosses patches at n=4, with *misaligned* slots
        qc.s(3)
        qc.cz(0, 3)   # also cross-patch and misaligned at n=4
        qc.swap(0, 1)
    qc.rz(0.4, 0)
    qc.sx(1)
    return qc


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def gate_counts(circuit: QuantumCircuit) -> tuple[int, int]:
    """(1q, 2q) physical gate counts, ignoring barriers and measurements."""
    one = two = 0
    for instruction in circuit.data:
        if instruction.operation.name in ("barrier", "measure"):
            continue
        if len(instruction.qubits) == 1:
            one += 1
        else:
            two += 1
    return one, two


def qubit_name(circuit: QuantumCircuit, qubit) -> str:
    """`register[index]`, matching the wire labels in the circuit drawing."""
    register, index = circuit.find_bit(qubit).registers[0]
    return f"{register.name}[{index}]"


def describe_layout(iceberg: IcebergSet) -> str:
    """Which physical qubit plays which role, so the drawing can be read."""
    layout = iceberg.layout
    lines = []
    for patch, module in enumerate(iceberg.modules):
        name = qubit_name(iceberg.circuit, module.x_anchor).split("[")[0]
        roles = [
            f"{name}[{layout.x_anchor}] = X-anchor",
            f"{name}[{layout.lane(0)}..{layout.lane(layout.k - 1)}]"
            f" = slots 0..{layout.k - 1}",
            f"{name}[{layout.z_anchor}] = Z-anchor",
        ]
        lines.append(f"  patch {patch}: " + ",  ".join(roles))
    return "\n".join(lines)


def report(
    program: QuantumCircuit,
    n: int,
    *,
    fault_tolerant: bool,
    draw: bool,
    verify: bool,
) -> dict:
    layout = CodeLayout(n)
    circuit, iceberg = encode_program(program, n, fault_tolerant=fault_tolerant)
    one, two = gate_counts(circuit)
    idle = iceberg.num_modules * layout.k - program.num_qubits

    print()
    print("=" * 78)
    print(
        f" [[{n},{layout.k},2]]   k={layout.k} slots/patch"
        f"   {iceberg.num_modules} patch(es)"
        f"   {circuit.num_qubits} physical qubits"
        + ("   (fault-tolerant)" if fault_tolerant else "")
    )
    print("=" * 78)
    print(describe_layout(iceberg))
    print(
        f"  {one} single-qubit + {two} two-qubit gates, depth {circuit.depth()}"
        f"   ({circuit.num_clbits} measured bits)"
        + (f",  {idle} idle slot(s)" if idle else "")
    )

    if draw:
        print()
        print(circuit.draw(output="text", fold=110, idle_wires=False))

    result = {"n": n, "k": layout.k, "patches": iceberg.num_modules,
              "qubits": circuit.num_qubits, "1q": one, "2q": two,
              "depth": circuit.depth()}
    if verify:
        result["verified"] = _verify(program, circuit)
    return result


def logical_distribution(
    circuit: QuantumCircuit,
    num_program_qubits: int,
    *,
    shots: int = 40_000,
) -> tuple[Dict[str, float], float]:
    """
    Simulate `circuit` noiselessly, postselect, and project onto the program.

    Returns (distribution keyed by logical qubit 0 leftmost, selection rate).
    The decoded string carries every slot of every patch, indexed
    `patch * k + slot`; under `encode_program`'s identity placement logical qubit
    i sits at index i and any idle slots come after, so the tail is dropped.

    Unseeded, which is now a choice rather than a workaround: a fixed seed used to make
    every one of `Simulator`'s `repetitions // 100` chunks a replay of the first. Chunks
    draw their own seeds from the configured one as of this revision.
    """
    from simulator.simulator import Simulator, SimulatorConfig

    from .iceberg_postselect import iceberg_postselect

    # Aer explicitly, against the Stim default. This function checks that the reference
    # encoder reproduces an *arbitrary* program, and `demo_program` earns that word by
    # including an rz -- which Stim, being a Clifford simulator, refuses outright. The
    # rest of the project routes rotation-bearing circuits through Clifford proxies
    # precisely so Stim can run them; the encoder check is the one place that must not.
    simulator = Simulator(
        SimulatorConfig(repetitions=shots, noiseless=True, stim=False)
    ).with_shot_progress(False)
    result = simulator.run(circuit)
    logical, selection_rate = iceberg_postselect(result.ideal_dist, result.key_order)

    projected: Dict[str, float] = {}
    for key, probability in logical.items():
        head = key[:num_program_qubits]
        projected[head] = projected.get(head, 0.0) + probability
    return projected, selection_rate


def reference_distribution(program: QuantumCircuit) -> Dict[str, float]:
    """Exact distribution of the unencoded program, qubit 0 leftmost."""
    from qiskit.quantum_info import Statevector

    return {
        key[::-1]: p
        for key, p in Statevector.from_instruction(program).probabilities_dict().items()
        if p > 1e-12
    }


def _verify(program: QuantumCircuit, circuit: QuantumCircuit) -> bool:
    """Simulate, postselect, and compare against the unencoded distribution."""
    if circuit.num_qubits > 26:
        print("  verification skipped: too many physical qubits to simulate")
        return False

    got, selection_rate = logical_distribution(circuit, program.num_qubits)
    want = reference_distribution(program)
    worst = max(
        abs(got.get(key, 0.0) - want.get(key, 0.0)) for key in set(got) | set(want)
    )
    ok = worst < 0.02
    print(
        f"  logical distribution vs unencoded reference: "
        f"{'MATCH' if ok else 'MISMATCH'} (max deviation {worst:.4f}, "
        f"selection rate {selection_rate:.3f})"
    )
    return ok


def per_gate(n: int, *, fault_tolerant: bool) -> None:
    """Show each logical operation's physical expansion, one at a time."""
    layout = CodeLayout(n)
    k = layout.k
    cases: Dict[str, Callable[[IcebergSet], None]] = {
        "X(0)": lambda ib: ib.X(0),
        "Z(0)": lambda ib: ib.Z(0),
        "S(0)": lambda ib: ib.S(0),
        "SX(0)": lambda ib: ib.SX(0, fault_tolerant=fault_tolerant),
        "H(0)": lambda ib: ib.H(0, fault_tolerant=fault_tolerant),
        "HH (all slots)": lambda ib: ib.HH([0], fault_tolerant=fault_tolerant),
        "RZ(0, 0.4)": lambda ib: ib.RZ(0, 0.4, fault_tolerant=fault_tolerant),
        "RX(0, 0.4)": lambda ib: ib.RX(0, 0.4, fault_tolerant=fault_tolerant),
        "CNOT(0, 1) intra-patch": lambda ib: ib.CNOT(0, 1, fault_tolerant=fault_tolerant),
        "CZ(0, 1) intra-patch": lambda ib: ib.CZ(0, 1, fault_tolerant=fault_tolerant),
        "XCX(0, 1) intra-patch": lambda ib: ib.XCX(0, 1, fault_tolerant=fault_tolerant),
        "SWAP(0, 1) intra-patch": lambda ib: ib.SWAP([[0, 0, 1]], fault_tolerant=fault_tolerant),
        f"CNOT(0, {k}) cross-patch, same slot": lambda ib: ib.CNOT(0, k, fault_tolerant=fault_tolerant),
        f"CNOT(0, {k + k - 1}) cross-patch, misaligned slots":
            lambda ib: ib.CNOT(0, k + k - 1, fault_tolerant=fault_tolerant),
        f"CZ(0, {k}) cross-patch": lambda ib: ib.CZ(0, k, fault_tolerant=fault_tolerant),
        f"XCX(0, {k}) cross-patch": lambda ib: ib.XCX(0, k, fault_tolerant=fault_tolerant),
        "CNOT_k_Transversal (all k slots at once)": lambda ib: ib.CNOT_k_Transversal(0, 1),
        "CZ_k_Transversal": lambda ib: ib.CZ_k_Transversal(0, 1),
        "XCX_k_Transversal": lambda ib: ib.XCX_k_Transversal(0, 1),
        "encode": lambda ib: None,
        "decode + measure": lambda ib: (ib.decode(), ib.measure()),
    }

    print()
    print("=" * 78)
    print(f" [[{n},{k},2]] — physical expansion of each logical operation")
    print("=" * 78)
    for label, apply in cases.items():
        circuit = QuantumCircuit()
        iceberg = IcebergSet(n_program=2 * k, n=n, circuit=circuit)
        if label == "encode":
            iceberg.encode()
        else:
            for module in iceberg.modules:
                module._encoded = True
            iceberg._encoded = True
            apply(iceberg)
        one, two = gate_counts(circuit)
        print(f"\n  {label}   ->   ({one} 1q, {two} 2q)")
        for instruction in circuit.data:
            if instruction.operation.name == "barrier":
                continue
            operands = ", ".join(qubit_name(circuit, q) for q in instruction.qubits)
            params = instruction.operation.params
            suffix = f"({params[0]:.4g})" if params and isinstance(params[0], float) else ""
            print(f"      {instruction.operation.name}{suffix} {operands}")


def via_translator(program: QuantumCircuit, *, fault_tolerant: bool, draw: bool) -> None:
    """Run the production compiler, for comparison with the reference encoder."""
    from translator.entrypoints import translate_circuit

    permutation = [(i // 2, i % 2) for i in range(program.num_qubits)]
    print()
    print("=" * 78)
    print(" production compiler (translator + merging), [[4,2,2]] only")
    print("=" * 78)
    print(f"  permutation (logical -> (patch, slot)): {permutation}")
    try:
        result = translate_circuit(program, permutation, fault_tolerant=fault_tolerant)
    except Exception as exc:  # placement can be rejected under the FT rules
        print(f"  translation failed: {type(exc).__name__}: {exc}")
        print(
            "  the translator rejects some placements the code itself supports —"
            " a cross-patch SWAP is the usual cause"
        )
        return

    one, two = gate_counts(result.circuit)
    print(
        f"  {one} single-qubit + {two} two-qubit gates, depth {result.circuit.depth()}"
        f"   ({result.circuit.num_qubits} physical qubits)"
    )
    print(
        f"  merge savings: {result.merge_savings_1q} 1q, {result.merge_savings_2q} 2q, "
        f"{result.merge_savings_weighted} weighted"
    )
    for entry in result.context.fault_log:
        print(f"  fault log: {entry}")
    if draw:
        print()
        print(result.circuit.draw(output="text", fold=110, idle_wires=False))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Encode a logical circuit onto [[n, n-2, 2]] Iceberg patches.",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[4, 6, 8],
        help="code size(s); must be even and >= 4 (default: 4 6 8)",
    )
    parser.add_argument(
        "--qubits", type=int, default=4, help="logical width of the demo program",
    )
    parser.add_argument(
        "--fault-tolerant", action="store_true", help="use the flagged FT gadgets",
    )
    parser.add_argument(
        "--no-draw", dest="draw", action="store_false", help="print counts but no circuit",
    )
    parser.add_argument(
        "--per-gate", action="store_true",
        help="instead of a program, expand each logical operation on its own",
    )
    parser.add_argument(
        "--translator", action="store_true",
        help="also run the production compiler (n=4 only)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="simulate and check the logical distribution against the unencoded program",
    )
    args = parser.parse_args(argv)

    for n in args.n:
        CodeLayout(n)  # fail fast on an invalid code size

    if args.per_gate:
        for n in args.n:
            per_gate(n, fault_tolerant=args.fault_tolerant)
        return 0

    program = demo_program(args.qubits)
    print("=" * 78)
    print(f" logical program: {program.num_qubits} qubits, {len(program.data)} gates")
    print("=" * 78)
    print(program.draw(output="text", fold=110))

    rows = [
        report(
            program, n,
            fault_tolerant=args.fault_tolerant,
            draw=args.draw,
            verify=args.verify,
        )
        for n in args.n
    ]

    if args.translator:
        via_translator(program, fault_tolerant=args.fault_tolerant, draw=args.draw)

    print()
    print("=" * 78)
    print(" summary (reference encoder, identity placement, no merging)")
    print("=" * 78)
    header = f"{'code':>12s} {'slots':>6s} {'patches':>8s} {'phys q':>7s} {'1q':>6s} {'2q':>6s} {'depth':>6s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        code = "[[{},{},2]]".format(row["n"], row["k"])
        print(
            f"{code:>12s} {row['k']:>6d} {row['patches']:>8d} "
            f"{row['qubits']:>7d} {row['1q']:>6d} {row['2q']:>6d} {row['depth']:>6d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
