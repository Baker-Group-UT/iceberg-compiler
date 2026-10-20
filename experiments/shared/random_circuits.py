"""
helpers.random_circuits.py - lightweight random circuit builders.
"""

from __future__ import annotations

import random
from math import pi
from typing import Iterable, List, Optional

from qiskit import QuantumCircuit

from translator.passes import cancel_inverse_gates

_ONE_QUBIT_GATES = {
    "h",
    "x",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
    "y",
    "rx",
    "rz",
}
_TWO_QUBIT_GATES = {"cx", "cz", "swap", "cy"}


def random_gate_circuit(
    num_qubits: int,
    num_gates: int,
    gates: Optional[Iterable[str]] = None,
    *,
    seed: Optional[int] = None,
    max_attempts: int = 50_000,
) -> QuantumCircuit:
    """
    Build a random circuit with uneven qubit depths and no immediate gate repeats.

    The returned circuit is post-processed with `cancel_inverse_gates` and will
    contain at least `num_gates` operations after cancellation.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    if num_gates < 0:
        raise ValueError("num_gates must be >= 0")

    if gates is None:
        gates = ["h", "x", "z", "s", "cx", "cz"]

    gate_list: List[str] = [str(name).lower() for name in gates]
    if not gate_list:
        return QuantumCircuit(num_qubits)

    rng = random.Random(seed)
    qc = QuantumCircuit(num_qubits)

    last_1q: List[Optional[str]] = [None] * num_qubits
    last_per_qubit: List[Optional[tuple[str, Optional[int], Optional[str]]]] = [None] * num_qubits

    def _append_one_qubit(name: str) -> bool:
        qubits = list(range(num_qubits))
        rng.shuffle(qubits)
        q = None
        for candidate in qubits:
            if last_1q[candidate] != name:
                q = candidate
                break
        if q is None:
            return False
        if name in {"rx", "rz"}:
            getattr(qc, name)(pi / 4, q)
        else:
            getattr(qc, name)(q)
        last_1q[q] = name
        last_per_qubit[q] = (name, None, None)
        return True

    def _append_two_qubit(name: str) -> bool:
        if num_qubits < 2:
            return False
        candidates = [
            (control, target)
            for control in range(num_qubits)
            for target in range(num_qubits)
            if control != target
        ]
        rng.shuffle(candidates)
        pair = None
        for control, target in candidates:
            if (
                last_per_qubit[control] == (name, target, "control")
                and last_per_qubit[target] == (name, control, "target")
            ):
                continue
            pair = (control, target)
            break
        if pair is None:
            return False
        control, target = pair
        getattr(qc, name)(control, target)
        last_1q[control] = None
        last_1q[target] = None
        last_per_qubit[control] = (name, target, "control")
        last_per_qubit[target] = (name, control, "target")
        return True

    cleaned = cancel_inverse_gates(qc)
    attempts = 0
    while len(cleaned.data) < num_gates:
        if attempts >= max_attempts:
            raise RuntimeError("Failed to build circuit without excessive cancellations.")
        attempts += 1
        name = rng.choice(gate_list)
        if name in _ONE_QUBIT_GATES:
            appended = _append_one_qubit(name)
        elif name in _TWO_QUBIT_GATES:
            appended = _append_two_qubit(name)
        else:
            raise ValueError(f"Unsupported gate name: {name}")

        if appended:
            cleaned = cancel_inverse_gates(qc)

    return cleaned


def random_program_circuit(
    num_qubits: int,
    gates_per_qubit: int,
    seed: int,
    *,
    immutable_ratio: float = 0.15,
    clifford_only: bool = True,
) -> QuantumCircuit:
    """
    A random circuit over the translator's gate set, with a controlled immutable share.

    `immutable_ratio` is the fraction of S/Rx/Rz -- gates merging cannot fuse, so they
    bound how much of the circuit the pass can reach. The paper's point about random
    circuits is that they carry fewer of these than the benchmarks do, which is why
    merging and packing find more to work with; making the share a knob is what lets a
    figure sweep that difference instead of asserting it.

    `clifford_only` restricts the immutable gates to S, and defaults to True here
    because anything that will be *simulated* needs it: the Stim backend rejects Rx and
    Rz outright, which is the same reason the paper simulates rotation-bearing
    benchmarks through Clifford proxies rather than directly. Set it False for
    translation-only figures, where the rotations are the point.

    Unlike `random_gate_circuit` this does not run `cancel_inverse_gates`, so the gate
    mix the caller asked for is the mix the circuit has. Cancellation would eat H pairs
    preferentially and quietly raise the immutable fraction above the requested one.
    """
    if not 0.0 <= immutable_ratio <= 0.6:
        raise ValueError("immutable_ratio must be in [0, 0.6]; two-qubit gates take 0.4")

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for _ in range(num_qubits * gates_per_qubit):
        roll = rng.random()
        if roll < 0.4 and num_qubits >= 2:
            left, right = rng.sample(range(num_qubits), 2)
            getattr(circuit, rng.choice(("cx", "cz")))(left, right)
        elif roll < 0.4 + immutable_ratio:
            _append_immutable(circuit, rng, clifford_only)
        else:
            getattr(circuit, rng.choice(("h", "x", "z")))(rng.randrange(num_qubits))
    return circuit


def _append_immutable(circuit: QuantumCircuit, rng, clifford_only: bool) -> None:
    """An S or a rotation: gates merging cannot fuse, so they bound its reach."""
    qubit = rng.randrange(circuit.num_qubits)
    choice = "s" if clifford_only else rng.choice(("s", "rz", "rx"))
    if choice == "s":
        circuit.s(qubit)
    else:
        getattr(circuit, choice)(rng.uniform(0.1, 3.0), qubit)


def immutable_fraction(circuit: QuantumCircuit) -> float:
    """Share of operations merging cannot fuse. What figure 10 plots against."""
    total = len(circuit.data)
    if total == 0:
        return 0.0
    immutable = sum(
        1 for op in circuit.data if op.operation.name in {"s", "sdg", "rz", "rx", "t", "tdg"}
    )
    return immutable / total


def plant_blocks(num_qubits: int, block_size: int, seed: int) -> List[List[int]]:
    """
    Partition qubits into blocks of `block_size`, on *shuffled* labels.

    The shuffle is load-bearing, not tidiness. If the planted blocks were
    `[0..b-1], [b..2b-1], ...` then the naive index-order packing is already the correct
    grouping, and a packer that recovers the structure perfectly still measures zero
    improvement -- which is exactly what the first version of this experiment measured
    (see `archive/grouping/evaluate.py`). Shuffling makes recovering the structure a real
    achievement rather than a coincidence.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    labels = list(range(num_qubits))
    random.Random(seed).shuffle(labels)
    return [labels[start:start + block_size] for start in range(0, num_qubits, block_size)]


def structured_program_circuit(
    num_qubits: int,
    gates_per_qubit: int,
    seed: int,
    *,
    blocks: Optional[List[List[int]]] = None,
    block_size: int = 2,
    locality: float = 0.85,
    two_qubit_ratio: float = 0.40,
    immutable_ratio: float = 0.0,
    clifford_only: bool = True,
    period: Optional[int] = None,
    cross_topology: str = "any",
) -> QuantumCircuit:
    """
    A random circuit whose interaction graph has community structure.

    `random_program_circuit` draws its two-qubit gates from a uniform random pair, which
    makes the interaction graph an Erdos-Renyi graph: every pair interacts about equally
    often, so a min-cut carries no signal and *every* partition is as good as any other.
    Packing cannot beat naive on such a circuit for a structural reason, and measuring it
    there says nothing about the packer. This generator plants blocks that interact mostly
    among themselves, so there is something to find.

    Knobs, and what each one is for:

    `locality`      probability a two-qubit gate stays inside its block. 1.0 is perfectly
                    separable, and around 0.5 collapses to the uniform case above. This is
                    the separability axis -- the natural x for a curve or a heatmap.
    `block_size`    size of a planted community. Independent of the code's k on purpose,
                    so a sweep can ask what happens when the natural community size does
                    not divide patch capacity.
    `two_qubit_ratio` / `immutable_ratio`
                    gate mix. The remainder is single-qubit H/X/Z. Immutable means S (or a
                    rotation when `clifford_only` is False) -- gates merging cannot fuse.
    `period`        when set, blocks take turns in round-robin over this many steps rather
                    than being sampled independently. That is temporal structure: a qubit
                    idles while other blocks are active, which is what drives idle noise.
                    None keeps the memoryless behaviour.
    `cross_topology` where a gate goes when it leaves its block. "ring" sends it to a
                    neighbouring block, giving the interaction graph a line/ring geometry
                    so movement distance means something; "any" picks a random other block.

    `clifford_only` defaults True because anything simulated needs it -- the Stim backend
    rejects Rx and Rz outright.
    """
    if not 0.0 <= locality <= 1.0:
        raise ValueError("locality must be in [0, 1]")
    if two_qubit_ratio < 0 or immutable_ratio < 0 or two_qubit_ratio + immutable_ratio > 1:
        raise ValueError("two_qubit_ratio + immutable_ratio must be in [0, 1]")
    if cross_topology not in {"any", "ring"}:
        raise ValueError("cross_topology must be 'any' or 'ring'")

    if blocks is None:
        blocks = plant_blocks(num_qubits, block_size, seed)
    usable = [block for block in blocks if len(block) >= 2]

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)
    for step in range(num_qubits * gates_per_qubit):
        roll = rng.random()
        if roll < two_qubit_ratio and num_qubits >= 2 and usable:
            index = (step % period) % len(usable) if period else rng.randrange(len(usable))
            block = usable[index]
            if rng.random() < locality:
                left, right = rng.sample(block, 2)
            else:
                left, right = _cross_block_pair(blocks, index, rng, cross_topology)
            getattr(circuit, rng.choice(("cx", "cz")))(left, right)
        elif roll < two_qubit_ratio + immutable_ratio:
            _append_immutable(circuit, rng, clifford_only)
        else:
            getattr(circuit, rng.choice(("h", "x", "z")))(rng.randrange(num_qubits))
    return circuit


def _cross_block_pair(blocks, index: int, rng, cross_topology: str):
    """One qubit from block `index`, one from another block."""
    others = [i for i in range(len(blocks)) if i != index and blocks[i]]
    if not others:
        return tuple(rng.sample(blocks[index], 2))
    if cross_topology == "ring":
        # Neighbour in the block ordering, so the interaction graph inherits a line/ring
        # geometry and cross-patch distance is a meaningful quantity.
        neighbours = [i for i in ((index - 1) % len(blocks), (index + 1) % len(blocks))
                      if i in others]
        target = rng.choice(neighbours or others)
    else:
        target = rng.choice(others)
    return rng.choice(blocks[index]), rng.choice(blocks[target])


def intra_block_fraction(circuit: QuantumCircuit, blocks: List[List[int]]) -> float:
    """
    Share of two-qubit gates whose endpoints share a block -- separability, as measured.

    The realised value of `locality`, which is what a figure should plot: the requested
    value is the control, this is what the circuit actually came out with.
    """
    owner = {qubit: index for index, block in enumerate(blocks) for qubit in block}
    inside = total = 0
    for inst in circuit.data:
        if len(inst.qubits) != 2:
            continue
        total += 1
        a, b = (circuit.find_bit(q).index for q in inst.qubits)
        inside += owner.get(a, -1) == owner.get(b, -2)
    return inside / total if total else 0.0
