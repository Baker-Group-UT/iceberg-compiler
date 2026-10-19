"""
check_majority_undo.py -- exact check of macro + undo merging (HH_MAJORITY_UNDO,
TRANSVERSAL_MAJORITY_UNDO) on programs built to trigger it.

Random Clifford programs almost never put an H on most lanes of a patch at once, so the
harness of `tests/test_merging_generalized.py` is reused with programs that do: H on a
strict majority of a patch's lanes, gates on a majority of two patches' lane pairs
(aligned and crossed), and random layers of both inside random Clifford context. Every
program is compared with the translated circuit's logical Clifford, exactly
(`tests.test_pipeline_logical_action.check_program`), and the undo records the pass
emitted are counted, so a pass that never fires cannot report a vacuous success.

    PYTHONPATH=. python -m experiments.tvd_explore.check_majority_undo
"""

from __future__ import annotations

import collections
import random
import sys

from qiskit import QuantumCircuit

from iceberg.utils import logical_capacity
from tests.random_programs import _append_two_qubit, random_clifford_program
from tests.test_pipeline_logical_action import check_program
from translator import explore_switches
from translator.merging import preprocessing

FIRED = collections.Counter()
_original_rewrite = preprocessing.build_rewritten_circuit


def _counting_rewrite(original, dag, records, layer_index, order_index):
    for record in records:
        if record.undo_ops:
            FIRED[record.macro_gate.name] += 1
            FIRED["undo_gates"] += len(record.undo_ops)
    return _original_rewrite(original, dag, records, layer_index, order_index)


preprocessing.build_rewritten_circuit = _counting_rewrite


def context(num_qubits, depth, rng):
    """Random Cliffords without program SWAPs, which the placement rules often refuse."""
    program = QuantumCircuit(num_qubits)
    for _ in range(depth):
        if rng.random() < 0.5:
            control, target = rng.sample(range(num_qubits), 2)
            _append_two_qubit(program, rng.choice(["cx", "cz", "xcx"]), control, target)
        else:
            getattr(program, rng.choice(["h", "s", "sx", "x", "z"]))(rng.randrange(num_qubits))
    return program


def sequential(num_qubits, k):
    return [(i // k, i % k) for i in range(num_qubits)]


def shuffled(num_qubits, k, rng):
    slots = sequential(num_qubits, k)
    rng.shuffle(slots)
    return slots


def majority_h(program, patch_qubits, rng):
    k = len(patch_qubits)
    wanted = rng.sample(patch_qubits, rng.randint(k // 2 + 1, k - 1))
    for q in wanted:
        program.h(q)


def majority_pairs(program, control_qubits, target_qubits, name, rng, crossed):
    k = len(control_qubits)
    targets = list(target_qubits)
    if crossed:
        rng.shuffle(targets)
    lanes = rng.sample(range(k), rng.randint(k // 2 + 1, k - 1))
    for lane in lanes:
        _append_two_qubit(program, name, control_qubits[lane], targets[lane])


def layered_program(num_patches, k, rng, layers=8):
    """Random layers of majority-H and majority-pair groups inside random Cliffords."""
    n = num_patches * k
    program = QuantumCircuit(n)
    patches = [list(range(p * k, (p + 1) * k)) for p in range(num_patches)]
    for _ in range(layers):
        roll = rng.random()
        if roll < 0.4:
            majority_h(program, rng.choice(patches), rng)
        elif roll < 0.8 and num_patches > 1:
            a, b = rng.sample(range(num_patches), 2)
            name = rng.choice(["cx", "cz", "xcx"])
            majority_pairs(program, patches[a], patches[b], name, rng, crossed=rng.random() < 0.3)
        else:
            program.compose(random_clifford_program(n, 3, rng=rng, include_xcx=True), inplace=True)
    return program


def run(n_code, programs, rng, label):
    k = logical_capacity(n_code)
    checked = rejected = 0
    for program, permutation in programs:
        try:
            check_program(program, permutation, n_code, enable_merging=True,
                          allow_transversal_swap_merges=True)
        except RuntimeWarning:
            rejected += 1
            continue
        checked += 1
    print(f"  n={n_code:2d} k={k:2d} {label:<22} checked {checked:3d}  rejected {rejected:2d}")
    return checked


def main():
    explore_switches.configure(HH_MAJORITY_UNDO=True, TRANSVERSAL_MAJORITY_UNDO=True)
    rng = random.Random(20261006)
    total = 0
    for n_code in (6, 8, 10, 12):
        k = logical_capacity(n_code)
        # 1. H on a strict majority of one patch, with and without surrounding context.
        progs = []
        for index in range(10):
            program = QuantumCircuit(2 * k)
            make = random_clifford_program if index % 2 else context
            kwargs = {"rng": rng} if index % 2 else {}
            program.compose(make(2 * k, 6, **kwargs) if kwargs else make(2 * k, 6, rng), inplace=True)
            majority_h(program, list(range(k)), rng)
            majority_h(program, list(range(k, 2 * k)), rng)
            program.compose(make(2 * k, 6, **kwargs) if kwargs else make(2 * k, 6, rng), inplace=True)
            for perm in (sequential(2 * k, k), shuffled(2 * k, k, rng)):
                progs.append((program, perm))
        total += run(n_code, progs, rng, "majority H")
        # 2. A gate on a majority of two patches' lane pairs, aligned and crossed.
        progs = []
        for name in ("cx", "cz", "xcx"):
            for crossed in (False, True):
                for _ in range(2):
                    program = QuantumCircuit(2 * k)
                    majority_pairs(program, list(range(k)), list(range(k, 2 * k)), name, rng, crossed)
                    progs.append((program, sequential(2 * k, k)))
        total += run(n_code, progs, rng, "majority pairs")
        # 3. Layered random programs over three patches, sequential and shuffled.
        progs = []
        for _ in range(6 if n_code <= 8 else 3):
            program = layered_program(3, k, rng)
            for perm in (sequential(3 * k, k), shuffled(3 * k, k, rng)):
                progs.append((program, perm))
        total += run(n_code, progs, rng, "layered, 3 patches")
        # 4. Uneven packing: last patch partly empty.
        progs = []
        for _ in range(4):
            n = 2 * k - 1
            program = QuantumCircuit(n)
            majority_h(program, list(range(k)), rng)
            program.compose(context(n, 8, rng), inplace=True)
            majority_h(program, list(range(k)), rng)
            progs.append((program, sequential(n, k)))
        total += run(n_code, progs, rng, "uneven packing")
    print(f"checked {total} programs exactly; undo records fired: {dict(FIRED)}")
    if not FIRED.get("hh") or not (FIRED.get("cxt") or FIRED.get("czt") or FIRED.get("xcxt")):
        print("WARNING: a macro family never fired with undo; the check is vacuous for it")
        sys.exit(1)


if __name__ == "__main__":
    main()
