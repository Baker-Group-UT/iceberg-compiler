"""
What happens when a merged macro cannot fire at translation time.

The merging pass replaces up to k program gates with one macro and *drops the
originals*. If the handler then decides the macro is inadmissible, something has to
put those gates back. Three defects conspired here, and each gets a test:

1. `translate_and_apply` discarded the return value of every compound handler, so a
   handler reporting "I emitted nothing" was read as success and the gates vanished.
   `Translation.translate_circuit` raises on `False`, and that guard was dead code for
   all ten names in `TABLE_COMPLEX` -- `cx`, `cz`, `xcx` and `h` among them.
2. `realign_target_slots` refused to align unless the control and target slot lists
   were a permutation of each other, which is only necessary when the patch is full.
3. `active_slots_in_patch` derived occupancy from *static* positions, so it went stale
   the moment an alignment swap moved a qubit into a previously-empty lane.

2 and 3 are the same mistake: assuming a patch is fully occupied. At k=2 an occupied
patch *is* full, which is why none of this showed while merging was [[4,2,2]]-only, and
why the existing suites missed it -- `test_merging_generalized.py` builds programs that
make each macro *fire*, never one that gets rejected.

The end-to-end cases below are the seeds that were silently wrong, checked against the
exact logical action rather than a sampled distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.random_circuits import random_gate_circuit
from tests.test_pipeline_logical_action import check_program
from translator.custom_gates import XCXGate
from translator.context import TranslationContext
from translator.gate_handlers import TABLE_COMPLEX, translate_and_apply
from translator.handlers.utils import active_slots_in_patch, realign_target_slots

#: The gate set `experiments.tests.testexp` fuzzes over, and `random_gate_circuit`'s
#: own default.
CLIFFORD_GATES = ["h", "x", "z", "s", "cx", "cz"]

#: (n_code, seed, permutation) triples whose macros get rejected at translation. Found
#: by the testexp fuzz; permutations are pinned rather than resampled so a change to the
#: permutation sampler cannot quietly make these vacuous.
REJECTING_CASES = [
    (8, 32, [(0, 4), (0, 2), (1, 3), (1, 0)]),
    (6, 63, None),
    (6, 66, None),
    (6, 97, None),
]


def program_for(seed: int) -> QuantumCircuit:
    return random_gate_circuit(4, 50, CLIFFORD_GATES, seed=seed)


def permutation_for(seed: int, n_code: int):
    """The permutation `generate_random_circuit` samples for this seed."""
    from permutations.code_sets import generate_uniform_code_sets
    from permutations.generator import random_permutation_from_circuit

    program = program_for(seed)
    return random_permutation_from_circuit(
        circuit=program,
        code_sizes=generate_uniform_code_sets(program.num_qubits, n=n_code),
        seed=seed,
    )


# --- 1. the failure has to reach the caller ---------------------------------------


def test_compound_handler_failure_reaches_the_caller():
    """
    A compound handler returning False must surface as False, not None.

    This is the defect that turned two false rejections into wrong answers instead of
    loud ones. Checked by substituting a handler that refuses, so the test does not
    depend on any particular gate being rejectable.
    """
    program = QuantumCircuit(2)
    program.cx(0, 1)
    context = TranslationContext.from_config(
        [(0, 0), (0, 1)], program, n_code=4, encode_override=True
    )
    op = program.data[0]

    original = TABLE_COMPLEX["cx"]
    TABLE_COMPLEX["cx"] = lambda op, context: False
    try:
        assert translate_and_apply(context, op) is False, (
            "a compound handler reporting failure was swallowed; "
            "Translation.translate_circuit would treat the dropped gate as success"
        )
    finally:
        TABLE_COMPLEX["cx"] = original


def test_compound_handler_success_is_still_success():
    """The propagation must not turn ordinary success into failure."""
    program = QuantumCircuit(2)
    program.cx(0, 1)
    context = TranslationContext.from_config(
        [(0, 0), (0, 1)], program, n_code=4, encode_override=True
    )
    assert translate_and_apply(context, program.data[0]) is None


# --- 2. realignment may use free lanes --------------------------------------------


@pytest.mark.parametrize(
    "control_slots, target_slots",
    [
        ([2, 4], [0, 3]),   # seed 32: disjoint, both reachable via empty lanes
        ([0, 1], [4, 5]),   # fully disjoint the other way
        ([1, 3], [1, 5]),   # partial overlap
        ([0, 2], [2, 0]),   # a true permutation, the case that always worked
    ],
)
def test_realign_uses_free_lanes_when_the_patch_is_not_full(control_slots, target_slots):
    """
    Any distinct target slots are reachable when the patch has room.

    `resolve_transversal_macro` checks coverage before realigning, so `target_slots` is
    every occupied lane of the patch and every other lane is free. The old permutation
    condition rejected the first three of these outright.
    """
    program = QuantumCircuit(4)
    for qubit in range(4):
        program.h(qubit)
    permutation = [(0, control_slots[0]), (0, control_slots[1]),
                   (1, target_slots[0]), (1, target_slots[1])]
    context = TranslationContext.from_config(
        permutation, program, n_code=8, encode_override=True
    )

    assert realign_target_slots(context, 1, control_slots, target_slots) is True
    assert active_slots_in_patch(context, 1) == set(control_slots), (
        "after realignment the target patch's occupied lanes must be the control lanes"
    )


def test_realign_rejects_malformed_requests():
    """Differing lengths or a repeated slot are still refused, emitting nothing."""
    program = QuantumCircuit(2)
    program.h(0)
    program.h(1)
    context = TranslationContext.from_config(
        [(0, 0), (1, 0)], program, n_code=8, encode_override=True
    )
    before = len(context.iceberg_set.circuit.data)
    assert realign_target_slots(context, 1, [0, 1], [0]) is False
    assert realign_target_slots(context, 1, [0, 0], [0, 1]) is False
    assert realign_target_slots(context, 1, [0, 1], [1, 1]) is False
    assert len(context.iceberg_set.circuit.data) == before


# --- 3. occupancy must be read live ----------------------------------------------


def test_active_slots_follow_swaps_into_empty_lanes():
    """
    Occupancy is a live property once a patch has free lanes.

    The old implementation computed `index % k_log` off the static permutation and
    documented the set as swap-invariant. That is true only for a full patch: swapping
    an occupied lane with a free one moves the set, and the next macro then read its own
    lanes as uncovered.
    """
    program = QuantumCircuit(2)
    program.h(0)
    program.h(1)
    context = TranslationContext.from_config(
        [(0, 0), (0, 3)], program, n_code=8, encode_override=True
    )
    assert active_slots_in_patch(context, 0) == {0, 3}

    # Slot 5 is empty at k=6, so this moves the occupied set rather than permuting it.
    context.iceberg_set.SWAP([[0, 3, 5]], fault_tolerant=False)
    context.update_states_on_swap(0, 3, 5)

    assert active_slots_in_patch(context, 0) == {0, 5}, (
        "occupancy was read from static positions and did not follow the swap"
    )


# --- 4. end to end: the seeds that were silently wrong ---------------------------


@pytest.mark.parametrize("n_code", [6, 8, 10])
@pytest.mark.parametrize("gate", ["cx", "cz", "xcx"])
def test_misaligned_cross_patch_gate_places_with_no_escape_hatch(n_code, gate):
    """
    A misaligned cross-patch gate must place even when all three FT predicates miss.

    The configuration that used to be refused: two patches whose occupied lanes are
    disjoint, so no lane matches (`is_full_transversal` false), neither patch holds a
    single qubit (`is_alone_in_patch` false), and the partners are not in `|0>`
    (`are_all_partners_zero` false). At k=2 an occupied patch is full so this cannot
    arise; at k>2 it is routine, and it cost 3 of 15 medium and up to 3 of 27 large
    QASMBench circuits at every k>2.

    The targeted construction is correct at any slot pair -- see
    `test_cross_patch_targeted_ops_any_slot_pair` -- so there was never anything to
    reject. Checked by exact tableau, not by whether it merely returned.
    """
    program = QuantumCircuit(4)
    for qubit in range(4):          # take the partners out of |0>, killing that hatch
        program.h(qubit)
    # cross-patch, and misaligned by construction. `xcx` has no QuantumCircuit method.
    if gate == "xcx":
        program.append(XCXGate(), [0, 2])
    else:
        getattr(program, gate)(0, 2)

    # patch 0 occupies slots {1, 3}; patch 1 occupies {0, 2}. Disjoint, so no lane pairs.
    permutation = [(0, 1), (0, 3), (1, 0), (1, 2)]
    check_program(program, permutation, n_code)


@pytest.mark.parametrize("n_code, seed, pinned", REJECTING_CASES)
def test_rejecting_macros_still_compute_the_program(n_code, seed, pinned):
    """
    A program whose macros get rejected must still translate to the right logical action.

    Exact tableau comparison, so this cannot pass on sampling luck. Every one of these
    seeds came out at fidelity ~0.5 before the fix; `seed 32` at k=6 was reported at
    TVD 0.75.
    """
    program = program_for(seed)
    permutation = pinned if pinned is not None else permutation_for(seed, n_code)
    check_program(program, permutation, n_code)
