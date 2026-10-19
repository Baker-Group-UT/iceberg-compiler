"""
passes.py — decide whether the translator's program-level passes are worth running.

`commute_hadamards` removes a Hadamard by carrying a basis flip forward in a per-qubit
frame, and pays for it by rewriting the two-qubit gates that frame passes through: CX
becomes XCX or CZ, CZ becomes CX, and so on. That is a good trade when the Hadamards
actually go away and a bad one when they do not -- if the frame is still live at the
end, the pass re-emits a trailing H, so the rewrite was paid for and nothing was saved.

How bad it gets, measured on QASMBench large proxies at the naive placement:

    adder_n64    112 of 112 Hadamards absorbed   commutation saves 11,272 units
    ising_n66      0 of  66 absorbed             commutation costs     422
    dnn_n51       52 of  52 absorbed             commutation costs   8,627
    qft_n29       29 of  29 absorbed by count,   commutation costs     856
                  none in effect

`dnn_n51` is why this module measures rather than estimates. Every Hadamard is
absorbed, so any accounting of the rewrite says it paid -- and it still cost 8,627
weighted units, because the damage is in the *schedule*, not the gate list. Removing
Hadamards also removes whole-patch `hh` macros the merging pass would have fired, and
lengthens the critical path in ways no per-gate sum can see.

So: translate both ways on the naive placement and compare. Two translations, decided
once per (program, code size) and cached. `translator.passes.commutation_gain` is the
cheap program-level estimate and is kept for callers who cannot afford a translation,
but it agreed with the measurement on only 21 of 28 cases here -- and every
disagreement was a false positive, recommending commutation on a circuit it hurt.
"""

from __future__ import annotations

from typing import Dict, Tuple

from qiskit import QuantumCircuit

__all__ = ["commutation_pays", "clear_cache"]

#: (circuit fingerprint, code size, merging) -> decision.
#:
#: Keyed by *content*, not by `id(program)`. CPython reuses ids, so an identity key served
#: a dead circuit's answer to a different circuit after about 155 allocations in a demo --
#: harmless inside one `pack()` call where the circuit stays alive, and wrong in exactly
#: the build-pack-discard loop a sweep or a figure script runs. `merging` is in the key
#: too: the decision is measured with merging matching what the caller will run, so two
#: callers differing in it were sharing one answer.
_CACHE: Dict[Tuple[Tuple, int, bool, str, bool], bool] = {}


def _fingerprint(program: QuantumCircuit) -> Tuple:
    """
    A hashable summary of what the circuit *is*, cheap enough to take per call.

    Gate name plus qubit indices plus parameters, in order -- everything the commutation
    decision can depend on. Costs one pass over the gate list, against the two full
    translations the cache exists to avoid, so the trade is not close.
    """
    return (
        program.num_qubits,
        tuple(
            (
                instruction.operation.name,
                tuple(program.find_bit(q).index for q in instruction.qubits),
                tuple(float(p) for p in instruction.operation.params),
            )
            for instruction in program.data
        ),
    )


def clear_cache() -> None:
    """Forget cached decisions. For tests, and for callers reusing circuit objects."""
    _CACHE.clear()


def commutation_pays(
    program: QuantumCircuit,
    n_code: int,
    *,
    merging: bool = True,
    use_cache: bool = True,
) -> bool:
    """
    Whether H commutation improves this circuit at this code size.

    Measured on the naive packing, with `merging` matching what the caller will run,
    because the two passes interact: commutation removes the Hadamards that whole-patch
    `hh` macros are built from, so judging it with merging off would miss the cost.

    Ties and rejections resolve to `True`, keeping the pipeline's existing default.
    """
    from iceberg.gadget_options import transversal_intra_enabled

    # The gadget options are part of the question: under transversal in-patch CZ / XCX the
    # same circuit can prefer the other setting. Keyed on the live flag, so a process that
    # plans under both never reuses an answer measured under the other.
    from translator.logical_optimization import state as logical_state

    # The logical optimisation pass is part of it as well: on, both translations below see
    # the optimised program, which can prefer the other setting.
    key = (_fingerprint(program), n_code, merging, logical_state(), transversal_intra_enabled())
    if use_cache and key in _CACHE:
        return _CACHE[key]

    from iceberg.utils import logical_capacity

    from .plan import measured_cost, naive_packing

    naive = naive_packing(program.num_qubits, logical_capacity(n_code))
    with_pass = measured_cost(
        program, naive, n_code, h_commutation=True, merging=merging
    )
    without = measured_cost(
        program, naive, n_code, h_commutation=False, merging=merging
    )
    if with_pass is None or without is None:
        decision = True
    else:
        decision = with_pass <= without

    if use_cache:
        _CACHE[key] = decision
    return decision
