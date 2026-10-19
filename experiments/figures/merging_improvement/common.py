"""
common.py -- the two things every script in this directory needs.

`pack_program` is the packing call the four benchmark scripts share. It used to be
`packer.merge_packer.MergeMaximizingPacker`, which is archived.

`stack_heights` is re-exported from `experiments.shared.figures`, where it lives now --
it is the house rule for drawing a stacked ablation rather than something this folder
owns, and every ablation figure has to clip the same way or the bars stop comparing.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import qiskit

from experiments.shared.figures import stack_heights
from packing.plan import plan_packing

Perm = List[Tuple[int, int]]

#: The code these figures evaluate: [[4, 2, 2]], two logical qubits per patch. This is
#: what `MergeMaximizingPacker` was fixed at -- it assigned qubits in pairs and could not
#: express anything else -- so pinning it here keeps the benchmark figures comparable
#: with the ones already in the paper. `merging_improvement_code_size.py` is the script
#: that varies it.
DEFAULT_N_CODE = 4


def pack_program(
    program_circuit: qiskit.QuantumCircuit,
    n_code: int = DEFAULT_N_CODE,
    *,
    verify: bool = True,
    h_commutation: Union[bool, str] = "auto",
    noise=None,
) -> Tuple[Perm, bool]:
    """
    Plan a placement with the `packing` package. Returns (placement, h_commutation).

    **Pass `noise` from any caller that simulates.** Bug B6: `plan_packing`'s noise costs
    default to zero, so without a profile the planner optimises weighted gates and depth
    while the simulator -- configured with `with_distance_noise` and `with_idle_noise` --
    charges movement and idling. The planner then answers a different question from the one
    being measured, and the packing arm is scored on a placement chosen for a machine that
    does not exist. `merging_improvement_benchmark_stylized_simulated` had exactly that
    mismatch; the gate-count-only figures legitimately pass nothing, because for them
    movement really is free.

    Two things about the call are easy to get wrong and both were wrong the first time:

    **Hand it the raw program.** `plan_packing` runs `translated_view`, which applies
    `cancel_inverse_gates` and `commute_hadamards` itself. Passing a circuit that has
    already been through those passes runs them twice and the model then prices a
    circuit the translator will never emit. The old `MergeMaximizingPacker` wanted the
    preprocessed circuit, which is why every call site here changed shape rather than
    just its import.

    **Honour the commutation decision, and translate under it.** `plan_packing` measures
    per (circuit, code size) whether H commutation pays and skips it where it does not --
    27 of 64 QASMBench proxies at n=4 choose off. The plan is priced under that answer,
    so a caller that translates the placement with the pass in the other state gets a
    circuit the plan never costed: measured 30% off, and worse than the naive arm
    `verify` had just guaranteed it beat. That is why the decision comes back with the
    placement instead of being assumed, and why every caller here feeds it into the
    translator config for *both* the packed row and the row it is differenced against.

    `verify` is left on. It translates the plan and the naive packing and keeps the
    cheaper, which is what makes the "+ Packing" segment safe to read as a gain -- the
    model over-counts merging by a couple of percent and on a few circuits that is
    enough to prefer a placement genuinely worse than sequential.
    """
    plan = plan_packing(
        program_circuit,
        n_code,
        h_commutation=h_commutation,
        verify=verify,
        noise=noise,
    )
    return plan.placement, plan.h_commutation


__all__ = ["DEFAULT_N_CODE", "pack_program", "stack_heights"]
