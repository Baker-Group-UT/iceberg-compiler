"""
Resource selection: which code, how many patches, and the permutation for them.

`packing` proper answers "given [[n, n-2, 2]] and P patches, which permutation?".
This subpackage answers the question above it -- what n and P should be -- because the
packer structurally cannot: three of its mechanisms are each monotone toward the
tightest packing, so patch count is an output of its objective rather than a decision
it makes. See `select` for the three.

    from packing.resource import NoiseKnobs, select_resources

    plan = select_resources(circuit, noise=NoiseKnobs(density_weight=2.0),
                            budget=ResourceBudget(max_physical_qubits=256))
    plan.placement, plan.n_code, plan.summary()

With `density_weight=0` (the default) this optimises accepted-shot rate alone and will
choose a tight packing. Raising it buys patch density -- more detectors per logical
qubit, fewer logical qubits behind each -- at the cost of throughput. That exchange
rate is an application property, so there is no honest default but zero.

`packing.compile_circuit` runs this stage and then `plan_packing` at the code size it
chose, and returns a physical circuit rather than a placement. Call it instead of this
when the answer wanted is a circuit; call this directly when the answer wanted is the
frontier.
"""

from .noise import ErrorEstimate, NoiseKnobs, estimate
from .select import Candidate, ResourcePlan, select_resources

__all__ = [
    "NoiseKnobs",
    "ErrorEstimate",
    "estimate",
    "Candidate",
    "ResourcePlan",
    "select_resources",
]
