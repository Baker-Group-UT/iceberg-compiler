"""
costs.py — physical gate counts of the logical operations, as a function of n.

Single source of truth for the compiler's cost model. `packer.work_table` and
`translator.merging.conflicts` both import from here; they used to carry
independently maintained copies of the n=4 numbers, which is exactly the kind of
duplication that makes a code family hard to generalize.

Every entry is (single-qubit gates, two-qubit gates) for the *non*-fault-tolerant
construction. Fault-tolerant mode multiplies the two-qubit count by the flagged
gadget's cost — 2 ancillas, 1 two-qubit gate becoming 6, plus 2 measurements —
uniformly and independently of n, so it does not change the relative ordering the
packer cares about. See `flagged_overhead` if an absolute FT number is needed.

What is and is not n-dependent
------------------------------
Almost nothing scales with n. Logical operators are weight-2 at every n, so the
targeted constructions touch a constant number of qubits:

    constant in n   X, Z, S, SX, H, CX, CZ, XCX, RZ, RX
    linear in n     XX/ZZ (n-2 lane Paulis), HH (n Hadamards), the transversal
                    CXT/CZT/XCXT (n gates), encode (n-1 CNOTs) and decode (2n of each)

That is the shape of the packing trade-off: targeted cross-patch gates cost the
same at any code size while their transversal counterparts get more expensive, so
the larger the patch, the less attractive transversal merging becomes relative to
targeted placement.

Placement caveat
----------------
The table is keyed by logical gate name only, but two of the entries differ
between intra- and cross-patch placement:

    CZ    intra-patch (1, 3)   cross-patch (0, 4)
    XCX   intra-patch (5, 3)   cross-patch (4, 4)

The entries below are the intra-patch numbers, matching what the tables they
replace have always used.

The patch-wise Pauli macros XX/ZZ are *whole-patch* entries: X^{\\otimes k} /
Z^{\\otimes k} over all k = n-2 lanes (k is even, so the anchor factors cancel
and one gate per lane remains). A partial fusion of m < k coincident Paulis is
not in the table; price it with `pauli_macro_cost(m)` = m + (m mod 2)
single-qubit gates, depth 1 at any m. A fused group split across patches
decomposes into one group per patch, each priced the same way -- which is why a
split pair costs (4, 0) against the fused pair's (2, 0).
"""

from __future__ import annotations

from typing import Dict, Tuple

from .layout import CodeLayout

__all__ = [
    "work_table",
    "depth_table",
    "gate_cost",
    "gate_depth",
    "pauli_macro_cost",
    "flagged_overhead",
    "WorkTable",
    "CARRIAGES",
    "CROSS_PATCH_COST",
    "carriage_cost",
    "carriage_weight",
]

WorkTable = Dict[str, Tuple[int, int]]

# Relative cost of a two-qubit gate to a single-qubit gate, used to collapse a
# (n1, n2) pair into a scalar. Calibrated at n=4; independent of n.
DEFAULT_WORK_WEIGHTS: Tuple[float, float] = (1.0, 13.0)

# A physical SWAP emits no pulses, so as a *gate count* (0, 0) is correct and stays --
# the tables are pinned against emitted circuits and a swap contributes nothing to
# either column. What a swap costs is **movement**: the simulator prices it as an atom
# exchange, both atoms carried, `move_error` each. That price lives in the carriage
# column below, not here. (A *program* SWAP is different again -- the translator
# implements it as a slot relabel and emits nothing at all; the packing tables key
# program swaps to this entry, and free is exactly right for them.)
SWAP_WEIGHT: Tuple[int, int] = (0, 0)

#: Atoms physically carried per logical operation -- the movement column the (n1, n2)
#: gate scheme lacks. A carriage is one pick-up/put-down, the unit the simulator
#: charges `move_error` for. Only two operations carry atoms *by construction* (the
#: cross-patch transport of the targeted/transversal gates is priced by the movement
#: models in `packing`, not per-op here):
#:
#:     SWAP   a physical lane/anchor exchange: both atoms travel, 2 carriages
#:     HH     transversal H plus the anchor swap, so the same 2
#:
#: Everything else is pulses only. Keyed like `work_table`; missing key means zero.
CARRIAGES: Dict[str, int] = {"SWAP": 2, "HH": 2}

#: Weighted-work units per carriage, on the same scale as `DEFAULT_WORK_WEIGHTS` --
#: one unit is one single-qubit gate's worth of error. Derived, not tuned:
#: `move_error / p1` at the neutral-atom defaults (1e-3 / 2e-4), so one carried atom
#: costs five single-qubit gates and one physical swap (2 carriages = 10) costs about
#: 2/3 of a two-qubit gate (13). Import the constants rather than restating them so
#: this cannot drift from the machine the simulator charges for.
def carriage_weight() -> float:
    """Weighted-work units per carried atom, from the simulator's own rates."""
    from simulator.noise_models import BASE_MOVE_ERROR, NEUTRAL_ATOM

    return BASE_MOVE_ERROR / NEUTRAL_ATOM["one_qubit_error"]


def carriage_cost(gate: str, weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS) -> float:
    """Movement cost of one logical `gate` in weighted-work units (0 for most gates)."""
    w1, _ = weights
    return CARRIAGES.get(gate, 0) * carriage_weight() * w1


#: Cross-patch (1q, 2q) overrides for the two-qubit gates whose construction changes
#: when their operands sit in different patches. Everything else costs the same either
#: way. Lived in `packing.tables` historically; moved here so `translator.merging` can
#: price the gates a transversal macro replaces without importing packing.
CROSS_PATCH_COST: Dict[str, Tuple[int, int]] = {
    "CZ": (0, 4),
    "XCX": (4, 4),
}


def work_table(n: int) -> WorkTable:
    """
    Physical (1q, 2q) gate counts per logical operation for a [[n, n-2, 2]] code.

    Verified against the emitted circuits in tests/test_iceberg_costs.py, which
    also asserts that `work_table(4)` reproduces the numbers the compiler was
    calibrated against.
    """
    CodeLayout(n)  # validate n

    return {
        # --- targeted Paulis: anchor plus one lane. Constant in n.
        "X": (2, 0),
        "Y": (2, 0),
        "Z": (2, 0),
        # --- whole-patch Pauli macros over all k = n-2 lanes; k is even, so the
        #     anchor factors cancel and one gate per lane remains. Partial
        #     fusions (m < k lanes) are priced by `pauli_macro_cost`, not here.
        "XX": (n - 2, 0),
        "ZZ": (n - 2, 0),
        # --- single-lane Cliffords
        "S": (2, 1),
        "SX": (4, 1),
        "H": (7, 4),
        # --- whole-patch Hadamard: transversal H plus the free anchor swap
        "HH": (n, 0),
        # --- rotations
        "RZ": (1, 2),
        "RX": (1, 2),
        "RZZ": (1, 2),
        # --- two-lane Cliffords (intra-patch; see module docstring)
        "CX": (0, 4),
        "CZ": (1, 3),
        "XCX": (5, 3),
        # --- transversal variants: one gate per data qubit
        "CXT": (0, n),
        "CZT": (0, n),
        "XCXT": (2 * n, n),
        # --- free / bookkeeping
        "SWAP": SWAP_WEIGHT,
        "MEASURE": (0, 0),
        # --- state preparation and syndrome extraction
        "ENCODE": (1, n - 1),
        "DECODE": (2 * n, 2 * n),
        **_transversal_intra_work(n),
    }


def _transversal_intra_work(n: int) -> Dict[str, Tuple[int, int]]:
    """Branch v2-tvd-explore: in-patch CZ / XCX as S (SX) on every data qubit at k = 2."""
    from translator import explore_switches

    if not (explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT and n == 4):
        return {}
    return {"CZ": (n + 2, 0), "XCX": (n + 2, 0)}


def _transversal_intra_depth(n: int) -> Dict[str, int]:
    from translator import explore_switches

    if not (explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT and n == 4):
        return {}
    return {"CZ": 2, "XCX": 2}


def depth_table(n: int) -> Dict[str, int]:
    """
    Circuit depth of each logical operation on a [[n, n-2, 2]] code.

    Measured from the emitted circuits, and pinned against them in
    tests/test_iceberg_costs.py exactly as `work_table` is.

    Every entry is **constant in n**, which is worth stating plainly because it is
    not obvious: HH is a transversal Hadamard layer plus the anchor swap, so two
    layers whatever n; the transversal two-qubit macros are n parallel gates, so one
    layer. Depth does not grow with the code size at all.

    Why this table matters as much as `work_table`
    ---------------------------------------------
    Gate counts and depth disagree about where a two-qubit gate wants to live, and
    for CNOT they disagree in *opposite directions*::

        CX    gates: intra (0,4) == cross (0,4)     -> indifferent
              depth: intra 4     >  cross 3         -> prefers to be split

        CZ    gates: intra (1,3) <  cross (0,4)     -> prefers to be together
              depth: intra 3     == cross 3         -> indifferent

    So a placement chosen on gate counts alone is blind to CNOT, and one chosen on
    depth alone pushes CNOT partners apart. A packer that minimises cut weight -- the
    natural reading of "put interacting qubits together" -- is therefore optimising
    against depth for every CNOT in the circuit.

    The transversal entries are the reason the trade-off is worth having. `CXT` is
    depth 1 and costs `(0, n)` while doing the work of k separate cross-patch CNOTs,
    which would be depth 3 each and `(0, 4k)` in total. At n=6 that is 78 weighted
    units and depth 1 against 208 and depth 12. Transversal placement wins on both
    axes at once -- but only when *all* k lanes of both patches pair up, which makes
    it an alignment problem rather than a clustering one.
    """
    CodeLayout(n)  # validate n

    return {
        # --- single-lane, constant in n
        "X": 1,
        "Y": 1,
        "Z": 1,
        "XX": 1,
        "ZZ": 1,
        "S": 3,
        "SX": 4,
        "H": 8,
        "RZ": 3,
        "RX": 3,
        "RZZ": 3,
        # --- whole-patch Hadamard: one transversal layer plus the anchor swap
        "HH": 2,
        # --- two-lane, intra-patch
        "CX": 4,
        "CZ": 3,
        "XCX": 6,
        # --- two-lane, cross-patch (see `CROSS_PATCH_DEPTH` users)
        "CX_CROSS": 3,
        "CZ_CROSS": 3,
        "XCX_CROSS": 5,
        # --- transversal: one layer of n parallel gates, k lanes at once
        "CXT": 1,
        "CZT": 1,
        "XCXT": 3,
        # --- free / bookkeeping
        "SWAP": 1,
        "MEASURE": 0,
        **_transversal_intra_depth(n),
    }


def gate_depth(gate: str, n: int, *, cross_patch: bool = False) -> int:
    """
    Depth of one logical `gate`, choosing the intra- or cross-patch construction.

    `cross_patch` silently falls back to the intra-patch entry for operations that
    have only one form, so callers can pass it uniformly.
    """
    table = depth_table(n)
    if cross_patch:
        crossed = table.get(f"{gate}_CROSS")
        if crossed is not None:
            return crossed
    return table[gate]


def pauli_macro_cost(m: int) -> Tuple[int, int]:
    """
    (1q, 2q) cost of fusing `m` coincident logical X's (or Z's) in one patch.

    `Xbar_i = X[x_anchor] X[lane_i]`, so a fused group emits one gate per lane
    plus the anchor when m is odd: m + (m % 2) single-qubit gates, depth 1 at
    any m. `work_table`'s XX/ZZ rows are this cost at m = k; the merging pass
    fuses any m >= 2 and must price the group it actually formed. m = 1
    reproduces the targeted X/Z entry, as it should.
    """
    if m < 1:
        raise ValueError("group size must be positive")
    return (m + (m % 2), 0)


def gate_cost(
    gate: str,
    n: int,
    weights: Tuple[float, float] = DEFAULT_WORK_WEIGHTS,
) -> float:
    """Scalar cost of one logical `gate` on a [[n, n-2, 2]] code."""
    one_qubit, two_qubit = work_table(n)[gate]
    w1, w2 = weights
    return w1 * one_qubit + w2 * two_qubit


def flagged_overhead() -> Tuple[int, int, int]:
    """
    Cost of wrapping one two-qubit gate in the flagged FT gadget.

    Returns (extra 1q gates, extra 2q gates, ancillas) — independent of n.
    """
    return (2, 5, 2)
