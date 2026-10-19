"""
explore_switches.py -- opt-in translation variants for the TVD explorations (branch
v2-tvd-explore only). Every default reproduces the shipped translator exactly.

Module-level rather than plumbed through `TranslatorConfig` because these are
experiments: `experiments/tvd_explore/` sets them per worker process. Anything that earns
its place gets a proper config field before it leaves this branch.
"""

#: Single-lane logical H on a shared patch as S . SX . S (equal to H up to a global
#: phase) instead of the 4-CZ lane gadget. Three in-block two-qubit gates instead of four:
#: fewer silent faults (0.6 p2 against 0.82) and fewer detected ones.
H_AS_S_SX_S = False

#: In-patch logical CZ / XCX at k = 2 as transversal single-qubit gates. On [[4,2,2]], S on
#: every data qubit maps Xbar_1 -> -Xbar_1 Zbar_2 and Xbar_2 -> -Xbar_2 Zbar_1 and fixes
#: Zbar, i.e. it is CZbar . Zbar_1 Zbar_2; conjugating by the whole-patch HH, SX on every
#: data qubit is XCXbar . Xbar_1 Xbar_2. The Pauli corrections are physical Z (X) on the two
#: lanes. No two-qubit gate, so no silent fault at first order and no movement, against the
#: 3-gate gadgets' 0.6 p2 silent. k = 2 only: at k > 2 the transversal gate entangles every
#: pair of lanes at once.
TRANSVERSAL_INTRA_TWO_QUBIT = False

#: Whole-patch HH when a strict majority of a patch's occupied lanes want H, followed by a
#: targeted H on each remaining lane to undo it (H is self-inverse, and HH is H on every
#: lane, so HH . H_u = H on the wanted lanes exactly). HH is (n, 0) plus the anchor swap's
#: two carriages, n + 10 weighted units, and leaks no silent error; a targeted H is (7, 4),
#: 59 units, and leaks 0.82 p2. So w wanted lanes cost w targeted H's against
#: HH + (k - w), cheaper on both counts exactly when w > k/2 (for any n < 49). Cannot fire
#: at k = 2, where a strict majority is both lanes and plain HH already covers it.
HH_MAJORITY_UNDO = False

#: The same idea for the transversal CXT / CZT / XCXT between two patches: fire the macro
#: when most lane pairs want the gate and undo it on the remaining pairs with targeted
#: cross-patch gates (all three are self-inverse). Taken only when the weighted work pays:
#: (2w - k) cross-patch singles saved against one macro of n two-qubit gates, so it needs
#: more than a bare majority at k >= 6. Cross-patch gadgets are silent-free either way, so
#: this is a gate-count (LSR) change, not a silent-error one.
TRANSVERSAL_MAJORITY_UNDO = False


#: The merging rewriter emits the whole circuit in ASAP-layer order whenever any macro
#: fires (`rewrite.build_rewritten_circuit` sorts every node by (layer, order)). Any
#: topological order computes the same thing, but the physical peephole pass only cancels
#: gates that end up adjacent, and gadgets on lanes sharing an anchor cancel at their
#: boundary only when nothing else touches the anchor in between -- which program order
#: (lane by lane) preserves and layer order (all lanes interleaved) breaks. On: the
#: rewritten circuit is re-linearized as close to program order as its dependencies allow.
MERGE_KEEP_PROGRAM_ORDER = False

#: Every arm emitted in ASAP-layer order, not only the ones merging rewrote. Logical gates
#: sharing a patch serialize on its anchors, so emission order sets the physical critical
#: path; the merging rewriter's re-sort is therefore a scheduling pass that only the
#: merging arms receive. On: the program reaching translation is layer-sorted in every
#: arm, baseline included, so a stage is credited with what it does rather than with the
#: order it happens to emit in.
#: Same as `TranslatorConfig.asap_order`, for scripts that set switches per worker.
LAYER_ORDER_ALL_ARMS = False

#: A solo patch's H (a patch holding one program qubit) as the targeted single-lane gadget
#: instead of the whole-patch HH it is promoted to by default. HH applies H to the spare
#: lane too, rotating it out of |0bar>, so after an odd number of them the spare's Zbar is
#: random at readout and cannot serve as a [[4,1,2]] gauge check. On: the spare is never
#: touched and stays a valid check, at the price of a (7, 4) gadget per H (against HH's
#: (n, 0) plus the anchor swap) that leaks silent error. For `experiments/solo_gauge/`.
SOLO_H_TARGETED = False

#: Keep the HH promotion, but before readout apply one targeted H to the spare lane(s) of
#: every solo patch that received an odd number of HH, so the spare ends back in its Zbar
#: frame and the [[4,1,2]] gauge check is measurable. One (7, 4) gadget per such patch,
#: rather than one per H. For `experiments/solo_gauge/`.
SOLO_GAUGE_PARITY_FIX = False

def configure(**switches) -> None:
    """Set switches by name; unknown names are an error, not a silent no-op."""
    for name, value in switches.items():
        if name not in globals() or name.startswith("_") or name == "configure":
            raise KeyError(f"unknown translation switch {name!r}")
        globals()[name] = value
