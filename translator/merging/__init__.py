"""
Gate merging
============

Rewrites a program circuit so that gates which can share one encoded construction
are replaced by a macro standing for that construction. Given a permutation, the
pass knows which program qubits share a patch and which lane each occupies, which
is what makes the substitution decidable.

Two families of macro, with different admissibility rules:

*whole-patch* -- `hh`, and the transversal `cxt` / `czt` / `xcxt`
    These act on every lane of the patch(es) involved, because they are built from
    physical gates applied to all n data qubits. A macro is therefore only a legal
    substitution when it covers every *occupied* lane, so at k slots per patch it
    takes k program gates to make one. Fusing a subset is not merely unprofitable
    but inexpressible: partial transversal CX sends the full-weight stabilizer
    `S_X = X^(x)n` outside the stabilizer group, so it is not a logical operation.
    Coverage is checked twice -- when detecting (`single_qubit`, `transversal`) and
    again against the live slot map when translating
    (`handlers.utils.resolve_transversal_macro`), since a SWAP may have moved a
    qubit in between.

*per-lane* -- `xx`, `zz`
    `Xbar_i = X[x_anchor] X[lane i]`, so any subset of lanes is legal and an
    even-sized subset cancels the anchor factor outright. That cancellation is the
    saving; no coverage condition applies.

At k=2 the two rules coincide, since a group of two lanes *is* the whole patch.
That is why the original pairwise implementation needed no notion of coverage, and
why generalizing past k=2 was mostly a matter of introducing one.

Pipeline
--------
`preprocessing.preprocess_program_for_merging_with_stats` runs, per pass:

    layer_by_node          assign each instruction an earliest layer and an order
    _build_patch_lanes     permutation -> patch -> occupied lanes, in slot order
    single_qubit.find_*    propose hh / xx / zz groups within a delay budget
    selection              pick a non-conflicting subset of those proposals
    transversal.find_*     propose cxt / czt / xcxt groups, honouring slot alignment
    conflicts.filter_*     drop single-qubit picks that the transversal picks reorder
    rewrite                emit the new circuit, delaying gates the macros displace

Alignment is the transversal detector's own problem: a group's control-slot to
target-slot map is a permutation of k slots, and only the identity can be emitted
directly. Anything else is realigned by permuting the target patch, decomposed into
transpositions and emitted as routing swaps around the macro. Routing swaps are atom
movement rather than gates -- priced (0, 0) -- so alignment costs nothing and the
binding constraint is scheduling, not cost.
"""

from .analysis import PairMergeAnalyzer

#: The code size the pass's default cost table is calibrated for. It no longer
#: bounds where merging *runs* -- detection, the macro gates and the handler guards
#: are general in k -- but `PairMergeAnalyzer` still scores qubits two at a time for
#: the packer, and `conflicts.DEFAULT_MERGE_WORK_TABLE` still defaults here. Pass
#: `n_code` to `preprocess_program_for_merging*` to price a different code size.
MERGING_CODE_SIZE = 4

from .gates import CXTGate, CZTGate, XCXTGate, HHGate, XXGate, ZZGate
from .models import MacroReplacement, MergeConflictInfo, PairMergeStats, PlannedOp, SoloMergeStats
from .conflicts import compute_merge_conflicts
from .selection import select_best_group
from .preprocessing import preprocess_program_for_merging, preprocess_program_for_merging_with_stats
from .settings import DEFAULT_SETTINGS, MergeSettings

__all__ = [
    "CXTGate",
    "CZTGate",
    "XCXTGate",
    "DEFAULT_SETTINGS",
    "HHGate",
    "MERGING_CODE_SIZE",
    "XXGate",
    "ZZGate",
    "MergeConflictInfo",
    "MergeSettings",
    "MacroReplacement",
    "PairMergeAnalyzer",
    "PairMergeStats",
    "PlannedOp",
    "SoloMergeStats",
    "compute_merge_conflicts",
    "select_best_group",
    "preprocess_program_for_merging",
    "preprocess_program_for_merging_with_stats",
]
