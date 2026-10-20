"""
translator.config.py - Configuration object for Translator behavior.
"""

from dataclasses import dataclass
from typing import Self


@dataclass
class TranslatorConfig:
    fault_tolerant: bool = False
    fault_tolerance_chance: float = -1.0
    verbose: bool = False
    n_code: int = 4  # Default [[4,2,2]] code size
    enable_merging: bool = True
    enable_h_commutation: bool = True
    #: When Hadamard commutation runs relative to merging.
    #:
    #: "before_merging" is the paper's order and the default. "after_merging" lets
    #: merging see the Hadamards first, which changes how credit is attributed between
    #: the two techniques -- the draft notes the ordering effect but could not measure
    #: it. "never" is equivalent to enable_h_commutation=False and is accepted so the
    #: three conditions can be selected by one field.
    h_commutation_stage: str = "before_merging"
    allow_transversal_swap_merges: bool = True
    encode_override: bool = False
    #: Emit every program in ASAP-layer order before translation. Opt-in; False keeps the
    #: shipped behaviour exactly.
    #:
    #: Logical gates on the lanes of one patch serialize on its anchors, so the order the
    #: translator meets them in sets the physical critical path -- the same gates in a
    #: different valid order give a different depth. Shipped, a program is translated in
    #: program order *unless* the merging pass rewrote it, because the rewriter re-emits
    #: the whole circuit in layer order whenever any macro fires. The merging stages
    #: therefore get a schedule the baseline and +H stages never do, and the depth they are
    #: credited with includes it (Fig 6 depth reduction: 28.9% shipped, 15.5% with every
    #: stage layer-sorted; gate counts move by at most 0.1 points, and a frame-simulation
    #: estimate moves LSR by about +2.6 points and TVD by +0.2). True layer-sorts every
    #: program, so stages differ only by what their passes do.
    asap_order: bool = False
    #: In-patch logical CZ / XCX at k = 2 as a transversal S (SX) layer plus two Pauli
    #: corrections instead of the three-two-qubit-gate gadget (`iceberg.gadget_options`).
    #: Part of the standard configuration (`packing.standard`, "w10T"); False here keeps
    #: this low-level default -- and every script written against it -- unchanged.
    transversal_intra_two_qubit: bool = False
    #: A Hadamard on a qubit alone in its patch as a whole-patch HH (H on every lane; the empty
    #: lanes are free to disturb). That is single-qubit merging, but the Hadamard handler
    #: applies it whether or not `enable_merging` is set, so on sparse layouts a bare baseline
    #: already has it: at one qubit per patch it makes the baseline 18% cheaper in 1q gates and
    #: 12% in 2q gates. False keeps such an H as the targeted single-lane gadget, for a baseline
    #: with no merging at all (`iceberg.gadget_options.solo_h_targeted`). True is the shipped
    #: behaviour.
    solo_hh_promotion: bool = True
    #: The merging pass's two-qubit macros: transversal CXT / CZT / XCXT across two patches.
    #: False keeps only its single-qubit macros (HH, XX, ZZ), to tell single- and two-qubit
    #: merging apart. True is the shipped behaviour.
    merge_two_qubit: bool = True
    #: Mid-circuit syndrome rounds: this many checks of every patch's two stabilizers
    #: (`IcebergModule.syndrome_round`, fault-tolerant at n = 4), at evenly spaced boundaries
    #: between the program's ASAP layers, on top of the end-of-circuit check. Registers
    #: `m_parity_round{r}_{patch}`; the post-selection rule reads them like any parity bit.
    #: 0 is the paper's scheme -- checks only at the end -- and keeps translations unchanged.
    syndrome_rounds: int = 0

    def to_dict(self) -> dict:
        """
        The settings as a dict. With the logical optimisation pass on (process-wide, see
        `translator.logical_optimization`) a `logical_optimization` entry names its version;
        off, the dict is exactly what it was before the pass existed.
        """
        from .logical_optimization import enabled, state

        settings = {
            "fault_tolerant": self.fault_tolerant,
            "verbose": self.verbose,
            "n_code": self.n_code,
            "enable_merging": self.enable_merging,
            "enable_h_commutation": self.enable_h_commutation,
            "h_commutation_stage": self.h_commutation_stage,
            "allow_transversal_swap_merges": self.allow_transversal_swap_merges,
            "encode_override": self.encode_override,
            "asap_order": self.asap_order,
            "transversal_intra_two_qubit": self.transversal_intra_two_qubit,
            "solo_hh_promotion": self.solo_hh_promotion,
            "merge_two_qubit": self.merge_two_qubit,
        }
        if self.syndrome_rounds:
            settings["syndrome_rounds"] = self.syndrome_rounds
        if enabled():
            settings["logical_optimization"] = state()
        return settings

    def disable_merging(self) -> Self:
        self.enable_merging = False
        return self
