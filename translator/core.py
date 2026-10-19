"""
translator.core.py - Translator, Translation, and TranslationResult classes.
Main functionality for this package (Qiskit version).
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit

from .context import TranslationContext
from .gate_handlers import translate_and_apply
from .config import TranslatorConfig
from .merging import preprocess_program_for_merging_with_stats
from .passes import cancel_inverse_gates, cancel_inverse_gates_through_swaps, commute_hadamards
from .logical_optimization import logically_optimized
from .utils import DEFAULT_CODE_SIZE

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    circuit: QuantumCircuit
    permutation: List[Tuple[int, int]]
    context: TranslationContext
    contains_faults: bool
    fault_log: List[str]
    verbose: bool
    fault_tolerant: bool
    merge_savings_1q: float = 0.0
    merge_savings_2q: float = 0.0
    merge_savings_weighted: float = 0.0

BARRIERS = False

class Translation:
    def __init__(
        self,
        permutation: List[Tuple[int, int]],
        program_circuit: QuantumCircuit,
        *,
        fault_tolerant: bool = False,
        verbose: bool = False,
        encode_override: bool = False,
        n_code: int = DEFAULT_CODE_SIZE,
        syndrome_rounds: int = 0,
    ):
        self.permutation = permutation
        self.syndrome_rounds = syndrome_rounds
        self.program_circuit = program_circuit
        self.fault_tolerant = fault_tolerant
        self.verbose = verbose
        self.encode_override = encode_override
        self.n_code = n_code

        self.context = TranslationContext.from_config(
            permutation,
            program_circuit,
            fault_tolerant=fault_tolerant,
            verbose=verbose,
            encode_override=encode_override,
            n_code=n_code,
        )

    def translate_gate(self, op):
        # `op` is an (inst, qargs, cargs) tuple from qc.data
        return translate_and_apply(self.context, op)

    def translate_circuit(self) -> TranslationResult:
        # Iterate over the program circuit instructions
        cuts = round_cuts(self.program_circuit, self.syndrome_rounds)
        rounds = 0
        for index, op in enumerate(self.program_circuit.data):
            flag = self.translate_gate(op)
            if flag is False:
                # gate application failed
                # TODO: make this pass further up the chain so we don't hault
                # permutation should be marked as infeasible in set
                raise RuntimeWarning("Gate application failed!")
            if BARRIERS:
                self.context.iceberg_set.BARRIER()
            if index in cuts:
                # every gadget leaves its patches in the code space, so a cut between two
                # program instructions is a valid place to check every patch
                self.context.iceberg_set.syndrome_round(rounds)
                rounds += 1

        from translator import explore_switches

        if explore_switches.SOLO_GAUGE_PARITY_FIX:
            _restore_solo_gauge_frames(self.context)
        if not self.encode_override:
            self.context.iceberg_set.decode()
            if BARRIERS:
                self.context.iceberg_set.BARRIER()
            self.context.iceberg_set.measure()
        processed_gates = cancel_inverse_gates(self.context.iceberg_set.circuit)
        processed_gates = cancel_inverse_gates_through_swaps(processed_gates)
        processed_gates = cancel_inverse_gates(processed_gates)
        return TranslationResult(
            circuit=processed_gates,
            permutation=self.permutation,
            context=self.context,
            # The name says what it means: True when the fault log has something in it.
            # This read `== 0` -- fault-free -- so every `df[df.contains_faults]` filter
            # selected exactly the population it was written to exclude, and the column
            # varied by permutation within a single non-FT scan, which made it look like
            # real signal.
            contains_faults=(len(self.context.fault_log) > 0),
            fault_log=self.context.fault_log,
            verbose=self.verbose,
            fault_tolerant=self.fault_tolerant,
        )


class Translator:
    def __init__(self, *, config: Optional[TranslatorConfig] = None):
        if config is None:
            config = TranslatorConfig()
        self.fault_tolerant = config.fault_tolerant
        self.verbose = config.verbose
        self.n_code = config.n_code
        self.enable_merging = config.enable_merging
        self.enable_h_commutation = config.enable_h_commutation
        self.h_commutation_stage = (
            config.h_commutation_stage if config.enable_h_commutation else "never"
        )
        if self.h_commutation_stage not in ("before_merging", "after_merging", "never"):
            raise ValueError(
                f"unknown h_commutation_stage {config.h_commutation_stage!r}"
            )
        self.allow_transversal_swap_merges = config.allow_transversal_swap_merges
        self.encode_override = config.encode_override
        self.asap_order = config.asap_order
        self.transversal_intra_two_qubit = config.transversal_intra_two_qubit
        self.solo_hh_promotion = config.solo_hh_promotion
        self.merge_two_qubit = config.merge_two_qubit
        self.syndrome_rounds = config.syndrome_rounds

    def translate(
        self,
        program_circuit: QuantumCircuit,
        permutation: List[Tuple[int, int]],
    ) -> TranslationResult:
        """
        Translate a program circuit and a permutation into a physical circuit.

        Runs under this config's gadget options (`iceberg.gadget_options`): with
        `transversal_intra_two_qubit` on, every construction and cost table consulted
        during the call -- the merging pass's savings included -- sees the transversal
        in-patch CZ / XCX.
        """
        from iceberg.gadget_options import solo_h_targeted, transversal_intra

        with transversal_intra(self.transversal_intra_two_qubit), \
                solo_h_targeted(not self.solo_hh_promotion):
            return self._translate_in_scope(program_circuit, permutation)

    def _translate_in_scope(
        self,
        program_circuit: QuantumCircuit,
        permutation: List[Tuple[int, int]],
    ) -> TranslationResult:
        # First pass: one verified round of logical optimisation (`logical_optimization`; on
        # unless AQORN_LOGICAL_OPT=0). `packing.tables.translated_view` runs the same, so
        # the planner prices exactly the program translated here.
        processed_circuit = cancel_inverse_gates(logically_optimized(program_circuit))

        def commute(circuit):
            return cancel_inverse_gates(commute_hadamards(circuit))

        def merge(circuit):
            if not self.enable_merging:
                return circuit, 0.0, 0.0, 0.0
            return preprocess_program_for_merging_with_stats(
                circuit,
                permutation,
                allow_transversal_swap_merges=self.allow_transversal_swap_merges,
                n_code=self.n_code,
                two_qubit=self.merge_two_qubit,
            )

        # Pass order is a knob, not a fact. Commutation runs first by default, and the
        # paper notes that this hands it the largest share of the credit -- it removes
        # the Hadamards that merging would otherwise have fused into whole-patch
        # macros. `h_commutation_stage` exists so that attribution can be measured
        # rather than argued about; see experiments/figures/updated/fig_pass_order.
        stage = self.h_commutation_stage
        if stage == "before_merging" and self.enable_h_commutation:
            processed_circuit = commute(processed_circuit)

        processed_circuit, merge_savings_1q, merge_savings_2q, merge_savings_weighted = (
            merge(processed_circuit)
        )

        if stage == "after_merging" and self.enable_h_commutation:
            # The circuit now contains macro gates. `commute_hadamards` flushes its
            # frame at any instruction it does not recognise, so macros are barriers to
            # commutation rather than hazards -- it simply finds less to do.
            processed_circuit = commute(processed_circuit)
        from translator import explore_switches

        # `asap_order` (see TranslatorConfig); the branch-only switch is the same thing for
        # the exploration scripts that set it per worker.
        if self.asap_order or explore_switches.LAYER_ORDER_ALL_ARMS:
            processed_circuit = _layer_sorted(processed_circuit)
        translation = Translation(
            permutation=permutation,
            program_circuit=processed_circuit,
            fault_tolerant=self.fault_tolerant,
            verbose=self.verbose,
            encode_override=self.encode_override,
            n_code=self.n_code,
            syndrome_rounds=self.syndrome_rounds,
        )
        translated = translation.translate_circuit()
        return TranslationResult(
            circuit=translated.circuit,
            permutation=translated.permutation,
            context=translated.context,
            contains_faults=translated.contains_faults,
            fault_log=translated.fault_log,
            verbose=translated.verbose,
            fault_tolerant=translated.fault_tolerant,
            merge_savings_1q=merge_savings_1q,
            merge_savings_2q=merge_savings_2q,
            merge_savings_weighted=merge_savings_weighted,
        )

    def translate_batch(
        self,
        program_circuit: QuantumCircuit,
        permutations: List[List[Tuple[int, int]]],
    ) -> List[TranslationResult]:
        """
        Translate a program circuit and a list of permutations into physical circuits.

        A permutation the translator refuses is skipped, not fatal. `translate` raises
        `RuntimeWarning` when a handler rejects a placement, which is a routine outcome
        under the fault-tolerance rules at k > 2 -- and as a bare comprehension one
        rejection took down the whole scan and discarded every placement already
        simulated. A multi-hour sweep losing all of its work to the last permutation in
        the list is a worse failure than a short frame.

        Alignment survives the skip because every `TranslationResult` carries its own
        `permutation`, and each scan task builds its rows from these results rather than
        by zipping against the input list.
        """
        translated: List[TranslationResult] = []
        rejected: List[List[Tuple[int, int]]] = []
        for permutation in permutations:
            try:
                translated.append(self.translate(program_circuit, permutation))
            except RuntimeWarning:
                rejected.append(permutation)
        if rejected:
            logger.warning(
                "translate_batch: %d of %d placements rejected and skipped; first was %s",
                len(rejected), len(permutations), rejected[0],
            )
        return translated


def _restore_solo_gauge_frames(context: TranslationContext) -> None:
    """
    `explore_switches.SOLO_GAUGE_PARITY_FIX`: a solo patch that received an odd number of
    whole-patch HH has its spare lane(s) in the X frame; one targeted H per spare lane
    rotates them back to |0bar>, where the final Z readout checks them ([[4,1,2]]).
    """
    counts = getattr(context, "solo_hh_count", {})
    for patch, count in sorted(counts.items()):
        if count % 2 == 0:
            continue
        occupied = {slot for p, slot in context.state_positions if p == patch}
        for slot in range(context.k_log):
            if slot not in occupied:
                context.iceberg_set.H_at(patch, slot, fault_tolerant=context.fault_tolerant)
                context.fault_log.append(f"spare fix-up H on patch {patch} slot {slot}")


def round_cuts(circuit, rounds: int) -> set:
    """
    Program-instruction indices after which `TranslatorConfig.syndrome_rounds` checks go.

    The boundaries split the program's ASAP layers into `rounds + 1` equal spans; a check goes
    after instruction i when every gate up to i lies before the boundary and every later gate
    at or after it. A layer-sorted program (`asap_order`) has such a cut at every boundary; a
    program in another order may have fewer, and gets fewer rounds.
    """
    if rounds <= 0:
        return set()
    layers = []
    next_free = {}
    for inst in circuit.data:
        if inst.operation.name in ("measure", "barrier"):
            layers.append(None)
            continue
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        layer = max((next_free.get(q, 0) for q in qubits), default=0)
        for q in qubits:
            next_free[q] = layer + 1
        layers.append(layer)
    gate_layers = [layer for layer in layers if layer is not None]
    if not gate_layers:
        return set()
    depth = max(gate_layers) + 1
    boundaries = sorted({round(depth * (r + 1) / (rounds + 1)) for r in range(rounds)}
                        - {0, depth})
    prefix_max, running = [], -1
    for layer in layers:
        if layer is not None:
            running = max(running, layer)
        prefix_max.append(running)
    suffix_min, running = [0] * (len(layers) + 1), float("inf")
    suffix_min[len(layers)] = running
    for i in range(len(layers) - 1, -1, -1):
        if layers[i] is not None:
            running = min(running, layers[i])
        suffix_min[i] = running
    cuts = set()
    for boundary in boundaries:
        for i in range(len(layers)):
            if 0 <= prefix_max[i] < boundary <= suffix_min[i + 1]:
                cuts.add(i)
                break
    return cuts


def _layer_sorted(circuit):
    """
    `circuit` re-emitted in ASAP-layer order: the same DAG, so the same unitary.

    Sorted by (earliest layer, insertion order) -- the key the merging rewriter emits in --
    which is a topological order because a node's layer exceeds every predecessor's.
    """
    from qiskit.converters import circuit_to_dag

    from .merging.layer import layer_by_node

    dag = circuit_to_dag(circuit)
    layers, order = layer_by_node(dag)
    out = circuit.copy_empty_like()
    for node in sorted(dag.topological_op_nodes(), key=lambda n: (layers[n], order[n])):
        out.append(node.op, node.qargs, node.cargs)
    return out
