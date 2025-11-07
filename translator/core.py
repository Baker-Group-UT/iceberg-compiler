"""
translator.core.py - Translator, Translation, and TranslationResult classes.
Main functionality for this package (Qiskit version).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit

from .context import TranslationContext
from .gate_handlers import translate_and_apply
from .config import TranslatorConfig
from .merging import preprocess_program_for_merging


@dataclass
class TranslationResult:
    circuit: QuantumCircuit
    permutation: List[Tuple[int, int]]
    context: TranslationContext
    contains_faults: bool
    fault_log: List[str]
    verbose: bool
    fault_tolerant: bool

BARRIERS = True

class Translation:
    def __init__(
        self,
        permutation: List[Tuple[int, int]],
        program_circuit: QuantumCircuit,
        *,
        fault_tolerant: bool = False,
        verbose: bool = False,
    ):
        self.permutation = permutation
        self.program_circuit = program_circuit
        self.fault_tolerant = fault_tolerant
        self.verbose = verbose

        self.context = TranslationContext.from_config(
            permutation, program_circuit, fault_tolerant=fault_tolerant, verbose=verbose
        )

    def translate_gate(self, op):
        # `op` is an (inst, qargs, cargs) tuple from qc.data
        return translate_and_apply(self.context, op)

    def translate_circuit(self) -> TranslationResult:
        # Iterate over the program circuit instructions
        for op in self.program_circuit.data:
            flag = self.translate_gate(op)
            if flag is False:
                # gate application failed
                # TODO: make this pass further up the chain so we don't hault
                # permutation should be marked as infeasible in set
                raise RuntimeWarning("Gate application failed!")

        self.context.iceberg_set.decode()
        if BARRIERS:
            self.context.iceberg_set.BARRIER()
        self.context.iceberg_set.measure()

        return TranslationResult(
            circuit=self.context.iceberg_set.circuit,
            permutation=self.permutation,
            context=self.context,
            contains_faults=(len(self.context.fault_log) == 0),
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
        self.enable_hh_merging = config.enable_hh_merging

    def translate(
        self,
        program_circuit: QuantumCircuit,
        permutation: List[Tuple[int, int]],
    ) -> TranslationResult:
        """
        Translate a program circuit and a permutation into a physical circuit.
        """
        if self.enable_hh_merging:
            processed_circuit = preprocess_program_for_merging(program_circuit, permutation)
        else:
            processed_circuit = program_circuit
        translation = Translation(
            permutation=permutation,
            program_circuit=processed_circuit,
            fault_tolerant=self.fault_tolerant,
            verbose=self.verbose,
        )
        return translation.translate_circuit()

    def translate_batch(
        self,
        program_circuit: QuantumCircuit,
        permutations: List[List[Tuple[int, int]]],
    ) -> List[TranslationResult]:
        """
        Translate a program circuit and a list of permutations into physical circuits.
        """
        return [self.translate(program_circuit, perm) for perm in permutations]
