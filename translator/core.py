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


@dataclass
class TranslationResult:
    circuit: QuantumCircuit
    permutation: List[Tuple[int, int]]
    context: TranslationContext
    contains_faults: bool
    fault_log: List[str]
    verbose: bool
    fault_tolerant: bool

BARRIERS = False

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

    def translate_gate(self, op, next_op=None):
        # `op` and `next_op` are (inst, qargs, cargs) tuples from qc.data
        return translate_and_apply(self.context, op, next_op=next_op)

    def translate_circuit(self) -> TranslationResult:
        # Iterate over the program circuit instructions
        ops = list(self.program_circuit.data)
        skip_next = False
        for i, op in enumerate(ops):
            if skip_next:
                skip_next = False
                continue
            next_op = ops[i + 1] if i + 1 < len(ops) else None
            consumed = self.translate_gate(op, next_op)
            if consumed is not None:
                if consumed:
                    skip_next = True
                else:
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

    def translate(
        self,
        program_circuit: QuantumCircuit,
        permutation: List[Tuple[int, int]],
    ) -> TranslationResult:
        """
        Translate a program circuit and a permutation into a physical circuit.
        """
        translation = Translation(
            permutation=permutation,
            program_circuit=program_circuit,
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
