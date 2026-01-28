"""Merge analysis helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .single_qubit import find_hh_merges_for_patch, find_xx_merges_for_patch, find_zz_merges_for_patch
from .layer import layer_by_node
from .models import PairMergeStats
from .settings import DEFAULT_SETTINGS, MergeSettings
from .transversal import _scan_patch_pair_for_transversal


class PairMergeAnalyzer:
    """
    Expose per-qubit-pair merge counts without requiring a concrete patch assignment.
    """

    def __init__(
        self,
        program_circuit: QuantumCircuit,
        *,
        settings: Optional[MergeSettings] = None,
        allow_transversal_swap_merges: bool = False,
    ) -> None:
        self._circuit = program_circuit
        self._settings = settings or DEFAULT_SETTINGS
        self._allow_swaps = allow_transversal_swap_merges
        self._dag = circuit_to_dag(program_circuit)
        self._layer_index, self._order_index = layer_by_node(self._dag)
        self._qubits = list(program_circuit.qubits)
        self._qubit_indices = {
            qubit: program_circuit.find_bit(qubit).index for qubit in self._qubits
        }
        self._cache: Dict[Tuple[int, int], PairMergeStats] = {}

    def analyze_pair(self, qubit_a: int, qubit_b: int) -> PairMergeStats:
        if qubit_a == qubit_b:
            raise ValueError("Cannot analyze merges for identical qubit indices.")
        if qubit_a < 0 or qubit_a >= len(self._qubits):
            raise IndexError(f"Qubit index {qubit_a} is out of range.")
        if qubit_b < 0 or qubit_b >= len(self._qubits):
            raise IndexError(f"Qubit index {qubit_b} is out of range.")

        key = tuple(sorted((qubit_a, qubit_b)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        qa = self._qubits[key[0]]
        qb = self._qubits[key[1]]
        stats = self._compute_pair_stats(qa, qb)
        self._cache[key] = stats
        return stats

    def _compute_pair_stats(self, qa, qb) -> PairMergeStats:
        hh_records = find_hh_merges_for_patch(
            self._layer_index,
            self._order_index,
            (qa, qb),
            self._settings,
            set(),
        )
        xx_records = find_xx_merges_for_patch(
            self._layer_index,
            self._order_index,
            (qa, qb),
            self._settings,
            set(),
        )
        zz_records = find_zz_merges_for_patch(
            self._layer_index,
            self._order_index,
            (qa, qb),
            self._settings,
            set(),
        )

        transversal_records = _scan_patch_pair_for_transversal(
            self._layer_index,
            self._order_index,
            patch_a=0,
            patch_b=1,
            qubits_a=(qa,),
            qubits_b=(qb,),
            qubit_indices=self._qubit_indices,
            consumed=set(),
            allow_transversal_swap_merges=self._allow_swaps,
        )

        cx_merges = sum(
            1
            for record in transversal_records
            if getattr(record.macro_gate, "name", "").lower() == "cxt"
        )
        cz_merges = sum(
            1
            for record in transversal_records
            if getattr(record.macro_gate, "name", "").lower() == "czt"
        )

        return PairMergeStats(
            hh_merges=len(hh_records),
            xx_merges=len(xx_records),
            zz_merges=len(zz_records),
            cx_merges=cx_merges,
            cz_merges=cz_merges,
        )
