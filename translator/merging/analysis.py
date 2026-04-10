"""Merge analysis helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .single_qubit import (
    find_hh_merges_for_patch,
    find_xx_merges_for_patch,
    find_zz_merges_for_patch,
)
from .layer import layer_by_node
from .models import MacroReplacement, PairMergeStats, SoloMergeStats
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
        self._solo_cache: Dict[int, SoloMergeStats] = {}

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

    def analyze_solo(self, qubit_index: int) -> SoloMergeStats:
        """Return per-qubit savings available when the qubit is alone in its patch."""
        cached = self._solo_cache.get(qubit_index)
        if cached is not None:
            return cached
        if qubit_index < 0 or qubit_index >= len(self._qubits):
            raise IndexError(f"Qubit index {qubit_index} is out of range.")
        qa = self._qubits[qubit_index]

        h_count = sum(
            1
            for node in self._dag.op_nodes()
            if len(node.qargs) == 1
            and node.qargs[0] == qa
            and getattr(node.op, "name", "").lower() == "h"
        )
        twoq_count = sum(
            1
            for node in self._dag.op_nodes()
            if len(node.qargs) == 2
            and qa in node.qargs
            and getattr(node.op, "name", "").lower() in ("cx", "cnot", "cz", "xcx")
        )

        # find_xx_merges_for_patch / find_zz_merges_for_patch require 2 qubits;
        # sequential X-X / Z-Z pairs on a solo qubit are eliminated by
        # cancel_inverse_gates() before scoring, so we leave these as 0.

        stats = SoloMergeStats(
            h_count=h_count,
            xx_merges=0,
            zz_merges=0,
            twoq_count=twoq_count,
        )
        self._solo_cache[qubit_index] = stats
        return stats

    def _compute_pair_stats(self, qa, qb) -> PairMergeStats:
        def _select_non_overlapping(records: list[MacroReplacement]) -> list[MacroReplacement]:
            used = set()
            ordered = sorted(
                records,
                key=lambda r: (
                    self._layer_index[r.second],
                    self._layer_index[r.first],
                    self._order_index.get(r.second, 0),
                    self._order_index.get(r.first, 0),
                ),
            )
            chosen: list[MacroReplacement] = []
            for record in ordered:
                if record.first in used or record.second in used:
                    continue
                chosen.append(record)
                used.add(record.first)
                used.add(record.second)
            return chosen

        hh_records = _select_non_overlapping(
            find_hh_merges_for_patch(
                self._layer_index,
                self._order_index,
                (qa, qb),
                self._settings,
                set(),
            )
        )
        xx_records = _select_non_overlapping(
            find_xx_merges_for_patch(
                self._layer_index,
                self._order_index,
                (qa, qb),
                self._settings,
                set(),
            )
        )
        zz_records = _select_non_overlapping(
            find_zz_merges_for_patch(
                self._layer_index,
                self._order_index,
                (qa, qb),
                self._settings,
                set(),
            )
        )

        transversal_records = _select_non_overlapping(
            _scan_patch_pair_for_transversal(
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
        xcx_merges = sum(
            1
            for record in transversal_records
            if getattr(record.macro_gate, "name", "").lower() == "xcxt"
        )
        hh_layer_distance = sum(
            abs(self._layer_index[record.first] - self._layer_index[record.second])
            for record in hh_records
        )
        cx_layer_distance = sum(
            abs(self._layer_index[record.first] - self._layer_index[record.second])
            for record in transversal_records
            if getattr(record.macro_gate, "name", "").lower() == "cxt"
        )
        cz_layer_distance = sum(
            abs(self._layer_index[record.first] - self._layer_index[record.second])
            for record in transversal_records
            if getattr(record.macro_gate, "name", "").lower() == "czt"
        )
        xcx_layer_distance = sum(
            abs(self._layer_index[record.first] - self._layer_index[record.second])
            for record in transversal_records
            if getattr(record.macro_gate, "name", "").lower() == "xcxt"
        )
        cx_between = 0
        cz_between = 0
        xcx_between = 0
        for node in self._dag.op_nodes():
            if len(node.qargs) != 2:
                continue
            name = getattr(node.op, "name", "").lower()
            if name not in ("cx", "cnot", "cz", "xcx"):
                continue
            if (node.qargs[0] == qa and node.qargs[1] == qb) or (
                node.qargs[0] == qb and node.qargs[1] == qa
            ):
                if name in ("cx", "cnot"):
                    cx_between += 1
                elif name == "cz":
                    cz_between += 1
                elif name == "xcx":
                    xcx_between += 1

        return PairMergeStats(
            hh_merges=len(hh_records),
            xx_merges=len(xx_records),
            zz_merges=len(zz_records),
            cx_merges=cx_merges,
            cz_merges=cz_merges,
            xcx_merges=xcx_merges,
            twoq_between=cx_between + cz_between + xcx_between,
            cx_between=cx_between,
            cz_between=cz_between,
            xcx_between=xcx_between,
            hh_layer_distance=hh_layer_distance,
            cx_layer_distance=cx_layer_distance,
            cz_layer_distance=cz_layer_distance,
            xcx_layer_distance=xcx_layer_distance,
        )
