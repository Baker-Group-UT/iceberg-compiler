"""
result_types.py - Postprocessed result container with Iceberg postselection.
"""
# pylint: disable=[C0116]

from dataclasses import dataclass
from typing import Dict, Any

from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit.qpy import dump as qpy_dump
import io

from simulator.result_types import SimulationResult


@dataclass
class PostselectedSimulationResult:
    """
    Final enriched result from simulation + Iceberg postselection.
    """
    permutation: Any                      # typically List[Tuple[int, int]]
    circuit: QuantumCircuit               # translated physical circuit
    contains_faults: bool
    raw_result: SimulationResult          # raw simulator output

    postselected_noisy_dist: Dict[str, float]
    postselected_ideal_dist: Dict[str, float]
    was_postselected: bool = True

    logical_fidelity: float = 0.0
    logical_selection_rate: float = 0.0
    logical_total_variation_distance: float = 0.0
    # % shots kept after physical postselection * % shots kept after logical postselection
    selection_rate_overall: float = 0.0
    merge_savings_1q: float = 0.0
    merge_savings_2q: float = 0.0
    merge_savings_weighted: float = 0.0

    def _circuit_serialize(self) -> Dict[str, Any]:
        """
        Prefer QASM 3 text for portability; fall back to QPY (binary, base64 via bytes)
        if QASM export fails (e.g., unsupported ops).
        """
        try:
            return {"format": "qasm3", "text": qasm3_dumps(self.circuit)}
        except Exception:
            buf = io.BytesIO()
            qpy_dump(self.circuit, buf)
            return {"format": "qpy", "bytes": buf.getvalue()}

    def to_dict(self) -> dict:
        return {
            "permutation": self.permutation,
            "circuit": self._circuit_serialize(),
            "contains_faults": self.contains_faults,
            "raw_result": self.raw_result.to_dict() if hasattr(self.raw_result, "to_dict") else self.raw_result,
            "postselected_noisy_dist": self.postselected_noisy_dist,
            "postselected_ideal_dist": self.postselected_ideal_dist,
            "logical_fidelity": self.logical_fidelity,
            "logical_selection_rate": self.logical_selection_rate,
            "logical_total_variation_distance": self.logical_total_variation_distance,
            "merge_savings_1q": self.merge_savings_1q,
            "merge_savings_2q": self.merge_savings_2q,
            "merge_savings_weighted": self.merge_savings_weighted,
        }
