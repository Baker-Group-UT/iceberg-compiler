"""
recorder.records - Typed data record containers for structured run saving (Qiskit version).

Each record defines its own serialization logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps, loads as qasm3_loads


# ------------------------
# DataFrameRecord
# ------------------------

@dataclass
class DataFrameRecord:
    df: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "columns": list(self.df.columns),
            "records": self.df.to_dict(orient="records"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataFrameRecord:
        df = pd.DataFrame.from_records(data["records"], columns=data.get("columns"))
        return cls(df=df)


# ------------------------
# CircuitRecord
# ------------------------

@dataclass
class CircuitRecord:
    format: str  # e.g., "qasm3" or "qiskit_serialized"
    text: str

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> CircuitRecord:
        """
        Serialize a Qiskit QuantumCircuit to QASM 3.0 (text-based).
        Falls back to Qiskit .qpy binary if QASM export fails.
        """
        try:
            text = qasm3_dumps(circuit)
            return cls(format="qasm3", text=text)
        except Exception:
            # Fallback to a generic repr-based serialization
            return cls(format="qiskit_serialized", text=circuit.qasm())

    def to_qiskit(self) -> Optional[QuantumCircuit]:
        """
        Attempt to deserialize back to a Qiskit QuantumCircuit.
        """
        if self.format == "qasm3":
            try:
                return qasm3_loads(self.text)
            except Exception:
                return None
        return None


# ------------------------
# SimulationResultRecord
# ------------------------

@dataclass
class SimulationResultRecord:
    results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.results

    @classmethod
    def from_dict(cls, d: dict) -> SimulationResultRecord:
        return cls(results=d)
