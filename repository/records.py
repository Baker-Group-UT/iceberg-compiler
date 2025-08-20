"""
recorder.records - Typed data record containers for structured run saving.

Each record defines its own serialization logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd
import cirq


# ------------------------
# DataFrameRecord
# ------------------------

@dataclass
class DataFrameRecord:
    df: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "columns": list(self.df.columns),
            "records": self.df.to_dict(orient="records")
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
    format: str  # "openqasm2" or "cirq_json"
    text: str

    @classmethod
    def from_circuit(cls, circuit: cirq.Circuit) -> CircuitRecord:
        try:
            text = cirq.qasm(circuit)  # type: ignore[attr-defined]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            return cls(format="openqasm2", text=text)
        except Exception:
            return cls(format="cirq_json", text=cirq.to_json(circuit))

    def to_cirq(self) -> Optional[cirq.Circuit]:
        if self.format == "cirq_json":
            try:
                return cirq.read_json(json_text=self.text)
            except Exception:
                return None
        return None


# ------------------------
# SimulationResultRecord (placeholder)
# ------------------------

@dataclass
class SimulationResultRecord:
    results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.results

    @classmethod
    def from_dict(cls, d: dict) -> SimulationResultRecord:
        return cls(results=d)
