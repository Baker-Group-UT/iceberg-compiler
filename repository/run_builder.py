"""
recorder.run_builder.py - contains the dataclass for managing run data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import pandas as pd
import cirq
import datetime as _dt
import uuid

from .manifest import RunManifest
from .records import (
    CircuitRecord,
    DataFrameRecord,
    SimulationResultRecord,
)


# ------------------------
# RunRecord (hydrated)
# ------------------------

@dataclass
class RunRecord:
    """
    Stores a run.
    args:
    - manifest
    - circuit
    - dataframe
    - sim_results
    """

    manifest: RunManifest
    circuit: CircuitRecord
    dataframe: Optional[DataFrameRecord] = None
    sim_results: Optional[SimulationResultRecord] = None

    def to_dict(self) -> dict:
        return {
            "meta": self.manifest.to_dict(),
            "circuit": asdict(self.circuit),
            "dataframe": self.dataframe.to_dict() if self.dataframe else None,
            "sim_results": self.sim_results.to_dict() if self.sim_results else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        return cls(
            manifest=RunManifest.from_dict(data["meta"]),
            circuit=CircuitRecord(**data["circuit"]),
            dataframe=DataFrameRecord.from_dict(data["dataframe"]) if data.get("dataframe") else None,
            sim_results=SimulationResultRecord.from_dict(data["sim_results"]) if data.get("sim_results") else None,
        )


# ------------------------
# RunBuilder (pipeline-side accumulation)
# ------------------------

@dataclass
class RunBuilder:
    circuit: cirq.Circuit
    circuit_name: str
    repetitions: int
    options: Dict[str, Any] = field(default_factory=dict)
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    dataframe: Optional[pd.DataFrame] = None
    sim_results: Optional[Dict[str, Any]] = None

    def attach_dataframe(self, df: pd.DataFrame) -> None:
        self.dataframe = df

    def attach_results(self, results: Dict[str, Any]) -> None:
        self.sim_results = results

    def finalize(self) -> RunRecord:
        circuit_rec = CircuitRecord.from_circuit(self.circuit)
        df_rec = DataFrameRecord(self.dataframe) if self.dataframe is not None else None
        res_rec = SimulationResultRecord(self.sim_results) if self.sim_results is not None else None

        contents = ["circuit"]
        if df_rec: contents.append("dataframe")
        if res_rec: contents.append("sim_results")

        manifest = RunManifest(
            run_id=str(uuid.uuid4()),
            created_utc=_dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            circuit_name=self.circuit_name,
            circuit_format=circuit_rec.format,
            repetitions=self.repetitions,
            contents=contents,
            options=self.options,
            extra_meta=self.extra_meta,
        )

        return RunRecord(
            manifest=manifest,
            circuit=circuit_rec,
            dataframe=df_rec,
            sim_results=res_rec,
        )
