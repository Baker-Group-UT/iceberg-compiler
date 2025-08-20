"""
recorder - Structured run recording, saving, and loading system.

Exposes:
- RunBuilder: pipeline-side object for accumulating circuit/run data
- RunRecord: hydrated container for saved runs
- RunManifest: metadata for run history and integrity
- RunRepository: disk storage interface
- DataFrameRecord, CircuitRecord, SimulationResultRecord: typed containers
"""

from .run_builder import RunBuilder, RunRecord
from .manifest import RunManifest
from .repository import RunRepository

from .records import (
    DataFrameRecord,
    CircuitRecord,
    SimulationResultRecord,
)

__all__ = [
    "RunBuilder",
    "RunRecord",
    "RunManifest",
    "RunRepository",
    "DataFrameRecord",
    "CircuitRecord",
    "SimulationResultRecord",
]