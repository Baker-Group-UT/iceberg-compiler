"""
recorder.manifest.py - Stores the class for a run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ------------------------
# RunManifest
# ------------------------

@dataclass
class RunManifest:
    run_id: str
    created_utc: str
    circuit_name: str
    circuit_format: str
    repetitions: int
    schema: int = 1
    contents: List[str] = field(default_factory=list)  # e.g. ["dataframe", "circuit"]
    integrity_sha256: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunManifest:
        return cls(**data)
