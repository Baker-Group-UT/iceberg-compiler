"""
recorder.repository.py - Handles saving and loading RunRecord objects.

Supports optional compression, automatic file naming, integrity hashing,
and loading runs by timestamp or circuit name.
"""

from __future__ import annotations

import os
import json
import gzip
import hashlib
from typing import List, Optional

from .run_builder import RunRecord


class RunRepository:
    """
    Repository for storing and retrieving finalized simulation runs.

    Saves and loads RunRecord objects to disk as JSON (optionally gzipped).
    Automatically handles filename formatting and content hashing.
    """

    def __init__(self, base_dir: str = "results/permutations"):
        """
        Args:
            base_dir: Directory to store or retrieve runs from.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path_for(self, record: RunRecord, compress: bool = True) -> str:
        """
        Generate a filename for a run based on timestamp, circuit name, and repetitions.

        Args:
            record: The RunRecord to save.
            compress: Whether the file will be saved as .gz.

        Returns:
            Full path to the file.
        """
        name = record.manifest.circuit_name.strip().replace(" ", "_")
        ts = record.manifest.created_utc.replace(":", "").replace("-", "").replace("Z", "")
        fname = f"{ts}_{name}_R{record.manifest.repetitions}.json"
        if compress:
            fname += ".gz"
        return os.path.join(self.base_dir, fname)

    def save_run(self, record: RunRecord, compress: bool = True) -> str:
        """
        Save a RunRecord to disk.

        Args:
            record: The RunRecord to save.
            compress: If True, save as .json.gz; else .json.

        Returns:
            Path to the saved file.
        """
        path = self._path_for(record, compress=compress)
        packed = record.to_dict()

        # Compute integrity hash
        integrity_src = json.dumps(packed, sort_keys=True)
        record.manifest.integrity_sha256 = hashlib.sha256(integrity_src.encode()).hexdigest()
        packed["meta"]["integrity_sha256"] = record.manifest.integrity_sha256

        # Save to disk
        if compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(packed, f, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(packed, f, ensure_ascii=False, indent=2)

        return path

    def load_run(self, path: str) -> RunRecord:
        """
        Load a RunRecord from a saved file.

        Args:
            path: Path to the saved .json or .json.gz file.

        Returns:
            Rehydrated RunRecord object.
        """
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        return RunRecord.from_dict(raw)

    def list_runs(self, circuit_name: Optional[str] = None) -> List[str]:
        """
        List all run files in the repository (optionally filtered by circuit name).

        Args:
            circuit_name: If provided, only include runs for this circuit.

        Returns:
            List of file paths (sorted newest to oldest).
        """
        files = [f for f in os.listdir(self.base_dir) if f.endswith((".json", ".json.gz"))]
        if circuit_name:
            name = circuit_name.strip().replace(" ", "_")
            files = [f for f in files if f"_{name}_" in f]

        return sorted(
            [os.path.join(self.base_dir, f) for f in files],
            key=os.path.getmtime,
            reverse=True
        )

    def load_latest(self, circuit_name: Optional[str] = None) -> Optional[RunRecord]:
        """
        Load the most recent run (optionally filtered by circuit name).

        Args:
            circuit_name: If provided, limit to runs for this circuit.

        Returns:
            Most recent RunRecord, or None if no matching runs exist.
        """
        paths = self.list_runs(circuit_name)
        return self.load_run(paths[0]) if paths else None