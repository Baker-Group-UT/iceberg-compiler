"""
Produce a simple circuit figure that highlights encoding, a single logical X
operation, and decoding within one Iceberg module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Provide Matplotlib a writable cache to avoid sandbox warnings/errors.
MPL_CACHE_DIR = THIS_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

FONT_CACHE_DIR = THIS_DIR / ".cache"
(FONT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib
from qiskit import QuantumCircuit

from iceberg.iceberg_set import IcebergSet

matplotlib.use("agg")


def build_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(name="iceberg_encoding_demo")
    iceberg = IcebergSet(
        n_program=1,
        n=4,
        circuit=circuit,
    )
    iceberg.encode()
    iceberg.X(0)
    iceberg.Z(0)
    iceberg.decode()
    iceberg.measure()
    return circuit


def main() -> None:
    circuit = build_circuit()
    out_path = THIS_DIR / "iceberg_encoding.svg"
    circuit.draw("mpl", filename=str(out_path), fold=-1)


if __name__ == "__main__":
    main()
