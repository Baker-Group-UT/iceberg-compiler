"""
Compare physical translations of a small logical program under two permutations.

The logical circuit:
    h(0); h(3); x(1); cx(0, 1); cx(3, 2); z(2)

Two permutations / packings are evaluated:
    - "trivial": [(0, 0), (0, 1), (1, 0), (1, 1)]
    - "optimal": [(0, 0), (1, 0), (1, 1), (0, 1)]

Each is translated via the `Translator` to produce a physical Iceberg circuit,
and the results are written as SVG drawings in this directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure matplotlib/font caches are writable inside the CLI sandbox.
MPL_CACHE_DIR = THIS_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

FONT_CACHE_DIR = THIS_DIR / ".cache"
(FONT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib
from qiskit import QuantumCircuit

from translator.config import TranslatorConfig
from translator.core import Translator

matplotlib.use("agg")


PERMUTATIONS: Dict[str, List[Tuple[int, int]]] = {
    "trivial": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "optimal": [(0, 0), (1, 0), (1, 1), (0, 1)],
}


def _build_program_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(10, name="logical_reference")
    qc.h(0)
    qc.h(3)
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(3, 2)
    qc.z(2)
    return qc


def _draw_svg(circuit: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit.draw("mpl", filename=str(path), fold=-1)


def main() -> None:
    output_dir = THIS_DIR
    program_circuit = _build_program_circuit()
    _draw_svg(program_circuit, output_dir / "logical_reference.svg")

    translator = Translator(config=TranslatorConfig(encode_override=True))
    for label, permutation in PERMUTATIONS.items():
        translation = translator.translate(program_circuit.copy(), permutation)
        _draw_svg(translation.circuit, output_dir / f"physical_{label}.svg")


if __name__ == "__main__":
    main()
