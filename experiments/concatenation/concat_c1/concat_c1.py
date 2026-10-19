"""
concat_c1.py -- what the inner level is handed: C1 for a two-qubit program, drawn by Qiskit.

The program H(0); CX(0, 1) on one outer [[4,2,2]] patch. C1, flattened (SWAPs relabelled,
measurements deferred): GHZ encoding of the four outer data qubits, the logical H and CX as
iceberg gadgets, and the end-of-circuit syndrome extraction into the two outer ancillas. Each of
the six wires is one logical qubit of an inner [[4,2,2]] patch. A circuit, so Qiskit draws it;
no --from-csv. Full width.

    PYTHONPATH=. venv/bin/python -m experiments.concatenation.concat_c1.concat_c1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = Path(__file__).resolve().parents[2] / "shared" / "aqorn.mplstyle"
NAME = "concat_c1"


def main() -> None:
    from qiskit import QuantumCircuit

    from experiments.concatenation import construct as C
    from experiments.shared.figures import project_root

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    program = QuantumCircuit(2)
    program.h(0)
    program.cx(0, 1)
    outer = C.outer_translate(program, [(0, 0), (0, 1)], commute=False, merging=False)
    flat = C.flatten(outer.circuit)
    circuit = flat.program.remove_final_measurements(inplace=False)
    labels = ["d0 (x-anchor)", "d1 (lane 0)", "d2 (lane 1)", "d3 (z-anchor)", "a0 (Z check)",
              "a1 (X check)"]
    named = QuantumCircuit(*[__import__("qiskit").QuantumRegister(1, name=l.split()[0]) for l in labels])
    named.compose(circuit, qubits=list(range(6)), inplace=True)
    plt.style.use(STYLE)
    out = Path(args.output_dir) if args.output_dir else None
    if out is None:
        root = project_root() if callable(project_root) else None
        out = (Path(root) / "results" / "figures") if root else Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        path = out / f"{NAME}.{suffix}"
        if path.exists():
            path = out / f"{NAME}_v2.{suffix}"
        named.draw(output="mpl", style={"name": "bw", "fontsize": 9, "subfontsize": 8},
                   fold=-1, filename=str(path))
        print(f"[{NAME}] saved {path}")
    print(f"[{NAME}] {circuit.size()} gates, depth {circuit.depth()}")


if __name__ == "__main__":
    main()
