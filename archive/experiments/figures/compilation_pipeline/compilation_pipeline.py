"""
compilation_pipeline.py
-----------------------

Saves circuit SVGs at every stage of the compilation pipeline for a user-defined
5-qubit logical circuit.  Each file corresponds to a stage described in the paper
figure caption:

    01_input.svg          – raw input program circuit
    02_gate_cancelled.svg – after first gate cancellation pass
    03_h_commuted.svg     – after H-commutation pass
    04_preprocessed.svg   – after second gate cancellation pass
    05_packed_macros.svg  – logical circuit with merged macro gates (HH, CXT, …)
    06_physical_raw.svg   – full iceberg-encoded circuit before final optimisation
    07_physical_final.svg – after final inverse-gate cancellation passes

Run with:

    python -m experiments.figures.compilation_pipeline.compilation_pipeline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from translator.core import Translation
from translator.merging import preprocess_program_for_merging_with_stats
from translator.passes import (
    cancel_inverse_gates,
    cancel_inverse_gates_through_swaps,
    commute_hadamards,
)

# Uncomment to use the merge-maximizing packer instead of the fixed permutation below.
# from packer.merge_packer import MergeMaximizingPacker

# ---------------------------------------------------------------------------
# Drawing settings
# ---------------------------------------------------------------------------

LOGICAL_FOLD  = -1   # -1 = no line-wrap for (compact) logical circuits
PHYSICAL_FOLD = 60   # wrap physical circuits at 60 gates per row
STYLE = {"name": "bw"}  # clean black-and-white style

OUTPUT_DIR = THIS_DIR


# ===========================================================================
# USER CIRCUIT — edit this section to define your 5-qubit input
# ===========================================================================

def build_input_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(5)

    # Example circuit designed to exercise every pipeline stage:
    #   - H-H pair on q0 (will cancel in stage 2)
    #   - H gates that H-commutation will absorb into XCX/CZ rewrites
    #   - CNOT pairs between co-patched qubits (q0/q1, q2/q3) eligible for
    #     CXT merge macros
    qc.h(0)
    qc.h(0)         # back-to-back H → cancels
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1, 2)
    qc.h(2)
    qc.cx(2, 3)
    qc.cx(0, 1)     # second CNOT on patch 0 pair → eligible for CXT merge
    qc.h(1)
    qc.cx(2, 3)     # second CNOT on patch 1 pair → eligible for CXT merge
    qc.h(3)
    qc.x(4)
    qc.h(4)
    qc.z(4)
    qc.cx(3, 4)
    qc.h(0)

    return qc


# ---------------------------------------------------------------------------
# Default permutation: tight trivial packing for 5 qubits into 3 patches
#   q0, q1 → patch 0 (slots 0, 1)
#   q2, q3 → patch 1 (slots 0, 1)
#   q4     → patch 2 (slot 0, solo)
# ---------------------------------------------------------------------------
DEFAULT_PERMUTATION = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]


# ===========================================================================
# End of user-editable section
# ===========================================================================


def save_circuit_svg(
    circuit: QuantumCircuit,
    path: Path,
    title: str | None = None,
    fold: int = LOGICAL_FOLD,
) -> None:
    """Draw a circuit and save it as an SVG file."""
    fig = circuit.draw(output="mpl", fold=fold, style=STYLE)
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}  ({circuit.num_qubits}q, {len(circuit)} gates)")


def run_pipeline(
    input_circuit: QuantumCircuit,
    permutation: list[tuple[int, int]],
    output_dir: Path,
) -> None:
    # ------------------------------------------------------------------
    # Stage 1 — input circuit
    # ------------------------------------------------------------------
    print("[1/7] Input circuit")
    save_circuit_svg(
        input_circuit,
        output_dir / "01_input.svg",
        title="Input program circuit",
    )

    # ------------------------------------------------------------------
    # Stage 2 — first gate cancellation
    # ------------------------------------------------------------------
    print("[2/7] Gate cancellation")
    gate_cancelled = cancel_inverse_gates(input_circuit)
    save_circuit_svg(
        gate_cancelled,
        output_dir / "02_gate_cancelled.svg",
        title="After gate cancellation",
    )

    # ------------------------------------------------------------------
    # Stage 3 — H commutation
    # ------------------------------------------------------------------
    print("[3/7] H commutation")
    h_commuted = commute_hadamards(gate_cancelled)
    save_circuit_svg(
        h_commuted,
        output_dir / "03_h_commuted.svg",
        title="After H commutation",
    )

    # ------------------------------------------------------------------
    # Stage 4 — second gate cancellation (= preprocessed circuit)
    # ------------------------------------------------------------------
    print("[4/7] Second gate cancellation")
    preprocessed = cancel_inverse_gates(h_commuted)
    save_circuit_svg(
        preprocessed,
        output_dir / "04_preprocessed.svg",
        title="After second gate cancellation (preprocessed)",
    )

    # ------------------------------------------------------------------
    # Stage 5 — merge preprocessing: insert macro gates
    # ------------------------------------------------------------------
    print("[5/7] Merge preprocessing (macro gates)")
    merged_circuit, s1q, s2q, _sw = preprocess_program_for_merging_with_stats(
        preprocessed, permutation
    )
    print(f"        merge savings: 1q={s1q:.1f}  2q={s2q:.1f}")
    save_circuit_svg(
        merged_circuit,
        output_dir / "05_packed_macros.svg",
        title="Logical circuit with merged macro gates",
    )

    # ------------------------------------------------------------------
    # Stage 6 — physical translation (before cancellation)
    # Drives Translation manually to capture raw iceberg circuit.
    # ------------------------------------------------------------------
    print("[6/7] Physical translation (raw, before cancellation)")
    translation = Translation(
        permutation=permutation,
        program_circuit=merged_circuit,
    )
    for op in translation.program_circuit.data:
        translation.translate_gate(op)
    translation.context.iceberg_set.decode()
    translation.context.iceberg_set.measure()

    raw_physical = translation.context.iceberg_set.circuit.copy()
    save_circuit_svg(
        raw_physical,
        output_dir / "06_physical_raw.svg",
        title="Physical circuit (after translation, before optimisation)",
        fold=PHYSICAL_FOLD,
    )

    # ------------------------------------------------------------------
    # Stage 7 — final cancellation passes
    # ------------------------------------------------------------------
    print("[7/7] Final cancellation passes")
    c1 = cancel_inverse_gates(raw_physical)
    c2 = cancel_inverse_gates_through_swaps(c1)
    final_physical = cancel_inverse_gates(c2)
    save_circuit_svg(
        final_physical,
        output_dir / "07_physical_final.svg",
        title="Physical circuit (after final optimisation)",
        fold=PHYSICAL_FOLD,
    )


def main() -> None:
    input_circuit = build_input_circuit()
    permutation = DEFAULT_PERMUTATION

    # Optionally use MergeMaximizingPacker to find the best permutation:
    from archive.packer.merge_packer import MergeMaximizingPacker
    from translator.passes import cancel_inverse_gates, commute_hadamards
    preprocessed_for_pack = cancel_inverse_gates(commute_hadamards(cancel_inverse_gates(input_circuit)))
    packer = MergeMaximizingPacker()
    permutation = packer.pack(preprocessed_for_pack, num_codes=3).permutation

    print(f"Input circuit: {input_circuit.num_qubits} qubits, {len(input_circuit)} gates")
    print(f"Permutation:   {permutation}")
    print(f"Output dir:    {OUTPUT_DIR}\n")

    run_pipeline(input_circuit, permutation, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
