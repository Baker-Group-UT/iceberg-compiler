"""The worked-example circuits behind the merging diagram (Fig. B), before and after.

This is not a plot but the qiskit rendering of the 13-instruction toy circuit the
merging diagram walks through: two patches at k = 4 (q0..q3 in patch 0, q4..q7 in
patch 1), designed so each admissibility mechanism fires exactly once.

Two circuits are drawn, and the pair is the point -- the second is the first put
through the real merging pass, so panel (d) traces a rewrite the compiler
actually produces rather than one drawn by hand:

    merging_example_circuit          the 13-instruction program
    merging_example_circuit_merged   the 8 instructions it emits as

    layer 1   H q0..q3          full coverage        -> hh macro
    layer 2   X q0, X q2        even 2-of-4 subset   -> xx macro
    layer 3   CX q0->q5, q1->q4, q2->q6, q3->q7
              covers both patches, slots 4<->5 crossed -> cxt after ONE routing swap
    layer 4   H q0, H q1, H q3  3 of 4 lanes         -> inadmissible, stays targeted

Thirteen instructions emit as six: hh, xx, cxt (+1 swap), and three targeted H.
The final H group is the coverage-failure contrast, not leftovers: same gate kind,
one lane short, full price.

`--trace` runs the circuit through the real merging pass
(translator.merging.preprocessing) and prints every stage of every pass --
layering, occupancy, proposed candidates, the selected subset, transversal
proposals with their routing swaps, the picks dropped by the transversal filter,
and the rewritten circuit -- so the diagram can be drawn against what the code
actually does rather than a story about it. The savings it reports are measured,
fit for the Fig. B caption. Transversal swap merges are enabled to match the
relaxed pipeline (the crossed cxt does not fire without them; --no-swap-merges
shows that).

Deviations from the standard figure script shape, on purpose:
  - No CSV cache and no --from-csv: there is no data, the circuit is the spec.
  - No seed: nothing is random.
  - Two figures from one script, since neither circuit means much alone.
  - No house style. The output is not a paper figure but Inkscape source material
    for the hand-drawn merging diagram, where every stroke gets replaced anyway,
    so it renders as a plain qiskit `bw` circuit on matplotlib defaults. Only the
    two font-export rcParams survive, and neither changes how the drawing looks:
    svg.fonttype none keeps labels as editable <text> instead of outlined paths
    (without it the SVG is useless in Inkscape), and pdf.fonttype 42 keeps the
    companion PDF free of Type 3 fonts in case it is ever dropped into the paper.

Run:
    PYTHONPATH=. venv/bin/python \
        experiments/figures/merging_example_circuit/merging_example_circuit.py [--trace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

NAME = "merging_example_circuit"
HERE = Path(__file__).resolve().parent

# Drawing the merged circuit runs the real merging pass, so the repo root has to
# be importable. Adding it here means the script runs plainly, without a
# PYTHONPATH incantation.
if str(HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[2]))

# One lane per program qubit: q0..q3 fill patch 0, q4..q7 fill patch 1 (k = 4).
PERMUTATION = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
N_CODE = 6  # k = n_code - 2 = 4

# Pipeline settings the merged circuit is produced at. Swap merges are on to
# match the relaxed pipeline; the crossed `cxt` does not fire without them.
ALLOW_SWAP_MERGES = True
NUM_PASSES = 2


def project_root() -> Path | None:
    """Nearest ancestor containing CLAUDE.md, or None when the repo stands alone."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "CLAUDE.md").exists():
            return parent
    return None


def output_dir(override: Path | None = None) -> Path:
    if override:
        target = override
    else:
        root = project_root()
        target = (root / "results" / "figures") if root else HERE / "out"
    target.mkdir(parents=True, exist_ok=True)
    return target


def toy_circuit() -> QuantumCircuit:
    """The 13-instruction merging example: patches {q0..q3} and {q4..q7} at k = 4."""
    qc = QuantumCircuit(8)
    # hh: an H on every occupied lane of patch 0.
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    # xx: an even (2-of-4) subset, anchors cancel.
    qc.x(0)
    qc.x(2)
    # cxt: controls fill patch 0, targets fill patch 1; slots 4 and 5 are crossed,
    # so alignment costs exactly one routing swap on patch 1.
    qc.cx(0, 5)
    qc.cx(1, 4)
    qc.cx(2, 6)
    qc.cx(3, 7)
    # Coverage failure: 3 of 4 lanes, inadmissible as hh, emitted targeted.
    qc.h(0)
    qc.h(1)
    qc.h(3)
    assert sum(1 for _ in qc.data) == 13, "the worked example is 13 instructions"
    return qc


def merged_circuit() -> QuantumCircuit:
    """The toy circuit after the real merging pass: 13 instructions emit as 8.

    Not a re-enactment -- this is `preprocess_program_for_merging` itself, at the
    settings the relaxed pipeline runs, so the "after" panel cannot show a
    rewrite the compiler would not produce. The two routing swaps are why the
    count lands at 8 rather than 6: the misaligned `cxt` pays for its
    realignment before the macro and gives it back after.
    """
    from translator.merging.preprocessing import preprocess_program_for_merging

    return preprocess_program_for_merging(
        toy_circuit(),
        PERMUTATION,
        allow_transversal_swap_merges=ALLOW_SWAP_MERGES,
        num_merge_passes=NUM_PASSES,
        n_code=N_CODE,
    )


def draw(out: Path) -> list[Path]:
    """Render the program circuit and its merged form; return every file written."""
    # Deliberately unstyled: stock matplotlib defaults, so this comes out as a
    # plain qiskit black-and-white circuit rather than a house-style figure.
    # "default" first in case a user matplotlibrc has opinions.
    plt.style.use("default")
    # The only two overrides, and neither is visual: text stays as <text> for
    # Inkscape, and the PDF embeds TrueType rather than Type 3.
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42

    written: list[Path] = []
    for stem, circuit in (
        (NAME, toy_circuit()),
        (f"{NAME}_merged", merged_circuit()),
    ):
        fig = circuit.draw(output="mpl", style="bw")
        for suffix in (".svg", ".pdf"):
            path = out / f"{stem}{suffix}"
            fig.savefig(path)
            written.append(path)
        plt.close(fig)
    return written


# --------------------------------------------------------------------------
# --trace: walk the real merging pass and narrate each stage.
# --------------------------------------------------------------------------

def _fmt_node(node, layer_index, qubit_indices) -> str:
    qs = ",".join(f"q{qubit_indices[q]}" for q in node.qargs)
    return f"{node.op.name.upper()}({qs})@L{layer_index[node]}"


def _fmt_record(rec, layer_index, qubit_indices) -> str:
    covered = ",".join(f"q{qubit_indices[q]}" for q in rec.macro_qargs)
    spans = sorted(layer_index[n] for n in rec.nodes)
    swaps = [
        f"{op.gate.name}({','.join(f'q{qubit_indices[q]}' for q in op.qargs)})"
        for op in (*rec.pre_ops, *rec.post_ops)
    ]
    swap_note = f"  swaps: {', '.join(swaps)}" if swaps else ""
    return (
        f"{rec.macro_gate.name} on [{covered}]  absorbs {len(rec.nodes)} gates"
        f"  layers L{spans[0]}..L{spans[-1]}{swap_note}"
    )


def trace(allow_swap_merges: bool = True) -> None:
    from qiskit.converters import circuit_to_dag

    from translator.merging.conflicts import (
        filter_single_qubit_conflicting_with_transversal,
        merge_work_table,
    )
    from translator.merging.layer import layer_by_node
    from translator.merging.preprocessing import (
        _build_patch_lanes,
        _merge_savings_for_records,
    )
    from translator.merging.selection import select_best_group
    from translator.merging.settings import DEFAULT_SETTINGS
    from translator.merging.single_qubit import (
        find_hh_merges_for_patch,
        find_xx_merges_for_patch,
        find_zz_merges_for_patch,
    )
    from translator.merging.transversal import find_transversal_merges

    work_table = merge_work_table(N_CODE)
    circuit = toy_circuit()
    total_1q = total_2q = 0.0

    print(f"program: {len(circuit.data)} instructions, n_code={N_CODE} (k={N_CODE - 2})")
    print(f"transversal swap merges: {'ON' if allow_swap_merges else 'OFF'}")
    print(circuit.draw(output="text"))

    for pass_number in (1, 2):
        print(f"\n=== pass {pass_number} " + "=" * 50)

        # Step 1: layer the circuit.
        dag = circuit_to_dag(circuit)
        layer_index, order_index = layer_by_node(dag)
        qubit_indices = {q: circuit.find_bit(q).index for q in circuit.qubits}
        print("\n[1] layers (gate -> time slice):")
        for node in sorted(layer_index, key=order_index.get):
            print(f"    {_fmt_node(node, layer_index, qubit_indices)}")

        # Step 2: occupancy from the permutation.
        patch_lanes = _build_patch_lanes(PERMUTATION, circuit.qubits)
        print("\n[2] occupancy (patch -> lanes):")
        for patch, qubits in sorted(patch_lanes.items()):
            lanes = ",".join(f"q{qubit_indices[q]}" for q in qubits)
            print(f"    patch {patch}: [{lanes}]")

        # Step 3: propose single-qubit-gate groups.
        consumed: set = set()
        candidates = []
        for _, qubits in patch_lanes.items():
            for finder in (find_hh_merges_for_patch, find_xx_merges_for_patch,
                           find_zz_merges_for_patch):
                candidates.extend(
                    finder(layer_index, order_index, qubits, DEFAULT_SETTINGS, consumed)
                )
        print(f"\n[3] proposed single-qubit groups ({len(candidates)}):")
        for rec in candidates:
            print(f"    {_fmt_record(rec, layer_index, qubit_indices)}")

        # Step 4: select a non-conflicting subset.
        selected = list(
            select_best_group(candidates, layer_index, order_index, qubit_indices,
                              dag=dag, work_table=work_table)
        )
        print(f"\n[4] selected ({len(selected)} of {len(candidates)}):")
        for rec in selected:
            print(f"    {_fmt_record(rec, layer_index, qubit_indices)}")
        consumed.update({r.first for r in selected})
        consumed.update({r.second for r in selected})

        # Step 5: propose transversal groups (alignment priced via routing swaps).
        transversal = find_transversal_merges(
            layer_index, order_index, patch_lanes, qubit_indices, consumed,
            allow_swap_merges, n_code=N_CODE,
        )
        print(f"\n[5] transversal groups ({len(transversal)}):")
        for rec in transversal:
            print(f"    {_fmt_record(rec, layer_index, qubit_indices)}")

        # Step 6: drop single-qubit picks the transversal choices reorder.
        survivors = filter_single_qubit_conflicting_with_transversal(
            selected, transversal, layer_index, order_index
        )
        dropped = [r for r in selected if r not in survivors]
        print(f"\n[6] dropped by transversal conflict ({len(dropped)}):")
        for rec in dropped:
            print(f"    {_fmt_record(rec, layer_index, qubit_indices)}")

        records = list(survivors) + list(transversal)
        if not records:
            print("\n    nothing merged; pass ends")
            break

        s1, s2 = _merge_savings_for_records(records, work_table)
        total_1q += s1
        total_2q += s2

        # Step 7: rewrite, delaying displaced gates.
        from translator.merging.rewrite import build_rewritten_circuit
        circuit = build_rewritten_circuit(circuit, dag, records, layer_index, order_index)
        print(f"\n[7] rewritten circuit ({len(circuit.data)} instructions), "
              f"pass savings: {s1:g} 1q, {s2:g} 2q physical gates")
        print(circuit.draw(output="text"))

    print("\n=== result " + "=" * 49)
    print(f"instructions: 13 -> {len(circuit.data)}")
    print(f"measured savings: {total_1q:g} 1q, {total_2q:g} 2q physical gates "
          f"(for the Fig. B caption)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where the SVG and PDF land (default: results/figures/)")
    parser.add_argument("--trace", action="store_true",
                        help="narrate the real merging pass on this circuit")
    parser.add_argument("--no-swap-merges", action="store_true",
                        help="with --trace: disable transversal swap merges")
    args = parser.parse_args()

    if args.trace:
        trace(allow_swap_merges=not args.no_swap_merges)
        return

    for path in draw(output_dir(args.output_dir)):
        size = path.stat().st_size
        print(f"wrote {path} ({size} bytes)")
        assert size > 0, f"{path} is empty"


if __name__ == "__main__":
    main()
