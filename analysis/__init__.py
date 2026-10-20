"""
analysis — measuring what the compiler produced.

Gathers the result-processing code that used to be scattered one file to a package:
`evaluator.gate_count_stats` (a package with a single module in it) and all of
`pipeline.analysis`. They do one job between them -- turn a translated circuit or a
simulation result into a number you can compare -- so they read better together than
as two half-packages either side of the pipeline boundary.

    gate_count_stats     one- and two-qubit gate totals; the metric every benchmark
                         in this repo is scored with
    postselection        Iceberg postselection, and the selection rate that comes with
                         it -- the quantity `packing.resource` predicts
    result_row           one simulation result -> one DataFrame row
    result_types         containers, including the ScanResult the scan tasks return
    metrics              fidelity and TVD
    print_df             table rendering
    extremal_selection   pick the best and worst permutations out of a scan
"""

from .gate_count_stats import one_two_qubit_gate_totals

__all__ = ["one_two_qubit_gate_totals"]
