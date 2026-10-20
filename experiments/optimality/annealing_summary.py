"""
annealing_summary.py -- does simulated annealing on AQORN's own objective beat AQORN's search?

Reads `packers_corpus.py --arms aqorn annealing --tag annealing` output and separates the three
things a difference can come from:
    search gap   annealing's model value (what both searches optimise) against AQORN's
    model gap    AQORN's full criterion AH* (verify's measured cost + silent term) against it
    measured     LSR and logical error rate at the circuit's Fig 6 alpha (frames)

    PYTHONPATH=. venv/bin/python -m experiments.optimality.annealing_summary \\
        --from-csv results/optimality_front/packers_corpus_annealing.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--from-csv", default="results/optimality_front/packers_corpus_annealing.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.from_csv)
    frame = frame[frame.get("correct", True) != False]  # noqa: E712
    wide = frame.pivot_table(index=["circuit", "num_qubits"], columns="arm",
                             values=["model_value", "cost", "lsr_fig6", "ler_fig6"])
    out = pd.DataFrame({
        "model_gap_pct": 100 * (wide["model_value"]["annealing"] / wide["model_value"]["aqorn"] - 1),
        "ah_gap_pct": 100 * (wide["cost"]["annealing"] / wide["cost"]["aqorn"] - 1),
        "lsr_pct": 100 * (wide["lsr_fig6"]["annealing"] / wide["lsr_fig6"]["aqorn"] - 1),
        "ler_pct": 100 * (wide["ler_fig6"]["annealing"] / wide["ler_fig6"]["aqorn"] - 1),
    }).reset_index()
    out["size"] = pd.cut(out["num_qubits"], [0, 9, 30, 1000], labels=["<10q", "10-30q", ">30q"])
    print(f"{len(out)} circuits; annealing vs AQORN (negative = annealing lower)")
    for label, column in (("search objective (model value)", "model_gap_pct"),
                          ("AQORN's full criterion AH*", "ah_gap_pct"),
                          ("measured LSR", "lsr_pct"), ("measured LER", "ler_pct")):
        series = out[column].dropna()
        better = (series < -1e-9).mean() if column != "lsr_pct" else (series > 1e-9).mean()
        print(f"  {label:32s} median {series.median():+.2f}%  mean {series.mean():+.2f}%  "
              f"annealing better on {better:.0%}  equal on {(series.abs() <= 1e-9).mean():.0%}")
    print(out.groupby("size", observed=True)[["model_gap_pct", "ah_gap_pct", "lsr_pct",
                                              "ler_pct"]].median().round(2).to_string())
    print("largest search gaps:")
    print(out.nsmallest(5, "model_gap_pct").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
