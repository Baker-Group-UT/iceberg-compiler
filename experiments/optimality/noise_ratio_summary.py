"""
noise_ratio_summary.py -- the numbers from `noise_ratio_packers.csv`, per idle/distance ratio.

At each ratio, at the calibrated noise level (`cal`: naive keeps its Fig 6 share of shots):
    vs naive   median logical-error (LER) and TVD reductions, geometric-mean LSR, for every arm
    paired     interaction partition + descent vs AQORN, AQORN's search alone (aqorn_raw) vs
               AQORN, and the two searches against each other: geometric-mean ratio, wins /
               losses, Wilcoxon signed-rank p, on LER, LSR and TVD
    model      the two searches on AQORN's objective at that ratio (lower is better)
    check      how often AQORN's final check fell back to naive; noiseless-check failures

    PYTHONPATH=. venv/bin/python -m experiments.optimality.noise_ratio_summary [--csv ...] [--alpha cal]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ARMS = ("interaction_partition", "interaction_partition_descent", "aqorn_raw", "aqorn")
SHORT = {"interaction_partition": "IP", "interaction_partition_descent": "IP+descent",
         "aqorn_raw": "AQORN search", "aqorn": "AQORN"}


def _paired(wide: pd.DataFrame, left: str, right: str) -> str:
    from scipy.stats import wilcoxon

    ok = wide[left].notna() & wide[right].notna() & (wide[left] > 0) & (wide[right] > 0)
    r = np.log(wide.loc[ok, left] / wide.loc[ok, right])
    if not len(r):
        return "n=0"
    nonzero = r[r.abs() > 1e-12]
    p = wilcoxon(nonzero).pvalue if len(nonzero) > 5 else float("nan")
    return (f"x{np.exp(r.mean()):.3f} (lower {int((r < -1e-12).sum())}, equal "
            f"{int((r.abs() <= 1e-12).sum())}, higher {int((r > 1e-12).sum())}, p={p:.2g}, n={len(r)})")


def summarize(frame: pd.DataFrame, alpha: str) -> None:
    lsr, ler, tvd = f"lsr_{alpha}", f"ler_{alpha}", f"tvd_{alpha}"
    bad = frame[frame["correct"] == False]  # noqa: E712
    print(f"rows {len(frame)}, circuits {frame.circuit.nunique()}, ratios "
          f"{sorted(frame.ratio.unique())}; noiseless-check failures {len(bad)}; "
          f"arm errors {int(frame['error'].notna().sum()) if 'error' in frame else 0}\n")
    for ratio, part in frame.groupby("ratio"):
        wide = {m: part.pivot_table(index="circuit", columns="arm", values=m)
                for m in (lsr, ler, tvd, "model_value")}
        fell = part[part.arm == "aqorn"].aqorn_fell_back.astype(bool)
        idle, distance = part.idle_scalar.iloc[0], part.distance_scalar.iloc[0]
        print(f"=== ratio {ratio:g} (idle x{idle:g}, distance x{distance:g}); AQORN fell back to "
              f"naive on {int(fell.sum())}/{len(fell)}; median alpha_cal "
              f"{part.drop_duplicates('circuit').alpha_cal.median():.3g}")
        visible = wide[tvd]["naive"] > 0
        for arm in ARMS:
            ler_change = 100 * (wide[ler][arm] / wide[ler]["naive"] - 1)
            tvd_change = 100 * (wide[tvd][arm][visible] / wide[tvd]["naive"][visible] - 1)
            lsr_gm = np.exp(np.log(wide[lsr][arm] / wide[lsr]["naive"]).mean())
            print(f"  {SHORT[arm]:13s} vs naive (median change): LER {ler_change.median():+6.1f}%  "
                  f"TVD {tvd_change.median():+6.1f}% ({int(visible.sum())} circ.)  "
                  f"LSR x{lsr_gm:.3f} (geomean)")
        for left, right in (("interaction_partition_descent", "aqorn"), ("aqorn_raw", "aqorn"),
                            ("interaction_partition_descent", "aqorn_raw")):
            print(f"  {SHORT[left]} / {SHORT[right]}:")
            for label, metric, sub in (("LER", ler, None), ("LSR", lsr, None),
                                       ("TVD", tvd, visible)):
                w = wide[metric] if sub is None else wide[metric][sub]
                print(f"      {label}: {_paired(w, left, right)}")
        mv = wide["model_value"]
        gap = 100 * (mv["interaction_partition_descent"] / mv["aqorn_raw"] - 1)
        print(f"  objective, IP+descent vs AQORN search: median {gap.median():+.2f}%, IP+descent "
              f"lower on {int((gap < -1e-9).sum())}, equal {int((gap.abs() <= 1e-9).sum())}, "
              f"higher {int((gap > 1e-9).sum())}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--csv", default="results/noise_ratio_packers/noise_ratio_packers.csv")
    parser.add_argument("--alpha", default="cal", choices=("cal", "fig6", "a1"))
    args = parser.parse_args()
    summarize(pd.read_csv(args.csv), args.alpha)


if __name__ == "__main__":
    main()
