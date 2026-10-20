"""
variant_summary.py -- the compiler decisions of 2026-10-10/11 read off `noise_ratio_packers
--variants` (Fig 6's 71 circuits x 7 idle/distance ratios, calibrated noise level `cal`).

Each comparison is paired per circuit at one ratio: geometric-mean ratio, lower / equal / higher,
Wilcoxon signed-rank p (zero differences dropped), on logical error (LER), shots kept (LSR) and
TVD (circuits whose naive TVD is above zero). Arms that land on the same placement share one
measurement, so an unchanged plan is an exact tie.

    commutation fix   aqorn_comm vs aqorn (the b664c94 standard). Adopt unless significantly
                      worse (p < 0.05) on LER or TVD at Fig 6's ratio (64).
    random restarts   aqorn_comm_norestart vs aqorn_comm. Drop them if LER, TVD and LSR stay
                      within +-1% (geometric mean) and n.s. at every ratio.
    grid alignment    aqorn_comm_grid vs aqorn_comm. Adopt only if significantly better on LER,
                      TVD or LSR at >= 1 ratio and never significantly worse.

    PYTHONPATH=. venv/bin/python -m experiments.optimality.variant_summary \
        [--csv results/noise_ratio_packers/noise_ratio_packers_variants.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

CSV = "results/noise_ratio_packers/noise_ratio_packers_variants.csv"
FIG6_RATIO = 64.0
COMPARISONS = (("commutation fix", "aqorn_comm", "aqorn"),
               ("no random restarts", "aqorn_comm_norestart", "aqorn_comm"),
               ("grid alignment", "aqorn_comm_grid", "aqorn_comm"))
#: +1: higher is better; -1: lower is better.
SENSE = {"ler": -1, "lsr": +1, "tvd": -1}


def paired(wide: pd.DataFrame, left: str, right: str) -> dict:
    from scipy.stats import wilcoxon

    ok = wide[left].notna() & wide[right].notna() & (wide[left] > 0) & (wide[right] > 0)
    r = np.log(wide.loc[ok, left] / wide.loc[ok, right])
    nonzero = r[r.abs() > 1e-12]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) > 5 else float("nan")
    return dict(gm=float(np.exp(r.mean())) if len(r) else float("nan"), p=p, n=len(r),
                lower=int((r < -1e-12).sum()), equal=int((r.abs() <= 1e-12).sum()),
                higher=int((r > 1e-12).sum()))


def table(frame: pd.DataFrame, alpha: str = "cal") -> pd.DataFrame:
    rows = []
    for ratio, part in frame.groupby("ratio"):
        visible = part[part.arm == "naive"].set_index("circuit")[f"tvd_{alpha}"] > 0
        for name, left, right in COMPARISONS:
            if left not in set(part.arm) or right not in set(part.arm):
                continue
            plans = part.pivot_table(index="circuit", columns="arm", values="placement",
                                     aggfunc="first")
            commutes = part.pivot_table(index="circuit", columns="arm", values="commute",
                                        aggfunc="first")
            changed = int(((plans[left] != plans[right]) | (commutes[left] != commutes[right])).sum())
            seconds = part.pivot_table(index="circuit", columns="arm", values="arm_seconds")
            for metric in ("ler", "lsr", "tvd"):
                wide = part.pivot_table(index="circuit", columns="arm", values=f"{metric}_{alpha}")
                if metric == "tvd":
                    wide = wide[visible.reindex(wide.index).fillna(False)]
                stats = paired(wide, left, right)
                better = SENSE[metric] * np.log(stats["gm"]) > 0
                significant = stats["p"] < 0.05
                rows.append(dict(ratio=ratio, comparison=name, metric=metric, changed=changed,
                                 seconds_left=seconds[left].median(),
                                 seconds_right=seconds[right].median(), **stats,
                                 verdict=("better" if better else "worse") if significant
                                 else "n.s."))
    return pd.DataFrame(rows)


def decide(tab: pd.DataFrame) -> dict:
    out = {}
    comm = tab[tab.comparison == "commutation fix"]
    worse = comm[(comm.ratio == FIG6_RATIO) & comm.metric.isin(["ler", "tvd"])
                 & (comm.verdict == "worse")]
    out["standard_commutation"] = worse.empty
    rest = tab[tab.comparison == "no random restarts"]
    neutral = ((rest.gm - 1).abs() <= 0.01) & (rest.verdict == "n.s.")
    out["drop_random_restarts"] = bool(len(rest)) and bool(neutral.all())
    grid = tab[tab.comparison == "grid alignment"]
    out["grid_alignment"] = bool((grid.verdict == "better").any()) and not bool(
        (grid.verdict == "worse").any())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--alpha", default="cal", choices=("cal", "fig6", "a1"))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    print(f"rows {len(frame)}, circuits {frame.circuit.nunique()}, ratios "
          f"{sorted(frame.ratio.unique())}; noiseless-check failures "
          f"{int((frame['correct'] == False).sum())}; arm errors "  # noqa: E712
          f"{int(frame['error'].notna().sum()) if 'error' in frame else 0}")
    for arm in ("aqorn", "aqorn_comm", "aqorn_comm_norestart", "aqorn_comm_grid"):
        part = frame[frame.arm == arm]
        if len(part):
            print(f"  {arm:22s} fell back {int(part.fell_back.eq(True).sum())}"
                  f"/{len(part)}, aligned {int(part.aligned.eq(True).sum())}, "
                  f"commute=False {int((~part.commute.astype(bool)).sum())}, median plan "
                  f"{part.arm_seconds.median():.2f} s, max {part.arm_seconds.max():.1f} s")
    tab = table(frame, args.alpha)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(tab.round(4).to_string(index=False))
    print("decisions:", decide(tab))
    out = args.csv.replace(".csv", f"_summary_{args.alpha}.csv")
    tab.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
