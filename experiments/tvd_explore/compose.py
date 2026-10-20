"""
compose.py -- build a variant's per-circuit plan table for `ablation_variant`.

    w10                weight 10 everywhere (Experiment8a)
    w10_exh<rule>      exhaustive measured placement (exhaustive_small.csv) for the circuits
                       it covered, selected by <rule>; weight 10 everywhere else

Rules (per circuit, over every candidate placement and commutation setting):
    fig6        lowest silent-per-kept at the circuit's Fig 6 alpha, keeping >= baseline's shots
    a1          the same at alpha = 1 (the hardware, not Fig 6's calibration)
    fig6_w10    lowest silent-per-kept at Fig 6 alpha keeping >= weight 10's shots

Writes results/tvd_explore/plans_<name>.csv (name, placement, commute, source).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUT = Path("results/tvd_explore")


def weight_plans(weight: float) -> pd.DataFrame:
    frame = pd.read_csv("results/revision/exp8/planner_static.csv")
    frame = frame[frame["silent_weight"] == weight][["name", "placement", "commute"]]
    return frame.assign(source=f"w{weight:g}")


def exhaustive_choice(rule: str, w10: pd.DataFrame) -> pd.DataFrame:
    from experiments.revision.exp8_undetected.frames import translate  # noqa: F401 (import check)

    frame = pd.read_csv(OUT / "exhaustive_small.csv")
    tag = "a1" if rule == "a1" else "fig6"
    rows = []
    for name, block in frame.groupby("name"):
        base = block[block["candidate"] == -1].iloc[0]
        pool = block[block["candidate"] >= 0]
        floor = 1 - base[f"pd_{tag}"]
        if rule == "fig6_w10":
            mine = w10.set_index("name").loc[name]
            match = pool[(pool["placement"] == mine["placement"]) & (pool["commute"] == bool(mine["commute"]))]
            floor = (1 - match[f"pd_{tag}"]).max() if not match.empty else floor
        keeps = pool[(1 - pool[f"pd_{tag}"]) >= floor - 1e-12]
        chosen = (keeps if not keeps.empty else pool).sort_values(f"sk_{tag}").iloc[0]
        rows.append({"name": name, "placement": chosen["placement"],
                     "commute": bool(chosen["commute"]), "source": f"exhaustive:{rule}"})
    return pd.DataFrame(rows)


def main(rule: str) -> Path:
    w10 = weight_plans(10.0)
    if rule == "w10":
        plans = w10
    else:
        picked = exhaustive_choice(rule, w10)
        plans = pd.concat([picked, w10[~w10["name"].isin(picked["name"])]], ignore_index=True)
    name = "w10" if rule == "w10" else f"w10_exh{rule}"
    path = OUT / f"plans_{name}.csv"
    plans.to_csv(path, index=False)
    print(f"wrote {path}: {plans['source'].value_counts().to_dict()}")
    return path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig6")
