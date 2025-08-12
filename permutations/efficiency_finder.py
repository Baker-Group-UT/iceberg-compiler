from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd

def _codes_used_from_perm(perm: List[Tuple[int, int]]) -> int:
    """Return number of distinct code patches in this permutation."""
    return len({p for p, _ in perm})

def _ensure_codes_used_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'codes_used' column if missing (derived from `permutation`)."""
    if "codes_used" not in df.columns:
        df = df.copy()
        df["codes_used"] = df["permutation"].apply(_codes_used_from_perm)
    return df

def extremal_per_codes(
    df: pd.DataFrame,
    *,
    metric: str = "total_gates",
    only_ft: bool = False,
    include_ties: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Find most- and least-efficient rows per number of codes used.
    """
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in df; have: {list(df.columns)}")

    df2 = _ensure_codes_used_column(df)

    if only_ft:
        if "fault_tolerant" not in df2.columns:
            raise ValueError("only_ft=True but 'fault_tolerant' column missing")
        df2 = df2[df2["fault_tolerant"]].copy()

    if df2.empty:
        empty = df2.copy()
        return {"best_by_codes": empty, "worst_by_codes": empty}

    # Compute per-row group mins/maxes as Series aligned to df2.index
    mins = df2.groupby("codes_used")[metric].transform("min")
    maxs = df2.groupby("codes_used")[metric].transform("max")

    if include_ties:
        best  = df2[df2[metric] == mins].copy()
        worst = df2[df2[metric] == maxs].copy()
    else:
        best  = (df2.sort_values(["codes_used", metric])
                    .groupby("codes_used", as_index=False).head(1).reset_index(drop=True))
        worst = (df2.sort_values(["codes_used", metric], ascending=[True, False])
                    .groupby("codes_used", as_index=False).head(1).reset_index(drop=True))

    best  = best.sort_values(["codes_used", metric]).reset_index(drop=True)
    worst = worst.sort_values(["codes_used", metric], ascending=[True, False]).reset_index(drop=True)

    return {"best_by_codes": best, "worst_by_codes": worst}

def add_efficiency_score(
    df: pd.DataFrame,
    *,
    lambda_codes: float = 3.0,
    metric_name: str = "efficiency_score",
) -> pd.DataFrame:
    """
    Add score = total_gates + lambda_codes * codes_used (lower is better).
    Tweak lambda_codes to penalize using more codes.
    """
    df2 = _ensure_codes_used_column(df).copy()
    if "total_gates" not in df2.columns:
        raise ValueError("'total_gates' column required")
    df2[metric_name] = df2["total_gates"] + lambda_codes * df2["codes_used"]
    return df2
