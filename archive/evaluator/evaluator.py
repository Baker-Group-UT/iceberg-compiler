"""
evaluator.py - Analyze permutations dataframes.
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd


def _perm_key(perm) -> Tuple[Tuple[int, int], ...]:
    """Normalize permutation to a tuple-of-tuples key."""
    if isinstance(perm, (list, tuple)):
        return tuple(tuple(t) for t in perm)
    return perm


def _codes_used_from_perm(perm: List[Tuple[int, int]]) -> int:
    """Number of unique patches used."""
    return len({p for p, _ in perm})


def _ensure_codes_used_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'codes_used' if missing."""
    if "codes_used" not in df.columns:
        df = df.copy()
        df["codes_used"] = df["permutation"].apply(_codes_used_from_perm)
    return df


def add_efficiency_score(
    df: pd.DataFrame,
    *,
    lambda_codes: float = 3.0,
    metric_name: str = "efficiency_score",
) -> pd.DataFrame:
    """
    efficiency_score = total_gates + lambda_codes * codes_used
    """
    df2 = _ensure_codes_used_column(df).copy()
    if "total_gates" not in df2.columns:
        raise ValueError("'total_gates' column required")
    df2[metric_name] = df2["total_gates"] + lambda_codes * df2["codes_used"]
    return df2


def attach_effectiveness_columns(df_summary: pd.DataFrame, eff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join effectiveness metrics (e.g., fidelity, logical_error_rate) onto summary by perm_key.
    """
    left = df_summary.copy()
    right = eff_df.copy()

    if "perm_key" not in left.columns:
        left["perm_key"] = left["permutation"].map(_perm_key)
    if "perm_key" not in right.columns:
        right["perm_key"] = right["permutation"].map(_perm_key)

    cols = [c for c in [
        "perm_key",
        "codes_used",
        "fidelity",
        "logical_error_rate",
        "postselected"
    ] if c in right.columns]

    merged = left.merge(right[cols], on="perm_key", how="left", suffixes=("", "_eff"))
    return merged.drop(columns=["perm_key"])
