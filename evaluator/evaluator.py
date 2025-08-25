"""
evaluator.py - This is the main file for analyzing permutations dataframes.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd

def _perm_key(perm) -> Tuple[Tuple[int, int], ...]:
    """Converts a permutation list or tuple to a normalized tuple-of-tuples key."""
    if isinstance(perm, (list, tuple)):
        return tuple(tuple(t) for t in perm)
    return perm

def _codes_used_from_perm(perm: List[Tuple[int, int]]) -> int:
    """Returns the number of unique patches used in the permutation."""
    return len({p for p, _ in perm})

def _ensure_codes_used_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'codes_used' column to the DataFrame if it is not already present."""
    if "codes_used" not in df.columns:
        df = df.copy()
        df["codes_used"] = df["permutation"].apply(_codes_used_from_perm)
    return df

# def extremal_per_codes(
#     df: pd.DataFrame,
#     *,
#     metric: str = "total_gates",
#     only_ft: bool = False,
#     include_ties: bool = True,
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Identifies best and worst permutations by the given metric for each code count bucket.

#     Args:
#         df: Input DataFrame with simulation metrics.
#         metric: Column to rank by (e.g., 'total_gates').
#         only_ft: Whether to restrict to fault-tolerant circuits only.
#         include_ties: Whether to return all tied best/worst entries or just one.

#     Returns:
#         Dictionary with 'best_by_codes' and 'worst_by_codes' DataFrames.
#     """
#     if metric not in df.columns:
#         raise ValueError(f"metric '{metric}' not found in df; have: {list(df.columns)}")

#     df2 = _ensure_codes_used_column(df)

#     if only_ft:
#         if "fault_tolerant" not in df2.columns:
#             raise ValueError("only_ft=True but 'fault_tolerant' column missing")
#         df2 = df2[df2["fault_tolerant"]].copy()

#     if df2.empty:
#         empty = df2.copy()
#         return {"best_by_codes": empty, "worst_by_codes": empty}

#     mins = df2.groupby("codes_used")[metric].transform("min")
#     maxs = df2.groupby("codes_used")[metric].transform("max")

#     if include_ties:
#         best = df2[df2[metric] == mins].copy()
#         worst = df2[df2[metric] == maxs].copy()
#     else:
#         best = (
#             df2.sort_values(["codes_used", metric])
#             .groupby("codes_used", as_index=False)
#             .head(1)
#             .reset_index(drop=True)
#         )
#         worst = (
#             df2.sort_values(["codes_used", metric], ascending=[True, False])
#             .groupby("codes_used", as_index=False)
#             .head(1)
#             .reset_index(drop=True)
#         )

#     return {
#         "best_by_codes": best.reset_index(drop=True),
#         "worst_by_codes": worst.reset_index(drop=True)
#     }

def add_efficiency_score(
    df: pd.DataFrame,
    *,
    lambda_codes: float = 3.0,
    metric_name: str = "efficiency_score",
) -> pd.DataFrame:
    """
    Adds a linear efficiency score to the DataFrame: total_gates + lambda * codes_used.

    Args:
        df: Input DataFrame.
        lambda_codes: Weighting factor for codes used.
        metric_name: Column name for the resulting score.

    Returns:
        New DataFrame with efficiency score column.
    """
    df2 = _ensure_codes_used_column(df).copy()
    if "total_gates" not in df2.columns:
        raise ValueError("'total_gates' column required")
    df2[metric_name] = df2["total_gates"] + lambda_codes * df2["codes_used"]
    return df2

def attach_effectiveness_columns(df_summary: pd.DataFrame, eff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins evaluation results (fidelity, LER, etc.) onto a summary DataFrame by perm_key.

    Args:
        df_summary: Summary DataFrame with permutation metadata.
        eff_df: DataFrame with evaluation metrics.

    Returns:
        Merged DataFrame with added effectiveness columns.
    """
    left = df_summary.copy()
    right = eff_df.copy()

    left["perm_key"] = left.get("perm_key") or left["permutation"].map(_perm_key)
    right["perm_key"] = right.get("perm_key") or right["permutation"].map(_perm_key)

    cols = [
        c for c in [
            "perm_key",
            "codes_used",
            "fidelity",
            "logical_error_rate",
            "postselected",
        ] if c in right.columns
    ]

    merged = left.merge(right[cols], on="perm_key", how="left", suffixes=("", "_eff"))
    return merged.drop(columns=["perm_key"])
